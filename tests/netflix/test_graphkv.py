"""
Tests for Netflix GraphKV model.
"""

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.org.netflix.graphkv import (
    _read_amplification,
    KV_READS_PER_LOGICAL_READ,
    MAX_EDGES_PER_TRAVERSAL,
    NflxGraphKVArguments,
)

# Property test configuration for GraphKV model.
# See tests/netflix/PROPERTY_TESTING.md for configuration options and examples.
PROPERTY_TEST_CONFIG = {
    "org.netflix.graphkv": {
        "extra_model_arguments": {
            "graphkv.avg-fanout-per-hop": 10,
            "graphkv.avg-traversal-depth": 1,
        },
    },
}


def _args(fanout: float, depth: int) -> NflxGraphKVArguments:
    return NflxGraphKVArguments.model_validate(
        {
            "graphkv.avg-fanout-per-hop": fanout,
            "graphkv.avg-traversal-depth": depth,
        }
    )


def test_fanout_costs_nothing_at_one_hop():
    # A single hop is one edge-index scan whatever it returns, so fan-out shows up
    # as response bytes rather than as extra backend requests.
    for fanout in (0.1, 1.0, 10.0, 1000.0):
        assert _read_amplification(_args(fanout, 1)) == KV_READS_PER_LOGICAL_READ


def test_depth_multiplies_by_the_frontier():
    # frontier(h) = fanout ** (h - 1), so depth 3 at fan-out 10 scans 1 + 10 + 100.
    assert _read_amplification(_args(10, 3)) == KV_READS_PER_LOGICAL_READ * 111
    # Fan-out below 1 shrinks the frontier instead of growing it.
    assert _read_amplification(_args(0.16, 3)) == KV_READS_PER_LOGICAL_READ * (
        1 + 0.16 + 0.16**2
    )


def test_frontier_growth_stops_at_the_traversal_edge_limit():
    huge = _read_amplification(_args(MAX_EDGES_PER_TRAVERSAL * 10, 3))
    # The first hop already blows past the edge limit, so hops 2 and 3 never run.
    assert huge == KV_READS_PER_LOGICAL_READ


# Two measured workload shapes and the backend KV load each produced.
# Rates are entities per second, not RPCs.
WORKLOADS = (
    {
        "shape": "node-heavy, shallow",
        "logical_read_per_second": 8_895,
        "logical_write_per_second": 268,
        # Below 1: most traversals return nothing.
        "fanout_per_hop": 0.16,
        "traversal_depth": 1,
        "kv_read_per_second": 11_338,
        "kv_write_per_second": 345,
        "state_gib": 53,
    },
    {
        "shape": "edge-heavy, shallow",
        "logical_read_per_second": 30_494,
        "logical_write_per_second": 13_785,
        "fanout_per_hop": 8.35,
        "traversal_depth": 1,
        "kv_read_per_second": 35_738,
        "kv_write_per_second": 18_003,
        "state_gib": 58,
    },
)


def _plan(model, read_per_second, write_per_second, state_gib, extra_args=None):
    plans = planner.plan_certain(
        model_name=model,
        region="us-east-1",
        desires=CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(read_per_second),
                estimated_write_per_second=certain_int(write_per_second),
            ),
            data_shape=DataShape(estimated_state_size_gib=certain_int(state_gib)),
        ),
        num_results=1,
        extra_model_arguments=extra_args or {},
    )
    assert plans, f"no plan for {model}"
    return plans[0].candidate_clusters


def test_production_shapes_land_near_their_key_value_counterparts():
    # Planning from logical traffic should land near the KeyValue plan built from
    # the backend load that traffic actually produced.
    for workload in WORKLOADS:
        graph_candidate = _plan(
            "org.netflix.graphkv",
            workload["logical_read_per_second"],
            workload["logical_write_per_second"],
            workload["state_gib"],
            {
                "graphkv.avg-fanout-per-hop": workload["fanout_per_hop"],
                "graphkv.avg-traversal-depth": workload["traversal_depth"],
            },
        )
        kv_candidate = _plan(
            "org.netflix.key-value",
            workload["kv_read_per_second"],
            workload["kv_write_per_second"],
            workload["state_gib"],
        )

        # The GraphKV plan carries its own app tier on top of everything the KV
        # plan has, so compare the shared clusters and allow that one extra tier.
        graph_clusters = {
            cluster.cluster_type: cluster for cluster in graph_candidate.regional
        }
        kv_clusters = {
            cluster.cluster_type: cluster for cluster in kv_candidate.regional
        }
        assert "dgwgraphkv" in graph_clusters
        for cluster_type, kv_cluster in kv_clusters.items():
            assert graph_clusters[cluster_type].count <= 2 * kv_cluster.count, (
                f"{workload['shape']} over-provisions {cluster_type}"
            )

        graph_cost = float(graph_candidate.total_annual_cost)
        kv_cost = float(kv_candidate.total_annual_cost)
        assert graph_cost < 1.5 * kv_cost, (
            f"{workload['shape']} costs {graph_cost:,.0f} against {kv_cost:,.0f} "
            f"for the KeyValue shard behind it"
        )

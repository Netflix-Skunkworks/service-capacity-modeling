"""
Tests for Netflix GraphKV model.
"""

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.org.netflix.graphkv import (
    _read_amplification,
    BYTES_PER_KV_RECORD,
    DEFAULT_MAX_TRAVERSAL_DEPTH,
    KV_READS_PER_LOGICAL_READ,
    KV_RECORDS_PER_LOGICAL_ITEM,
    KV_WRITES_PER_LOGICAL_WRITE,
    MAX_EDGES_PER_TRAVERSAL,
    NflxGraphKVArguments,
    NflxGraphKVCapacityModel,
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


def test_depth_beyond_the_fleet_default_is_accepted():
    # GraphTraversalService.getMaxDepth prefers a namespace's own
    # TraversalConfig.maxDepth whenever it is positive and never clamps it to the
    # shard default, so a namespace can be configured deeper than
    # DEFAULT_MAX_TRAVERSAL_DEPTH and the model has to plan it, not reject it.
    deeper = _read_amplification(_args(2.0, DEFAULT_MAX_TRAVERSAL_DEPTH + 2))
    assert deeper > _read_amplification(_args(2.0, DEFAULT_MAX_TRAVERSAL_DEPTH))
    # The edge limit, not the depth field, is what stops a typo from exploding: at
    # fan-out 1000 the second hop already exceeds it, so hops 3..50 never run.
    assert _read_amplification(_args(1_000.0, 50)) == KV_READS_PER_LOGICAL_READ * 1_001


def _composed_kv_data_shape(data_shape: DataShape) -> DataShape:
    """Run the key-value composition transform alone, without the planner."""
    desires = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(1_000),
            estimated_write_per_second=certain_int(100),
        ),
        data_shape=data_shape,
    )
    ((_, transform),) = NflxGraphKVCapacityModel.compose_with(desires, {})
    return transform(desires).data_shape


def test_state_size_alone_passes_through():
    # A size in GiB is already a backend size, so it reaches KeyValue untouched.
    size_only = _composed_kv_data_shape(
        DataShape(estimated_state_size_gib=certain_int(100))
    )
    assert size_only.estimated_state_size_gib.mid == 100


def test_explicit_state_size_wins_over_an_item_count():
    # Expanding the item count on top of a size the caller gave us would charge the
    # fan-out twice and silently discard what they asked for.
    both = _composed_kv_data_shape(
        DataShape(
            estimated_state_size_gib=certain_int(100),
            estimated_state_item_count=certain_int(1_000_000_000),
        )
    )
    assert both.estimated_state_size_gib.mid == 100


def test_item_count_alone_expands_by_the_stored_record_multiplier():
    # Only an item count has to be expanded, and it expands by records STORED per
    # logical entity -- not by the write-REQUEST multiplier, which is a different
    # quantity that happens to share its value today.
    count_only = _composed_kv_data_shape(
        DataShape(estimated_state_item_count=certain_int(1_000_000_000))
    )
    expected_gib = (
        1_000_000_000 * KV_RECORDS_PER_LOGICAL_ITEM * BYTES_PER_KV_RECORD / 1024**3
    )
    assert count_only.estimated_state_size_gib.mid == pytest.approx(expected_gib)


def test_stored_size_ignores_the_write_request_multiplier(monkeypatch):
    # The record count and the request count share a value today, so the test above
    # cannot tell which one sizes the disk. Pull them apart and check that moving the
    # throughput knob does not resize storage.
    module = "service_capacity_modeling.models.org.netflix.graphkv"
    monkeypatch.setattr(f"{module}.KV_RECORDS_PER_LOGICAL_ITEM", 4.0)
    monkeypatch.setattr(f"{module}.KV_WRITES_PER_LOGICAL_WRITE", 99.0)

    count_only = _composed_kv_data_shape(
        DataShape(estimated_state_item_count=certain_int(1_000_000_000))
    )
    expected_gib = 1_000_000_000 * 4.0 * BYTES_PER_KV_RECORD / 1024**3
    assert count_only.estimated_state_size_gib.mid == pytest.approx(expected_gib)


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


def test_amplification_matches_measured_backend_load():
    # The direct guard on the constants: what the model multiplies logical traffic
    # by, against what each workload's backend load actually was. Two-sided on
    # purpose -- an upper bound alone passes an arbitrarily under-estimated model,
    # and under-estimating is the direction that under-provisions.
    for workload in WORKLOADS:
        modeled_read = _read_amplification(
            _args(workload["fanout_per_hop"], workload["traversal_depth"])
        )
        operations = (
            (
                "read",
                modeled_read,
                workload["kv_read_per_second"] / workload["logical_read_per_second"],
            ),
            (
                "write",
                KV_WRITES_PER_LOGICAL_WRITE,
                workload["kv_write_per_second"] / workload["logical_write_per_second"],
            ),
        )
        for operation, modeled, measured in operations:
            headroom = modeled / measured
            assert 1.0 <= headroom <= 1.25, (
                f"{workload['shape']} {operation} amplification is {headroom:.2f}x "
                f"the backend load this workload actually produced. The band is "
                f"deliberately tight; re-check the measurements before widening it."
            )


def test_production_shapes_land_near_their_key_value_counterparts():
    # Planning from logical traffic should land near the KeyValue plan built from
    # the backend load that traffic actually produced -- near from both sides.
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
            assert cluster_type in graph_clusters, (
                f"{workload['shape']} is missing the {cluster_type} tier that the "
                f"KeyValue plan provisions"
            )
            count_ratio = graph_clusters[cluster_type].count / kv_cluster.count
            assert 0.75 <= count_ratio <= 2.0, (
                f"{workload['shape']} sizes {cluster_type} at {count_ratio:.2f}x the "
                f"KeyValue shard behind it"
            )

        # Compared as a ratio: the graph plan adds its own tier on top of everything
        # the KV plan plans, so it has to cost more, but not by a multiple.
        cost_ratio = float(graph_candidate.total_annual_cost) / float(
            kv_candidate.total_annual_cost
        )
        assert 1.0 < cost_ratio < 1.5, (
            f"{workload['shape']} costs {cost_ratio:.2f}x the KeyValue shard behind it"
        )

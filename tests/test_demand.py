import dataclasses

import pytest

from service_capacity_modeling.capacity_planner import PlannerArguments
from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.demand import Demand
from service_capacity_modeling.demand import fit
from service_capacity_modeling.demand import Fit
from service_capacity_modeling.demand import fit_all
from service_capacity_modeling.demand import Layout
from service_capacity_modeling.demand import ncu_per_node
from service_capacity_modeling.demand import NodeRules
from service_capacity_modeling.demand import Resource
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import AlternativeOptions
from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import CurrentClusters
from service_capacity_modeling.interface import CurrentRegionClusterCapacity
from service_capacity_modeling.interface import Excuse
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern

HARDWARE = shapes.region("us-east-1")
GP3 = HARDWARE.drives["gp3"]

READ_ONLY_KV = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        access_pattern=AccessPattern.latency,
        estimated_read_per_second=certain_int(50_000),
        estimated_write_per_second=certain_int(0),
        estimated_mean_read_latency_ms=certain_float(2.0),
        estimated_mean_write_latency_ms=certain_float(0),
        estimated_mean_read_size_bytes=certain_int(2048),
    ),
    data_shape=DataShape(estimated_state_size_gib=certain_int(1000)),
)
READ_ONLY_KV_ARGS = {"total_num_partitions": 12}
CASSANDRA = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(100_000),
        estimated_write_per_second=certain_int(100_000),
        estimated_mean_read_latency_ms=certain_float(0.5),
        estimated_mean_write_latency_ms=certain_float(0.4),
    ),
    data_shape=DataShape(estimated_state_size_gib=certain_int(10)),
)


def test_ncu_normalizes_cpu_speed():
    assert ncu_per_node(HARDWARE.instances["m6id.2xlarge"]) == pytest.approx(14.0)
    assert ncu_per_node(HARDWARE.instances["m7a.2xlarge"]) == pytest.approx(
        25.1, abs=0.1
    )


WITH_ALTERNATIVES = PlannerArguments(alternatives=AlternativeOptions())


def test_alternatives_are_off_by_default():
    plan = planner.plan_certain(
        model_name="org.netflix.read-only-kv",
        region="us-east-1",
        desires=READ_ONLY_KV,
        extra_model_arguments=READ_ONLY_KV_ARGS,
    )[0]

    assert not plan.candidate_clusters.regional[0].alternatives


def test_planner_lists_alternatives_that_host_the_same_partition_layout():
    plan = planner.plan_certain(
        model_name="org.netflix.read-only-kv",
        region="us-east-1",
        desires=READ_ONLY_KV,
        extra_model_arguments=READ_ONLY_KV_ARGS,
        planner_arguments=WITH_ALTERNATIVES,
    )[0]
    cluster = plan.candidate_clusters.regional[0]
    params = cluster.cluster_params
    partition_gib = 1000 * 1.15 / 12  # data x default disk buffer / partitions

    assert cluster.alternatives, "expected at least one alternative"
    for alt in cluster.alternatives:
        instance = HARDWARE.instances[alt.instance]
        assert alt.count == cluster.count
        assert instance.family != cluster.instance.family
        # Each node can hold the primary's partitions...
        ppn = params["read-only-kv.partitions_per_node"]
        assert alt.supply_per_node["disk_gib"] >= ppn * partition_gib
        # ...and each node supplies the CPU the layout needs from it.
        assert alt.supply_per_node["ncu"] >= alt.need_per_node["ncu"]


def test_fit_sizes_every_shape_like_the_model():
    """The demand path must agree with the model's own per-shape plan."""
    plans = planner.plan_certain(
        model_name="org.netflix.read-only-kv",
        region="us-east-1",
        desires=READ_ONLY_KV,
        extra_model_arguments=READ_ONLY_KV_ARGS,
        num_results=1000,
        planner_arguments=PlannerArguments(max_results_per_family=100),
    )
    stated = planner.demand(
        model_name="org.netflix.read-only-kv",
        region="us-east-1",
        desires=READ_ONLY_KV,
        drive=HARDWARE.instances["i4i.2xlarge"].drive,
        extra_model_arguments=READ_ONLY_KV_ARGS,
    )
    assert stated is not None and len(plans) > 10
    demand, rules = stated
    for plan in plans:
        cluster = plan.candidate_clusters.regional[0]
        sized = fit(demand, rules, cluster.instance, cluster.instance.drive)
        assert isinstance(sized, Fit), cluster.instance.name
        assert sized.count == cluster.count, cluster.instance.name


def test_models_without_demand_get_alternatives_by_planning_again():
    plan = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=CASSANDRA,
        planner_arguments=WITH_ALTERNATIVES,
    )[0]
    zones = plan.candidate_clusters.zonal
    primary = zones[0]

    assert primary.alternatives
    assert all(z.alternatives == primary.alternatives for z in zones)
    families = [HARDWARE.instances[a.instance].family for a in primary.alternatives]
    assert primary.instance.family not in families
    assert len(set(families)) == len(families)
    costs = [a.annual_cost for a in primary.alternatives]
    assert costs == sorted(costs)
    for alt in primary.alternatives:
        assert alt.count == primary.count
        assert alt.annual_cost <= primary.annual_cost * 1.5
        assert alt.binding is None


def test_alternatives_are_certain_planning_only():
    with pytest.raises(ValueError, match="certain planning only"):
        planner.plan(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=CASSANDRA,
            planner_arguments=WITH_ALTERNATIVES,
        )


def test_fixed_count_caller_picks_types_for_a_target():
    """How an autoscaler holding the node count fixed would call SCM."""
    stated = planner.demand(
        model_name="org.netflix.read-only-kv",
        region="us-east-1",
        desires=READ_ONLY_KV,
        drive=HARDWARE.instances["i4i.2xlarge"].drive,
        extra_model_arguments=READ_ONLY_KV_ARGS,
    )
    assert stated is not None
    demand, rules = stated
    primary = planner.plan_certain(
        model_name="org.netflix.read-only-kv",
        region="us-east-1",
        desires=READ_ONLY_KV,
        extra_model_arguments=READ_ONLY_KV_ARGS,
    )[0].candidate_clusters.regional[0]
    # The caller brings its own CPU target and pins only the node count.
    demand = dataclasses.replace(demand, ncu=Interval(low=100, mid=140, high=180))
    local = [i for i in HARDWARE.instances.values() if i.drive is not None]

    fits, excuses = fit_all(demand, rules, local, pinned=Layout(primary.count))

    assert fits
    for f in fits:
        params = f.layout.params
        copies = params["read-only-kv.replica_count"]
        assert f.count == primary.count
        assert copies >= 2
        assert copies * params["read-only-kv.nodes_for_one_copy"] == f.count
        assert f.supply_per_node[Resource.ncu] * f.count >= 140
    assert [f.annual_cost for f in fits] == sorted(f.annual_cost for f in fits)
    # Every rejection names the resource that did not fit.
    assert excuses and all(e.reason for e in excuses)


def test_count_only_pin_names_the_short_resource():
    """A count the shape cannot host names what ran short, here CPU: 4 nodes
    of 14 NCU cannot serve 63 NCU at any partition placement."""
    stated = planner.demand(
        model_name="org.netflix.read-only-kv",
        region="us-east-1",
        desires=READ_ONLY_KV,
        drive=GP3,
        extra_model_arguments=READ_ONLY_KV_ARGS,
    )
    assert stated is not None
    demand, rules = stated
    demand = dataclasses.replace(demand, ncu=certain_float(63))
    m6id = HARDWARE.instances["m6id.2xlarge"]

    excuse = fit(demand, rules, m6id, m6id.drive, pinned=Layout(4))

    assert isinstance(excuse, Excuse)
    assert excuse.reason == "Cannot host 4 nodes: ncu"
    assert excuse.bottleneck == Bottleneck.cpu


def test_count_only_pin_honors_a_partitions_per_node_cap():
    """A caller can cap partitions per node, so a deploy never puts more data
    on a node than the plan it came from."""
    small = READ_ONLY_KV.model_copy(deep=True)
    small.data_shape.estimated_state_size_gib = certain_int(48)
    stated = planner.demand(
        model_name="org.netflix.read-only-kv",
        region="us-east-1",
        desires=small,
        drive=GP3,
        extra_model_arguments=READ_ONLY_KV_ARGS,
    )
    assert stated is not None
    demand, rules = stated
    demand = dataclasses.replace(demand, ncu=certain_float(63))
    r6id = HARDWARE.instances["r6id.4xlarge"]

    def placed(pinned: Layout) -> tuple:
        sized = fit(demand, rules, r6id, r6id.drive, pinned=pinned)
        assert isinstance(sized, Fit), sized
        params = sized.layout.params
        return (
            params["read-only-kv.partitions_per_node"],
            params["read-only-kv.replica_count"],
        )

    assert placed(Layout(6)) == (12, 6)
    capped = Layout(6, params={"read-only-kv.max_partitions_per_node": 6})
    assert placed(capped) == (6, 3)


def test_fit_all_ranks_by_the_callers_key():
    stated = planner.demand(
        model_name="org.netflix.read-only-kv",
        region="us-east-1",
        desires=READ_ONLY_KV,
        drive=GP3,
        extra_model_arguments=READ_ONLY_KV_ARGS,
    )
    assert stated is not None
    demand, rules = stated
    demand = dataclasses.replace(demand, ncu=certain_float(60))
    m6id = [HARDWARE.instances[f"m6id.{size}xlarge"] for size in ("2", "4")]

    def ncu_from_170(f: Fit) -> float:
        return abs(f.supply_per_node[Resource.ncu] * f.count - 170)

    cheapest, _ = fit_all(demand, rules, m6id, pinned=Layout(6))
    nearest, _ = fit_all(demand, rules, m6id, pinned=Layout(6), key=ncu_from_170)

    assert [f.instance.name for f in cheapest] == ["m6id.2xlarge", "m6id.4xlarge"]
    assert [f.instance.name for f in nearest] == ["m6id.4xlarge", "m6id.2xlarge"]


def test_existing_cluster_keeps_whole_core_count():
    """Converting the model's cores to NCU and back must not add a core."""
    desires = READ_ONLY_KV.model_copy(deep=True)
    desires.query_pattern.estimated_read_per_second = certain_int(5_000)
    desires.data_shape.estimated_state_size_gib = certain_int(100)
    desires.current_clusters = CurrentClusters(
        regional=[
            CurrentRegionClusterCapacity(
                cluster_instance_name="m6id.2xlarge",
                cluster_instance_count=certain_int(6),
                cpu_utilization=certain_float(80.0),
            )
        ]
    )
    plan = planner.plan_certain(
        model_name="org.netflix.read-only-kv",
        region="us-east-1",
        desires=desires,
        extra_model_arguments=READ_ONLY_KV_ARGS,
        instance_families=["m6id"],
    )[0]
    cluster = plan.candidate_clusters.regional[0]

    assert plan.requirements.regional[0].cpu_cores.mid == 24
    assert (cluster.instance.name, cluster.count) == ("m6id.2xlarge", 3)


def test_layout_rule_cannot_skip_a_resource():
    """A layout rule places nodes, but every node must still supply each
    resource it carries, including ones the rule ignores."""
    demand = Demand(
        ncu=certain_float(10),
        mem_gib=certain_float(10),
        disk_gib=certain_float(10),
        network_mbps=certain_float(1_000_000),
    )
    rules = NodeRules(layout=lambda instance, need, have, pinned: Layout(count=2))

    result = fit(demand, rules, HARDWARE.instances["m6id.2xlarge"], GP3)

    assert isinstance(result, Excuse)
    assert "network_mbps" in result.reason

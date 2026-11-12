import pytest

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
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
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


def test_ncu_normalizes_cpu_speed():
    assert ncu_per_node(HARDWARE.instances["m6id.2xlarge"]) == pytest.approx(14.0)
    assert ncu_per_node(HARDWARE.instances["m7a.2xlarge"]) == pytest.approx(
        25.1, abs=0.1
    )


def test_ceiling_layout_counts_the_binding_resource():
    demand = Demand(
        ncu=Interval(low=120, mid=140, high=160),
        mem_gib=Interval(low=60, mid=90, high=120),
        disk_gib=Interval(low=600, mid=800, high=1000),
        network_mbps=Interval(low=1000, mid=2000, high=3000),
    )
    rules = NodeRules(reserve_mem_gib=lambda _: 4)

    m7a = fit(demand, rules, HARDWARE.instances["m7a.2xlarge"], GP3)
    weaker = fit(
        demand, rules, HARDWARE.instances["m6id.2xlarge"], GP3, pinned=Layout(6)
    )

    assert isinstance(m7a, Fit)
    assert m7a.count == 6  # ceil(140 / 25.1); memory needs only ceil(90 / 26.5)
    assert m7a.binding == Resource.ncu
    assert isinstance(weaker, Excuse)  # 6 x 14 NCU is below the 120 floor


def test_alternatives_meet_mid_unless_caller_accepts_low():
    demand = Demand(
        ncu=Interval(low=120, mid=140, high=160),
        mem_gib=certain_float(60),
        disk_gib=certain_float(600),
        network_mbps=certain_float(1000),
    )
    m5 = HARDWARE.instances["m5.4xlarge"]  # 21.6 NCU: 6 nodes give 129

    assert isinstance(fit(demand, NodeRules(), m5, GP3, pinned=Layout(6)), Excuse)
    assert isinstance(
        fit(demand, NodeRules(), m5, GP3, pinned=Layout(6), point="low"), Fit
    )


def test_planner_lists_alternatives_that_host_the_same_partition_layout():
    plan = planner.plan_certain(
        model_name="org.netflix.read-only-kv",
        region="us-east-1",
        desires=READ_ONLY_KV,
        extra_model_arguments=READ_ONLY_KV_ARGS,
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
    pinned = Layout(count=primary.count, params=dict(primary.cluster_params))
    local = [i for i in HARDWARE.instances.values() if i.drive is not None]

    fits, excuses = fit_all(demand, rules, local, pinned=pinned, point="mid")

    assert primary.instance.name in {f.instance.name for f in fits}
    assert all(f.count == primary.count for f in fits)
    assert [f.annual_cost for f in fits] == sorted(f.annual_cost for f in fits)
    # Every rejection names the resource that did not fit.
    assert excuses and all(e.reason for e in excuses)

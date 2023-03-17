from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import AccessPattern, certain_float
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern


def test_evcache_high_qps():
    qps = 100_000
    high_qps = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=qps // 10, mid=qps, high=qps * 10, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=qps // 10, mid=qps, high=qps * 10, confidence=0.98
            ),
            estimated_write_size_bytes=Interval(
                low=10, mid=100, high=1000, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=10, mid=100, high=1000, confidence=0.98
            ),
            estimated_state_item_count=Interval(
                low=10, mid=100, high=1000, confidence=0.98
            ),
        ),
    )
    plan = planner.plan(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=high_qps,
    )

    assert len(plan.least_regret) >= 2

    lr = plan.least_regret[0]
    # EVCache should regret having too little RAM, disk and spending too much
    assert all(k in lr.requirements.regrets for k in ("spend", "mem", "disk"))

    # EVCache should be pretty cheap for 100k QPS
    assert lr.candidate_clusters.annual_costs["evcache.zonal-clusters"] < 10000
    # Without replication shouldn't have network costs
    assert len(lr.candidate_clusters.annual_costs.keys()) == 1

    zc = lr.candidate_clusters.zonal[0]

    if zc.instance.drive is not None:
        # If we end up with disk we want at least 100 GiB of disk per zone
        assert zc.count * zc.instance.drive.size_gib > 100
    else:
        # If we end up with RAM we want at least 100 GiB of ram per zone
        assert zc.count * zc.instance.ram_gib > 100


def test_evcache_large_data():
    qps = 10_000
    large_data = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=qps // 10, mid=qps, high=qps * 10, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=qps // 10, mid=qps, high=qps * 10, confidence=0.98
            ),
            estimated_write_size_bytes=Interval(
                low=1000, mid=5000, high=10_000, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=100, mid=5000, high=10_000, confidence=0.98
            )
        ),
    )
    plan = planner.plan(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=large_data,
    )

    assert len(plan.least_regret) >= 1

    lr = plan.least_regret[0]
    # EVCache should regret having too little RAM, disk and spending too much
    assert all(k in lr.requirements.regrets for k in ("spend", "mem", "disk"))

    # EVCache should be somewhat expensive due to the large amount of data
    assert lr.candidate_clusters.annual_costs["evcache.zonal-clusters"] > 10_000
    # Without replication shouldn't have network costs
    assert len(lr.candidate_clusters.annual_costs.keys()) == 1

    zc = lr.candidate_clusters.zonal[0]

    # For the sheer volume of data, it probably doesn't make sense for the least regretful cluster to not have disk.
    assert zc.instance.drive is not None

    # We want at least 1 TiB of disk per zone
    assert zc.count * zc.instance.drive.size_gib > 1000


def test_evcache_replication():
    high_qps = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            access_pattern=AccessPattern.latency,
            estimated_read_per_second=Interval(
                low=10_000, mid=100_000, high=1_000_000, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=10_000, mid=100_000, high=1_000_000, confidence=0.98
            ),
        ),
        # This should work out to around 200 GiB of state
        data_shape=DataShape(
            estimated_state_item_count=Interval(
                low=100_000_000, mid=1_000_000_000, high=10_000_000_000, confidence=0.98
            )
        ),
    )
    plan = planner.plan(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=high_qps,
        num_regions=3,
        extra_model_arguments={"cross_region_replication": "sets"},
    )
    assert len(plan.least_regret) >= 2

    lr = plan.least_regret[0]
    # EVCache should regret having too little RAM, disk and spending too much
    assert all(k in lr.requirements.regrets for k in ("spend", "mem", "disk"))
    assert lr.requirements.zonal[0].disk_gib.mid > 200

    # EVCache compute should be pretty cheap for 100k RPS with 10k WPS
    assert lr.candidate_clusters.annual_costs["evcache.zonal-clusters"] < 10000

    set_inter_region = lr.candidate_clusters.annual_costs["evcache.net.inter.region"]

    # With replication should have network costs
    assert 10000 < set_inter_region < 40000
    assert (
        50000 < lr.candidate_clusters.annual_costs["evcache.net.intra.region"] < 120000
    )

    delete_plan = planner.plan(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=high_qps,
        num_regions=3,
        extra_model_arguments={
            "cross_region_replication": "evicts",
            "copies_per_region": 3,
        },
    )

    lr = delete_plan.least_regret[0]

    # Evicts should be cheaper than sets
    evict_inter_region = lr.candidate_clusters.annual_costs["evcache.net.inter.region"]
    assert evict_inter_region < set_inter_region

    # With replication should have network costs
    assert 5000 < evict_inter_region < 15000
    assert (
        12000 < lr.candidate_clusters.annual_costs["evcache.net.intra.region"] < 40000
    )


def test_evcache_compare_working_sets():
    small = CapacityDesires(
        service_tier=2,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=10_000, mid=100_000, high=1_000_000, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=10_000, mid=100_000, high=1_000_000, confidence=0.98
            ),
            estimated_write_size_bytes=Interval(
                low=10, mid=100, high=1000, confidence=0.98
            )
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=10, mid=100, high=1000, confidence=0.98
            ),
            estimated_working_set_percent=certain_float(0.10)
        ),
    )
    large = small.copy(deep=True)
    large.data_shape.estimated_working_set_percent = certain_float(0.90)

    plan_small = planner.plan(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=small,
    )
    plan_large = planner.plan(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=large,
    )

    assert len(plan_small.least_regret) >= 2
    assert len(plan_large.least_regret) >= 2

    lr_small = plan_small.least_regret[0]
    lr_large = plan_large.least_regret[0]

    # Only the plan whose desires contain the smaller working set percentage should care about disk.
    assert all(k in lr_small.requirements.regrets for k in ("spend", "mem", "disk"))
    assert all(k in lr_large.requirements.regrets for k in ("spend", "mem"))

    # Smaller working set percentage should lead to fewer costs.
    assert lr_small.candidate_clusters.annual_costs["evcache.zonal-clusters"] < \
           lr_large.candidate_clusters.annual_costs["evcache.zonal-clusters"]

    # The large difference in working set percentage should lead to a difference in RAM.
    assert lr_small.candidate_clusters.zonal[0].instance.ram_gib < \
           lr_large.candidate_clusters.zonal[0].instance.ram_gib

    # The small working set percentage should lead to picking an instance with both memory and disk.
    assert lr_small.candidate_clusters.zonal[0].instance.drive is not None
    assert lr_small.candidate_clusters.zonal[0].instance.ram_gib is not None

    # The large working set percentage should lead to only memory (no disk).
    assert lr_large.candidate_clusters.zonal[0].instance.drive is None
    assert lr_large.candidate_clusters.zonal[0].instance.ram_gib is not None

    # Without replication shouldn't have network costs
    assert len(lr_small.candidate_clusters.annual_costs.keys()) == 1
    assert len(lr_large.candidate_clusters.annual_costs.keys()) == 1


def test_evcache_compare_tiers():
    low = CapacityDesires(
        service_tier=0,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=10_000, mid=100_000, high=1_000_000, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=10_000, mid=100_000, high=1_000_000, confidence=0.98
            ),
            estimated_write_size_bytes=Interval(
                low=10, mid=100, high=1000, confidence=0.98
            )
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=10, mid=100, high=1000, confidence=0.98
            ),
        ),
    )
    high = low.copy(deep=True)
    high.service_tier = 3

    plan_low = planner.plan(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=low,
    )
    plan_high = planner.plan(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=high,
    )

    assert len(plan_low.least_regret) >= 2
    assert len(plan_high.least_regret) >= 2

    lr_low = plan_low.least_regret[0]
    lr_high = plan_high.least_regret[0]

    # EVCache should regret having too little RAM, disk and spending too much
    assert all(k in lr_low.requirements.regrets for k in ("spend", "mem", "disk"))
    assert all(k in lr_high.requirements.regrets for k in ("spend", "mem", "disk"))

    # Lower tier should lead to greater costs.
    assert lr_low.candidate_clusters.annual_costs["evcache.zonal-clusters"] > \
           lr_high.candidate_clusters.annual_costs["evcache.zonal-clusters"]

    # Large difference in tiers should lead to different instance family types.
    assert lr_low.candidate_clusters.zonal[0].instance.family != lr_high.candidate_clusters.zonal[0].instance.family

    # Without replication shouldn't have network costs
    assert len(lr_low.candidate_clusters.annual_costs.keys()) == 1
    assert len(lr_high.candidate_clusters.annual_costs.keys()) == 1

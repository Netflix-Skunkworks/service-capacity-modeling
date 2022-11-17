from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern


def test_evcache_high_qps():
    high_qps = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=10_000, mid=100_000, high=1_000_000, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=10_000, mid=100_000, high=1_000_000, confidence=0.98
            ),
            estimated_write_size_bytes=Interval(
                low=10, mid=100, high=1000, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=10, mid=100, high=1000, confidence=0.98
            )
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

    # EVCache should be pretty cheap for 100k RPS with 10k WPS
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

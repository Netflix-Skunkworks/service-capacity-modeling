from service_capacity_modeling.capacity_planner import planner
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
                low=1_000, mid=10_000, high=100_000, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
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

    # EVCache should be pretty cheap for 100k RPS with 10k WPS
    assert lr.candidate_clusters.total_annual_cost < 10000

    zc = lr.candidate_clusters.zonal[0]

    if zc.instance.drive is not None:
        # If we end up with disk we want at least 100 GiB of disk per zone
        assert zc.count * zc.instance.drive.size_gib > 100
    else:
        # If we end up with RAM we want at least 100 GiB of ram per zone
        assert zc.count * zc.instance.ram_gib > 100

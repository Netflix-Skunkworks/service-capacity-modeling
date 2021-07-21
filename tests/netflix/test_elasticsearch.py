from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern


def test_es_increasing_qps_simple():
    qps_values = (100, 1000, 10_000, 100_000)
    zonal_result = []
    for qps in qps_values:
        simple = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=Interval(
                    low=qps // 10, mid=qps, high=qps * 10, confidence=0.98
                ),
                estimated_write_per_second=Interval(
                    low=qps // 10, mid=qps, high=qps * 10, confidence=0.98
                ),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=Interval(
                    low=20, mid=200, high=2000, confidence=0.98
                ),
            ),
        )

        cap_plan = planner.plan(
            model_name="org.netflix.elasticsearch",
            region="us-east-1",
            desires=simple,
            simulations=256,
        )

        # Check the ES cluster
        zlr = cap_plan.least_regret[0].candidate_clusters.zonal[0]
        zlr_cpu = zlr.count * zlr.instance.cpu
        zlr_cost = cap_plan.least_regret[0].candidate_clusters.total_annual_cost
        zlr_family = zlr.instance.family
        if zlr.instance.drive is None:
            assert sum(dr.size_gib for dr in zlr.attached_drives) >= 200
        else:
            assert zlr.instance.drive.size_gib >= 50

        zonal_result.append(
            (
                zlr_family,
                zlr_cpu,
                zlr_cost,
                cap_plan.least_regret[0].requirements.zonal[0],
            )
        )

    # We should generally want cheap CPUs for Cassandra
    assert all(r[0] in ("r5", "r5d", "m5", "m5d") for r in zonal_result)

    # Should have more capacity as requirement increases
    x = [r[1] for r in zonal_result]
    assert x[0] < x[-1]
    assert sorted(x) == x

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern


def test_kv_increasing_qps_simple():
    qps_values = (100, 1000, 10_000, 100_000)
    zonal_result = []
    regional_result = []
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
            model_name="org.netflix.key-value",
            region="us-east-1",
            desires=simple,
            simulations=256,
        )

        # Check the C* cluster
        zlr = cap_plan.least_regret[0].candidate_clusters.zonal[0]
        zlr_cpu = zlr.count * zlr.instance.cpu
        zlr_cost = cap_plan.least_regret[0].candidate_clusters.total_annual_cost
        zlr_family = zlr.instance.family
        if zlr.instance.drive is None:
            assert sum(dr.size_gib for dr in zlr.attached_drives) >= 200
        else:
            assert zlr.instance.drive.size_gib >= 100

        zonal_result.append(
            (
                zlr_family,
                zlr_cpu,
                zlr_cost,
                cap_plan.least_regret[0].requirements.zonal[0],
            )
        )

        # Check the Java cluster
        rlr = cap_plan.least_regret[0].candidate_clusters.regional[0]
        regional_result.append(
            (rlr.instance.family, rlr.count * rlr.instance.cpu, zlr_cost)
        )
        # We should never be paying for ephemeral drives
        assert rlr.instance.drive is None

    # We should generally want cheap CPUs for Cassandra
    assert all(r[0] in ("r5", "m5d", "m5", "i3en") for r in zonal_result)

    # We just want ram and cpus for a java app
    assert all(r[0] in ("m5", "r5") for r in regional_result)

    # Should have more capacity as requirement increases
    x = [r[1] for r in zonal_result]
    assert x[0] < x[-1]
    assert sorted(x) == x

    # Should have more capacity as requirement increases
    x = [r[1] for r in regional_result]
    assert x[0] < x[-1]
    assert sorted(x) == x

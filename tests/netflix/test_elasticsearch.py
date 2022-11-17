from collections import defaultdict

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern


def test_es_increasing_qps_simple():
    qps_values = (100, 1000, 10_000, 100_000)
    zonal_result = defaultdict(list)
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
        for zonal in cap_plan.least_regret[0].candidate_clusters.zonal:
            zonal_result[zonal.cluster_type].append(zonal_summary(zonal))

    expected_families = set(["r", "m", "i"])
    for cluster_type in list(zonal_result.keys()):
        zonal_by_increasing_qps = zonal_result[cluster_type]

        families = {r[0] for r in zonal_by_increasing_qps}
        for f in families:
            assert f[0] in expected_families

        # Should have more CPU and Disk capacity as requirement increases
        cpu = [r[2] for r in zonal_by_increasing_qps]
        assert cpu[0] <= cpu[-1], f"cpu for {cluster_type} going down as QPS went up?"

        cost = [r[3] for r in zonal_by_increasing_qps]
        assert (
            cost[0] <= cost[-1]
        ), f"cost for {cluster_type} going down as QPS went up?"

        disk = [r[4] for r in zonal_by_increasing_qps]
        assert (
            disk[0] <= disk[-1]
        ), f"disk for {cluster_type} going down as QPS went up?"


def zonal_summary(zlr):
    zlr_cpu = zlr.count * zlr.instance.cpu
    zlr_cost = zlr.annual_cost
    zlr_family = zlr.instance.family
    zlr_drive_gib = sum(dr.size_gib for dr in zlr.attached_drives)
    if zlr.instance.drive is not None:
        zlr_drive_gib += zlr.instance.drive.size_gib
    zlr_drive_gib *= zlr.count

    return (
        zlr_family,
        zlr.count,
        zlr_cpu,
        zlr_cost,
        zlr_drive_gib,
    )

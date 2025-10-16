from collections import Counter
from collections import defaultdict

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.org.netflix.elasticsearch import (
    NflxElasticsearchArguments,
)
from service_capacity_modeling.models.org.netflix.elasticsearch import (
    NflxElasticsearchDataCapacityModel,
)


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

    expected_families = {"r", "m", "c", "i"}
    for cluster_type in list(zonal_result.keys()):
        zonal_by_increasing_qps = zonal_result[cluster_type]

        families = {r[0] for r in zonal_by_increasing_qps}
        for f in families:
            assert f[0] in expected_families

        # Should have more CPU and Disk capacity as requirement increases
        cpu = [r[2] for r in zonal_by_increasing_qps]
        assert cpu[0] <= cpu[-1], f"cpu for {cluster_type} going down as QPS went up?"

        cost = [r[3] for r in zonal_by_increasing_qps]
        assert cost[0] <= cost[-1], (
            f"cost for {cluster_type} going down as QPS went up?"
        )

        disk = [r[4] for r in zonal_by_increasing_qps]
        assert disk[0] <= disk[-1], (
            f"disk for {cluster_type} going down as QPS went up?"
        )


def test_es_simple_mean_percentiles():
    simple = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=100, mid=1000, high=10_000, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=100, mid=1000, high=10_000, confidence=0.98
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

    assert len(cap_plan.mean) > 0, "mean is empty"
    assert all(mean_plan for mean_plan in cap_plan.mean), (
        "One or more mean plans are empty"
    )

    assert len(cap_plan.percentiles) > 0, "percentiles are empty"
    assert all(percentile_plan for percentile_plan in cap_plan.percentiles.values()), (
        "One or more percentile plans are empty"
    )


def test_es_simple_certain():
    simple = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=100, mid=1000, high=10_000, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=100, mid=1000, high=10_000, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=20, mid=200, high=2000, confidence=0.98
            ),
        ),
    )

    cap_plan = planner.plan_certain(
        model_name="org.netflix.elasticsearch",
        region="us-east-1",
        desires=simple,
    )

    assert len(cap_plan) > 0, "Resulting cap_plan is empty"

    for plan in cap_plan:
        assert plan, "One or more plans is empty"
        assert plan.candidate_clusters, "candidate_clusters is empty"
        assert plan.candidate_clusters.zonal, "candidate_clusters.zonal is empty"
        assert len(plan.candidate_clusters.zonal) == 9, (
            "len(candidate_clusters.zonal) != 9"
        )

        cluster_type_counts = Counter(
            zone.cluster_type for zone in plan.candidate_clusters.zonal
        )

        assert len(cluster_type_counts) == 3, "Expecting 3 cluster types"
        assert cluster_type_counts["elasticsearch-search"] == 3, (
            "Expecting exactly 3 search nodes"
        )
        assert cluster_type_counts["elasticsearch-master"] == 3, (
            "Expecting exactly 3 master nodes"
        )
        assert cluster_type_counts["elasticsearch-data"] >= 3, (
            "Expecting at least 3 data nodes"
        )


def test_es_simple_certain_state_size_only():
    estimated_state_size_gib = 10_000
    expected_allocated_disk_size_gib = (
        NflxElasticsearchDataCapacityModel.default_buffers().default.ratio
        * NflxElasticsearchArguments.model_fields.get("copies_per_region").default
        * estimated_state_size_gib
    )

    simple = CapacityDesires(
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=estimated_state_size_gib,
                mid=estimated_state_size_gib,
                high=estimated_state_size_gib,
                confidence=1.0,
            ),
        ),
    )

    cap_plan = planner.plan_certain(
        model_name="org.netflix.elasticsearch",
        region="us-east-1",
        desires=simple,
    )

    assert len(cap_plan) > 0, "Resulting cap_plan is empty"

    for plan in cap_plan:
        assert plan, "One or more plans is empty"
        assert plan.candidate_clusters, "candidate_clusters is empty"
        assert plan.candidate_clusters.zonal, "candidate_clusters.zonal is empty"
        assert len(plan.candidate_clusters.zonal) == 9, (
            "len(candidate_clusters.zonal) != 9"
        )

        cluster_type_counts = Counter(
            zone.cluster_type for zone in plan.candidate_clusters.zonal
        )

        assert len(cluster_type_counts) == 3, "Expecting 3 cluster types"
        assert cluster_type_counts["elasticsearch-search"] == 3, (
            "Expecting exactly 3 search nodes"
        )
        assert cluster_type_counts["elasticsearch-master"] == 3, (
            "Expecting exactly 3 master nodes"
        )
        assert cluster_type_counts["elasticsearch-data"] >= 3, (
            "Expecting at least 3 data nodes"
        )

        # Verify total disk space for elasticsearch-data nodes
        # exceeds expected_allocated_disk_size_gib
        total_allocated_disk_gib = sum(
            zone.count
            * (
                sum(dr.size_gib for dr in zone.attached_drives)
                + (
                    zone.instance.drive.size_gib
                    if zone.instance.drive is not None
                    else 0
                )
            )
            for zone in plan.candidate_clusters.zonal
            if zone.cluster_type == "elasticsearch-data"
        )
        assert total_allocated_disk_gib >= expected_allocated_disk_size_gib, (
            f"Total disk space for elasticsearch-data nodes "
            f"({total_allocated_disk_gib} GiB) must be greater than "
            f"{expected_allocated_disk_size_gib} GiB"
        )


def zonal_summary(zlr):
    zlr_cpu = zlr.count * zlr.instance.cpu
    zlr_cost = zlr.annual_cost
    zlr_family = zlr.instance.family
    zlr_instance_name = zlr.instance.name
    zlr_drive_gib = sum(dr.size_gib for dr in zlr.attached_drives)
    if zlr.instance.drive is not None:
        zlr_drive_gib += zlr.instance.drive.size_gib
    zlr_drive_gib *= zlr.count

    return (
        zlr_family,
        zlr_instance_name,
        zlr.count,
        zlr_cpu,
        zlr_cost,
        zlr_drive_gib,
    )

"""Tests for Cassandra interchangeable zonal plans."""

import pytest

from service_capacity_modeling.capacity_planner import PlannerArguments, planner
from service_capacity_modeling.interface import (
    CapacityDesires,
    CurrentClusters,
    CurrentZoneClusterCapacity,
    DataShape,
    QueryPattern,
    certain_float,
    certain_int,
)
from service_capacity_modeling.models import RANK_PENALTIES
from service_capacity_modeling.models.interchangeable import (
    InterchangeableOptions,
    plan_interchangeable,
)
from service_capacity_modeling.models.org.netflix.cassandra_interchangeable import (
    CassandraInterchangeable,
)
from service_capacity_modeling.models.org.netflix.interchangeable_models import (
    INTERCHANGEABLE_MODELS,
)

MODEL = "org.netflix.cassandra"
REGION = "us-east-1"
FAMILIES = ["m6id", "r6id", "i4i", "m6i", "r6i"]
REGRET = 1.0
DESIRES = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(60_000),
        estimated_write_per_second=certain_int(60_000),
    ),
    data_shape=DataShape(estimated_state_size_gib=certain_int(4_000)),
)


def _with_current(disk_utilization_gib):
    """DESIRES with an existing i4i cluster given only by instance name."""
    current = CurrentZoneClusterCapacity(
        cluster_instance_name="i4i.4xlarge",
        cluster_instance_count=certain_int(8),
        cpu_utilization=certain_float(30),
        disk_utilization_gib=certain_float(disk_utilization_gib),
    )
    return DESIRES.model_copy(
        update={"current_clusters": CurrentClusters(zonal=[current] * 3)}
    )


def _singles(args, desires=DESIRES):
    return planner.plan_certain(
        model_name=MODEL,
        region=REGION,
        desires=desires,
        extra_model_arguments=args,
        instance_families=FAMILIES,
        num_results=10_000,
        # plan_interchangeable plans without the preferred-family penalty.
        planner_arguments=PlannerArguments(
            max_results_per_family=10_000, preferred_family_penalty=0
        ),
    )


def _model(args, desires=DESIRES):
    candidates = {}
    for plan in _singles(args, desires):
        candidates.setdefault(plan.candidate_clusters.zonal[0].instance.name, plan)
    return candidates, CassandraInterchangeable(candidates, desires, args)


@pytest.mark.parametrize("args", [{}, {"cluster_size_mode": "doubling"}])
def test_single_type_preserves_planner_layout_and_rank(args):
    candidates, model = _model(args)
    layouts = model.layouts()
    assert candidates
    for name, original in candidates.items():
        layout = next(layout for layout in layouts if model.serves(name, layout))
        resized = model.resize(original, layout, [name])
        assert (
            resized.candidate_clusters.zonal[0].count
            == original.candidate_clusters.zonal[0].count
        )
        assert (
            resized.candidate_clusters.zonal[0].attached_drives
            == original.candidate_clusters.zonal[0].attached_drives
        )
        assert resized.rank == pytest.approx(original.rank)


# A current cluster adds the family-migration penalty, and one that reports no
# disk usage makes the model estimate data size from the desires instead.
@pytest.mark.parametrize(
    "desires",
    [DESIRES, _with_current(0), _with_current(300)],
    ids=["new", "current-no-disk", "current-with-disk"],
)
def test_returned_types_pass_planner_at_shared_count_and_configuration(desires):
    args = {}
    result = plan_interchangeable(
        planner,
        MODEL,
        INTERCHANGEABLE_MODELS[MODEL],
        REGION,
        desires,
        args,
        # A high regret makes local-disk plans grow to reach more families.
        InterchangeableOptions(max_fallback_types=2, missing_fallback_regret=REGRET),
        instance_families=FAMILIES,
        num_results=100,
        max_results_per_family=100,
    )
    assert result.plans
    assert any(item.fallback_instance_types for item in result.plans)
    assert any(
        item.fallback_instance_types
        and item.plan.candidate_clusters.zonal[0].attached_drives
        for item in result.plans
    )
    assert any(
        item.fallback_instance_types
        and item.plan.candidate_clusters.zonal[0].count > item.single_type_node_count
        for item in result.plans
    )
    plans_by_count = {}
    for item in result.plans:
        cluster = item.plan.candidate_clusters.zonal[0]
        if cluster.count not in plans_by_count:
            at_count = _singles({"required_cluster_size": cluster.count}, desires)
            plans_by_count[cluster.count] = {
                p.candidate_clusters.zonal[0].instance.name: p for p in at_count
            }
        by_name = plans_by_count[cluster.count]
        for name in [item.preferred_instance_type, *item.fallback_instance_types]:
            assert name in by_name
            checked = by_name[name].candidate_clusters.zonal[0]
            assert checked.count == cluster.count
            assert (
                checked.cluster_params["cassandra.heap.gib"]
                == cluster.cluster_params["cassandra.heap.gib"]
            )
            assert (
                checked.cluster_params["cassandra.compaction.min_threshold"]
                == cluster.cluster_params["cassandra.compaction.min_threshold"]
            )
            if cluster.attached_drives:
                assert checked.attached_drives == cluster.attached_drives
            else:
                assert not checked.attached_drives
                assert (
                    checked.instance.drive.drive_type
                    == cluster.instance.drive.drive_type
                )
        zones = item.plan.candidate_clusters.zonal
        assert all(zone.count == cluster.count for zone in zones)
        assert float(
            item.plan.candidate_clusters.annual_costs["cassandra.zonal-clusters"]
        ) == pytest.approx(sum(zone.annual_cost for zone in zones))
        missing = 2 - len(item.fallback_instance_types)
        assert item.plan.rank == pytest.approx(
            by_name[item.preferred_instance_type].rank * (1 + REGRET * missing)
        )


def test_resizing_recomputes_density_penalty():
    desires = DESIRES.model_copy(deep=True)
    desires.data_shape.estimated_state_size_gib = certain_int(30_000)
    candidates, model = _model({}, desires)
    for name, plan in candidates.items():
        cluster = plan.candidate_clusters.zonal[0]
        penalty = cluster.cluster_params.get(RANK_PENALTIES, {}).get(
            "ephemeral_maintenance", 0
        )
        if penalty <= 0.2 or cluster.count >= 64:
            continue
        layout = next(
            layout
            for layout in model.layouts()
            if layout.zone_count == cluster.count + 1 and model.serves(name, layout)
        )
        resized = model.resize(plan, layout, [name])
        oracle = next(
            p
            for p in _singles({"required_cluster_size": layout.zone_count}, desires)
            if p.candidate_clusters.zonal[0].instance.name == name
        )
        assert resized.rank == pytest.approx(oracle.rank)
        assert (
            resized.candidate_clusters.zonal[0].cluster_params[RANK_PENALTIES][
                "ephemeral_maintenance"
            ]
            < penalty
        )
        return
    pytest.fail("workload did not produce a resizable dense local-disk plan")

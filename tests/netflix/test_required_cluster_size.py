"""required_cluster_size for EVCache and Elasticsearch data, as for Cassandra."""

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import certain_int

REGION = "us-east-1"
DESIRES = {
    "org.netflix.evcache": CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(500_000),
            estimated_write_per_second=certain_int(50_000),
            estimated_mean_read_size_bytes=certain_int(2_000),
            estimated_mean_write_size_bytes=certain_int(2_000),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_int(600)),
    ),
    "org.netflix.elasticsearch.node": CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(10_000),
            estimated_write_per_second=certain_int(5_000),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_int(2_000)),
    ),
}


def _plan(model, args):
    return planner.plan_certain_explained(
        model_name=model,
        region=REGION,
        desires=DESIRES[model],
        extra_model_arguments=args,
        num_results=100,
        max_results_per_family=100,
    )


@pytest.mark.parametrize("model", list(DESIRES))
def test_every_plan_has_the_required_zone_size(model):
    best = _plan(model, {}).plans[0].candidate_clusters.zonal[0].count
    required = best + 3
    plans = _plan(model, {"required_cluster_size": required}).plans
    assert plans
    assert all(
        zone.count == required for p in plans for zone in p.candidate_clusters.zonal
    )


@pytest.mark.parametrize("model", list(DESIRES))
def test_a_size_too_small_for_the_workload_is_excused(model):
    result = _plan(model, {"required_cluster_size": 1})
    excuses = [e for es in result.excuses_by_model.values() for e in es]
    assert any(e.bottleneck == Bottleneck.cluster_size for e in excuses)

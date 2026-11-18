import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import CurrentClusters
from service_capacity_modeling.interface import CurrentZoneClusterCapacity
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import Service

ATLAS_COST_PER_SERIES = 0.00315
DESIRES = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(20_000),
        estimated_write_per_second=certain_int(20_000),
    ),
    data_shape=DataShape(estimated_state_size_gib=certain_int(500)),
)


@pytest.fixture
def priced_atlas(monkeypatch):
    services = planner.hardware_shapes.region("us-east-1").services
    monkeypatch.setitem(
        services,
        "atlas.series",
        Service(name="Atlas time series", annual_cost_per_series=ATLAS_COST_PER_SERIES),
    )


def _plan(extra_model_arguments):
    return planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=DESIRES,
        extra_model_arguments=extra_model_arguments,
        num_results=1,
    )[0]


def test_public_pricing_leaves_atlas_unpriced():
    plan = _plan({})
    assert "cassandra.atlas.series" not in plan.candidate_clusters.annual_costs


@pytest.mark.usefixtures("priced_atlas")
def test_atlas_cost_scales_with_planned_nodes_and_tables():
    plan = _plan({"table_count": 11})
    clusters = plan.candidate_clusters
    node_count = sum(cluster.count for cluster in clusters.zonal)
    # 4,085 node-level series plus 52 system and 11 application tables at
    # 208 series each.
    series_per_node = 4085 + 63 * 208
    assert float(clusters.annual_costs["cassandra.atlas.series"]) == pytest.approx(
        node_count * series_per_node * ATLAS_COST_PER_SERIES
    )

    more_tables = _plan({"table_count": 111}).candidate_clusters
    assert (
        more_tables.annual_costs["cassandra.atlas.series"]
        > (clusters.annual_costs["cassandra.atlas.series"])
    )


@pytest.mark.usefixtures("priced_atlas")
def test_baseline_extraction_prices_current_nodes():
    zone = CurrentZoneClusterCapacity(
        cluster_instance_name="m7a.2xlarge",
        cluster_instance_count=certain_int(32),
        cluster_type="cassandra",
    )
    desires = DESIRES.model_copy(
        update={"current_clusters": CurrentClusters(zonal=[zone] * 3)}
    )

    baseline = planner.extract_baseline_plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
        extra_model_arguments={"table_count": 11},
    )

    # 96 nodes x 17,189 series is about 1.65M series, close to the 1.645M
    # Atlas reports for the cluster these constants were measured on.
    atlas = {s.service_type: s for s in baseline.candidate_clusters.services}[
        "cassandra.atlas.series"
    ]
    assert atlas.service_params == {
        "node_count": 96,
        "table_count": 11,
        "series_per_node": 17_189,
    }
    assert atlas.annual_cost == pytest.approx(96 * 17_189 * ATLAS_COST_PER_SERIES)

"""Tests for what a composed model inherits from its parent's desires."""

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern

# Sized to sit near the memory bound of the backing datastore, so a leaked
# reserve visibly changes the shape it picks. A disk-bound namespace would
# hide the difference.
BASE_QUERY_PATTERN = QueryPattern(
    estimated_read_per_second=Interval(low=500, mid=5000, high=50_000, confidence=0.98),
    estimated_write_per_second=Interval(
        low=250, mid=2500, high=25_000, confidence=0.98
    ),
)
BASE_STATE_SIZE = Interval(low=20, mid=200, high=2000, confidence=0.98)

# Models that plan a datastore of their own on separate shapes. The parent's
# app memory reserve describes the parent's instances, not theirs.
COMPOSED_MODELS = [
    "org.netflix.key-value",
    "org.netflix.time-series",
    "org.netflix.graphkv",
    "org.netflix.entity",
]


def _desires(app_mem_gib: float) -> CapacityDesires:
    return CapacityDesires(
        service_tier=1,
        query_pattern=BASE_QUERY_PATTERN.model_copy(deep=True),
        data_shape=DataShape(
            estimated_state_size_gib=BASE_STATE_SIZE,
            reserved_instance_app_mem_gib=app_mem_gib,
        ),
    )


def _clusters_by_type(model_name: str, app_mem_gib: float) -> dict:
    plan = planner.plan_certain(
        model_name=model_name, region="us-east-1", desires=_desires(app_mem_gib)
    )[0]
    clusters = list(plan.candidate_clusters.zonal) + list(
        plan.candidate_clusters.regional
    )
    return {c.cluster_type: f"{c.instance.name}x{c.count}" for c in clusters}


@pytest.mark.parametrize("model_name", COMPOSED_MODELS)
def test_app_memory_reserve_does_not_reach_composed_models(model_name):
    """A parent's app memory reserve stops at the parent's own tier.

    reserved_instance_app_mem_gib is memory per instance of the tier it was
    written for. Composed models run on their own shapes, so inheriting the
    value made them hold back memory for an app that is not on the box --
    shrinking page cache and pushing them onto larger instances.
    """
    lean = _clusters_by_type(model_name, 4)
    heavy = _clusters_by_type(model_name, 24)

    datastore_types = set(lean) - {"dgwkv", "dgwts", "dgwgraphkv", "dgwentity"}
    assert datastore_types, f"{model_name} composed no datastore"

    for cluster_type in datastore_types:
        assert heavy[cluster_type] == lean[cluster_type], (
            f"{model_name}: {cluster_type} changed with the parent's app reserve"
        )


def test_directly_planned_models_keep_the_caller_reserve():
    """Addressing a datastore by name still applies the caller's reserve.

    Only inheritance across a composition boundary is dropped. Someone who
    plans Cassandra directly is describing Cassandra's own instances.
    """
    lean = _clusters_by_type("org.netflix.cassandra", 4)
    heavy = _clusters_by_type("org.netflix.cassandra", 24)

    assert heavy["cassandra"] != lean["cassandra"]


def test_parent_tier_still_gets_the_caller_reserve():
    """The reserve is dropped for children, not for the model asked for."""
    lean = _clusters_by_type("org.netflix.key-value", 4)
    heavy = _clusters_by_type("org.netflix.key-value", 24)

    assert heavy["dgwkv"] != lean["dgwkv"]


def test_transforms_compose_through_more_than_one_level():
    """GraphKV fans out to KeyValue, which fans out to Cassandra.

    Each compose_with transform used to receive the original user desires, so
    a two-level chain dropped the middle one: Cassandra never saw GraphKV's
    read and write amplification and was sized for logical graph traffic
    rather than the backend KeyValue operations it actually serves.
    """
    by_model = dict(
        planner._sub_models(  # pylint: disable=protected-access
            "org.netflix.graphkv", _desires(24), extra_model_arguments={}
        )
    )

    logical = by_model["org.netflix.graphkv"].query_pattern
    backend = by_model["org.netflix.cassandra"].query_pattern

    assert backend.estimated_read_per_second.mid > (
        logical.estimated_read_per_second.mid * 10
    )
    assert backend.estimated_write_per_second.mid > (
        logical.estimated_write_per_second.mid
    )
    assert by_model["org.netflix.cassandra"].data_shape.estimated_state_size_gib.mid > (
        by_model["org.netflix.graphkv"].data_shape.estimated_state_size_gib.mid
    )


def test_the_rule_holds_through_an_aggregator():
    """org.netflix.elasticsearch owns no instances; it splits into node roles.

    The reserve still stops at the boundary rather than reaching the node
    roles through it. Entity is the only composition that reaches
    Elasticsearch in practice, and it strips one level higher anyway, so this
    keeps the two paths agreeing instead of carving out a second mode.
    """
    lean = _clusters_by_type("org.netflix.elasticsearch", 1)
    heavy = _clusters_by_type("org.netflix.elasticsearch", 64)

    assert heavy["elasticsearch-data"] == lean["elasticsearch-data"]

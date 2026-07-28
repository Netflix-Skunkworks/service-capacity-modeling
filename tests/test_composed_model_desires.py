"""Tests for what a composed model inherits from its parent's desires."""

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import CurrentClusters
from service_capacity_modeling.interface import CurrentRegionClusterCapacity
from service_capacity_modeling.interface import CurrentZoneClusterCapacity
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.org.netflix.elasticsearch import (
    nflx_elasticsearch_capacity_model,
)

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


def test_aggregators_pass_the_reserve_to_their_node_roles():
    """A model that owns no instances is not a tier boundary.

    org.netflix.elasticsearch plans nothing itself; it splits Elasticsearch
    into data, master and search nodes. The caller's reserve describes those
    nodes, so dropping it there would discard an explicit input rather than
    stop a leak.
    """
    assert not nflx_elasticsearch_capacity_model.plans_own_cluster()

    lean = _clusters_by_type("org.netflix.elasticsearch", 1)
    heavy = _clusters_by_type("org.netflix.elasticsearch", 64)

    assert heavy["elasticsearch-data"] != lean["elasticsearch-data"]


def _current_zonal(cluster_type=None):
    return CurrentZoneClusterCapacity(
        cluster_instance_name="m5.xlarge",
        cluster_instance_count=certain_int(3),
        cluster_type=cluster_type,
        cpu_utilization=certain_int(50),
        memory_utilization_gib=certain_int(8),
        network_utilization_mbps=certain_int(100),
        disk_utilization_gib=certain_int(100),
    )


def _current_regional(cluster_type=None):
    return CurrentRegionClusterCapacity(
        cluster_instance_name="m5.xlarge",
        cluster_instance_count=certain_int(3),
        cluster_type=cluster_type,
        cpu_utilization=certain_int(50),
        memory_utilization_gib=certain_int(8),
        network_utilization_mbps=certain_int(100),
        disk_utilization_gib=certain_int(100),
    )


def _current_types(desires):
    current = desires.current_clusters
    assert current is not None
    return {cluster.cluster_type for cluster in (*current.zonal, *current.regional)}


def _submodels(model_name, desires, extra_model_arguments=None):
    return dict(
        planner._sub_models(  # pylint: disable=protected-access
            model_name,
            desires,
            extra_model_arguments=extra_model_arguments or {},
        )
    )


def test_composed_models_receive_only_their_current_cluster_type():
    desires = _desires(24)
    desires.current_clusters = CurrentClusters(
        zonal=[_current_zonal("evcache"), _current_zonal("cassandra")],
        regional=[_current_regional("dgwkv")],
    )

    by_model = _submodels(
        "org.netflix.key-value",
        desires,
        extra_model_arguments={"kv_force_evcache": True},
    )

    assert _current_types(by_model["org.netflix.key-value"]) == {"dgwkv"}
    assert _current_types(by_model["org.netflix.cassandra"]) == {"cassandra"}
    assert _current_types(by_model["org.netflix.evcache"]) == {"evcache"}


def test_composite_plan_accepts_typed_regional_and_zonal_current_clusters():
    desires = _desires(24)
    desires.current_clusters = CurrentClusters(
        zonal=[_current_zonal("cassandra")],
        regional=[_current_regional("dgwkv")],
    )

    plans = planner.plan_certain(
        model_name="org.netflix.key-value",
        region="us-east-1",
        desires=desires,
        num_results=1,
    )

    assert plans


def test_role_split_models_filter_current_clusters_by_role():
    desires = _desires(1)
    desires.current_clusters = CurrentClusters(
        zonal=[
            _current_zonal("elasticsearch-data"),
            _current_zonal("elasticsearch-master"),
            _current_zonal("elasticsearch-search"),
        ]
    )

    by_model = _submodels("org.netflix.elasticsearch", desires)

    assert _current_types(by_model["org.netflix.elasticsearch.node"]) == {
        "elasticsearch-data"
    }
    assert _current_types(by_model["org.netflix.elasticsearch.master"]) == {
        "elasticsearch-master"
    }
    assert _current_types(by_model["org.netflix.elasticsearch.search"]) == {
        "elasticsearch-search"
    }


def test_untyped_current_cluster_remains_available_to_direct_model():
    desires = _desires(4)
    desires.current_clusters = CurrentClusters(zonal=[_current_zonal()])

    by_model = _submodels("org.netflix.cassandra", desires)

    assert _current_types(by_model["org.netflix.cassandra"]) == {None}


def test_multilevel_composition_preserves_ancestor_transforms():
    desires = _desires(4)

    by_model = _submodels("org.netflix.graphkv", desires)

    key_value = by_model["org.netflix.key-value"].query_pattern
    cassandra = by_model["org.netflix.cassandra"].query_pattern
    assert key_value.estimated_read_per_second.mid == 55 * 5000
    assert key_value.estimated_write_per_second.mid == 3 * 2500
    assert (
        cassandra.estimated_read_per_second.mid
        == key_value.estimated_read_per_second.mid
    )
    assert (
        cassandra.estimated_write_per_second.mid
        == key_value.estimated_write_per_second.mid
    )

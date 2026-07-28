"""Tests for picking a model's own entry out of desires.current_clusters."""

from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import CurrentClusters
from service_capacity_modeling.interface import CurrentRegionClusterCapacity
from service_capacity_modeling.interface import CurrentZoneClusterCapacity
from service_capacity_modeling.models.common import current_cluster_capacity


def _zonal(name: str, cluster_type=None, count: int = 3):
    return CurrentZoneClusterCapacity(
        cluster_instance_name=name,
        cluster_type=cluster_type,
        cluster_instance_count=certain_int(count),
        cpu_utilization=certain_float(10),
    )


def _regional(name: str, cluster_type=None, count: int = 3):
    return CurrentRegionClusterCapacity(
        cluster_instance_name=name,
        cluster_type=cluster_type,
        cluster_instance_count=certain_int(count),
        cpu_utilization=certain_float(10),
    )


def _desires(**kwargs) -> CapacityDesires:
    return CapacityDesires(service_tier=1, current_clusters=CurrentClusters(**kwargs))


def test_picks_the_entry_matching_the_model():
    """Cassandra and EVCache are both zonal in a composed KeyValue plan."""
    desires = _desires(
        zonal=[
            _zonal("i7ie.large", cluster_type="evcache", count=14),
            _zonal("i4i.2xlarge", cluster_type="cassandra", count=4),
        ],
        regional=[_regional("r8i.xlarge", cluster_type="dgwkv")],
    )

    assert current_cluster_capacity(desires, "cassandra").cluster_instance_name == (
        "i4i.2xlarge"
    )
    assert current_cluster_capacity(desires, "evcache").cluster_instance_name == (
        "i7ie.large"
    )
    assert current_cluster_capacity(desires, "dgwkv").cluster_instance_name == (
        "r8i.xlarge"
    )


def test_unlabelled_entries_fall_back_to_the_first():
    """Callers that predate cluster_type keep the old behaviour."""
    desires = _desires(zonal=[_zonal("i4i.2xlarge"), _zonal("i3en.xlarge")])

    picked = current_cluster_capacity(desires, "cassandra")
    assert picked.cluster_instance_name == "i4i.2xlarge"


def test_labelled_but_unmatched_means_no_current_cluster():
    """If the caller labelled other tiers and not ours, we have none.

    Falling back to the first entry here is what let Cassandra size itself
    against EVCache's topology.
    """
    desires = _desires(zonal=[_zonal("i7ie.large", cluster_type="evcache")])

    assert current_cluster_capacity(desires, "cassandra") is None


def test_zonal_is_preferred_over_regional_for_the_legacy_fallback():
    desires = _desires(
        zonal=[_zonal("i4i.2xlarge")], regional=[_regional("r8i.xlarge")]
    )

    picked = current_cluster_capacity(desires, "cassandra")
    assert picked.cluster_instance_name == "i4i.2xlarge"


def test_no_current_clusters_at_all():
    assert (
        current_cluster_capacity(CapacityDesires(service_tier=1), "cassandra") is None
    )
    assert current_cluster_capacity(_desires(), "cassandra") is None

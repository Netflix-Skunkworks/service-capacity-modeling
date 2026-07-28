"""Tests for picking a model's own entry out of desires.current_clusters."""

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import CurrentClusters
from service_capacity_modeling.interface import CurrentRegionClusterCapacity
from service_capacity_modeling.interface import CurrentZoneClusterCapacity
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.common import current_cluster_capacity
from service_capacity_modeling.models.org.netflix.elasticsearch import (
    nflx_elasticsearch_capacity_model,
)
from service_capacity_modeling.models.org.netflix.postgres import (
    nflx_postgres_capacity_model,
)


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


def test_a_sibling_being_labelled_does_not_discard_my_unlabelled_entry():
    """Half-labelled payloads are what a cluster_type rollout looks like.

    Antigravity may start labelling one tier before another. Deciding whether
    my entry is trustworthy by looking at labels on *other* entries throws
    mine away for no reason -- Cassandra would lose its node count, disk
    utilization and reference shape because a sibling got labelled first.
    """
    desires = _desires(
        zonal=[_zonal("i4i.2xlarge"), _zonal("i7ie.large", cluster_type="evcache")]
    )

    picked = current_cluster_capacity(desires, "cassandra")
    assert picked is not None
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


def test_a_facade_model_answers_to_each_backend_label():
    """Postgres plans an Aurora cluster, so its existing one carries that label.

    A single cluster_type cannot express that, which is why the selector takes
    several and the model declares them.
    """
    assert set(nflx_postgres_capacity_model.current_cluster_types()) == {
        "aurora-cluster",
        "rds-cluster",
    }

    for label in ("aurora-cluster", "rds-cluster"):
        desires = _desires(regional=[_regional("db.r6g.xlarge", cluster_type=label)])
        picked = current_cluster_capacity(
            desires, *nflx_postgres_capacity_model.current_cluster_types()
        )
        assert picked is not None, f"{label} should match Postgres"
        assert picked.cluster_instance_name == "db.r6g.xlarge"


def test_a_model_declaring_no_label_matches_nothing_by_name():
    """An aggregator owns no clusters, so it has no label to match on.

    It still gets the unlabelled fallback, which is what callers had before
    cluster_type was read at all.
    """
    assert not nflx_elasticsearch_capacity_model.current_cluster_types()

    labelled = _desires(zonal=[_zonal("i4i.2xlarge", cluster_type="cassandra")])
    assert current_cluster_capacity(labelled) is None

    unlabelled = _desires(zonal=[_zonal("i4i.2xlarge")])
    assert current_cluster_capacity(unlabelled) is not None


def test_no_current_clusters_at_all():
    assert (
        current_cluster_capacity(CapacityDesires(service_tier=1), "cassandra") is None
    )
    assert current_cluster_capacity(_desires(), "cassandra") is None


def _cassandra_desires(zonal):
    return CapacityDesires(
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
                low=100, mid=1000, high=4000, confidence=0.98
            )
        ),
        current_clusters=CurrentClusters(zonal=zonal),
    )


def _live_cluster(instance_name, cluster_type, count, disk_gib):
    return CurrentZoneClusterCapacity(
        cluster_instance_name=instance_name,
        cluster_instance=shapes.instance(instance_name),
        cluster_type=cluster_type,
        cluster_drive=Drive(name="gp3", drive_type="attached-ssd", size_gib=1000),
        cluster_instance_count=certain_int(count),
        cpu_utilization=certain_float(20),
        disk_utilization_gib=certain_float(disk_gib),
        network_utilization_mbps=certain_float(100),
    )


def test_cassandra_ignores_another_tier_in_a_composed_request():
    """EVCache listed first must not become Cassandra's current cluster.

    Both are zonal, so a composed KeyValue request puts them in the same
    list. Reading index 0 meant Cassandra sized itself against EVCache's
    node count and disk.
    """
    cassandra_only = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=_cassandra_desires(
            [_live_cluster("i4i.2xlarge", "cassandra", count=4, disk_gib=200)]
        ),
    )
    with_evcache_first = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=_cassandra_desires(
            [
                _live_cluster("i7ie.large", "evcache", count=40, disk_gib=900),
                _live_cluster("i4i.2xlarge", "cassandra", count=4, disk_gib=200),
            ]
        ),
    )

    assert cassandra_only and with_evcache_first
    assert len(cassandra_only) == len(with_evcache_first)
    for own, composed in zip(cassandra_only, with_evcache_first):
        own_zone = own.candidate_clusters.zonal[0]
        composed_zone = composed.candidate_clusters.zonal[0]
        assert own_zone.instance.name == composed_zone.instance.name
        assert own_zone.count == composed_zone.count


def test_evcache_ignores_another_tier_in_a_composed_request():
    """The other half of the same collision.

    Cassandra and EVCache are both zonal, so whichever the caller listed first
    was the entry both models read. Fixing only Cassandra leaves EVCache
    sizing itself off Cassandra's node count and disk.
    """
    evcache_only = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=_cassandra_desires(
            [_live_cluster("i7ie.large", "evcache", count=6, disk_gib=100)]
        ),
    )
    with_cassandra_first = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=_cassandra_desires(
            [
                _live_cluster("i4i.2xlarge", "cassandra", count=40, disk_gib=900),
                _live_cluster("i7ie.large", "evcache", count=6, disk_gib=100),
            ]
        ),
    )

    assert evcache_only and with_cassandra_first
    assert len(evcache_only) == len(with_cassandra_first)
    for own, composed in zip(evcache_only, with_cassandra_first):
        own_zone = own.candidate_clusters.zonal[0]
        composed_zone = composed.candidate_clusters.zonal[0]
        assert own_zone.instance.name == composed_zone.instance.name
        assert own_zone.count == composed_zone.count

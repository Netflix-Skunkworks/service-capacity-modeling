from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern


uncertain_mid = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(
            low=1000, mid=10000, high=100000, confidence=0.98
        ),
        estimated_write_per_second=Interval(
            low=1000, mid=10000, high=100000, confidence=0.98
        ),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(low=100, mid=500, high=1000, confidence=0.98),
    ),
)


def test_repeated_plans():
    results = []
    for _ in range(5):
        results.append(
            planner.plan(
                model_name="org.netflix.cassandra",
                region="us-east-1",
                desires=uncertain_mid,
            ).model_dump_json()
        )

    a = [hash(x) for x in results]
    # We should end up with consistent results
    assert all(i == a[0] for i in a)


def test_compositional():
    direct_result = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=uncertain_mid,
        num_results=4,
        explain=True,
    )
    composed_result = planner.plan(
        model_name="org.netflix.key-value",
        region="us-east-1",
        desires=uncertain_mid,
        num_results=4,
        explain=True,
    )

    count = len(direct_result.least_regret)
    direct_clusters = []
    for i in range(count):
        direct_cluster = direct_result.least_regret[i].candidate_clusters.zonal[0]
        direct_clusters.append(direct_cluster)
        # FIXME(josephl): It appears that since we are now zipping the
        # regional and zonal clusters we can get repeats in the zonal.
        # This is odd to me but not related to the 6th gen instances
        composed_cluster = composed_result.least_regret[i].candidate_clusters.zonal[0]
        assert composed_cluster in direct_clusters

        java = composed_result.least_regret[i].candidate_clusters.regional[0]
        assert java.cluster_type == "dgwkv"
        # usually like 15 * 4 = ~50
        assert 100 > java.count * java.instance.cpu > 20


def test_multiple_options():
    result = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=uncertain_mid,
        num_results=4,
        simulations=6,
    )
    least_regret = result.least_regret
    # With only 128 simulations we have < 4 instance families
    assert len(least_regret) < 4
    families = [lr.candidate_clusters.zonal[0].instance.family for lr in least_regret]
    for f in families:
        assert f in {"i3en", "m6id", "m5d", "i4i", "i3", "c6id", "c5d"}

    # With 1024 simulations we get a 4th instance family (r5)
    result = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=uncertain_mid,
        num_results=4,
        simulations=1024,
    )
    least_regret = result.least_regret
    assert len(least_regret) == 4

    families = [lr.candidate_clusters.zonal[0].instance.family for lr in least_regret]
    for f in families:
        assert f[0] in ("i", "r", "c", "m")

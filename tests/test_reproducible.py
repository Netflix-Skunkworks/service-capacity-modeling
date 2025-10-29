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


def test_multiple_options_diversify_with_more_simulations():
    """
    This test appears strange at first. The goal is to show that with less
    simulations we would see a smaller subset of the diverse world of outputs
    than with more simulations
    """

    # These values happen to work today but may not work in the future with
    # changes to the CP inputs (instances, costs, performance).
    # Feel free to change the numbers as long as it fits the below assertion
    arbitrary_num_results = 8
    arbitrary_small_number = 12
    arbitrary_large_number = 1024
    assert arbitrary_small_number < arbitrary_large_number

    less_simulations_result = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=uncertain_mid,
        num_results=arbitrary_num_results,
        simulations=arbitrary_small_number,
    )
    more_simulations_result = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=uncertain_mid,
        num_results=arbitrary_num_results,
        simulations=arbitrary_large_number,
    )

    # Potentially brittle assertion. This is the part likely to break
    # The idea is that we should see more options with more simulations.
    less_simulations_famlies = {
        lr.candidate_clusters.zonal[0].instance.family
        for lr in less_simulations_result.least_regret
    }
    more_simulations_families = {
        lr.candidate_clusters.zonal[0].instance.family
        for lr in more_simulations_result.least_regret
    }
    assert len(less_simulations_famlies) < len(more_simulations_families)

    expected_family_types = {"i", "r", "c", "m"}
    for f in less_simulations_famlies:
        assert f[0] in expected_family_types
    for f in more_simulations_families:
        assert f[0] in expected_family_types

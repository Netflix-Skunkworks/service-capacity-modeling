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
    """Test that key-value composition produces identical Cassandra plans.

    The key-value model composes with Cassandra via `lambda x: x` (identity),
    meaning the Cassandra sub-model must receive identical inputs and produce
    byte-for-byte identical outputs. This is the strictest possible test of
    compositional correctness.

    Note: The final least_regret results may differ due to reduce_by_family()
    filtering across both regional and zonal dimensions, but that is a
    presentation concern - the underlying Cassandra planning must be identical.
    """
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

    # Strictest test: Cassandra regret clusters must be EXACTLY identical
    # (same plans, same regrets, same order) since key-value uses `lambda x: x`
    direct_cass = direct_result.explanation.regret_clusters_by_model[
        "org.netflix.cassandra"
    ]
    composed_cass = composed_result.explanation.regret_clusters_by_model[
        "org.netflix.cassandra"
    ]
    assert len(direct_cass) == len(composed_cass)
    for i, ((d_plan, _, d_regret), (c_plan, _, c_regret)) in enumerate(
        zip(direct_cass, composed_cass)
    ):
        assert d_plan == c_plan, f"Plan {i} differs"
        assert d_regret == c_regret, f"Regret {i} differs: {d_regret} vs {c_regret}"

    # Verify the composed results have the expected structure
    for lr in composed_result.least_regret:
        # Zonal cluster should be Cassandra
        assert lr.candidate_clusters.zonal[0].cluster_type == "cassandra"
        # Regional cluster should be the key-value Java app
        java = lr.candidate_clusters.regional[0]
        assert java.cluster_type == "dgwkv"
        # Sanity check on Java app sizing (~48 total CPUs: 6 x 8 vCPU instances,
        # but may vary with CPU architecture or pricing improvements)
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
    arbitrary_num_results = 12
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

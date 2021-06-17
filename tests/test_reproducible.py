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
            ).json()
        )

    a = [hash(x) for x in results]
    # We should end up with consistent results
    assert all(i == a[0] for i in a)


def test_multiple_options():
    result = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=uncertain_mid,
        num_results=4,
        simulations=24,
    )
    least_regret = result.least_regret
    # With only 128 simulations we only have 3 instance families
    assert len(least_regret) == 3
    families = [lr.candidate_clusters.zonal[0].instance.family for lr in least_regret]
    assert set(families) == set(("r5", "m5d", "m5"))

    # With 1024 simulations we get a 4th instance family (i3en)
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
    assert set(families) == set(("r5", "i3en", "m5d", "m5"))

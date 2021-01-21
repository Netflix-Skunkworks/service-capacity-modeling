from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern


uncertain_mid = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(
            low=1000, mid=10000, high=100000, confidence=0.9
        ),
        estimated_write_per_second=Interval(
            low=1000, mid=10000, high=100000, confidence=0.9
        ),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(low=100, mid=500, high=1000, confidence=0.9),
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

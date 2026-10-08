import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern


@pytest.mark.parametrize(
    "state_size_gib,writes_per_second,expected_instances",
    [
        (
            9,
            999,
            {
                "rawhollow-logkeeper": "r6a.large",
                "rawhollow-writer": "r6a.xlarge",
                "rawhollow-producer": "r6a.2xlarge",
                "rawhollow-explorer": "r6a.large",
            },
        ),
        (
            10,
            1_000,
            {
                "rawhollow-logkeeper": "r6a.xlarge",
                "rawhollow-writer": "r6a.2xlarge",
                "rawhollow-producer": "r6a.2xlarge",
                "rawhollow-explorer": "r6a.large",
            },
        ),
        (
            100,
            1_000,
            {
                "rawhollow-logkeeper": "r6a.xlarge",
                "rawhollow-writer": "r6a.2xlarge",
                "rawhollow-producer": "r6a.4xlarge",
                "rawhollow-explorer": "r6a.xlarge",
            },
        ),
    ],
)
def test_raw_hollow_preserves_role_sizing_thresholds(
    state_size_gib: int,
    writes_per_second: int,
    expected_instances: dict[str, str],
):
    desires = CapacityDesires(
        query_pattern=QueryPattern(
            estimated_write_per_second=certain_int(writes_per_second),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=certain_int(state_size_gib),
        ),
    )

    plans = planner.plan_certain(
        model_name="org.netflix.raw-hollow",
        region="us-east-1",
        desires=desires,
    )

    assert len(plans) == 1
    clusters = {
        cluster.cluster_type: cluster
        for cluster in plans[0].candidate_clusters.regional
    }
    assert {role: cluster.instance.name for role, cluster in clusters.items()} == (
        expected_instances
    )
    assert {role: cluster.count for role, cluster in clusters.items()} == {
        "rawhollow-logkeeper": 5,
        "rawhollow-writer": 2,
        "rawhollow-producer": 1,
        "rawhollow-explorer": 1,
    }


def test_raw_hollow_models_are_registered():
    assert {
        "org.netflix.raw-hollow",
        "org.netflix.raw-hollow.logkeeper",
        "org.netflix.raw-hollow.writer",
        "org.netflix.raw-hollow.producer",
        "org.netflix.raw-hollow.explorer",
    } <= planner.models.keys()


def test_raw_hollow_uncertain_plan_composes_all_roles():
    desires = CapacityDesires(
        query_pattern=QueryPattern(
            estimated_write_per_second=Interval(
                low=100,
                mid=1_000,
                high=10_000,
                confidence=0.98,
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=1,
                mid=100,
                high=1_000,
                confidence=0.98,
            ),
        ),
    )

    plan = planner.plan(
        model_name="org.netflix.raw-hollow",
        region="us-east-1",
        desires=desires,
        simulations=32,
    ).least_regret[0]

    assert {cluster.cluster_type for cluster in plan.candidate_clusters.regional} == {
        "rawhollow-logkeeper",
        "rawhollow-writer",
        "rawhollow-producer",
        "rawhollow-explorer",
    }

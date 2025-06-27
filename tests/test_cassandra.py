import pytest

from service_capacity_modeling.models.org.netflix.cassandra import (
    NflxCassandraCapacityModel,
)


@pytest.mark.parametrize(
    "tier, extra_model_arguments, expected_result",
    [
        # Non-critical tier, no required_cluster_size
        (2, {}, None),
        # Non-critical tier, required_cluster_size provided
        (2, {"required_cluster_size": 5}, 5),
        # Critical tier, required_cluster_size >= CRITICAL_TIER_MIN_CLUSTER_SIZE
        (0, {"required_cluster_size": 3}, 3),
        (0, {"required_cluster_size": 2}, 2),
        # Critical tier, no required_cluster_size
        (0, {}, None),
    ],
)
def test_get_required_cluster_size_valid(tier, extra_model_arguments, expected_result):
    result = NflxCassandraCapacityModel.get_required_cluster_size(
        tier, extra_model_arguments
    )
    assert result == expected_result


@pytest.mark.parametrize(
    "tier, extra_model_arguments, expected_exception",
    [
        # Critical tier(s), required_cluster_size < CRITICAL_TIER_MIN_CLUSTER_SIZE
        (
            1,
            {"required_cluster_size": 1},
            ValueError,
        ),
        (
            0,
            {"required_cluster_size": 1},
            ValueError,
        ),
    ],
)
def test_get_required_cluster_size_exceptions(
    tier, extra_model_arguments, expected_exception
):
    with pytest.raises(expected_exception):
        NflxCassandraCapacityModel.get_required_cluster_size(
            tier, extra_model_arguments
        )

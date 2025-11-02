import pytest

from service_capacity_modeling import capacity_planner
from service_capacity_modeling.capacity_planner import CapacityPlanner
from service_capacity_modeling.models.org import netflix


@pytest.fixture(scope="session", autouse=True)
def configure_test_planner():
    """
    Configure the global planner instance with optimized settings for testing.

    This fixture automatically runs before all tests and replaces the default
    planner with one that uses fewer simulations (32 instead of 128).

    This provides a significant speedup (4x) for tests involving uncertain
    capacity desires (Intervals) while maintaining sufficient statistical
    confidence for test validation.
    """
    # Create a new planner with reduced simulations for faster tests
    # Use 16 simulations for much faster tests (was 128 in production, 32 initially)
    # This provides ~8x speedup while maintaining sufficient coverage
    test_planner = CapacityPlanner(
        default_num_simulations=16,  # Reduced from 128 default
        default_num_results=2,
    )

    # Register the Netflix models (same as production planner)
    test_planner.register_group(netflix.models)

    # Replace the global planner instance
    capacity_planner.planner = test_planner

    yield test_planner

    # No cleanup needed - planner is replaced for entire test session

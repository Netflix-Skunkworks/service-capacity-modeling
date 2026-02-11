"""
Registry for property test configurations.

This module stores the centralized registry of model configurations
to avoid circular imports between conftest.py and property_test_utils.py.
"""

from typing import Any
from typing import Dict

# Registry for property test configurations
# This will be populated by conftest.py during test discovery
_PROPERTY_TEST_CONFIGS: Dict[str, Dict[str, Any]] = {}


def register_property_test_config(model_name: str, config: Dict[str, Any]):
    """
    Register a property test configuration for a model.

    Args:
        model_name: Full model name (e.g., "org.netflix.postgres")
        config: Configuration dictionary with keys:
            - extra_model_arguments: Dict of required model arguments
            - exclude_from_universal_tests: Set to True to skip universal tests
            - exclude_from_high_qps_tests: Set to True to skip high QPS tests
            - supports_tier_0: Set to False if model doesn't support tier 0
            - qps_range: Tuple of (min_qps, max_qps) for property test generation
            - data_range_gib: Tuple of (min_gib, max_gib) for property test generation
            - skip_tests: List of test function names to skip for this model
    """
    _PROPERTY_TEST_CONFIGS[model_name] = config


def get_property_test_config(model_name: str) -> Dict[str, Any]:
    """Get the property test configuration for a model."""
    return _PROPERTY_TEST_CONFIGS.get(model_name, {})


def get_all_registered_configs() -> Dict[str, Dict[str, Any]]:
    """Get all registered property test configurations."""
    return _PROPERTY_TEST_CONFIGS.copy()

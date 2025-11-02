"""
Pytest configuration for Netflix model tests.

This module automatically discovers and registers property test configurations
from individual model test files. Each test file can optionally define a
PROPERTY_TEST_CONFIG dictionary to customize property testing behavior.
"""

import importlib
import pkgutil

from tests.netflix.property_test_registry import get_all_registered_configs
from tests.netflix.property_test_registry import register_property_test_config


# Automatically discover and load property test configs from all test modules
def _discover_property_configs():
    """Scan all test_*.py files in this package for PROPERTY_TEST_CONFIG."""
    import tests.netflix as test_package

    package_path = test_package.__path__
    package_name = test_package.__name__

    for _, module_name, _ in pkgutil.iter_modules(package_path):
        if not module_name.startswith("test_"):
            continue

        try:
            module = importlib.import_module(f"{package_name}.{module_name}")
            if hasattr(module, "PROPERTY_TEST_CONFIG"):
                config = getattr(module, "PROPERTY_TEST_CONFIG")
                if isinstance(config, dict):
                    # Config can be a single model or multiple models
                    for model_name, model_config in config.items():
                        register_property_test_config(model_name, model_config)
        except Exception:  # pylint: disable=broad-exception-caught
            # Silently skip modules that fail to import or don't have valid config
            pass


# Run discovery on import
_discover_property_configs()


def pytest_generate_tests(metafunc):
    """
    Dynamically generate test parametrization based on registered configs.

    This hook is called during test collection to allow dynamic parametrization.
    It replaces static UNIVERSAL_TEST_MODELS and HIGH_QPS_TEST_MODELS lists
    with dynamically computed lists based on registered configs.
    """
    if "model_name" in metafunc.fixturenames:
        # Import here to ensure all test modules have been scanned
        from tests.netflix.property_test_utils import get_all_model_names

        all_models = get_all_model_names()

        # Exclude only models with exclude_from_universal_tests
        # (all tests now use model-specific ranges)
        excluded = set()
        for model_name, config in get_all_registered_configs().items():
            if config.get("exclude_from_universal_tests", False):
                excluded.add(model_name)
        models = [m for m in all_models if m not in excluded]

        metafunc.parametrize("model_name", models)

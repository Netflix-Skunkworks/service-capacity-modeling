"""
Tests for Netflix identifier model.
"""

# Property test configuration for Identifier model.
# See tests/netflix/PROPERTY_TESTING.md for configuration options and examples.
PROPERTY_TEST_CONFIG = {
    "org.netflix.identifier": {
        "extra_model_arguments": {},
        "read_qps_range": (10, 1000),
        "write_qps_range": (1, 100),
        "data_range_gib": (1, 50),
    },
}

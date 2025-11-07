"""
Tests for Netflix control model.
"""

# Property test configuration for Control model.
# See tests/netflix/PROPERTY_TESTING.md for configuration options and examples.
PROPERTY_TEST_CONFIG = {
    "org.netflix.control": {
        "extra_model_arguments": {},
        # Control caches reads in memory, only writes go to Aurora
        # Read QPS can be high, but write QPS is typically low
        "read_qps_range": (10, 1000),
        "write_qps_range": (1, 100),
        "data_range_gib": (1, 50),
    },
}

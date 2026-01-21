"""
Tests for Netflix entity model.
"""

# Property test configuration for Entity model.
# See tests/netflix/PROPERTY_TESTING.md for configuration options and examples.
PROPERTY_TEST_CONFIG = {
    "org.netflix.entity": {
        # Entity doesn't support tier 0
        # (tier 0 support is inferred from tier_range[0] > 0)
        "tier_range": (1, 2),
        # Cap QPS to prevent Aurora from disappearing at high load
        "qps_range": (100, 10000),
    },
}

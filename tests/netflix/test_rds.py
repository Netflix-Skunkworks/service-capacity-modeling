"""
Tests for Netflix RDS model.
"""

# Property test configuration for RDS model.
# RDS is used for composition with Entity/Control
PROPERTY_TEST_CONFIG = {
    "org.netflix.rds": {
        # RDS doesn't support tier 0 (uses EC2, not managed service tiers)
        "tier_range": (1, 2),
        # RDS proxy handles limited QPS due to connection pooling
        "qps_range": (100, 5000),
        # Smaller data footprint (connection metadata only)
        "data_range_gib": (1, 50),
    },
}

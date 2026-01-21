"""
Tests for Netflix Postgres model.
"""

# Property test configuration for Postgres model.
# Postgres is excluded from universal tests as it's used as a composition target.
PROPERTY_TEST_CONFIG = {
    "org.netflix.postgres": {
        "exclude_from_universal_tests": True,
    },
}

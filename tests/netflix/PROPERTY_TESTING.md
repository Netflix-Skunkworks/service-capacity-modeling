# Property Testing Configuration

This directory uses automated property-based testing to verify universal properties across all capacity models.

## How It Works

Universal property tests are defined in `test_all_models_properties.py` and automatically run against all models. Each model can customize this behavior by defining a `PROPERTY_TEST_CONFIG` dictionary in its test file (e.g., `test_postgres.py`, `test_entity.py`).

The configuration is automatically discovered and registered by `conftest.py` on import.

## Configuration Options

Add a `PROPERTY_TEST_CONFIG` dictionary to your model's test file to customize property testing:

```python
PROPERTY_TEST_CONFIG = {
    "org.netflix.your-model": {
        # Required planning arguments (if any)
        "extra_model_arguments": {
            "num_regions": 1,
            "your_param": "value",
        },

        # Valid QPS range for property tests (default: 1000-50000)
        "qps_range": (100, 1000),

        # Valid data size range in GiB (default: 100-1000)
        "data_range_gib": (10, 100),

        # Tier comparison override (default: (0, 2) for tier 0 vs tier 2)
        # NOTE: If tier_range[0] > 0, tier 0 support is automatically disabled
        "tier_range": (1, 2),

        # Skip all universal property tests (use sparingly)
        "exclude_from_universal_tests": True,
    },
}
```

## Examples

### Database Model with Restricted Scale
See `tests/netflix/test_postgres.py`:
```python
PROPERTY_TEST_CONFIG = {
    "org.netflix.postgres": {
        "extra_model_arguments": {"num_regions": 1},
        "qps_range": (100, 1000),
        "data_range_gib": (10, 100),
        "tier_range": (1, 2),
    },
}
```

### Model with Required Arguments
See `tests/netflix/test_counter.py`:
```python
PROPERTY_TEST_CONFIG = {
    "org.netflix.counter": {
        "extra_model_arguments": {
            "counter.mode": "exact",
            "counter.cardinality": "high",
        },
    },
}
```

### Excluding from Universal Tests
See `tests/netflix/test_kafka.py`:
```python
PROPERTY_TEST_CONFIG = {
    "org.netflix.kafka": {
        "exclude_from_universal_tests": True,
    },
}
```

## What Gets Tested

Universal property tests verify:
- **Determinism**: Same input → same output
- **Feasibility**: Valid input → at least one plan
- **QPS Monotonicity**: Higher QPS → more CPU (or equal)
- **Tier Capacity**: Tier 0 ≥ Tier 2 in at least one dimension (CPU/RAM/storage/cost)
- **Cost Positivity**: All plans have positive annual cost
- **Instance Count Positivity**: All clusters have at least one instance

## When to Write Model-Specific Tests

Model-specific tests (in the same file as `PROPERTY_TEST_CONFIG`) should test:
- Specific instance types selected
- Specific cost ranges
- Edge cases (capacity limits, unsupported tiers)
- Regression behaviors
- Implementation details not covered by universal properties

Universal property tests verify general correctness across all models, while model-specific tests verify specific implementation details.

# Service Capacity Modeling - Claude Instructions

## Project Overview

This is a capacity planning library for Netflix infrastructure services. It models resource requirements (CPU, memory, disk, network) for various data stores and computes optimal cluster configurations.

## Key Directories

```
service_capacity_modeling/
├── models/org/netflix/     # Netflix-specific capacity models
│   ├── partition_capacity.py  # Partition-based algorithms + fault tolerance
│   ├── read_only_kv.py        # Read-only KV store model
│   ├── cassandra.py           # Cassandra model
│   └── ...
├── hardware/               # Instance type definitions
└── interface.py           # Core data structures (CapacityDesires, etc.)

tests/netflix/              # Netflix model tests
```

## Running Tests

```bash
# Run specific test file
tox -e py312 -- tests/netflix/test_partition_capacity.py -v

# Run with pattern matching
tox -e py312 -- tests/netflix/ -k "fault_tolerance" -v

# Run all tests (slow, ~5 min)
tox -e py312
```

## Code Style

- **Pre-commit hooks run automatically** on commit via tox
- Use `ruff` for formatting (runs in pre-commit)
- Line length: 88 characters
- Type hints required for public APIs
- Docstrings for public functions

### Linting Pragmas

When needed, use these patterns:
```python
def complex_function(...):  # noqa: C901
    """Allow complex function."""

def many_args(a, b, c, d, e, f, g):  # pylint: disable=R0917
    """Allow many positional arguments."""
```

## Testing Patterns

### Hypothesis Property Testing

Use Hypothesis for testing mathematical properties and finding edge cases:

```python
from hypothesis import given, settings
from hypothesis import strategies as st

@given(
    n_partitions=st.integers(10, 500),
    rf=st.integers(2, 5),
)
@settings(max_examples=100, deadline=None)
def test_property(self, n_partitions, rf):
    """Property: describe the invariant being tested."""
    # Test a PROPERTY of the function, not its correctness
    result = some_function(n_partitions, rf)
    assert result >= 0  # Example invariant
```

### Monte Carlo Simulation

Use Monte Carlo for validating closed-form math against simulation:

```python
def test_closed_form_matches_simulation(self):
    """Validate math for specific representative cases."""
    closed = closed_form_function(12, 3, 2, 100)
    simulated = simulate_function(12, 3, 2, 100, n_trials=10000, seed=42)

    # Allow 2-3% tolerance for simulation variance
    assert abs(closed - simulated) < 0.03
```

**Important**: Do NOT combine Hypothesis with Monte Carlo (double-randomization is slow and redundant). Use each tool for its strength:
- **Monte Carlo**: Validate math for specific representative cases
- **Hypothesis**: Test properties across input space without simulation

## Fault Tolerance Module

Located in `partition_capacity.py`, this module computes system-wide availability under AZ failure.

### Key Insight: Non-linear Averaging

The correct formula for system availability is:
```
P(system available) = (1/n_zones) × Σ_z (1 - p_z)^P
```

NOT `(1 - avg_p)^P`. Must compute per-zone availability FIRST, then average.

### Tier Configuration

| Tier | Target Availability | Min RF | Use Case |
|------|---------------------|--------|----------|
| 0    | 99.9%               | 3      | Critical services |
| 1    | 99%                 | 3      | Important services |
| 2    | 95%                 | 2      | Standard services |
| 3    | 80%                 | 2      | Test/dev |

### Zone-Aware Comparison

The `FaultTolerantResult` includes `zone_aware_cost` and `zone_aware_savings` to show potential cost savings if the system used zone-aware placement (where RF=2 guarantees cross-AZ spread).

## Worktrees

Active worktrees for this project:
- `~/worktrees/mho-fault-tolerance-v2` - Fault tolerance v2 implementation

## Common Issues

### Pre-commit Hook Failures

If pre-commit reformats files, just `git add` and commit again:
```bash
git add <file>
git commit -m "message"  # Will pass second time
```

### Hypothesis Counterexamples

When Hypothesis finds a counterexample, it's usually revealing a bug in your assumptions, not your code. Example: "more nodes → better availability" is FALSE because more nodes per zone = more ways to place all replicas in the same zone.

# Service Capacity Modeling

![Build Status](https://github.com/Netflix-Skunkworks/service-capacity-modeling/actions/workflows/python-build.yml/badge.svg)

A generic toolkit for modeling capacity requirements in the cloud. Pricing
information included in this repository are public prices.

**NOTE**: Netflix confidential information should never enter this repo. Please
remember this repository is public when making changes to it.

## Trying it out

Run the tests:
```bash
# Test the capacity planner on included netflix models
$ tox -e py38

# Run a single test with a debugger attached if the test fails
$ .tox/py38/bin/pytest -n0 -k test_java_heap_heavy --pdb --pdbcls=IPython.terminal.debugger:Pdb

# Verify all type contracts
$ tox -e mypy
```

Run IPython for interactively using the library:
```
tox -e dev -- ipython
```

## Example of Provisioning a Database
Fire up ipython and let's capacity plan a Tier 1 (important to the product aka
"prod") Cassandra database.

```python
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import FixedInterval, Interval
from service_capacity_modeling.interface import QueryPattern, DataShape

db_desires = CapacityDesires(
    # This service is important to the business, not critical (tier 0)
    service_tier=1,
    query_pattern=QueryPattern(
        # Not sure exactly how much QPS we will do, but we think around
        # 10,000 reads and 10,000 writes per second.
        estimated_read_per_second=Interval(
            low=1000, mid=10000, high=100000, confidence=0.9
        ),
        estimated_write_per_second=Interval(
            low=1000, mid=10000, high=100000, confidence=0.9
        ),
    ),
    # Not sure how much data, but we think it'll be below 1 TiB
    data_shape=DataShape(
        estimated_state_size_gib=Interval(low=100, mid=100, high=1000, confidence=0.9),
    ),
)
```

Now we can load up some models and do some capacity planning

```python
from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.models.org import netflix
import pprint

# Load up the Netflix capacity models
planner.register_group(netflix.models)

cap_plan = planner.plan(
    model_name="org.netflix.cassandra",
    region="us-east-1",
    desires=db_desires,
    # Simulate the possible requirements 512 times
    simulations=512,
    # Request 3 diverse hardware families to be returned
    num_results=3,
)

# The range of requirements in hardware resources (CPU, RAM, Disk, etc ...)
requirements = cap_plan.requirements

# The ordered list of least regretful choices for the requirement
least_regret = cap_plan.least_regret

# Show the range of requirements for a single zone
pprint.pprint(requirements.zonal[0].model_dump())

# Show our least regretful choices of hardware in least regret order
# So for example if we can buy the first set of computers we would prefer
# to do that but we might not have availability in that family in which
# case we'd buy the second one.
for choice in range(3):
    num_clusters = len(least_regret[choice].candidate_clusters.zonal)
    print(f"Our #{choice + 1} choice is {num_clusters} zones of:")
    pprint.pprint(least_regret[choice].candidate_clusters.zonal[0].model_dump())

```

Note that we _can_ customize more information given what we know about the
use case, but each model (e.g. Cassandra) supplies reasonable defaults.

For example we can specify a lot more information

```python
from service_capacity_modeling.interface import CapacityDesires, QueryPattern, Interval, FixedInterval, DataShape

db_desires = CapacityDesires(
    # This service is important to the business, not critical (tier 0)
    service_tier=1,
    query_pattern=QueryPattern(
        # Not sure exactly how much QPS we will do, but we think around
        # 50,000 reads and 45,000 writes per second with a rather narrow
        # bound
        estimated_read_per_second=Interval(
            low=40_000, mid=50_000, high=60_000, confidence=0.9
        ),
        estimated_write_per_second=Interval(
            low=42_000, mid=45_000, high=50_000, confidence=0.9
        ),
        # This use case might do some partition scan queries that are
        # somewhat expensive, so we hint a rather expensive ON-CPU time
        # that a read will consume on the entire cluster.
        estimated_mean_read_latency_ms=Interval(
            low=0.1, mid=4, high=20, confidence=0.9
        ),
        # Writes at LOCAL_ONE are pretty cheap
        estimated_mean_write_latency_ms=Interval(
            low=0.1, mid=0.4, high=0.8, confidence=0.9
        ),
        # We want single digit latency, note that this is not a p99 of 10ms
        # but defines the interval where 98% of latency falls to be between
        # 0.4 and 10 milliseconds. Think of:
        #   low = "the minimum reasonable latency"
        #   high = "the maximum reasonable latency"
        #   mid = "value between low and high such that I want my distribution
        #          to skew left or right"
        read_latency_slo_ms=FixedInterval(
            low=0.4, mid=4, high=10, confidence=0.98
        ),
        write_latency_slo_ms=FixedInterval(
            low=0.4, mid=4, high=10, confidence=0.98
        )
    ),
    # Not sure how much data, but we think it'll be below 1 TiB
    data_shape=DataShape(
        estimated_state_size_gib=Interval(low=100, mid=500, high=1000, confidence=0.9),
    ),
)
```

## Example of provisioning a caching cluster

In this example we tweak the QPS up, on CPU time of operations down
and SLO down. This more closely approximates a caching workload

```python
from service_capacity_modeling.interface import CapacityDesires, QueryPattern, Interval, FixedInterval, DataShape
from service_capacity_modeling.capacity_planner import planner

cache_desires = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        # Not sure exactly how much QPS we will do, but we think around
        # 10,000 reads and 10,000 writes per second.
        estimated_read_per_second=Interval(
            low=10_000, mid=100_000, high=1_000_000, confidence=0.9
        ),
        estimated_write_per_second=Interval(
            low=1_000, mid=20_000, high=100_000, confidence=0.9
        ),
        # Memcache is consistently fast at queries
        estimated_mean_read_latency_ms=Interval(
            low=0.05, mid=0.2, high=0.4, confidence=0.9
        ),
        estimated_mean_write_latency_ms=Interval(
            low=0.05, mid=0.2, high=0.4, confidence=0.9
        ),
        # Caches usually have tighter SLOs
        read_latency_slo_ms=FixedInterval(
            low=0.4, mid=0.5, high=5, confidence=0.98
        ),
        write_latency_slo_ms=FixedInterval(
            low=0.4, mid=0.5, high=5, confidence=0.98
        )
    ),
    # Not sure how much data, but we think it'll be below 1000
    data_shape=DataShape(
        estimated_state_size_gib=Interval(low=100, mid=200, high=500, confidence=0.9),
    ),
)

cache_cap_plan = planner.plan(
    model_name="org.netflix.cassandra",
    region="us-east-1",
    desires=cache_desires,
    allow_gp2=True,
)

requirement = cache_cap_plan.requirement
least_regret = cache_cap_plan.least_regret
```

## Notebooks

We have a demo notebook in `notebooks` you can use to experiment. Start it with

```
tox -e notebook -- jupyter notebook notebooks/demo.ipynb
```

## Development

To contribute to this project:

1. Make your change in a branch. Consider making a new model if you are making
   significant changes and registering it as a different name.
2. Write a unit test using `pytest` in the `tests` folder.
3. Ensure your tests pass via `tox` or debug them with:
```
tox -e py38 -- -k test_<your_functionality> --pdb --pdbcls=IPython.terminal.debugger:Pdb
```

### PyCharm IDE Setup
Use one of the test environments for IDE development, e.g. `tox -e py310` and then
`Add New Interpreter -> Add Local -> Select Existing -> Navigate to (workdir)/.tox/py310`.

### Running CLIs
Use the `dev` virtual environment via `tox -e dev`. Then execute CLIs via that env.

## Release
Any successful `main` build will trigger a release to PyPI, defaulting to a patch bump based on the setupmeta
[distance algorithm](https://github.com/codrsquad/setupmeta/blob/main/docs/versioning.rst#distance). If
you are significantly adding to the API please follow the below instructions to bump the base version. Since we
are still in `0.` we do not do major version bumps.

### Bumping a minor or major
From latest `main`, bump at least the `minor` to get a new base version:
```shell
git tag v0.4.0
git push origin HEAD --tags
```
Now setupmeta will bump the patch from this version, e.g. `0.4.1`.

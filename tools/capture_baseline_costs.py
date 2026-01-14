#!/usr/bin/env python3
"""
Capture current cost outputs for regression testing.

This script runs capacity planning for various scenarios and captures
the cost breakdowns to use as baselines for regression tests.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import (
    CapacityDesires,
    certain_float,
    certain_int,
    DataShape,
    Interval,
    QueryPattern,
)


def capture_costs(model_name, region, desires, extra_args=None, scenario_name=""):
    """Capture all cost breakdown for a planning scenario."""
    try:
        cap_plans = planner.plan_certain(
            model_name=model_name,
            region=region,
            desires=desires,
            num_results=1,
            extra_model_arguments=extra_args or {},
        )

        if not cap_plans:
            return {"error": "No capacity plans generated"}

        cap_plan = cap_plans[0]
        clusters = cap_plan.candidate_clusters

        result = {
            "scenario": scenario_name,
            "model": model_name,
            "region": region,
            "service_tier": desires.service_tier,
            "annual_costs": {k: float(v) for k, v in clusters.annual_costs.items()},
            "total_annual_cost": float(clusters.total_annual_cost),
            "cluster_count": len(clusters.zonal) + len(clusters.regional),
            "service_count": len(clusters.services),
        }

        # Add instance info
        if clusters.zonal:
            result["instance_name"] = clusters.zonal[0].instance.name
            result["instance_count"] = clusters.zonal[0].count
            result["deployment"] = "zonal"
        elif clusters.regional:
            result["instance_name"] = clusters.regional[0].instance.name
            result["instance_count"] = clusters.regional[0].count
            result["deployment"] = "regional"

        return result
    except Exception as e:
        return {"error": str(e), "scenario": scenario_name}


# Define test scenarios for each service
scenarios = []

# RDS scenarios
rds_small = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(200),
        estimated_write_per_second=certain_int(100),
        estimated_mean_read_latency_ms=certain_float(10),
        estimated_mean_write_latency_ms=certain_float(10),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(50),
    ),
)

rds_tier3 = CapacityDesires(
    service_tier=3,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(200),
        estimated_write_per_second=certain_int(100),
        estimated_mean_read_latency_ms=certain_float(20),
        estimated_mean_write_latency_ms=certain_float(20),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(200),
    ),
)

scenarios.extend([
    ("org.netflix.rds", "us-east-1", rds_small, None, "rds_small_tier1"),
    ("org.netflix.rds", "us-east-1", rds_tier3, None, "rds_tier3"),
])

# Aurora scenarios
aurora_small = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(100),
        estimated_write_per_second=certain_int(100),
        estimated_mean_read_latency_ms=certain_float(10),
        estimated_mean_write_latency_ms=certain_float(10),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(50),
    ),
)

aurora_tier3 = CapacityDesires(
    service_tier=3,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(200),
        estimated_write_per_second=certain_int(100),
        estimated_mean_read_latency_ms=certain_float(10),
        estimated_mean_write_latency_ms=certain_float(10),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(200),
    ),
)

scenarios.extend([
    ("org.netflix.aurora", "us-east-1", aurora_small, None, "aurora_small_tier1"),
    ("org.netflix.aurora", "us-east-1", aurora_tier3, None, "aurora_tier3"),
])

# Cassandra scenarios
cassandra_small_high_qps = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(100_000),
        estimated_write_per_second=certain_int(100_000),
        estimated_mean_read_latency_ms=certain_float(0.5),
        estimated_mean_write_latency_ms=certain_float(0.4),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(10),
    ),
)

cassandra_high_writes = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(10_000),
        estimated_write_per_second=certain_int(500_000),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(300),
    ),
)

scenarios.extend([
    ("org.netflix.cassandra", "us-east-1", cassandra_small_high_qps,
     {"require_local_disks": True}, "cassandra_small_high_qps_local"),
    ("org.netflix.cassandra", "us-east-1", cassandra_high_writes,
     {"require_local_disks": False, "copies_per_region": 2}, "cassandra_high_writes_ebs"),
])

# Kafka scenarios
kafka_medium = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(50_000),
        estimated_write_per_second=certain_int(50_000),
        estimated_mean_write_size_bytes=certain_int(1024),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(1000),
        estimated_state_item_count=Interval(
            low=1_000_000, mid=10_000_000, high=100_000_000, confidence=0.98
        ),
    ),
)

scenarios.extend([
    ("org.netflix.kafka", "us-east-1", kafka_medium,
     {"require_local_disks": True}, "kafka_medium_local"),
])

# EVCache scenarios
evcache_small = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(100_000),
        estimated_write_per_second=certain_int(10_000),
        estimated_mean_read_latency_ms=certain_float(1.0),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(10),
        estimated_state_item_count=Interval(
            low=1_000_000, mid=10_000_000, high=20_000_000, confidence=0.98
        ),
    ),
)

evcache_large = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(500_000),
        estimated_write_per_second=certain_int(50_000),
        estimated_mean_read_latency_ms=certain_float(1.0),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(500),
        estimated_state_item_count=Interval(
            low=10_000_000, mid=100_000_000, high=200_000_000, confidence=0.98
        ),
    ),
)

scenarios.extend([
    ("org.netflix.evcache", "us-east-1", evcache_small,
     {"cross_region_replication": "none"}, "evcache_small_no_replication"),
    ("org.netflix.evcache", "us-east-1", evcache_large,
     {"cross_region_replication": "sets", "copies_per_region": 2}, "evcache_large_with_replication"),
])

# Capture all scenarios
results = []
for model, region, desires, extra_args, scenario_name in scenarios:
    print(f"Capturing: {scenario_name}...")
    result = capture_costs(model, region, desires, extra_args, scenario_name)
    results.append(result)

    if "error" in result:
        print(f"  ERROR: {result['error']}")
    else:
        print(f"  Total cost: ${result['total_annual_cost']:,.2f}")
        print(f"  Cost breakdown: {list(result['annual_costs'].keys())}")

# Save results
output_file = Path(__file__).parent / "baseline_costs.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {output_file}")
print(f"Total scenarios captured: {len([r for r in results if 'error' not in r])}/{len(results)}")

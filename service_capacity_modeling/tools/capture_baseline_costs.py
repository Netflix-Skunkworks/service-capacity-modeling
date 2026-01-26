#!/usr/bin/env python3
"""
Capture current cost outputs for regression testing.

This script runs capacity planning for various scenarios and captures
the cost breakdowns to use as baselines for regression tests.

Usage:
    python -m service_capacity_modeling.tools.capture_baseline_costs
"""

import json
from pathlib import Path
from typing import Any

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import (
    AccessConsistency,
    CapacityDesires,
    certain_float,
    certain_int,
    ClusterCapacity,
    Consistency,
    DataShape,
    GlobalConsistency,
    Interval,
    QueryPattern,
)


def _format_cluster(cluster: ClusterCapacity, deployment: str) -> dict[str, Any]:
    """Format a single cluster's details."""
    info: dict[str, Any] = {
        "cluster_type": cluster.cluster_type,
        "deployment": deployment,
        "instance": cluster.instance.name,
        "count": cluster.count,
        "annual_cost": float(cluster.annual_cost),
    }

    # Add attached drives if present
    if cluster.attached_drives:
        drives = []
        for drive in cluster.attached_drives:
            size_gib = int(drive.size_gib) if drive.size_gib else 0
            drives.append(f"{drive.name} : {size_gib}GB")
        info["attached_drives"] = sorted(drives)

    return info


def capture_costs(
    model_name: str,
    region: str,
    desires: CapacityDesires,
    extra_args: dict[str, Any] | None = None,
    scenario_name: str = "",
) -> dict[str, Any]:
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
            return {"error": "No capacity plans generated", "scenario": scenario_name}

        cap_plan = cap_plans[0]
        candidate = cap_plan.candidate_clusters

        # Build cluster details for each cluster
        cluster_details = []
        for zonal_cluster in candidate.zonal:
            cluster_details.append(_format_cluster(zonal_cluster, "zonal"))
        for regional_cluster in candidate.regional:
            cluster_details.append(_format_cluster(regional_cluster, "regional"))

        result = {
            "scenario": scenario_name,
            "model": model_name,
            "region": region,
            "service_tier": desires.service_tier,
            "total_annual_cost": float(candidate.total_annual_cost),
            "clusters": cluster_details,
            "annual_costs": dict(
                sorted((k, float(v)) for k, v in candidate.annual_costs.items())
            ),
        }

        return result
    except (ValueError, KeyError, AttributeError) as e:
        return {"error": str(e), "scenario": scenario_name}


# Define test scenarios for each service
# Each scenario: (model_name, region, desires, extra_args, scenario_name)
scenarios: list[tuple[str, str, CapacityDesires, dict[str, Any] | None, str]] = []

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

scenarios.extend(
    [
        ("org.netflix.rds", "us-east-1", rds_small, None, "rds_small_tier1"),
        ("org.netflix.rds", "us-east-1", rds_tier3, None, "rds_tier3"),
    ]
)

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

scenarios.extend(
    [
        ("org.netflix.aurora", "us-east-1", aurora_small, None, "aurora_small_tier1"),
        ("org.netflix.aurora", "us-east-1", aurora_tier3, None, "aurora_tier3"),
    ]
)

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

scenarios.extend(
    [
        (
            "org.netflix.cassandra",
            "us-east-1",
            cassandra_small_high_qps,
            {"require_local_disks": True},
            "cassandra_small_high_qps_local",
        ),
        (
            "org.netflix.cassandra",
            "us-east-1",
            cassandra_high_writes,
            {"require_local_disks": False, "copies_per_region": 2},
            "cassandra_high_writes_ebs",
        ),
    ]
)

# Kafka scenarios - Kafka uses throughput-based sizing via write_size
# 100 MiB/s throughput with 2 consumers, 1 producer
throughput = 100 * 1024 * 1024  # 100 MiB/s
kafka_throughput = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(low=1, mid=2, high=2, confidence=0.98),
        estimated_write_per_second=Interval(low=1, mid=1, high=1, confidence=0.98),
        estimated_mean_write_size_bytes=Interval(
            low=throughput, mid=throughput, high=throughput * 2, confidence=0.98
        ),
    ),
)

scenarios.extend(
    [
        (
            "org.netflix.kafka",
            "us-east-1",
            kafka_throughput,
            {"require_local_disks": False},
            "kafka_100mib_throughput",
        ),
    ]
)

# EVCache scenarios
# Tiny EVCache - small cluster to show spread cost (< 10 instances = spread penalty)
evcache_tiny = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(1_000),
        estimated_write_per_second=certain_int(100),
        estimated_mean_read_latency_ms=certain_float(1.0),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(1),
        estimated_state_item_count=Interval(
            low=10_000, mid=100_000, high=200_000, confidence=0.98
        ),
    ),
)

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

scenarios.extend(
    [
        (
            "org.netflix.evcache",
            "us-east-1",
            evcache_tiny,
            {"cross_region_replication": "none"},
            "evcache_tiny_with_spread",
        ),
        (
            "org.netflix.evcache",
            "us-east-1",
            evcache_small,
            {"cross_region_replication": "none"},
            "evcache_small_no_replication",
        ),
        (
            "org.netflix.evcache",
            "us-east-1",
            evcache_large,
            {"cross_region_replication": "sets", "copies_per_region": 2},
            "evcache_large_with_replication",
        ),
    ]
)

# Key-Value scenarios (composite: Cassandra + EVCache)
# Uses evcache_large desires with eventual consistency to enable caching layer
kv_with_cache = evcache_large.model_copy(deep=True)
kv_with_cache.query_pattern.access_consistency = GlobalConsistency(
    same_region=Consistency(target_consistency=AccessConsistency.eventual),
    cross_region=Consistency(target_consistency=AccessConsistency.best_effort),
)

scenarios.extend(
    [
        (
            "org.netflix.key-value",
            "us-east-1",
            kv_with_cache,
            None,
            "kv_with_cache",
        ),
    ]
)

# Export as dict for tests to import (single source of truth)
SCENARIOS: dict[str, dict[str, Any]] = {
    name: {
        "model": model,
        "region": region,
        "desires": desires,
        "extra_args": extra_args,
    }
    for model, region, desires, extra_args, name in scenarios
}


if __name__ == "__main__":
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
    output_file = Path(__file__).parent / "data" / "baseline_costs.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")  # Ensure trailing newline for pre-commit

    print(f"\nResults saved to: {output_file}")
    success_count = len([r for r in results if "error" not in r])
    print(f"Total scenarios captured: {success_count}/{len(results)}")

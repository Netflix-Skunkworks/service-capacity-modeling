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
    Buffer,
    BufferComponent,
    BufferIntent,
    Buffers,
    CapacityDesires,
    certain_float,
    Drive,
    certain_int,
    ClusterCapacity,
    Consistency,
    CurrentClusters,
    CurrentZoneClusterCapacity,
    DataShape,
    GlobalConsistency,
    Interval,
    QueryPattern,
)

BASELINE_UNCERTAIN_SIMULATIONS = 16
BASELINE_UNCERTAIN_NUM_RESULTS = 3


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

    # Add cluster_params if present (e.g., replica_count, partitions_per_node)
    if cluster.cluster_params:
        info["cluster_params"] = dict(sorted(cluster.cluster_params.items()))

    return info


def _capture_candidate(candidate: Any) -> dict[str, Any]:
    """Serialize a candidate cluster set into a stable regression snapshot."""
    cluster_details = []
    for zonal_cluster in candidate.zonal:
        cluster_details.append(_format_cluster(zonal_cluster, "zonal"))
    for regional_cluster in candidate.regional:
        cluster_details.append(_format_cluster(regional_cluster, "regional"))

    return {
        "total_annual_cost": float(candidate.total_annual_cost),
        "clusters": cluster_details,
        "annual_costs": dict(
            sorted((k, float(v)) for k, v in candidate.annual_costs.items())
        ),
    }


def _capture_plan_sequence(plans: Any) -> list[dict[str, Any]]:
    return [_capture_candidate(plan.candidate_clusters) for plan in plans]


def _capture_counted_excuse(excuse: Any) -> dict[str, Any]:
    return {
        "instance": excuse.instance,
        "drive": excuse.drive,
        "reason": excuse.reason,
        "count": excuse.count,
        "bottleneck": (
            str(excuse.bottleneck) if excuse.bottleneck is not None else None
        ),
        "tags": sorted(str(tag) for tag in excuse.tags),
    }


def _capture_regret_summary(summary: Any) -> dict[str, Any]:
    return {
        "plan": _capture_candidate(summary.plan.candidate_clusters),
        "equivalent_plan_count": summary.equivalent_plan_count,
        "selected_total_regret": summary.selected_total_regret,
        "min_total_regret": summary.min_total_regret,
        "max_total_regret": summary.max_total_regret,
        "mean_total_regret": summary.mean_total_regret,
        "selected_regret_components": dict(
            sorted(summary.selected_regret_components.items())
        ),
        "mean_regret_components": dict(sorted(summary.mean_regret_components.items())),
        "selected_regret_components_by_model": {
            model: dict(sorted(components.items()))
            for model, components in sorted(
                summary.selected_regret_components_by_model.items()
            )
        },
        "mean_regret_components_by_model": {
            model: dict(sorted(components.items()))
            for model, components in sorted(
                summary.mean_regret_components_by_model.items()
            )
        },
        "representative_models": sorted(summary.representative_desires_by_model.keys()),
    }


def _capture_error(
    scenario_name: str,
    error: Exception,
    model_name: str,
    region: str,
    desires: CapacityDesires,
) -> dict[str, Any]:
    return {
        "error": str(error),
        "scenario": scenario_name,
        "model": model_name,
        "region": region,
        "service_tier": desires.service_tier,
    }


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

        result = {
            "scenario": scenario_name,
            "model": model_name,
            "region": region,
            "service_tier": desires.service_tier,
        }
        result.update(_capture_candidate(cap_plans[0].candidate_clusters))
        return result
    except (ValueError, KeyError, AttributeError) as e:
        return _capture_error(scenario_name, e, model_name, region, desires)


def capture_uncertain(  # pylint: disable=too-many-positional-arguments
    model_name: str,
    region: str,
    desires: CapacityDesires,
    extra_args: dict[str, Any] | None = None,
    scenario_name: str = "",
    simulations: int = BASELINE_UNCERTAIN_SIMULATIONS,
    num_results: int = BASELINE_UNCERTAIN_NUM_RESULTS,
) -> dict[str, Any]:
    """Capture a compact snapshot from the stochastic planner."""
    try:
        cap_plan = planner.plan(
            model_name=model_name,
            region=region,
            desires=desires,
            simulations=simulations,
            num_results=num_results,
            extra_model_arguments=extra_args or {},
        )
        return {
            "scenario": scenario_name,
            "model": model_name,
            "region": region,
            "service_tier": desires.service_tier,
            "simulations": simulations,
            "num_results": num_results,
            "least_regret": _capture_plan_sequence(cap_plan.least_regret),
            "mean": _capture_plan_sequence(cap_plan.mean),
            "percentiles": {
                str(percentile): _capture_plan_sequence(plans)
                for percentile, plans in sorted(cap_plan.percentiles.items())
            },
        }
    except (ValueError, KeyError, AttributeError) as e:
        return _capture_error(scenario_name, e, model_name, region, desires)


def capture_uncertain_explained(  # pylint: disable=too-many-positional-arguments
    model_name: str,
    region: str,
    desires: CapacityDesires,
    extra_args: dict[str, Any] | None = None,
    scenario_name: str = "",
    simulations: int = BASELINE_UNCERTAIN_SIMULATIONS,
    num_results: int = BASELINE_UNCERTAIN_NUM_RESULTS,
) -> dict[str, Any]:
    """Capture the richer uncertain explainability surface."""
    try:
        explained = planner.plan_explained(
            model_name=model_name,
            region=region,
            desires=desires,
            simulations=simulations,
            num_results=num_results,
            extra_model_arguments=extra_args or {},
        )
        return {
            "scenario": scenario_name,
            "model": model_name,
            "region": region,
            "service_tier": desires.service_tier,
            "simulations": simulations,
            "num_results": num_results,
            "least_regret_summaries": [
                _capture_regret_summary(summary)
                for summary in explained.least_regret_summaries
            ],
            "excuse_summary": [
                _capture_counted_excuse(excuse) for excuse in explained.excuse_summary
            ],
        }
    except (ValueError, KeyError, AttributeError) as e:
        return _capture_error(scenario_name, e, model_name, region, desires)


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

# Cassandra vertical scaling baseline — unconstrained instance sizing
# Documents which instances the planner prefers for a typical workload
cassandra_vertical_baseline = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(350_000),
        estimated_write_per_second=certain_int(30_000),
        estimated_mean_read_latency_ms=certain_float(0.8),
        estimated_mean_write_latency_ms=certain_float(0.5),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(500),
        estimated_compression_ratio=Interval(low=1, mid=1, high=1, confidence=1),
    ),
)

scenarios.append(
    (
        "org.netflix.cassandra",
        "us-east-1",
        cassandra_vertical_baseline,
        {"require_local_disks": False},
        "cassandra_vertical_baseline",
    )
)

# Cassandra timeseries — large write-heavy EBS cluster.
# Anonymized from a production cluster with ~200 TiB state, 64 nodes/zone.
# Key: NO memory_utilization_gib — the legacy model must infer working set
# from drive latency and read SLO alone, which overestimates memory needs
# and produces zero results for large archival clusters.
cassandra_timeseries_ebs = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(
            low=120_000, mid=240_000, high=480_000, confidence=0.98
        ),
        estimated_write_per_second=Interval(
            low=340_000, mid=680_000, high=1_000_000, confidence=0.98
        ),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(
            low=180_000, mid=200_000, high=220_000, confidence=0.98
        ),
        estimated_compression_ratio=certain_float(1.0),
    ),
    current_clusters=CurrentClusters(
        zonal=[
            CurrentZoneClusterCapacity(
                cluster_instance_name="r6a.4xlarge",
                cluster_drive=Drive(
                    name="gp3",
                    drive_type="attached-ssd",
                    size_gib=5600,
                ),
                cluster_instance_count=certain_int(64),
                cpu_utilization=Interval(low=5, mid=15, high=45, confidence=1),
                disk_utilization_gib=certain_float(3000),
                network_utilization_mbps=certain_float(50),
            ),
        ]
        * 3
    ),
    # Existing cluster: don't inflate observed disk by the model's storage buffer.
    buffers=Buffers(
        derived={
            "storage": Buffer(
                ratio=1.0,
                intent=BufferIntent.scale_down,
                components=[BufferComponent.storage],
            ),
        },
    ),
)

scenarios.append(
    (
        "org.netflix.cassandra",
        "us-east-1",
        cassandra_timeseries_ebs,
        {
            "require_local_disks": False,
            "experimental_memory_model": True,
        },
        "cassandra_timeseries_ebs",
    )
)

# Cassandra KV dense — read-heavy lookup workload on small EBS instances.
# 50 TiB state served from a hot subset, 32 r6a.2xlarge nodes/zone.
# Exercises the page cache cap on small instances where memory easily becomes
# the bottleneck: the cap requires ~1024 GiB total page cache but disk/CPU
# only need ~9 nodes. Soft memory should reduce from ~24 → ~9 nodes.
cassandra_kv_dense_ebs = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(
            low=100_000, mid=200_000, high=400_000, confidence=0.98
        ),
        estimated_write_per_second=Interval(
            low=10_000, mid=20_000, high=40_000, confidence=0.98
        ),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(
            low=40_000, mid=50_000, high=60_000, confidence=0.98
        ),
        estimated_compression_ratio=certain_float(1.0),
    ),
    current_clusters=CurrentClusters(
        zonal=[
            CurrentZoneClusterCapacity(
                cluster_instance_name="r6a.2xlarge",
                cluster_drive=Drive(
                    name="gp3",
                    drive_type="attached-ssd",
                    size_gib=2400,
                ),
                cluster_instance_count=certain_int(32),
                cpu_utilization=Interval(low=8, mid=20, high=50, confidence=1),
                disk_utilization_gib=certain_float(1500),
                network_utilization_mbps=certain_float(30),
            ),
        ]
        * 3
    ),
    buffers=Buffers(
        derived={
            "storage": Buffer(
                ratio=1.0,
                intent=BufferIntent.scale_down,
                components=[BufferComponent.storage],
            ),
        },
    ),
)

scenarios.append(
    (
        "org.netflix.cassandra",
        "us-east-1",
        cassandra_kv_dense_ebs,
        {
            "require_local_disks": False,
            "require_attached_disks": True,
            "experimental_memory_model": True,
        },
        "cassandra_kv_dense_ebs",
    )
)

# Cassandra KV compact — small lookup workload on 2xlarge EBS instances.
# 20 TiB state, 16 r6a.2xlarge nodes/zone, light read traffic.
# Most extreme memory-bound case: page cache cap requires ~512 GiB total
# but disk/CPU only need ~4 nodes. Soft memory should reduce from ~12 → ~4.
cassandra_kv_compact_ebs = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(
            low=50_000, mid=100_000, high=200_000, confidence=0.98
        ),
        estimated_write_per_second=Interval(
            low=2_500, mid=5_000, high=10_000, confidence=0.98
        ),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(
            low=16_000, mid=20_000, high=24_000, confidence=0.98
        ),
        estimated_compression_ratio=certain_float(1.0),
    ),
    current_clusters=CurrentClusters(
        zonal=[
            CurrentZoneClusterCapacity(
                cluster_instance_name="r6a.2xlarge",
                cluster_drive=Drive(
                    name="gp3",
                    drive_type="attached-ssd",
                    size_gib=1800,
                ),
                cluster_instance_count=certain_int(16),
                cpu_utilization=Interval(low=5, mid=12, high=35, confidence=1),
                disk_utilization_gib=certain_float(1200),
                network_utilization_mbps=certain_float(15),
            ),
        ]
        * 3
    ),
    buffers=Buffers(
        derived={
            "storage": Buffer(
                ratio=1.0,
                intent=BufferIntent.scale_down,
                components=[BufferComponent.storage],
            ),
        },
    ),
)

scenarios.append(
    (
        "org.netflix.cassandra",
        "us-east-1",
        cassandra_kv_compact_ebs,
        {
            "require_local_disks": False,
            "require_attached_disks": True,
            "experimental_memory_model": True,
        },
        "cassandra_kv_compact_ebs",
    )
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

# 500 MiB/s high-throughput Kafka — exercises EBS volume sizing in
# compute_stateful_zone to guard against regressions in max_node_disk_gib.
high_throughput = 500 * 1024 * 1024  # 500 MiB/s
kafka_high_throughput = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(low=2, mid=4, high=6, confidence=0.98),
        estimated_write_per_second=Interval(low=1, mid=1, high=1, confidence=0.98),
        estimated_mean_write_size_bytes=Interval(
            low=high_throughput,
            mid=high_throughput,
            high=high_throughput * 2,
            confidence=0.98,
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
        (
            "org.netflix.kafka",
            "us-east-1",
            kafka_high_throughput,
            {"require_local_disks": False},
            "kafka_500mib_throughput",
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

# Read-Only KV scenarios (partition-aware algorithm)
# Large dataset with many partitions
read_only_kv_large = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(20_000),
        estimated_mean_read_latency_ms=certain_float(2.0),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(48_000),
    ),
)

# Medium dataset
read_only_kv_medium = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(20_000),
        estimated_mean_read_latency_ms=certain_float(2.0),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(1397),
    ),
)

# Small dataset
read_only_kv_small = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(17_000),
        estimated_mean_read_latency_ms=certain_float(2.0),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(60),
    ),
)

scenarios.extend(
    [
        (
            "org.netflix.read-only-kv",
            "us-east-1",
            read_only_kv_large,
            {"total_num_partitions": 512, "min_replica_count": 4},
            "read_only_kv_large",
        ),
        (
            "org.netflix.read-only-kv",
            "us-east-1",
            read_only_kv_medium,
            {"total_num_partitions": 8, "min_replica_count": 3},
            "read_only_kv_medium",
        ),
        (
            "org.netflix.read-only-kv",
            "us-east-1",
            read_only_kv_small,
            {"total_num_partitions": 16, "min_replica_count": 4},
            "read_only_kv_small",
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

UNCERTAIN_SCENARIOS: dict[str, dict[str, Any]] = {
    name: SCENARIOS[name]
    for name in (
        "cassandra_timeseries_ebs",
        "cassandra_kv_dense_ebs",
        "cassandra_kv_compact_ebs",
        "kafka_100mib_throughput",
        "evcache_large_with_replication",
        "kv_with_cache",
    )
}

UNCERTAIN_EXPLAINED_SCENARIOS: dict[str, dict[str, Any]] = {
    name: SCENARIOS[name]
    for name in (
        "cassandra_timeseries_ebs",
        "cassandra_kv_dense_ebs",
        "cassandra_kv_compact_ebs",
        "kv_with_cache",
    )
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

    uncertain_results = []
    for scenario_name, scenario in UNCERTAIN_SCENARIOS.items():
        print(f"Capturing uncertain: {scenario_name}...")
        result = capture_uncertain(
            model_name=scenario["model"],
            region=scenario["region"],
            desires=scenario["desires"],
            extra_args=scenario["extra_args"],
            scenario_name=scenario_name,
        )
        uncertain_results.append(result)
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(
                "  Least regret families: "
                + ", ".join(
                    p["clusters"][0]["instance"]
                    for p in result["least_regret"]
                    if p["clusters"]
                )
            )

    # Save deterministic results
    output_dir = Path(__file__).parent / "data"
    output_file = output_dir / "baseline_costs.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")  # Ensure trailing newline for pre-commit

    uncertain_output_file = output_dir / "baseline_uncertain.json"
    with open(uncertain_output_file, "w", encoding="utf-8") as f:
        json.dump(uncertain_results, f, indent=2, sort_keys=True)
        f.write("\n")

    uncertain_explained_results = []
    for scenario_name, scenario in UNCERTAIN_EXPLAINED_SCENARIOS.items():
        print(f"Capturing uncertain explained: {scenario_name}...")
        result = capture_uncertain_explained(
            model_name=scenario["model"],
            region=scenario["region"],
            desires=scenario["desires"],
            extra_args=scenario["extra_args"],
            scenario_name=scenario_name,
        )
        uncertain_explained_results.append(result)
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(
                f"  Summaries: {len(result['least_regret_summaries'])} plans, "
                f"{len(result['excuse_summary'])} counted excuses"
            )

    uncertain_explained_output_file = output_dir / "baseline_uncertain_explained.json"
    with open(uncertain_explained_output_file, "w", encoding="utf-8") as f:
        json.dump(uncertain_explained_results, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"\nResults saved to: {output_file}")
    success_count = len([r for r in results if "error" not in r])
    print(f"Total scenarios captured: {success_count}/{len(results)}")
    uncertain_success_count = len([r for r in uncertain_results if "error" not in r])
    print(f"Uncertain results saved to: {uncertain_output_file}")
    print(
        f"Total uncertain scenarios captured: "
        f"{uncertain_success_count}/{len(uncertain_results)}"
    )
    uncertain_explained_success_count = len(
        [r for r in uncertain_explained_results if "error" not in r]
    )
    print(f"Uncertain explained results saved to: {uncertain_explained_output_file}")
    print(
        "Total uncertain explained scenarios captured: "
        f"{uncertain_explained_success_count}/{len(uncertain_explained_results)}"
    )

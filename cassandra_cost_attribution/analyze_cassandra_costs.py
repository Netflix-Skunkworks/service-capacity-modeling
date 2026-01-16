#!/usr/bin/env python3
"""
Analyze the Cassandra cost CSV to determine the best metrics for
cost attribution.
"""

import csv
from collections import defaultdict
from statistics import mean, stdev
from typing import Any


def load_cost_data() -> list[dict[str, Any]]:
    """Load and parse the cost analysis CSV."""
    data = []
    with open("cassandra_cost_analysis.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(
                {
                    "reads": int(row["reads_per_sec"]),
                    "writes": int(row["writes_per_sec"]),
                    "data_gib": int(row["data_size_gib"]),
                    "cost": float(row["total_annual_cost"]),
                    "cost_per_read": float(row["cost_per_read"]),
                    "cost_per_write": float(row["cost_per_write"]),
                    "cost_per_gib": float(row["cost_per_gib"]),
                    "cost_per_total_ops": float(row["cost_per_total_ops"]),
                    "instance_count": int(row["instance_count"]),
                    "cpu_cores": int(row["total_cpu_cores"]),
                }
            )
    return data


def print_metric_statistics(data: list[dict[str, Any]]) -> None:
    """Print statistics for cost per metric."""
    print("1. COST PER METRIC STATISTICS")
    print("-" * 80)

    metrics = {
        "Cost per Read ($/read/year)": [d["cost_per_read"] for d in data],
        "Cost per Write ($/write/year)": [d["cost_per_write"] for d in data],
        "Cost per GiB ($/GiB/year)": [d["cost_per_gib"] for d in data],
        "Cost per Total Op ($/op/year)": [d["cost_per_total_ops"] for d in data],
    }

    for metric_name, values in metrics.items():
        print(f"\n{metric_name}:")
        print(f"  Min:    {min(values):.6f}")
        print(f"  Max:    {max(values):.6f}")
        print(f"  Mean:   {mean(values):.6f}")
        print(f"  StdDev: {stdev(values):.6f}")
        cv = stdev(values) / mean(values)
        print(f"  Coefficient of Variation: {cv:.2%}")


def print_workload_patterns(data: list[dict[str, Any]]) -> None:
    """Print cost patterns by workload type."""
    print("\n" + "=" * 80)
    print("2. COST PATTERNS BY WORKLOAD TYPE")
    print("-" * 80)

    # Categorize workloads
    read_heavy = [d for d in data if d["reads"] > 10 * d["writes"]]
    write_heavy = [d for d in data if d["writes"] > 10 * d["reads"]]
    balanced = [d for d in data if d not in read_heavy and d not in write_heavy]

    print(f"\nRead-Heavy workloads ({len(read_heavy)} scenarios):")
    if read_heavy:
        print(f"  Avg cost: ${mean([d['cost'] for d in read_heavy]):,.2f}/year")
        print(f"  Cost per read: ${mean([d['cost_per_read'] for d in read_heavy]):.6f}")
        print(
            f"  Cost per write: ${mean([d['cost_per_write'] for d in read_heavy]):.6f}"
        )

    print(f"\nWrite-Heavy workloads ({len(write_heavy)} scenarios):")
    if write_heavy:
        print(f"  Avg cost: ${mean([d['cost'] for d in write_heavy]):,.2f}/year")
        print(
            f"  Cost per read: ${mean([d['cost_per_read'] for d in write_heavy]):.6f}"
        )
        print(
            f"  Cost per write: ${mean([d['cost_per_write'] for d in write_heavy]):.6f}"
        )

    print(f"\nBalanced workloads ({len(balanced)} scenarios):")
    if balanced:
        print(f"  Avg cost: ${mean([d['cost'] for d in balanced]):,.2f}/year")
        print(f"  Cost per read: ${mean([d['cost_per_read'] for d in balanced]):.6f}")
        print(f"  Cost per write: ${mean([d['cost_per_write'] for d in balanced]):.6f}")


def print_scaling_analysis(data: list[dict[str, Any]]) -> None:
    """Print analysis of how cost scales with workload parameters."""
    print("\n" + "=" * 80)
    print("3. SCALING ANALYSIS")
    print("-" * 80)

    # Group by data size and analyze cost scaling
    by_data_size = defaultdict(list)
    for d in data:
        by_data_size[d["data_gib"]].append(d)

    print("\nHow cost scales with operations (holding data size constant):")
    for data_gib in sorted(by_data_size.keys()):
        scenarios = by_data_size[data_gib]
        min_cost = min(s["cost"] for s in scenarios)
        max_cost = max(s["cost"] for s in scenarios)
        print(f"\n  Data size: {data_gib} GiB")
        variation = max_cost - min_cost
        print(
            f"    Cost range: ${min_cost:,.2f} - ${max_cost:,.2f} "
            f"(${variation:,.2f} variation)"
        )
        print(f"    Cost multiplier: {max_cost / min_cost:.1f}x")

    # Group by total ops and analyze cost scaling
    total_ops_buckets = defaultdict(list)
    for d in data:
        total_ops = d["reads"] + d["writes"]
        total_ops_buckets[total_ops].append(d)

    print("\nHow cost scales with data size (holding ops roughly constant):")
    for total_ops in sorted(total_ops_buckets.keys())[:5]:
        scenarios = total_ops_buckets[total_ops]
        if len(scenarios) > 1:
            min_cost = min(s["cost"] for s in scenarios)
            max_cost = max(s["cost"] for s in scenarios)
            print(f"\n  Total ops: {total_ops:,}/sec")
            variation = max_cost - min_cost
            print(
                f"    Cost range: ${min_cost:,.2f} - ${max_cost:,.2f} "
                f"(${variation:,.2f} variation)"
            )
            print(f"    Cost multiplier: {max_cost / min_cost:.1f}x")


def print_recommendations() -> None:
    """Print recommendations for cost attribution."""
    print("\n" + "=" * 80)
    print("4. RECOMMENDATIONS FOR COST ATTRIBUTION")
    print("-" * 80)
    print("""
Based on the coefficient of variation (lower = more stable/predictable):

1. BEST OPTION - Weighted Combination:
   - Use a formula like: cost = (A * reads) + (B * writes) + (C * data)
   - This accounts for the fact that reads, writes, and data size all
     drive cost
   - Weights can be determined via regression analysis on this dataset

2. ALTERNATIVE - Total Operations:
   - Cost per total operation has moderate variation
   - Simple to explain and implement
   - May underweight data size impact

3. NOT RECOMMENDED - Single Metric:
   - Cost per read/write/GiB alone all have high variation
   - They depend too much on the other dimensions to be used alone

4. PRACTICAL APPROACH - Bucketing:
   - Classify tables into workload types (read-heavy, write-heavy,
     balanced)
   - Use different attribution formulas for each type
   - Within each type, use a weighted combination of metrics

Example weighted formula (requires regression to determine A, B, C):
  table_cost = cluster_total_cost * (
      (A * table_reads / cluster_reads) +
      (B * table_writes / cluster_writes) +
      (C * table_data_gib / cluster_data_gib)
  ) where A + B + C = 1
    """)


def print_next_steps() -> None:
    """Print suggested next steps."""
    print("\n" + "=" * 80)
    print("5. SUGGESTED NEXT STEPS")
    print("-" * 80)
    print("""
1. Run regression analysis on the CSV to determine optimal weights for:
      cost ~ w1*reads + w2*writes + w3*data_size

2. Validate the model:
   - Compare predicted costs vs actual costs from the CSV
   - Calculate R-squared to measure goodness of fit

3. For your cluster, gather per-table metrics:
   - Reads/sec per table (from metrics)
   - Writes/sec per table (from metrics)
   - Data size per table (from disk usage)

4. Apply the weighted formula to attribute costs to each table

5. Sanity check: sum of all table costs should equal cluster cost
    """)


def analyze_cost_drivers() -> None:
    """Analyze which metrics best correlate with cost."""
    data = load_cost_data()

    print("=" * 80)
    print("CASSANDRA COST ATTRIBUTION ANALYSIS")
    print("=" * 80)
    print()

    print_metric_statistics(data)
    print_workload_patterns(data)
    print_scaling_analysis(data)
    print_recommendations()
    print_next_steps()


if __name__ == "__main__":
    analyze_cost_drivers()

#!/usr/bin/env python3
"""
Generate a CSV table showing the relationship between Cassandra workload
parameters and cost. This helps determine the best metrics for attributing
cluster costs to individual tables.
"""

import csv
import sys
from itertools import product

# pylint: disable=import-error
from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import (
    CapacityDesires,
    DataShape,
    QueryPattern,
    certain_int,
)


def run_capacity_analysis() -> None:
    """
    Run capacity planning across various combinations of reads/sec,
    writes/sec, and data size. Output results to CSV for cost
    attribution analysis.
    """

    # Define the parameter ranges to test
    read_rates = [1_000, 10_000, 50_000, 100_000, 500_000]
    write_rates = [1_000, 10_000, 50_000, 100_000, 500_000]
    data_sizes = [10, 100, 500, 1_000, 5_000]  # GiB

    # CSV output
    fieldnames = [
        "reads_per_sec",
        "writes_per_sec",
        "data_size_gib",
        "total_annual_cost",
        "instance_type",
        "instance_count",
        "total_cpu_cores",
        "total_ram_gib",
        "total_disk_gib",
        "cost_per_read",
        "cost_per_write",
        "cost_per_gib",
        "cost_per_total_ops",
    ]

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()

    # Test each combination
    for reads, writes, data_gib in product(read_rates, write_rates, data_sizes):
        try:
            desires = CapacityDesires(
                service_tier=1,
                query_pattern=QueryPattern(
                    estimated_read_per_second=certain_int(reads),
                    estimated_write_per_second=certain_int(writes),
                ),
                data_shape=DataShape(
                    estimated_state_size_gib=certain_int(data_gib),
                ),
            )

            # Get the capacity plan
            cap_plan = planner.plan_certain(
                model_name="org.netflix.cassandra",
                region="us-east-1",
                desires=desires,
                num_results=1,
            )[0]

            # Extract metrics
            cluster = cap_plan.candidate_clusters.zonal[0]
            total_cost = cap_plan.candidate_clusters.total_annual_cost

            total_cpu = cluster.count * cluster.instance.cpu
            total_ram = cluster.count * cluster.instance.ram_gib

            # Calculate total disk (local + attached)
            total_disk = 0
            if cluster.instance.drive:
                total_disk += cluster.count * cluster.instance.drive.size_gib
            if cluster.attached_drives:
                for drive in cluster.attached_drives:
                    total_disk += cluster.count * drive.size_gib

            # Calculate cost attribution metrics
            total_ops = reads + writes
            cost_per_read = total_cost / reads if reads > 0 else 0
            cost_per_write = total_cost / writes if writes > 0 else 0
            cost_per_gib = total_cost / data_gib if data_gib > 0 else 0
            cost_per_total_ops = total_cost / total_ops if total_ops > 0 else 0

            row = {
                "reads_per_sec": reads,
                "writes_per_sec": writes,
                "data_size_gib": data_gib,
                "total_annual_cost": f"{total_cost:.2f}",
                "instance_type": cluster.instance.name,
                "instance_count": cluster.count,
                "total_cpu_cores": total_cpu,
                "total_ram_gib": f"{total_ram:.2f}",
                "total_disk_gib": f"{total_disk:.2f}",
                "cost_per_read": f"{cost_per_read:.6f}",
                "cost_per_write": f"{cost_per_write:.6f}",
                "cost_per_gib": f"{cost_per_gib:.2f}",
                "cost_per_total_ops": f"{cost_per_total_ops:.6f}",
            }

            writer.writerow(row)
            sys.stdout.flush()

        except Exception as e:  # pylint: disable=broad-exception-caught
            # Log failures but continue
            sys.stderr.write(
                f"Failed for reads={reads}, writes={writes}, data={data_gib}: {e}\n"
            )
            continue


if __name__ == "__main__":
    run_capacity_analysis()

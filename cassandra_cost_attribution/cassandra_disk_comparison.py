#!/usr/bin/env python3
"""
Compare cost attribution for EBS vs local SSD Cassandra configurations.
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


def run_disk_comparison_analysis() -> None:  # pylint: disable=too-many-locals
    """
    Run capacity planning for both EBS and local SSD configurations.
    Compare cost attribution multipliers between disk types.
    """

    # Define parameter ranges - smaller set for comparison
    read_rates = [1_000, 10_000, 50_000, 100_000]
    write_rates = [1_000, 10_000, 50_000, 100_000]
    data_sizes = [100, 500, 1_000, 5_000]  # GiB

    # Test both disk configurations
    disk_configs = [
        {"name": "ebs", "require_attached_disks": True, "require_local_disks": False},
        {
            "name": "local_ssd",
            "require_attached_disks": False,
            "require_local_disks": True,
        },
    ]

    # CSV output
    fieldnames = [
        "disk_type",
        "reads_per_sec",
        "writes_per_sec",
        "data_size_gib",
        "total_annual_cost",
        "instance_type",
        "instance_count",
        "total_cpu_cores",
        "total_ram_gib",
        "total_disk_gib",
        "disk_iops",
        "cost_per_read",
        "cost_per_write",
        "cost_per_gib",
        "cost_per_total_ops",
    ]

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()

    for disk_config in disk_configs:
        disk_type = disk_config["name"]
        sys.stderr.write(f"Processing {disk_type} configurations...\n")

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

                # Get the capacity plan with disk-specific configuration
                cap_plan = planner.plan_certain(
                    model_name="org.netflix.cassandra",
                    region="us-east-1",
                    desires=desires,
                    extra_model_arguments={
                        "require_attached_disks": disk_config["require_attached_disks"],
                        "require_local_disks": disk_config["require_local_disks"],
                    },
                    num_results=1,
                )[0]

                # Extract metrics
                cluster = cap_plan.candidate_clusters.zonal[0]
                total_cost = cap_plan.candidate_clusters.total_annual_cost

                total_cpu = cluster.count * cluster.instance.cpu
                total_ram = cluster.count * cluster.instance.ram_gib

                # Calculate total disk and IOPS
                total_disk = 0
                disk_iops = 0

                if cluster.instance.drive:
                    # Local SSD
                    total_disk += cluster.count * cluster.instance.drive.size_gib
                    disk_iops = cluster.instance.drive.read_io_per_s or 0

                if cluster.attached_drives:
                    # EBS
                    for drive in cluster.attached_drives:
                        total_disk += cluster.count * drive.size_gib
                        disk_iops = drive.read_io_per_s or 0

                # Calculate cost attribution metrics
                total_ops = reads + writes
                cost_per_read = total_cost / reads if reads > 0 else 0
                cost_per_write = total_cost / writes if writes > 0 else 0
                cost_per_gib = total_cost / data_gib if data_gib > 0 else 0
                cost_per_total_ops = total_cost / total_ops if total_ops > 0 else 0

                row = {
                    "disk_type": disk_type,
                    "reads_per_sec": reads,
                    "writes_per_sec": writes,
                    "data_size_gib": data_gib,
                    "total_annual_cost": f"{total_cost:.2f}",
                    "instance_type": cluster.instance.name,
                    "instance_count": cluster.count,
                    "total_cpu_cores": total_cpu,
                    "total_ram_gib": f"{total_ram:.2f}",
                    "total_disk_gib": f"{total_disk:.2f}",
                    "disk_iops": disk_iops,
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
                    f"Failed for {disk_type}, reads={reads}, "
                    f"writes={writes}, data={data_gib}: {e}\n"
                )
                continue

        sys.stderr.write(f"Completed {disk_type} configurations.\n\n")


if __name__ == "__main__":
    run_disk_comparison_analysis()

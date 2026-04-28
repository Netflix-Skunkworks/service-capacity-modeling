"""One-off debug: write-heavy Cassandra + experimental memory model.

Before the rps_working_set fix, the observational memory path dropped the
rps bound, oversizing memory for write-heavy workloads and forcing
large-RAM instances. After the fix, a write-heavy workload on an
SLO-limited drive should see ``ws`` pulled down by ``rps_ws`` and allow
smaller-RAM candidates.
"""

from __future__ import annotations

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires

_DESIRE = CapacityDesires.model_validate(
    {
        "service_tier": 1,
        "query_pattern": {
            "access_pattern": "latency",
            "estimated_read_per_second": {
                "low": 90,
                "mid": 3285,
                "high": 12286,
                "confidence": 0.98,
            },
            "estimated_write_per_second": {
                "low": 1958,
                "mid": 89446,
                "high": 735928,
                "confidence": 0.98,
            },
        },
        "data_shape": {
            "estimated_state_size_gib": {
                "low": 428.25,
                "mid": 3089.25,
                "high": 8461.166666666666,
                "confidence": 0.98,
            },
            "estimated_compression_ratio": {
                "low": 1,
                "mid": 1,
                "high": 1,
                "confidence": 1,
            },
        },
        "current_clusters": {
            "zonal": [
                {
                    "cluster_instance_name": "r7a.4xlarge",
                    "cluster_drive": {
                        "name": "gp3",
                        "drive_type": "attached-ssd",
                        "size_gib": 2524,
                        "annual_cost": 0,
                    },
                    "cluster_instance_count": {
                        "low": 8,
                        "mid": 8,
                        "high": 8,
                        "confidence": 1,
                    },
                    "cluster_type": "cassandra",
                    "cpu_utilization": {
                        "low": 2.86,
                        "mid": 21.86,
                        "high": 83.44,
                        "confidence": 0.98,
                    },
                    "network_utilization_mbps": {
                        "low": 4.92,
                        "mid": 240.37,
                        "high": 2364.30,
                        "confidence": 0.98,
                    },
                    "disk_utilization_gib": {
                        "low": 53.53,
                        "mid": 386.16,
                        "high": 1057.65,
                        "confidence": 0.98,
                    },
                }
            ],
            "current_asg_size": 3,
            "max_disk_used_gib": 461.32,
        },
    }
)

_EXTRA_ARGS = {
    "experimental_memory_model": True,
    "require_local_disks": False,
    "require_attached_disks": True,
}


def test_4xlarge_class_instances_are_candidates():
    """4xlarge boxes (~61 GiB RAM) must fit the memory partition.

    Regression guard: heap (30) + base (3) + page_cache_capacity (28) = 61
    is the tightest budget the planner targets. If any of those caps
    drift such that 64-GiB-class instances stop appearing as candidates,
    this test fails.
    """
    plans = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=_DESIRE,
        extra_model_arguments=_EXTRA_ARGS,
        num_results=200,
    )
    candidate_names = {p.candidate_clusters.zonal[0].instance.name for p in plans}
    four_xlarge_candidates = {n for n in candidate_names if ".4xlarge" in n}
    assert four_xlarge_candidates, (
        f"No 4xlarge-class instance is a candidate. Candidates: {candidate_names}"
    )


def test_m7a_4xlarge_fits_same_cluster_size_as_r7a_4xlarge():
    """Page-cache demand should not force the larger-RAM r7a shape."""
    plans = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=_DESIRE,
        instance_families=["m7a", "r7a"],
        extra_model_arguments=_EXTRA_ARGS,
        max_results_per_family=20,
        num_results=100,
    )
    by_instance = {p.candidate_clusters.zonal[0].instance.name: p for p in plans}

    m7a = by_instance["m7a.4xlarge"].candidate_clusters.zonal[0]
    r7a = by_instance["r7a.4xlarge"].candidate_clusters.zonal[0]

    assert m7a.count == r7a.count == 8
    assert m7a.cluster_params["required_nodes_by_type"]["memory"] == 8

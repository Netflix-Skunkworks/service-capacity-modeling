"""Tests for per-resource node count breakdown in compute_stateful_zone."""

import pytest

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.models.common import compute_stateful_zone

EBS = Drive(name="gp3", size_gib=0)
M5_4XL = shapes.instance("m5.4xlarge")  # 16 vCPU, 64 GiB
I4I_4XL = shapes.instance("i4i.4xlarge")

EXPECTED_KEYS = {"cpu", "memory", "network", "storage", "disk_iops"}


@pytest.mark.parametrize(
    "cores,mem,disk,net,expected_binding",
    [
        (48, 10, 100, 100, "cpu"),  # ceil(48/16)=3 dominates
        (4, 200, 100, 100, "memory"),  # ceil(200/64)=4 dominates
        (4, 10, 100, 60_000, "network"),  # ceil(60000/10000)=6 dominates
    ],
    ids=["cpu-bound", "memory-bound", "network-bound"],
)
def test_binding_resource(cores, mem, disk, net, expected_binding):
    cluster = compute_stateful_zone(
        instance=M5_4XL,
        drive=EBS,
        needed_cores=cores,
        needed_disk_gib=disk,
        needed_memory_gib=mem,
        needed_network_mbps=net,
    )
    nrb = cluster.cluster_params["nodes_required_by"]
    assert set(nrb.keys()) == EXPECTED_KEYS
    assert max(nrb, key=nrb.get) == expected_binding


def test_storage_bound_local():
    cluster = compute_stateful_zone(
        instance=I4I_4XL,
        drive=I4I_4XL.drive,
        needed_cores=4,
        needed_disk_gib=20_000,
        needed_memory_gib=10,
        needed_network_mbps=100,
    )
    nrb = cluster.cluster_params["nodes_required_by"]
    assert max(nrb, key=nrb.get) == "storage"


def test_write_buffer_merged_into_memory():
    """Write buffer inflates memory count (same RAM, different slice)."""
    cluster = compute_stateful_zone(
        instance=M5_4XL,
        drive=EBS,
        needed_cores=4,
        needed_disk_gib=100,
        needed_memory_gib=10,
        needed_network_mbps=100,
        reserve_memory=lambda x: 4.0,
        write_buffer=lambda x: 0.25,  # 0.25 GiB per node
        required_write_buffer_gib=2.0,  # ceil(2.0/0.25) = 8 nodes
    )
    assert cluster.cluster_params["nodes_required_by"]["memory"] == 8

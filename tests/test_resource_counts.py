"""Tests for node-count explainability in compute_stateful_zone."""

import pytest

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import NodeCountConstraint
from service_capacity_modeling.models.common import compute_stateful_zone

EBS = Drive(name="gp3", size_gib=0)
M5_4XL = shapes.instance("m5.4xlarge")  # 16 vCPU, 64 GiB
I4I_4XL = shapes.instance("i4i.4xlarge")

EXPECTED_KEYS = {
    "cpu",
    "memory",
    "network",
    "disk_capacity",
    "disk_iops",
    "cluster_size",
    "min_count",
}


@pytest.mark.parametrize(
    "cores,mem,disk,net,expected_bottleneck",
    [
        (48, 10, 100, 100, "cpu"),  # ceil(48/16)=3 dominates
        (4, 200, 100, 100, "memory"),  # ceil(200/64)=4 dominates
        (4, 10, 100, 60_000, "network"),  # ceil(60000/10000)=6 dominates
    ],
    ids=["cpu-bound", "memory-bound", "network-bound"],
)
def test_count_bottleneck_resource(cores, mem, disk, net, expected_bottleneck):
    cluster = compute_stateful_zone(
        instance=M5_4XL,
        drive=EBS,
        needed_cores=cores,
        needed_disk_gib=disk,
        needed_memory_gib=mem,
        needed_network_mbps=net,
    )
    context = cluster.node_count_context
    assert context is not None
    assert {k.value for k in context.required_nodes_by_type} == EXPECTED_KEYS
    assert context.resource_bottleneck == NodeCountConstraint(expected_bottleneck)


def test_storage_bound_local():
    cluster = compute_stateful_zone(
        instance=I4I_4XL,
        drive=I4I_4XL.drive,
        needed_cores=4,
        needed_disk_gib=20_000,
        needed_memory_gib=10,
        needed_network_mbps=100,
    )
    counts = cluster.cluster_params["required_nodes_by_type"]
    assert cluster.cluster_params["resource_bottleneck"] == "disk_capacity"
    assert counts["disk_capacity"] > counts["cpu"]


def test_attached_drive_iops_overflow_recalculates_per_node_iops():
    cluster = compute_stateful_zone(
        instance=M5_4XL,
        drive=Drive(
            name="tiny-ebs",
            size_gib=0,
            max_scale_size_gib=1_000,
            max_scale_io_per_s=1_000,
        ),
        needed_cores=4,
        needed_disk_gib=100,
        needed_memory_gib=10,
        needed_network_mbps=100,
        required_disk_ios=lambda _size, count: (1200 / count, 0.0),
    )

    attached_drive = cluster.attached_drives[0]
    counts = cluster.cluster_params["required_nodes_by_type"]
    assert counts["disk_iops"] == 2
    assert cluster.count == 2
    assert attached_drive.read_io_per_s == 600
    assert attached_drive.read_io_per_s < attached_drive.max_io_per_s


def test_attached_drive_capacity_overflow_recalculates_per_node_ios():
    cluster = compute_stateful_zone(
        instance=M5_4XL,
        drive=Drive(
            name="tiny-ebs",
            size_gib=0,
            max_scale_size_gib=100,
            max_scale_io_per_s=1_000,
        ),
        needed_cores=4,
        needed_disk_gib=200,
        needed_memory_gib=10,
        needed_network_mbps=100,
        required_disk_ios=lambda size, _count: (size * 3.0, 0.0),
        max_node_disk_gib=lambda d: int(d.max_size_gib),
    )

    attached_drive = cluster.attached_drives[0]
    assert cluster.node_count_context is not None
    counts = {
        k.value: v for k, v in cluster.node_count_context.required_nodes_by_type.items()
    }
    assert counts["disk_capacity"] == 2
    assert cluster.count == 2
    assert attached_drive.size_gib == 100
    assert attached_drive.read_io_per_s == 400


def test_gp2_attached_drive_recomputes_per_node_size_after_count_increase():
    cluster = compute_stateful_zone(
        instance=M5_4XL,
        drive=Drive(
            name="gp2",
            size_gib=0,
            max_scale_size_gib=1_000,
        ),
        needed_cores=4,
        needed_disk_gib=100,
        needed_memory_gib=10,
        needed_network_mbps=100,
        required_disk_ios=lambda _size, count: (7500 / count, 0.0),
        max_node_disk_gib=lambda d: int(d.max_size_gib),
    )

    attached_drive = cluster.attached_drives[0]
    assert cluster.node_count_context is not None
    counts = {
        k.value: v for k, v in cluster.node_count_context.required_nodes_by_type.items()
    }
    assert counts["disk_capacity"] == 3
    assert cluster.count == 3
    assert attached_drive.size_gib == 900
    assert attached_drive.read_io_per_s == 2600


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
    assert cluster.cluster_params["required_nodes_by_type"]["memory"] == 8


def test_cluster_size_is_reported_when_rounding_adds_nodes():
    cluster = compute_stateful_zone(
        instance=M5_4XL,
        drive=EBS,
        needed_cores=48,  # ceil(48/16) = 3 nodes
        needed_disk_gib=100,
        needed_memory_gib=10,
        needed_network_mbps=100,
        cluster_size=lambda x: x if x % 2 == 0 else x + 1,  # round up to even
    )
    counts = cluster.cluster_params["required_nodes_by_type"]
    assert counts["cpu"] == 3
    assert counts["cluster_size"] == 4
    assert counts["min_count"] == 0
    assert cluster.cluster_params["resource_bottleneck"] == "cpu"
    assert cluster.count == 4


def test_min_count_is_reported_when_it_adds_nodes():
    cluster = compute_stateful_zone(
        instance=M5_4XL,
        drive=EBS,
        needed_cores=4,  # ceil(4/16) = 1 node
        needed_disk_gib=100,
        needed_memory_gib=10,
        needed_network_mbps=100,
        min_count=6,
    )
    counts = cluster.cluster_params["required_nodes_by_type"]
    assert counts["cpu"] == 1
    assert counts["cluster_size"] == 1
    assert counts["min_count"] == 6
    assert cluster.cluster_params["resource_bottleneck"] == "cpu"
    assert cluster.count == 6


def test_topology_constraints_do_not_override_stronger_resource_limits():
    cluster = compute_stateful_zone(
        instance=M5_4XL,
        drive=EBS,
        needed_cores=640,  # ceil(640/16) = 40 nodes
        needed_disk_gib=100,
        needed_memory_gib=10,
        needed_network_mbps=100,
        min_count=6,
    )
    counts = cluster.cluster_params["required_nodes_by_type"]
    assert counts["cpu"] == 40
    assert counts["min_count"] == 6
    assert cluster.cluster_params["resource_bottleneck"] == "cpu"

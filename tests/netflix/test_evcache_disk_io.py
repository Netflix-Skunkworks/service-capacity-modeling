from copy import deepcopy

import pytest

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import (
    Buffer,
    BufferIntent,
    Buffers,
    CapacityDesires,
    CurrentClusters,
    CurrentZoneClusterCapacity,
    DataShape,
    Drive,
    QueryPattern,
    RegionContext,
    certain_float,
)
from service_capacity_modeling.models.org.netflix import evcache
from service_capacity_modeling.models.org.netflix.evcache_disk_io import (
    disk_io_requirement,
)
from service_capacity_modeling.tools.capture_baseline_costs import _format_cluster


def plan(
    *,
    observation=None,
    compute=1.5,
    disk_buffer=None,
    growth=1,
    intent="scale",
    components=("compute", "disk_iops"),
    copies=3,
    storage_buffer=1,
):
    instance = shapes.instance("i4i.large").model_copy(deep=True)
    instance.drive.read_io_per_s = 100
    instance.drive.write_io_per_s = 50
    instance.drive.size_gib = 1000
    instance.drive.block_size_kib = 4
    desired = {"compute": Buffer(ratio=compute, components=["compute"])}
    desired["storage"] = Buffer(ratio=storage_buffer, components=["storage"])
    if disk_buffer is not None:
        desired["io"] = Buffer(ratio=disk_buffer, components=["disk_iops"])
    desires = CapacityDesires(
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_float(10),
            estimated_write_per_second=certain_float(1),
            estimated_mean_read_size_bytes=certain_float(1024),
            estimated_mean_write_size_bytes=certain_float(1024),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_float(200)),
        current_clusters=CurrentClusters(
            zonal=[
                CurrentZoneClusterCapacity(
                    cluster_type="evcache",
                    cluster_instance_name=instance.name,
                    cluster_instance=instance,
                    cluster_instance_count=certain_float(10),
                    cluster_drive=instance.drive,
                    cpu_utilization=certain_float(1),
                    memory_utilization_gib=certain_float(1),
                    disk_utilization_gib=certain_float(20),
                    network_utilization_mbps=certain_float(1),
                )
            ]
        ),
        buffers=Buffers(
            default=Buffer(ratio=1),
            desired=desired,
            derived={
                "event": Buffer(
                    ratio=growth,
                    intent=BufferIntent(intent),
                    components=list(components),
                )
            },
        ),
    )
    args = {"copies_per_region": copies}
    if observation is not None:
        args["observed_disk_io"] = observation
    return evcache.NflxEVCacheCapacityModel.capacity_plan(
        instance, instance.drive, RegionContext(), desires, args
    )


def observation(**overrides):
    return {
        "read_bytes_per_second": 1000 * 4096,
        "write_bytes_per_second": 0,
        "read_ops_per_second": 1000,
        "write_ops_per_second": 0,
        **overrides,
    }


def io_params(result):
    return result.candidate_clusters.zonal[0].cluster_params["evcache.disk_io"]


def test_observed_disk_ops_increase_the_planned_node_count():
    result = plan(observation=observation())
    assert result.candidate_clusters.zonal[0].count == 15
    assert io_params(result)["source"] == "observed"
    assert (
        result.candidate_clusters.zonal[0].node_count_context.required_nodes_by_type[
            "disk_iops"
        ]
        == 15
    )


def test_dedicated_io_buffer_overrides_compute_instead_of_multiplying():
    result = plan(observation=observation(), compute=3, disk_buffer=1.2)
    assert io_params(result)["buffer_ratio"] == 1.2
    assert io_params(result)["required_nodes"] == 12


def test_event_growth_applies_once_when_rule_names_both_components():
    result = plan(observation=observation(), growth=1.17)
    assert io_params(result)["required_nodes"] == 18
    assert io_params(result)["scale"] == 1.17


def test_bytes_and_operations_can_each_govern_node_count():
    # Operations need 10 nodes, but larger operations need 20 for throughput.
    result = plan(observation=observation(read_bytes_per_second=2000 * 4096), compute=1)
    assert io_params(result)["required_nodes"] == 20
    # Small operations must not bypass the operations constraint.
    result = plan(observation=observation(read_bytes_per_second=1000), compute=1)
    assert io_params(result)["required_nodes"] == 10


def test_read_and_write_share_the_device_budget():
    result = plan(
        observation=observation(
            write_ops_per_second=500, write_bytes_per_second=500 * 4096
        ),
        compute=1,
    )
    assert io_params(result)["required_nodes"] == 20


def test_zero_observation_is_accepted_and_not_replaced_with_request_estimate():
    result = plan(observation={key: 0 for key in observation()})
    assert io_params(result)["source"] == "observed"
    assert io_params(result)["required_nodes"] == 0


def test_missing_observation_retains_request_estimation_with_resolved_buffer():
    result = plan(compute=2)
    assert io_params(result)["source"] == "requests"
    assert io_params(result)["buffer_ratio"] == 2


def test_existing_capacity_floor_is_preserved_for_io():
    result = plan(
        observation=observation(read_ops_per_second=1, read_bytes_per_second=4096),
        compute=1,
        intent="scale_up",
    )
    assert io_params(result)["required_nodes"] == 10


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
@pytest.mark.parametrize("field", list(observation()))
def test_invalid_explicit_disk_observation_is_rejected(field, value):
    with pytest.raises(ValueError):
        plan(observation=observation(**{field: value}))


def test_partial_explicit_disk_observation_is_rejected():
    value = deepcopy(observation())
    value.pop("read_ops_per_second")
    with pytest.raises(ValueError):
        plan(observation=value)


def test_compute_only_event_scales_disk_and_storage_does_not_add_a_margin():
    result = plan(
        observation=observation(),
        growth=1.17,
        components=("compute",),
        storage_buffer=3,
    )
    assert io_params(result)["required_nodes"] == 18
    assert io_params(result)["buffer_ratio"] == 1.5


@pytest.mark.parametrize("copies", [1, 2, 3, 6])
def test_replica_count_does_not_multiply_one_replicas_io(copies):
    result = plan(observation=observation(), copies=copies)
    assert len(result.candidate_clusters.zonal) == copies
    assert all(cluster.count == 15 for cluster in result.candidate_clusters.zonal)


def test_zero_traffic_preserves_existing_capacity_for_scale_up():
    result = plan(observation={key: 0 for key in observation()}, intent="scale_up")
    assert io_params(result)["required_nodes"] == 10


def test_explicit_throughput_can_be_the_binding_limit():
    drive = Drive(
        name="bounded",
        size_gib=100,
        read_io_per_s=10000,
        write_io_per_s=10000,
        throughput=1,
    )
    demand = evcache.EVCacheDiskIO(**observation(read_bytes_per_second=2 * 1024**2))
    assert demand.node_load(drive) == 2


def test_missing_drive_capability_does_not_mean_unlimited_io():
    with pytest.raises(ValueError, match="Missing read IO capacity"):
        evcache.EVCacheDiskIO(**observation()).node_load(Drive(name="unknown"))


def test_request_diagnostics_snapshot_ignores_sampled_rate_drift():
    cluster = plan().candidate_clusters.zonal[0]
    changed = cluster.model_copy(deep=True)
    changed.cluster_params["evcache.disk_io"]["observation"][
        "write_bytes_per_second"
    ] += 0.001
    assert _format_cluster(cluster, "zonal") == _format_cluster(changed, "zonal")
    assert (
        cluster.cluster_params["evcache.disk_io"]["observation"]
        != changed.cluster_params["evcache.disk_io"]["observation"]
    )
    changed.cluster_params["evcache.disk_io"]["required_nodes"] += 1
    assert _format_cluster(cluster, "zonal") != _format_cluster(changed, "zonal")


def test_explicit_disk_observations_remain_in_snapshot():
    cluster = plan(observation=observation()).candidate_clusters.zonal[0]
    changed = cluster.model_copy(deep=True)
    changed.cluster_params["evcache.disk_io"]["observation"][
        "write_bytes_per_second"
    ] += 0.001
    assert _format_cluster(cluster, "zonal") != _format_cluster(changed, "zonal")


@pytest.fixture
def bounded_io_case():
    instance = shapes.instance("i4i.large").model_copy(deep=True)
    instance.drive.read_io_per_s = 1024
    instance.drive.write_io_per_s = 1024
    instance.drive.block_size_kib = 4
    current = CurrentZoneClusterCapacity(
        cluster_type="evcache",
        cluster_instance_name=instance.name,
        cluster_instance=instance.model_copy(deep=True),
        cluster_instance_count=certain_float(10),
        cluster_drive=instance.drive.model_copy(deep=True),
        disk_utilization_gib=certain_float(20),
    )
    desires = CapacityDesires(
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_float(1),
            estimated_write_per_second=certain_float(0),
            estimated_mean_read_size_bytes=certain_float(4096),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_float(200)),
        current_clusters=CurrentClusters(zonal=[current]),
        buffers=Buffers(
            default=Buffer(ratio=1),
            derived={
                "preserve_io": Buffer(
                    ratio=1,
                    intent=BufferIntent.scale_up,
                    components=["disk_iops"],
                )
            },
        ),
    )
    return instance, current, desires


@pytest.mark.parametrize("source", ["observed", "requests"])
@pytest.mark.parametrize("override_iops,expected", [(None, 10), (512, 5)])
def test_sparse_current_drive_uses_known_hardware_capabilities(
    bounded_io_case, source, override_iops, expected
):
    instance, current, desires = bounded_io_case
    drive = {"name": instance.drive.name, "size_gib": instance.drive.size_gib}
    if override_iops is not None:
        drive["read_io_per_s"] = override_iops
    current.cluster_drive = Drive(**drive)
    args = (
        {
            "observed_disk_io": observation(
                read_ops_per_second=1, read_bytes_per_second=4096
            )
        }
        if source == "observed"
        else {}
    )
    result = evcache.NflxEVCacheCapacityModel.capacity_plan(
        instance, instance.drive, RegionContext(), desires, args
    )
    assert io_params(result)["required_nodes"] == expected


@pytest.mark.parametrize("source", ["observed", "requests"])
def test_sparse_current_drive_preserves_zero_traffic_capacity(bounded_io_case, source):
    instance, current, desires = bounded_io_case
    current.cluster_drive = Drive(
        name=instance.drive.name, size_gib=instance.drive.size_gib
    )
    desires.query_pattern.estimated_read_per_second = certain_float(0)
    args = (
        {"observed_disk_io": {key: 0 for key in observation()}}
        if source == "observed"
        else {}
    )
    result = evcache.NflxEVCacheCapacityModel.capacity_plan(
        instance, instance.drive, RegionContext(), desires, args
    )
    assert io_params(result)["required_nodes"] == 10


@pytest.mark.parametrize(
    "profile,expected,small_expected",
    [
        ((1, 1, 1024, 4), 10, 10),
        ((None, 2, 4, 4), 20, 20),
        ((2, None, 4, 4), 10, 5),
        ((8, None, 4, 4), 20, 10),
        ((None, None, 8, 4), 20, 10),
    ],
)
def test_zero_demand_floor_uses_effective_drive_bandwidth(
    bounded_io_case,
    profile,
    expected,
    small_expected,
):
    current_throughput, candidate_throughput, current_block, candidate_block = profile
    instance, current, desires = bounded_io_case
    current.cluster_drive.throughput = current_throughput
    current.cluster_drive.block_size_kib = current_block
    instance.drive.throughput = candidate_throughput
    instance.drive.block_size_kib = candidate_block
    zero = evcache.EVCacheDiskIO(**{key: 0 for key in observation()})
    small = evcache.EVCacheDiskIO(
        **observation(read_ops_per_second=1, read_bytes_per_second=4096)
    )
    assert disk_io_requirement(instance, desires, zero, current)[0] == expected
    # Positive traffic supplies a mix; zero traffic must preserve every axis.
    assert disk_io_requirement(instance, desires, small, current)[0] == small_expected


@pytest.mark.parametrize("direction", ["read", "write"])
@pytest.mark.parametrize(
    "throughput_pair", [(4, None), (None, 4), (4, 4), (None, None)]
)
def test_zero_demand_preserves_only_supported_drive_directions(
    bounded_io_case, direction, throughput_pair
):
    instance, current, desires = bounded_io_case
    other = "write" if direction == "read" else "read"
    for drive in (
        instance.drive,
        current.cluster_instance.drive,
        current.cluster_drive,
    ):
        setattr(drive, f"{other}_io_per_s", None)
    current.cluster_drive.throughput, instance.drive.throughput = throughput_pair
    zero = {key: 0 for key in observation()}
    small = {
        **zero,
        f"{direction}_ops_per_second": 1,
        f"{direction}_bytes_per_second": 4096,
    }
    for rates in (small, zero):
        count, _ = disk_io_requirement(
            instance, desires, evcache.EVCacheDiskIO(**rates), current
        )
        assert count == 10

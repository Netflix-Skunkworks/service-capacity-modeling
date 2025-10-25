import logging
import math
from decimal import Decimal
from typing import Any
from typing import Dict
from typing import Optional

from pydantic import BaseModel
from pydantic import Field

from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import Buffer
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import Buffers
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRequirement
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import Consistency
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import GlobalConsistency
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.interface import ZoneClusterCapacity
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.common import buffer_for_components
from service_capacity_modeling.models.common import compute_stateful_zone
from service_capacity_modeling.models.common import get_effective_disk_per_node_gib
from service_capacity_modeling.models.common import normalize_cores
from service_capacity_modeling.models.common import simple_network_mbps
from service_capacity_modeling.models.common import sqrt_staffed_cores
from service_capacity_modeling.models.common import working_set_from_drive_and_slo
from service_capacity_modeling.stats import dist_for_interval

logger = logging.getLogger(__name__)


# Pebble does Leveled compaction with tieres of size??
# (FIXME) What does pebble actually do
def _crdb_io_per_read(node_size_gib: float, sstable_size_mb: int = 1000) -> int:
    gb = node_size_gib * 1024
    sstables = max(1, gb // sstable_size_mb)
    # 10 sstables per level, plus 1 for L0 (avg)
    levels = 1 + int(math.ceil(math.log(sstables, 10)))
    return levels


def _estimate_cockroachdb_requirement(  # noqa=E501 pylint: disable=too-many-positional-arguments
    instance: Instance,
    desires: CapacityDesires,
    working_set: float,
    reads_per_second: float,
    max_rps_to_disk: int,
    zones_per_region: int = 3,
    copies_per_region: int = 3,
) -> CapacityRequirement:
    """Estimate the capacity required for one zone given a regional desire

    The input desires should be the **regional** desire, and this function will
    return the zonal capacity requirement
    """
    needed_cores = normalize_cores(
        # Keep half of the cores free for background work (compaction, backup, index)
        core_count=sqrt_staffed_cores(desires) * 2,
        target_shape=instance,
        reference_shape=desires.reference_shape,
    )
    # Keep half of the bandwidth available for backup
    needed_network_mbps = simple_network_mbps(desires) * 2

    needed_disk = math.ceil(
        (1.0 / desires.data_shape.estimated_compression_ratio.mid)
        * desires.data_shape.estimated_state_size_gib.mid
        * copies_per_region
    )

    # Rough estimate of how many instances we would need just for the the CPU
    # Note that this is a lower bound, we might end up with more.
    rough_count = math.ceil(needed_cores / instance.cpu)

    # Generally speaking we want fewer than some number of reads per second
    # hitting disk per instance. If we don't have many reads we don't need to
    # hold much data in memory.
    instance_rps = max(1, reads_per_second // rough_count)
    disk_rps = instance_rps * _crdb_io_per_read(max(1, needed_disk // rough_count))
    rps_working_set = min(1.0, disk_rps / max_rps_to_disk)

    # If disk RPS will be smaller than our target because there are no
    # reads, we don't need to hold as much data in memory
    needed_memory = min(working_set, rps_working_set) * needed_disk

    # Now convert to per zone
    needed_cores = max(1, needed_cores // zones_per_region)
    needed_disk = max(1, needed_disk // zones_per_region)
    needed_memory = max(1, int(needed_memory // zones_per_region))
    logger.debug(
        "Need (cpu, mem, disk, working) = (%s, %s, %s, %f)",
        needed_cores,
        needed_memory,
        needed_disk,
        working_set,
    )

    return CapacityRequirement(
        requirement_type="crdb-zonal",
        reference_shape=desires.reference_shape,
        cpu_cores=certain_int(needed_cores),
        mem_gib=certain_float(needed_memory),
        disk_gib=certain_float(needed_disk),
        network_mbps=certain_float(needed_network_mbps),
        context={
            "working_set": min(working_set, rps_working_set),
            "rps_working_set": rps_working_set,
            "disk_slo_working_set": working_set,
            "replication_factor": copies_per_region,
            "compression_ratio": round(
                1.0 / desires.data_shape.estimated_compression_ratio.mid, 2
            ),
            "read_per_second": reads_per_second,
        },
    )


def _upsert_params(cluster: ZoneClusterCapacity, params: Dict[str, Any]) -> None:
    if cluster.cluster_params:
        cluster.cluster_params.update(params)
    else:
        cluster.cluster_params = params


# pylint: disable=too-many-locals
def _estimate_cockroachdb_cluster_zonal(  # noqa=E501 pylint: disable=too-many-positional-arguments
    instance: Instance,
    drive: Drive,
    desires: CapacityDesires,
    zones_per_region: int = 3,
    copies_per_region: int = 3,
    max_local_data_per_node_gib: int = 2048,
    max_regional_size: int = 288,
    max_rps_to_disk: int = 500,
    min_vcpu_per_instance: int = 4,
    license_fee_per_core: float = 0.0,
) -> Optional[CapacityPlan]:
    if instance.cpu < min_vcpu_per_instance:
        return None

    # Right now CRDB doesn't deploy to cloud drives, just adding this
    # here and leaving the capability to handle cloud drives for the future
    if instance.drive is None:
        return None

    rps = desires.query_pattern.estimated_read_per_second.mid // zones_per_region

    # Based on the disk latency and the read latency SLOs we adjust our
    # working set to keep more or less data in RAM. Faster drives need
    # less fronting RAM.
    ws_drive = instance.drive or drive
    working_set = working_set_from_drive_and_slo(
        drive_read_latency_dist=dist_for_interval(ws_drive.read_io_latency_ms),
        read_slo_latency_dist=dist_for_interval(
            desires.query_pattern.read_latency_slo_ms
        ),
        estimated_working_set=desires.data_shape.estimated_working_set_percent,
        # CRDB has looser latency SLOs but we still want a lot of the data
        # hot in cache. Target the 95th percentile of disk latency to
        # keep in RAM.
        target_percentile=0.95,
    ).mid

    requirement = _estimate_cockroachdb_requirement(
        instance=instance,
        desires=desires,
        working_set=working_set,
        reads_per_second=rps,
        zones_per_region=zones_per_region,
        copies_per_region=copies_per_region,
        max_rps_to_disk=max_rps_to_disk,
    )

    base_mem = (
        desires.data_shape.reserved_instance_app_mem_gib
        + desires.data_shape.reserved_instance_system_mem_gib
    )

    disk_buffer_ratio = buffer_for_components(
        buffers=desires.buffers, components=[BufferComponent.disk]
    ).ratio
    max_data_per_node_gib = get_effective_disk_per_node_gib(
        instance,
        drive,
        disk_buffer_ratio,
        max_local_data_per_node_gib=max_local_data_per_node_gib,
    )
    needed_disk_gib = requirement.disk_gib.mid * disk_buffer_ratio
    min_count = math.ceil(needed_disk_gib / max_data_per_node_gib)

    cluster = compute_stateful_zone(
        instance=instance,
        drive=drive,
        needed_cores=int(requirement.cpu_cores.mid),
        needed_disk_gib=needed_disk_gib,
        needed_memory_gib=requirement.mem_gib.mid,
        needed_network_mbps=requirement.network_mbps.mid,
        # Take into account the reads per read
        # from the per node dataset using leveled compaction
        # FIXME: I feel like this can be improved
        required_disk_ios=lambda size, count: (
            _crdb_io_per_read(size) * math.ceil(rps / count),
            # TODO: presumably there are some write IOs here
            0,
        ),
        # cockroachdb clusters will autobalance across available nodes
        cluster_size=lambda x: x,
        min_count=min_count,
        # Sidecars/System takes away memory from cockroachdb
        # cockroachdb by default uses --max-sql-memory of 25% of system memory
        # that cannot be used for caching
        reserve_memory=lambda x: base_mem + 0.25 * x,
        # TODO: Figure out how much memory CRDB needs to buffer writes
        # in memtables in order to only flush occasionally
        # write_buffer=...
        # required_write_buffer_gib=...
    )

    # Communicate to the actual provision that if we want reduced RF
    params = {"cockroachdb.copies": copies_per_region}
    _upsert_params(cluster, params)

    # cockroachdb clusters generally should try to stay under some total number
    # of nodes. Orgs do this for all kinds of reasons such as
    #   * Security group limits. Since you must have < 500 rules if you're
    #       ingressing public ips)
    #   * Maintenance. If your restart script does one node at a time you want
    #       smaller clusters so your restarts don't take months.
    #   * NxN network issues. Sometimes smaller clusters of bigger nodes
    #       are better for network propagation
    if cluster.count > (max_regional_size // zones_per_region):
        return None

    ec2_cost = zones_per_region * cluster.annual_cost
    license_fee = zones_per_region * (cluster.instance.cpu * license_fee_per_core)

    cluster.cluster_type = "cockroachdb"
    clusters = Clusters(
        annual_costs={"cockroachdb-zonal": Decimal(ec2_cost + license_fee)},
        zonal=[cluster] * zones_per_region,
        regional=[],
    )

    return CapacityPlan(
        requirements=Requirements(zonal=[requirement] * zones_per_region),
        candidate_clusters=clusters,
    )


class NflxCockroachDBArguments(BaseModel):
    copies_per_region: int = Field(
        default=3,
        description="How many copies of the data will exist e.g. RF=3. If unsupplied"
        " this will be deduced from durability and consistency desires",
    )
    max_regional_size: int = Field(
        default=288,
        description="What is the maximum size of a cluster in this region",
    )
    max_local_disk_gib: int = Field(
        default=2048,
        description="The maximum amount of data we store per machine",
    )
    max_rps_to_disk: int = Field(
        default=500,
        description="How many disk IOs should be allowed to hit disk per instance",
    )


class NflxCockroachDBCapacityModel(CapacityModel):
    @staticmethod
    def default_buffers() -> Buffers:
        return Buffers(
            default=Buffer(ratio=1.2),
        )

    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        # (FIXME): Need crdb input
        # TODO: Use read requirements to compute RF.
        copies_per_region: int = extra_model_arguments.get("copies_per_region", 3)
        max_regional_size: int = extra_model_arguments.get("max_regional_size", 500)
        max_rps_to_disk: int = extra_model_arguments.get("max_rps_to_disk", 500)
        # Very large nodes are hard to recover
        max_local_data_per_node_gib: int = extra_model_arguments.get(
            "max_local_data_per_node_gib",
            extra_model_arguments.get("max_local_disk_gib", 2048),
        )

        # Cockroach Labs recommends a minimum of 8 vCPUs and strongly
        # recommends no fewer than 4 vCPUs per node.
        min_vcpu_per_instance: int = extra_model_arguments.get(
            "min_vcpu_per_instance", 4
        )
        license_fee_per_core: float = context.services[
            "crdb_core_license"
        ].annual_cost_per_core

        return _estimate_cockroachdb_cluster_zonal(
            instance=instance,
            drive=drive,
            desires=desires,
            zones_per_region=context.zones_in_region,
            copies_per_region=copies_per_region,
            max_regional_size=max_regional_size,
            max_local_data_per_node_gib=max_local_data_per_node_gib,
            max_rps_to_disk=max_rps_to_disk,
            min_vcpu_per_instance=min_vcpu_per_instance,
            license_fee_per_core=license_fee_per_core,
        )

    @staticmethod
    def description() -> str:
        return "Netflix Streaming CockroachDB Model"

    @staticmethod
    def extra_model_arguments_schema() -> Dict[str, Any]:
        return NflxCockroachDBArguments.model_json_schema()

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> CapacityDesires:
        acceptable_consistency = {
            None,
            AccessConsistency.linearizable,
            AccessConsistency.linearizable_stale,
            AccessConsistency.serializable,
            AccessConsistency.serializable_stale,
            AccessConsistency.never,
        }
        for key, value in user_desires.query_pattern.access_consistency:
            if value.target_consistency not in acceptable_consistency:
                raise ValueError(
                    f"CockroachDB can only provide {acceptable_consistency} access."
                    f"User asked for {key}={value}"
                )

        buffers = NflxCockroachDBCapacityModel.default_buffers()
        if user_desires.query_pattern.access_pattern == AccessPattern.latency:
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.latency,
                    access_consistency=GlobalConsistency(
                        same_region=Consistency(
                            target_consistency=AccessConsistency.serializable,
                        ),
                        cross_region=Consistency(
                            target_consistency=AccessConsistency.serializable,
                        ),
                    ),
                    estimated_mean_read_size_bytes=Interval(
                        low=128, mid=1024, high=65536, confidence=0.95
                    ),
                    estimated_mean_write_size_bytes=Interval(
                        low=128, mid=1024, high=65536, confidence=0.95
                    ),
                    # (FIXME): Need crdb input
                    # CockroachDB reads and writes can take CPU time as
                    # JOINs and such can be hard to predict.
                    estimated_mean_read_latency_ms=Interval(
                        low=1, mid=2, high=100, confidence=0.98
                    ),
                    # Writes typically involve transactions which can be
                    # expensive, but it's rare for it to have a huge tail
                    estimated_mean_write_latency_ms=Interval(
                        low=1, mid=4, high=200, confidence=0.98
                    ),
                    # Assume point queries "Single digit millisecond SLO"
                    read_latency_slo_ms=FixedInterval(
                        minimum_value=1,
                        maximum_value=100,
                        low=1,
                        mid=10,
                        high=20,
                        confidence=0.98,
                    ),
                    write_latency_slo_ms=FixedInterval(
                        minimum_value=1,
                        maximum_value=100,
                        low=1,
                        mid=10,
                        high=20,
                        confidence=0.98,
                    ),
                ),
                # Most latency sensitive cockroachdb clusters are in the
                # < 100GiB range
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=10, mid=20, high=100, confidence=0.98
                    ),
                    # Pebble compresses with Snappy by default
                    estimated_compression_ratio=Interval(
                        minimum_value=1.1,
                        maximum_value=8,
                        low=1.5,
                        mid=2.4,
                        high=4,
                        confidence=0.98,
                    ),
                    # CRDB doesn't have a sidecar, but it does have data
                    # gateway taking about 1 MiB of memory
                    reserved_instance_app_mem_gib=0.001,
                ),
                buffers=buffers,
            )
        else:
            return CapacityDesires(
                # (FIXME): Need to pair with crdb folks on the exact values
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.throughput,
                    access_consistency=GlobalConsistency(
                        same_region=Consistency(
                            target_consistency=AccessConsistency.serializable,
                        ),
                        cross_region=Consistency(
                            target_consistency=AccessConsistency.serializable,
                        ),
                    ),
                    estimated_mean_read_size_bytes=Interval(
                        low=128, mid=4096, high=65536, confidence=0.95
                    ),
                    estimated_mean_write_size_bytes=Interval(
                        low=128, mid=4096, high=65536, confidence=0.95
                    ),
                    # (FIXME): Need crdb input
                    # CockroachDB analytics reads probably take extra time
                    # as they are full table scanning or doing complex JOINs
                    estimated_mean_read_latency_ms=Interval(
                        low=1, mid=20, high=100, confidence=0.98
                    ),
                    # Throughput writes typically involve large transactions
                    # which can be expensive, but it's rare for it to have a
                    # huge tail
                    estimated_mean_write_latency_ms=Interval(
                        low=1, mid=20, high=100, confidence=0.98
                    ),
                    # Assume scan queries "Tens of millisecond SLO"
                    read_latency_slo_ms=FixedInterval(
                        minimum_value=1,
                        maximum_value=100,
                        low=10,
                        mid=20,
                        high=99,
                        confidence=0.98,
                    ),
                    write_latency_slo_ms=FixedInterval(
                        minimum_value=1,
                        maximum_value=100,
                        low=10,
                        mid=20,
                        high=99,
                        confidence=0.98,
                    ),
                ),
                # Most throughput sensitive cockroachdb clusters are in the
                # < 100GiB range
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=10, mid=20, high=100, confidence=0.98
                    ),
                    # Pebble compresses with Snappy by default
                    estimated_compression_ratio=Interval(
                        minimum_value=1.1,
                        maximum_value=8,
                        low=1.5,
                        mid=2.4,
                        high=4,
                        confidence=0.98,
                    ),
                    # CRDB doesn't have a sidecar, but it does have data
                    # gateway taking about 1 MiB of memory
                    reserved_instance_app_mem_gib=0.001,
                ),
                buffers=buffers,
            )


nflx_cockroachdb_capacity_model = NflxCockroachDBCapacityModel()

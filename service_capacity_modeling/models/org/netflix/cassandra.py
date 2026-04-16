# pylint: disable=too-many-lines
import logging
import math
from typing import Any
from typing import Callable
from typing import Dict
from typing import FrozenSet
from typing import List
from typing import Optional
from typing import Tuple
from typing import Set
from typing import Union

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import Buffer
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import Buffers
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRegretParameters
from service_capacity_modeling.interface import CapacityRequirement
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import Consistency
from service_capacity_modeling.interface import CurrentClusterCapacity
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.explainability import STATEFUL_DATASTORE_FAMILIES
from service_capacity_modeling.interface import Excuse
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import GlobalConsistency
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import normalized_aws_size
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.interface import ServiceCapacity
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models import CostAwareModel
from service_capacity_modeling.models import RANK_PENALTIES
from service_capacity_modeling.models.common import COUNT_BOTTLENECK
from service_capacity_modeling.models.common import REQUIRED_NODES_BY_TYPE
from service_capacity_modeling.models.common import buffer_for_components
from service_capacity_modeling.models.common import compute_stateful_zone
from service_capacity_modeling.models.common import DerivedBuffers
from service_capacity_modeling.models.common import EFFECTIVE_DISK_PER_NODE_GIB
from service_capacity_modeling.models.common import get_effective_disk_per_node_gib
from service_capacity_modeling.models.common import network_services
from service_capacity_modeling.models.common import normalize_cores
from service_capacity_modeling.models.common import upsert_params
from service_capacity_modeling.models.common import simple_network_mbps
from service_capacity_modeling.models.common import sqrt_staffed_cores
from service_capacity_modeling.models.common import working_set_from_drive_and_slo
from service_capacity_modeling.models.common import zonal_requirements_from_current
from service_capacity_modeling.models.org.netflix.cassandra_memory import (
    _get_base_memory,
    _cass_heap,
    estimate_memory_experimental,
    estimate_memory_legacy,
)
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.models.utils import is_power_of_2
from service_capacity_modeling.models.utils import next_doubling
from service_capacity_modeling.models.utils import next_power_of_2
from service_capacity_modeling.stats import dist_for_interval

logger = logging.getLogger(__name__)

BACKGROUND_BUFFER = "background"
CRITICAL_TIERS: Set[int] = {0, 1}
# cluster size aka nodes per ASG
CRITICAL_TIER_MIN_CLUSTER_SIZE = 2

# --- CRR network and backup cost helpers ---
#
# Empirically derived from billing validation against 10 production clusters.
# Two clusters with known app-level write sizes were used to fit a linear model:
#   - cluster A: app=240B, billing=1,241B
#   - cluster B: app=6,700B (20KB / 3:1 compression), billing=10,415B
#
# Linear regression: wire_bytes = intercept + slope * app_bytes
#   slope = (10415 - 1241) / (6700 - 240) = 1.42
#   intercept = 1241 - 1.42 * 240 = 900
#
# Physical interpretation:
#   - 900B fixed: internode MessageOut framing, partition key, clustering column
#     metadata, per-mutation timestamps, CRC checksums
#   - 1.42x proportional: per-cell serialization overhead (column name, flags,
#     timestamp, TTL per cell ~ 42% of raw value bytes)
#
# Validated against 5 additional clusters using default write sizes (256B):
#   wire = 900 + 1.42*256 = 1,264B vs billing median 1,136-1,467B (within 15%)
_CASSANDRA_MUTATION_FIXED_OVERHEAD_BYTES = 900
_CASSANDRA_SERIALIZATION_AMPLIFICATION = 1.42

# Empirically derived from 8 production clusters (Feb 2026).
# For each cluster: effective_retention = (actual_backup_gib - state_gib) / daily_write_gib
# Across 6 LCS clusters (excluding TTL-heavy outliers): median = 14.4 days
#
# This reflects the SSTable lifecycle under LCS:
#   1. SSTable created on memtable flush -> backed up to S3
#   2. Compacted into new SSTable -> new SSTable backed up
#   3. Old SSTable deleted from live cluster
#   4. Retained in backup for ~7 additional days post-deletion
#
# Configurable via backup_retention_days in extra_model_arguments for clusters
# with different compaction strategies (TWCS) or aggressive TTLs.
_DEFAULT_BACKUP_RETENTION_DAYS = 14.0

# Known Cassandra write size defaults (from default_desires() and interface.py).
# Used to detect whether estimated_mean_write_size_bytes was user-supplied or
# model-defaulted. Interval has frozen=True (hashable), so set membership works.
_KNOWN_WRITE_SIZE_DEFAULTS = {
    Interval(low=64, mid=256, high=1024, confidence=0.95),  # latency pattern
    Interval(low=128, mid=1024, high=65536, confidence=0.95),  # throughput pattern
    Interval(low=512, mid=512, high=512, confidence=1.0),  # interface fallback
}


def _cassandra_wire_write_size(app_write_size_bytes: float) -> int:
    """Convert application-level write size to on-wire bytes for network cost.

    Cassandra mutations carry fixed protocol overhead (internode framing,
    partition metadata) plus proportional serialization overhead (per-cell
    metadata). See module-level constants for derivation.
    """
    return int(
        _CASSANDRA_MUTATION_FIXED_OVERHEAD_BYTES
        + _CASSANDRA_SERIALIZATION_AMPLIFICATION * app_write_size_bytes
    )


def _is_write_size_defaulted(desires: CapacityDesires) -> bool:
    """Detect whether estimated_mean_write_size_bytes was user-supplied or defaulted.

    After merge_with(), user-supplied values override model defaults via
    ExcludeUnsetModel.model_dump(exclude_unset=True). We detect defaults by
    exact equality against the known default Intervals (all 4 params: low, mid,
    high, confidence).
    """
    return (
        desires.query_pattern.estimated_mean_write_size_bytes
        in _KNOWN_WRITE_SIZE_DEFAULTS
    )


def _write_buffer_gib_zone(
    desires: CapacityDesires, zones_per_region: int, flushes_before_compaction: int = 4
) -> float:
    # Cassandra has to buffer writes before flushing to disk, and assuming
    # we will compact every 4 flushes and we want no more than 2 redundant
    # compactions in an hour, we want <= 4**2 = 16 flushes per hour
    # or a flush of data every 3600 / 16 = 225 seconds
    write_bytes_per_second = (
        desires.query_pattern.estimated_write_per_second.mid
        * desires.query_pattern.estimated_mean_write_size_bytes.mid
    )

    compactions_per_hour = 2
    hour_in_seconds = 60 * 60

    write_buffer_gib = (
        (write_bytes_per_second * hour_in_seconds)
        / (flushes_before_compaction**compactions_per_hour)
    ) / (1 << 30)

    return float(write_buffer_gib) / zones_per_region


def _get_cores_from_desires(desires: CapacityDesires, instance: Instance) -> int:
    cpu_buffer = buffer_for_components(
        buffers=desires.buffers, components=[BACKGROUND_BUFFER]
    )

    # We have no existing utilization to go from
    reference_shape = desires.reference_shape
    # Keep half of the cores free for background work (compaction, backup, repair).
    needed_cores = math.ceil(sqrt_staffed_cores(desires) * cpu_buffer.ratio)

    needed_cores = normalize_cores(
        core_count=needed_cores,
        target_shape=instance,
        reference_shape=reference_shape,
    )
    return needed_cores


def _get_disk_from_desires(desires: CapacityDesires, copies_per_region: int) -> int:
    disk_buffer = buffer_for_components(
        buffers=desires.buffers, components=[BufferComponent.disk]
    )
    # Do not add disk buffers now as memory calculation is done on the disk usage
    return math.ceil(
        (1.0 / desires.data_shape.estimated_compression_ratio.mid)
        * desires.data_shape.estimated_state_size_gib.mid
        * copies_per_region
        * disk_buffer.ratio
    )


def _get_min_count(
    tier: int,
    required_cluster_size: Optional[int],
    needed_disk_gib: float,
    disk_per_node_gib: float,
    cluster_size_lambda: Callable[[int], int],
) -> int:
    """
    Compute the minimum number of nodes required for a zone.

    This function is used to prevent the planner from allocating clusters that
    would exceed the max data per node or under the required cluster size for
    a tier or existing cluster
    """

    # Cassandra clusters should aim to be at least 2 nodes per zone to start
    # out with for tier 0 or tier 1. This gives us more room to "up-color"]
    # clusters.
    min_nodes_for_tier = 2 if tier in CRITICAL_TIERS else 0

    # Prevent allocating clusters that exceed the max data per node.
    min_nodes_for_disk = math.ceil(needed_disk_gib / disk_per_node_gib)

    # Take the max of the following in order to avoid:
    # (1) if `required_cluster_size` < `min_nodes_for_disk`, don't let the planner
    #     pick a shape that would exceed the max data per node
    #
    #     For example, if we need 4TiB of disk, and the max data per node is 1TiB,
    #     Regardless of the `required_cluster_size`, we cannot allocate less than 4
    #     nodes because that would exceed the max data per node.
    #
    # (2) if `required_cluster_size` > `min_nodes_for_disk`, don't let the
    #     node density requirement affect the min count because the required
    #     cluster size already meets the node density requirement.
    #
    #     For example, if we need 4TiB of disk, and the max data per node is 1TiB,
    #     and the upstream requires >= 8 nodes, we can allocate 8 nodes because
    #     each node would only have 500GB of data.
    min_count = max(
        min_nodes_for_tier,
        required_cluster_size or 0,
        min_nodes_for_disk,
    )
    # Ensure that the min count is an increment of the cluster size constraint (doubling)
    return cluster_size_lambda(min_count)


def _zonal_requirement_for_new_cluster(
    desires: CapacityDesires,
    instance: Instance,
    copies_per_region: int,
    zones_per_region: int,
) -> CapacityRequirement:
    needed_cores = _get_cores_from_desires(desires, instance)
    needed_disk = _get_disk_from_desires(desires, copies_per_region)

    # Keep some of the bandwidth available for backup and repair streaming
    network_buffer = buffer_for_components(
        buffers=desires.buffers, components=[BACKGROUND_BUFFER]
    )
    needed_network_mbps = simple_network_mbps(desires) * network_buffer.ratio

    # Now convert to per zone
    needed_cores = max(1, needed_cores // zones_per_region)
    needed_disk = max(1, needed_disk // zones_per_region)

    return CapacityRequirement(
        requirement_type=NflxCassandraCapacityModel.cluster_type,
        cpu_cores=certain_int(needed_cores),
        disk_gib=certain_float(needed_disk),
        network_mbps=certain_float(needed_network_mbps),
    )


# pylint: disable=too-many-locals
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-statements
def _estimate_cassandra_requirement(
    instance: Instance,
    desires: CapacityDesires,
    working_set: float,
    reads_per_second: float,
    max_rps_to_disk: int,
    zones_per_region: int = 3,
    copies_per_region: int = 3,
    experimental_memory_model: bool = False,
    max_page_cache_gib: float = 32.0,
) -> CapacityRequirement:
    # Input: regional desires → Output: zonal requirement
    disk_buffer = buffer_for_components(
        buffers=desires.buffers, components=[BufferComponent.disk]
    )
    reference_shape = desires.reference_shape
    current_capacity = _get_current_capacity(desires)

    # If the cluster is already provisioned
    if current_capacity and desires.current_clusters is not None:
        capacity_requirement = zonal_requirements_from_current(
            desires.current_clusters,
            desires.buffers,
            instance,
            reference_shape,
        )
        disk_derived_buffer = DerivedBuffers.for_components(
            desires.buffers.derived, [BufferComponent.disk]
        )
        disk_used_gib = (
            current_capacity.disk_utilization_gib.mid
            * current_capacity.cluster_instance_count.mid
            * disk_derived_buffer.scale
        )
    else:
        # If the cluster is not yet provisioned
        capacity_requirement = _zonal_requirement_for_new_cluster(
            desires, instance, copies_per_region, zones_per_region
        )
        disk_used_gib = capacity_requirement.disk_gib.mid / disk_buffer.ratio

    needed_cores = math.ceil(capacity_requirement.cpu_cores.mid)
    needed_disk = capacity_requirement.disk_gib.mid
    needed_network_mbps = capacity_requirement.network_mbps.mid

    # it can be 0 for cases where disk utilization is not passed as a part of current cluster capacity
    if needed_disk == 0:
        needed_disk = max(
            1, _get_disk_from_desires(desires, copies_per_region) // zones_per_region
        )
        # We want to compute memory based on the data size.
        disk_used_gib = needed_disk / disk_buffer.ratio

    # Rough estimate of how many instances we would need just for the CPU
    # Note that this is a lower bound, we might end up with more.
    estimated_cores_per_region = math.ceil(
        (needed_cores * zones_per_region) / instance.cpu
    )
    # Generally speaking we want fewer than some number of reads per second
    # hitting disk per instance. If we don't have many reads we don't need to
    # hold much data in memory.
    instance_rps = max(1, reads_per_second // estimated_cores_per_region)
    disk_rps = instance_rps * _cass_io_per_read(
        max(1, (disk_used_gib * zones_per_region) // estimated_cores_per_region)
    )
    rps_working_set = min(1.0, disk_rps / max_rps_to_disk)

    # Cassandra can defer writes either by buffering in memory or by
    # waiting longer before recompacting (the min-threshold on the
    # L0 compactions or STCS compactions)
    min_threshold = 4
    write_buffer_gib = _write_buffer_gib_zone(
        desires=desires,
        zones_per_region=zones_per_region,
        flushes_before_compaction=min_threshold,
    )

    while write_buffer_gib > 12 and min_threshold < 16:
        min_threshold *= 2
        write_buffer_gib = _write_buffer_gib_zone(
            desires=desires,
            zones_per_region=zones_per_region,
            flushes_before_compaction=min_threshold,
        )

    # Compute effective working set and memory requirement.
    if experimental_memory_model:
        mem = estimate_memory_experimental(
            current_capacity=current_capacity,
            working_set=working_set,
            rps_working_set=rps_working_set,
            disk_used_gib=disk_used_gib,
            desires=desires,
            write_buffer_gib=write_buffer_gib,
            max_page_cache_gib=max_page_cache_gib,
        )
    else:
        mem = estimate_memory_legacy(
            working_set=working_set,
            rps_working_set=rps_working_set,
            disk_used_gib=disk_used_gib,
            zones_per_region=zones_per_region,
            write_buffer_gib=write_buffer_gib,
        )
    effective_working_set = mem.effective_working_set
    needed_memory = mem.needed_memory_gib
    write_buffer_gib = mem.write_buffer_gib

    logger.debug(
        "Need (cpu, mem, disk, working) = (%s, %s, %s, %f)",
        needed_cores,
        needed_memory,
        needed_disk,
        working_set,
    )

    return CapacityRequirement(
        requirement_type=NflxCassandraCapacityModel.cluster_type,
        reference_shape=reference_shape,
        cpu_cores=certain_int(needed_cores),
        mem_gib=certain_float(needed_memory),
        disk_gib=certain_float(needed_disk),
        network_mbps=certain_float(needed_network_mbps),
        context={
            "working_set": effective_working_set,
            "rps_working_set": rps_working_set,
            "disk_slo_working_set": working_set,
            "replication_factor": copies_per_region,
            "compression_ratio": round(
                1.0 / desires.data_shape.estimated_compression_ratio.mid, 2
            ),
            "read_per_second": reads_per_second,
            "write_buffer_gib": write_buffer_gib,
            "min_threshold": min_threshold,
        },
    )


def _get_current_cluster_size(desires: CapacityDesires) -> int:
    current_capacity = _get_current_capacity(desires)
    if current_capacity is None:
        return 0
    return math.ceil(current_capacity.cluster_instance_count.mid)


def _get_current_capacity(desires: CapacityDesires) -> Optional[CurrentClusterCapacity]:
    if desires.current_clusters is None:
        return None
    if desires.current_clusters.zonal:
        return desires.current_clusters.zonal[0]
    if desires.current_clusters.regional:
        return desires.current_clusters.regional[0]
    return None


def _get_cluster_size_lambda(
    current_cluster_size: int,
    required_cluster_size: Optional[int],
) -> Callable[[int], int]:
    if required_cluster_size:
        return lambda x: next_doubling(x, base=required_cluster_size)
    elif current_cluster_size and not is_power_of_2(current_cluster_size):
        return lambda x: next_doubling(x, base=current_cluster_size)
    else:  # New provisionings
        return next_power_of_2


def _compute_penalties(
    instance: Instance,
    large_instance_regret: float,
    current_family: Optional[str] = None,
    different_family_regret: float = 0.10,  # Empirical; see NflxCassandraArguments
) -> Dict[str, float]:
    """Compute named penalties from regret coefficients.

    Penalties inflate the plan rank used by plan_certain() sorting:
        rank = compute_cost * (1 + sum(penalties.values())) + service_cost

    All plans get a cost-proportional rank, so penalties act as
    percentage cost adjustments rather than absolute barriers.
    """
    penalties: Dict[str, float] = {}

    # Prefer horizontal scaling: penalize instance sizes above 8xlarge
    instance_size = float(normalized_aws_size(instance.name))
    if large_instance_regret > 0 and instance_size > 8:
        penalties["large_instance"] = large_instance_regret * (instance_size - 8) / 8

    # Penalize switching to a different instance family
    if (
        different_family_regret > 0
        and current_family
        and instance.family != current_family
    ):
        penalties["family_migration"] = different_family_regret

    return penalties


# pylint: disable=too-many-locals
# pylint: disable=too-many-return-statements
# flake8: noqa: C901
def _estimate_cassandra_cluster_zonal(  # pylint: disable=too-many-positional-arguments
    instance: Instance,
    drive: Drive,
    context: RegionContext,
    desires: CapacityDesires,
    zones_per_region: int = 3,
    copies_per_region: int = 3,
    require_local_disks: bool = False,
    require_attached_disks: bool = False,
    required_cluster_size: Optional[int] = None,
    max_rps_to_disk: int = 500,
    max_local_data_per_node_gib: int = 1280,
    max_attached_data_per_node_gib: int = 2048,
    max_regional_size: int = 192,
    max_write_buffer_percent: float = 0.25,
    max_table_buffer_percent: float = 0.11,
    large_instance_regret: float = 0.2,
    different_family_regret: float = 0.10,
    experimental_memory_model: bool = False,
    max_page_cache_gib: float = 32.0,
    backup_retention_days: Optional[float] = None,
) -> Union[CapacityPlan, Excuse, None]:
    drive_name = drive.name

    # Netflix Cassandra doesn't like to deploy on really small instances
    if instance.cpu < 2 or instance.ram_gib <= 16:
        return Excuse(
            instance=instance.name,
            drive=drive_name,
            reason=(
                f"Instance too small: {instance.cpu} vCPUs "
                f"(min 2), {instance.ram_gib:.0f} GiB RAM (requires > 16 GiB)"
            ),
            context={
                "cpu": instance.cpu,
                "ram_gib": instance.ram_gib,
                "min_cpu": 2,
                "min_ram_gib_exclusive": 16,
            },
            bottleneck=Bottleneck.cpu if instance.cpu < 2 else Bottleneck.memory,
        )

    # if we're not allowed to use gp2, skip EBS only types
    if instance.drive is None and require_local_disks:
        return Excuse(
            instance=instance.name,
            drive=drive_name,
            reason=f"Requires local disks but {instance.name} is EBS-only",
            context={
                "has_local_drive": False,
                "require_local_disks": True,
            },
            bottleneck=Bottleneck.drive_type,
        )

    # if we're not allowed to use local disks, skip ephems
    if instance.drive is not None and require_attached_disks:
        return Excuse(
            instance=instance.name,
            drive=drive_name,
            reason=f"Requires attached disks but {instance.name} has local drives",
            context={
                "instance_drive": str(instance.drive),
                "require_attached_disks": True,
            },
            bottleneck=Bottleneck.drive_type,
        )

    # Cassandra deploys on gp3 only (gp2 is legacy)
    if drive.name != "gp3":
        return Excuse(
            instance=instance.name,
            drive=drive_name,
            reason=f"Unsupported drive type: {drive_name}",
            context={
                "drive_name": drive_name,
                "supported_drives": ["gp3"],
            },
            bottleneck=Bottleneck.drive_type,
        )

    rps = desires.query_pattern.estimated_read_per_second.mid // zones_per_region
    write_per_sec = (
        desires.query_pattern.estimated_write_per_second.mid // zones_per_region
    )
    write_bytes_per_sec = round(
        write_per_sec * desires.query_pattern.estimated_mean_write_size_bytes.mid
    )
    read_bytes_per_sec = rps * desires.query_pattern.estimated_mean_read_size_bytes.mid
    # Write IO will be 1 to commitlog + 2 writes (plus 2 reads) in the first
    # hour during compaction.
    # Writes are sequential
    # Reads are random
    write_io_per_sec = (1 + 4) * max(
        1, write_bytes_per_sec // (drive.seq_io_size_kib * 1024)
    )
    read_io_per_sec = max(rps, read_bytes_per_sec // (drive.rand_io_size_kib * 1024))

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
        # This is about right for a database, a cache probably would want
        # to increase this even more.
        target_percentile=0.95,
    ).mid

    requirement = _estimate_cassandra_requirement(
        instance=instance,
        desires=desires,
        working_set=working_set,
        reads_per_second=rps,
        max_rps_to_disk=max_rps_to_disk,
        zones_per_region=zones_per_region,
        copies_per_region=copies_per_region,
        experimental_memory_model=experimental_memory_model,
        max_page_cache_gib=max_page_cache_gib,
    )

    # Adjust the min count to adjust to prevent too much data on a single
    needed_disk_gib = int(requirement.disk_gib.mid)
    disk_buffer_ratio = buffer_for_components(
        buffers=desires.buffers, components=[BufferComponent.disk]
    ).ratio

    # For existing EBS clusters, raise disk caps to at least the observed
    # values so _get_min_count and compute_stateful_zone don't reject the
    # current topology or inflate node count.
    current_capacity = _get_current_capacity(desires)
    is_ebs = instance.drive is None
    is_existing = (
        current_capacity is not None
        and current_capacity.disk_utilization_gib is not None
        and current_capacity.disk_utilization_gib.mid > 0
    )
    ebs_disk_floor = 0
    if experimental_memory_model and is_ebs and is_existing:
        assert current_capacity is not None
        assert current_capacity.disk_utilization_gib is not None
        observed_disk_per_node = int(current_capacity.disk_utilization_gib.mid)
        max_attached_data_per_node_gib = max(
            max_attached_data_per_node_gib, observed_disk_per_node
        )
        ebs_disk_floor = int(observed_disk_per_node * disk_buffer_ratio)

    disk_per_node_gib = get_effective_disk_per_node_gib(
        instance,
        drive,
        disk_buffer_ratio,
        max_local_data_per_node_gib=max_local_data_per_node_gib,
        max_attached_data_per_node_gib=max_attached_data_per_node_gib,
    )

    current_cluster_size = _get_current_cluster_size(desires)
    cluster_size_lambda = _get_cluster_size_lambda(
        current_cluster_size, required_cluster_size
    )
    min_count = _get_min_count(
        tier=desires.service_tier,
        required_cluster_size=required_cluster_size,
        needed_disk_gib=needed_disk_gib,
        disk_per_node_gib=disk_per_node_gib,
        cluster_size_lambda=cluster_size_lambda,
    )

    base_mem = _get_base_memory(desires)

    heap_fn = _cass_heap_for_write_buffer(
        instance=instance,
        max_zonal_size=max_regional_size // zones_per_region,
        write_buffer_gib=requirement.context["write_buffer_gib"],
        buffer_percent=(max_write_buffer_percent * max_table_buffer_percent),
    )

    def max_node_disk(d: Drive) -> int:
        return max(math.ceil(d.max_size_gib / 3), ebs_disk_floor)

    # Apply memory-only derived buffers to the write-buffer requirement so
    # scale_down caps both page cache and memtable space at current allocation.
    raw_write_buffer_gib = float(requirement.context["write_buffer_gib"])
    if raw_write_buffer_gib > 0 and current_capacity:
        try:
            current_instance = current_capacity.cluster_instance or shapes.instance(
                current_capacity.cluster_instance_name
            )
        except KeyError:
            current_instance = None
        if current_instance is not None:
            # Per-node write buffer = heap × max_write_buffer_percent × 0.25,
            # matching the write_buffer lambda passed to compute_stateful_zone.
            existing_write_buffer = (
                current_capacity.cluster_instance_count.mid
                * _cass_heap(current_instance.ram_gib)
                * max_write_buffer_percent
                * 0.25
            )
            memory_derived = DerivedBuffers.for_components(
                desires.buffers.derived,
                [BufferComponent.memory],
                component_fallbacks={},
            )
            raw_write_buffer_gib = memory_derived.calculate_requirement(
                current_usage=raw_write_buffer_gib,
                existing_capacity=existing_write_buffer,
            )

    cluster = compute_stateful_zone(
        instance=instance,
        drive=drive,
        needed_cores=int(requirement.cpu_cores.mid),
        needed_disk_gib=needed_disk_gib,
        needed_memory_gib=int(requirement.mem_gib.mid),
        needed_network_mbps=requirement.network_mbps.mid,
        # Take into account the reads per read
        # from the per node dataset using leveled compaction
        required_disk_ios=lambda size, count: (
            _cass_io_per_read(size) * math.ceil(read_io_per_sec / count),
            write_io_per_sec / count,
        ),
        # C* clusters provision in powers of 2 because doubling
        cluster_size=cluster_size_lambda,
        min_count=min_count,
        # TODO: Take reserve memory calculation into account during buffer calculation
        # C* heap usage takes away from OS page cache memory
        reserve_memory=lambda x: base_mem + heap_fn(x),
        # C* heap buffers the writes at roughly a rate of
        # memtable_cleanup_threshold * memtable_size. At Netflix this
        # is 0.11 * 25 * heap
        write_buffer=lambda x: heap_fn(x) * max_write_buffer_percent * 0.25,
        required_write_buffer_gib=raw_write_buffer_gib,
        max_node_disk_gib=max_node_disk,
        include_node_count_breakdown=True,
    )

    # Communicate to the actual provision that if we want reduced RF
    params = {
        "cassandra.keyspace.rf": copies_per_region,
        # In order to handle high write loads we have to shift memory
        # to heap memory, communicate with C* about this
        "cassandra.heap.gib": heap_fn(instance.ram_gib),
        "cassandra.heap.write.percent": max_write_buffer_percent,
        "cassandra.heap.table.percent": max_table_buffer_percent,
        "cassandra.compaction.min_threshold": requirement.context["min_threshold"],
        EFFECTIVE_DISK_PER_NODE_GIB: disk_per_node_gib,
        "cassandra.storage_buffer_ratio": round(disk_buffer_ratio, 2),
        "cassandra.compute_buffer_ratio": round(
            getattr(desires.buffers.desired.get("compute"), "ratio", 1.5),
            2,
        ),
    }
    upsert_params(cluster, params)

    # All penalties inflate plan.rank = cost * (1 + sum(penalties)),
    # which controls plan_certain() sort order. Penalties are also stored
    # in cluster_params[RANK_PENALTIES] for the regret() override.
    current_capacity = _get_current_capacity(desires)
    current_family = (
        current_capacity.cluster_instance.family
        if current_capacity and current_capacity.cluster_instance
        else None
    )
    penalties = _compute_penalties(
        instance=instance,
        large_instance_regret=large_instance_regret,
        current_family=current_family,
        different_family_regret=different_family_regret,
    )
    if penalties:
        upsert_params(cluster, {RANK_PENALTIES: penalties})

    # Sometimes we don't want modify cluster topology, so only allow
    # topologies that match the desired zone size
    if required_cluster_size is not None and cluster.count != required_cluster_size:
        required_nodes_by_type = cluster.cluster_params.get(REQUIRED_NODES_BY_TYPE, {})
        count_bottleneck = cluster.cluster_params.get(COUNT_BOTTLENECK, "unknown")
        return Excuse(
            instance=instance.name,
            drive=drive_name,
            reason=(
                f"Cluster size {cluster.count} "
                f"(count bottleneck: {count_bottleneck}) "
                f"!= required {required_cluster_size}"
            ),
            context={
                "computed_count": cluster.count,
                "required_cluster_size": required_cluster_size,
                "required_nodes_by_type": required_nodes_by_type,
                "count_bottleneck": count_bottleneck,
            },
            bottleneck=Bottleneck.cluster_size,
        )

    # Cassandra clusters generally should try to stay under some total number
    # of nodes. Orgs do this for all kinds of reasons such as
    #   * Security group limits. Since you must have < 500 rules if you're
    #       ingressing public ips)
    #   * Maintenance. If your restart script does one node at a time you want
    #       smaller clusters so your restarts don't take months.
    #   * Schema propagation. Since C* must gossip out changes to schema the
    #       duration of this can increase a lot with > 500 node clusters.
    max_zonal = max_regional_size // zones_per_region
    if cluster.count > max_zonal:
        return Excuse(
            instance=instance.name,
            drive=drive_name,
            reason=(
                f"Cluster too large: {cluster.count} nodes > max {max_zonal} per zone"
            ),
            context={
                "zonal_count": cluster.count,
                "max_zonal": max_zonal,
                "needed_disk_gib": needed_disk_gib,
                "disk_per_node_gib": disk_per_node_gib,
            },
            bottleneck=Bottleneck.disk_capacity,
        )

    # Calculate service costs (network + backup)
    cap_services = NflxCassandraCapacityModel.service_costs(
        service_type=NflxCassandraCapacityModel.service_name,
        context=context,
        desires=desires,
        extra_model_arguments={
            "copies_per_region": copies_per_region,
            "backup_retention_days": backup_retention_days,
        },
    )

    cluster.cluster_type = NflxCassandraCapacityModel.cluster_type
    zonal_clusters = [cluster] * zones_per_region

    # Account for the clusters, backup, and network costs
    cluster_costs = NflxCassandraCapacityModel.cluster_costs(
        service_type=NflxCassandraCapacityModel.service_name,
        zonal_clusters=zonal_clusters,
    )
    # annual_costs combines cluster infra + services for the Clusters object
    annual_costs = {**cluster_costs}
    annual_costs.update({s.service_type: s.annual_cost for s in cap_services})

    clusters = Clusters(
        annual_costs=annual_costs,
        zonal=zonal_clusters,
        regional=[],
        services=cap_services,
    )

    # Apply penalties only to compute (cluster infrastructure) costs, not
    # service costs (network, backup). Service costs are fixed regardless of
    # instance choice, so penalizing them inflates the effective threshold —
    # e.g. a 10% penalty on $700K total when services are $600K acts as a
    # 47% penalty on the $100K compute component.
    total_penalty = sum(penalties.values())
    compute_cost = float(sum(cluster_costs.values()))
    service_cost = float(sum(s.annual_cost for s in cap_services))
    plan_rank = compute_cost * (1 + total_penalty) + service_cost

    return CapacityPlan(
        requirements=Requirements(zonal=[requirement] * zones_per_region),
        candidate_clusters=clusters,
        rank=plan_rank,
    )


# C* LCS has 160 MiB sstables by default and 10 sstables per level
def _cass_io_per_read(node_size_gib: float, sstable_size_mb: int = 160) -> int:
    gb = node_size_gib * 1024
    sstables = max(1, gb // sstable_size_mb)
    # 10 sstables per level, plus 1 for L0 (avg)
    levels = 1 + int(math.ceil(math.log(sstables, 10)))
    # One disk IO per data read and one per index read (assume we miss
    # the key cache)
    return 2 * levels


def _cass_heap_for_write_buffer(
    instance: Instance,
    write_buffer_gib: float,
    max_zonal_size: int,
    buffer_percent: float,
) -> Callable[[float], float]:
    # If there is no way we can get enough heap with the max zonal size, try
    # letting max heap grow to 31 GiB per node to get more write buffer
    if write_buffer_gib > (
        max_zonal_size * _cass_heap(instance.ram_gib) * buffer_percent
    ):
        return lambda x: _cass_heap(x, max_heap_gib=30)
    else:
        return _cass_heap


def _target_rf(desires: CapacityDesires, user_copies: Optional[int]) -> int:
    if user_copies is not None:
        assert user_copies > 1
        return user_copies

    # Due to the relaxed durability and consistency requirements we can
    # run with RF=2
    consistency = desires.query_pattern.access_consistency.same_region
    if (
        desires.data_shape.durability_slo_order.mid < 1000
        and consistency is not None
        and consistency.target_consistency != AccessConsistency.read_your_writes
    ):
        return 2
    return 3


def _adaptive_storage_buffer_ratio(
    zonal_data_gib: float,
    max_ratio: float = 4.0,
    min_ratio: float = 2.0,
    midpoint_gib: float = 10_000,
    steepness: float = 0.8,
) -> float:
    """Logistic decay from max_ratio to min_ratio as data size grows.

    Large clusters have enormous absolute headroom even at lower ratios,
    so a fixed 4x buffer over-provisions them and rejects cheaper instance
    types that physically fit the data.
    """
    if zonal_data_gib <= 0:
        return max_ratio
    x = math.log(zonal_data_gib / midpoint_gib) * steepness
    t = 1.0 / (1.0 + math.exp(-x))
    return max_ratio - t * (max_ratio - min_ratio)


def _estimate_zonal_data_gib(user_desires: CapacityDesires, rf: int) -> float:
    """Estimate compressed on-disk bytes for one zone.

    Prefers actual current_clusters data; falls back to estimated_state_size_gib
    divided by compression ratio (matching the on-disk units of the first path).
    """
    if (
        user_desires.current_clusters is not None
        and user_desires.current_clusters.zonal
    ):
        cc = user_desires.current_clusters.zonal[0]
        if (
            cc.disk_utilization_gib is not None
            and cc.disk_utilization_gib.mid > 0
            and cc.cluster_instance_count is not None
        ):
            return cc.disk_utilization_gib.mid * cc.cluster_instance_count.mid

    if user_desires.data_shape is not None:
        state = user_desires.data_shape.estimated_state_size_gib
        if state.mid > 0:
            # Use state size as-is with the user's compression ratio.
            # estimated_compression_ratio defaults to 1.0 (no compression)
            # when unset, so this path naturally operates in logical GiB.
            # The adaptive buffer midpoint is calibrated for this.
            cr = user_desires.data_shape.estimated_compression_ratio.mid
            return state.mid / max(cr, 1.0) * rf / 3

    return 0.0


def _adaptive_compute_buffer_ratio(
    write_weighted_throughput_mbps: float,
    max_ratio: float = 1.5,
    min_ratio: float = 1.3,
    midpoint_mbps: float = 100.0,
    steepness: float = 0.8,
) -> float:
    """Logistic decay from max_ratio to min_ratio as write-weighted throughput grows.

    At high throughput, sqrt staffing provides natural statistical multiplexing
    headroom. Uses write-weighted throughput (read MB/s + 3x write MB/s) to
    account for both operation size and write CPU overhead (memtable flushes,
    compaction pressure scale with write size, not just write count).
    """
    if write_weighted_throughput_mbps <= 0:
        return max_ratio
    x = math.log(write_weighted_throughput_mbps / midpoint_mbps) * steepness
    t = 1.0 / (1.0 + math.exp(-x))
    return max_ratio - t * (max_ratio - min_ratio)


class NflxCassandraArguments(BaseModel):
    """Configuration arguments for the Netflix Cassandra capacity model.

    This model centralizes all tunable parameters with their defaults.
    Use `from_extra_model_arguments()` to parse a dict into a validated instance.
    """

    copies_per_region: Optional[int] = Field(
        default=None,
        description="How many copies of the data will exist e.g. RF=3. If unsupplied"
        " this will be deduced from durability and consistency desires",
    )
    require_local_disks: bool = Field(
        default=True,
        description="If local (ephemeral) drives are required",
    )
    require_attached_disks: bool = Field(
        default=False,
        description="If attached (ebs) drives are required",
    )
    required_cluster_size: Optional[int] = Field(
        default=None,
        description="Require zonal clusters to be this size (force vertical scaling)",
    )
    max_rps_to_disk: int = Field(
        default=500,
        description="How many disk IOs should be allowed to hit disk per instance",
    )
    max_regional_size: int = Field(
        default=192,
        description="What is the maximum size of a cluster in this region",
    )
    max_local_data_per_node_gib: int = Field(
        default=1280,
        description="Maximum data per node for local disk instances (GiB)",
    )
    max_attached_data_per_node_gib: int = Field(
        default=2048,
        description="Maximum data per node for attached disk instances (GiB)",
    )
    max_write_buffer_percent: float = Field(
        default=0.25,
        description="The amount of heap memory that can be used to buffer writes. "
        "Note that if there are more than 100k writes this will "
        "automatically adjust to 0.5",
    )
    max_table_buffer_percent: float = Field(
        default=0.11,
        description="How much of heap memory can be used for a single table. "
        "Note that if there are more than 100k writes this will "
        "automatically adjust to 0.2",
    )
    large_instance_regret: float = Field(
        default=0.2,
        description="Graduated cost penalty for instance sizes above 8xlarge. "
        "Adds penalty * max(0, (normalized_size - 8) / 8) * cost to the "
        "effective sort cost. Prevents AWS pricing rounding from favoring "
        "larger instances. Set to 0 to disable.",
    )
    different_family_regret: float = Field(
        default=0.10,
        description="Minimum annual savings threshold to justify switching "
        "instance families (e.g. m6id -> c6id). Reservations are "
        "family-specific, so switching means paying on-demand (~2.8x "
        "reserved) until new reservations are procured. Empirical: "
        "10% balances migration friction (on-demand premium for ~1-2 "
        "months) against locking in suboptimal families. Increase to "
        "0.15-0.20 for risk-averse clusters. Only applies when "
        "current_clusters is set. Set to 0 to disable.",
    )
    experimental_memory_model: bool = Field(
        default=False,
        description="Enable experimental memory model. When True, derives working "
        "set from page cache capped at max_page_cache_gib instead of theoretical "
        "disk/SLO estimate. When False (default), uses the legacy memory sizing.",
    )
    max_page_cache_gib: float = Field(
        default=32.0,
        description="Maximum page cache (GiB) to assume per node when computing "
        "working set in the experimental memory model. Caps the effective page "
        "cache at this value regardless of instance RAM. Set to 0 to disable "
        "the cap. Only applies when experimental_memory_model is True.",
    )
    backup_retention_days: Optional[float] = Field(
        default=None,
        description="Effective backup retention in days for write-throughput backup cost. "
        "Default 14.0 (derived from LCS production clusters). Lower for TWCS or "
        "aggressive TTL workloads where SSTables expire before retention matters.",
    )
    adaptive_storage_buffer: bool = Field(
        default=True,
        description="Use a data-size-adaptive storage buffer instead of the fixed 4x. "
        "Large clusters get a lower ratio (down to min_storage_buffer_ratio) "
        "because they already have enormous absolute headroom in GiB.",
    )
    max_storage_buffer_ratio: float = Field(
        default=4.0,
        description="Storage buffer ratio for tiny clusters (adaptive upper bound).",
    )
    min_storage_buffer_ratio: float = Field(
        default=2.0,
        description="Storage buffer ratio for very large clusters (adaptive lower bound).",
    )

    adaptive_compute_buffer: bool = Field(
        default=True,
        description="Use a traffic-adaptive compute buffer instead of fixed 1.5x. "
        "Large clusters get a lower ratio (down to min_compute_buffer_ratio) "
        "because sqrt staffing provides natural headroom at scale.",
    )
    max_compute_buffer_ratio: float = Field(
        default=1.5,
        description="Compute success buffer for tiny clusters (adaptive upper bound).",
    )
    min_compute_buffer_ratio: float = Field(
        default=1.3,
        description="Compute success buffer for very large clusters (adaptive lower bound).",
    )

    @model_validator(mode="after")
    def _check_storage_buffer_bounds(self) -> "NflxCassandraArguments":
        if self.min_storage_buffer_ratio > self.max_storage_buffer_ratio:
            raise ValueError(
                f"min_storage_buffer_ratio ({self.min_storage_buffer_ratio}) "
                f"must be <= max_storage_buffer_ratio ({self.max_storage_buffer_ratio})"
            )
        return self

    @model_validator(mode="after")
    def _check_compute_buffer_bounds(self) -> "NflxCassandraArguments":
        if self.min_compute_buffer_ratio > self.max_compute_buffer_ratio:
            raise ValueError(
                f"min_compute_buffer_ratio ({self.min_compute_buffer_ratio}) "
                f"must be <= max_compute_buffer_ratio ({self.max_compute_buffer_ratio})"
            )
        return self

    @classmethod
    def from_extra_model_arguments(
        cls, extra_model_arguments: Dict[str, Any]
    ) -> "NflxCassandraArguments":
        """Parse extra_model_arguments dict into a validated NflxCassandraArguments.

        This centralizes default values - any field not in extra_model_arguments
        will use the default defined in this model.

        Handles legacy field name mappings:
        - max_local_disk_gib -> max_local_data_per_node_gib (if not explicitly set)
        """
        # Handle legacy field name: max_local_disk_gib -> max_local_data_per_node_gib
        args = dict(extra_model_arguments)
        if "max_local_data_per_node_gib" not in args and "max_local_disk_gib" in args:
            args["max_local_data_per_node_gib"] = args["max_local_disk_gib"]

        # Pydantic will use defaults for any missing fields
        return cls.model_validate(args)


class NflxCassandraCapacityModel(CapacityModel, CostAwareModel):
    service_name = "cassandra"
    cluster_type = "cassandra"

    def __init__(self) -> None:
        pass

    @staticmethod
    def allowed_cloud_drives() -> Tuple[Optional[str], ...]:
        return ("gp3",)

    @staticmethod
    def preferred_families() -> Optional[FrozenSet[str]]:
        return STATEFUL_DATASTORE_FAMILIES

    @staticmethod
    def get_required_cluster_size(
        tier: int, extra_model_arguments: Dict[str, Any]
    ) -> Optional[int]:
        required_cluster_size: Optional[int] = (
            math.ceil(extra_model_arguments["required_cluster_size"])
            if "required_cluster_size" in extra_model_arguments
            else None
        )

        if tier not in CRITICAL_TIERS or required_cluster_size is None:
            return required_cluster_size

        # If the upstream explicitly set a cluster size, make sure it is
        # at least CRITICAL_TIER_MIN_CLUSTER_SIZE. We cannot do a max
        # of the two because the horizontal scaling is disabled
        if required_cluster_size < CRITICAL_TIER_MIN_CLUSTER_SIZE:
            raise ValueError(
                f"Required cluster size must be at least "
                f"{CRITICAL_TIER_MIN_CLUSTER_SIZE=} when "
                f"service tier({tier}) is a "
                f"critical tier({CRITICAL_TIERS}). "
                f"If it is an existing cluster, horizontally "
                f"scale the cluster to be >= "
                f"{CRITICAL_TIER_MIN_CLUSTER_SIZE}"
            )

        return required_cluster_size

    @staticmethod
    def service_costs(
        service_type: str,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> List[ServiceCapacity]:
        # C* service costs: network + backup
        args = NflxCassandraArguments.from_extra_model_arguments(extra_model_arguments)
        copies_per_region: int = _target_rf(
            desires, extra_model_arguments.get("copies_per_region")
        )

        # Compute overhead-adjusted wire write size for CRR network cost.
        # Cassandra mutations carry ~900B fixed overhead plus 1.42x proportional
        # overhead vs raw app payload. See module-level constants for derivation.
        current_write_size = desires.query_pattern.estimated_mean_write_size_bytes.mid
        wire_write_size = _cassandra_wire_write_size(current_write_size)
        write_size_defaulted = _is_write_size_defaulted(desires)

        # Adjust desires to use wire write size for network cost calculation.
        # We copy desires rather than modifying the shared function in common.py,
        # since the overhead is Cassandra-specific (other models use
        # network_services() with their own wire formats).
        adjusted_desires = desires.model_copy(deep=True)
        adjusted_desires.query_pattern.estimated_mean_write_size_bytes = certain_int(
            wire_write_size
        )

        services: List[ServiceCapacity] = []

        # TODO(homatthew): Move cost confidence/warning signals to a top-level
        # field on CapacityPlan (e.g., cost_warnings or cost_metadata). This is
        # part of a larger explainability feature for the capacity planner —
        # surfacing which costs are estimated vs measured, which inputs were
        # defaulted, and how confident the model is in each cost component.
        # For now, service_params carries the flag per-service.
        net_services = network_services(
            service_type, context, adjusted_desires, copies_per_region
        )
        for svc in net_services:
            svc.service_params["write_size_defaulted"] = write_size_defaulted
        services.extend(net_services)

        if desires.data_shape.durability_slo_order.mid >= 1000:
            blob = context.services.get("blob.standard", None)
            if blob:
                # Snapshot component: data-at-rest per zone
                backup_disk_gib = max(
                    1,
                    _get_disk_from_desires(desires, copies_per_region)
                    // context.zones_in_region,
                )

                # Write-throughput component: backup storage is dominated by
                # continuous SSTable uploads, not just the data-at-rest snapshot.
                # Uses overhead-adjusted wire size because SSTables include
                # full serialized mutations (cell metadata, timestamps, bloom
                # filter contributions), not just raw app payload.
                wps = desires.query_pattern.estimated_write_per_second.mid
                daily_write_gib = (wps * wire_write_size * 86400) / (1024**3)
                retention_days = (
                    args.backup_retention_days or _DEFAULT_BACKUP_RETENTION_DAYS
                )

                # Total = state snapshot + retained write volume
                backup_total_gib = backup_disk_gib + daily_write_gib * retention_days

                services.append(
                    ServiceCapacity(
                        service_type=f"{service_type}.backup.{blob.name}",
                        annual_cost=blob.annual_cost_gib(backup_total_gib),
                        service_params={
                            "snapshot_gib": backup_disk_gib,
                            "daily_write_gib": round(daily_write_gib, 1),
                            "retention_days": retention_days,
                            "write_size_defaulted": write_size_defaulted,
                        },
                    )
                )

        return services

    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Union[CapacityPlan, Excuse, None]:
        # Parse extra_model_arguments into a validated model with centralized defaults
        args = NflxCassandraArguments.from_extra_model_arguments(extra_model_arguments)

        # Use durability and consistency to compute RF if not explicitly set
        copies_per_region = _target_rf(desires, args.copies_per_region)

        # Validate required_cluster_size for critical tiers
        required_cluster_size: Optional[int] = (
            NflxCassandraCapacityModel.get_required_cluster_size(
                desires.service_tier, extra_model_arguments
            )
        )

        # Apply caps to buffer percentages
        max_write_buffer_percent = min(0.5, args.max_write_buffer_percent)
        max_table_buffer_percent = min(0.5, args.max_table_buffer_percent)

        # Adjust heap defaults for high write clusters
        if (
            desires.query_pattern.estimated_write_per_second.mid >= 100_000
            and desires.data_shape.estimated_state_size_gib.mid >= 100
        ):
            max_write_buffer_percent = max(0.5, max_write_buffer_percent)
            max_table_buffer_percent = max(0.2, max_table_buffer_percent)

        result = _estimate_cassandra_cluster_zonal(
            instance=instance,
            drive=drive,
            context=context,
            desires=desires,
            zones_per_region=context.zones_in_region,
            copies_per_region=copies_per_region,
            require_local_disks=args.require_local_disks,
            require_attached_disks=args.require_attached_disks,
            required_cluster_size=required_cluster_size,
            max_rps_to_disk=args.max_rps_to_disk,
            max_regional_size=args.max_regional_size,
            max_local_data_per_node_gib=args.max_local_data_per_node_gib,
            max_attached_data_per_node_gib=args.max_attached_data_per_node_gib,
            max_write_buffer_percent=max_write_buffer_percent,
            max_table_buffer_percent=max_table_buffer_percent,
            large_instance_regret=args.large_instance_regret,
            different_family_regret=args.different_family_regret,
            experimental_memory_model=args.experimental_memory_model,
            max_page_cache_gib=args.max_page_cache_gib,
            backup_retention_days=args.backup_retention_days,
        )

        return result

    @staticmethod
    def regret(
        regret_params: CapacityRegretParameters,
        optimal_plan: CapacityPlan,
        proposed_plan: CapacityPlan,
    ) -> Dict[str, float]:
        regrets = CapacityModel.regret(regret_params, optimal_plan, proposed_plan)

        # Large instance size is a pairwise property — choosing 32xlarge over
        # 8xlarge has operational risk (fewer nodes, harder to scale) that
        # varies by scenario. Family migration is not pairwise — it's a fixed
        # switching cost independent of which plan is optimal, so it only
        # needs the rank penalty to bias selection toward the current family.
        if proposed_plan.candidate_clusters.zonal:
            params = proposed_plan.candidate_clusters.zonal[0].cluster_params
            penalties = params.get(RANK_PENALTIES, {})
            if "large_instance" in penalties:
                # Apply regret only to compute costs (same rationale as rank)
                compute_cost = float(
                    sum(c.annual_cost for c in proposed_plan.candidate_clusters.zonal)
                )
                regrets["large_instance"] = penalties["large_instance"] * compute_cost

        return regrets

    @staticmethod
    def description() -> str:
        return "Netflix Streaming Cassandra Model"

    @staticmethod
    def extra_model_arguments_schema() -> Dict[str, Any]:
        return NflxCassandraArguments.model_json_schema()

    @staticmethod
    def default_buffers(
        storage_ratio: float = 4.0, compute_ratio: float = 1.5
    ) -> Buffers:
        return Buffers(
            default=Buffer(ratio=1.5),
            desired={
                "compute": Buffer(
                    ratio=compute_ratio, components=[BufferComponent.compute]
                ),
                "storage": Buffer(
                    ratio=storage_ratio, components=[BufferComponent.storage]
                ),
                # Cassandra reserves headroom in both cpu and network for background
                # work and tasks
                "background": Buffer(
                    ratio=2.0,
                    components=[
                        BufferComponent.cpu,
                        BufferComponent.network,
                        BACKGROUND_BUFFER,
                    ],
                ),
            },
        )

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> CapacityDesires:
        acceptable_consistency = {
            None,
            AccessConsistency.best_effort,
            AccessConsistency.eventual,
            AccessConsistency.read_your_writes,
            AccessConsistency.never,
        }
        for key, value in user_desires.query_pattern.access_consistency:
            if value.target_consistency not in acceptable_consistency:
                raise ValueError(
                    f"Cassandra can only provide {acceptable_consistency} access."
                    f"User asked for {key}={value}"
                )

        # Lower RF = less write compute
        rf = _target_rf(
            user_desires, extra_model_arguments.get("copies_per_region", None)
        )
        if rf < 3:
            rf_write_latency = Interval(low=0.2, mid=0.6, high=2, confidence=0.98)
        else:
            rf_write_latency = Interval(low=0.4, mid=1, high=2, confidence=0.98)

        # Compute adaptive storage buffer ratio based on data size
        args = NflxCassandraArguments.from_extra_model_arguments(extra_model_arguments)
        storage_ratio = args.max_storage_buffer_ratio
        if args.adaptive_storage_buffer:
            storage_ratio = _adaptive_storage_buffer_ratio(
                _estimate_zonal_data_gib(user_desires, rf),
                max_ratio=args.max_storage_buffer_ratio,
                min_ratio=args.min_storage_buffer_ratio,
            )

        # Compute adaptive compute buffer ratio based on total RPS
        compute_ratio = args.max_compute_buffer_ratio
        if args.adaptive_compute_buffer:
            # Write-weighted throughput as a proxy for CPU load. Writes are
            # ~3x more expensive per byte than reads (memtable flushes and
            # compaction pressure scale with write size, not just write count).
            qp = user_desires.query_pattern
            write_weighted_throughput_mbps = (
                qp.estimated_read_per_second.mid * qp.estimated_mean_read_size_bytes.mid
                + 3.0
                * qp.estimated_write_per_second.mid
                * qp.estimated_mean_write_size_bytes.mid
            ) / (1024 * 1024)
            compute_ratio = _adaptive_compute_buffer_ratio(
                write_weighted_throughput_mbps,
                max_ratio=args.max_compute_buffer_ratio,
                min_ratio=args.min_compute_buffer_ratio,
            )

        # By supplying these buffers we can deconstruct observed utilization into
        # load versus buffer.
        buffers = NflxCassandraCapacityModel.default_buffers(
            storage_ratio=storage_ratio,
            compute_ratio=compute_ratio,
        )
        if user_desires.query_pattern.access_pattern == AccessPattern.latency:
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.latency,
                    access_consistency=GlobalConsistency(
                        same_region=Consistency(
                            target_consistency=AccessConsistency.read_your_writes,
                        ),
                        cross_region=Consistency(
                            target_consistency=AccessConsistency.eventual,
                        ),
                    ),
                    estimated_mean_read_size_bytes=Interval(
                        low=128, mid=1024, high=65536, confidence=0.95
                    ),
                    estimated_mean_write_size_bytes=Interval(
                        low=64, mid=256, high=1024, confidence=0.95
                    ),
                    # Cassandra point queries usualy take just around 2ms
                    # of on CPU time for reads and 1ms for writes
                    estimated_mean_read_latency_ms=Interval(
                        low=0.4, mid=2, high=5, confidence=0.98
                    ),
                    estimated_mean_write_latency_ms=rf_write_latency,
                    # Assume point queries, "Single digit milliseconds SLO"
                    read_latency_slo_ms=FixedInterval(
                        minimum_value=0.2,
                        maximum_value=10,
                        low=0.4,
                        mid=2,
                        high=5,
                        confidence=0.98,
                    ),
                    write_latency_slo_ms=FixedInterval(
                        minimum_value=0.2,
                        maximum_value=10,
                        low=0.4,
                        mid=1,
                        high=4,
                        confidence=0.98,
                    ),
                ),
                # Most latency sensitive cassandra clusters are in the
                # < 1TiB range
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=10, mid=100, high=1000, confidence=0.98
                    ),
                    # Cassandra compresses with LZ4 by default
                    estimated_compression_ratio=Interval(
                        minimum_value=1.1,
                        maximum_value=8,
                        low=2,
                        mid=3,
                        high=5,
                        confidence=0.98,
                    ),
                    # We dynamically allocate the C* JVM memory in the plan
                    # but account for the Priam sidecar here
                    reserved_instance_app_mem_gib=4,
                ),
                buffers=buffers,
            )
        else:
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.throughput,
                    access_consistency=GlobalConsistency(
                        same_region=Consistency(
                            target_consistency=AccessConsistency.read_your_writes,
                        ),
                        cross_region=Consistency(
                            target_consistency=AccessConsistency.eventual,
                        ),
                    ),
                    estimated_mean_read_size_bytes=Interval(
                        low=128, mid=1024, high=65536, confidence=0.95
                    ),
                    estimated_mean_write_size_bytes=Interval(
                        low=128, mid=1024, high=65536, confidence=0.95
                    ),
                    # Cassandra scan queries usually take longer
                    estimated_mean_read_latency_ms=Interval(
                        low=0.2, mid=5, high=20, confidence=0.98
                    ),
                    # Usually throughput clusters are running RF=2
                    # Maybe revise this?
                    estimated_mean_write_latency_ms=Interval(
                        low=0.2, mid=0.6, high=2, confidence=0.98
                    ),
                    # Assume they're scanning -> slow reads
                    read_latency_slo_ms=FixedInterval(
                        minimum_value=1,
                        maximum_value=100,
                        low=2,
                        mid=8,
                        high=90,
                        confidence=0.98,
                    ),
                    # Assume they're doing BATCH writes
                    write_latency_slo_ms=FixedInterval(
                        minimum_value=0.5,
                        maximum_value=20,
                        low=1,
                        mid=2,
                        high=8,
                        confidence=0.98,
                    ),
                ),
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=100, mid=1000, high=4000, confidence=0.98
                    ),
                    # Cassandra compresses with LZ4 by default
                    estimated_compression_ratio=Interval(
                        low=2, mid=3, high=5, confidence=0.98
                    ),
                    # We dynamically allocate the C* JVM memory in the plan
                    # but account for the Priam sidecar here
                    reserved_instance_app_mem_gib=4,
                ),
                buffers=buffers,
            )


nflx_cassandra_capacity_model = NflxCassandraCapacityModel()

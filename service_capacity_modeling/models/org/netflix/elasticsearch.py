import logging
import math
from decimal import Decimal
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRequirement
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.common import compute_stateful_zone
from service_capacity_modeling.models.common import simple_network_mbps
from service_capacity_modeling.models.common import sqrt_staffed_cores
from service_capacity_modeling.models.common import working_set_from_drive_and_slo
from service_capacity_modeling.stats import dist_for_interval


logger = logging.getLogger(__name__)


def _target_rf(desires: CapacityDesires, user_copies: Optional[int]) -> int:
    if user_copies is not None:
        assert user_copies > 1
        return user_copies

    # Due to the relaxed durability and consistency requirements we can
    # run with RF=2
    if desires.data_shape.durability_slo_order.mid < 1000:
        return 2
    return 3


# Looks like Elasticsearch (Lucene) uses a tiered merge strategy of 10
# segments of 512 megs per
# https://lucene.apache.org/core/8_1_0/core/org/apache/lucene/index/TieredMergePolicy.html#setSegmentsPerTier(double)
# (FIXME) Verify what elastic merge actually does
def _es_io_per_read(node_size_gib, segment_size_mb=512):
    size_mib = node_size_gib * 1024
    segments = max(1, size_mib // segment_size_mb)
    # 10 segments per tier, plus 1 for L0 (avg)
    levels = 1 + int(math.ceil(math.log(segments, 10)))
    return levels


def _estimate_elasticsearch_requirement(
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
    # Keep half of the cores free for background work (merging mostly)
    needed_cores = math.ceil(sqrt_staffed_cores(desires) * 1.5)
    # Keep half of the bandwidth available for backup
    needed_network_mbps = simple_network_mbps(desires) * 2

    needed_disk = math.ceil(
        (1.0 / desires.data_shape.estimated_compression_ratio.mid)
        * desires.data_shape.estimated_state_size_gib.mid
        * copies_per_region,
    )

    # Rough estimate of how many instances we would need just for the the CPU
    # Note that this is a lower bound, we might end up with more.
    needed_cores = math.ceil(
        max(1, needed_cores // (instance.cpu_ghz / desires.core_reference_ghz))
    )
    rough_count = math.ceil(needed_cores / instance.cpu)

    # Generally speaking we want fewer than some number of reads per second
    # hitting disk per instance. If we don't have many reads we don't need to
    # hold much data in memory.
    instance_rps = max(1, reads_per_second // rough_count)
    disk_rps = instance_rps * _es_io_per_read(max(1, needed_disk // rough_count))
    rps_working_set = min(1.0, disk_rps / max_rps_to_disk)

    # If disk RPS will be smaller than our target because there are no
    # reads, we don't need to hold as much data in memory
    needed_memory = min(working_set, rps_working_set) * needed_disk

    # Now convert to per zone
    needed_cores = needed_cores // zones_per_region
    needed_disk = needed_disk // zones_per_region
    needed_memory = int(needed_memory // zones_per_region)
    logger.debug(
        "Need (cpu, mem, disk, working) = (%s, %s, %s, %f)",
        needed_cores,
        needed_memory,
        needed_disk,
        working_set,
    )

    return CapacityRequirement(
        requirement_type="elasticsearch-data-zonal",
        core_reference_ghz=desires.core_reference_ghz,
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


def _upsert_params(cluster, params):
    if cluster.cluster_params:
        cluster.cluster_params.update(params)
    else:
        cluster.cluster_params = params


# pylint: disable=too-many-locals
def _estimate_elasticsearch_cluster_zonal(
    instance: Instance,
    drive: Drive,
    desires: CapacityDesires,
    zones_per_region: int = 3,
    copies_per_region: int = 3,
    max_local_disk_gib: int = 4096,
    max_regional_size: int = 240,
    max_rps_to_disk: int = 500,
) -> Optional[CapacityPlan]:

    # Netflix Elasticsearch doesn't like to deploy on really small instances
    if instance.cpu < 2 or instance.ram_gib < 14:
        return None

    # (FIXME): Need elasticsearch input
    # Right now Elasticsearch doesn't deploy to cloud drives, just adding this
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
        # Elasticsearch has looser latency SLOs, target the 90th percentile of disk
        # latency to keep in RAM.
        target_percentile=0.90,
    ).mid

    requirement = _estimate_elasticsearch_requirement(
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

    cluster = compute_stateful_zone(
        instance=instance,
        drive=drive,
        needed_cores=int(requirement.cpu_cores.mid),
        needed_disk_gib=int(requirement.disk_gib.mid),
        needed_memory_gib=int(requirement.mem_gib.mid),
        needed_network_mbps=requirement.network_mbps.mid,
        # Assume that by provisioning enough memory we'll get
        # a 90% hit rate, but take into account the reads per read
        # from the per node dataset using leveled compaction
        # FIXME: I feel like this can be improved
        required_disk_ios=lambda x: _es_io_per_read(x) * math.ceil(0.1 * rps),
        # Elasticsearch requires ephemeral disks to be % full because tiered
        # merging can make progress as long as there is some headroom
        required_disk_space=lambda x: x * 1.4,
        max_local_disk_gib=max_local_disk_gib,
        # elasticsearch clusters can autobalance via shard placement
        cluster_size=lambda x: x,
        min_count=1,
        # Sidecars/System takes away memory from elasticsearch
        # Elasticsearch uses half of available system max of 32 for compressed
        # oops
        reserve_memory=lambda x: base_mem + max(32, x / 2),
        core_reference_ghz=requirement.core_reference_ghz,
    )

    # Communicate to the actual provision that if we want reduced RF
    params = {"elasticsearch.copies": copies_per_region}
    _upsert_params(cluster, params)

    # elasticsearch clusters generally should try to stay under some total number
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

    cluster.cluster_type = "elasticsearch-data"
    clusters = Clusters(
        total_annual_cost=round(Decimal(ec2_cost), 2),
        zonal=[cluster] * zones_per_region,
        regional=list(),
    )

    return CapacityPlan(
        requirements=Requirements(zonal=[requirement] * zones_per_region),
        candidate_clusters=clusters,
    )


class NflxElasticsearchCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        # (FIXME): Need elasticsearch input
        # TODO: Use durability requirements to compute RF.
        copies_per_region: int = _target_rf(
            desires, extra_model_arguments.get("copies_per_region", None)
        )
        max_regional_size: int = extra_model_arguments.get("max_regional_size", 240)
        max_rps_to_disk: int = extra_model_arguments.get("max_rps_to_disk", 1000)
        # Very large nodes are hard to recover
        max_local_disk_gib: int = extra_model_arguments.get("max_local_disk_gib", 5000)

        return _estimate_elasticsearch_cluster_zonal(
            instance=instance,
            drive=drive,
            desires=desires,
            zones_per_region=context.zones_in_region,
            copies_per_region=copies_per_region,
            max_regional_size=max_regional_size,
            max_local_disk_gib=max_local_disk_gib,
            max_rps_to_disk=max_rps_to_disk,
        )

    @staticmethod
    def description():
        return "Netflix Streaming Elasticsearch Model"

    @staticmethod
    def extra_model_arguments() -> Sequence[Tuple[str, str, str]]:
        return (
            (
                "copies_per_region",
                "int = 3",
                "How many copies of the data will exist e.g. RF=3. If unsupplied"
                " this will be deduced from durability and consistency desires",
            ),
            (
                "max_regional_size",
                # Twice the size of our largest cluster
                "int = 240",
                "What is the maximum size of a cluster in this region",
            ),
            (
                "max_local_disk_gib",
                # Nodes larger than 4 TiB are painful to recover
                "int = 4096",
                "The maximum amount of data we store per machine",
            ),
            (
                "max_rps_to_disk",
                "int = 1000",
                "How many disk IOs should be allowed to hit disk per instance",
            ),
        )

    @staticmethod
    def default_desires(user_desires, extra_model_arguments: Dict[str, Any]):
        acceptable_consistency = set(
            (
                AccessConsistency.best_effort,
                AccessConsistency.eventual,
                AccessConsistency.never,
                None,
            )
        )
        for key, value in user_desires.query_pattern.access_consistency:
            if value.target_consistency not in acceptable_consistency:
                raise ValueError(
                    f"Elasticsearch can only provide {acceptable_consistency} access."
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

        if user_desires.query_pattern.access_pattern == AccessPattern.latency:
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.latency,
                    estimated_mean_read_size_bytes=Interval(
                        low=128, mid=4096, high=131072, confidence=0.98
                    ),
                    estimated_mean_write_size_bytes=Interval(
                        low=128, mid=4096, high=131072, confidence=0.98
                    ),
                    # Elasticsearch reads and writes can take CPU time as
                    # large cardinality search and such can be hard to predict.
                    estimated_mean_read_latency_ms=Interval(
                        low=1, mid=2, high=100, confidence=0.98
                    ),
                    # Writes depend heavily on rf and consistency
                    estimated_mean_write_latency_ms=rf_write_latency,
                    # Assume point queries "Single digit millisecond SLO"
                    read_latency_slo_ms=FixedInterval(
                        low=1,
                        mid=10,
                        high=100,
                        confidence=0.98,
                    ),
                    write_latency_slo_ms=FixedInterval(
                        low=1,
                        mid=10,
                        high=100,
                        confidence=0.98,
                    ),
                ),
                # Most latency sensitive elasticsearch clusters are in the
                # < 100GiB range
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=10, mid=100, high=1000, confidence=0.98
                    ),
                    # Netflix Elasticsearch compresses with Deflate (gzip)
                    # by default
                    estimated_compression_ratio=Interval(
                        minimum_value=1.4,
                        maximum_value=8,
                        low=2,
                        mid=3,
                        high=4,
                        confidence=0.98,
                    ),
                    # Elasticsearch has a 1 GiB sidecar
                    reserved_instance_app_mem_gib=1,
                ),
            )
        else:
            return CapacityDesires(
                # (FIXME): Need to pair with ES folks on the exact values
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.throughput,
                    # Bulk writes and reads are larger in general
                    estimated_mean_read_size_bytes=Interval(
                        low=128, mid=4096, high=131072, confidence=0.95
                    ),
                    estimated_mean_write_size_bytes=Interval(
                        low=4096, mid=16384, high=1048576, confidence=0.95
                    ),
                    # (FIXME): Need es input
                    # Elasticsearch analytics reads probably take extra time
                    # as they are scrolling or doing complex aggregations
                    estimated_mean_read_latency_ms=Interval(
                        low=1, mid=20, high=100, confidence=0.98
                    ),
                    # Throughput writes typically involve large bulks
                    # which can be expensive, but it's rare for it to have a
                    # huge tail
                    estimated_mean_write_latency_ms=Interval(
                        low=1, mid=10, high=100, confidence=0.98
                    ),
                    # Assume scan queries "Tens of millisecond SLO"
                    read_latency_slo_ms=FixedInterval(
                        minimum_value=1,
                        maximum_value=100,
                        low=10,
                        mid=20,
                        high=100,
                        confidence=0.98,
                    ),
                    write_latency_slo_ms=FixedInterval(
                        minimum_value=1,
                        maximum_value=100,
                        low=10,
                        mid=20,
                        high=100,
                        confidence=0.98,
                    ),
                ),
                # Most throughput elasticsearch clusters are in the
                # < 1TiB range
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=100, mid=1000, high=10000, confidence=0.98
                    ),
                    # Netflix Elasticsearch compresses with Deflate (gzip)
                    # by default
                    estimated_compression_ratio=Interval(
                        minimum_value=1.4,
                        maximum_value=8,
                        low=2,
                        mid=3,
                        high=4,
                        confidence=0.98,
                    ),
                    # Elasticsearch has a 1 GiB sidecar
                    reserved_instance_app_mem_gib=1,
                ),
            )


nflx_elasticsearch_capacity_model = NflxElasticsearchCapacityModel()

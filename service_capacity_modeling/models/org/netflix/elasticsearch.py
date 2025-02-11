import logging
import math
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

from pydantic import BaseModel
from pydantic import Field

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
from service_capacity_modeling.interface import ZoneClusterCapacity
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.common import compute_stateful_zone
from service_capacity_modeling.models.common import normalize_cores
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


def _estimate_elasticsearch_requirement(  # noqa: E501 pylint: disable=too-many-positional-arguments
    node_type: str,
    instance: Instance,
    desires: CapacityDesires,
    working_set: float,
    reads_per_second: float,
    max_rps_to_disk: int,
    zones_in_region: int = 3,
    copies_per_region: int = 3,
    jvm_memory_overhead=1.2,
) -> CapacityRequirement:
    """Estimate the capacity required for one zone given a regional desire

    The input desires should be the **regional** desire, and this function will
    return the zonal capacity requirement
    """
    # Keep half of the cores free for background work (merging mostly)
    needed_cores = normalize_cores(
        core_count=math.ceil(sqrt_staffed_cores(desires) * 1.5),
        target_shape=instance,
        reference_shape=desires.reference_shape,
    )
    # Keep half of the bandwidth available for backup
    needed_network_mbps = simple_network_mbps(desires) * 2

    needed_disk = math.ceil(
        (1.0 / desires.data_shape.estimated_compression_ratio.mid)
        * desires.data_shape.estimated_state_size_gib.mid
        * copies_per_region,
    )

    # Rough estimate of how many instances we would need just for the CPU
    # Note that this is a lower bound, we might end up with more.
    rough_count = math.ceil(needed_cores / instance.cpu)

    # Generally speaking we want fewer than some number of reads per second
    # hitting disk per instance. If we don't have many reads we don't need to
    # hold much data in memory.
    instance_rps = max(1, reads_per_second // rough_count)
    disk_rps = instance_rps * _es_io_per_read(max(1, needed_disk // rough_count))
    rps_working_set = min(1.0, disk_rps / max_rps_to_disk)

    mem_allocation_mbps = needed_network_mbps * jvm_memory_overhead
    heap_allocation_gibps = (mem_allocation_mbps / 8) / 1024
    network_heap = heap_allocation_gibps * 2

    # If disk RPS will be smaller than our target because there are no
    # reads, we don't need to hold as much data in memory
    needed_memory = min(working_set, rps_working_set) * needed_disk + network_heap

    # Now convert to per zone
    needed_cores = needed_cores // zones_in_region
    needed_disk = needed_disk // zones_in_region
    needed_memory = int(needed_memory // zones_in_region)
    logger.debug(
        "Need (cpu, mem, disk, working) = (%s, %s, %s, %f)",
        needed_cores,
        needed_memory,
        needed_disk,
        working_set,
    )

    return CapacityRequirement(
        requirement_type=f"elasticsearch-{node_type}-zonal",
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


def _upsert_params(cluster, params):
    if cluster.cluster_params:
        cluster.cluster_params.update(params)
    else:
        cluster.cluster_params = params


# pylint: disable=too-many-locals


class NflxElasticsearchArguments(BaseModel):
    copies_per_region: int = Field(
        default=3,
        description="How many copies of the data will exist e.g. RF=3. If not supplied"
        " this will be deduced from durability and consistency desires",
    )
    max_regional_size: int = Field(
        # Twice the size of our largest cluster
        default=240,
        description="What is the maximum size of a cluster in this region",
    )
    max_local_disk_gib: int = Field(
        # Nodes larger than 8 TiB are painful to recover
        default=8192,
        description="The maximum amount of data we store per machine",
    )
    max_rps_to_disk: int = Field(
        default=1000,
        description="How many disk IOs should be allowed to hit disk per instance",
    )


class NflxElasticsearchDataCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:

        copies_per_region: int = _target_rf(
            desires, extra_model_arguments.get("copies_per_region", None)
        )
        max_regional_size: int = extra_model_arguments.get("max_regional_size", 120)
        max_rps_to_disk: int = extra_model_arguments.get("max_rps_to_disk", 1000)
        # Very large nodes are hard to recover
        max_local_disk_gib: int = extra_model_arguments.get("max_local_disk_gib", 8192)

        # the ratio of traffic that should be handled by search nodes.
        #  0.0 = no search nodes, all searches handled by data nodes
        #  1.0 = requests split 50/50 between search and data nodes
        search_to_data_rps_ratio = extra_model_arguments.get(
            "search_to_data_rps_ratio", 0.0
        )

        # Netflix Elasticsearch doesn't like to deploy on really small instances
        if instance.cpu < 2 or instance.ram_gib < 24:
            return None

        # Right now Elasticsearch doesn't deploy to cloud drives
        if instance.drive is None:
            return None

        zones_in_region = context.zones_in_region

        _rps = desires.query_pattern.estimated_read_per_second.mid // zones_in_region

        data_rps = _rps / (search_to_data_rps_ratio + 1)

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

        data_requirement = _estimate_elasticsearch_requirement(
            node_type="data",
            instance=instance,
            desires=desires,
            working_set=working_set,
            reads_per_second=data_rps,
            zones_in_region=zones_in_region,
            copies_per_region=copies_per_region,
            max_rps_to_disk=max_rps_to_disk,
        )
        base_mem = (
            desires.data_shape.reserved_instance_app_mem_gib
            + desires.data_shape.reserved_instance_system_mem_gib
        )

        data_write_per_sec = (
            desires.query_pattern.estimated_write_per_second.mid // zones_in_region
        )
        data_write_bytes_per_sec = (
            data_write_per_sec
            * desires.query_pattern.estimated_mean_write_size_bytes.mid
        )

        # Write IO will be 1 to translog + 5 read+writes in the first hour
        # during segment merges. Drives don't differentiate writes from reads
        # for the most part, so we just account for the merge reads here
        # as writes.
        # https://aws.amazon.com/ebs/volume-types/ says IOPS are 16k for
        # io2/gp2 so for now we're just hardcoding.
        data_write_io_per_sec = (1 + 10) * max(1, data_write_bytes_per_sec // 16384)

        data_cluster = compute_stateful_zone(
            instance=instance,
            drive=drive,
            needed_cores=int(data_requirement.cpu_cores.mid),
            needed_disk_gib=int(data_requirement.disk_gib.mid),
            needed_memory_gib=int(data_requirement.mem_gib.mid),
            needed_network_mbps=data_requirement.network_mbps.mid,
            # Take into account the reads per read
            # from the per node dataset using leveled compaction
            required_disk_ios=lambda size, count: (
                _es_io_per_read(size) * math.ceil(data_rps / count),
                data_write_io_per_sec / count,
            ),
            # Elasticsearch requires ephemeral disks to be % full because tiered
            # merging can make progress as long as there is some headroom
            required_disk_space=lambda x: x * 1.33,
            max_local_disk_gib=max_local_disk_gib,
            # Elasticsearch clusters can auto-balance via shard placement
            cluster_size=lambda x: x,
            min_count=1,
            # Sidecars/System takes away memory from Elasticsearch
            # which uses half of available system max of 32 for compressed oops
            reserve_memory=lambda x: base_mem + max(32, x / 2),
        )
        data_cluster.cluster_type = "elasticsearch-data"

        # Communicate to the actual provision that if we want reduced RF
        params = {"elasticsearch.copies": copies_per_region}
        _upsert_params(data_cluster, params)

        # Elasticsearch clusters generally should try to stay under some total number
        # of nodes. Orgs do this for all kinds of reasons such as
        #  * Maintenance. If your restart script does one node at a time you want
        #    smaller clusters so your restarts don't take months.
        #  * NxN network issues. Sometimes smaller clusters of bigger nodes
        #    are better for network propagation
        if data_cluster.count > (max_regional_size // zones_in_region):
            return None

        ec2_costs = {
            "elasticsearch-data.zonal-clusters": zones_in_region
            * data_cluster.annual_cost
        }

        clusters = Clusters(
            annual_costs=ec2_costs,
            zonal=[data_cluster] * zones_in_region,
            regional=[],
        )

        plan = CapacityPlan(
            requirements=Requirements(zonal=[data_requirement] * zones_in_region),
            candidate_clusters=clusters,
        )

        return plan


class NflxElasticsearchMasterCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        # Only accept running on instances with a lot of RAM and a few CPUs
        if instance.ram_gib <= 24:
            return None
        if instance.cpu <= 2:
            return None

        zones_in_region = context.zones_in_region
        requirement = CapacityRequirement(
            requirement_type="elasticsearch-master-zonal",
            cpu_cores=certain_int(2),
            mem_gib=certain_int(24),
            context={},
        )

        cluster = ZoneClusterCapacity(
            cluster_type="elasticsearch-master",
            count=1,
            instance=instance,
            attached_drives=[],
            annual_cost=instance.annual_cost,
        )

        # TODO(josephl): This probably needs network transfer costs like
        # C*, EVCache, etc ... have
        ec2_cost = zones_in_region * cluster.annual_cost
        clusters = Clusters(
            annual_costs={"elasticsearch-master.zonal-clusters": ec2_cost},
            zonal=[cluster] * zones_in_region,
        )

        return CapacityPlan(
            requirements=Requirements(zonal=[requirement] * zones_in_region),
            candidate_clusters=clusters,
        )


class NflxElasticsearchSearchCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        # Only accept running on instances with a lot of RAM and a few CPUs
        if instance.ram_gib <= 24:
            return None
        if instance.cpu <= 2:
            return None

        zones_in_region = context.zones_in_region
        requirement = CapacityRequirement(
            requirement_type="elasticsearch-search-zonal",
            cpu_cores=certain_int(2),
            mem_gib=certain_int(24),
            context={},
        )

        cluster = ZoneClusterCapacity(
            cluster_type="elasticsearch-search",
            count=1,
            instance=instance,
            attached_drives=[],
            annual_cost=instance.annual_cost,
        )

        ec2_cost = zones_in_region * cluster.annual_cost
        clusters = Clusters(
            annual_costs={"elasticsearch-search.zonal-clusters": ec2_cost},
            zonal=[cluster] * zones_in_region,
        )

        return CapacityPlan(
            requirements=Requirements(zonal=[requirement] * zones_in_region),
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
        return None

    @staticmethod
    def description():
        return "Netflix Streaming Elasticsearch Model"

    @staticmethod
    def compose_with(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> Tuple[Tuple[str, Callable[[CapacityDesires], CapacityDesires]], ...]:
        def _modify_data_desires(desires: CapacityDesires) -> CapacityDesires:
            # data node's model use the full desires
            return desires

        def _modify_master_desires(desires: CapacityDesires) -> CapacityDesires:
            # master node's model doesn't use anything from the desires
            return desires

        def _modify_search_desires(desires: CapacityDesires) -> CapacityDesires:
            # search node's model doesn't use anything from the desires
            return desires

        return (
            ("org.netflix.elasticsearch.node", _modify_data_desires),
            ("org.netflix.elasticsearch.master", _modify_master_desires),
            ("org.netflix.elasticsearch.search", _modify_search_desires),
        )

    @staticmethod
    def extra_model_arguments_schema() -> Dict[str, Any]:
        return NflxElasticsearchArguments.model_json_schema()

    @staticmethod
    def default_desires(user_desires, extra_model_arguments: Dict[str, Any]):
        acceptable_consistency = {
            AccessConsistency.best_effort,
            AccessConsistency.eventual,
            AccessConsistency.never,
            None,
        }
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
                # Most latency sensitive Elasticsearch clusters are in the <100GiB range
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=10, mid=100, high=1000, confidence=0.98
                    ),
                    # Netflix Elasticsearch compresses with Deflate (gzip) by default
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
nflx_elasticsearch_data_capacity_model = NflxElasticsearchDataCapacityModel()
nflx_elasticsearch_master_capacity_model = NflxElasticsearchMasterCapacityModel()
nflx_elasticsearch_search_capacity_model = NflxElasticsearchSearchCapacityModel()

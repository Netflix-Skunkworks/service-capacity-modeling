from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

from pydantic import Field

from .stateless_java import nflx_java_app_capacity_model
from .stateless_java import NflxJavaAppArguments
from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import Consistency
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import GlobalConsistency
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.models import CapacityModel


# ===========================================================================
# GraphKV read/write amplification model
#
# A single logical GraphKV operation fans out into many backend KV operations.
# We split the inputs to that fan-out into two buckets:
#
#   1. Namespace inputs  (NflxGraphKVArguments below) -- supplied per namespace
#      via extra_model_arguments. They describe the *shape of the graph*.
#   2. Model assumptions (the MODULE CONSTANTS below) -- fleet-wide engine /
#      workload constants we hold fixed and tune in one place.
#
# Caching (write-back EVCache) and time-travel (temporal index) are
# intentionally excluded from this model.
# ===========================================================================

# Server-side ceiling on traversal depth, so a namespace cannot ask the model to
# walk deeper than the engine will.
MAX_TRAVERSAL_DEPTH = 3


class NflxGraphKVArguments(NflxJavaAppArguments):
    """Per-namespace inputs describing the graph shape.

    These are the only values a namespace owner supplies; everything else in
    the amplification calculation is a fixed model assumption (see the module
    constants below).
    """

    avg_fanout_per_hop: float = Field(
        default=10.0,
        ge=0.0,
        alias="graphkv.avg-fanout-per-hop",
        description=(
            "Average number of edges a traversal follows out of one frontier "
            "node in one hop, i.e. the branching factor. Grows the frontier the "
            "next hop has to scan, so it only costs backend reads when the "
            "traversal is deeper than one hop; at depth 1 it costs response "
            "bytes, not requests."
        ),
    )
    avg_traversal_depth: int = Field(
        default=1,
        ge=1,
        le=MAX_TRAVERSAL_DEPTH,
        alias="graphkv.avg-traversal-depth",
        description=(
            "Average number of hops a traversal actually walks, 1 to "
            f"{MAX_TRAVERSAL_DEPTH}. This is the achieved depth, not the "
            "configured ceiling, which is set to the maximum far more often "
            "than it is reached."
        ),
    )


# --- Model assumptions (NOT namespace config) -----------------------------
# Empirically derived from two production workloads (Jul 2026), one node-heavy
# and one edge-heavy, each compared against the backend load it actually caused.
#
# Backend reads per logical read at depth 1: a traversal reads its source entity
# and scans one edge index, and a point read costs about one. Measured 1.17 to
# 1.37 across both workloads at mean and peak.
KV_READS_PER_LOGICAL_READ = 1.4
# Backend write requests per logical write. A node write is one request; an edge
# write is two, one per direction, plus one more when the edge carries
# properties. Measured 1.2 on the node-heavy workload and 1.6 on the edge-heavy
# one; held above the midpoint because under-provisioning writes costs more than
# over-provisioning them.
KV_WRITES_PER_LOGICAL_WRITE = 1.5
# Bytes one backend record occupies including key and metadata. Measured at 122 B
# and 207 B on the two workloads; rounded up for the same reason.
BYTES_PER_KV_RECORD = 256
# Server-side cap on edges walked in one traversal, which bounds the geometric
# frontier growth below.
MAX_EDGES_PER_TRAVERSAL = 100_000


def _read_amplification(args: NflxGraphKVArguments) -> float:
    """Backend KV reads per logical GraphKV read.

    Each frontier node is expanded by its own KV request -- GraphKV does not
    batch a hop's frontier into one call -- so request count is driven by how
    many nodes get expanded, which is geometric in fan-out over hops:

        frontier(1) = 1, frontier(h + 1) = frontier(h) * fanout
        reads = KV_READS_PER_LOGICAL_READ * sum(frontier(1..depth))

    Fan-out therefore costs nothing at depth 1 (the one scan returns all the
    neighbors at once, as response bytes); it only multiplies request count once
    there is a second hop to expand into.
    """
    frontier, scans, edges = 1.0, 0.0, 0.0
    for _ in range(args.avg_traversal_depth):
        scans += frontier
        edges += frontier * args.avg_fanout_per_hop
        if edges >= MAX_EDGES_PER_TRAVERSAL:
            break
        frontier *= args.avg_fanout_per_hop
    return KV_READS_PER_LOGICAL_READ * scans


class NflxGraphKVCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        graphkv_app = nflx_java_app_capacity_model.capacity_plan(
            instance=instance,
            drive=drive,
            context=context,
            desires=desires,
            extra_model_arguments=extra_model_arguments,
        )
        if graphkv_app is None:
            return None

        for cluster in graphkv_app.candidate_clusters.regional:
            cluster.cluster_type = "dgwgraphkv"
        return graphkv_app

    @staticmethod
    def description() -> str:
        return "Netflix Streaming Graph Abstraction"

    @staticmethod
    def extra_model_arguments_schema() -> Dict[str, Any]:
        return NflxGraphKVArguments.model_json_schema()

    @staticmethod
    def compose_with(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> Tuple[Tuple[str, Callable[[CapacityDesires], CapacityDesires]], ...]:
        def _modify_kv_desires(
            user_desires: CapacityDesires,
        ) -> CapacityDesires:
            relaxed = user_desires.model_copy(deep=True)

            # Per-namespace graph shape drives how each logical read/write fans
            # out into backend KV operations. See _read_amplification and the
            # model constants above.
            args = NflxGraphKVArguments.model_validate(extra_model_arguments)
            relaxed.query_pattern.estimated_read_per_second = (
                user_desires.query_pattern.estimated_read_per_second.scale(
                    _read_amplification(args)
                )
            )
            relaxed.query_pattern.estimated_write_per_second = (
                user_desires.query_pattern.estimated_write_per_second.scale(
                    KV_WRITES_PER_LOGICAL_WRITE
                )
            )

            # An item count is logical nodes plus edges, so it has to be expanded
            # into backend KV records to become a stored size. A state size in GiB
            # is already a backend size -- re-deriving an item count from it and
            # expanding that would charge the fan-out twice.
            item_count = relaxed.data_shape.estimated_state_item_count
            if item_count is not None:
                relaxed.data_shape.estimated_state_size_gib = item_count.scale(
                    KV_WRITES_PER_LOGICAL_WRITE * BYTES_PER_KV_RECORD / 1024**3
                )
            return relaxed

        return (("org.netflix.key-value", _modify_kv_desires),)

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> CapacityDesires:
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
                        low=64, mid=128, high=1024, confidence=0.95
                    ),
                    estimated_mean_read_latency_ms=Interval(
                        low=0.2, mid=1, high=2, confidence=0.98
                    ),
                    estimated_mean_write_latency_ms=Interval(
                        low=0.2, mid=1, high=2, confidence=0.98
                    ),
                    # "Single digit milliseconds SLO"
                    read_latency_slo_ms=FixedInterval(
                        minimum_value=0.2,
                        maximum_value=10,
                        low=1,
                        mid=3,
                        high=6,
                        confidence=0.98,
                    ),
                    write_latency_slo_ms=FixedInterval(
                        minimum_value=0.2,
                        maximum_value=10,
                        low=0.4,
                        mid=2,
                        high=5,
                        confidence=0.98,
                    ),
                ),
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=10, mid=50, high=200, confidence=0.98
                    ),
                    reserved_instance_app_mem_gib=8,
                ),
            )
        else:
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
                        low=64, mid=128, high=1024, confidence=0.95
                    ),
                    estimated_mean_read_latency_ms=Interval(
                        low=0.2, mid=4, high=6, confidence=0.98
                    ),
                    estimated_mean_write_latency_ms=Interval(
                        low=0.2, mid=1, high=2, confidence=0.98
                    ),
                    # Assume they're doing GetItems scans -> slow reads
                    read_latency_slo_ms=FixedInterval(
                        minimum_value=1,
                        maximum_value=100,
                        low=1,
                        mid=8,
                        high=90,
                        confidence=0.98,
                    ),
                    # Assume they're doing PutRecords (BATCH)
                    write_latency_slo_ms=FixedInterval(
                        minimum_value=1,
                        maximum_value=20,
                        low=2,
                        mid=4,
                        high=10,
                        confidence=0.98,
                    ),
                ),
                # Most throughput GraphKV clusters are large
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=100, mid=1000, high=4000, confidence=0.98
                    ),
                    reserved_instance_app_mem_gib=8,
                ),
            )


nflx_graphkv_capacity_model = NflxGraphKVCapacityModel()

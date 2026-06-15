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


class NflxGraphKVArguments(NflxJavaAppArguments):
    """Per-namespace inputs describing the graph shape.

    These are the only values a namespace owner supplies; everything else in
    the amplification calculation is a fixed model assumption (see the module
    constants below).
    """

    edge_mapping_count: int = Field(
        default=5,
        alias="graphkv.edge-mapping-count",
        description=(
            "Number of registered edge-mappings (distinct "
            "(from_type, edge_type, to_type) triples) in the namespace. Drives "
            "READ amplification: a traversal issues one KV scan per edge-mapping "
            "per hop. Does not change per-write cost."
        ),
    )
    edge_mapping_property_count: int = Field(
        default=1,
        alias="graphkv.edge-mapping-property-count",
        description=(
            "Average number of properties stored on an edge. Drives WRITE "
            "amplification: each property is a separate KV item, so an edge "
            "write fans out to (2 link items + this many property items)."
        ),
    )
    node_mapping_count: int = Field(
        default=5,
        alias="graphkv.node-mapping-count",
        description=(
            "Number of registered node-mappings (distinct node types) in the "
            "namespace. Used only to estimate the node-vs-edge WRITE mix; it "
            "does not change per-operation amplification on its own."
        ),
    )
    node_mapping_property_count: int = Field(
        default=2,
        alias="graphkv.node-mapping-property-count",
        description=(
            "Average number of properties stored on a node. Drives WRITE "
            "amplification: each property is a separate KV item, so a node "
            "write fans out to (1 metadata item + this many property items)."
        ),
    )


# --- Model assumptions (NOT namespace config) -----------------------------
# Every edge is persisted twice: the forward link plus its inverse/reverse
# link (bidirectional symmetry / inverse index). Edge properties are written
# once, on the aligned direction only.
EDGE_DIRECTION_COPIES = 2
# A node write always stores one metadata item in addition to its properties.
NODE_BASE_ITEMS = 1
# Average out-degree a single edge-mapping contributes at a node, i.e. how many
# neighbors of one edge-type a node has (formerly "average_node_fanout").
AVG_FANOUT_PER_EDGE_MAPPING = 10
# Directions scanned per hop: 1 for unidirectional, 2 if a namespace must scan
# both OUT and IN. Held at 1 as the fleet default.
DIRECTION_SCAN_FACTOR = 1
# Property reads charged per visited neighbor. Properties are co-located under
# the entity id, so one scan returns all of them (filter OR hydrate => 1).
PROPERTY_READS_PER_NEIGHBOR = 1
# Representative traversal depth in hops. Kept at 1 so the default estimate
# stays bounded; the geometric fan-out below scales correctly if this is
# raised to model multi-hop traversals.
TRAVERSAL_DEPTH = 1


def _write_amplification(args: NflxGraphKVArguments) -> float:
    """Backend KV item-writes per logical write.

      edge write -> EDGE_DIRECTION_COPIES link items + 1 item per edge property
      node write -> NODE_BASE_ITEMS metadata item + 1 item per node property

    Blended by the node-vs-edge write mix. We don't know the real mix, so we
    proxy it with the registered mapping counts (more edge-mappings than
    node-mappings => more edge-heavy writes).
    """
    edge_write_amp = EDGE_DIRECTION_COPIES + args.edge_mapping_property_count
    node_write_amp = NODE_BASE_ITEMS + args.node_mapping_property_count

    total_mappings = args.edge_mapping_count + args.node_mapping_count
    edge_write_fraction = (
        args.edge_mapping_count / total_mappings if total_mappings > 0 else 0.5
    )
    return (
        edge_write_fraction * edge_write_amp
        + (1 - edge_write_fraction) * node_write_amp
    )


def _read_amplification(args: NflxGraphKVArguments) -> float:
    """Backend KV reads per logical traversal.

    Per hop, per frontier node:
      enumeration = edge_mappings_traversed * DIRECTION_SCAN_FACTOR
                    (one scan per edge-mapping; each returns ~fanout neighbors --
                    note this is ADDITIVE in edge-mappings, not multiplicative)
      hydration   = neighbors * PROPERTY_READS_PER_NEIGHBOR

    Summed geometrically over TRAVERSAL_DEPTH hops, since the frontier grows by
    the per-node neighbor count each hop.
    """
    edge_mappings_traversed = args.edge_mapping_count
    neighbors_per_hop = AVG_FANOUT_PER_EDGE_MAPPING * edge_mappings_traversed
    enumeration_reads = edge_mappings_traversed * DIRECTION_SCAN_FACTOR
    hydration_reads = neighbors_per_hop * PROPERTY_READS_PER_NEIGHBOR
    read_amp_per_hop = enumeration_reads + hydration_reads

    branch = neighbors_per_hop
    if branch <= 1 or TRAVERSAL_DEPTH <= 1:
        return read_amp_per_hop * TRAVERSAL_DEPTH
    # Frontier at hop h is branch**h, so total reads =
    # read_amp_per_hop * sum_{h=0}^{D-1} branch**h.
    hop_multiplier = (float(branch) ** TRAVERSAL_DEPTH - 1) / (branch - 1)
    return read_amp_per_hop * hop_multiplier


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
            # out into backend KV operations. See _read_amplification /
            # _write_amplification and the model constants above.
            args = NflxGraphKVArguments.model_validate(extra_model_arguments)
            relaxed.query_pattern.estimated_read_per_second = (
                user_desires.query_pattern.estimated_read_per_second.scale(
                    _read_amplification(args)
                )
            )
            relaxed.query_pattern.estimated_write_per_second = (
                user_desires.query_pattern.estimated_write_per_second.scale(
                    _write_amplification(args)
                )
            )

            item_count = relaxed.data_shape.estimated_state_item_count
            if item_count is None:
                # assume 1 KB items
                if (
                    user_desires.query_pattern.estimated_mean_write_size_bytes
                    is not None
                ):
                    item_size_gib = (
                        user_desires.query_pattern.estimated_mean_write_size_bytes.mid
                        / 1024**3
                    )
                else:
                    item_size_gib = 1 / 1024**2  # type: ignore[unreachable]
                item_count = user_desires.data_shape.estimated_state_size_gib.scale(
                    1 / item_size_gib
                )
            # item_count is the number of *logical* nodes/edges. Each one fans
            # out into _write_amplification() backend KV items (2 links + one
            # item per edge property; 1 metadata item + one per node property),
            # and every item written is an item stored. Size each KV item at the
            # ~512 B needed to track its id and metadata write_ts.
            relaxed.data_shape.estimated_state_size_gib = item_count.scale(
                _write_amplification(args) * 512 / 1024**3
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

"""EXPERIMENTAL: capacity models that support instance fallback plans.

Most entries build a :class:`NodeCountAdapter` from the model's own size
limits.
"""

from typing import Any
from typing import Dict

from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.models.instance_fallback import (
    InstanceFallbackAdapterFactory,
)
from service_capacity_modeling.models.instance_fallback import NodeCountAdapter
from service_capacity_modeling.models.org.netflix.cassandra import (
    CassandraClusterSizeMode,
    NflxCassandraArguments,
    NflxCassandraCapacityModel,
    _get_cluster_size_lambda,
    _get_current_cluster_size,
)
from service_capacity_modeling.models.org.netflix.read_only_kv_instance_fallback import (  # noqa: E501
    ReadOnlyKVInstanceFallbackAdapter,
)


def _cassandra(
    candidates: Dict[str, CapacityPlan],
    desires: CapacityDesires,
    extra_model_arguments: Dict[str, Any],
) -> NodeCountAdapter:
    args = NflxCassandraArguments.from_extra_model_arguments(extra_model_arguments)
    required = NflxCassandraCapacityModel.get_required_cluster_size(
        desires.service_tier, extra_model_arguments
    )
    round_count = _get_cluster_size_lambda(
        args.cluster_size_mode or CassandraClusterSizeMode.unrestricted,
        _get_current_cluster_size(desires),
        required,
    )
    return NodeCountAdapter(
        candidates,
        desires,
        extra_model_arguments,
        cluster_type=NflxCassandraCapacityModel.cluster_type,
        max_regional_size=args.max_regional_size,
        # One Cassandra configuration per zone.
        shared_params=("cassandra.heap.gib", "cassandra.compaction.min_threshold"),
        allowed_count=lambda count: round_count(count) == count
        and (required is None or count == required),
    )


def _evcache(
    candidates: Dict[str, CapacityPlan],
    desires: CapacityDesires,
    extra_model_arguments: Dict[str, Any],
) -> NodeCountAdapter:
    return NodeCountAdapter(
        candidates,
        desires,
        extra_model_arguments,
        cluster_type="evcache",
        max_regional_size=int(extra_model_arguments.get("max_regional_size", 10000)),
    )


def _elasticsearch_data(
    candidates: Dict[str, CapacityPlan],
    desires: CapacityDesires,
    extra_model_arguments: Dict[str, Any],
) -> NodeCountAdapter:
    return NodeCountAdapter(
        candidates,
        desires,
        extra_model_arguments,
        cluster_type="elasticsearch-data",
        max_regional_size=int(extra_model_arguments.get("max_regional_size", 120)),
    )


INSTANCE_FALLBACK_ADAPTERS: Dict[str, InstanceFallbackAdapterFactory] = {
    "org.netflix.cassandra": _cassandra,
    "org.netflix.evcache": _evcache,
    # Data nodes only: master and search nodes are separate clusters.
    "org.netflix.elasticsearch.node": _elasticsearch_data,
    # A node count does not describe a read-only KV layout; see its module.
    "org.netflix.read-only-kv": ReadOnlyKVInstanceFallbackAdapter,
}

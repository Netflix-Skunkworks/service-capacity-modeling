"""EXPERIMENTAL: capacity models that support instance fallback plans."""

from typing import Dict

from service_capacity_modeling.models.instance_fallback import (
    InstanceFallbackAdapterFactory,
)
from service_capacity_modeling.models.org.netflix.cassandra_instance_fallback import (
    CassandraInstanceFallbackAdapter,
)
from service_capacity_modeling.models.org.netflix.read_only_kv_instance_fallback import (  # noqa: E501
    ReadOnlyKVInstanceFallbackAdapter,
)

INSTANCE_FALLBACK_ADAPTERS: Dict[str, InstanceFallbackAdapterFactory] = {
    "org.netflix.cassandra": CassandraInstanceFallbackAdapter,
    "org.netflix.read-only-kv": ReadOnlyKVInstanceFallbackAdapter,
}

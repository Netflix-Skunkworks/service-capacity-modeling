"""EXPERIMENTAL: capacity models that support interchangeable capacity plans."""

from typing import Dict

from service_capacity_modeling.models.interchangeable import (
    InterchangeableModelFactory,
)
from service_capacity_modeling.models.org.netflix.cassandra_interchangeable import (
    CassandraInterchangeable,
)
from service_capacity_modeling.models.org.netflix.read_only_kv_interchangeable import (
    ReadOnlyKVInterchangeable,
)

INTERCHANGEABLE_MODELS: Dict[str, InterchangeableModelFactory] = {
    "org.netflix.cassandra": CassandraInterchangeable,
    "org.netflix.read-only-kv": ReadOnlyKVInterchangeable,
}

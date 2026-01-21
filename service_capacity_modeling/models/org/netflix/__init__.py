from typing import Any
from typing import Dict

from .aurora import nflx_aurora_capacity_model
from .cassandra import nflx_cassandra_capacity_model
from .control import nflx_control_capacity_model
from .counter import nflx_counter_capacity_model
from .crdb import nflx_cockroachdb_capacity_model
from .ddb import nflx_ddb_capacity_model
from .elasticsearch import nflx_elasticsearch_capacity_model
from .elasticsearch import nflx_elasticsearch_data_capacity_model
from .elasticsearch import nflx_elasticsearch_master_capacity_model
from .elasticsearch import nflx_elasticsearch_search_capacity_model
from .entity import nflx_entity_capacity_model
from .evcache import nflx_evcache_capacity_model
from .graphkv import nflx_graphkv_capacity_model
from .kafka import nflx_kafka_capacity_model
from .key_value import nflx_key_value_capacity_model
from .postgres import nflx_postgres_capacity_model
from .rds import nflx_rds_capacity_model
from .stateless_java import nflx_java_app_capacity_model
from .time_series import nflx_time_series_capacity_model
from .wal import nflx_wal_capacity_model
from .zookeeper import nflx_zookeeper_capacity_model


def models() -> Dict[str, Any]:
    return {
        "org.netflix.cassandra": nflx_cassandra_capacity_model,
        "org.netflix.stateless-java": nflx_java_app_capacity_model,
        "org.netflix.key-value": nflx_key_value_capacity_model,
        "org.netflix.time-series": nflx_time_series_capacity_model,
        "org.netflix.counter": nflx_counter_capacity_model,
        "org.netflix.zookeeper": nflx_zookeeper_capacity_model,
        "org.netflix.evcache": nflx_evcache_capacity_model,
        "org.netflix.elasticsearch": nflx_elasticsearch_capacity_model,
        "org.netflix.elasticsearch.node": nflx_elasticsearch_data_capacity_model,
        "org.netflix.elasticsearch.master": nflx_elasticsearch_master_capacity_model,
        "org.netflix.elasticsearch.search": nflx_elasticsearch_search_capacity_model,
        "org.netflix.entity": nflx_entity_capacity_model,
        "org.netflix.control": nflx_control_capacity_model,
        "org.netflix.cockroachdb": nflx_cockroachdb_capacity_model,
        "org.netflix.aurora": nflx_aurora_capacity_model,
        "org.netflix.postgres": nflx_postgres_capacity_model,
        "org.netflix.rds": nflx_rds_capacity_model,
        "org.netflix.kafka": nflx_kafka_capacity_model,
        "org.netflix.dynamodb": nflx_ddb_capacity_model,
        "org.netflix.wal": nflx_wal_capacity_model,
        "org.netflix.graphkv": nflx_graphkv_capacity_model,
    }

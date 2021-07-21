from .cassandra import nflx_cassandra_capacity_model
from .elasticsearch import nflx_elasticsearch_capacity_model
from .evcache import nflx_evcache_capacity_model
from .key_value import nflx_key_value_capacity_model
from .time_series import nflx_time_series_capacity_model
from .rds import nflx_rds_capacity_model
from .stateless_java import nflx_java_app_capacity_model
from .zookeeper import nflx_zookeeper_capacity_model
from .entity import nflx_entity_capacity_model


def models():
    return {
        "org.netflix.cassandra": nflx_cassandra_capacity_model,
        "org.netflix.stateless-java": nflx_java_app_capacity_model,
        "org.netflix.key-value": nflx_key_value_capacity_model,
        "org.netflix.time-series": nflx_time_series_capacity_model,
        "org.netflix.zookeeper": nflx_zookeeper_capacity_model,
        "org.netflix.evcache": nflx_evcache_capacity_model,
        "org.netflix.rds": nflx_rds_capacity_model,
        "org.netflix.elasticsearch": nflx_elasticsearch_capacity_model,
        "org.netflix.entity": nflx_entity_capacity_model,
    }

from .cassandra import NflxCassandraCapacityModel
from .key_value import NflxKeyValueCapacityModel
from .stateless_java import NflxJavaAppCapacityModel


def models():
    return {
        "org.netflix.cassandra": NflxCassandraCapacityModel(),
        "org.netflix.stateless-java": NflxJavaAppCapacityModel(),
        "org.netflix.key-value": NflxKeyValueCapacityModel(),
    }

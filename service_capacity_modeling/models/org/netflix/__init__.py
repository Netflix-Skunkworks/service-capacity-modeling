from .cassandra import NflxCassandraCapacityModel
from .stateless_java import NflxJavaAppCapacityModel


def models():
    return {
        "org.netflix.cassandra": NflxCassandraCapacityModel(),
        "org.netflix.stateless_java": NflxJavaAppCapacityModel(),
    }

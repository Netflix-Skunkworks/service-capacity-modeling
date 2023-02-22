from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern


user_desires = CapacityDesires(
    service_tier=0,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(100000),
        estimated_write_per_second=certain_int(100000),
        estimated_mean_read_size_bytes=Interval(
            low=10, mid=100, high=1000, confidence=0.98
        ),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(10),
    ),
)


def test_cassandra_merge():
    cass_defaults = planner.models["org.netflix.cassandra"].default_desires(
        user_desires, {}
    )
    merged = user_desires.merge_with(cass_defaults)

    # Should come from the user
    assert merged.service_tier == 0
    assert merged.query_pattern.estimated_read_per_second.mid == 100000
    assert merged.query_pattern.estimated_mean_read_size_bytes.low == 10
    assert merged.data_shape.estimated_state_size_gib.mid == 10

    # Should come from cassandra
    assert merged.query_pattern.estimated_mean_read_latency_ms.mid == 2.0
    assert merged.query_pattern.estimated_mean_write_latency_ms.mid == 1.0

    # Should come from overall defaults
    assert merged.core_reference_ghz == 2.3

    # Should come from the default size producing the count
    # 10 GiB / 512 byte items = 20971520 items
    assert merged.data_shape.estimated_state_item_count is not None
    assert int(merged.data_shape.estimated_state_item_count.mid) == 41943040

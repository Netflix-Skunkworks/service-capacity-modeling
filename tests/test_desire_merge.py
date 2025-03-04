from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import Buffer
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import BufferIntent
from service_capacity_modeling.interface import Buffers
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.common import buffer_for_components

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
    buffers=Buffers(
        desired={
            "custom": Buffer(ratio=3.8, components=["custom"]),
            "custom-cpu": Buffer(ratio=3.0, components=[BufferComponent.cpu]),
        },
        derived={
            "compute": Buffer(
                intent=BufferIntent.scale, ratio=2, components=["compute"]
            )
        },
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
    assert merged.buffers.derived.get("compute") is not None
    assert merged.buffers.derived["compute"].ratio == 2.0
    assert merged.buffers.derived["compute"].intent == BufferIntent.scale

    # Should come from cassandra model
    assert merged.query_pattern.estimated_mean_read_latency_ms.mid == 2.0
    assert merged.query_pattern.estimated_mean_write_latency_ms.mid == 1.0

    # Should come from overall defaults
    assert merged.reference_shape.cpu_ghz == 2.3

    # Should come from the default size producing the count
    # 10 GiB / 512 byte items = 20971520 items
    assert merged.data_shape.estimated_state_item_count is not None
    assert int(merged.data_shape.estimated_state_item_count.mid) == 41943040

    # Buffer tests
    # The custom component should just be itself
    assert (
        buffer_for_components(buffers=merged.buffers, components=["custom"]).ratio
        == 3.8
    )
    # The custom cpu buffer should multiply with the default 1.5 compute buffer
    # AND the 2.0 background buffer
    assert (
        buffer_for_components(
            buffers=merged.buffers, components=[BufferComponent.cpu]
        ).ratio
        == 3.0 * 1.5 * 2.0
    )
    # The network should just have the default 1.5 compute and 2.0 background
    assert (
        buffer_for_components(
            buffers=merged.buffers, components=[BufferComponent.network]
        ).ratio
        == 1.5 * 2.0
    )
    # Disk should just come from the default storage
    assert (
        buffer_for_components(
            buffers=merged.buffers, components=[BufferComponent.disk]
        ).ratio
        == 4.0
    )

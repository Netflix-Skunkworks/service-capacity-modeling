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


# Elasticsearch has submodels which the C* test doesn't cover
def test_es_merge():
    """Test that Elasticsearch parent desires get merged with sub-models correctly"""
    # Test input values
    USER_TIER = 1
    USER_RPS_READ = 50000
    USER_RPS_WRITE = 10000
    USER_SIZE_LOW = 100
    USER_SIZE_MID = 500
    USER_SIZE_HIGH = 2000
    USER_STATE_GIB = 500
    USER_BUFFER_RATIO = 2.5

    # Expected values from model defaults
    EXPECTED_READ_LATENCY_MS = 2.0
    EXPECTED_COMPRESSION_RATIO = 3.0
    EXPECTED_RESERVED_MEM_GIB = 1
    EXPECTED_DATA_NODE_BUFFER = 1.33
    EXPECTED_DEFAULT_BUFFER = 1.5

    # Create user desires for elasticsearch
    es_user_desires = CapacityDesires(
        service_tier=USER_TIER,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(USER_RPS_READ),
            estimated_write_per_second=certain_int(USER_RPS_WRITE),
            estimated_mean_read_size_bytes=Interval(
                low=USER_SIZE_LOW,
                mid=USER_SIZE_MID,
                high=USER_SIZE_HIGH,
                confidence=0.95,
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=certain_int(USER_STATE_GIB),
        ),
        buffers=Buffers(
            desired={
                "custom-es": Buffer(
                    ratio=USER_BUFFER_RATIO, components=[BufferComponent.cpu]
                ),
            },
        ),
    )

    # Step 1: Merge user desires with parent model defaults
    es_parent_model = planner.models["org.netflix.elasticsearch"]
    parent_defaults = es_parent_model.default_desires(es_user_desires, {})
    merged_parent = es_user_desires.merge_with(parent_defaults)

    # Step 2: Get sub-models via compose_with
    sub_model_specs = es_parent_model.compose_with(es_user_desires, {})

    # Should have 3 sub-models (data, master, search)
    assert len(sub_model_specs) == 3

    # Step 3: Test each sub-model merge
    expected_models = {
        "org.netflix.elasticsearch.node": False,
        "org.netflix.elasticsearch.master": False,
        "org.netflix.elasticsearch.search": False,
    }

    for model_name, modifier_func in sub_model_specs:
        assert model_name in expected_models, f"Unexpected sub-model: {model_name}"

        # Apply the modifier to get modified desires
        modified_desires = modifier_func(merged_parent)

        # Merge with sub-model defaults
        sub_model = planner.models[model_name]
        sub_defaults = sub_model.default_desires(modified_desires, {})
        merged_sub = modified_desires.merge_with(sub_defaults)

        # Verify user values propagated through the merge chain for ALL sub-models
        assert merged_sub.service_tier == USER_TIER
        assert merged_sub.query_pattern.estimated_read_per_second.mid == USER_RPS_READ
        assert merged_sub.query_pattern.estimated_write_per_second.mid == USER_RPS_WRITE
        assert merged_sub.data_shape.estimated_state_size_gib.mid == USER_STATE_GIB

        # Verify user's custom buffer survived all the merges
        assert merged_sub.buffers.desired.get("custom-es") is not None
        assert (
            buffer_for_components(merged_sub.buffers, [BufferComponent.cpu]).ratio
            == USER_BUFFER_RATIO
        )

        # Verify parent model defaults were applied (from org.netflix.elasticsearch)
        assert (
            merged_sub.query_pattern.estimated_mean_read_latency_ms.mid
            == EXPECTED_READ_LATENCY_MS
        )
        assert (
            merged_sub.data_shape.estimated_compression_ratio.mid
            == EXPECTED_COMPRESSION_RATIO
        )
        assert (
            merged_sub.data_shape.reserved_instance_app_mem_gib
            == EXPECTED_RESERVED_MEM_GIB
        )

        # Verify user read size bytes overrode both parent and sub-model defaults
        assert (
            merged_sub.query_pattern.estimated_mean_read_size_bytes.mid == USER_SIZE_MID
        )

        # Model-specific buffer assertions using buffer_for_components
        if model_name == "org.netflix.elasticsearch.node":
            # Data node has specific buffer defaults (1.33 default buffer)
            # The default buffer affects all components
            assert (
                buffer_for_components(
                    merged_sub.buffers, [BufferComponent.network]
                ).ratio
                == EXPECTED_DATA_NODE_BUFFER
            )
        else:
            # Other nodes should have the standard default buffer (1.5)
            assert (
                buffer_for_components(
                    merged_sub.buffers, [BufferComponent.network]
                ).ratio
                == EXPECTED_DEFAULT_BUFFER
            )

        expected_models[model_name] = True  # Mark as seen

    # Verify all expected models were seen
    assert all(v for v in expected_models.values()), "Not all sub-models were tested"


def test_merge_with_missing_nested_structures():
    """Test merge behavior when query_pattern or data_shape are missing"""

    # Case 1: User provides query_pattern but the defaults don't
    user_with_query = CapacityDesires(
        service_tier=2,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(1000),
        ),
    )
    defaults_without_query = CapacityDesires(
        service_tier=1,
        data_shape=DataShape(
            estimated_state_size_gib=certain_int(100),
        ),
    )
    merged = user_with_query.merge_with(defaults_without_query)

    # User values should win
    assert merged.service_tier == 2
    assert merged.query_pattern.estimated_read_per_second.mid == 1000
    # Defaults should be present
    assert merged.data_shape.estimated_state_size_gib.mid == 100

    # Case 2: Defaults provide query_pattern, user doesn't
    user_without_query = CapacityDesires(
        service_tier=2,
        data_shape=DataShape(
            estimated_state_size_gib=certain_int(200),
        ),
    )
    defaults_with_query = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(5000),
        ),
    )
    merged = user_without_query.merge_with(defaults_with_query)

    # User values should win
    assert merged.service_tier == 2
    assert merged.data_shape.estimated_state_size_gib.mid == 200
    # Defaults should fill in missing values
    assert merged.query_pattern.estimated_read_per_second.mid == 5000

    # Case 3: Both provide nested structures with partial overlap
    user_partial = CapacityDesires(
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(3000),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=certain_int(50),
        ),
    )
    defaults_partial = CapacityDesires(
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(1000),
            estimated_write_per_second=certain_int(2000),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=certain_int(100),
            estimated_compression_ratio=Interval(low=2, mid=3, high=4),
        ),
    )
    merged = user_partial.merge_with(defaults_partial)

    # User values override defaults
    assert merged.query_pattern.estimated_read_per_second.mid == 3000
    assert merged.data_shape.estimated_state_size_gib.mid == 50
    # Defaults fill in missing fields
    assert merged.query_pattern.estimated_write_per_second.mid == 2000
    assert merged.data_shape.estimated_compression_ratio.mid == 3


def test_merge_with_missing_buffers():
    """Test merge behavior when buffers are partially or completely missing"""

    # Case 1: User has no buffers, defaults have buffers
    user_no_buffers = CapacityDesires(
        service_tier=1,
    )
    defaults_with_buffers = CapacityDesires(
        buffers=Buffers(
            default=Buffer(ratio=2.0),
            desired={
                "compute": Buffer(ratio=1.5, components=[BufferComponent.compute]),
            },
        ),
    )
    merged = user_no_buffers.merge_with(defaults_with_buffers)

    # Should get buffers from defaults
    assert buffer_for_components(merged.buffers, [BufferComponent.compute]).ratio == 1.5
    # Storage should get the default buffer
    assert buffer_for_components(merged.buffers, [BufferComponent.storage]).ratio == 2.0

    # Case 2: User has buffers, defaults have different buffers
    user_with_buffers = CapacityDesires(
        buffers=Buffers(
            desired={
                "custom": Buffer(ratio=3.0, components=[BufferComponent.compute]),
            },
        ),
    )
    defaults_with_buffers = CapacityDesires(
        buffers=Buffers(
            default=Buffer(ratio=2.0),
            desired={
                "compute": Buffer(ratio=1.5, components=[BufferComponent.compute]),
            },
        ),
    )
    merged = user_with_buffers.merge_with(defaults_with_buffers)

    # Should have both user and default buffers
    assert (
        buffer_for_components(merged.buffers, [BufferComponent.compute]).ratio
        == 1.5 * 3
    )  # From defaults
    assert (
        buffer_for_components(merged.buffers, [BufferComponent.storage]).ratio == 2.0
    )  # Default buffer

    # Case 3: User overrides a buffer that exists in defaults
    user_override = CapacityDesires(
        buffers=Buffers(
            desired={
                "compute": Buffer(ratio=5.0, components=[BufferComponent.compute]),
            },
        ),
    )
    merged = user_override.merge_with(defaults_with_buffers)
    assert buffer_for_components(merged.buffers, [BufferComponent.compute]).ratio == 5.0
    assert buffer_for_components(merged.buffers, [BufferComponent.storage]).ratio == 2.0

    # Case 4: Defaults do not have any explicit buffers
    default_without_buffers = CapacityDesires()
    merged = user_override.merge_with(default_without_buffers)

    # User value should win
    assert buffer_for_components(merged.buffers, [BufferComponent.compute]).ratio == 5.0
    assert buffer_for_components(merged.buffers, [BufferComponent.storage]).ratio == 1.5

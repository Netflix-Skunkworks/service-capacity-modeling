import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import Buffer
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import Buffers
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.common import buffer_for_components
from service_capacity_modeling.models.common import EFFECTIVE_DISK_PER_NODE_GIB
from service_capacity_modeling.models.org.netflix.cassandra import (
    CASSANDRA_DISK_UTILIZATION_LIMIT,
)
from service_capacity_modeling.models.org.netflix.cassandra import (
    CASSANDRA_MAX_ATTACHED_DATA_PER_NODE_GIB,
)
from service_capacity_modeling.models.org.netflix.cassandra import (
    _with_disk_utilization_buffer,
)
from service_capacity_modeling.models.org.netflix.cassandra import (
    NflxCassandraArguments,
)


def _ebs_desires(
    buffers: Buffers | None = None, state_gib: int = 20_000
) -> CapacityDesires:
    return CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(1_000),
            estimated_write_per_second=certain_int(1_000),
            estimated_mean_read_latency_ms=certain_float(1),
            estimated_mean_write_latency_ms=certain_float(1),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=certain_int(state_gib),
            estimated_compression_ratio=certain_float(1.0),
        ),
        buffers=buffers or Buffers(),
    )


def _plan_ebs(desires: CapacityDesires, **extra_model_arguments):
    return planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
        extra_model_arguments={
            "require_attached_disks": True,
            "require_local_disks": False,
            "copies_per_region": 3,
            "adaptive_storage_buffer": False,
            "cluster_size_mode": "unrestricted",
            **extra_model_arguments,
        },
    )[0].candidate_clusters.zonal[0]


def test_ebs_applies_attached_disk_buffer_multiplier():
    result = _plan_ebs(_ebs_desires(), max_storage_buffer_ratio=4.0)

    assert result.count == 14
    assert result.attached_drives[0].size_gib == 2900
    assert result.cluster_params[EFFECTIVE_DISK_PER_NODE_GIB] == 3100
    assert result.cluster_params["cassandra.storage_buffer_ratio"] == 2.0


def test_ebs_default_allows_1_5_tib_data_without_heating_disk_utilization():
    args = NflxCassandraArguments.from_extra_model_arguments({})
    result = _plan_ebs(
        _ebs_desires(state_gib=15_360),
        adaptive_storage_buffer=True,
    )

    assert args.max_attached_data_per_node_gib == 1536
    assert CASSANDRA_MAX_ATTACHED_DATA_PER_NODE_GIB == 1536
    assert result.count == 10
    assert result.cluster_params[EFFECTIVE_DISK_PER_NODE_GIB] == 2800
    assert result.cluster_params["cassandra.storage_buffer_ratio"] == pytest.approx(
        1 / 0.55,
        abs=0.01,
    )
    data_per_node_gib = 15_360 / result.count
    assert data_per_node_gib == CASSANDRA_MAX_ATTACHED_DATA_PER_NODE_GIB
    assert data_per_node_gib / result.attached_drives[0].size_gib < 0.55


def test_ebs_default_density_limit_is_strict():
    result = _plan_ebs(
        _ebs_desires(state_gib=15_361),
        adaptive_storage_buffer=True,
    )

    assert result.count == 11
    assert 15_361 / result.count < CASSANDRA_MAX_ATTACHED_DATA_PER_NODE_GIB


def test_ebs_multiplies_explicit_storage_buffer():
    result = _plan_ebs(
        _ebs_desires(
            Buffers(
                desired={
                    "storage": Buffer(
                        ratio=3.0,
                        components=[BufferComponent.storage],
                    )
                }
            )
        )
    )

    assert result.count == 14
    assert result.attached_drives[0].size_gib == 2600
    assert result.cluster_params[EFFECTIVE_DISK_PER_NODE_GIB] == 2800
    assert result.cluster_params["cassandra.storage_buffer_ratio"] == pytest.approx(
        1 / 0.55,
        abs=0.01,
    )


def test_ebs_hotter_buffer_respects_disk_utilization_cap_with_adaptive_storage():
    result = _plan_ebs(
        _ebs_desires(),
        adaptive_storage_buffer=True,
    )
    stricter = _plan_ebs(
        _ebs_desires(),
        adaptive_storage_buffer=True,
        max_disk_utilization=0.50,
    )

    assert result.cluster_params["cassandra.storage_buffer_ratio"] == pytest.approx(
        1 / 0.55,
        abs=0.01,
    )
    assert result.cluster_params[EFFECTIVE_DISK_PER_NODE_GIB] == 2800
    assert stricter.cluster_params["cassandra.storage_buffer_ratio"] == pytest.approx(
        1 / 0.50,
        abs=0.01,
    )
    assert stricter.cluster_params[EFFECTIVE_DISK_PER_NODE_GIB] == 3100


def test_disk_utilization_cap_does_not_weaken_default_buffer_fallback():
    original = _ebs_desires()
    desires = _with_disk_utilization_buffer(original, max_disk_utilization=0.55)

    disk_buffer = buffer_for_components(
        buffers=desires.buffers,
        components=[BufferComponent.disk],
    )

    assert disk_buffer.ratio == pytest.approx(1 / 0.55, abs=0.01)
    assert CASSANDRA_DISK_UTILIZATION_LIMIT not in original.buffers.desired


def test_max_disk_utilization_must_be_positive():
    with pytest.raises(ValueError, match="max_disk_utilization"):
        NflxCassandraArguments.from_extra_model_arguments({"max_disk_utilization": 0})

from pytest import approx

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import Buffer
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import Buffers
from service_capacity_modeling.models.common import cpu_headroom_target


def test_m5_same_headroom_as_r5():
    m5xl = planner.instance("m5.xlarge")
    r5xl = planner.instance("r5.xlarge")

    m5_headroom = cpu_headroom_target(m5xl)
    assert m5_headroom is not None
    r5_headroom = cpu_headroom_target(r5xl)
    assert r5_headroom is not None
    assert m5_headroom == r5_headroom


def test_larger_instance_needs_less_headroom():
    m5xl = cpu_headroom_target(planner.instance("m5.xlarge"))
    assert m5xl is not None
    m52xl = cpu_headroom_target(planner.instance("m5.2xlarge"))
    assert m52xl is not None
    assert m52xl < m5xl


def test_seventh_gen_instance_reflects_non_ht_boost():
    m6i4xl = cpu_headroom_target(planner.instance("m6i.4xlarge"))
    m7a4xl = cpu_headroom_target(planner.instance("m7a.4xlarge"))
    assert m7a4xl < m6i4xl
    assert m7a4xl == approx(0.16, rel=0.05)


def test_has_some_headroom():
    # For the purpose of headroom shapes and planner should both see the same
    # thing
    m5_instance = shapes.instance("m5.xlarge")
    r5_instance = planner.instance("r5.xlarge")
    assert m5_instance.cpu == 4
    m5_headroom = cpu_headroom_target(m5_instance)
    r5_headroom = cpu_headroom_target(r5_instance)
    assert m5_headroom is not None
    assert r5_headroom is not None
    assert m5_headroom > 0
    assert m5_headroom == r5_headroom

    assert 0.3 < r5_headroom < 0.5


def test_2xl_headroom_with_buffer():
    # TODO (jolynch) I think these numbers are probably just a little too
    #  aggressive with small instances, but maybe that's right ... let's see
    m6i2xl = planner.instance("m6i.2xlarge")

    headroom = cpu_headroom_target(m6i2xl)
    assert headroom == approx(0.28, rel=0.05)

    buffer_1_5x = Buffers(desired={BufferComponent.compute: 1.5})
    headroom_with_1_5x_buffer = cpu_headroom_target(m6i2xl, buffers=buffer_1_5x)
    assert headroom_with_1_5x_buffer > headroom
    assert headroom_with_1_5x_buffer == approx(0.52, rel=0.05)

    buffer_2_0x = Buffers(desired={BufferComponent.compute: 2})
    headroom_with_2_0x_buffer = cpu_headroom_target(m6i2xl, buffers=buffer_2_0x)
    assert headroom_with_2_0x_buffer > headroom_with_1_5x_buffer > headroom
    assert headroom_with_2_0x_buffer == approx(0.64, rel=0.05)


def test_2xl_headroom_with_buffer_no_hyperthreading():
    buffer_2x = Buffers(desired={BufferComponent.compute: 2})
    m7a_2xl = planner.instance("m7a.2xlarge")
    effective_headroom = cpu_headroom_target(m7a_2xl, buffers=buffer_2x)

    # Seventh gen cpus are cores not threads, should be able to run with < 2x buffer

    assert cpu_headroom_target(m7a_2xl) == approx(0.22, rel=0.05)
    assert cpu_headroom_target(m7a_2xl) < effective_headroom


def test_default_buffer():
    buffer = Buffers(default=4)
    for component in BufferComponent:
        assert buffer.buffer_for_component(component) == Buffer(
            component=component, source="default", ratio=4
        )


def test_common_buffer_fallbacks():
    buffers = Buffers(desired={"compute": 2.0, "storage": 3.2})

    assert buffers.buffer_for_component(BufferComponent.cpu) == Buffer(
        component="cpu", ratio=2.0, source="compute"
    )
    assert buffers.buffer_for_component(BufferComponent.network) == Buffer(
        component="network", ratio=2.0, source="compute"
    )
    assert buffers.buffer_for_component(BufferComponent.disk) == Buffer(
        component="disk", ratio=3.2, source="storage"
    )
    assert buffers.buffer_for_component(BufferComponent.memory) == Buffer(
        component="memory", ratio=3.2, source="storage"
    )


def test_precise_buffers():
    buffers = Buffers(
        desired={
            BufferComponent.cpu: 2.0,
            BufferComponent.network: 2.5,
            BufferComponent.disk: 3.0,
            BufferComponent.memory: 3.5,
            BufferComponent.memory_traffic: 4.0,
            BufferComponent.memory_read_cache: 4.5,
            BufferComponent.memory_write_cache: 5.0,
        }
    )

    assert buffers.buffer_for_component(BufferComponent.cpu) == Buffer(
        component="cpu", ratio=2.0
    )
    assert buffers.buffer_for_component(BufferComponent.network) == Buffer(
        component="network", ratio=2.5
    )
    assert buffers.buffer_for_component(BufferComponent.disk) == Buffer(
        component="disk", ratio=3.0
    )
    assert buffers.buffer_for_component(BufferComponent.memory) == Buffer(
        component="memory", ratio=3.5
    )
    assert buffers.buffer_for_component(BufferComponent.memory_traffic) == Buffer(
        component="memory-traffic", ratio=4.0
    )
    assert buffers.buffer_for_component(BufferComponent.memory_read_cache) == Buffer(
        component="memory-read-cache", ratio=4.5
    )
    assert buffers.buffer_for_component(BufferComponent.memory_write_cache) == Buffer(
        component="memory-write-cache", ratio=5.0
    )

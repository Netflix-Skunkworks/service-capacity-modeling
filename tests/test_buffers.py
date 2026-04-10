import pytest
from pydantic import ValidationError
from pytest import approx

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import Buffer
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import BufferIntent
from service_capacity_modeling.interface import Buffers
from service_capacity_modeling.models.common import buffer_for_components
from service_capacity_modeling.models.common import cpu_headroom_target
from service_capacity_modeling.models.common import DerivedBuffers


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

    buffer_1_5x = Buffers(desired={BufferComponent.compute: Buffer(ratio=1.5)})
    headroom_with_1_5x_buffer = cpu_headroom_target(m6i2xl, buffers=buffer_1_5x)
    assert headroom_with_1_5x_buffer > headroom
    assert headroom_with_1_5x_buffer == approx(0.52, rel=0.05)

    buffer_2_0x = Buffers(desired={BufferComponent.compute: Buffer(ratio=2.0)})
    headroom_with_2_0x_buffer = cpu_headroom_target(m6i2xl, buffers=buffer_2_0x)
    assert headroom_with_2_0x_buffer > headroom_with_1_5x_buffer > headroom
    assert headroom_with_2_0x_buffer == approx(0.64, rel=0.05)


def test_2xl_headroom_with_buffer_no_hyperthreading():
    buffer_2x = Buffers(desired={BufferComponent.compute: Buffer(ratio=2.0)})
    m7a_2xl = planner.instance("m7a.2xlarge")
    effective_headroom = cpu_headroom_target(m7a_2xl, buffers=buffer_2x)

    # Seventh gen cpus are cores not threads, should be able to run with < 2x buffer

    assert cpu_headroom_target(m7a_2xl) == approx(0.22, rel=0.05)
    assert cpu_headroom_target(m7a_2xl) < effective_headroom


def test_default_buffer():
    buffers = Buffers(default=Buffer(ratio=4, sources={"default": Buffer(ratio=4)}))
    for component in BufferComponent:
        if BufferComponent.is_generic(component):
            continue
        result = buffer_for_components(buffers, [component])
        assert result.ratio == 4
        assert component in result.components
        assert not result.sources


def test_common_buffer_fallbacks():
    buffers = Buffers(
        desired={
            "compute": Buffer(ratio=2.0, components=[BufferComponent.compute]),
            "storage": Buffer(ratio=3.2, components=[BufferComponent.storage]),
        }
    )

    assert buffer_for_components(buffers, [BufferComponent.cpu]) == Buffer(
        ratio=2.0,
        components=["compute", "cpu"],
        sources={"compute": buffers.desired["compute"]},
    )

    assert buffer_for_components(buffers, [BufferComponent.network]) == Buffer(
        ratio=2.0,
        components=["compute", "network"],
        sources={"compute": buffers.desired["compute"]},
    )

    assert buffer_for_components(buffers, [BufferComponent.disk]) == Buffer(
        ratio=3.2,
        components=["disk", "storage"],
        sources={"storage": buffers.desired["storage"]},
    )

    assert buffer_for_components(buffers, [BufferComponent.memory]) == Buffer(
        ratio=3.2,
        components=["memory", "storage"],
        sources={"storage": buffers.desired["storage"]},
    )


def test_precise_buffers():
    buffers = Buffers(
        desired={
            "custom-cpu": Buffer(ratio=2.0, components=[BufferComponent.cpu]),
            "custom-network": Buffer(ratio=2.5, components=[BufferComponent.network]),
            "custom-disk": Buffer(ratio=3.0, components=[BufferComponent.disk]),
            "custom-memory": Buffer(ratio=3.5, components=[BufferComponent.memory]),
        }
    )

    assert buffer_for_components(buffers, [BufferComponent.cpu]) == Buffer(
        ratio=2.0,
        components=["compute", "cpu"],
        sources={"custom-cpu": buffers.desired["custom-cpu"]},
    )
    assert buffer_for_components(buffers, [BufferComponent.network]) == Buffer(
        ratio=2.5,
        components=["compute", "network"],
        sources={"custom-network": buffers.desired["custom-network"]},
    )
    assert buffer_for_components(buffers, [BufferComponent.disk]) == Buffer(
        ratio=3.0,
        components=["disk", "storage"],
        sources={"custom-disk": buffers.desired["custom-disk"]},
    )
    assert buffer_for_components(buffers, [BufferComponent.memory]) == Buffer(
        ratio=3.5,
        components=["memory", "storage"],
        sources={"custom-memory": buffers.desired["custom-memory"]},
    )


def test_compute_cpu_and_network_multiply():
    buffers = Buffers(
        desired={
            "compute": Buffer(ratio=3, components=[BufferComponent.compute]),
            "cpu": Buffer(ratio=5, components=[BufferComponent.cpu]),
            "network": Buffer(ratio=7, components=[BufferComponent.network]),
        }
    )

    assert buffer_for_components(buffers, [BufferComponent.cpu]) == Buffer(
        ratio=15,
        components=["compute", "cpu"],
        sources={
            "compute": buffers.desired["compute"],
            "cpu": buffers.desired["cpu"],
        },
    )

    assert buffer_for_components(buffers, [BufferComponent.network]) == Buffer(
        ratio=21,
        components=["compute", "network"],
        sources={
            "compute": buffers.desired["compute"],
            "network": buffers.desired["network"],
        },
    )

    # Querying for generic component should raise error
    with pytest.raises(ValueError):
        buffer_for_components(buffers, [BufferComponent.compute])


def test_storage_disk_memory_multiply():
    buffers = Buffers(
        desired={
            "storage": Buffer(ratio=2, components=[BufferComponent.storage]),
            "disk": Buffer(ratio=3, components=[BufferComponent.disk]),
            "memory": Buffer(ratio=5, components=[BufferComponent.memory]),
        }
    )

    assert buffer_for_components(buffers, [BufferComponent.disk]) == Buffer(
        ratio=6,
        components=["disk", "storage"],
        sources={
            "storage": buffers.desired["storage"],
            "disk": buffers.desired["disk"],
        },
    )

    assert buffer_for_components(buffers, [BufferComponent.memory]) == Buffer(
        ratio=10,
        components=["memory", "storage"],
        sources={
            "storage": buffers.desired["storage"],
            "memory": buffers.desired["memory"],
        },
    )

    # Querying for generic component should raise error
    with pytest.raises(ValueError):
        buffer_for_components(buffers, [BufferComponent.storage])


def test_component_validation():
    """Test that buffer_for_components rejects generic components"""
    buffers = Buffers(
        desired={
            "cpu": Buffer(ratio=2.0, components=[BufferComponent.cpu]),
            "compute": Buffer(ratio=3.0, components=[BufferComponent.compute]),
            "storage": Buffer(ratio=4.0, components=[BufferComponent.storage]),
        }
    )

    # Specific components should work fine
    buffer_for_components(buffers, [BufferComponent.cpu])
    buffer_for_components(buffers, [BufferComponent.disk])
    buffer_for_components(buffers, [BufferComponent.memory])
    buffer_for_components(buffers, [BufferComponent.network])

    # Generic components should raise ValueError
    with pytest.raises(ValueError):
        buffer_for_components(buffers, [BufferComponent.compute])

    with pytest.raises(ValueError):
        buffer_for_components(buffers, [BufferComponent.storage])

    # Multiple generic components should also raise ValueError
    with pytest.raises(ValueError):
        buffer_for_components(
            buffers, [BufferComponent.compute, BufferComponent.storage]
        )


def test_derived_buffer_validation():
    """Test that DerivedBuffers.for_components rejects generic components"""
    derived_buffers = {
        "cpu": Buffer(ratio=2.0, components=[BufferComponent.cpu]),
        "compute": Buffer(ratio=3.0, components=[BufferComponent.compute]),
    }

    # Specific components should work fine
    DerivedBuffers.for_components(derived_buffers, [BufferComponent.cpu])
    DerivedBuffers.for_components(derived_buffers, [BufferComponent.disk])
    DerivedBuffers.for_components(derived_buffers, [BufferComponent.memory])
    DerivedBuffers.for_components(derived_buffers, [BufferComponent.network])

    # Generic components should raise ValueError
    with pytest.raises(ValueError):
        DerivedBuffers.for_components(derived_buffers, [BufferComponent.compute])

    with pytest.raises(ValueError):
        DerivedBuffers.for_components(derived_buffers, [BufferComponent.storage])


# ──────────────────────────────────────────────────────────────────
# Buffer floor/ceiling validation
# ──────────────────────────────────────────────────────────────────


def test_buffer_floor_ceiling_validation():
    """floor and ceiling must be positive and floor <= ceiling"""
    Buffer(ratio=1.5, floor=1.0)
    Buffer(ratio=1.5, ceiling=2.0)
    Buffer(ratio=1.5, floor=0.5, ceiling=1.0)

    with pytest.raises(ValidationError):
        Buffer(floor=0)
    with pytest.raises(ValidationError):
        Buffer(ceiling=-1)
    with pytest.raises(ValidationError):
        Buffer(floor=2.0, ceiling=1.0)


def test_desired_buffer_rejects_floor_ceiling():
    """Desired buffers must not set floor or ceiling"""
    with pytest.raises(ValidationError):
        Buffers(desired={"mem": Buffer(ratio=1.5, floor=1.0, components=["memory"])})
    with pytest.raises(ValidationError):
        Buffers(desired={"mem": Buffer(ratio=1.5, ceiling=1.0, components=["memory"])})


def test_derived_mixed_intent_and_bounds_rejected():
    """Cannot mix legacy intent sugar with explicit floor/ceiling"""
    with pytest.raises(ValidationError):
        Buffers(
            derived={
                "mem": Buffer(
                    intent=BufferIntent.scale_up,
                    ratio=1.5,
                    floor=1.0,
                    components=["memory"],
                )
            }
        )
    with pytest.raises(ValidationError):
        Buffers(
            derived={
                "disk": Buffer(
                    intent=BufferIntent.scale_down,
                    ratio=0.8,
                    ceiling=1.0,
                    components=["disk"],
                )
            }
        )


def test_preserve_with_non_unit_ratio_rejected():
    """preserve intent requires ratio=1.0"""
    with pytest.raises(ValidationError):
        Buffers(
            derived={
                "mem": Buffer(
                    intent=BufferIntent.preserve,
                    ratio=2.0,
                    components=["memory"],
                )
            }
        )


# ──────────────────────────────────────────────────────────────────
# DerivedBuffers normalization from legacy intents
# ──────────────────────────────────────────────────────────────────


def test_preserve_normalizes_to_bounds():
    """preserve intent normalizes to scale=1, floor=1, ceiling=1"""
    derived = {
        "mem": Buffer(intent=BufferIntent.preserve, components=[BufferComponent.memory])
    }
    db = DerivedBuffers.for_components(derived, [BufferComponent.memory])
    assert db.scale == 1
    assert db.floor == 1
    assert db.ceiling == 1
    assert db.is_preserve


def test_scale_up_normalizes_floor():
    """scale_up intent normalizes to floor=1, no ceiling"""
    derived = {
        "cpu": Buffer(
            intent=BufferIntent.scale_up,
            ratio=1.5,
            components=[BufferComponent.cpu],
        )
    }
    db = DerivedBuffers.for_components(derived, [BufferComponent.cpu])
    assert db.scale == 1.5
    assert db.floor == 1
    assert db.ceiling is None
    assert not db.is_preserve


def test_scale_down_normalizes_ceiling():
    """scale_down intent normalizes to ceiling=1, no floor"""
    derived = {
        "disk": Buffer(
            intent=BufferIntent.scale_down,
            ratio=0.8,
            components=[BufferComponent.disk],
        )
    }
    db = DerivedBuffers.for_components(derived, [BufferComponent.disk])
    assert db.scale == 0.8
    assert db.ceiling == 1
    assert db.floor is None
    assert not db.is_preserve


def test_scale_normalizes_no_bounds():
    """plain scale intent: just scale, no floor/ceiling"""
    derived = {
        "net": Buffer(
            intent=BufferIntent.scale,
            ratio=2.0,
            components=[BufferComponent.network],
        )
    }
    db = DerivedBuffers.for_components(derived, [BufferComponent.network])
    assert db.scale == 2.0
    assert db.floor is None
    assert db.ceiling is None


# ──────────────────────────────────────────────────────────────────
# DerivedBuffers normalization from new-style floor/ceiling
# ──────────────────────────────────────────────────────────────────


def test_new_style_floor():
    """New-style Buffer with explicit floor resolves correctly"""
    derived = {"mem": Buffer(ratio=1.5, floor=1.0, components=[BufferComponent.memory])}
    db = DerivedBuffers.for_components(derived, [BufferComponent.memory])
    assert db.scale == 1.0  # no intent → no scale contribution
    assert db.floor == 1.0
    assert db.ceiling is None


def test_new_style_ceiling():
    """New-style Buffer with explicit ceiling resolves correctly"""
    derived = {
        "disk": Buffer(ratio=0.8, ceiling=1.0, components=[BufferComponent.disk])
    }
    db = DerivedBuffers.for_components(derived, [BufferComponent.disk])
    assert db.scale == 1.0
    assert db.ceiling == 1.0
    assert db.floor is None


def test_new_style_floor_and_ceiling():
    """New-style Buffer with both floor and ceiling"""
    derived = {
        "mem": Buffer(
            ratio=1.0, floor=0.8, ceiling=1.2, components=[BufferComponent.memory]
        )
    }
    db = DerivedBuffers.for_components(derived, [BufferComponent.memory])
    assert db.floor == 0.8
    assert db.ceiling == 1.2


# ──────────────────────────────────────────────────────────────────
# DerivedBuffers combination semantics
# ──────────────────────────────────────────────────────────────────


def test_multiple_floors_merge_max():
    """Multiple buffers with floors → take the max"""
    derived = {
        "a": Buffer(floor=0.5, components=[BufferComponent.memory]),
        "b": Buffer(floor=0.8, components=[BufferComponent.memory]),
    }
    db = DerivedBuffers.for_components(derived, [BufferComponent.memory])
    assert db.floor == 0.8


def test_multiple_ceilings_merge_min():
    """Multiple buffers with ceilings → take the min"""
    derived = {
        "a": Buffer(ceiling=1.5, components=[BufferComponent.memory]),
        "b": Buffer(ceiling=1.2, components=[BufferComponent.memory]),
    }
    db = DerivedBuffers.for_components(derived, [BufferComponent.memory])
    assert db.ceiling == 1.2


def test_merged_floor_exceeds_ceiling_rejected():
    """Merged policy where floor > ceiling raises ValueError"""
    derived = {
        "a": Buffer(floor=2.0, components=[BufferComponent.memory]),
        "b": Buffer(ceiling=1.0, components=[BufferComponent.memory]),
    }
    with pytest.raises(ValueError, match="floor.*ceiling"):
        DerivedBuffers.for_components(derived, [BufferComponent.memory])


# ──────────────────────────────────────────────────────────────────
# calculate_requirement with normalized policies
# ──────────────────────────────────────────────────────────────────


def test_preserve_calculate_requirement():
    """Preserve-normalized policy returns existing_capacity"""
    db = DerivedBuffers(scale=1, floor=1, ceiling=1)
    assert db.is_preserve
    # Regardless of current_usage or desired_buffer_ratio, result = existing
    assert db.calculate_requirement(50, 100) == 100
    assert db.calculate_requirement(150, 100) == 100
    assert db.calculate_requirement(50, 100, desired_buffer_ratio=2.0) == 100


def test_scale_up_calculate_requirement():
    """scale_up: floor=1 means never drop below existing capacity"""
    db = DerivedBuffers(scale=1.5, floor=1)
    # Under-provisioned: need 150, have 100 → scale up to 150
    assert db.calculate_requirement(100, 100) == 150
    # Over-provisioned: need 30, have 100 → floor pins to 100
    assert db.calculate_requirement(20, 100) == 100


def test_scale_down_calculate_requirement():
    """scale_down: ceiling=1 means never exceed existing capacity"""
    db = DerivedBuffers(scale=0.8, ceiling=1)
    # Over-provisioned: need 80, have 100 → scale down to 80
    assert db.calculate_requirement(100, 100) == 80
    # Under-provisioned: need 160, have 100 → ceiling caps to 100
    assert db.calculate_requirement(200, 100) == 100


def test_plain_scale_calculate_requirement():
    """Plain scale with no bounds: just multiply"""
    db = DerivedBuffers(scale=2.0)
    assert db.calculate_requirement(50, 100) == 100
    assert db.calculate_requirement(50, 100, desired_buffer_ratio=1.5) == 150

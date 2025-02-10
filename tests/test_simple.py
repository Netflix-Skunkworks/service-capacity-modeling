from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.simple import get_simple_instance_headroom_target
from service_capacity_modeling.simple import (
    get_simple_instance_headroom_target_for_name,
)


def test_invalid_instance_returns_none():
    assert get_simple_instance_headroom_target_for_name("invalid.instance") is None


def test_m5_same_headroom_as_r5():
    m5_headroom = get_simple_instance_headroom_target_for_name("m5.xlarge")
    assert m5_headroom is not None
    r5_headroom = get_simple_instance_headroom_target_for_name("r5.xlarge")
    assert r5_headroom is not None
    assert m5_headroom == r5_headroom


def test_larger_instance_needs_less_headroom():
    xlarge_headroom = get_simple_instance_headroom_target_for_name("m5.xlarge")
    assert xlarge_headroom is not None
    twoxlarge_headroom = get_simple_instance_headroom_target_for_name("m5.2xlarge")
    assert twoxlarge_headroom is not None
    assert twoxlarge_headroom < xlarge_headroom


def test_has_some_headroom():
    m5_instance = shapes.hardware.regions["us-east-1"].instances["m5.xlarge"]
    r5_instance = shapes.hardware.regions["us-east-1"].instances["r5.xlarge"]
    assert m5_instance.cpu == r5_instance.cpu
    assert m5_instance.cpu == 4
    m5_headroom = get_simple_instance_headroom_target(m5_instance)
    r5_headroom = get_simple_instance_headroom_target(m5_instance)
    assert m5_headroom is not None
    assert r5_headroom is not None
    assert m5_headroom > 0
    assert m5_headroom == r5_headroom

from service_capacity_modeling.simple import get_simple_instance_headroom_target

def test_invalid_instance_returns_none():
    assert get_simple_instance_headroom_target("invalid.instance") is None

def test_m5_same_headroom_as_r5():
    m5_headroom = get_simple_instance_headroom_target("m5.xlarge")
    assert m5_headroom is not None
    r5_headroom = get_simple_instance_headroom_target("r5.xlarge")
    assert r5_headroom is not None
    assert m5_headroom == r5_headroom

def test_larger_instance_needs_less_headroom():
    xlarge_headroom = get_simple_instance_headroom_target("m5.xlarge")
    assert xlarge_headroom is not None
    twoxlarge_headroom = get_simple_instance_headroom_target("m5.2xlarge") 
    assert twoxlarge_headroom is not None
    assert twoxlarge_headroom < xlarge_headroom

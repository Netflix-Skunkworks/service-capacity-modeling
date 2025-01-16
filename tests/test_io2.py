from service_capacity_modeling.hardware import shapes

io2 = shapes.hardware.regions["us-east-1"].drives["io2"]


def test_io2_pricing():
    d = io2.model_copy()
    d.size_gib = 100
    d.read_io_per_s = 33000
    assert d.annual_cost == 25662

    d.read_io_per_s = 31000
    d.write_io_per_s = 34000
    assert d.annual_cost == 50394

    d.read_io_per_s = 50000
    d.write_io_per_s = 34000
    assert d.annual_cost == 61110

    d.read_io_per_s = 75000
    d.write_io_per_s = 34000
    # (32000 * 0.78 + (64000-32000) * 0.552 + (75000 - 64000) * 0.384)
    # + ((32000 * 0.78) + (34000 - 32000) * 0.552)
    # 150
    assert d.annual_cost == 73062

    d.read_io_per_s = 256000
    d.write_io_per_s = 34000
    assert d.annual_cost == 142566

from service_capacity_modeling.hardware import shapes


def test_blob():
    s3 = shapes.region("us-east-1").services["blob.standard"]
    assert s3.annual_cost_per_gib > 0
    assert s3.annual_cost_per_write_io > 0
    assert s3.annual_cost_per_read_io > 0

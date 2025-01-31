from service_capacity_modeling.hardware import shapes


def test_blob():
    s3 = shapes.region("us-east-1").services["blob.standard"]
    assert s3.annual_cost_gib(1) > 0
    assert s3.annual_cost_per_write_io > 0
    assert s3.annual_cost_per_read_io > 0


def test_loaded_from_ec2_and_overrides():
    ec2 = shapes.region("us-east-1").instances["m5.large"]
    assert ec2.annual_cost > 0
    ec2 = shapes.region("us-east-1").instances["db.r5.large"]
    assert ec2.annual_cost > 0

    assert shapes.region("us-east-1").zones_in_region == 3

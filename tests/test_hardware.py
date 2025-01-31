from fractions import Fraction

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


def test_overrides_correct_order():
    m6idxl = shapes.region("us-east-1").instances["m6id.xlarge"]
    m6id4xl = shapes.region("us-east-1").instances["m6id.4xlarge"]

    # these are in the overrides file
    assert m6idxl.annual_cost == 781.66
    assert m6id4xl.annual_cost == 3126.64


def test_normalized_size():
    sizes = (
        "large",
        "xlarge",
        "2xlarge",
        "4xlarge",
        "8xlarge",
        "12xlarge",
        "16xlarge",
        "24xlarge",
    )

    expected = (
        Fraction(1, 2),
        Fraction(1),
        Fraction(2),
        Fraction(4),
        Fraction(8),
        Fraction(12),
        Fraction(16),
        Fraction(24),
    )
    hw = shapes.region("us-east-1").instances
    for i, size in enumerate(sizes):
        name = "m5." + size
        assert hw[name].normalized_size == expected[i]


def test_r6id():
    r6id_24xl = shapes.region("us-east-1").instances["r6id.24xlarge"]
    assert r6id_24xl is not None
    assert r6id_24xl.cpu == 96

from fractions import Fraction

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import DriveType
from service_capacity_modeling.interface import normalized_aws_size


def test_services():
    s3 = shapes.region("us-east-1").services["blob.standard"]
    assert s3.annual_cost_gib(1) > 0
    assert s3.annual_cost_per_write_io > 0
    assert s3.annual_cost_per_read_io > 0

    assert "dynamo.standard" in shapes.region("us-east-1").services


def test_drives():
    gp3 = shapes.region("us-east-1").drives["gp3"]
    assert gp3.drive_type == DriveType.attached_ssd
    assert gp3.max_scale_size_gib == 16384


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


def test_auto_file_loaded():
    m7axl = shapes.region("us-east-1").instances["m7a.xlarge"]
    assert m7axl.cpu == 4


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
    for i, size in enumerate(sizes):
        name = "m5." + size
        assert normalized_aws_size(name) == expected[i]


def test_r6id():
    r6id_24xl = shapes.region("us-east-1").instances["r6id.24xlarge"]
    assert r6id_24xl is not None
    assert r6id_24xl.cpu == 96

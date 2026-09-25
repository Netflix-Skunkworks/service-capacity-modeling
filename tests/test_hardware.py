from fractions import Fraction

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import Drive
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
    assert gp3.max_scale_size_gib == 65536
    assert gp3.max_scale_io_per_s == 16000
    assert gp3.max_scale_io_per_s_per_gib == 500
    assert gp3.max_scale_throughput == 2000
    assert gp3.max_scale_throughput_per_io == 0.25
    assert gp3.pricing_source == "public"
    assert gp3.pricing_region == "us-east-1"


def test_gp3_prices_shared_iops_and_throughput_once():
    gp3 = shapes.region("us-east-1").price_drive(
        Drive(
            name="gp3",
            size_gib=1000,
            read_io_per_s=10000,
            write_io_per_s=6000,
            provisioned_io_per_s=16000,
            throughput=1000,
        )
    )

    assert gp3.provisioned_io_per_s == 16000
    assert gp3.throughput == 1000
    assert gp3.annual_cost == 2160
    assert gp3.annual_cost_components == {
        "capacity": 960,
        "iops": 780,
        "throughput": 420,
        "read_iops": 0,
        "write_iops": 0,
    }


def test_directional_drive_pricing_remains_supported():
    drive = Drive(
        name="legacy",
        size_gib=10,
        read_io_per_s=4000,
        write_io_per_s=5000,
        annual_cost_per_gib=1,
        annual_cost_per_read_io=[(3000, 0), (10000, 1)],
        annual_cost_per_write_io=[(3000, 0), (10000, 2)],
    )

    assert drive.annual_cost == 5010


def test_shared_iops_pricing_accepts_legacy_directional_current_drive():
    gp3 = shapes.region("us-east-1").price_drive(
        Drive(name="gp3", size_gib=1000, read_io_per_s=10_000, write_io_per_s=6_000)
    )

    assert gp3.provisioned_io_per_s == 16_000
    assert gp3.annual_cost_components["iops"] == 780
    assert gp3.annual_cost_components["read_iops"] == 0
    assert gp3.annual_cost_components["write_iops"] == 0


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

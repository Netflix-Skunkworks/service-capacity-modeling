import copy

import botocore
import pytest
from botocore.stub import Stubber
from pytest import approx

from service_capacity_modeling.interface import Platform
from service_capacity_modeling.tools import auto_shape
from service_capacity_modeling.tools.auto_shape import deduce_cpu_perf
from service_capacity_modeling.tools.auto_shape import deduce_io_perf
from service_capacity_modeling.tools.auto_shape import guess_iops_per_gib
from service_capacity_modeling.tools.auto_shape import pull_family
from service_capacity_modeling.tools.auto_shape import pull_rds_family
from service_capacity_modeling.tools.auto_shape import CPUPerformance
from service_capacity_modeling.tools.auto_shape import deduce_cpu_ipc_scale
from tests.tools import mock_data


@pytest.fixture
def mock_ec2():
    ec2 = botocore.session.get_session().create_client("ec2", region_name="us-east-1")
    return ec2


def _stub_family(client, family):
    stub = Stubber(client)
    request = {
        "Filters": [
            {"Name": "instance-type", "Values": [f"{family}.*"]},
        ],
    }
    stub.add_response("describe_instance_types", mock_data[family], request)
    stub.activate()
    return stub


def test_pull_family_r6id(mock_ec2):
    _stub_family(mock_ec2, "r6id")
    cpu_perf = deduce_cpu_perf("r6id")
    io_perf = deduce_io_perf("r6id", "ssd", None)
    assert io_perf is not None

    shape = pull_family(
        ec2_client=mock_ec2,
        family="r6id",
        cpu_perf=cpu_perf,
        io_perf=io_perf,
    )[0]

    assert shape.name == "r6id.2xlarge"
    assert shape.cpu == 8
    assert shape.cpu_ghz == 3.5
    assert shape.cpu_ipc_scale == 1.0
    # AWS claims it has 64 GiB, real launches say 62, treating their GiB (base 2)
    # number as GB (base 10) gives 61. This is fine.
    assert 61 < shape.ram_gib < 62
    assert shape.net_mbps == 3125
    assert shape.drive is not None
    assert shape.drive.single_tenant
    assert shape.drive.read_io_per_s == approx(134167, rel=0.01)
    assert shape.ebs_baseline_iops == 12000
    assert shape.ebs_max_iops == 40000
    # AWS reports decimal MB/s; gp3 throughput is MiB/s.
    assert shape.ebs_baseline_throughput_mib_per_s == 298.02
    assert shape.ebs_max_throughput_mib_per_s == 1192.09
    assert shape.platforms == [Platform.amd64, Platform.amd64_intel]


def test_pull_family_m7a(mock_ec2):
    _stub_family(mock_ec2, "m7a")
    cpu_perf = deduce_cpu_perf("m7a")
    io_perf = deduce_io_perf("m7a", "ssd", None)
    assert io_perf is None

    shape = pull_family(
        ec2_client=mock_ec2,
        family="m7a",
        cpu_perf=cpu_perf,
        io_perf=io_perf,
    )[0]

    assert shape.name == "m7a.12xlarge"
    assert shape.cpu == 48
    assert shape.cpu_ghz == 3.7
    assert shape.cpu_ipc_scale == 1.5
    # AWS claims it has 192 GiB, real launches say 184, treating their GiB (base 2)
    # number as GB (base 10) gives 183. This is fine.
    assert 182 < shape.ram_gib < 184
    assert shape.net_mbps == approx(18750)
    assert shape.drive is None
    assert shape.ebs_baseline_iops == 60000
    assert shape.ebs_max_iops == 60000
    assert shape.ebs_baseline_throughput_mib_per_s == 1788.14
    assert shape.ebs_max_throughput_mib_per_s == 1788.14
    assert shape.platforms == [Platform.amd64, Platform.amd64_amd]


@pytest.mark.parametrize(
    "archs,manufacturer,expected",
    [
        (["arm64"], "AWS", [Platform.arm64, Platform.arm64_graviton]),
        (["i386", "x86_64"], "Intel", [Platform.amd64, Platform.amd64_intel]),
        # Unknown vendors get no subtag rather than a guess
        (["x86_64"], "Hygon", [Platform.amd64]),
        (["x86_64"], None, [Platform.amd64]),
    ],
)
def test_pull_family_platform_tags(mock_ec2, archs, manufacturer, expected):
    response = copy.deepcopy(mock_data["m7a"])
    processor = response["InstanceTypes"][0]["ProcessorInfo"]
    processor["SupportedArchitectures"] = archs
    processor.pop("Manufacturer")
    if manufacturer is not None:
        processor["Manufacturer"] = manufacturer
    stub = Stubber(mock_ec2)
    stub.add_response(
        "describe_instance_types",
        response,
        {"Filters": [{"Name": "instance-type", "Values": ["m7a.*"]}]},
    )
    stub.activate()

    shapes = pull_family(ec2_client=mock_ec2, family="m7a")
    assert [s.platforms for s in shapes] == [expected]


def test_pull_rds_family_has_no_ec2_platform_tags(monkeypatch):
    # Managed shapes must never carry amd64/arm64 (or vendor) tags, otherwise
    # they would become candidates for EC2-only models.
    rds = botocore.session.get_session().create_client("rds", region_name="us-east-1")
    stub = Stubber(rds)
    stub.add_response(
        "describe_orderable_db_instance_options",
        {
            "OrderableDBInstanceOptions": [
                {"DBInstanceClass": "db.r7g.large", "Engine": "aurora-postgresql"},
                {"DBInstanceClass": "db.r7g.xlarge", "Engine": "aurora-postgresql"},
            ]
        },
        {"Engine": "aurora-postgresql"},
    )
    stub.activate()
    monkeypatch.setattr(auto_shape.boto3, "client", lambda *a, **kw: None)
    monkeypatch.setattr(
        auto_shape,
        "lookup_ec2_instance_specs",
        lambda *a, **kw: (2, 1, 2.6, 16.0, 12500.0),
    )

    shapes = pull_rds_family(rds, family="r7g", db_engines=["aurora-postgresql"])

    assert [s.name for s in shapes] == ["db.r7g.large", "db.r7g.xlarge"]
    assert all(s.platforms == [Platform.aurora_postgres] for s in shapes)


def test_guess_iops_per_gib():
    should_exist = ("i4i", "m6id", "r6id", "i3", "i3en")
    for family in should_exist:
        if guess_iops_per_gib(family) is None:
            assert family == "did not exist"

    assert guess_iops_per_gib("random shape") is None


# Tests IPC and HT scale factors
def test_deduce_cpu_ipc_scale_no_ht_multiplies_both_factors():
    # no-HT (vcpu == cores): arch IPC × 1.5
    perf = CPUPerformance(ipc_scale_factor=1.2995)  # GENOA_IPC
    result = deduce_cpu_ipc_scale(vcpu_count=2, cpu_cores=2, cpu_perf=perf)
    assert result == approx(1.2995 * 1.5, rel=0.01)  # ≈ 1.949


def test_deduce_cpu_ipc_scale_ht_on_no_ht_factor():
    # HT on (vcpu != cores): arch IPC only, ht_factor = 1.0
    perf = CPUPerformance(ipc_scale_factor=1.15)  # MILAN_IPC
    result = deduce_cpu_ipc_scale(vcpu_count=2, cpu_cores=1, cpu_perf=perf)
    assert result == approx(1.15)


def test_deduce_cpu_ipc_scale_no_perf_ht_on():
    # no arch IPC provided, HT on: defaults to 1.0
    result = deduce_cpu_ipc_scale(vcpu_count=2, cpu_cores=1, cpu_perf=None)
    assert result == approx(1.0)


def test_deduce_cpu_ipc_scale_no_perf_no_ht():
    # no arch IPC provided, HT off: 1.0 × 1.5
    result = deduce_cpu_ipc_scale(vcpu_count=2, cpu_cores=2, cpu_perf=None)
    assert result == approx(1.5)

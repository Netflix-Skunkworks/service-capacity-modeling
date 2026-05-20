import botocore
import pytest
from botocore.stub import Stubber
from pytest import approx

from service_capacity_modeling.tools.auto_shape import deduce_cpu_perf
from service_capacity_modeling.tools.auto_shape import deduce_io_perf
from service_capacity_modeling.tools.auto_shape import guess_iops_per_gib
from service_capacity_modeling.tools.auto_shape import pull_family
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

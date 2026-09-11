"""Tests for hardware-only instance scale factors."""

import math

import pytest
from pydantic import ValidationError

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.models.instance_scale import InstanceScaleFactors
from service_capacity_modeling.models.instance_scale import scale_factors
from service_capacity_modeling.models.plan_comparison import ResourceType


@pytest.mark.parametrize(
    "name",
    [
        "c5.2xlarge",  # hyperthreaded, EBS only
        "c7a.2xlarge",  # non-hyperthreaded
        "i3.2xlarge",  # local disk, cpu_cores unset
        "m5d.2xlarge",  # cpu_ipc_scale unset (defaults to 1.0)
    ],
)
def test_identity_is_exactly_one(name: str) -> None:
    factors = scale_factors(name, name)

    for dimension in factors.dimensions.values():
        assert dimension.factor == 1.0, dimension
        assert factors.is_limiting(dimension)
    assert factors.limiting.factor == 1.0


def test_hyperthreaded_to_physical_cores() -> None:
    """c5.2xlarge is 8 vCPU = 4 HT cores; c7a.2xlarge is 8 vCPU = 8 real cores.

    cpu_ipc_scale already folds in both the architectural IPC uplift and the
    1.5x non-SMT factor, so the compute factor must come from
    vCPU x GHz x cpu_ipc_scale (11.83 -> 25.10, i.e. ~0.471).
    """
    compute = scale_factors("c5.2xlarge", "c7a.2xlarge").compute

    assert compute.factor == pytest.approx(0.471, rel=0.05)
    # Regression guard: cores x cpu_ipc_scale would double-count SMT and drop
    # clock frequency, landing near 0.256 -- 1.8x optimistic, the unsafe way.
    assert compute.factor > 0.4


def test_memory_binds_rather_than_compute() -> None:
    """c5 -> c7a nearly doubles compute but keeps RAM identical at 15.26 GiB."""
    factors = scale_factors("c5.2xlarge", "c7a.2xlarge")

    assert factors.memory.factor == 1.0
    assert factors.limiting.resource is ResourceType.mem_gib
    assert factors.limiting_resource is ResourceType.mem_gib
    assert factors.limiting_factor == 1.0
    assert factors.is_limiting(factors.memory)

    # The safety valve for a caller that narrowed to compute.
    assert factors.compute.factor < 1.0
    assert not factors.is_limiting(factors.compute)


def test_same_family_size_up_halves() -> None:
    factors = scale_factors("c7a.2xlarge", "c7a.4xlarge")

    assert factors.compute.factor == 0.5
    assert factors.memory.factor == 0.5
    assert factors.network.factor == 0.5
    assert factors.limiting_factor == 0.5

    # All three tie at the max, so every one of them binds.
    assert factors.is_limiting(factors.compute)
    assert factors.is_limiting(factors.memory)
    assert factors.is_limiting(factors.network)


def test_local_disk_dimension_present_and_binding() -> None:
    """i3.2xlarge 1900 GiB -> i4i.2xlarge 1746 GiB: storage binds, not compute."""
    factors = scale_factors("i3.2xlarge", "i4i.2xlarge")
    disk = factors.disk

    assert disk is not None
    assert disk.from_capacity == 1900
    assert disk.to_capacity == 1746
    assert disk.factor == pytest.approx(1900 / 1746)
    assert disk.is_representable
    assert factors.limiting.resource is ResourceType.disk_gib
    assert not factors.is_limiting(factors.compute)


def test_losing_local_disk_is_unrepresentable() -> None:
    """Dropping 1900 GiB of local disk for an EBS-only shape must not read fine."""
    factors = scale_factors("i3.2xlarge", "r7a.2xlarge")
    disk = factors.disk

    assert disk is not None
    assert disk.factor == math.inf
    assert not disk.is_representable
    assert factors.limiting.resource is ResourceType.disk_gib
    assert factors.limiting_factor == math.inf
    # Memory looks nearly neutral here, which is exactly why disk has to surface.
    assert factors.memory.factor == pytest.approx(1.0, rel=0.01)


def test_no_local_disk_omits_the_dimension() -> None:
    factors = scale_factors("c5.2xlarge", "c7a.2xlarge")

    assert factors.disk is None
    assert ResourceType.disk_gib not in factors.dimensions


def test_uncurated_ipc_scale_is_flagged_not_raised() -> None:
    """m5d.* ships no cpu_ipc_scale, so compute rests on the 1.0 default.

    It happens to be right (Skylake is the 1.0 baseline), but it is un-asserted
    data, so it is surfaced rather than silently folded in -- and rather than
    raising, which would refuse a swap whose factor is in fact correct.
    """
    factors = scale_factors("m5d.2xlarge", "m7i.2xlarge")

    assert not factors.compute.curated
    assert factors.uncurated == [factors.compute]
    assert factors.memory.curated
    assert factors.network.curated


def test_curated_shapes_report_nothing_uncurated() -> None:
    factors = scale_factors("c5.2xlarge", "c7a.2xlarge")

    assert factors.compute.curated
    assert factors.uncurated == []


def _synthetic(name: str, **overrides: float) -> Instance:
    params = {
        "name": name,
        "cpu": 8,
        "cpu_ghz": 2.3,
        "cpu_ipc_scale": 1.0,
        "ram_gib": 32.0,
        "net_mbps": 2000.0,
    }
    params.update(overrides)
    return Instance(**params)  # type: ignore[arg-type]


def test_ipc_scale_applied_exactly_once() -> None:
    """Independent of shape data: a 1.5x ipc_scale is worth exactly 1.5x.

    If the hyperthreading factor inside cpu_ipc_scale were combined with a
    separate cpu_cores term, this would come out at 0.333 rather than 0.667.
    """
    factors = scale_factors(
        _synthetic("synth.ht", cpu_ipc_scale=1.0),
        _synthetic("synth.noht", cpu_ipc_scale=1.5),
    )

    assert factors.compute.factor == pytest.approx(1 / 1.5)
    assert factors.memory.factor == 1.0
    assert factors.network.factor == 1.0


def test_clock_frequency_counts_toward_compute() -> None:
    factors = scale_factors(
        _synthetic("synth.slow", cpu_ghz=2.0),
        _synthetic("synth.fast", cpu_ghz=4.0),
    )

    assert factors.compute.factor == pytest.approx(0.5)


def test_synthetic_local_disk_to_ebs() -> None:
    with_disk = _synthetic("synth.disk")
    with_disk.drive = Drive(name="ephem", size_gib=500)

    factors = scale_factors(with_disk, _synthetic("synth.ebs"))

    assert factors.disk is not None
    assert factors.disk.factor == math.inf

    # Reversed: the source has no local disk, so there is nothing to preserve.
    reverse = scale_factors(_synthetic("synth.ebs"), with_disk)
    assert reverse.disk is None


def test_names_and_instances_are_equivalent() -> None:
    by_name = scale_factors("c5.2xlarge", "c7a.2xlarge")
    by_instance = scale_factors(
        shapes.instance("c5.2xlarge"), shapes.instance("c7a.2xlarge")
    )

    assert by_name.model_dump() == by_instance.model_dump()
    assert by_name.from_instance == "c5.2xlarge"
    assert by_name.to_instance == "c7a.2xlarge"


def test_unknown_instance_raises() -> None:
    with pytest.raises(KeyError):
        scale_factors("c5.2xlarge", "not.a.real.shape")


def test_explain_mentions_workload_blindness_and_limiting_dimension() -> None:
    explanation = scale_factors("c5.2xlarge", "c7a.2xlarge").explain()

    assert "workload-blind" in explanation
    assert "does not rightsize" in explanation
    assert "limited by mem_gib" in explanation
    assert "[limiting]" in explanation


def test_serializes_with_limiting_answer() -> None:
    """A downstream service consumer gets the safe factor without recomputing."""
    dumped = scale_factors("c5.2xlarge", "c7a.2xlarge").model_dump()

    assert dumped["limiting_resource"] == ResourceType.mem_gib
    assert dumped["limiting_factor"] == 1.0
    assert dumped["dimensions"][ResourceType.cpu]["factor"] == pytest.approx(
        0.471, rel=0.05
    )


def test_which_dimension_binds_is_not_serialized_per_dimension() -> None:
    """Binding is relational, so it lives on the parent, not on each dimension.

    Pins that a per-dimension flag is not reintroduced into the wire contract,
    where it could go stale relative to the dimensions it describes.
    """
    dumped = scale_factors("c5.2xlarge", "c7a.2xlarge").model_dump()

    for dimension in dumped["dimensions"].values():
        assert "is_limiting" not in dimension


def test_round_trip_recomputes_the_limiting_dimension() -> None:
    """Derived state is recomputed on deserialization, never carried."""
    factors = scale_factors("i3.2xlarge", "i4i.2xlarge")
    restored = InstanceScaleFactors.model_validate(factors.model_dump())

    assert restored.limiting_resource is factors.limiting_resource
    assert restored.limiting_factor == factors.limiting_factor
    assert restored.disk is not None
    assert restored.is_limiting(restored.disk)
    assert not restored.is_limiting(restored.compute)


def test_empty_dimensions_is_rejected() -> None:
    """A scale factor with nothing to compare is meaningless, not empty."""
    with pytest.raises(ValidationError):
        InstanceScaleFactors(from_instance="c5.2xlarge", to_instance="c7a.2xlarge")

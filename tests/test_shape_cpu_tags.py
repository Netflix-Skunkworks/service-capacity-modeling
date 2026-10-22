"""Shipped shapes carry a consistent CPU architecture and microarchitecture.

EC2 shapes carry exactly one architecture platform (amd64/arm64) and a curated
cpu_microarch on that architecture. Managed shapes (Aurora, ElastiCache) carry
neither: an amd64/arm64 tag would make them candidates for EC2-only models.
"""

import json
from pathlib import Path
from typing import Dict
from typing import List

import pytest
from pydantic import ValidationError

from service_capacity_modeling.interface import CPUMicroarch
from service_capacity_modeling.interface import CPUVendor
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Platform

SHAPES_DIR = (
    Path(__file__).resolve().parent.parent
    / "service_capacity_modeling/hardware/profiles/shapes/aws"
)
MANAGED_PREFIXES = ("db.", "cache.")
# Synthetic shape that exercises lifecycle filtering, not real hardware
SYNTHETIC_SHAPES = {"f5.large.deprecated"}
ARCHS = (Platform.amd64, Platform.arm64)


def _shipped_instances() -> Dict[str, Instance]:
    instances: Dict[str, Instance] = {}
    for path in sorted(SHAPES_DIR.glob("*.json")):
        for name, shape in json.loads(path.read_text()).get("instances", {}).items():
            instances[name] = Instance(**shape)
    return instances


def test_ec2_shapes_have_one_arch_and_a_microarch_on_it() -> None:
    bad: List[str] = []
    for name, instance in _shipped_instances().items():
        if name.startswith(MANAGED_PREFIXES) or name in SYNTHETIC_SHAPES:
            continue
        microarch = instance.cpu_microarch
        if microarch is None or instance.platforms != [microarch.isa]:
            bad.append(f"{name}: {instance.platforms} {instance.cpu_microarch}")
    assert not bad, bad


def test_managed_shapes_have_no_ec2_cpu_tags() -> None:
    bad = [
        f"{name}: {instance.platforms} {instance.cpu_microarch}"
        for name, instance in _shipped_instances().items()
        if name.startswith(MANAGED_PREFIXES)
        and (set(ARCHS).intersection(instance.platforms) or instance.cpu_microarch)
    ]
    assert not bad, bad


def test_microarch_derives_vendor_and_isa() -> None:
    shape = Instance(
        name="m7g.xlarge",
        cpu=4,
        cpu_ghz=2.6,
        ram_gib=16,
        net_mbps=1,
        platforms=[Platform.arm64],
        cpu_microarch=CPUMicroarch.graviton3,
    )
    assert shape.cpu_vendor == CPUVendor.graviton
    assert CPUMicroarch.graviton3.isa == Platform.arm64
    assert "cpu_vendor" not in shape.model_dump()


def test_microarch_must_run_on_a_listed_platform() -> None:
    # A Graviton microarch on an amd64-only shape is contradictory data
    with pytest.raises(ValidationError, match="graviton3"):
        Instance(
            name="m7g.xlarge",
            cpu=4,
            cpu_ghz=2.6,
            ram_gib=16,
            net_mbps=1,
            cpu_microarch=CPUMicroarch.graviton3,
        )


def test_shapes_without_microarch_have_no_vendor() -> None:
    shape = Instance(name="custom.large", cpu=2, cpu_ghz=3, ram_gib=8, net_mbps=1)
    assert shape.cpu_microarch is None and shape.cpu_vendor is None

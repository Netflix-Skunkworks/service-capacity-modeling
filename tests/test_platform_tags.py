"""Shipped shapes carry consistent CPU architecture and vendor platform tags.

EC2 shapes carry exactly one architecture tag (amd64/arm64) plus exactly one
matching vendor subtag. Managed shapes (Aurora, ElastiCache) carry neither:
an amd64/arm64 tag would make them candidates for EC2-only models.
"""

import json
from pathlib import Path
from typing import Dict
from typing import List

from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Platform

SHAPES_DIR = (
    Path(__file__).resolve().parent.parent
    / "service_capacity_modeling/hardware/profiles/shapes/aws"
)
MANAGED_PREFIXES = ("db.", "cache.")
# Synthetic shape that exercises lifecycle filtering, not real hardware
SYNTHETIC_SHAPES = {"f5.large.deprecated"}
VENDOR_ARCH = {
    Platform.amd64_intel: Platform.amd64,
    Platform.amd64_amd: Platform.amd64,
    Platform.arm64_graviton: Platform.arm64,
}
ARCH_AND_VENDOR = set(VENDOR_ARCH) | set(VENDOR_ARCH.values())


def _shipped_instances() -> Dict[str, Instance]:
    instances: Dict[str, Instance] = {}
    for path in sorted(SHAPES_DIR.glob("*.json")):
        for name, shape in json.loads(path.read_text()).get("instances", {}).items():
            instances[name] = Instance(**shape)
    return instances


def test_ec2_shapes_have_arch_and_matching_vendor() -> None:
    bad: List[str] = []
    for name, instance in _shipped_instances().items():
        if name.startswith(MANAGED_PREFIXES) or name in SYNTHETIC_SHAPES:
            continue
        vendors = [p for p in instance.platforms if p in VENDOR_ARCH]
        archs = [p for p in instance.platforms if p in (Platform.amd64, Platform.arm64)]
        if len(vendors) != 1 or archs != [VENDOR_ARCH[vendors[0]]]:
            bad.append(f"{name}: {instance.platforms}")
    assert not bad, bad


def test_managed_shapes_have_no_ec2_platform_tags() -> None:
    bad = [
        f"{name}: {instance.platforms}"
        for name, instance in _shipped_instances().items()
        if name.startswith(MANAGED_PREFIXES)
        and ARCH_AND_VENDOR.intersection(instance.platforms)
    ]
    assert not bad, bad

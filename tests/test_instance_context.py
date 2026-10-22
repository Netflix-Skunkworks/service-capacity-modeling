import json
import random
from fractions import Fraction
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import pytest

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.hardware.context import HardwareProvider
from service_capacity_modeling.hardware.context import instance_context
from service_capacity_modeling.hardware.context import InstanceContext
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Platform
from service_capacity_modeling.tools.auto_shape import _parse_family

SHAPES_DIR = (
    Path(__file__).resolve().parent.parent
    / "service_capacity_modeling/hardware/profiles/shapes/aws"
)
# Synthetic shape that exercises lifecycle filtering, not a real AWS name
SYNTHETIC_SHAPES = {"f5.large.deprecated"}


def _one_shape_per_family() -> List[Instance]:
    families: Dict[str, List[Instance]] = {}
    for path in sorted(SHAPES_DIR.glob("*.json")):
        for name, shape in json.loads(path.read_text()).get("instances", {}).items():
            if name not in SYNTHETIC_SHAPES:
                families.setdefault(name.rsplit(".", 1)[0], []).append(
                    Instance(**shape)
                )
    # Seeding per family picks a "random" size that is stable across runs
    return [
        random.Random(family).choice(sorted(shapes, key=lambda i: i.name))
        for family, shapes in sorted(families.items())
    ]


@pytest.mark.parametrize("instance", _one_shape_per_family(), ids=lambda i: i.name)
def test_parse_shipped_shapes(instance: Instance) -> None:
    series, generation, vendor = _parse_family(instance.family)
    family = instance.family.removeprefix("db.").removeprefix("cache.")
    assert family.startswith(f"{series}{generation}")

    parsed = instance_context(instance)
    assert parsed is not None
    assert (parsed.provider, parsed.series, parsed.generation, parsed.vendor) == (
        "aws",
        series,
        generation,
        vendor,
    )
    # EC2 shapes carry a curated microarch checked against AWS; managed do not
    if instance.cpu_microarch is not None:
        assert instance.cpu_vendor == vendor
        assert parsed.arch in instance.platforms


@pytest.mark.parametrize(
    "name,expected",
    [
        # Graviton with more option letters after the "g"
        ("m7gd.xlarge", ("m", 7, "graviton")),
        ("c7gn.large", ("c", 7, "graviton")),
        ("c6gn.large", ("c", 6, "graviton")),
        ("r8gd.large", ("r", 8, "graviton")),
        ("im4gn.large", ("im", 4, "graviton")),
        ("is4gen.large", ("is", 4, "graviton")),
        ("hpc7g.4xlarge", ("hpc", 7, "graviton")),
        ("a1.large", ("a", 1, "graviton")),
        # Managed shapes parse as their underlying family
        ("db.r7g.large", ("r", 7, "graviton")),
        ("cache.r7g.large", ("r", 7, "graviton")),
        ("m7a.large", ("m", 7, "amd")),
        ("r6id.large", ("r", 6, "intel")),
        ("i4i.large", ("i", 4, "intel")),
        ("i3en.large", ("i", 3, "intel")),
        ("r7iz.large", ("r", 7, "intel")),
        ("m5zn.large", ("m", 5, "intel")),
        ("x2iedn.xlarge", ("x", 2, "intel")),
        ("c7i-flex.large", ("c", 7, "intel")),
        # Accelerator hosts the name says nothing about
        ("p5.48xlarge", ("p", 5, "amd")),
        ("g6e.xlarge", ("g", 6, "amd")),
        ("inf2.xlarge", ("inf", 2, "amd")),
        ("p6-b200.48xlarge", ("p", 6, "intel")),
        ("p6e-gb200.36xlarge", ("p", 6, "arm")),
        ("u-6tb1.112xlarge", ("u", 1, "intel")),
        ("u7i-12tb.224xlarge", ("u", 7, "intel")),
        ("mac1.metal", ("mac", 1, "intel")),
        ("mac2-m2pro.metal", ("mac", 2, "arm")),
        ("mac-m4.metal", ("mac", 4, "arm")),
    ],
)
def test_parse_aws_names(name: str, expected: tuple) -> None:
    assert _parse_family(name.rsplit(".", 1)[0]) == expected
    parsed = InstanceContext.from_name(name)
    assert (parsed.provider, parsed.series, parsed.generation, parsed.vendor) == (
        "aws",
        *expected,
    )
    assert parsed.arch == ("amd64" if expected[2] in ("intel", "amd") else "arm64")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("n2-standard-8", ("gcp", "n", 2, "intel", "amd64")),
        ("n2d-highmem-16", ("gcp", "n", 2, "amd", "amd64")),
        ("c3d-highcpu-4", ("gcp", "c", 3, "amd", "amd64")),
        ("c4a-standard-8", ("gcp", "c", 4, "arm", "arm64")),
        ("t2a-standard-4", ("gcp", "t", 2, "arm", "arm64")),
        ("c4n-standard-2", ("gcp", "c", 4, "intel", "amd64")),
        ("e2-medium", ("gcp", "e", 2, "intel", "amd64")),
        ("a3-highgpu-8g", ("gcp", "a", 3, "intel", "amd64")),
        ("a4x-highgpu-4g", ("gcp", "a", 4, "arm", "arm64")),
        ("z3-highmem-88", ("gcp", "z", 3, "intel", "amd64")),
        ("Standard_D16as_v5", ("azure", "D", 5, "amd", "amd64")),
        ("Standard_E8ads_v5", ("azure", "E", 5, "amd", "amd64")),
        ("Standard_D4ps_v5", ("azure", "D", 5, "arm", "arm64")),
        ("Standard_D4s_v3", ("azure", "D", 3, "intel", "amd64")),
        ("Standard_M8-2ms_v2", ("azure", "M", 2, "intel", "amd64")),
        ("Standard_M48ds_1_v3", ("azure", "M", 3, "intel", "amd64")),
        ("Standard_NC4as_T4_v3", ("azure", "NC", 3, "amd", "amd64")),
        ("Standard_HB120-96rs_v3", ("azure", "HB", 3, "amd", "amd64")),
        ("Standard_L8s_v2", ("azure", "L", 2, "amd", "amd64")),
        ("Standard_L8s_v3", ("azure", "L", 3, "intel", "amd64")),
    ],
)
def test_parse_gcp_and_azure_names(name: str, expected: tuple) -> None:
    parsed = InstanceContext.from_name(name)
    assert (
        parsed.provider,
        parsed.series,
        parsed.generation,
        parsed.vendor,
        parsed.arch,
    ) == expected
    assert InstanceContext.from_name(name, HardwareProvider(expected[0])) == parsed


def test_instance_context_details() -> None:
    m7gd = InstanceContext.from_name("m7gd.xlarge")
    assert (m7gd.attributes, m7gd.size) == ("gd", Fraction(1))
    assert m7gd.arch == Platform.arm64
    assert InstanceContext.from_name("m7gd").size is None
    # The same family means different hardware depending on the provider
    assert InstanceContext.from_name("c4a", HardwareProvider.gcp).vendor == "arm"
    assert InstanceContext.from_name("c4a", HardwareProvider.aws).vendor == "amd"


@pytest.mark.parametrize(
    "name,size",
    [
        # AWS sizes are xlarge multiples, GCE and Azure count vCPUs (4 per xlarge)
        ("t3.nano", Fraction(1, 32)),
        ("m7a.large", Fraction(1, 2)),
        ("r6id.24xlarge", Fraction(24)),
        ("c7i.metal-24xl", Fraction(24)),
        ("db.r7g.2xlarge", Fraction(2)),
        ("n2-standard-8", Fraction(2)),
        ("c3-highmem-6-lssd", Fraction(3, 2)),
        ("e2-medium", None),
        ("a3-highgpu-8g", None),
        ("Standard_D16as_v5", Fraction(4)),
        ("Standard_M8-2ms_v2", Fraction(2)),
    ],
)
def test_instance_context_size(name: str, size: Optional[Fraction]) -> None:
    parsed = InstanceContext.from_name(name).size
    assert parsed == size
    assert size is None or isinstance(parsed, Fraction)


@pytest.mark.parametrize("name", ["", "m.large", "not-an-instance!", "Standard_"])
def test_instance_context_rejects_garbage(name: str) -> None:
    with pytest.raises(ValueError):
        InstanceContext.from_name(name)


def test_unparseable_instance_has_no_context() -> None:
    fake = Instance(
        name="f5.large.deprecated", cpu=2, cpu_ghz=3.1, ram_gib=8, net_mbps=1
    )
    assert instance_context(fake) is None
    with pytest.raises(ValueError):
        InstanceContext.from_name(fake.name)


def test_instance_context_stays_off_the_schema() -> None:
    # Display-only context must not become part of Instance or plan output
    instance = shapes.instance("m7a.xlarge")
    assert not hasattr(instance, "instance_context")
    context = instance_context(instance)
    assert context is not None and context.size == Fraction(1)
    # Even a direct dump leaves out the Fraction size
    assert "size" not in context.model_dump()

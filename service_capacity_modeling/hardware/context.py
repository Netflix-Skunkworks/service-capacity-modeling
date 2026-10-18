"""What instance type names say about hardware, for display only

Nothing here is a performance or quality measure. See InstanceContext.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional

from pydantic import ConfigDict
from pydantic import Field

from service_capacity_modeling.enum_utils import enum_docstrings
from service_capacity_modeling.enum_utils import StrEnum
from service_capacity_modeling.interface import ExcludeUnsetModel
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import normalized_aws_size
from service_capacity_modeling.interface import Platform


@enum_docstrings
class HardwareProvider(StrEnum):
    """Hardware provider whose instance type naming convention a name follows"""

    aws = "aws"
    gcp = "gcp"
    azure = "azure"


@enum_docstrings
class CPUVendor(StrEnum):
    """CPU vendor as far as an instance type name reveals it"""

    intel = "intel"
    amd = "amd"
    graviton = "graviton"
    arm = "arm"
    """Any other Arm host, e.g. GCE Axion/Ampere, Azure Cobalt/Ampere,
    NVIDIA Grace or Apple silicon"""


# Naming conventions, note letters mean different things per provider (AWS
# "a" is AMD and "d" is local disk, GCE "a" is Arm and "d" is AMD):
#   https://docs.aws.amazon.com/ec2/latest/instancetypes/instance-type-names.html
#   https://docs.cloud.google.com/compute/docs/machine-resource
#   https://learn.microsoft.com/en-us/azure/virtual-machines/vm-naming-conventions
_FAMILY_RE = re.compile(r"([a-z]+?)(\d+)([a-z]*(?:-[a-z0-9]+)?)")
_AZURE_RE = re.compile(
    r"(?:Standard_|Basic_)?([A-Z]+)(\d+)(?:-\d+)?([a-z]*)"
    r"(?:_[A-Za-z0-9]+?)*?(?:_v(\d+))?"
)
_VENDOR_LETTERS = {
    HardwareProvider.aws: {
        "a": CPUVendor.amd,
        "g": CPUVendor.graviton,
        "i": CPUVendor.intel,
    },
    HardwareProvider.gcp: {"a": CPUVendor.arm, "d": CPUVendor.amd},
}
# Accelerator families whose AMD host CPU is not in the name, per
# describe-instance-types ProcessorInfo.Manufacturer
_AWS_AMD_HOSTS = frozenset(
    {"f2", "g5", "g6", "g6e", "g6f", "gr6", "gr6f", "inf2", "p5", "p5e"}
)


class InstanceContext(ExcludeUnsetModel):
    """Additional context for humans decoding names

    This is context to show people, e.g. what changes between two shapes being
    compared. It must NEVER be used as a measure of relative performance or
    hardware quality: use cpu_ghz, cpu_ipc_scale and friends for that, which is
    why most fields here are strings. The name is only a hint, so prefer
    Instance.platforms tags where a shape has them.

    Examples below: AWS m7gd.2xlarge, GCE c3d-highmem-8-lssd and Azure
    Standard_NC24ads_A100_v4.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    provider: HardwareProvider
    """Whose naming convention the name follows: aws, gcp, azure"""

    series: str
    """Family letters before the generation: m, c, NC"""

    generation: int
    """The provider's own generation number: 7 (m7), 3 (c3), 4 (_v4). Not a
    performance measure, and only comparable within one provider and series"""

    vendor: CPUVendor
    """Host CPU vendor: graviton, amd, amd. From option letters plus known
    exceptions where the name says nothing (e.g. AWS p5 is AMD)"""

    arch: Platform
    """CPU architecture implied by vendor: arm64, amd64, amd64"""

    attributes: str = ""
    """Option letters after the generation, vendor letter included: gd
    (Graviton, local disk), d (AMD), ads (AMD, local disk, premium storage)"""

    size: Optional[Fraction] = Field(default=None, exclude=True)
    """Size in AWS xlarge units (4 vCPUs): 2, 2, 6. Not a performance measure.
    AWS: normalized_aws_size (large = 1/2, 2xlarge = 2, metal = 48).
    GCE: trailing vCPU count / 4 (highmem-8 -> 2).
    Azure: leading vCPU count / 4 (NC24 -> 6; constrained E16-8 uses 16).
    None when the name has no count (e2-medium, a3-highgpu-8g)"""

    @classmethod
    def from_name(  # noqa: C901
        cls, name: str, provider: Optional[HardwareProvider] = None
    ) -> InstanceContext:
        """Parses an instance type name, guessing the provider if not given

        Accepts a full name (m7gd.xlarge, n2d-standard-8, Standard_D16as_v5)
        or an AWS/GCE family (m7gd, n2d).
        """
        if provider is None:
            if name.startswith(("Standard_", "Basic_")) or re.search(r"_v\d+$", name):
                provider = HardwareProvider.azure
            else:
                aws = "." in name or "-" not in name
                provider = HardwareProvider.aws if aws else HardwareProvider.gcp
        if provider == HardwareProvider.azure:
            return cls._parse_azure(name)

        family, size = name, None
        if provider == HardwareProvider.aws:
            family = family.removeprefix("db.").removeprefix("cache.")
            if "." in family:
                size = normalized_aws_size(family)
                family = family.rsplit(".", 1)[0]
            # Older high memory shapes put the generation last (u-6tb1), and
            # the newest Mac shapes have none, so use the Apple chip's (mac-m4)
            family = re.sub(r"^u-(\d+tb)(\d+)$", r"u\2-\1", family)
            family = re.sub(r"^mac-(m(\d+)[a-z]*)$", r"mac\2-\1", family)
        elif "-" in family:
            family, machine_type = family.split("-", 1)
            size = _vcpu_xlarge_units(machine_type)
        m = _FAMILY_RE.fullmatch(family)
        if m is None:
            raise ValueError(f"Cannot parse {provider} instance family {family!r}")
        series, generation, attributes = m.groups()

        vendor = _VENDOR_LETTERS[provider].get(attributes[:1], CPUVendor.intel)
        if provider == HardwareProvider.aws and series == "a":
            vendor = CPUVendor.graviton
        elif provider == HardwareProvider.aws and family in _AWS_AMD_HOSTS:
            vendor = CPUVendor.amd
        elif provider == HardwareProvider.aws and (
            (series == "mac" and generation != "1") or "-gb" in attributes
        ):
            vendor = CPUVendor.arm  # Apple silicon or NVIDIA Grace (p6e-gb200)
        elif (
            provider == HardwareProvider.gcp and series == "a" and attributes[:1] == "x"
        ):
            vendor = CPUVendor.arm  # A4X hosts are NVIDIA Grace
        return cls(
            provider=provider,
            series=series,
            generation=int(generation),
            vendor=vendor,
            arch=_vendor_arch(vendor),
            attributes=attributes,
            size=size,
        )

    @classmethod
    def _parse_azure(cls, name: str) -> InstanceContext:
        # Azure puts the vCPU count where AWS/GCE put the generation, and the
        # generation in a trailing _vN (absent means v1)
        m = _AZURE_RE.fullmatch(name)
        if m is None:
            raise ValueError(f"Cannot parse azure instance name {name!r}")
        series, vcpus, attributes, version = m.groups()
        vendor = CPUVendor.intel
        if "a" in attributes or series == "HB" or (series == "L" and version == "2"):
            # HB and Lsv2 run on AMD EPYC without the "a" feature letter
            vendor = CPUVendor.amd
        elif "p" in attributes:
            vendor = CPUVendor.arm
        return cls(
            provider=HardwareProvider.azure,
            series=series,
            generation=int(version or 1),
            vendor=vendor,
            arch=_vendor_arch(vendor),
            attributes=attributes,
            size=_vcpu_xlarge_units(vcpus),
        )


def _vcpu_xlarge_units(size: str) -> Optional[Fraction]:
    # GCE and Azure count vCPUs where AWS counts xlarges (4 vCPUs). GCE
    # shared-core and GPU types (e2-medium, a3-highgpu-8g) don't say.
    m = re.search(r"(?:^|-)(\d+)(?:-lssd|-metal)?$", size)
    return Fraction(int(m.group(1)), 4) if m else None


def _vendor_arch(vendor: CPUVendor) -> Platform:
    if vendor in (CPUVendor.graviton, CPUVendor.arm):
        return Platform.arm64
    return Platform.amd64


def instance_context(instance: Instance) -> Optional[InstanceContext]:
    """What this shape's name says about its hardware, for display only.

    Never use it as a performance or quality measure (see InstanceContext).
    None if the name can't be parsed, e.g. synthetic shapes.
    """
    try:
        return InstanceContext.from_name(instance.name)
    except ValueError:
        return None

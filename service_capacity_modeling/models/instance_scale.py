"""Hardware-only scale factors between two instance shapes.

.. warning::

    **This is deliberately workload-blind and is NOT a capacity recommender.**

    It answers exactly one question: "to preserve the capacity currently
    provisioned, how many ``to_instance`` do I need per ``from_instance``?" It
    is a pure function of two hardware shapes -- no telemetry, no planner, no
    service calls.

    It does **not** rightsize. It preserves the existing provisioning decision,
    including any overprovisioning, so that a hardware migration does not
    silently change a cluster's risk posture. Rightsizing stays a separate,
    deliberate decision made elsewhere.

    It knows nothing about utilization, memory-bandwidth sensitivity,
    ASG minimums, or blast radius on very large shapes. Callers own
    all of that. If you want a capacity recommendation, use
    ``planner.plan_certain``.

Example usage::

    import math

    from service_capacity_modeling.models.instance_scale import scale_factors

    factors = scale_factors("c5.2xlarge", "c7a.2xlarge")

    # The safe answer: the least favorable dimension binds, because whichever
    # resource runs out first is the real constraint.
    nodes = math.ceil(current_count * factors.limiting.factor)

    # Narrowing to one dimension asserts "this cluster has headroom on the
    # others". Often true and legitimate, but say so deliberately, and check
    # whether the dimension you picked is actually the binding one.
    compute = factors.compute
    if not factors.is_limiting(compute):
        logger.warning(
            "compute factor %.2fx is not binding; %s binds at %.2fx",
            compute.factor,
            factors.limiting_resource,
            factors.limiting_factor,
        )

Factors are raw floats; rounding is the caller's, since only the caller knows
its ASG minimums and whether a partial node means anything.
"""

import math
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from pydantic import computed_field
from pydantic import Field

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import ExcludeUnsetModel
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.models.common import get_disk_size_gib
from service_capacity_modeling.models.plan_comparison import ResourceType
from service_capacity_modeling.models.plan_comparison import to_reference_cores

# Order used to break ties when several dimensions share the max factor, so the
# reported limiting dimension is deterministic.
_DIMENSION_ORDER: Tuple[ResourceType, ...] = (
    ResourceType.cpu,
    ResourceType.mem_gib,
    ResourceType.network_mbps,
    ResourceType.disk_gib,
)


class DimensionScale(ExcludeUnsetModel):
    """How one resource dimension scales between two instance shapes."""

    resource: ResourceType
    """Which resource dimension this is"""

    from_capacity: float
    """Per-instance capacity of the shape being migrated away from"""

    to_capacity: float
    """Per-instance capacity of the shape being migrated to"""

    curated: bool = True
    """Whether every shape value feeding this dimension was explicitly set.

    False means at least one side fell back to a pydantic default rather than a
    curated value, so the factor may be wrong in a way that looks right. Today
    this only ever happens for compute, via an unset ``cpu_ipc_scale``.
    """

    @computed_field(return_type=float)  # type: ignore
    @property
    def factor(self) -> float:
        """How many ``to`` instances are needed per ``from`` instance.

        - ``> 1.0`` means the target is weaker here, so you need more boxes
        - ``< 1.0`` means the target is stronger here, so you need fewer
        - ``inf`` means the target has none of this resource at all (e.g.
          migrating off local disk onto an EBS-only shape), so no number of
          them preserves the current capacity
        """
        if self.to_capacity == 0:
            if self.from_capacity == 0:
                return 1.0
            return float("inf")
        return self.from_capacity / self.to_capacity

    @property
    def is_representable(self) -> bool:
        """False when no instance count can preserve this dimension."""
        return math.isfinite(self.factor)

    def __str__(self) -> str:
        if not self.is_representable:
            detail = (
                f"{self.resource.value}: unrepresentable "
                f"({self.from_capacity:.2f} -> none)"
            )
        else:
            detail = (
                f"{self.resource.value}: {self.factor:.3f}x "
                f"({self.from_capacity:.2f} -> {self.to_capacity:.2f})"
            )
        if not self.curated:
            detail += " [uncurated shape data]"
        return detail


class InstanceScaleFactors(ExcludeUnsetModel):
    """Per-dimension hardware scale factors for one instance type swap.

    See the module docstring: this is workload-blind and preserves current
    provisioning rather than rightsizing it.
    """

    from_instance: str
    """Name of the shape being migrated away from"""

    to_instance: str
    """Name of the shape being migrated to"""

    dimensions: Dict[ResourceType, DimensionScale] = Field(min_length=1)
    """Scale factors keyed by resource dimension.

    Compute, memory and network are always present. Disk is present only when
    the ``from`` shape has local instance storage, since for EBS-only shapes
    there is no local capacity to preserve.
    """

    @property
    def limiting(self) -> DimensionScale:
        """The least favorable dimension: the one that actually binds.

        This is the safe default. Whichever resource runs out first is the real
        constraint, so scaling by anything smaller under-provisions.
        """
        return max(
            (self.dimensions[r] for r in _DIMENSION_ORDER if r in self.dimensions),
            key=lambda d: d.factor,
        )

    @computed_field(return_type=ResourceType)  # type: ignore
    @property
    def limiting_resource(self) -> ResourceType:
        """Which dimension binds, so a surprising factor explains itself."""
        return self.limiting.resource

    @computed_field(return_type=float)  # type: ignore
    @property
    def limiting_factor(self) -> float:
        """The safe factor: the max (least favorable) across dimensions."""
        return self.limiting.factor

    def is_limiting(self, dimension: DimensionScale) -> bool:
        """Whether this dimension is (one of) the least favorable, i.e. binding.

        This is the check for a caller that narrowed to a single dimension: the
        cheap signal that the dimension they picked is not the one that binds.
        Every dimension exactly at the max counts, so an identity swap reports
        all of them.
        """
        return dimension.factor == self.limiting_factor

    @property
    def compute(self) -> DimensionScale:
        """Effective compute: vCPU count x clock x IPC scale.

        Named "compute" rather than "cpu" on purpose: this is not a vCPU count.
        Comparing vCPUs would make hyperthreaded and non-hyperthreaded shapes
        look equivalent when they are not.
        """
        return self.dimensions[ResourceType.cpu]

    @property
    def memory(self) -> DimensionScale:
        return self.dimensions[ResourceType.mem_gib]

    @property
    def network(self) -> DimensionScale:
        return self.dimensions[ResourceType.network_mbps]

    @property
    def disk(self) -> Optional[DimensionScale]:
        """Local instance storage, or None if the ``from`` shape has none."""
        return self.dimensions.get(ResourceType.disk_gib)

    @property
    def uncurated(self) -> List[DimensionScale]:
        """Dimensions computed from at least one defaulted shape value."""
        return [
            self.dimensions[r]
            for r in _DIMENSION_ORDER
            if r in self.dimensions and not self.dimensions[r].curated
        ]

    def explain(self) -> str:
        if math.isfinite(self.limiting_factor):
            headline = (
                f"{self.from_instance} -> {self.to_instance}: "
                f"{self.limiting_factor:.3f}x per instance "
                f"(limited by {self.limiting_resource.value})"
            )
        else:
            headline = (
                f"{self.from_instance} -> {self.to_instance}: no instance count "
                f"preserves {self.limiting_resource.value}; "
                f"{self.to_instance} has none of it"
            )
        lines = [
            headline,
            "  WARNING: hardware-only, workload-blind. Preserves current",
            "  provisioning; does not rightsize.",
        ]
        for resource in _DIMENSION_ORDER:
            if resource not in self.dimensions:
                continue
            dimension = self.dimensions[resource]
            marker = " [limiting]" if self.is_limiting(dimension) else ""
            lines.append(f"  {dimension}{marker}")
        if self.uncurated:
            lines.append(
                "  Uncurated shape data on: "
                + ", ".join(d.resource.value for d in self.uncurated)
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.explain()


def scale_factors(
    from_instance: Union[str, Instance],
    to_instance: Union[str, Instance],
) -> InstanceScaleFactors:
    """How many ``to_instance`` are needed per ``from_instance``, by dimension.

    .. warning::

        Hardware-only and deliberately workload-blind. This preserves the
        capacity currently provisioned -- including any overprovisioning -- and
        does **not** rightsize. It ignores utilization, memory-bandwidth
        sensitivity, ASG minimums and blast radius. It is not a capacity
        recommender; see ``planner.plan_certain`` for that.

    Multiply your current instance count by ``result.limiting.factor`` and round
    up. Narrowing to a single dimension (``result.compute``) asserts that the
    cluster has headroom on the others.

    Instance specs do not vary by region in SCM's data -- regions differ only in
    pricing and lifecycle -- so there is no region argument. Names are resolved
    through the global shape catalog and raise ``KeyError`` if unknown.
    """
    frm = (
        shapes.instance(from_instance)
        if isinstance(from_instance, str)
        else from_instance
    )
    to = shapes.instance(to_instance) if isinstance(to_instance, str) else to_instance

    # Effective compute, not vCPU count. cpu_ipc_scale already folds in both the
    # per-core IPC uplift and the hyperthreading factor (1.5x for non-SMT shapes,
    # see tools/auto_shape.py::deduce_cpu_ipc_scale), so the repo-canonical
    # measure multiplies it by the *vCPU* count. Using cpu_cores here would
    # double-count SMT and drop clock frequency entirely.
    dimensions: Dict[ResourceType, DimensionScale] = {
        ResourceType.cpu: DimensionScale(
            resource=ResourceType.cpu,
            from_capacity=to_reference_cores(frm.cpu, frm),
            to_capacity=to_reference_cores(to.cpu, to),
            # An unset cpu_ipc_scale silently defaults to 1.0, which computes as
            # "no IPC difference" and would look like a right answer.
            curated=(
                "cpu_ipc_scale" in frm.model_fields_set
                and "cpu_ipc_scale" in to.model_fields_set
            ),
        ),
        ResourceType.mem_gib: DimensionScale(
            resource=ResourceType.mem_gib,
            from_capacity=frm.ram_gib,
            to_capacity=to.ram_gib,
        ),
        ResourceType.network_mbps: DimensionScale(
            resource=ResourceType.network_mbps,
            from_capacity=frm.net_mbps,
            to_capacity=to.net_mbps,
        ),
    }

    # Only meaningful if there is local storage to preserve. When the source has
    # local disk and the target does not, the factor is infinite rather than
    # absent: dropping the dimension would make that swap read as fine.
    from_disk_gib = get_disk_size_gib(None, frm)
    if from_disk_gib > 0:
        dimensions[ResourceType.disk_gib] = DimensionScale(
            resource=ResourceType.disk_gib,
            from_capacity=from_disk_gib,
            to_capacity=get_disk_size_gib(None, to),
        )

    return InstanceScaleFactors(
        from_instance=frm.name,
        to_instance=to.name,
        dimensions=dimensions,
    )

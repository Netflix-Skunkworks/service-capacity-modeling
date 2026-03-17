"""Explainability types for the capacity planner.

This module contains the family graph (FamilyTrait, FamilyEdge, FamilyGraph)
and ExplainedPlans — types used to explain *why* the planner rejected
certain instance/drive combinations and what alternatives exist.

Core contract types (Bottleneck, Excuse) live in interface.py because they
are part of the CapacityModel.capacity_plan() return type.
"""

from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import DriveType
from service_capacity_modeling.interface import ExcludeUnsetModel
from service_capacity_modeling.interface import Excuse
from service_capacity_modeling.interface import Instance


class FamilyTrait(ExcludeUnsetModel):
    """Intrinsic hardware properties of an instance family.

    All numeric values are derived from hardware shapes data via
    from_instance(). Within an AWS family, ratios (ram/vcpu,
    disk/vcpu) are constant across instance sizes.
    """

    family: str
    memory_gib_per_vcpu: float
    has_local_disk: bool
    local_disk_gib_per_vcpu: Optional[float] = None
    drive_type: Optional[DriveType] = None

    @classmethod
    def from_instance(cls, instance: Instance) -> FamilyTrait:
        """Derive family traits from any instance in the family.

        Since ratios are constant within a family, any size works.
        """
        drive = instance.drive
        has_local = drive is not None
        return cls(
            family=instance.family,
            memory_gib_per_vcpu=round(instance.ram_gib / instance.cpu, 2),
            has_local_disk=has_local,
            local_disk_gib_per_vcpu=(
                round(drive.size_gib / instance.cpu, 1)
                if drive is not None and drive.size_gib > 0
                else None
            ),
            drive_type=drive.drive_type if drive is not None else None,
        )


class FamilyEdge(ExcludeUnsetModel):
    """A directed trade-off edge between two instance families.

    Edges encode domain knowledge: what improves and what degrades
    when switching from one family to another. These are human-authored
    and cannot be derived from hardware data.
    """

    from_family: str
    to_family: str
    trade_off: str
    improves: List[Bottleneck] = []
    degrades: List[Bottleneck] = []


class FamilyGraph(ExcludeUnsetModel):
    """Soft DAG of instance family trade-off relationships.

    Nodes are FamilyTraits (derived from hardware shapes).
    Edges are FamilyEdges (authored domain knowledge).
    """

    traits: Dict[str, FamilyTrait] = {}
    edges: List[FamilyEdge] = []

    def suggest_alternatives(self, excuse: Excuse) -> List[FamilyEdge]:
        """Return edges to families that improve the excuse's bottleneck."""
        if not excuse.bottleneck:
            return []
        excuse_family = excuse.instance.rsplit(".", 1)[0]
        return [
            e
            for e in self.edges
            if e.from_family == excuse_family and excuse.bottleneck in e.improves
        ]


class ExplainedPlans(ExcludeUnsetModel):
    """Plans + excuses + family context. Both structured and renderable."""

    plans: Sequence[CapacityPlan]
    excuses: Sequence[Excuse] = []
    family_graph: FamilyGraph = FamilyGraph()

    def render(self) -> str:
        """Render a markdown report for human/AI consumption."""
        if not self.excuses:
            return ""

        lines: List[str] = ["## Why shapes were rejected", ""]

        # Group excuses by tag category
        current = [e for e in self.excuses if "current_shape" in e.tags]
        same_fam = [
            e
            for e in self.excuses
            if any(t in e.tags for t in ("same_family",))
            and "current_shape" not in e.tags
        ]
        diff_fam = [e for e in self.excuses if "different_family" in e.tags]
        untagged = [
            e
            for e in self.excuses
            if not e.tags
            or not any(
                t in e.tags
                for t in (
                    "current_shape",
                    "same_family",
                    "different_family",
                )
            )
        ]

        for section_title, section_excuses in [
            ("Current shape", current),
            ("Same family", same_fam),
            ("Different families", diff_fam),
            ("Other", untagged),
        ]:
            if not section_excuses:
                continue
            lines.append(f"### {section_title}")
            for exc in section_excuses:
                line = f"- **{exc.instance}** ({exc.drive}): {exc.reason}"
                if exc.bottleneck:
                    alts = self.family_graph.suggest_alternatives(exc)
                    alt_str = (
                        ", ".join(f"{a.to_family} ({a.trade_off})" for a in alts)
                        if alts
                        else "none"
                    )
                    line += (
                        f"\n  Bottleneck: {exc.bottleneck} | Alternatives: {alt_str}"
                    )
                lines.append(line)
            lines.append("")

        # Family trade-off map
        if self.family_graph.edges:
            lines.append("### Family trade-off map")
            for edge in self.family_graph.edges:
                lines.append(
                    f"- {edge.from_family}"
                    f" --[{', '.join(edge.improves)}]--> "
                    f"{edge.to_family} ({edge.trade_off})"
                )
            lines.append("")

        return "\n".join(lines)

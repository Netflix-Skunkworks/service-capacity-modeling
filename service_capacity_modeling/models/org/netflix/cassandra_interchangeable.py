"""EXPERIMENTAL: Cassandra adapter for interchangeable capacity plans.

Every Cassandra node in a zone owns an equal token range and so the same share
of data and traffic, whatever its instance type. Whether a type can serve a
layout therefore does not depend on the other types in the zone.

A layout is the per-zone node count plus the settings every node in the zone
must share: drive configuration, heap size and compaction threshold. A type
serves a layout when it has the same settings in its own plan and the count is
at least its own planned count. More nodes give each node less work, and the
count must still pass the model's size rounding, ``required_cluster_size`` and
``max_regional_size``.

EBS layouts are limited to a type's own count and exact volume configuration:
the model does not expose volume sizing at an arbitrary count.
"""

from collections.abc import Sequence
from decimal import Decimal
from typing import Any, NamedTuple

from service_capacity_modeling.interface import (
    BufferComponent,
    CapacityDesires,
    CapacityPlan,
)
from service_capacity_modeling.models import RANK_PENALTIES
from service_capacity_modeling.models.common import (
    DerivedBuffers,
    buffer_for_components,
)
from service_capacity_modeling.models.interchangeable import InterchangeableModel
from service_capacity_modeling.models.org.netflix.cassandra import (
    CassandraClusterSizeMode,
    NflxCassandraArguments,
    NflxCassandraCapacityModel,
    _compute_penalties,
    _get_cluster_size_lambda,
    _get_current_capacity,
    _get_current_cluster_size,
    _with_disk_utilization_buffer,
)


class CassandraLayout(NamedTuple):
    zone_count: int
    drive: str
    heap_gib: float
    compaction_threshold: int


def _layout(plan: CapacityPlan) -> CassandraLayout:
    cluster = plan.candidate_clusters.zonal[0]
    if cluster.attached_drives:
        drive = "ebs:" + cluster.attached_drives[0].model_dump_json()
    else:
        assert cluster.instance.drive is not None
        drive = "local:" + str(cluster.instance.drive.drive_type)
    return CassandraLayout(
        cluster.count,
        drive,
        cluster.cluster_params["cassandra.heap.gib"],
        cluster.cluster_params["cassandra.compaction.min_threshold"],
    )


class CassandraInterchangeable(InterchangeableModel[CassandraLayout]):
    def __init__(
        self,
        candidates: dict[str, CapacityPlan],
        desires: CapacityDesires,
        extra_model_arguments: dict[str, Any],
    ) -> None:
        super().__init__(candidates, desires, extra_model_arguments)
        self.args = NflxCassandraArguments.from_extra_model_arguments(
            extra_model_arguments
        )
        self.sized_desires = desires.merge_with(
            NflxCassandraCapacityModel.default_desires(desires, extra_model_arguments)
        )
        self.current = _get_current_capacity(self.sized_desires)
        self.required_count = NflxCassandraCapacityModel.get_required_cluster_size(
            desires.service_tier, extra_model_arguments
        )
        self.round_count = _get_cluster_size_lambda(
            self.args.cluster_size_mode or CassandraClusterSizeMode.unrestricted,
            _get_current_cluster_size(self.sized_desires),
            self.required_count,
        )

    def _max_zone_count(self, plan: CapacityPlan) -> int:
        return self.args.max_regional_size // len(plan.candidate_clusters.zonal)

    def _allowed_zone_count(self, count: int) -> bool:
        """Whether the model's own size rules allow this per-zone count."""
        return self.round_count(count) == count and (
            self.required_count is None or count == self.required_count
        )

    def layouts(self) -> Sequence[CassandraLayout]:
        layouts = set()
        for plan in self.candidates.values():
            own = _layout(plan)
            if own.drive.startswith("ebs:"):
                layouts.add(own)
            else:
                layouts.update(
                    own._replace(zone_count=count)
                    for count in range(own.zone_count, self._max_zone_count(plan) + 1)
                    if self._allowed_zone_count(count)
                )
        return sorted(layouts)

    def serves(self, instance_name: str, layout: CassandraLayout) -> bool:
        plan = self.candidates[instance_name]
        own = _layout(plan)
        if own[1:] != layout[1:]:
            return False
        if own.drive.startswith("ebs:"):
            return own.zone_count == layout.zone_count
        return own.zone_count <= layout.zone_count <= self._max_zone_count(
            plan
        ) and self._allowed_zone_count(layout.zone_count)

    def resize(
        self, plan: CapacityPlan, layout: CassandraLayout, instance_names: Sequence[str]
    ) -> CapacityPlan:
        del instance_names
        clusters = plan.candidate_clusters
        preferred = clusters.zonal[0]
        service_cost = sum(s.annual_cost for s in clusters.services)
        resized = preferred.model_copy(deep=True)
        resized.count = layout.zone_count
        if not resized.attached_drives:
            # Same source the model uses: observed disk when the current
            # cluster reports it, otherwise data size from the desires.
            if (
                self.current is not None
                and self.current.disk_utilization_gib.mid > 0
                and self.args.keyspace_topology is None
            ):
                scale = DerivedBuffers.for_components(
                    self.sized_desires.buffers.derived, [BufferComponent.disk]
                ).scale
                disk_used = (
                    self.current.disk_utilization_gib.mid
                    * self.current.cluster_instance_count.mid
                    * scale
                )
            else:
                buffered = _with_disk_utilization_buffer(
                    self.sized_desires, self.args.max_disk_utilization
                )
                ratio = buffer_for_components(
                    buffered.buffers, [BufferComponent.disk]
                ).ratio
                disk_used = plan.requirements.zonal[0].disk_gib.mid / ratio
            current_family = (
                self.current.cluster_instance.family
                if self.current and self.current.cluster_instance
                else None
            )
            penalties = _compute_penalties(
                instance=resized.instance,
                large_instance_regret=self.args.large_instance_regret,
                data_per_node_gib=disk_used / layout.zone_count,
                ephemeral_maintenance_regret=self.args.ephemeral_maintenance_regret,
                current_family=current_family,
                different_family_regret=self.args.different_family_regret,
            )
            resized.cluster_params.pop(RANK_PENALTIES, None)
            if penalties:
                resized.cluster_params[RANK_PENALTIES] = penalties
        zonal = [resized.model_copy(deep=True) for _ in clusters.zonal]
        infra = NflxCassandraCapacityModel.cluster_costs(
            service_type=NflxCassandraCapacityModel.service_name, zonal_clusters=zonal
        )
        costs = {**infra, **{s.service_type: s.annual_cost for s in clusters.services}}
        updated = clusters.model_copy(
            update={
                "zonal": zonal,
                "annual_costs": {k: Decimal(str(v)) for k, v in costs.items()},
            }
        )
        penalty = sum(resized.cluster_params.get(RANK_PENALTIES, {}).values())
        rank = sum(infra.values()) * (1 + penalty) + service_cost
        return plan.model_copy(update={"candidate_clusters": updated, "rank": rank})

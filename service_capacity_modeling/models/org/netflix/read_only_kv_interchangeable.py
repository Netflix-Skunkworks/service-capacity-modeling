"""EXPERIMENTAL: read-only KV adapter for interchangeable capacity plans.

Every read-only KV node holds the same ``ppn`` partitions and serves the same
share of reads, whatever its instance type, so whether a type can serve a
layout does not depend on the other types in the cluster.

A layout is ``(nodes_for_one_copy, replica_count)``. Type ``t`` serves it when

    ceil(n_partitions / nodes_for_one_copy) <= ppn_cap[t]
    nodes_for_one_copy * replica_count * cores[t] >= cores_needed[t]

where ``ppn_cap[t]`` is the smaller of the partitions that fit on its disk
and, when SST memory placement is on, in its memory; ``cores_needed[t]`` is
already normalized for core frequency and IPC. Layouts are ordered as
``search_for_min_nodes`` breaks ties: fewest nodes, then higher RF, so a type
on its own gets back the planner's layout.
"""

import math
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.models import RANK_PENALTIES
from service_capacity_modeling.models.common import buffer_for_components
from service_capacity_modeling.models.common import EFFECTIVE_DISK_PER_NODE_GIB
from service_capacity_modeling.models.common import upsert_params
from service_capacity_modeling.models.interchangeable import InterchangeableModel
from service_capacity_modeling.models.org.netflix.partition_aware_algorithm import (
    _partitions_that_fit,
    CapacityProblem,
)
from service_capacity_modeling.models.org.netflix.read_only_kv import (
    NflxReadOnlyKVArguments,
    NflxReadOnlyKVCapacityModel,
)

_SST_MEMORY_PLACEMENT = "read-only-kv.sst_memory_placement_enabled"

# (nodes_for_one_copy, replica_count)
ReadOnlyKVLayout = Tuple[int, int]


def ppn_cap(problem: CapacityProblem) -> int:
    """Most partitions one node of the problem's type can hold."""
    disk_ppn = _partitions_that_fit(
        problem.disk_per_node_gib, problem.partition_size_gib
    )
    if problem.memory_per_partition_gib is None:
        return disk_ppn
    assert problem.memory_per_node_gib is not None
    return min(
        disk_ppn,
        _partitions_that_fit(
            problem.memory_per_node_gib, problem.memory_per_partition_gib
        ),
    )


class ReadOnlyKVInterchangeable(InterchangeableModel[ReadOnlyKVLayout]):
    def __init__(
        self,
        candidates: Dict[str, CapacityPlan],
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> None:
        super().__init__(candidates, desires, extra_model_arguments)
        self.args = NflxReadOnlyKVArguments.model_validate(extra_model_arguments)
        # The planner sizes on desires merged with the model's defaults, which
        # supply the disk buffer, so partition size must use the same merge.
        sized_desires = desires.merge_with(
            NflxReadOnlyKVCapacityModel.default_desires(desires, extra_model_arguments)
        )
        partition_size_gib = (
            sized_desires.data_shape.estimated_state_size_gib.mid
            / self.args.total_num_partitions
            * buffer_for_components(
                buffers=sized_desires.buffers, components=[BufferComponent.disk]
            ).ratio
        )
        # Rebuild the problem each single-type plan was sized on.
        self.problems: Dict[str, CapacityProblem] = {}
        for name, plan in candidates.items():
            cluster = plan.candidate_clusters.regional[0]
            params = cluster.cluster_params
            memory_placement = params.get(_SST_MEMORY_PLACEMENT, False)
            self.problems[name] = CapacityProblem(
                n_partitions=self.args.total_num_partitions,
                partition_size_gib=partition_size_gib,
                disk_per_node_gib=params[EFFECTIVE_DISK_PER_NODE_GIB],
                memory_per_partition_gib=(
                    params["read-only-kv.buffered_sst_memory_per_partition_gib"]
                    if memory_placement
                    else None
                ),
                memory_per_node_gib=(
                    params["read-only-kv.available_sst_memory_per_node_gib"]
                    if memory_placement
                    else None
                ),
                cpu_per_node=cluster.instance.cpu,
                cpu_needed=math.ceil(plan.requirements.regional[0].cpu_cores.mid),
                min_rf=self.args.min_replica_count,
                max_nodes=self.args.max_regional_size,
            )
        self.ppn_caps = {name: ppn_cap(p) for name, p in self.problems.items()}

    def layouts(self) -> Sequence[ReadOnlyKVLayout]:
        n_partitions = self.args.total_num_partitions
        max_nodes = self.args.max_regional_size
        layouts: List[ReadOnlyKVLayout] = []
        for n1 in sorted(
            {math.ceil(n_partitions / ppn) for ppn in range(1, n_partitions + 1)}
        ):
            for rf in range(self.args.min_replica_count, max_nodes // n1 + 1):
                layouts.append((n1, rf))
        # At least 2 nodes for availability, as in search_for_min_nodes.
        return sorted(
            layouts, key=lambda layout: (max(layout[0] * layout[1], 2), -layout[1])
        )

    def serves(self, instance_name: str, layout: ReadOnlyKVLayout) -> bool:
        n1, rf = layout
        problem = self.problems[instance_name]
        return (
            math.ceil(problem.n_partitions / n1) <= self.ppn_caps[instance_name]
            and n1 * rf * problem.cpu_per_node >= problem.cpu_needed
        )

    def resize(
        self,
        plan: CapacityPlan,
        layout: ReadOnlyKVLayout,
        instance_names: Sequence[str],
    ) -> CapacityPlan:
        n1, rf = layout
        n_partitions = self.args.total_num_partitions
        # Report the densest ppn every listed type holds within this
        # nodes_for_one_copy step, as search_for_min_nodes reports for one type
        # (for one node per copy that is the full cap, even above n_partitions).
        ppn = min(self.ppn_caps[n] for n in instance_names)
        if n1 > 1:
            ppn = min(ppn, (n_partitions - 1) // (n1 - 1))

        preferred = plan.candidate_clusters.regional[0]
        single_rf = preferred.cluster_params["read-only-kv.replica_count"]
        resized = preferred.model_copy(deep=True)
        resized.count = max(n1 * rf, 2)
        params: Dict[str, Any] = {
            "read-only-kv.replica_count": rf,
            "read-only-kv.partitions_per_node": ppn,
            "read-only-kv.nodes_for_one_copy": n1,
        }
        if preferred.cluster_params.get(_SST_MEMORY_PLACEMENT, False):
            params["read-only-kv.total_buffered_sst_memory_gib"] = (
                preferred.cluster_params[
                    "read-only-kv.buffered_sst_memory_per_copy_gib"
                ]
                * rf
            )
        upsert_params(resized, params)
        clusters = plan.candidate_clusters.model_copy(
            update={
                "regional": [resized],
                "annual_costs": NflxReadOnlyKVCapacityModel.cluster_costs(
                    service_type=NflxReadOnlyKVCapacityModel.service_name,
                    regional_clusters=[resized],
                ),
            }
        )
        # Regional memory scales with RF, as in the model's own plan.
        requirement = plan.requirements.regional[0]
        regional = requirement.model_copy(
            update={"mem_gib": requirement.mem_gib.scale(rf / single_rf)}
        )
        penalty_ratio = sum(resized.cluster_params.get(RANK_PENALTIES, {}).values())
        return plan.model_copy(
            update={
                "candidate_clusters": clusters,
                "requirements": plan.requirements.model_copy(
                    update={"regional": [regional]}
                ),
                "rank": clusters.total_annual_cost * (1 + penalty_ratio),
            }
        )

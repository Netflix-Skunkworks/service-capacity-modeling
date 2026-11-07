"""EXPERIMENTAL: capacity plans with instance fallbacks.

An instance fallback plan is one cluster layout that holds whether each node
launches as the preferred instance type or as any of its fallback types, so an
autoscaling group can launch the fallbacks, in order, when the preferred type
runs short of capacity.

Why the search is exact
-----------------------
A model supplies an :class:`InstanceFallbackAdapter`: the layouts it could deploy
and a test of whether one instance type can serve one layout. The test must not
depend on which other types share the cluster, which holds when every node gets
the same work whatever its hardware (equal partitions, an equal token range).
The types are then interchangeable: a mixed fleet serves a layout exactly when
each of its types does, so the planner walks the layouts once and filters the
types for each one instead of searching combinations of types. Models that
place work in proportion to node capacity cannot satisfy this and should not
register.

Most models need no adapter of their own: :class:`NodeCountAdapter` covers
zonal models whose layout is a node count. A model writes its own adapter only
when a node count does not describe what each node holds.

Choosing fallbacks
------------------
Fallbacks are the smallest serving size of each other family, ordered by the
rank of their own single-type plans. Reaching another family can cost nodes, so
for each preferred type the planner prices every fallback count up to
``max_fallback_types`` at the first layout, in the model's preference order,
that reaches it. It takes the most fallbacks that add at most
``max_fallback_cost_increase`` to the preferred type's own cost, and drops the
preferred type if that is fewer than ``min_fallback_types``. Plans then rank as
the model ranks them. No family is preferred over another.

To remove the experiment, delete this module, the adapters that subclass
:class:`InstanceFallbackAdapter`, ``models/org/netflix/instance_fallback_adapters.py``
and their tests.
"""

from collections.abc import Hashable
from decimal import Decimal
from typing import Any
from typing import Callable
from typing import Dict
from typing import FrozenSet
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from service_capacity_modeling.capacity_planner import CapacityPlanner
from service_capacity_modeling.capacity_planner import PlannerArguments
from service_capacity_modeling.capacity_planner import _resolve_cluster_instances
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import ClusterCapacity
from service_capacity_modeling.interface import Lifecycle
from service_capacity_modeling.interface import ZoneClusterCapacity
from service_capacity_modeling.models.utils import reduce_by_family

# Large enough that the planner returns every single-type candidate.
_ALL_CANDIDATES = 10_000


def planned_cluster(plan: CapacityPlan) -> ClusterCapacity:
    """The plan's first cluster: regional for regional models, else zonal."""
    clusters = plan.candidate_clusters
    return (clusters.regional or clusters.zonal)[0]


class InstanceFallbackOptions(BaseModel):
    min_fallback_types: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Fewest fallback types a plan must have. Preferred types "
        "that cannot reach this within max_fallback_cost_increase are left out; "
        "0 keeps plans with no fallbacks.",
    )
    max_fallback_types: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Most fallback instance types to attach to one plan.",
    )
    max_fallback_cost_increase: float = Field(
        default=0.2,
        ge=0,
        description="Most the plan's annual cost, priced at the preferred "
        "type, may grow to fit the fallbacks, as a fraction of the preferred "
        "type's own plan.",
    )

    @model_validator(mode="after")
    def _min_within_max(self) -> "InstanceFallbackOptions":
        if self.min_fallback_types > self.max_fallback_types:
            raise ValueError("min_fallback_types exceeds max_fallback_types")
        return self


class InstanceFallbackPlan(BaseModel):
    plan: CapacityPlan = Field(
        description="The layout holds on the preferred type and every "
        "fallback type; costs assume the preferred type."
    )
    preferred_instance_type: str
    fallback_instance_types: List[str] = Field(
        description="Exact fallback types in launch preference order, at most "
        "one per family."
    )
    single_type_node_count: int = Field(
        description="Node count of the preferred type's own plan, per zone "
        "for zonal models; the difference from the plan's count is what the "
        "fallbacks add."
    )


class InstanceFallbackPlans(BaseModel):
    plans: List[InstanceFallbackPlan]


class InstanceFallbackAdapter:
    """What a capacity model supplies to plan instance fallbacks.

    ``candidates`` maps each instance type to its best single-type plan, best
    rank first. A layout is any hashable value the adapter chooses; the
    planner only passes layouts back to ``serves`` and ``resize``.
    """

    def __init__(
        self,
        candidates: Dict[str, CapacityPlan],
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> None:
        self.candidates = candidates
        self.desires = desires
        self.extra_model_arguments = extra_model_arguments

    def layouts(self) -> Sequence[Any]:
        """Every deployable layout, ordered the way the model's own planner
        prefers layouts for any single type (usually fewest nodes first)."""
        raise NotImplementedError

    def serves(self, instance_name: str, layout: Any) -> bool:
        """Whether nodes of this type alone can serve the layout. Must not
        depend on the other types in the cluster."""
        raise NotImplementedError

    def resize(
        self, plan: CapacityPlan, layout: Any, instance_names: Sequence[str]
    ) -> CapacityPlan:
        """Apply the layout to the preferred type's plan, re-priced at the
        preferred type and ranked as the model's own ``capacity_plan`` would
        rank it. ``instance_names`` lists the preferred type and its
        fallbacks, all of which serve the layout."""
        raise NotImplementedError


InstanceFallbackAdapterFactory = Callable[
    [Dict[str, CapacityPlan], CapacityDesires, Dict[str, Any]],
    InstanceFallbackAdapter,
]


class NodeCountLayout(NamedTuple):
    zone_count: int
    # Settings every node in the zone shares: the attached drive configuration
    # (one launch template) and the model's shared cluster params.
    shared: Tuple[Hashable, ...]


class NodeCountAdapter(InstanceFallbackAdapter):
    """The adapter for zonal models where a layout is just a node count.

    It fits models that give every node in a zone an equal share of the work
    (an equal token range, consistent hashing, balanced shards). A type whose
    own plan has ``n`` nodes per zone then serves any larger count, since more
    nodes only give each node less work, up to ``max_regional_size``. A count
    beyond a type's own adds no type, so the candidates' own counts are the
    only layouts that matter.

    ``resize`` keeps the preferred type's cluster, changes its count and
    re-prices it. Rank penalties keep the dollar value of the preferred type's
    own plan, so the rank grows by exactly the added cost; penalties that
    shrink with more nodes are slightly overstated.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        candidates: Dict[str, CapacityPlan],
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
        *,
        cluster_type: str,
        max_regional_size: int,
        shared_params: Sequence[str] = (),
        allowed_count: Callable[[int], bool] = lambda count: True,
    ) -> None:
        super().__init__(candidates, desires, extra_model_arguments)
        self.cluster_type = cluster_type
        self.allowed_count = allowed_count
        self.own: Dict[str, NodeCountLayout] = {}
        self.max_zone_count: Dict[str, int] = {}
        for name, plan in candidates.items():
            zones = self._zones(plan)
            cluster = zones[0]
            self.own[name] = NodeCountLayout(
                cluster.count,
                (
                    _drive_signature(cluster),
                    *(cluster.cluster_params.get(key) for key in shared_params),
                ),
            )
            self.max_zone_count[name] = max_regional_size // len(zones)

    def _zones(self, plan: CapacityPlan) -> List[ZoneClusterCapacity]:
        zones = [
            c
            for c in plan.candidate_clusters.zonal
            if c.cluster_type == self.cluster_type
        ]
        assert zones, f"plan has no zonal {self.cluster_type} cluster"
        return zones

    def layouts(self) -> Sequence[NodeCountLayout]:
        counts = {own.zone_count for own in self.own.values()}
        shared = {own.shared for own in self.own.values()}
        return sorted(
            NodeCountLayout(count, s)
            for count in counts
            for s in shared
            if self.allowed_count(count)
        )

    def serves(self, instance_name: str, layout: NodeCountLayout) -> bool:
        own = self.own[instance_name]
        return (
            own.shared == layout.shared
            and own.zone_count
            <= layout.zone_count
            <= self.max_zone_count[instance_name]
            and self.allowed_count(layout.zone_count)
        )

    def resize(
        self, plan: CapacityPlan, layout: NodeCountLayout, instance_names: Sequence[str]
    ) -> CapacityPlan:
        del instance_names
        clusters = plan.candidate_clusters
        zonal = [c.model_copy(deep=True) for c in clusters.zonal]
        resized = [c for c in zonal if c.cluster_type == self.cluster_type]
        for cluster in resized:
            cluster.count = layout.zone_count
        cost_key = f"{self.cluster_type}.zonal-clusters"
        assert cost_key in clusters.annual_costs, cost_key
        updated = clusters.model_copy(
            update={
                "zonal": zonal,
                # model_copy skips validation, so convert to Decimal here.
                "annual_costs": {
                    **clusters.annual_costs,
                    cost_key: Decimal(str(sum(c.annual_cost for c in resized))),
                },
            }
        )
        added = updated.total_annual_cost - clusters.total_annual_cost
        return plan.model_copy(
            update={"candidate_clusters": updated, "rank": plan.rank + added}
        )


def _drive_signature(cluster: ClusterCapacity) -> str:
    if cluster.attached_drives:
        return "attached:" + ",".join(
            d.model_dump_json() for d in cluster.attached_drives
        )
    if cluster.instance.drive is not None:
        return f"local:{cluster.instance.drive.drive_type}"
    return "none"


def _smallest_by_family(
    layout_count: int,
    served: Dict[str, FrozenSet[int]],
    instances: Dict[str, Instance],
    rank: Dict[str, int],
) -> List[Dict[str, str]]:
    """The smallest serving size of each family, per layout index.

    A second size of one family adds little protection: sizes of a family
    share a hardware pool, so they tend to run short together.
    """
    smallest: List[Dict[str, str]] = [{} for _ in range(layout_count)]
    for name in sorted(served, key=lambda n: (instances[n].cpu, rank[n])):
        for i in served[name]:
            smallest[i].setdefault(instances[name].family, name)
    return smallest


def _most_fallbacks_within_budget(
    plan: CapacityPlan,
    choices: List[Tuple[CapacityPlan, List[str]]],
    options: InstanceFallbackOptions,
) -> Optional[Tuple[CapacityPlan, List[str]]]:
    """The choice with the most fallbacks, at least ``min_fallback_types``, that
    adds at most ``max_fallback_cost_increase`` to the preferred type's own plan."""
    budget = plan.candidate_clusters.total_annual_cost * (
        1 + options.max_fallback_cost_increase
    )
    affordable = [
        (p, f)
        for p, f in choices
        if len(f) >= options.min_fallback_types
        and p.candidate_clusters.total_annual_cost <= budget
    ]
    return max(affordable, key=lambda c: len(c[1])) if affordable else None


# pylint: disable-next=too-many-positional-arguments,too-many-locals
def plan_with_instance_fallbacks(
    planner: CapacityPlanner,
    model_name: str,
    model_factory: InstanceFallbackAdapterFactory,
    region: str,
    desires: CapacityDesires,
    extra_model_arguments: Dict[str, Any],
    options: InstanceFallbackOptions,
    lifecycles: Optional[Sequence[Lifecycle]] = None,
    instance_families: Optional[Sequence[str]] = None,
    num_results: int = 3,
    max_results_per_family: int = 1,
    num_regions: int = 3,
) -> InstanceFallbackPlans:
    """Plan a model and give each plan fallback instance types.

    Every single-type candidate is a possible fallback, including candidates
    that ranking and family reduction drop from the returned plans.
    """
    singles = planner.plan_certain(
        model_name=model_name,
        region=region,
        desires=desires,
        lifecycles=lifecycles,
        instance_families=instance_families,
        num_results=_ALL_CANDIDATES,
        num_regions=num_regions,
        extra_model_arguments=extra_model_arguments,
        # Fallbacks exist to ride out capacity shortages, so no family is
        # preferred; instance_families is the way to exclude families.
        planner_arguments=PlannerArguments(
            max_results_per_family=_ALL_CANDIDATES, preferred_family_penalty=0
        ),
    )
    # The planner returns plans best rank first, so the first plan for each
    # instance type is its best option.
    candidates: Dict[str, CapacityPlan] = {}
    for plan in singles:
        candidates.setdefault(planned_cluster(plan).instance.name, plan)
    if not candidates:
        return InstanceFallbackPlans(plans=[])

    # Give the adapter the desires the model saw: the planner resolves current
    # clusters given only by instance name on its own copy of the desires.
    resolved = desires.model_copy(deep=True)
    _resolve_cluster_instances(resolved)
    model = model_factory(candidates, resolved, extra_model_arguments)
    layouts = model.layouts()
    instances = {
        name: planned_cluster(plan).instance for name, plan in candidates.items()
    }
    rank = {name: i for i, name in enumerate(candidates)}
    served: Dict[str, FrozenSet[int]] = {
        name: frozenset(
            i for i, layout in enumerate(layouts) if model.serves(name, layout)
        )
        for name in candidates
    }
    smallest_by_family = _smallest_by_family(len(layouts), served, instances, rank)

    target = options.max_fallback_types
    revised: List[InstanceFallbackPlan] = []
    for name, plan in candidates.items():
        family = instances[name].family
        # Each fallback count, priced at the first layout that reaches it.
        # Layout indices follow the model's preference.
        choices: List[Tuple[CapacityPlan, List[str]]] = [(plan, [])]
        for i in sorted(served[name]):
            reach = min(target, len(smallest_by_family[i]) - 1)
            if reach < len(choices):
                continue
            others = sorted(
                (n for f, n in smallest_by_family[i].items() if f != family),
                key=rank.__getitem__,
            )
            for count in range(len(choices), reach + 1):
                fallbacks = others[:count]
                choices.append(
                    (model.resize(plan, layouts[i], [name, *fallbacks]), fallbacks)
                )
            if reach == target:
                break
        picked = _most_fallbacks_within_budget(plan, choices, options)
        if picked is None:
            continue
        chosen, fallbacks = picked
        revised.append(
            InstanceFallbackPlan(
                plan=chosen,
                preferred_instance_type=name,
                fallback_instance_types=fallbacks,
                single_type_node_count=planned_cluster(plan).count,
            )
        )

    revised.sort(
        key=lambda p: (p.plan.rank, p.plan.candidate_clusters.total_annual_cost)
    )
    by_plan = {id(p.plan): p for p in revised}
    kept = reduce_by_family(
        [p.plan for p in revised], max_results_per_family=max_results_per_family
    )[:num_results]
    return InstanceFallbackPlans(plans=[by_plan[id(plan)] for plan in kept])

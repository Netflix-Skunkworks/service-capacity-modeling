"""EXPERIMENTAL: interchangeable capacity plans.

An interchangeable plan is one cluster layout that holds whether each node
launches as the preferred instance type or as any of its fallback types, so an
autoscaling group can launch the fallbacks, in order, when the preferred type
runs short of capacity.

Why the search is exact
-----------------------
A model supplies an :class:`InterchangeableModel`: the layouts it could deploy
and a test of whether one instance type can serve one layout. The test must not
depend on which other types share the cluster, which holds when every node gets
the same work whatever its hardware (equal partitions, an equal token range).
A mixed fleet then serves a layout exactly when each of its types does, so the
planner walks the layouts once and filters the types for each one instead of
searching combinations of types. Models that place work in proportion to node
capacity cannot satisfy this and should not register.

Choosing fallbacks
------------------
Fallbacks are the smallest serving size of each other family, ordered by the
rank of their own single-type plans. Reaching another family can cost nodes, so
for each preferred type the planner prices every fallback count up to
``max_fallback_types`` at the first layout, in the model's preference order,
that reaches it. It takes the most fallbacks that add at most
``max_fallback_cost`` to the preferred type's own cost, and drops the
preferred type if that is fewer than ``min_fallback_types``. Plans then rank as
the model ranks them. No family is preferred over another.

To remove the experiment, delete this module, the model adapters that subclass
:class:`InterchangeableModel`, ``models/org/netflix/interchangeable_models.py``
and their tests.
"""

from typing import Any
from typing import Callable
from typing import Dict
from typing import FrozenSet
from typing import List
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
from service_capacity_modeling.models.utils import reduce_by_family

# Large enough that the planner returns every single-type candidate.
_ALL_CANDIDATES = 10_000


def planned_cluster(plan: CapacityPlan) -> ClusterCapacity:
    """The plan's first cluster: regional for regional models, else zonal."""
    clusters = plan.candidate_clusters
    return (clusters.regional or clusters.zonal)[0]


class InterchangeableOptions(BaseModel):
    min_fallback_types: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Fewest fallback types a plan must have. Preferred types "
        "that cannot reach this within max_fallback_cost are left out.",
    )
    max_fallback_types: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Most fallback instance types to attach to one plan.",
    )
    max_fallback_cost: float = Field(
        default=0.2,
        ge=0,
        description="Most the fallbacks may add to the preferred type's own "
        "annual cost, as a fraction of it.",
    )

    @model_validator(mode="after")
    def _min_within_max(self) -> "InterchangeableOptions":
        if self.min_fallback_types > self.max_fallback_types:
            raise ValueError("min_fallback_types exceeds max_fallback_types")
        return self


class InterchangeablePlan(BaseModel):
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


class InterchangeablePlans(BaseModel):
    plans: List[InterchangeablePlan]


class InterchangeableModel:
    """What a capacity model supplies to plan interchangeable instance types.

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


InterchangeableModelFactory = Callable[
    [Dict[str, CapacityPlan], CapacityDesires, Dict[str, Any]],
    InterchangeableModel,
]


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
    options: InterchangeableOptions,
) -> Optional[Tuple[CapacityPlan, List[str]]]:
    """The choice with the most fallbacks, at least ``min_fallback_types``, that
    adds at most ``max_fallback_cost`` to the preferred type's own plan."""
    budget = plan.candidate_clusters.total_annual_cost * (1 + options.max_fallback_cost)
    affordable = [
        (p, f)
        for p, f in choices
        if len(f) >= options.min_fallback_types
        and p.candidate_clusters.total_annual_cost <= budget
    ]
    return max(affordable, key=lambda c: len(c[1])) if affordable else None


# pylint: disable-next=too-many-positional-arguments,too-many-locals
def plan_interchangeable(
    planner: CapacityPlanner,
    model_name: str,
    model_factory: InterchangeableModelFactory,
    region: str,
    desires: CapacityDesires,
    extra_model_arguments: Dict[str, Any],
    options: InterchangeableOptions,
    lifecycles: Optional[Sequence[Lifecycle]] = None,
    instance_families: Optional[Sequence[str]] = None,
    num_results: int = 3,
    max_results_per_family: int = 1,
    num_regions: int = 3,
) -> InterchangeablePlans:
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
        return InterchangeablePlans(plans=[])

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
    revised: List[InterchangeablePlan] = []
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
            InterchangeablePlan(
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
    return InterchangeablePlans(plans=[by_plan[id(plan)] for plan in kept])

"""EXPERIMENTAL: interchangeable capacity plans.

An interchangeable plan is one cluster layout that holds whether each node
launches as the preferred instance type or as any of its fallback types. An
autoscaling group can then launch the fallbacks, in order, when the preferred
type runs short of capacity. Fallbacks exist only to ride out those shortages,
so cost never decides whether a type is a fallback.

How the search stays exact
--------------------------
A model supplies an :class:`InterchangeableModel`: the layouts it could deploy
and a test of whether one instance type can serve one layout. The test must not
depend on which other types share the cluster. That holds when the system gives
every node the same work whatever its hardware (equal partitions, an equal
token range, an equal traffic share). A mixed fleet then serves a layout exactly
when each of its types does, so instead of searching combinations of types the
planner walks the layouts once and filters the types for each one.

For each preferred type, the chosen layout is the model's most preferred layout
that the preferred type serves and that serves the most other families, up to
``max_fallback_types``. The fallbacks are the smallest serving size in each of
those families, ordered by the rank of their own single-type plans. Plans are
then ranked by their re-sized cost, so cost still picks the plan.

Models that place work in proportion to node capacity cannot satisfy the
contract and should not register.

To remove the experiment, delete this module, the model adapters that subclass
:class:`InterchangeableModel`, ``models/org/netflix/interchangeable_models.py``
and their tests.
"""

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Hashable
from typing import List
from typing import Optional
from typing import Sequence
from typing import TypeVar

from pydantic import BaseModel
from pydantic import Field

from service_capacity_modeling.capacity_planner import CapacityPlanner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import ClusterCapacity
from service_capacity_modeling.interface import Lifecycle
from service_capacity_modeling.models.utils import reduce_by_family

# Large enough that the planner returns every single-type candidate.
_ALL_CANDIDATES = 10_000

Layout = TypeVar("Layout", bound=Hashable)


def planned_cluster(plan: CapacityPlan) -> ClusterCapacity:
    """The plan's first cluster: regional for regional models, else zonal."""
    clusters = plan.candidate_clusters
    return (clusters.regional or clusters.zonal)[0]


class InterchangeableOptions(BaseModel):
    max_fallback_types: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Most fallback instance types to attach to one plan. This "
        "limits how many alternatives a plan lists, not how many nodes they "
        "add; compare the plan's count with single_type_node_count.",
    )


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


class DominanceEdge(BaseModel):
    """``stronger`` serves every layout ``weaker`` serves."""

    weaker: str
    stronger: str


class InterchangeablePlans(BaseModel):
    plans: List[InterchangeablePlan]
    dominance_edges: List[DominanceEdge] = Field(
        description="Transitive reduction of which candidate types can replace "
        "which with no extra nodes. An edge proves a fallback is free; a "
        "missing edge does not prove it costs nodes."
    )


class InterchangeableModel(ABC, Generic[Layout]):
    """What a capacity model supplies to plan interchangeable instance types.

    ``candidates`` maps each instance type to its best single-type plan, best
    rank first.
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

    @abstractmethod
    def layouts(self) -> Sequence[Layout]:
        """Every deployable layout, ordered the way the model's own planner
        prefers layouts for any single type (usually fewest nodes first)."""

    @abstractmethod
    def serves(self, instance_name: str, layout: Layout) -> bool:
        """Whether nodes of this type alone can serve the layout. Must not
        depend on the other types in the cluster."""

    @abstractmethod
    def resize(
        self, plan: CapacityPlan, layout: Layout, instance_names: Sequence[str]
    ) -> CapacityPlan:
        """Apply the layout to the preferred type's plan, re-priced and
        re-ranked at the preferred type. ``instance_names`` lists the preferred
        type and its fallbacks, all of which serve the layout."""


InterchangeableModelFactory = Callable[
    [Dict[str, CapacityPlan], CapacityDesires, Dict[str, Any]],
    InterchangeableModel[Any],
]


def dominance_edges(served: Dict[str, FrozenSet[int]]) -> List[DominanceEdge]:
    """Transitive reduction of layout-set inclusion between instance types.

    Types that serve exactly the same layouts are ordered by name so the
    graph stays acyclic.
    """
    names = sorted(served)
    above = {
        weaker: {
            stronger
            for stronger in names
            if stronger != weaker
            and served[weaker] <= served[stronger]
            and (served[weaker] != served[stronger] or stronger > weaker)
        }
        for weaker in names
    }
    return [
        DominanceEdge(weaker=weaker, stronger=stronger)
        for weaker in names
        for stronger in sorted(above[weaker])
        if not any(stronger in above[middle] for middle in above[weaker])
    ]


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
        max_results_per_family=_ALL_CANDIDATES,
    )
    # The planner returns plans best rank first, so the first plan for each
    # instance type is its best option.
    candidates: Dict[str, CapacityPlan] = {}
    for plan in singles:
        candidates.setdefault(planned_cluster(plan).instance.name, plan)
    if not candidates:
        return InterchangeablePlans(plans=[], dominance_edges=[])

    model = model_factory(candidates, desires, extra_model_arguments)
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
    # The smallest serving size of each family, per layout. A second size of
    # one family adds little protection: sizes of a family share a hardware
    # pool, so they tend to run short together.
    smallest_by_family: List[Dict[str, str]] = [{} for _ in layouts]
    for name in sorted(candidates, key=lambda n: (instances[n].cpu, rank[n])):
        for i in served[name]:
            smallest_by_family[i].setdefault(instances[name].family, name)

    wanted = options.max_fallback_types
    revised: List[InterchangeablePlan] = []
    for name, plan in candidates.items():
        family = instances[name].family
        single_type_node_count = planned_cluster(plan).count
        best: Optional[int] = None
        best_count = 0
        # Layout indices follow the model's preference, so the first layout
        # that reaches the most families wins.
        for i in sorted(served[name]):
            count = min(wanted, len(smallest_by_family[i]) - 1)
            if best is None or count > best_count:
                best, best_count = i, count
            if count == wanted:
                break
        fallbacks: List[str] = []
        if best is not None and best_count > 0:
            others = [n for f, n in smallest_by_family[best].items() if f != family]
            fallbacks = sorted(others, key=rank.__getitem__)[:wanted]
            plan = model.resize(plan, layouts[best], [name, *fallbacks])
        revised.append(
            InterchangeablePlan(
                plan=plan,
                preferred_instance_type=name,
                fallback_instance_types=fallbacks,
                single_type_node_count=single_type_node_count,
            )
        )

    revised.sort(
        key=lambda p: (p.plan.rank, p.plan.candidate_clusters.total_annual_cost)
    )
    by_plan = {id(p.plan): p for p in revised}
    kept = reduce_by_family(
        [p.plan for p in revised], max_results_per_family=max_results_per_family
    )[:num_results]
    return InterchangeablePlans(
        plans=[by_plan[id(plan)] for plan in kept],
        dominance_edges=dominance_edges(served),
    )

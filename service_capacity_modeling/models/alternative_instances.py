"""Choose alternative instance types that can run one shared layout.

An autoscaling group that cannot launch its primary instance type launches the
alternative types its mixed instances policy lists. That is safe only if every
listed type can run the same layout with the same attached drives. The planner
plans every type at each layout the model offers
(``CapacityModel.candidate_layouts``) and this module picks, for each primary
type, the layout and alternatives to use.

Alternatives follow a policy, not a search for the cheapest mixed fleet: at most
one per family, priced at the layout and listed cheapest first, within the
limits in ``AlternativeInstanceOptions``.
"""

from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import ClusterCapacity
from service_capacity_modeling.interface import Excuse
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import AlternativeInstanceOptions


def planned_cluster(plan: CapacityPlan) -> ClusterCapacity:
    """The plan's first cluster: regional for regional models, else zonal."""
    clusters = plan.candidate_clusters
    return (clusters.regional or clusters.zonal)[0]


def _cost(plan: CapacityPlan) -> float:
    """Cluster cost: the plan's total without services such as network and
    backup, which do not change with the instance type or node count."""
    clusters = plan.candidate_clusters
    return float(clusters.total_annual_cost) - sum(
        s.annual_cost for s in clusters.services
    )


def _drives(plan: CapacityPlan) -> List[str]:
    """The planned cluster's attached drives, which one launch template fixes."""
    return [d.model_dump_json() for d in planned_cluster(plan).attached_drives]


def _with_alternatives(
    plan: CapacityPlan, instances: Sequence[Instance]
) -> CapacityPlan:
    """A copy of the plan whose clusters list ``instances`` as alternatives."""
    clusters = plan.candidate_clusters
    update = {"alternative_instances": list(instances)}
    return plan.model_copy(
        update={
            "candidate_clusters": clusters.model_copy(
                update={
                    "zonal": [c.model_copy(update=update) for c in clusters.zonal],
                    "regional": [
                        c.model_copy(update=update) for c in clusters.regional
                    ],
                }
            )
        }
    )


def _cheapest_alternative_per_family(
    at_layout: Dict[str, CapacityPlan],
    primary: CapacityPlan,
    max_alternative_cost: float,
    rank: Dict[str, int],
    limit: int,
) -> List[CapacityPlan]:
    """Up to ``limit`` alternatives for ``primary`` at this layout: one type per
    other family, the one with the fewest CPUs, with the same attached drives
    and cost at most ``max_alternative_cost``, listed cheapest first.

    A second size of one family adds little protection: sizes of a family share
    a hardware pool, so they tend to run short together."""
    family = planned_cluster(primary).instance.family
    drives = _drives(primary)
    by_family: Dict[str, CapacityPlan] = {}
    for name in sorted(
        at_layout,
        key=lambda n: (planned_cluster(at_layout[n]).instance.cpu, rank[n]),
    ):
        plan = at_layout[name]
        other = planned_cluster(plan).instance.family
        if (
            other != family
            and _drives(plan) == drives
            and _cost(plan) <= max_alternative_cost
        ):
            by_family.setdefault(other, plan)
    return sorted(
        by_family.values(),
        key=lambda p: (_cost(p), rank[planned_cluster(p).instance.name]),
    )[:limit]


def choose_alternative_instances(
    own_plans: Dict[str, CapacityPlan],
    plans_by_layout: Sequence[Dict[str, CapacityPlan]],
    options: AlternativeInstanceOptions,
) -> Tuple[List[CapacityPlan], List[Excuse]]:
    """Each primary type's plan at a shared layout, listing its alternatives.

    ``own_plans`` maps each instance type to its single-type plan, best rank first.
    ``plans_by_layout`` maps, per layout in the model's preference order, each type
    to its plan at that layout. For each primary type this keeps the layout,
    within ``max_plan_cost_increase`` of its own plan, that reaches the most
    alternatives, the earliest on a tie. A primary type with fewer than
    ``min_alternatives`` is dropped with an Excuse.
    """
    rank = {name: i for i, name in enumerate(own_plans)}
    chosen: List[CapacityPlan] = []
    excuses: List[Excuse] = []
    for name, plan in own_plans.items():
        max_plan_cost = _cost(plan) * (1 + options.max_plan_cost_increase)
        max_alternative_cost = _cost(plan) * (1 + options.max_alternative_cost_increase)
        best: Optional[Tuple[CapacityPlan, List[CapacityPlan]]] = None
        for at_layout in plans_by_layout:
            primary = at_layout.get(name)
            if primary is None or _cost(primary) > max_plan_cost:
                continue
            alternatives = _cheapest_alternative_per_family(
                {n: p for n, p in at_layout.items() if n in rank},
                primary,
                max_alternative_cost,
                rank,
                options.max_alternatives,
            )
            if best is None or len(alternatives) > len(best[1]):
                best = (primary, alternatives)
            if len(alternatives) == options.max_alternatives:
                break
        if best is None or len(best[1]) < options.min_alternatives:
            excuses.append(
                Excuse(
                    instance=name,
                    drive=planned_cluster(plan).attached_drives[0].name
                    if planned_cluster(plan).attached_drives
                    else "",
                    reason=(
                        f"Fewer than {options.min_alternatives} alternative "
                        "types within the cost limits"
                    ),
                    context={"alternatives": len(best[1]) if best else 0},
                    bottleneck=Bottleneck.cost,
                )
            )
            continue
        primary, alternatives = best
        instances = [planned_cluster(p).instance for p in alternatives]
        chosen.append(_with_alternatives(primary, instances))
    return chosen, excuses

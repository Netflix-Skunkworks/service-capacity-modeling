from typing import Optional

from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRegretParameters
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import Instance


class CapacityModel:
    """Stateless interface for defining a capacity model

    To define a capacity model you must implement two pure functions.

    The first, `capacity_plan` method calculates the best possible
    CapacityPlan (cluster layout) given a concrete `Instance` shape (cpu,
    memory, disk etc ...) and attachable `Drive` (e.g. cloud drives like GP2)
    along with a concrete instead of CapacityDesires (qps, latency, etc ...).

    The second, `regret` method calculates a relative regret (higher is worse)
    of two capacity plans. For example if the optimal plan computed by
    `capacity_plan` would allocate `100` CPUs and the proposed plan only
    allocates `50` we regret that choice as we are under-provisioning.

    """

    def __init__(self):
        pass

    @staticmethod
    def capacity_plan(
        instance: Instance, drive: Drive, desires: CapacityDesires, **kwargs
    ) -> Optional[CapacityPlan]:
        # quiet pylint
        (_, _, _) = (instance, drive, desires)

    @staticmethod
    def regret(
        regret_params: CapacityRegretParameters,
        optimal_plan: CapacityPlan,
        proposed_plan: CapacityPlan,
    ) -> float:
        # Under an assumption that cost is a reasonable single dimension
        # to compare to clusters on, we penalize under-provisioned (cheap)
        # clusters more than expensive ones.
        optimal_cost = optimal_plan.candidate_clusters.total_annual_cost.mid
        plan_cost = proposed_plan.candidate_clusters.total_annual_cost.mid

        if plan_cost >= optimal_cost:
            return (plan_cost - optimal_cost) * regret_params.over_provision_cost
        else:
            return (optimal_cost - plan_cost) * regret_params.under_provision_cost

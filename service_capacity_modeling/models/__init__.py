from typing import Optional

from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRegretParameters
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import QueryPattern


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
        return None

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

    @staticmethod
    def description() -> str:
        return "No description"

    @staticmethod
    def default_desires(user_desires: CapacityDesires):
        if user_desires.query_pattern.access_pattern == AccessPattern.latency:
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.latency,
                    estimated_mean_read_latency_ms=certain_float(1),
                    estimated_mean_write_latency_ms=certain_float(1),
                    # "Single digit milliseconds"
                    read_latency_slo_ms=FixedInterval(
                        low=0.4, mid=4, high=10, confidence=0.98
                    ),
                    write_latency_slo_ms=FixedInterval(
                        low=0.4, mid=4, high=10, confidence=0.98
                    ),
                ),
                data_shape=DataShape(),
            )
        else:
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.throughput,
                    estimated_mean_read_latency_ms=certain_float(10),
                    estimated_mean_write_latency_ms=certain_float(20),
                    # "Tens of milliseconds"
                    read_latency_slo_ms=FixedInterval(
                        low=10, mid=50, high=100, confidence=0.98
                    ),
                    write_latency_slo_ms=FixedInterval(
                        low=10, mid=50, high=100, confidence=0.98
                    ),
                )
            )

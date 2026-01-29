from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRegretParameters
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import Consistency
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import GlobalConsistency
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Platform
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import ClusterCapacity
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import ServiceCapacity

__all__ = [
    "AccessConsistency",
    "AccessPattern",
    "CapacityDesires",
    "CapacityPlan",
    "CapacityRegretParameters",
    "certain_float",
    "Consistency",
    "DataShape",
    "Drive",
    "FixedInterval",
    "GlobalConsistency",
    "Instance",
    "Platform",
    "QueryPattern",
    "RegionContext",
    "CapacityModel",
    "CostAwareModel",
]

__common_regrets__ = frozenset(("spend", "disk", "mem"))


def _disk_regret(  # noqa: C901
    regret_params: CapacityRegretParameters,
    optimal_plan: CapacityPlan,
    proposed_plan: CapacityPlan,
) -> float:
    # type -> disk
    optimal_disk = {}
    plan_disk = {}

    for zonal in optimal_plan.requirements.zonal:
        typ = zonal.requirement_type
        if typ not in optimal_disk:
            optimal_disk[typ] = 0.0
        optimal_disk[typ] += zonal.disk_gib.mid

    for regional in optimal_plan.requirements.regional:
        typ = regional.requirement_type
        if typ not in optimal_disk:
            optimal_disk[typ] = 0.0
        optimal_disk[typ] += regional.disk_gib.mid

    for zonal in proposed_plan.requirements.zonal:
        typ = zonal.requirement_type
        if typ not in plan_disk:
            plan_disk[typ] = 0.0
        plan_disk[typ] += zonal.disk_gib.mid

    for regional in proposed_plan.requirements.regional:
        typ = regional.requirement_type
        if typ not in plan_disk:
            plan_disk[typ] = 0.0
        plan_disk[typ] += regional.disk_gib.mid

    # We regret not having the disk space for a dataset, but do not
    # regret lacking disk space
    regret = 0.0
    for typ in optimal_disk.keys() | plan_disk.keys():
        od, pd = optimal_disk.get(typ, 0), plan_disk.get(typ, 0)
        if od > pd:
            regret += (
                (od - pd) * regret_params.disk.under_provision_cost
            ) ** regret_params.disk.exponent
    return regret


class CostAwareModel:
    """Mixin for models that implement cost calculation methods.

    Models using this mixin MUST define:
        service_name: str  # prefix for cost keys (e.g., "cassandra")
        cluster_type: str  # filters which clusters this model costs

    Example:
        class MyModel(CapacityModel, CostAwareModel):
            service_name = "myservice"
            cluster_type = "myservice"
    """

    # Subclasses MUST override these (validated in register_model)
    service_name: str
    cluster_type: str

    @staticmethod
    def service_costs(
        service_type: str,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> List[ServiceCapacity]:
        """Calculate additional service costs (network, backup, etc)."""
        raise NotImplementedError(
            f"service_costs() must be implemented by {service_type} model"
        )

    @staticmethod
    def cluster_costs(
        service_type: str,
        zonal_clusters: Sequence["ClusterCapacity"] = (),
        regional_clusters: Sequence["ClusterCapacity"] = (),
    ) -> Dict[str, float]:
        """Calculate cluster infrastructure costs (instances, drives)."""
        raise NotImplementedError(
            f"cluster_costs() must be implemented by {service_type} model"
        )


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

    def __init__(self) -> None:
        pass

    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        """Given a concrete hardware shape and desires, return a candidate

        This is the only required method on this interface. Your model
        must either:
            * Return None to indicate there is no viable use of the
              instance/drive which meets the user's desires
            * Return a CapapacityPlan containing the model's calculation
              of how much CPU/RAM/disk etc ... that is required and
        """
        # quiet pylint
        (_, _, _, _, _) = (instance, drive, context, desires, extra_model_arguments)
        return None

    @staticmethod
    def regret(
        regret_params: CapacityRegretParameters,
        optimal_plan: CapacityPlan,
        proposed_plan: CapacityPlan,
    ) -> Dict[str, float]:
        """Optional cost model for how much we regret a choice

        After the capacity planner has simulated a bunch of possible outcomes
        We need to evaluate each cluster against the optimal choice for a
        given requirement.

        Our default model of cost is just related to money and disk footprint:

        Under an assumption that cost is a reasonable single dimension
        to compare to clusters on, we penalize under-provisioned (cheap)
        clusters more than expensive ones.

        :return: A componentized regret function where the total regret is
        the sum of all componenets. This is not just a single number so as you
        develop more complex regret functions you can debug why clusters are
        or are not being chosen
        """
        regrets = {"spend": 0.0, "disk": 0.0, "mem": 0.0}

        # Have to subtract out service costs which should be proprotional
        # to the input not the output.
        optimal_service_spend = sum(
            s.annual_cost
            for s in optimal_plan.candidate_clusters.services
            if s.regret_cost is False
        )
        optimal_cost = (
            float(optimal_plan.candidate_clusters.total_annual_cost)
            - optimal_service_spend
        )
        plan_service_spend = sum(
            s.annual_cost
            for s in proposed_plan.candidate_clusters.services
            if s.regret_cost is False
        )
        plan_cost = (
            float(proposed_plan.candidate_clusters.total_annual_cost)
            - plan_service_spend
        )

        if "spend" in optimal_plan.requirements.regrets:
            if plan_cost >= optimal_cost:
                regrets["spend"] = (
                    (plan_cost - optimal_cost) * regret_params.spend.over_provision_cost
                ) ** regret_params.spend.exponent
            else:
                regrets["spend"] = (
                    (optimal_cost - plan_cost)
                    * regret_params.spend.under_provision_cost
                ) ** regret_params.spend.exponent

        if "disk" in optimal_plan.requirements.regrets:
            regrets["disk"] = _disk_regret(
                regret_params=regret_params,
                optimal_plan=optimal_plan,
                proposed_plan=proposed_plan,
            )

        if "mem" in optimal_plan.requirements.regrets:
            optimal_mem = sum(
                req.mem_gib.mid for req in optimal_plan.requirements.zonal
            )
            optimal_mem += sum(
                req.mem_gib.mid for req in optimal_plan.requirements.regional
            )

            plan_mem = sum(req.mem_gib.mid for req in proposed_plan.requirements.zonal)
            plan_mem += sum(
                req.mem_gib.mid for req in proposed_plan.requirements.regional
            )

            # Running out of memory is particularly costly because it often
            # can cause an outage that is hard to get out of. We do not regret
            # too much memory
            if optimal_mem > plan_mem:
                regrets["mem"] = (
                    (optimal_mem - plan_mem) * regret_params.mem.under_provision_cost
                ) ** regret_params.mem.exponent

        for regret in optimal_plan.requirements.regrets:
            if regret not in __common_regrets__:
                regrets["regret"] = optimal_plan.requirements.regret(
                    name=regret, optimal_plan=optimal_plan, proposed_plan=proposed_plan
                )

        return regrets

    @staticmethod
    def description() -> str:
        """Optional description of the model"""
        return "No description"

    @staticmethod
    def extra_model_arguments_schema() -> Dict[str, Any]:
        """Optional json schema of extra keyword arguments

        Some models might take additional arguments to capacity_plan.
        They can convey that context to callers here along with a
        description of each argument
        """
        return {"type": "object"}

    @staticmethod
    def run_hardware_simulation() -> bool:
        """Optional to skip hardware simulation

        Some models, managed services, do not
        require simulating through hardware (instances, drives).

        """
        return True

    @staticmethod
    def compose_with(
        user_desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Tuple[Tuple[str, Callable[[CapacityDesires], CapacityDesires]], ...]:
        """Return additional model names to compose with this one

        The second element of the tuple is a capacity desire transform that
        takes the original user desire and modifies it for the composed
        model.

        (("model1", lambda x: x),
         ("model2", lambda x: transform(x)))

        Often used for dependencies.
        """
        # quiet pylint
        (_, _) = user_desires, extra_model_arguments
        return tuple()

    @staticmethod
    def allowed_platforms() -> Tuple[Platform, ...]:
        """Return which platforms this model accepts.

        Most software can run on amd64 (Intel and AMD), but some models might
        accept others
        """
        return (Platform.amd64,)

    @staticmethod
    def allowed_cloud_drives() -> Tuple[Optional[str], ...]:
        """Return which cloud drives this model accepts.

        This will _override_ the lifecycle of the drive.

        Note that an empty tuple means _all_ drives. If you want to
        only accept local hard drives emit None
        """
        return tuple()

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> CapacityDesires:
        """Optional defaults to apply given a user desires

        Often users do not know what the on-cpu time of their queries
        will be, but the models often have a good idea. For example
        databases usually have some range of on-cpu time for point queries
        (latency access) versus throughput.

        This is also a good place to throw ValueError on AccessPattern
        or AccessConsistency that cannot be met.
        """
        _ = extra_model_arguments

        unlikely_consistency_models = (
            AccessConsistency.linearizable,
            AccessConsistency.serializable,
        )
        target = (
            user_desires.query_pattern.access_consistency.same_region.target_consistency
        )
        if target in unlikely_consistency_models:
            raise ValueError(
                f"Most services do not support {unlikely_consistency_models}"
            )

        if user_desires.query_pattern.access_pattern == AccessPattern.latency:
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.latency,
                    access_consistency=GlobalConsistency(
                        same_region=Consistency(
                            target_consistency=AccessConsistency.read_your_writes,
                            staleness_slo_sec=FixedInterval(low=0, mid=0.1, high=1),
                        ),
                        cross_region=Consistency(
                            target_consistency=AccessConsistency.best_effort,
                            staleness_slo_sec=FixedInterval(low=10, mid=60, high=600),
                        ),
                    ),
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
                    access_consistency=GlobalConsistency(
                        same_region=Consistency(
                            target_consistency=AccessConsistency.read_your_writes,
                            staleness_slo_sec=FixedInterval(low=0, mid=0.1, high=1),
                        ),
                        cross_region=Consistency(
                            target_consistency=AccessConsistency.best_effort,
                            staleness_slo_sec=FixedInterval(low=10, mid=60, high=600),
                        ),
                    ),
                    estimated_mean_read_latency_ms=certain_float(2),
                    estimated_mean_write_latency_ms=certain_float(4),
                    # "Tens of milliseconds"
                    read_latency_slo_ms=FixedInterval(
                        low=10, mid=50, high=100, confidence=0.98
                    ),
                    write_latency_slo_ms=FixedInterval(
                        low=10, mid=50, high=100, confidence=0.98
                    ),
                ),
                data_shape=DataShape(),
            )

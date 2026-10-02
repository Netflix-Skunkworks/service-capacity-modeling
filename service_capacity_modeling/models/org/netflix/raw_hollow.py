from typing import Any
from typing import Callable
from typing import Dict
from typing import FrozenSet
from typing import Optional
from typing import Tuple

from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRequirement
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import RegionClusterCapacity
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.models import CapacityModel


def _logkeeper_instance(desires: CapacityDesires) -> str:
    if desires.data_shape.estimated_state_size_gib.mid < 10:
        return "r6a.large"
    return "r6a.xlarge"


def _writer_instance(desires: CapacityDesires) -> str:
    if desires.query_pattern.estimated_write_per_second.mid < 1_000:
        return "r6a.xlarge"
    return "r6a.2xlarge"


def _producer_instance(desires: CapacityDesires) -> str:
    if desires.data_shape.estimated_state_size_gib.mid < 100:
        return "r6a.2xlarge"
    return "r6a.4xlarge"


def _explorer_instance(desires: CapacityDesires) -> str:
    if desires.data_shape.estimated_state_size_gib.mid < 100:
        return "r6a.large"
    return "r6a.xlarge"


def _raw_hollow_role_plan(
    role: str,
    count: int,
    target_instance: Callable[[CapacityDesires], str],
    *,
    instance: Instance,
    drive: Drive,
    desires: CapacityDesires,
) -> Optional[CapacityPlan]:
    if drive.name != "gp2" or instance.name != target_instance(desires):
        return None

    cluster_type = f"rawhollow-{role}"
    cluster = RegionClusterCapacity(
        cluster_type=cluster_type,
        count=count,
        instance=instance,
    )
    requirement = CapacityRequirement(
        requirement_type=cluster_type,
        reference_shape=instance,
        cpu_cores=certain_int(instance.cpu * count),
        mem_gib=certain_float(instance.ram_gib * count),
        network_mbps=certain_float(instance.net_mbps * count),
    )
    return CapacityPlan(
        requirements=Requirements(
            regional=[requirement],
            regrets=("spend", "mem"),
        ),
        candidate_clusters=Clusters(
            annual_costs={
                f"{cluster_type}.regional-clusters": cluster.annual_cost,
            },
            regional=[cluster],
        ),
    )


class NflxRawHollowRoleCapacityModel(CapacityModel):
    @staticmethod
    def description() -> str:
        return "Netflix RawHollow Role Model"

    @staticmethod
    def preferred_families() -> Optional[FrozenSet[str]]:
        return frozenset(("r6a",))

    @staticmethod
    def allowed_cloud_drives() -> Tuple[Optional[str], ...]:
        return ("gp2",)


class NflxRawHollowLogkeeperCapacityModel(NflxRawHollowRoleCapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        _ = (context, extra_model_arguments)
        return _raw_hollow_role_plan(
            "logkeeper",
            5,
            _logkeeper_instance,
            instance=instance,
            drive=drive,
            desires=desires,
        )


class NflxRawHollowWriterCapacityModel(NflxRawHollowRoleCapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        _ = (context, extra_model_arguments)
        return _raw_hollow_role_plan(
            "writer",
            2,
            _writer_instance,
            instance=instance,
            drive=drive,
            desires=desires,
        )


class NflxRawHollowProducerCapacityModel(NflxRawHollowRoleCapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        _ = (context, extra_model_arguments)
        return _raw_hollow_role_plan(
            "producer",
            1,
            _producer_instance,
            instance=instance,
            drive=drive,
            desires=desires,
        )


class NflxRawHollowExplorerCapacityModel(NflxRawHollowRoleCapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        _ = (context, extra_model_arguments)
        return _raw_hollow_role_plan(
            "explorer",
            1,
            _explorer_instance,
            instance=instance,
            drive=drive,
            desires=desires,
        )


class NflxRawHollowCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        _ = (instance, drive, context, desires, extra_model_arguments)
        return None

    @staticmethod
    def compose_with(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> Tuple[Tuple[str, Callable[[CapacityDesires], CapacityDesires]], ...]:
        _ = (user_desires, extra_model_arguments)
        return (
            ("org.netflix.raw-hollow.logkeeper", lambda desires: desires),
            ("org.netflix.raw-hollow.writer", lambda desires: desires),
            ("org.netflix.raw-hollow.producer", lambda desires: desires),
            ("org.netflix.raw-hollow.explorer", lambda desires: desires),
        )

    @staticmethod
    def description() -> str:
        return "Netflix RawHollow Model"


nflx_raw_hollow_capacity_model = NflxRawHollowCapacityModel()
nflx_raw_hollow_logkeeper_capacity_model = NflxRawHollowLogkeeperCapacityModel()
nflx_raw_hollow_writer_capacity_model = NflxRawHollowWriterCapacityModel()
nflx_raw_hollow_producer_capacity_model = NflxRawHollowProducerCapacityModel()
nflx_raw_hollow_explorer_capacity_model = NflxRawHollowExplorerCapacityModel()

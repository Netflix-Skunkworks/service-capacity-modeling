from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

from .aurora import nflx_aurora_capacity_model
from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import Consistency
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import GlobalConsistency
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import Platform
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.models import CapacityModel


class NflxPostgresCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        if desires.service_tier == 0:
            return None

        plan = None
        if Platform.aurora_postgres in instance.platforms:
            plan = nflx_aurora_capacity_model.capacity_plan(
                instance=instance,
                drive=drive,
                context=context,
                desires=desires,
                extra_model_arguments=extra_model_arguments,
            )
        return plan

    @staticmethod
    def description() -> str:
        return "Netflix Postgres Model"

    @staticmethod
    def allowed_platforms() -> Tuple[Platform, ...]:
        return Platform.aurora_postgres, Platform.amd64

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> CapacityDesires:
        return CapacityDesires(
            query_pattern=QueryPattern(
                access_pattern=AccessPattern.latency,
                access_consistency=GlobalConsistency(
                    same_region=Consistency(
                        target_consistency=AccessConsistency.serializable_stale,
                    ),
                    cross_region=Consistency(
                        target_consistency=AccessConsistency.never,
                    ),
                ),
                # can't really make latency/throughput trade-offs with RDS
                estimated_mean_read_size_bytes=Interval(
                    low=128, mid=1024, high=65536, confidence=0.90
                ),
                estimated_mean_write_size_bytes=Interval(
                    low=64, mid=512, high=2048, confidence=0.90
                ),
                estimated_mean_read_latency_ms=Interval(
                    low=1, mid=4, high=100, confidence=0.90
                ),
                estimated_mean_write_latency_ms=Interval(
                    low=1, mid=6, high=200, confidence=0.90
                ),
                read_latency_slo_ms=FixedInterval(
                    minimum_value=1,
                    maximum_value=100,
                    low=1,
                    mid=10,
                    high=20,
                    confidence=0.98,
                ),
                write_latency_slo_ms=FixedInterval(
                    minimum_value=1,
                    maximum_value=100,
                    low=1,
                    mid=10,
                    high=20,
                    confidence=0.98,
                ),
            ),
            # Assume that the working set is between 20% by default
            data_shape=DataShape(
                estimated_working_set_percent=Interval(
                    low=0.05, mid=0.10, high=0.20, confidence=0.8
                )
            ),
        )


nflx_postgres_capacity_model = NflxPostgresCapacityModel()

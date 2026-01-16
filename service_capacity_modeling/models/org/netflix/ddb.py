import math
from typing import Any
from typing import Dict
from typing import Optional

from pydantic import BaseModel
from pydantic import Field

from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRequirement
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import Consistency
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import GlobalConsistency
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.interface import Service
from service_capacity_modeling.interface import ServiceCapacity
from service_capacity_modeling.models import CapacityModel

_TIER_TARGET_UTILIZATION_MAPPING = {
    0: 0.20,
    1: 0.40,
    2: 0.60,
    3: 0.80,
}


class NflxDynamoDBArguments(BaseModel):
    transactional_write_percent: float = Field(
        default=0,
        description="Coordinated writes, if multiple items in single write "
        "then all-or-nothing guarantee for writes. "
        "Default is zero, assuming most writes are single item puts",
    )
    transactional_read_percent: float = Field(
        default=0,
        description="Coordinated reads, if multiple items in "
        "single read then all-or-nothing guarantee for read. "
        "Default is zero",
    )
    target_max_annual_cost: float = Field(
        default=0,
        description="Target to determine max capacity units for auto-scale. "
        "Model does best effort to predict the capacity units but "
        "does not guarantee that actual costs will be less than "
        "what is specified here. Base costs based on provided requirements "
        "itself can exceed the target, if base costs are lower then "
        "actual costs can be 2x-3x of target",
    )


class _ReadPlan(BaseModel):
    read_capacity_units: int = Field(
        default=0,
        description="read capacity units needed to support the requested reads",
    )
    total_annual_read_cost: float = Field(
        default=0,
        description="annual cost to consume the provisioned read capacity units",
    )


class _WritePlan(BaseModel):
    write_capacity_units: int = Field(
        default=0,
        description="write capacity units needed to support the requested writes",
    )
    replicated_write_capacity_units: int = Field(
        default=0,
        description="write capacity units needed to "
        "support the requested writes for global tables",
    )
    total_annual_write_cost: float = Field(
        default=0,
        description="annual cost to consume the provisioned write capacity units",
    )


class _DataStoragePlan(BaseModel):
    total_data_storage_gib: float = Field(
        default=0,
        description="total amount of data stored in gib",
    )
    total_annual_data_storage_cost: float = Field(
        default=0,
        description="annual cost to store the requested data",
    )


class _DataBackupPlan(BaseModel):
    total_backup_data_storage_gib: float = Field(
        default=0,
        description="total amount of data stored for backup in gib",
    )
    total_annual_backup_cost: float = Field(
        default=0,
        description="annual cost to backup the stored data",
    )


class _DataTransferPlan(BaseModel):
    total_data_transfer_gib: float = Field(
        default=0,
        description="amount of data transferred, "
        "included the global table cross region replication",
    )
    total_annual_data_transfer_cost: float = Field(
        default=0,
        description="annual cost for data transfer",
    )


def _get_read_consistency_percentages(
    desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
) -> Dict[str, float]:
    # either all the required attributes are supplied in extra_model_arguments
    # or is derived based on the access consistency
    eventual_read_percent = extra_model_arguments.get("eventual_read_percent", 0.0)
    assert eventual_read_percent >= 0.0
    transactional_read_percent = extra_model_arguments.get(
        "transactional_read_percent", 0.0
    )
    assert transactional_read_percent >= 0.0
    strong_read_percent = extra_model_arguments.get("strong_read_percent", 0.0)
    assert strong_read_percent >= 0.0
    total_percent = (
        eventual_read_percent + transactional_read_percent + strong_read_percent
    )
    if total_percent == 0:
        access_consistency = desires.query_pattern.access_consistency.same_region
        target_consistency = access_consistency.target_consistency
        if target_consistency == AccessConsistency.serializable:
            transactional_read_percent = 1.0
            eventual_read_percent = 0.0
            strong_read_percent = 0.0
        elif target_consistency in (
            AccessConsistency.read_your_writes,
            AccessConsistency.linearizable,
        ):
            transactional_read_percent = 0.0
            eventual_read_percent = 0.0
            strong_read_percent = 1.0
        else:
            transactional_read_percent = 0.0
            eventual_read_percent = 1.0
            strong_read_percent = 0.0
    total_percent = (
        eventual_read_percent + transactional_read_percent + strong_read_percent
    )
    assert total_percent == 1, (
        "eventual_read_percent, transactional_read_percent, strong_read_percent"
        " should sum to 1"
    )
    return {
        "transactional_read_percent": transactional_read_percent,
        "eventual_read_percent": eventual_read_percent,
        "strong_read_percent": strong_read_percent,
    }


def _get_write_consistency_percentages(
    desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
) -> Dict[str, float]:
    # either all the required attributes are supplied in extra_model_arguments
    # or is derived based on the access consistency
    transactional_write_percent: float = extra_model_arguments.get(
        "transactional_write_percent", 0
    )
    assert transactional_write_percent >= 0.0
    non_transactional_write_percent: float = extra_model_arguments.get(
        "non_transactional_write_percent", 0
    )
    assert non_transactional_write_percent >= 0.0
    total_percent = transactional_write_percent + non_transactional_write_percent
    if total_percent == 0:
        access_consistency = desires.query_pattern.access_consistency.same_region
        target_consistency = access_consistency.target_consistency
        if target_consistency == AccessConsistency.serializable:
            transactional_write_percent = 1.0
            non_transactional_write_percent = 0.0
        else:
            transactional_write_percent = 0.0
            non_transactional_write_percent = 1.0
    total_percent = transactional_write_percent + non_transactional_write_percent
    assert total_percent == 1, (
        "transactional_write_percent, non_transactional_write_percent should sum to 1"
    )
    return {
        "transactional_write_percent": transactional_write_percent,
        "non_transactional_write_percent": non_transactional_write_percent,
    }


def _mean_write_item_size_bytes(desires: CapacityDesires) -> float:
    mean_item_size = desires.query_pattern.estimated_mean_write_size_bytes.mid
    return mean_item_size


def _mean_read_item_size_bytes(desires: CapacityDesires) -> float:
    mean_item_size = desires.query_pattern.estimated_mean_read_size_bytes.mid
    return mean_item_size


def _get_dynamo_standard(context: RegionContext) -> Service:
    number_of_regions = context.num_regions
    dynamo_service = (
        context.services.get("dynamo.standard.global", None)
        if number_of_regions > 1
        else context.services.get("dynamo.standard", None)
    )
    if not dynamo_service:
        raise ValueError("DynamoDB Service is not available in context")
    return dynamo_service


def _get_dynamo_backup(context: RegionContext) -> Service:
    dynamo_backup = context.services.get("dynamo.backup.continuous", None)
    if not dynamo_backup:
        raise ValueError("DynamoDB Backup is not available in context")
    return dynamo_backup


def _get_dynamo_transfer(
    context: RegionContext,
) -> Service:
    transfer_costs = context.services.get("dynamo.transfer", None)
    if not transfer_costs:
        raise ValueError("DynamoDB Transfer cost is not available in context")
    return transfer_costs


def _plan_writes(
    context: RegionContext,
    desires: CapacityDesires,
    extra_model_arguments: Dict[str, Any],
) -> _WritePlan:
    mean_item_size = _mean_write_item_size_bytes(desires)

    # For items up to 1 KB in size,
    # one WCU can perform one standard write request per second
    rounded_wcus_per_item = math.ceil(max(1.0, mean_item_size / 1024))

    write_percentages = _get_write_consistency_percentages(
        desires, extra_model_arguments
    )

    transactional_write_percent = write_percentages["transactional_write_percent"]
    baseline_wcus_non_transactional = (
        desires.query_pattern.estimated_write_per_second.mid
        * (1 - transactional_write_percent)
        * rounded_wcus_per_item
    )

    # Transactional write requests require two WCUs
    baseline_wcus_transactional = (
        desires.query_pattern.estimated_write_per_second.mid
        * transactional_write_percent
        * 2
        * rounded_wcus_per_item
    )
    # we do not break down reserved vs non-reserved
    total_baseline_wcus = math.ceil(
        baseline_wcus_non_transactional + baseline_wcus_transactional
    )

    # 8760 hours in a year (365 * 24)
    total_baseline_wcus_hours = 8760 * total_baseline_wcus
    number_of_regions = context.num_regions

    # pick the right pricing based on context
    dynamo_service_standard = _get_dynamo_standard(context)

    # for global replication each region will additionally consume rWCUs
    total_annual_cost_wcu = round(
        (
            number_of_regions
            * total_baseline_wcus_hours
            * dynamo_service_standard.annual_cost_per_write_io
        ),
        2,
    )
    if number_of_regions > 1:
        return _WritePlan(
            replicated_write_capacity_units=total_baseline_wcus_hours,
            write_capacity_units=0,
            total_annual_write_cost=total_annual_cost_wcu,
        )
    return _WritePlan(
        write_capacity_units=total_baseline_wcus_hours,
        replicated_write_capacity_units=0,
        total_annual_write_cost=total_annual_cost_wcu,
    )


def _plan_reads(
    context: RegionContext,
    desires: CapacityDesires,
    extra_model_arguments: Dict[str, Any],
) -> _ReadPlan:
    read_percentages = _get_read_consistency_percentages(desires, extra_model_arguments)
    transactional_read_percent = read_percentages["transactional_read_percent"]
    eventual_read_percent = read_percentages["eventual_read_percent"]
    strong_read_percent = read_percentages["strong_read_percent"]
    mean_item_size = _mean_read_item_size_bytes(desires)

    # items up to 4 KB in size
    rounded_rcus_per_item = math.ceil(max(1.0, mean_item_size / (4 * 1024)))
    estimated_read_per_second = desires.query_pattern.estimated_read_per_second.mid

    # one RCU can perform two eventually consistent read
    baseline_rcus_eventually_consistent = (
        estimated_read_per_second * eventual_read_percent * 0.5 * rounded_rcus_per_item
    )

    # one RCU can perform one strongly consistent read
    baseline_rcus_strong_consistent = (
        estimated_read_per_second * strong_read_percent * 1 * rounded_rcus_per_item
    )

    # two RCUs can perform one transactional read
    baseline_rcus_transact = (
        estimated_read_per_second
        * transactional_read_percent
        * 2
        * rounded_rcus_per_item
    )

    total_baseline_rcus = math.ceil(
        baseline_rcus_eventually_consistent
        + baseline_rcus_strong_consistent
        + baseline_rcus_transact
    )

    # 8760 hours in a year (365 * 24)
    total_baseline_rcus_hours = 8760 * total_baseline_rcus
    dynamo_service_standard = _get_dynamo_standard(context)
    total_annual_cost_rcu_hour = round(
        (total_baseline_rcus_hours * dynamo_service_standard.annual_cost_per_read_io), 2
    )
    return _ReadPlan(
        read_capacity_units=total_baseline_rcus_hours,
        total_annual_read_cost=total_annual_cost_rcu_hour,
    )


def _plan_storage(
    context: RegionContext,
    desires: CapacityDesires,
) -> _DataStoragePlan:
    dynamo_service_standard = _get_dynamo_standard(context)
    annual_storage_cost = round(
        (
            dynamo_service_standard.annual_cost_gib(
                math.ceil(desires.data_shape.estimated_state_size_gib.mid)
            )
        ),
        2,
    )
    return _DataStoragePlan(
        total_data_storage_gib=round(
            desires.data_shape.estimated_state_size_gib.mid, 2
        ),
        total_annual_data_storage_cost=annual_storage_cost,
    )


def _plan_data_transfer(
    context: RegionContext,
    desires: CapacityDesires,
) -> _DataTransferPlan:
    number_of_regions = context.num_regions
    if not number_of_regions > 1:
        return _DataTransferPlan(
            total_data_transfer_gib=0, total_annual_data_transfer_cost=0
        )
    mean_item_size_bytes = _mean_write_item_size_bytes(desires)
    writes_per_second = desires.query_pattern.estimated_write_per_second.mid
    # 31,536,000 seconds in a year (365 * 24 * 60 * 60)
    # 1024 * 1024 * 1024 = 1Gib
    annual_data_written_gib = round(
        ((31536000 / (1024 * 1024 * 1024)) * writes_per_second * mean_item_size_bytes),
        2,
    )
    transfer_costs = _get_dynamo_transfer(context)
    annual_transfer_cost_to_another_region = transfer_costs.annual_cost_gib(
        annual_data_written_gib
    )
    annual_transfer_cost = round(
        annual_transfer_cost_to_another_region * (number_of_regions - 1), 2
    )
    return _DataTransferPlan(
        total_data_transfer_gib=annual_data_written_gib * (number_of_regions - 1),
        total_annual_data_transfer_cost=annual_transfer_cost,
    )


def _plan_backup(
    context: RegionContext,
    desires: CapacityDesires,
) -> _DataBackupPlan:
    number_of_regions = context.num_regions
    dynamo_backup_continuous = _get_dynamo_backup(context)
    annual_pitr_cost = dynamo_backup_continuous.annual_cost_gib(
        math.ceil(desires.data_shape.estimated_state_size_gib.mid)
    )
    # normalizing the cost as pitr is not charged per region
    annual_pitr_cost_normalized = round((annual_pitr_cost / number_of_regions), 2)
    return _DataBackupPlan(
        total_backup_data_storage_gib=round(
            desires.data_shape.estimated_state_size_gib.mid, 2
        ),
        total_annual_backup_cost=annual_pitr_cost_normalized,
    )


class NflxDynamoDBCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        # refer https://aws.amazon.com/dynamodb/pricing/provisioned/
        write_plan = _plan_writes(context, desires, extra_model_arguments)
        read_plan = _plan_reads(context, desires, extra_model_arguments)
        storage_plan = _plan_storage(context, desires)
        backup_plan = _plan_backup(context, desires)
        data_transfer_plan = _plan_data_transfer(context, desires)
        target_max_annual_cost: float = extra_model_arguments.get(
            "target_max_annual_cost", 0
        )
        target_util_percentage = 0.80
        if desires.service_tier in _TIER_TARGET_UTILIZATION_MAPPING:
            target_util_percentage = _TIER_TARGET_UTILIZATION_MAPPING[
                desires.service_tier
            ]

        requirement_context = {
            "read_capacity_units": read_plan.read_capacity_units,
            "write_capacity_units": write_plan.write_capacity_units,
            "data_transfer_gib": data_transfer_plan.total_data_transfer_gib,
            "target_utilization_percentage": target_util_percentage,
        }
        requirement_context["replicated_write_capacity_units"] = (
            write_plan.replicated_write_capacity_units
        )

        dynamo_costs = {
            "dynamo.regional-writes": write_plan.total_annual_write_cost,
            "dynamo.regional-reads": read_plan.total_annual_read_cost,
            "dynamo.regional-storage": storage_plan.total_annual_data_storage_cost,
        }

        dynamo_costs["dynamo.regional-transfer"] = (
            data_transfer_plan.total_annual_data_transfer_cost
        )

        dynamo_costs["dynamo.data-backup"] = backup_plan.total_annual_backup_cost

        total_annual_costs = round(sum(dynamo_costs.values()), 2)
        total_write_capacity_units = (
            write_plan.write_capacity_units + write_plan.replicated_write_capacity_units
        )
        max_read_capacity_units = max(1, read_plan.read_capacity_units)
        max_write_capacity_units = max(1, total_write_capacity_units)

        if target_max_annual_cost > 0:
            requirement_context["target_max_annual_cost"] = target_max_annual_cost
            annual_balance_target = target_max_annual_cost - total_annual_costs
            if annual_balance_target > 0:
                max_read_capacity_units += math.ceil(
                    (annual_balance_target / max(1.0, read_plan.total_annual_read_cost))
                    * read_plan.read_capacity_units,
                )
                max_write_capacity_units += math.ceil(
                    (
                        annual_balance_target
                        / max(1.0, write_plan.total_annual_write_cost)
                    )
                    * total_write_capacity_units,
                )

        requirement = CapacityRequirement(
            requirement_type="dynamo-regional",
            cpu_cores=certain_int(0),
            disk_gib=certain_float(storage_plan.total_data_storage_gib),
            context=requirement_context,
        )
        dynamo_services = [
            ServiceCapacity(
                service_type="dynamo.standard",
                annual_cost=(
                    write_plan.total_annual_write_cost
                    + read_plan.total_annual_read_cost
                    + storage_plan.total_annual_data_storage_cost
                    + data_transfer_plan.total_annual_data_transfer_cost
                ),
                service_params={
                    "read_capacity_units": {
                        "estimated": read_plan.read_capacity_units,
                        "auto_scale": {
                            "min": 1,
                            "max": max_read_capacity_units,
                            "target_utilization_percentage": target_util_percentage,
                        },
                    },
                    "write_capacity_units": {
                        "estimated": (
                            write_plan.write_capacity_units
                            + write_plan.replicated_write_capacity_units
                        ),
                        "auto_scale": {
                            "min": 1,
                            "max": max_write_capacity_units,
                            "target_utilization_percentage": target_util_percentage,
                        },
                    },
                },
            ),
            ServiceCapacity(
                service_type="dynamo.backup",
                annual_cost=backup_plan.total_annual_backup_cost,
            ),
        ]

        clusters = Clusters(
            annual_costs=dynamo_costs,
            zonal=[],
            regional=[],
            services=dynamo_services,
        )

        return CapacityPlan(
            requirements=Requirements(regional=[requirement]),
            candidate_clusters=clusters,
        )

    @staticmethod
    def description() -> str:
        return "Netflix Streaming DynamoDB Model"

    @staticmethod
    def extra_model_arguments_schema() -> Dict[str, Any]:
        return NflxDynamoDBArguments.model_json_schema()

    @staticmethod
    def run_hardware_simulation() -> bool:
        return False

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> CapacityDesires:
        acceptable_consistency = {
            "same_region": {
                None,
                AccessConsistency.serializable,
                AccessConsistency.linearizable,
                AccessConsistency.best_effort,
                AccessConsistency.eventual,
                AccessConsistency.read_your_writes,
                AccessConsistency.never,
            },
            "cross_region": {
                None,
                # https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/globaltables_HowItWorks.html#V2globaltables_HowItWorks.CommonTasks
                AccessConsistency.linearizable_stale,
                AccessConsistency.best_effort,
                AccessConsistency.eventual,
                AccessConsistency.never,
            },
        }

        for key, value in user_desires.query_pattern.access_consistency:
            if value.target_consistency not in acceptable_consistency[key]:
                raise ValueError(
                    f"DynamoDB can only provide {acceptable_consistency[key]} access."
                    f"User asked for {key}={value}"
                )

        if user_desires.query_pattern.access_pattern == AccessPattern.latency:
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.latency,
                    access_consistency=GlobalConsistency(
                        same_region=Consistency(
                            target_consistency=AccessConsistency.read_your_writes,
                        ),
                        cross_region=Consistency(
                            target_consistency=AccessConsistency.eventual,
                        ),
                    ),
                    estimated_mean_read_size_bytes=Interval(
                        low=128, mid=1024, high=65536, confidence=0.95
                    ),
                    estimated_mean_write_size_bytes=Interval(
                        low=64, mid=256, high=1024, confidence=0.95
                    ),
                    estimated_mean_read_latency_ms=Interval(
                        low=5, mid=10, high=15, confidence=0.98
                    ),
                    estimated_mean_write_latency_ms=Interval(
                        low=5, mid=10, high=15, confidence=0.98
                    ),
                    # Assume point queries, "Single digit milliseconds SLO"
                    read_latency_slo_ms=FixedInterval(
                        minimum_value=2,
                        maximum_value=20,
                        low=5,
                        mid=10,
                        high=15,
                        confidence=0.98,
                    ),
                    write_latency_slo_ms=FixedInterval(
                        minimum_value=2,
                        maximum_value=20,
                        low=5,
                        mid=10,
                        high=15,
                        confidence=0.98,
                    ),
                ),
                # Most latency sensitive clusters are in the
                # < 1TiB range
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=10, mid=100, high=1000, confidence=0.98
                    ),
                ),
            )
        else:
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.throughput,
                    access_consistency=GlobalConsistency(
                        same_region=Consistency(
                            target_consistency=AccessConsistency.read_your_writes,
                        ),
                        cross_region=Consistency(
                            target_consistency=AccessConsistency.eventual,
                        ),
                    ),
                    estimated_mean_read_size_bytes=Interval(
                        low=512, mid=2048, high=131072, confidence=0.95
                    ),
                    estimated_mean_write_size_bytes=Interval(
                        low=128, mid=1024, high=65536, confidence=0.95
                    ),
                    estimated_mean_read_latency_ms=Interval(
                        low=10, mid=20, high=30, confidence=0.98
                    ),
                    estimated_mean_write_latency_ms=Interval(
                        low=10, mid=20, high=30, confidence=0.98
                    ),
                    read_latency_slo_ms=FixedInterval(
                        minimum_value=5,
                        maximum_value=100,
                        low=10,
                        mid=40,
                        high=90,
                        confidence=0.98,
                    ),
                    write_latency_slo_ms=FixedInterval(
                        minimum_value=5,
                        maximum_value=100,
                        low=10,
                        mid=40,
                        high=90,
                        confidence=0.98,
                    ),
                ),
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=100, mid=1000, high=4000, confidence=0.98
                    ),
                ),
            )


nflx_ddb_capacity_model = NflxDynamoDBCapacityModel()

import math
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from pydantic import BaseModel
from pydantic import Field

from service_capacity_modeling.interface import AccessConsistency, certain_float
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRequirement
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


class NflxDynamoDBArguments(BaseModel):
    number_of_regions: int = Field(
        default=1,
        description="How many regions the dynamoDB table needs to be created. "
        "This determines if we should use wcu's "
        "or rwcu's and to correctly estimate the transfer costs",
    )
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
    estimated_mean_item_size_bytes: int = Field(
        default=0,
        description="Estimated avg item size. If not supplies, "
        "then estimated_mean_write_size_bytes from desires will "
        "be used",
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


def _mean_item_size_bytes(
    desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
) -> float:
    mean_item_size = extra_model_arguments.get("estimated_mean_item_size_bytes", 0)
    if mean_item_size == 0:
        mean_item_size = desires.query_pattern.estimated_mean_write_size_bytes.mid
    return mean_item_size


def _get_num_regions(extra_model_arguments: Dict[str, Any]) -> int:
    return extra_model_arguments.get("number_of_regions", 1)


def _get_dynamo_standard(
    context: RegionContext, extra_model_arguments: Dict[str, Any]
) -> Service:
    number_of_regions = _get_num_regions(extra_model_arguments)
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
) -> List[Tuple[float, float]]:
    transfer_costs = context.services.get("dynamo.transfer", None)
    if not transfer_costs:
        raise ValueError(
            "DynamoDB Transfer cost is not available in context"
        )
    return transfer_costs.annual_cost_per_gib


def _plan_writes(
    context: RegionContext,
    desires: CapacityDesires,
    extra_model_arguments: Dict[str, Any],
) -> _WritePlan:
    # For items up to 1 KB in size,
    # one WCU can perform one standard write request per second

    mean_item_size = _mean_item_size_bytes(desires, extra_model_arguments)

    rounded_wcus_per_item = math.ceil(max(1.0, mean_item_size / 1024))

    transactional_write_percent: float = extra_model_arguments.get(
        "transactional_write_percent", 0
    )
    assert transactional_write_percent >= 0.0
    baseline_wcus_non_transactional = (
        desires.query_pattern.estimated_write_per_second.mid
        * (1 - transactional_write_percent)
        * rounded_wcus_per_item
    )
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

    # 8760 hours in a year
    total_baseline_wcus_hours = 8760 * total_baseline_wcus
    number_of_regions = _get_num_regions(extra_model_arguments)
    dynamo_service_standard = _get_dynamo_standard(context, extra_model_arguments)

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
    eventual_consistency_percent = extra_model_arguments.get(
        "eventual_read_percent", 1.0
    )
    assert eventual_consistency_percent >= 0.0
    transactional_read_percent = extra_model_arguments.get(
        "transactional_read_percent", 0.0
    )
    assert transactional_read_percent >= 0.0
    if (
        desires.query_pattern.access_consistency.same_region.target_consistency
        == AccessConsistency.read_your_writes
    ):
        eventual_consistency_percent = 0.0
    strong_consistency_percent = (
        1 - eventual_consistency_percent - transactional_read_percent
    )
    assert strong_consistency_percent >= 0.0
    mean_item_size = _mean_item_size_bytes(desires, extra_model_arguments)
    rounded_rcus_per_item = math.ceil(max(1.0, mean_item_size / (4 * 1024)))
    estimated_read_per_second = desires.query_pattern.estimated_read_per_second.mid
    baseline_rcus_eventually_consistent = (
        estimated_read_per_second
        * eventual_consistency_percent
        * 0.5
        * rounded_rcus_per_item
    )
    baseline_rcus_strong_consistent = (
        estimated_read_per_second
        * strong_consistency_percent
        * 1
        * rounded_rcus_per_item
    )
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

    # 8760 hours in a year
    total_baseline_rcus_hours = 8760 * total_baseline_rcus
    dynamo_service_standard = _get_dynamo_standard(context, extra_model_arguments)
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
    extra_model_arguments: Dict[str, Any],
) -> _DataStoragePlan:
    dynamo_service_standard = _get_dynamo_standard(context, extra_model_arguments)
    annual_storage_cost = round(
        (
            math.ceil(desires.data_shape.estimated_state_size_gib.mid)
            * dynamo_service_standard.annual_cost_per_gib[0][1]
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
    extra_model_arguments: Dict[str, Any],
) -> _DataTransferPlan:
    number_of_regions = _get_num_regions(extra_model_arguments)
    if not number_of_regions > 1:
        return _DataTransferPlan(
            total_data_transfer_gib=0, total_annual_data_transfer_cost=0
        )
    mean_item_size_bytes = _mean_item_size_bytes(desires, extra_model_arguments)
    writes_per_second = desires.query_pattern.estimated_write_per_second.mid
    # 31,536,000 seconds in a year
    annual_data_written_gib = round(
        ((31536000 / (1024 * 1024 * 1024)) * writes_per_second * mean_item_size_bytes),
        2,
    )
    _annual_data_written = annual_data_written_gib
    transfer_costs = _get_dynamo_transfer(context)
    annual_transfer_cost_to_another_region = 0.0
    for transfer_cost in transfer_costs:
        if not _annual_data_written > 0:
            break
        if transfer_cost[0] > 0:
            annual_transfer_cost_to_another_region += (
                min(_annual_data_written, transfer_cost[0]) * transfer_cost[1]
            )
            _annual_data_written -= transfer_cost[0]
        else:
            # final remaining data transfer cost
            annual_transfer_cost_to_another_region += (
                _annual_data_written * transfer_cost[1]
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
    extra_model_arguments: Dict[str, Any],
) -> _DataBackupPlan:
    number_of_regions = _get_num_regions(extra_model_arguments)
    dynamo_backup_continuous = _get_dynamo_backup(context)
    annual_pitr_cost = (
        math.ceil(desires.data_shape.estimated_state_size_gib.mid)
        * dynamo_backup_continuous.annual_cost_per_gib[0][1]
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
        write_plan = _plan_writes(context, desires, extra_model_arguments)
        read_plan = _plan_reads(context, desires, extra_model_arguments)
        storage_plan = _plan_storage(context, desires, extra_model_arguments)
        backup_plan = _plan_backup(context, desires, extra_model_arguments)
        data_transfer_plan = _plan_data_transfer(
            context, desires, extra_model_arguments
        )
        requirement_context = {
            "read_capacity_units": read_plan.read_capacity_units,
            "write_capacity_units": write_plan.write_capacity_units,
            "data_transfer_gib": data_transfer_plan.total_data_transfer_gib,
        }
        requirement_context[
            "replicated_write_capacity_units"
        ] = write_plan.replicated_write_capacity_units
        requirement = CapacityRequirement(
            requirement_type="dynamo-regional",
            core_reference_ghz=0,
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
                    "read_capacity_units": read_plan.read_capacity_units,
                    "write_capacity_units": (
                        write_plan.write_capacity_units
                        + write_plan.replicated_write_capacity_units
                    ),
                },
            ),
            ServiceCapacity(
                service_type="dynamo.backup",
                annual_cost=backup_plan.total_annual_backup_cost,
            ),
        ]
        dynamo_costs = {
            "dynamo.regional-writes": write_plan.total_annual_write_cost,
            "dynamo.regional-reads": read_plan.total_annual_read_cost,
            "dynamo.regional-storage": storage_plan.total_annual_data_storage_cost,
        }

        dynamo_costs[
            "dynamo.regional-transfer"
        ] = data_transfer_plan.total_annual_data_transfer_cost

        dynamo_costs["dynamo.data-backup"] = backup_plan.total_annual_backup_cost

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
    def description():
        return "Netflix Streaming DynamoDB Model"

    @staticmethod
    def extra_model_arguments_schema() -> Dict[str, Any]:
        return NflxDynamoDBArguments.schema()

    @staticmethod
    def run_hardware_simulation() -> bool:
        return False

    @staticmethod
    def default_desires(user_desires, extra_model_arguments: Dict[str, Any]):
        acceptable_consistency = {
            "same_region": {
                None,
                AccessConsistency.best_effort,
                AccessConsistency.eventual,
                AccessConsistency.read_your_writes,
                AccessConsistency.never,
            },
            "cross_region": {
                None,
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

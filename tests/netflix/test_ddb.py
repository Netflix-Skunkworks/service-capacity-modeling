from decimal import Decimal

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import Consistency
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import GlobalConsistency
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern

storage_only_desires = CapacityDesires(
    service_tier=0,
    query_pattern=QueryPattern(
        access_consistency=GlobalConsistency(
            same_region=Consistency(
                target_consistency=AccessConsistency.read_your_writes
            )
        ),
        estimated_read_per_second=certain_int(0),
        estimated_write_per_second=certain_int(0),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(low=10, mid=100, high=1000, confidence=0.98),
    ),
)

read_desires = CapacityDesires(
    service_tier=0,
    query_pattern=QueryPattern(
        access_consistency=GlobalConsistency(
            same_region=Consistency(target_consistency=AccessConsistency.eventual)
        ),
        estimated_read_per_second=Interval(
            low=100, mid=1000, high=10000, confidence=0.98
        ),
        estimated_write_per_second=certain_int(0),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(low=10, mid=100, high=1000, confidence=0.98),
    ),
)

write_desires = CapacityDesires(
    service_tier=0,
    query_pattern=QueryPattern(
        estimated_write_per_second=Interval(
            low=100, mid=1000, high=10000, confidence=0.98
        )
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(low=10, mid=100, high=1000, confidence=0.98),
    ),
)


def test_plan_storage():
    result = planner.plan(
        model_name="org.netflix.ddb",
        region="us-east-1",
        desires=storage_only_desires,
        extra_model_arguments={"number_of_regions": 1},
    )
    mean_plan = result.mean
    assert len(mean_plan) > 0
    assert mean_plan[0].requirements.regional[0].disk_gib == certain_int(100)
    regional_req = {
        "read_capacity_units": 0,
        "write_capacity_units": 0,
        "data_transfer_gib": 0.0,
        "replicated_write_capacity_units": 0,
    }
    assert mean_plan[0].requirements.regional[0].context == regional_req
    annual_costs = {
        "dynamo.regional-writes": Decimal("0.0"),
        "dynamo.regional-reads": Decimal("0.0"),
        "dynamo.regional-storage": Decimal("300.0"),
        "dynamo.regional-transfer": Decimal("0.0"),
        "dynamo.data-backup": Decimal("240.0"),
    }
    assert mean_plan[0].candidate_clusters.annual_costs == annual_costs
    assert round(mean_plan[0].candidate_clusters.total_annual_cost, 0) == round(540, 0)

    # global table
    result = planner.plan(
        model_name="org.netflix.ddb",
        region="us-east-1",
        desires=storage_only_desires,
        extra_model_arguments={"number_of_regions": 3},
    )
    mean_plan = result.mean
    assert len(mean_plan) > 0
    assert mean_plan[0].requirements.regional[0].disk_gib == certain_int(100)
    regional_req = {
        "read_capacity_units": 0,
        "write_capacity_units": 0,
        "data_transfer_gib": 0.0,
        "replicated_write_capacity_units": 0,
    }
    assert mean_plan[0].requirements.regional[0].context == regional_req
    annual_costs = {
        "dynamo.regional-writes": Decimal("0.0"),
        "dynamo.regional-reads": Decimal("0.0"),
        "dynamo.regional-storage": Decimal("300.0"),
        "dynamo.regional-transfer": Decimal("0.0"),
        # backup costs are normalized per region as backup is done in one region
        "dynamo.data-backup": Decimal("80.0"),
    }
    assert mean_plan[0].candidate_clusters.annual_costs == annual_costs
    assert round(mean_plan[0].candidate_clusters.total_annual_cost, 2) == round(380, 0)


def test_plan_reads():
    result = planner.plan(
        model_name="org.netflix.ddb",
        region="us-east-1",
        desires=read_desires,
        extra_model_arguments={
            "number_of_regions": 1,
            "eventual_read_percent": 0.30,
            "transactional_read_percent": 0.10,
        },
    )
    mean_plan = result.mean
    assert len(mean_plan) > 0
    assert mean_plan[0].requirements.regional[0].disk_gib == certain_int(100)
    regional_req = {
        "read_capacity_units": 8322000,
        "write_capacity_units": 0,
        "data_transfer_gib": 0.0,
        "replicated_write_capacity_units": 0,
    }
    assert mean_plan[0].requirements.regional[0].context == regional_req
    annual_costs = {
        "dynamo.regional-writes": Decimal("0.0"),
        "dynamo.regional-reads": Decimal("1081.86"),
        "dynamo.regional-storage": Decimal("300.0"),
        "dynamo.regional-transfer": Decimal("0.0"),
        "dynamo.data-backup": Decimal("240.0"),
    }
    assert mean_plan[0].candidate_clusters.annual_costs == annual_costs
    assert round(mean_plan[0].candidate_clusters.total_annual_cost, 2) == round(
        Decimal(1621.86), 2
    )

    # global table
    result = planner.plan(
        model_name="org.netflix.ddb",
        region="us-east-1",
        desires=read_desires,
        extra_model_arguments={
            "number_of_regions": 3,
            "eventual_read_percent": 0.30,
            "transactional_read_percent": 0.10,
        },
    )
    mean_plan = result.mean
    assert len(mean_plan) > 0
    assert mean_plan[0].requirements.regional[0].disk_gib == certain_int(100)
    regional_req = {
        "read_capacity_units": 8322000,
        "write_capacity_units": 0,
        "data_transfer_gib": 0.0,
        "replicated_write_capacity_units": 0,
    }
    assert mean_plan[0].requirements.regional[0].context == regional_req
    annual_costs = {
        "dynamo.regional-writes": Decimal("0.0"),
        "dynamo.regional-reads": Decimal("1081.86"),
        "dynamo.regional-storage": Decimal("300.0"),
        "dynamo.regional-transfer": Decimal("0.0"),
        # backup costs are normalized per region as backup is done in one region
        "dynamo.data-backup": Decimal("80.0"),
    }
    assert mean_plan[0].candidate_clusters.annual_costs == annual_costs
    assert round(mean_plan[0].candidate_clusters.total_annual_cost, 2) == round(
        Decimal(1461.86), 2
    )


def test_plan_reads_large_item():
    result = planner.plan(
        model_name="org.netflix.ddb",
        region="us-east-1",
        desires=read_desires,
        extra_model_arguments={
            "number_of_regions": 3,
            "eventual_read_percent": 0.30,
            "transactional_read_percent": 0.10,
            "estimated_mean_item_size_bytes": 5798,
        },
    )
    mean_plan = result.mean
    assert len(mean_plan) > 0
    assert mean_plan[0].requirements.regional[0].disk_gib == certain_int(100)
    regional_req = {
        "read_capacity_units": 16644000,
        "write_capacity_units": 0,
        "data_transfer_gib": 0.0,
        "replicated_write_capacity_units": 0,
    }
    assert mean_plan[0].requirements.regional[0].context == regional_req
    annual_costs = {
        "dynamo.regional-writes": Decimal("0.0"),
        "dynamo.regional-reads": Decimal("2163.72"),
        "dynamo.regional-storage": Decimal("300.0"),
        "dynamo.regional-transfer": Decimal("0.0"),
        "dynamo.data-backup": Decimal("80.0"),
    }
    assert mean_plan[0].candidate_clusters.annual_costs == annual_costs
    assert round(mean_plan[0].candidate_clusters.total_annual_cost, 2) == round(
        Decimal(2543.72), 2
    )


def test_plan_writes():
    result = planner.plan(
        model_name="org.netflix.ddb",
        region="us-east-1",
        desires=write_desires,
        extra_model_arguments={
            "number_of_regions": 1,
            "transactional_write_percent": 0.1,
        },
    )
    mean_plan = result.mean
    assert len(mean_plan) > 0
    assert mean_plan[0].requirements.regional[0].disk_gib == certain_int(100)
    regional_req = {
        "read_capacity_units": 0,
        "write_capacity_units": 9636000,
        "data_transfer_gib": 0.0,
        "replicated_write_capacity_units": 0,
    }
    assert mean_plan[0].requirements.regional[0].context == regional_req
    annual_costs = {
        "dynamo.regional-writes": Decimal("6263.4"),
        "dynamo.regional-reads": Decimal("0.0"),
        "dynamo.regional-storage": Decimal("300.0"),
        "dynamo.regional-transfer": Decimal("0.0"),
        "dynamo.data-backup": Decimal("240.0"),
    }
    assert mean_plan[0].candidate_clusters.annual_costs == annual_costs
    assert round(mean_plan[0].candidate_clusters.total_annual_cost, 2) == round(
        Decimal(6803.40), 2
    )

    # global table
    result = planner.plan(
        model_name="org.netflix.ddb",
        region="us-east-1",
        desires=write_desires,
        extra_model_arguments={
            "number_of_regions": 3,
            "transactional_write_percent": 0.1,
        },
    )
    mean_plan = result.mean
    assert len(mean_plan) > 0
    assert mean_plan[0].requirements.regional[0].disk_gib == certain_int(100)
    regional_req = {
        "read_capacity_units": 0,
        "write_capacity_units": 0,
        "data_transfer_gib": 15037.54,
        "replicated_write_capacity_units": 9636000,
    }
    assert mean_plan[0].requirements.regional[0].context == regional_req
    annual_costs = {
        "dynamo.regional-writes": Decimal("28185.3"),
        "dynamo.regional-reads": Decimal("0.0"),
        "dynamo.regional-storage": Decimal("300.0"),
        "dynamo.regional-transfer": Decimal("1353.38"),
        # backup costs are normalized per region as backup is done in one region
        "dynamo.data-backup": Decimal("80.0"),
    }
    assert mean_plan[0].candidate_clusters.annual_costs == annual_costs
    assert round(mean_plan[0].candidate_clusters.total_annual_cost, 2) == round(
        Decimal(29918.68), 2
    )


def test_plan_writes_large_item():
    result = planner.plan(
        model_name="org.netflix.ddb",
        region="us-east-1",
        desires=write_desires,
        extra_model_arguments={
            "number_of_regions": 3,
            "transactional_write_percent": 0.1,
            "estimated_mean_item_size_bytes": 5798,
        },
    )
    mean_plan = result.mean
    assert len(mean_plan) > 0
    assert mean_plan[0].requirements.regional[0].disk_gib == certain_int(100)
    regional_req = {
        "read_capacity_units": 0,
        "write_capacity_units": 0,
        "data_transfer_gib": 340576.70,
        "replicated_write_capacity_units": 57816000,
    }
    assert mean_plan[0].requirements.regional[0].context == regional_req
    annual_costs = {
        "dynamo.regional-writes": Decimal("169111.8"),
        "dynamo.regional-reads": Decimal("0.0"),
        "dynamo.regional-storage": Decimal("300.0"),
        "dynamo.regional-transfer": Decimal("30177.82"),
        "dynamo.data-backup": Decimal("80.0"),
    }
    assert mean_plan[0].candidate_clusters.annual_costs == annual_costs
    assert round(mean_plan[0].candidate_clusters.total_annual_cost, 2) == round(
        Decimal(199669.62), 2
    )

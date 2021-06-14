import json

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires, Interval
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int

tier_0 = CapacityDesires(
    service_tier=0,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(200),
        estimated_write_per_second=certain_int(100),
        estimated_mean_read_latency_ms=certain_float(20),
        estimated_mean_write_latency_ms=certain_float(20),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(200),
    ),
)

small_footprint = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(200),
        estimated_write_per_second=certain_int(100),
        estimated_mean_read_latency_ms=certain_float(10),
        estimated_mean_write_latency_ms=certain_float(10),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(50),
    ),
)

mid_footprint = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(300),
        estimated_write_per_second=certain_int(150),
        estimated_mean_read_latency_ms=certain_float(20),
        estimated_mean_write_latency_ms=certain_float(20),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(400),
    ),
)

large_footprint = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(1000),
        estimated_write_per_second=certain_int(800),
        estimated_mean_read_latency_ms=certain_float(20),
        estimated_mean_write_latency_ms=certain_float(20),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(800),
        estimated_working_set_percent=Interval(
            low=0.05,
            mid=0.30,
            high=0.50,
            confidence=0.8
        )
    ),
)

tier_3 = CapacityDesires(
    service_tier=3,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(200),
        estimated_write_per_second=certain_int(100),
        estimated_mean_read_latency_ms=certain_float(20),
        estimated_mean_write_latency_ms=certain_float(20),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(200),
    ),
)


def test_tier_0_not_supported():
    cap_plan = planner.plan_certain(
        model_name="org.netflix.rds",
        region="us-east-1",
        desires=tier_0,
    )
    # RDS can't support tier 0 service
    assert len(cap_plan) == 0


def test_small_footprint():
    cap_plan = planner.plan_certain(
        model_name="org.netflix.rds",
        region="us-east-1",
        desires=small_footprint,
    )
    assert cap_plan[0].candidate_clusters.regional[0].instance.name == "r5.xlarge"


def test_medium_footprint():
    cap_plan = planner.plan_certain(
        model_name="org.netflix.rds",
        region="us-east-1",
        desires=mid_footprint
    )
    assert cap_plan[0].candidate_clusters.regional[0].instance.name == "r5.8xlarge"


def test_large_footprint():
    cap_plan = planner.plan_certain(
        model_name="org.netflix.rds",
        region="us-east-1",
        desires=large_footprint,
    )
    assert cap_plan[0].candidate_clusters.regional[0].instance.name == "r5.8xlarge"


def test_tier_3():
    cap_plan = planner.plan_certain(
        model_name="org.netflix.rds",
        region="us-east-1",
        desires=tier_3,
    )
    assert cap_plan[0].candidate_clusters.regional[0].instance.name == "r5.4xlarge"
    print(cap_plan[1].candidate_clusters.regional[0].instance.name)


def test_cap_plan():
    desire_json = """{
      "deploy_desires": {
        "capacity": {
          "data_shape": {
            "estimated_state_size_gib": {
              "confidence": 0.98,
              "high": 1000,
              "low": 10,
              "mid": 100
            }
          },
          "query_pattern": {
            "access_pattern": "latency",
            "estimated_read_per_second": {
              "confidence": 0.98,
              "high": 10000,
              "low": 100,
              "mid": 1000
            },
            "estimated_write_per_second": {
              "confidence": 0.98,
              "high": 1000,
              "low": 10,
              "mid": 100
            }
          }
        },
        "config": {
          "cdc.enable": true,
          "context": "Test",
          "context-memo": "test",
          "nflx-sensitivedata": true,
          "rds.action": "create-new",
          "rds.engine": "mysql"
        },
        "consumers": [
          {
            "group": "read-write",
            "type": "email",
            "value": "saroskar@netflix.com"
          }
        ],
        "locations": [
          {
            "account": "persistence_test",
            "regions": [
              "us-east-1"
            ]
          }
        ],
        "owners": [
          {
            "group": "owner",
            "type": "google-group",
            "value": "dabp@netflix.com"
          },
          {
            "group": "owner",
            "type": "pager",
            "value": "PR06TV6"
          },
          {
            "group": "owner",
            "type": "slack",
            "value": "data-gateway-help"
          },
          {
            "group": "owner",
            "type": "email",
            "value": "saroskar@netflix.com"
          }
        ],
        "service_tier": 1,
        "version_set": {
          "base": {
            "kind": "branch",
            "value": "release/rds"
          },
          "config": {
            "kind": "branch",
            "value": "release/rds"
          }
        }
      }
    }"""
    desire = json.loads(desire_json)
    capacity = desire['deploy_desires']['capacity']
    print(capacity)
    my_desire = CapacityDesires(service_tier=desire["deploy_desires"]["service_tier"],
                                query_pattern=capacity["query_pattern"],
                                data_shape=capacity["data_shape"])
    cap_plan = planner.plan_certain(
        model_name="org.netflix.rds",
        region="us-east-1",
        desires=my_desire,
    )
    assert cap_plan[0].candidate_clusters.regional[0].instance.name == "m5.8xlarge"

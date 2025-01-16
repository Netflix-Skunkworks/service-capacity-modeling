import json
from decimal import Decimal

from service_capacity_modeling.interface import Clusters


def test_total_annual_cost():
    """make sure total_annual_cost is calculated and dumped correctly"""
    cluster = Clusters(
        annual_costs={"right-zonal": Decimal(1234), "right-regional": Decimal(234)}
    )
    expected_total = float(cluster.total_annual_cost)

    assert expected_total == cluster.model_dump().get("total_annual_cost")
    assert expected_total == cluster.model_dump().get("total_annual_cost")
    assert expected_total == json.loads(cluster.model_dump_json()).get(
        "total_annual_cost"
    )
    assert expected_total == json.loads(cluster.model_dump_json()).get(
        "total_annual_cost"
    )

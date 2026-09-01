import json
from unittest.mock import Mock

from service_capacity_modeling.tools.fetch_pricing import fetch_elasticache_pricing


def _price_item(
    instance_type: str,
    *,
    engine: str = "Valkey",
    usage_type: str = "NodeUsage:cache.r7g.large",
    hourly_price: str = "0.1752",
    upfront_price: str = "2071.9152",
) -> str:
    return json.dumps(
        {
            "product": {
                "attributes": {
                    "instanceType": instance_type,
                    "cacheEngine": engine,
                    "usagetype": usage_type,
                }
            },
            "terms": {
                "OnDemand": {
                    "term": {
                        "priceDimensions": {
                            "hourly": {
                                "unit": "Hrs",
                                "pricePerUnit": {"USD": hourly_price},
                            }
                        },
                    }
                },
                "Reserved": {
                    "term": {
                        "termAttributes": {
                            "LeaseContractLength": "3yr",
                            "PurchaseOption": "All Upfront",
                            "OfferingClass": "standard",
                        },
                        "priceDimensions": {
                            "upfront": {
                                "unit": "Quantity",
                                "pricePerUnit": {"USD": upfront_price},
                            }
                        },
                    }
                },
            },
        }
    )


def test_fetch_elasticache_pricing_selects_on_demand_valkey_nodes():
    paginator = Mock()
    paginator.paginate.return_value = [
        {
            "PriceList": [
                _price_item("cache.r7g.large"),
                _price_item(
                    "cache.r7g.large",
                    usage_type="USE1-SyncDurability-NodeUsage:cache.r7g.large",
                ),
                _price_item("cache.r6g.large"),
                _price_item("cache.r7g.xlarge", engine="Redis"),
            ]
        }
    ]
    pricing_client = Mock()
    pricing_client.get_paginator.return_value = paginator

    assert fetch_elasticache_pricing(pricing_client) == {
        "cache.r7g.large": {"annual_cost": 1534.75}
    }
    paginator.paginate.assert_called_once_with(
        ServiceCode="AmazonElastiCache",
        Filters=[
            {
                "Type": "TERM_MATCH",
                "Field": "location",
                "Value": "US East (N. Virginia)",
            },
            {"Type": "TERM_MATCH", "Field": "cacheEngine", "Value": "Valkey"},
        ],
    )

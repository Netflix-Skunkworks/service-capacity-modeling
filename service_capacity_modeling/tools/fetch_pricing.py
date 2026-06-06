import argparse
import json
import os
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import boto3


def extract_3yr_upfront_price(price_data: Dict[str, Any]) -> Optional[float]:
    instance_type = price_data["product"]["attributes"]["instanceType"]

    # Look through Reserved terms
    if price_data.get("terms") is None:
        return None

    reserved_terms = price_data["terms"].get("Reserved", {})

    for term in reserved_terms.values():
        term_attrs = term["termAttributes"]
        # Find 3yr All Upfront terms
        if (
            term_attrs.get("LeaseContractLength") == "3yr"
            and term_attrs.get("PurchaseOption") == "All Upfront"
            and term_attrs.get("OfferingClass") == "standard"
        ):
            # Get upfront fee
            for dim in term["priceDimensions"].values():
                if dim["unit"] == "Quantity":
                    upfront = float(dim["pricePerUnit"]["USD"])
                    annual_cost = upfront / 3
                    annual_cost_rounded = round(annual_cost, 2)
                    print(f"{instance_type}: {annual_cost_rounded}")
                    return annual_cost_rounded

    return None


def fetch_ec2_pricing(pricing_client: Any) -> Dict[str, Dict[str, Union[float, str]]]:
    paginator = pricing_client.get_paginator("get_products")

    instances = {}

    filter_params = {
        "ServiceCode": "AmazonEC2",
        "Filters": [
            {
                "Type": "TERM_MATCH",
                "Field": "location",
                "Value": "US East (N. Virginia)",
            },
            {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
            {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
            {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
            {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
        ],
    }

    for page in paginator.paginate(**filter_params):
        for price_item in page["PriceList"]:
            price_data = json.loads(price_item)

            # Extract instance type
            attributes = price_data.get("product", {}).get("attributes", {})
            instance_type = attributes.get("instanceType")

            if not instance_type:
                continue

            annual_cost = extract_3yr_upfront_price(price_data)
            if annual_cost:
                instance_info: Dict[str, Union[float, str]] = {
                    "annual_cost": annual_cost
                }
                if "deprecated" in str(instance_type).lower():
                    instance_info["lifecycle"] = "deprecated"
                instances[instance_type] = instance_info
    return instances


# A 3yr Savings Plan offering, in seconds (3 * 365 days).
THREE_YEAR_SECONDS = 94608000
# Hours in a year, used to annualize the effective hourly Savings Plan rate.
HOURS_PER_YEAR = 8760


def _savings_plan_instance_type(result: Dict[str, Any]) -> Optional[str]:
    """Pull the instance type out of an offering rate result.

    Prefer an explicit ``instanceType`` property if present, otherwise parse it
    out of the ``usageType`` (e.g. "USE1-BoxUsage:i7ie.xlarge" -> "i7ie.xlarge").
    """
    for prop in result.get("properties", []):
        if prop.get("name") == "instanceType" and prop.get("value"):
            return str(prop["value"])

    usage_type = str(result.get("usageType", ""))
    if ":" in usage_type:
        return usage_type.split(":", 1)[1]
    return None


def fetch_ec2_savings_plan_pricing(sp_client: Any, region: str) -> Dict[str, float]:
    """Fetch 3yr All Upfront EC2 Instance Savings Plan rates for a region.

    Returns {instance_type: annual_cost}. The EC2 Instance Savings Plan is
    family+region locked with the deepest discount, making it the closest analog
    to the now-deprecated 3yr All Upfront standard Reserved Instance.
    """
    instances: Dict[str, float] = {}
    next_token: Optional[str] = None
    while True:
        kwargs: Dict[str, Any] = {
            "savingsPlanTypes": ["EC2Instance"],
            "savingsPlanPaymentOptions": ["All Upfront"],
            "serviceCodes": ["AmazonEC2"],
            "filters": [
                {"name": "region", "values": [region]},
                {"name": "tenancy", "values": ["shared"]},
                {"name": "productDescription", "values": ["Linux/UNIX"]},
            ],
            "maxResults": 100,
        }
        if next_token:
            kwargs["nextToken"] = next_token

        response = sp_client.describe_savings_plans_offering_rates(**kwargs)
        for result in response.get("searchResults", []):
            offering = result.get("savingsPlanOffering", {})
            if offering.get("durationSeconds") != THREE_YEAR_SECONDS:
                continue

            instance_type = _savings_plan_instance_type(result)
            if instance_type is None:
                continue
            # De-dupe defensively: keep the first 3yr match per instance type.
            if instance_type in instances:
                continue

            annual_cost = round(float(result["rate"]) * HOURS_PER_YEAR, 2)
            instances[instance_type] = annual_cost
            print(f"{instance_type} (SP): {annual_cost}")

        next_token = response.get("nextToken")
        if not next_token:
            break
    return instances


def fetch_rds_pricing(pricing_client: Any) -> Dict[str, Dict[str, Union[float, str]]]:
    paginator = pricing_client.get_paginator("get_products")

    instances = {}

    filter_params = {
        "ServiceCode": "AmazonRDS",
        "Filters": [
            {
                "Type": "TERM_MATCH",
                "Field": "location",
                "Value": "US East (N. Virginia)",
            },
            {
                "Type": "TERM_MATCH",
                "Field": "databaseEngine",
                "Value": "Aurora PostgreSQL",
            },
            {"Type": "TERM_MATCH", "Field": "deploymentOption", "Value": "Single-AZ"},
        ],
    }

    for page in paginator.paginate(**filter_params):
        for price_item in page["PriceList"]:
            price_data = json.loads(price_item)

            # Extract instance type
            attributes = price_data.get("product", {}).get("attributes", {})
            instance_type = attributes.get("instanceType")

            if not instance_type:
                continue

            annual_cost = extract_3yr_upfront_price(price_data)
            if annual_cost:
                instance_info: Dict[str, Union[float, str]] = {
                    "annual_cost": annual_cost
                }
                if "deprecated" in str(instance_type).lower():
                    instance_info["lifecycle"] = "deprecated"
                instances[instance_type] = instance_info
    return instances


def fetch_pricing(region: str) -> None:
    pricing_client = boto3.client("pricing", region_name=region)
    ec2_instances = fetch_ec2_pricing(pricing_client)
    rds_instances = fetch_rds_pricing(pricing_client)

    # AWS no longer sells standard Reserved Instances for newer families
    # (e.g. i7ie), so their instance types have no RI price above. Fall back to
    # the 3yr All Upfront EC2 Instance Savings Plan rate for any instance type
    # the RI path missed.
    sp_client = boto3.client("savingsplans", region_name=region)
    sp_instances = fetch_ec2_savings_plan_pricing(sp_client, region)
    filled = 0
    for instance_type, annual_cost in sp_instances.items():
        if instance_type not in ec2_instances:
            ec2_instances[instance_type] = {"annual_cost": annual_cost}
            filled += 1
    print(f"\nFilled {filled} EC2 instance types from Savings Plan rates")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ec2_output_file = os.path.join(
        project_root,
        "hardware",
        "profiles",
        "pricing",
        "aws",
        "3yr-reserved_ec2.json",
    )
    # we bolt on the other info, as a hack until we can improve prior layers
    ec2_output = {
        "us-east-1": {
            "instances": ec2_instances,
            "drives": {},
            "services": {},
            "zones_in_region": 3,
        }
    }
    with open(ec2_output_file, "w", encoding="utf-8") as f:
        json.dump(ec2_output, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nEC2 pricing data written to {ec2_output_file}")
    rds_output_file = os.path.join(
        project_root,
        "hardware",
        "profiles",
        "pricing",
        "aws",
        "3yr-reserved_rds.json",
    )
    rds_output = {
        "us-east-1": {
            "instances": rds_instances,
            "drives": {},
            "services": {},
            "zones_in_region": 3,
        }
    }
    with open(rds_output_file, "w", encoding="utf-8") as f:
        json.dump(rds_output, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"RDS pricing data written to {rds_output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch EC2 and RDS Reserved Instance pricing data."
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    args = parser.parse_args()

    fetch_pricing(args.region)


if __name__ == "__main__":
    main()

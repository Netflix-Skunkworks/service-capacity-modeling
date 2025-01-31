import argparse
import json
import os

import boto3


def extract_3yr_upfront_price(price_data):
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


def fetch_pricing(region: str):
    # Initialize pricing client
    pricing_client = boto3.client("pricing", region_name=region)

    # Get all EC2 pricing for Reserved Instances
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
                instances[instance_type] = {"annual_cost": annual_cost}
                if "deprecated" in instance_type.lower():
                    instances[instance_type]["lifecycle"] = "deprecated"

    # Create final output structure
    # we bolt on the other info, as a hack until we can improve prior layers
    output = {
        "us-east-1": {
            "instances": instances,
            "drives": {},
            "services": {},
            "zones_in_region": 3,
        }
    }

    # Write to JSON file
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_file = os.path.join(
        project_root,
        "service_capacity_modeling",
        "hardware",
        "profiles",
        "pricing",
        "aws",
        "3yr-reserved_ec2.json",
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Pricing data written to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch EC2 Reserved Instance pricing data."
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

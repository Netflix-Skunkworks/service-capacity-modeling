import argparse
import boto3
import json
import sys
import os
from collections import OrderedDict
from fractions import Fraction
from typing import Optional
from typing import Sequence

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import Instance


def pull_family(
    instance_filter: str, xlarge_read_iops: Optional[float]=None, xlarge_write_iops: Optional[float]=None, region: str = "us-east-1"
) -> Sequence[Instance]:
    # pylint: disable=unused-argument
    """
    This will emulate

    aws --region us-east-1 \
      ec2 describe-instance-types \
      --filters 'Name=instance-type,Values=m7a.*'

    And then convert from base 10 for memory and disk to base 2 and generate the
    shape from that.

    Returns a list of Instance for hardware. Then we can compose to writing
    json files.
    """
    # Initialize ec2 client
    ec2_client = boto3.client("ec2", region_name=region)

    paginator = ec2_client.get_paginator("describe_instance_types")
    filter_params = {
        "Filters": [
            {
                "Name": "instance-type",
                "Values": [instance_filter]
            },
        ],
    }

    hasSSD = False
    instance_jsons_dict = {}
    disk_single_tenant_max_size = -1
    for page in paginator.paginate(**filter_params):
        for instance_type_json in page["InstanceTypes"]:
            print(json.dumps(instance_type_json))
            if "metal" in instance_type_json["InstanceType"]:
                print("Exclude metal for now...")
                continue
            if instance_type_json["InstanceStorageSupported"] is True:
                if instance_type_json["InstanceStorageInfo"]["Disks"][0]["Type"] == "ssd":
                    hasSSD = True
                    if instance_type_json["InstanceStorageInfo"]["Disks"][0]["Count"] == 1:
                        # The biggest size for count equal 1 help us identify whether it's single tenant drive or not
                        disk_single_tenant_max_size = max(disk_single_tenant_max_size, instance_type_json["InstanceStorageInfo"]["TotalSizeInGB"])
                else:
                    raise Exception("Instance drive is not of type SSD. Please consult for the right read_io_latency.")

            instance_jsons_dict[instance_type_json["InstanceType"].split('.')[1]] = instance_type_json

    if "xlarge" not in instance_jsons_dict:
        raise Exception("No xlarge result returned from AWS api. Please check your input.")

    if hasSSD and (xlarge_read_iops is None or xlarge_write_iops is None):
        raise Exception("Instance shape has SSD. User should input xlarge_read_iops and xlarge_write_iops.")

    results = []
    # First build up xlarge instance shape, then we will use it as a reference.
    xlarge_shape = Instance(
        name=instance_jsons_dict["xlarge"]["InstanceType"],
        cpu=instance_jsons_dict["xlarge"]["VCpuInfo"]["DefaultVCpus"],
        cpu_ghz=instance_jsons_dict["xlarge"]["ProcessorInfo"]["SustainedClockSpeedInGhz"],
        ram_gib=round(instance_jsons_dict["xlarge"]["MemoryInfo"]["SizeInMiB"] * 10 ** 6 / 2 ** 30, 2),
        net_mbps=round(instance_jsons_dict["xlarge"]["NetworkInfo"]["NetworkCards"][0]["BaselineBandwidthInGbps"] * 10 ** 3),
        drive=None if not hasSSD else Drive(
            name="ephem",
            size_gib=int(instance_jsons_dict["xlarge"]["InstanceStorageInfo"]["TotalSizeInGB"] * 10 ** 9 / 2 ** 30),
            read_io_latency_ms=FixedInterval(
                low=0.1, mid=0.125, high=0.17, confidence=0.9, minimum_value=0.05, maximum_value=2.05
            ),
            read_io_per_s=xlarge_read_iops,
            write_io_per_s=xlarge_write_iops,
            block_size_kib=4,
            single_tenant=True if instance_jsons_dict["xlarge"]["InstanceStorageInfo"]["Disks"][0]["Count"] == 1 and instance_jsons_dict["xlarge"]["InstanceStorageInfo"]["TotalSizeInGB"] == disk_single_tenant_max_size else False,
        )
    )
    results.append(xlarge_shape)

    # Build up the rest
    for instance_type_scale, instance_type_json in instance_jsons_dict.items():
        if instance_type_scale == "xlarge":
            continue

        new_shape = Instance(
            name=instance_type_json["InstanceType"],
            cpu=instance_type_json["VCpuInfo"]["DefaultVCpus"],
            cpu_ghz=instance_type_json["ProcessorInfo"]["SustainedClockSpeedInGhz"],
            ram_gib=round(instance_type_json["MemoryInfo"]["SizeInMiB"] * 10 ** 6 / 2 ** 30, 2),
            net_mbps=round(instance_type_json["NetworkInfo"]["NetworkCards"][0]["BaselineBandwidthInGbps"] * 10 ** 3),
            drive=None
        )

        if hasSSD:
            scale_factor = new_shape.normalized_size / xlarge_shape.normalized_size
            new_shape.drive = Drive(
                name="ephem",
                size_gib=int(instance_jsons_dict["xlarge"]["InstanceStorageInfo"]["TotalSizeInGB"] * 10 ** 9 / 2 ** 30),
                read_io_latency_ms=FixedInterval(
                    low=0.1, mid=0.125, high=0.17, confidence=0.9, minimum_value=0.05, maximum_value=2.05
                ),
                read_io_per_s=int(round(xlarge_read_iops * scale_factor)),
                write_io_per_s=int(round(xlarge_write_iops * scale_factor)),
                block_size_kib=4,
                single_tenant=True if instance_type_json["InstanceStorageInfo"]["Disks"][0]["Count"] == 1 and instance_type_json["InstanceStorageInfo"]["TotalSizeInGB"] == disk_single_tenant_max_size else False,
            )

        results.append(new_shape)
    results = sorted(results, key=lambda i: i.normalized_size)
    # print(results)
    return results


def main(args) -> int:
    instance_gen = args.template.split(".")[0]
    result = pull_family(instance_gen + ".*", args.riops, args.wiops, args.region)
    json_shapes = {}
    for shape in result:
        model_dict = shape.model_dump(exclude_unset=True)
        # Hardware shouldn't have costs
        if "annual_cost" in model_dict:
            del model_dict["annual_cost"]
        if "drive" in model_dict and model_dict["drive"] is not None and "annual_cost" in model_dict["drive"]:
            del model_dict["drive"]["annual_cost"]

        json_shapes[shape.name] = model_dict

    print(f"[{instance_gen}[aws.json] Hardware Shapes", file=sys.stderr)
    print(json.dumps(json_shapes, indent=2))

    # Write to JSON file
    if args.dry is False:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_file = os.path.join(
            project_root,
            "hardware",
            "profiles",
            "shapes",
            f"aws_{instance_gen}.json",
        )
        print(f"output json file to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_shapes, f, indent=2)
            f.write("\n")

    return 0


if __name__ == "__main__":
    instances = set()
    families = set()
    regions = set()
    for ar in shapes.hardware.regions.keys():
        regions.add(ar)

    parser = argparse.ArgumentParser(
        prog="Project shapes from instance family filter",
        description="Input the target instance family filter like m7a.* and projects all the instance shapes in a json file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-r", "--region", choices=regions, default="us-east-1")
    parser.add_argument("-template", type=str, default="m7a")
    parser.add_argument("-riops", type=float, default=None)
    parser.add_argument("-wiops", type=float, default=None)
    parser.add_argument("-dry", type=bool, default=True)
    sys.exit(main(parser.parse_args()))

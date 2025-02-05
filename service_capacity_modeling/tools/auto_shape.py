import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

import boto3
from pydantic import BaseModel

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import DriveType
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import normalized_aws_size


class IOPerformance(BaseModel):
    xl_read_iops: float
    xl_write_iops: float
    latency: FixedInterval
    single_tenant_size: int = 0


latency_curve_ms: Dict[str, FixedInterval] = {
    "6th-gen-ssd": FixedInterval(
        low=0.1,
        mid=0.125,
        high=0.170,
        confidence=0.9,
        minimum_value=0.050,
        maximum_value=2.000,
    ),
    "5th-gen-ssd": FixedInterval(
        low=0.080,
        mid=0.125,
        high=0.200,
        confidence=0.9,
        minimum_value=0.070,
        maximum_value=2.000,
    ),
}

aws_io_data = {
    "m": "https://docs.aws.amazon.com/ec2/latest/instancetypes/gp.html#gp_instance-store",
    "c": "https://docs.aws.amazon.com/ec2/latest/instancetypes/co.html#co_instance-store",
    "r": "https://docs.aws.amazon.com/ec2/latest/instancetypes/mo.html#co_instance-store",
    "i": "https://docs.aws.amazon.com/ec2/latest/instancetypes/so.html#so_instance-store",
}
aws_instance_link = "https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-type-specifications.html"


def _gb_to_gib(inp: float) -> int:
    return int(round((inp * 10**9) / 2.0**30))


def _drive(
    drive_type: DriveType, io_perf: Optional[IOPerformance], scale: Fraction, data: Dict
) -> Optional[Drive]:
    if drive_type.name.startswith("attached") or io_perf is None:
        return None

    size = data["InstanceStorageInfo"]["TotalSizeInGB"]
    size_gib = _gb_to_gib(size)
    single_drive_size = data["InstanceStorageInfo"]["Disks"][0]["SizeInGB"]
    single_tenant = False
    if single_drive_size == io_perf.single_tenant_size:
        single_tenant = True

    return Drive(
        name="ephem",
        size_gib=int(size_gib),
        write_io_per_s=int(round(io_perf.xl_write_iops * scale)),
        read_io_per_s=int(round(io_perf.xl_read_iops * scale)),
        read_io_latency_ms=io_perf.latency,
        single_tenant=single_tenant,
    )


def pull_family(
    instance_filter: str,
    region: str = "us-east-1",
    io_perf: Optional[IOPerformance] = None,
    debug: bool = False,
) -> Sequence[Instance]:
    # pylint: disable=unused-argument
    # flake8: noqa: C901
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

    request = ec2_client.get_paginator("describe_instance_types")
    payload = {
        "Filters": [
            {"Name": "instance-type", "Values": [instance_filter]},
        ],
    }

    def debug_log(msg: str):
        if debug:
            print(msg, file=sys.stderr)

    instance_jsons_dict = {}
    disk_type = DriveType.attached_ssd
    disk_max_size = -1
    for instance_page in request.paginate(**payload):
        for instance_type_json in instance_page["InstanceTypes"]:
            debug_log(json.dumps(instance_type_json))
            if "metal" in instance_type_json["InstanceType"]:
                debug_log("Exclude metal for now...")
                continue

            if instance_type_json["InstanceStorageSupported"] is True:
                typ = instance_type_json["InstanceStorageInfo"]["Disks"][0]["Type"]
                disk_type = DriveType["local_" + typ.lower()]
                for disk in instance_type_json["InstanceStorageInfo"]["Disks"]:
                    # The biggest size drive is the single tenant one
                    disk_max_size = max(disk_max_size, int(disk["SizeInGB"]))

            instance_jsons_dict[
                instance_type_json["InstanceType"].split(".")[1]
            ] = instance_type_json

    if disk_type.name.startswith("local"):
        if (
            io_perf is None
            or io_perf.xl_read_iops is None
            or io_perf.xl_write_iops is None
        ):
            family = instance_filter.split(".")[0]
            link = aws_io_data.get(family[0], aws_instance_link)
            print(
                "Instance shape has ephemeral storage. You must pass --xl-iops with "
                "data either from fio benchmarking or from AWS's page for the "
                "appropriate family:\n"
                f"Search for {family}.xlarge in {link}\n",
                file=sys.stderr,
            )
            sys.exit(2)
        debug_log(
            f"{disk_type.name} IO data:\n"
            f"xlarge  read IOPS: {io_perf.xl_read_iops}\n"
            f"xlarge write IOPS: {io_perf.xl_write_iops}\n"
            f"latency curve: {io_perf.latency.model_dump_json()}\n"
        )
        io_perf = io_perf.model_copy(
            update={
                "single_tenant_size": disk_max_size,
            }
        )

    results = []
    # Now build the instance shapes from the data
    for _, data in instance_jsons_dict.items():
        name = data["InstanceType"]
        try:
            normalized_size = normalized_aws_size(name)
        except AssertionError:
            print(name)

        drive = _drive(disk_type, io_perf, scale=normalized_size, data=data)
        new_shape = Instance(
            name=data["InstanceType"],
            cpu=data["VCpuInfo"]["DefaultVCpus"],
            cpu_ghz=data["ProcessorInfo"]["SustainedClockSpeedInGhz"],
            ram_gib=round(data["MemoryInfo"]["SizeInMiB"] * 10**6 / 2.0**30, 2),
            net_mbps=round(
                data["NetworkInfo"]["NetworkCards"][0]["BaselineBandwidthInGbps"]
                * 10**3
            ),
            drive=drive,
        )

        results.append(new_shape)
    results = sorted(results, key=lambda i: normalized_aws_size(i.name))
    return results


def parse_iops(inp: str) -> Optional[Tuple[int, int]]:
    """Parses strings like 100,000/50,000 to (100000, 50000)"""
    if inp is None:
        return None
    # AWS often gives like 117,000 / 57,000
    if inp.count("/") == 1:
        left, right = inp.split("/")
        left = left.strip().replace(",", "")
        right = right.strip().replace(",", "")
        return int(left), int(right)
    else:
        raise argparse.ArgumentTypeError(
            "xlarge IOPS should be given in <random read>/<write> format. "
            "For example r5d.xlarge would be 59000/29000."
        )


def parse_io_curve(inp: str) -> FixedInterval:
    return latency_curve_ms[inp]


def main(args) -> int:
    iops = args.xl_iops
    if iops is None:
        io_perf = None
    else:
        io_perf = IOPerformance(
            latency=parse_io_curve(args.io_latency_curve),
            xl_read_iops=args.xl_iops[0],
            xl_write_iops=args.xl_iops[1],
        )

    for family in args.families:
        family_shapes = pull_family(
            instance_filter=family + ".*",
            region=args.region,
            io_perf=io_perf,
            debug=args.debug,
        )
        json_shapes: Dict[str, Dict[str, Instance]] = {"instances": {}}
        for shape in family_shapes:
            model_dict = shape.model_dump(exclude_unset=True)
            # Hardware shouldn't have costs
            if "annual_cost" in model_dict:
                del model_dict["annual_cost"]
            if (
                "drive" in model_dict
                and model_dict["drive"] is not None
                and "annual_cost" in model_dict["drive"]
            ):
                del model_dict["drive"]["annual_cost"]

            json_shapes["instances"][shape.name] = model_dict

        print(f"[{family}[aws.json] Hardware Shapes", file=sys.stderr)
        print(json.dumps(json_shapes, indent=2))

        # Write to JSON file if requested
        if args.output_path is not None:
            path: Path = args.output_path
            if path.is_dir():
                output_path = Path(path, f"auto_{family}.json")
            else:
                output_path = path
            with open(output_path, "wt", encoding="utf-8") as fd:
                json.dump(json_shapes, fd, indent=2)
                fd.write("\n")

    return 0


if __name__ == "__main__":
    families = set()
    regions = set()
    for ar in shapes.hardware.regions.keys():
        regions.add(ar)

    try:
        client = boto3.client("ec2", region_name="us-east-1")
    except:
        print(
            "Unable to connect to EC2. Do you have AWS credentials refreshed?",
            file=sys.stderr,
        )
        sys.exit(1)

    result = client.describe_instance_type_offerings()
    offerings = result["InstanceTypeOfferings"]
    for offering in offerings:
        families.add(offering["InstanceType"].rsplit(".", 1)[0])

    parser = argparse.ArgumentParser(
        prog="Project shapes from instance family filter",
        description="Input the target instance family filter like m7a.* and generates shape data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--xl-iops",
        default=None,
        type=parse_iops,
        help=(
            "The xlarge size's iops expressed as <rand 4k reads>/<write>. "
            "For example '100,000 / 50,000'. To find this information use AWS's "
            f"spec: {aws_instance_link} OR use fio to benchmark"
        ),
    )
    parser.add_argument(
        "--io-latency-curve", default="6th-gen-ssd", choices=latency_curve_ms.keys()
    )
    parser.add_argument("--region", choices=regions, default="us-east-1")
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Output file path to copy the result to. If not given only stdout will occur",
    )
    parser.add_argument("--debug", action="store_true", help="Show verbose output")

    parser.add_argument(
        "families",
        nargs="+",
        type=str,
        default="m7a",
        metavar="family",
        choices=sorted(list(families)),
    )
    sys.exit(main(parser.parse_args()))

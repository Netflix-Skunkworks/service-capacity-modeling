import argparse
import json
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

import boto3
import botocore
from pydantic import BaseModel

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import DriveType
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import normalized_aws_size


# Default latency curves from FIO testing per generation
latency_curve_ms: Dict[str, FixedInterval] = {
    "7th-gen-ephemeral": FixedInterval(
        low=0.071,
        mid=0.079,
        high=0.258,
        confidence=0.9,
        minimum_value=0.026,
        maximum_value=2.153,
    ),
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
    "ssd": FixedInterval(low=0.08, mid=0.10, high=0.2),
    "hdd": FixedInterval(low=1, mid=2, high=10),
}

# Default xlarge -> (4k random read, write) iops tables
aws_xlarge_iops = {
    # General Purpose and Memory Share IOPs
    "5ad": (59_000, 29_000),
    "5d": (59_000, 29_000),
    "5dn": (58_000, 29_000),
    "6gd": (53_750, 22_500),
    "6id": (67_083, 33_542),
    "6idn": (67_083, 33_542),
    "7gd": (67_083, 33_542),
    # Compute has smaller IOPs
    "c5ad": (32_566, 14_211),
    "c5d": (40_000, 18_000),
    "c6gd": (53_750, 22_500),
    "c6id": (67_083, 33_542),
    "c7gd": (67_083, 33_542),
    # Storage has more
    # https://docs.aws.amazon.com/ec2/latest/instancetypes/so.html#so_instance-store
    "i3": (206_250, 70_000),
    "i3en": (85_000, 65_000),
    "i4g": (62_500, 50_000),
    "i4i": (100_000, 55_000),
    "i7i": (150_000, 82_500),
    "i7ie": (108_333, 86_666),
    "i8g": (150_000, 82_500),
}


def guess_iops(family: str) -> Optional[Tuple[int, int]]:
    if family[0] in ("m", "r"):
        return aws_xlarge_iops.get(family[1:])
    return aws_xlarge_iops.get(family)


aws_io_links = {
    "m": "https://docs.aws.amazon.com/ec2/latest/instancetypes/gp.html#gp_instance-store",
    "c": "https://docs.aws.amazon.com/ec2/latest/instancetypes/co.html#co_instance-store",
    "r": "https://docs.aws.amazon.com/ec2/latest/instancetypes/mo.html#mo_instance-store",
    "i": "https://docs.aws.amazon.com/ec2/latest/instancetypes/so.html#so_instance-store",
}
aws_instance_link = "https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-type-specifications.html"


class CPUPerformance(BaseModel):
    ipc_scale_factor: Optional[float] = None


class IOPerformance(BaseModel):
    xl_read_iops: float
    xl_write_iops: float
    latency: FixedInterval
    single_tenant_size: int = 0


def _gb_to_gib(inp: float) -> int:
    return int(round((inp * 10**9) / 2.0**30))


def deduce_cpu_ipc_scale(
    vcpu_count: int,
    cpu_cores: int,
    cpu_perf: Optional[CPUPerformance] = None,
) -> float:
    """
    Deduce CPU IPC scale factor from vCPU and core counts.
    If all vCPUs are full cores (no SMT), use 1.5, otherwise 1.0.
    """
    if cpu_perf is not None and cpu_perf.ipc_scale_factor is not None:
        return cpu_perf.ipc_scale_factor
    if vcpu_count == cpu_cores:
        return 1.5
    return 1.0


def convert_mib_to_gib(size_mib: float) -> float:
    """Convert AWS MiB memory size to GiB."""
    return round(size_mib * 10**6 / 2.0**30, 2)


def convert_gbps_to_mbps(bandwidth_gbps: float) -> float:
    """Convert AWS Gbps network bandwidth to Mbps."""
    return round(bandwidth_gbps * 1000)


def _engine_to_platform(engine: str) -> str:
    """
    Map RDS engine name to Platform enum value.
    Currently only supports aurora-postgresql.
    """
    if engine == "aurora-postgresql":
        return "aurora_postgres"
    raise ValueError(f"Engine '{engine}' is not supported")


def _drive(
    drive_type: DriveType,
    io_perf: Optional[IOPerformance],
    scale: Fraction,
    data: Dict[str, Any],
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
    ec2_client: Any,
    family: str,
    cpu_perf: Optional[CPUPerformance] = None,
    io_perf: Optional[IOPerformance] = None,
    debug: bool = False,
) -> Sequence[Instance]:
    # flake8: noqa: C901
    """
    Pull Instance shapes from AWS APIs.

    Emulates something like the following

    aws --region us-east-1 \
      ec2 describe-instance-types \
      --filters 'Name=instance-type,Values=m7a.*'

    And then convert from base 10 for memory and disk to base 2 and generate the
    shape from that.

    Returns a list of Instance for hardware. Then we can compose to writing
    json files.
    """
    request = ec2_client.get_paginator("describe_instance_types")
    payload = {
        "Filters": [
            {
                "Name": "instance-type",
                "Values": [
                    family + ".*",
                ],
            },
        ],
    }

    def debug_log(msg: str) -> None:
        if debug:
            print(msg, file=sys.stderr)

    instance_jsons_dict = {}
    disk_type = DriveType.attached_ssd
    disk_max_size = -1
    for instance_page in request.paginate(**payload):
        for instance_type_json in instance_page["InstanceTypes"]:
            debug_log(json.dumps(instance_type_json))
            if "metal" in instance_type_json["InstanceType"]:
                debug_log("Excluding metal for now.")
                continue

            if instance_type_json["InstanceStorageSupported"] is True:
                typ = instance_type_json["InstanceStorageInfo"]["Disks"][0]["Type"]
                disk_type = DriveType["local_" + typ.lower()]
                for disk in instance_type_json["InstanceStorageInfo"]["Disks"]:
                    # The biggest size drive is the single tenant one
                    disk_max_size = max(disk_max_size, int(disk["SizeInGB"]))

            instance_jsons_dict[instance_type_json["InstanceType"].split(".")[1]] = (
                instance_type_json
            )

    if disk_type.name.startswith("local"):
        if (
            io_perf is None
            or io_perf.xl_read_iops is None
            or io_perf.xl_write_iops is None
        ):
            link = aws_io_links.get(family[0], aws_instance_link)
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
        vcpu_count = data["VCpuInfo"]["DefaultVCpus"]
        cpu_cores = data["VCpuInfo"]["DefaultCores"]
        cpu_ipc_scale_factor = deduce_cpu_ipc_scale(vcpu_count, cpu_cores, cpu_perf)

        new_shape = Instance(
            name=data["InstanceType"],
            cpu=vcpu_count,
            cpu_cores=cpu_cores,
            cpu_ghz=data["ProcessorInfo"]["SustainedClockSpeedInGhz"],
            cpu_ipc_scale=cpu_ipc_scale_factor,
            ram_gib=convert_mib_to_gib(data["MemoryInfo"]["SizeInMiB"]),
            net_mbps=convert_gbps_to_mbps(
                data["NetworkInfo"]["NetworkCards"][0]["BaselineBandwidthInGbps"]
            ),
            drive=drive,
        )

        results.append(new_shape)
    results = sorted(results, key=lambda i: normalized_aws_size(i.name))
    return results


def lookup_ec2_instance_specs(
    ec2_client: Any,
    instance_type: str,
    debug: bool = False,
) -> Tuple[int, int, float, float, float]:
    """
    Look up EC2 instance specifications.
    Returns: (vcpu_count, cpu_cores, cpu_ghz, ram_gib, net_mbps)
    Returns zeros/defaults if lookup fails.
    """
    try:
        ec2_response = ec2_client.describe_instance_types(InstanceTypes=[instance_type])
        if ec2_response["InstanceTypes"]:
            ec2_data = ec2_response["InstanceTypes"][0]
            vcpu_count = ec2_data["VCpuInfo"]["DefaultVCpus"]
            cpu_cores = ec2_data["VCpuInfo"]["DefaultCores"]
            cpu_ghz = ec2_data["ProcessorInfo"]["SustainedClockSpeedInGhz"]
            ram_gib = convert_mib_to_gib(ec2_data["MemoryInfo"]["SizeInMiB"])
            net_mbps = convert_gbps_to_mbps(
                ec2_data["NetworkInfo"]["NetworkCards"][0]["BaselineBandwidthInGbps"]
            )
            if debug:
                print(
                    f"Looked up {instance_type} -> {vcpu_count} vCPUs, "
                    f"{cpu_cores} cores, {cpu_ghz} GHz, {ram_gib} GiB, {net_mbps} Mbps",
                    file=sys.stderr,
                )
            return vcpu_count, cpu_cores, cpu_ghz, ram_gib, net_mbps
        else:
            raise RuntimeError("No EC2 instance types returned")
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        if debug:
            print(
                f"WARNING: Could not look up {instance_type}: {e}. Using defaults.",
                file=sys.stderr,
            )
        return 0, 0, 2.5, 0.0, 1000.0


def pull_rds_family(  # pylint: disable=too-many-locals
    rds_client: Any,
    family: str,
    db_engines: Sequence[str],
    cpu_perf: Optional[CPUPerformance] = None,
    debug: bool = False,
) -> Sequence[Instance]:
    """
    Pull RDS Instance shapes from AWS RDS APIs.
    Queries describe_orderable_db_instance_options for the specified engines
    and family, then constructs Instance objects with appropriate platform tags.
    Returns a list of Instance for hardware shapes.
    """

    def debug_log(msg: str) -> None:
        if debug:
            print(msg, file=sys.stderr)

    ec2_client = boto3.client("ec2", region_name=rds_client.meta.region_name)
    instance_data_map = {}
    for engine in db_engines:
        debug_log(f"Querying RDS API for engine={engine}, family=db.{family}.*")
        paginator = rds_client.get_paginator("describe_orderable_db_instance_options")
        page_iterator = paginator.paginate(Engine=engine)
        for page in page_iterator:
            for option in page["OrderableDBInstanceOptions"]:
                db_instance_class = option["DBInstanceClass"]
                if not db_instance_class.startswith(f"db.{family}."):
                    continue
                if db_instance_class not in instance_data_map:
                    instance_data_map[db_instance_class] = {
                        "data": option,
                        "platforms": set(),
                    }
                platform = _engine_to_platform(engine)
                instance_data_map[db_instance_class]["platforms"].add(platform)
    if not instance_data_map:
        print(
            f"ERROR: No RDS instances found for family 'db.{family}' "
            f"with engines {db_engines}",
            file=sys.stderr,
        )
        return []
    debug_log(f"Found {len(instance_data_map)} unique instance classes")
    results = []
    for db_instance_class, info in instance_data_map.items():
        option = info["data"]
        platforms = sorted(info["platforms"])
        _, size = db_instance_class.rsplit(".", 1)
        ec2_instance_type = f"{family}.{size}"
        vcpu_count, cpu_cores, cpu_ghz, ram_gib, net_mbps = lookup_ec2_instance_specs(
            ec2_client, ec2_instance_type, debug
        )
        cpu_ipc_scale_factor = deduce_cpu_ipc_scale(vcpu_count, cpu_cores, cpu_perf)
        debug_log(
            f"{db_instance_class}: cpu={vcpu_count}, cores={cpu_cores}, "
            f"ghz={cpu_ghz}, ram={ram_gib} GiB, net={net_mbps} Mbps, "
            f"platforms={platforms}"
        )
        from service_capacity_modeling.interface import Platform

        new_instance = Instance(
            name=db_instance_class,
            cpu=vcpu_count,
            cpu_cores=cpu_cores,
            cpu_ghz=cpu_ghz,
            cpu_ipc_scale=cpu_ipc_scale_factor,
            ram_gib=ram_gib,
            net_mbps=net_mbps,
            drive=None,
            platforms=[Platform[p] for p in platforms],
        )
        results.append(new_instance)
    results = sorted(
        results, key=lambda i: normalized_aws_size(i.name.replace("db.", ""))
    )
    return results


def parse_iops(inp: Optional[str]) -> Optional[Tuple[int, int]]:
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


def parse_db_engines(engines_str: str) -> Sequence[str]:
    """Parse comma-separated database engines. Currently only aurora-postgresql is supported."""
    engines = [e.strip() for e in engines_str.split(",")]
    supported_engines = {"aurora-postgresql"}
    for engine in engines:
        if engine not in supported_engines:
            raise argparse.ArgumentTypeError(
                f"Database engine '{engine}' is not implemented yet. "
                f"Currently supported: {', '.join(sorted(supported_engines))}"
            )
    return engines


def _parse_family(family: str) -> Tuple[str, int, str]:
    series = family[0]
    num = re.findall(r"\d+", family)
    assert len(num) == 1
    gen = int(num[0])

    vendor = "intel"
    if family[-1] == "a":
        vendor = "amd"
    if family[-1] == "g":
        vendor = "graviton"

    return series, gen, vendor


def parse_io_curve(inp: str, family: str) -> FixedInterval:
    series, gen, _ = _parse_family(family)

    if series in ("i", "d") and gen < 7:
        # i4i is actually 6th gen
        # i3/i3en/d3 is actually 5th gen
        # AWS fixed this after 7th gen
        gen += 2

    if f"{gen}th-gen-{inp}" in latency_curve_ms:
        return latency_curve_ms[f"{gen}th-gen-{inp}"]

    return latency_curve_ms[inp]


def deduce_io_perf(
    family: str, curve: str, iops: Optional[Tuple[int, int]]
) -> Optional[IOPerformance]:
    if iops is None:
        guess = guess_iops(family)
        if guess is None:
            return None
        else:
            return IOPerformance(
                latency=parse_io_curve(curve, family),
                xl_read_iops=guess[0],
                xl_write_iops=guess[1],
            )
    else:
        return IOPerformance(
            latency=parse_io_curve(curve, family),
            xl_read_iops=iops[0],
            xl_write_iops=iops[1],
        )


def deduce_cpu_perf(
    family: str, ipc_scale_factor: Optional[float] = None
) -> Optional[CPUPerformance]:
    if ipc_scale_factor is not None:
        return CPUPerformance(
            ipc_scale_factor=ipc_scale_factor,
        )
    else:
        _, gen, vendor = _parse_family(family)
        if vendor == "intel" and gen == 7:
            # The 7th generation intel machines have a lower base clock
            # But in real world workloads are around 15% faster. So
            # (1.15 * 3.5) / 3.2 = 1.25
            return CPUPerformance(ipc_scale_factor=1.25)
    return CPUPerformance()


def main(args: Any) -> int:
    for family in args.families:
        is_rds = family.startswith("db.")
        if is_rds:
            rds_family = family[3:]
            if not args.db_engines:
                print(
                    f"ERROR: Family '{family}' appears to be RDS (starts with 'db.'), "
                    "but --db-engines was not specified. Please provide "
                    "--db-engines with comma-separated list like 'aurora-postgresql'",
                    file=sys.stderr,
                )
                return 1
            if args.xl_iops or args.io_latency_curve != "ssd":
                print(
                    "WARNING: --xl-iops and --io-latency-curve are ignored for RDS "
                    "instances (db.* families) as they use managed storage.",
                    file=sys.stderr,
                )
            cpu_perf = deduce_cpu_perf(
                family=rds_family,
                ipc_scale_factor=args.cpu_ipc_scale,
            )
            rds_client = boto3.client("rds", region_name=args.region)
            family_shapes = pull_rds_family(
                rds_client=rds_client,
                family=rds_family,
                db_engines=args.db_engines,
                cpu_perf=cpu_perf,
                debug=args.debug,
            )
            output_filename = f"auto_db_{rds_family}.json"
        else:
            io_perf = deduce_io_perf(
                family=family, curve=args.io_latency_curve, iops=args.xl_iops
            )
            cpu_perf = deduce_cpu_perf(
                family=family,
                ipc_scale_factor=args.cpu_ipc_scale,
            )
            ec2_client = boto3.client("ec2", region_name=args.region)
            family_shapes = pull_family(
                ec2_client=ec2_client,
                family=family,
                cpu_perf=cpu_perf,
                io_perf=io_perf,
                debug=args.debug,
            )
            output_filename = f"auto_{family}.json"
        json_shapes: Dict[str, Dict[str, Any]] = {"instances": {}}
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
        print(f"[{family}] Hardware Shapes", file=sys.stderr)
        print(json.dumps(json_shapes, indent=2))
        # Write to JSON file if requested
        if args.output_path is not None:
            path: Path = args.output_path
            if path.is_dir():
                output_path = Path(path, output_filename)
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
        result = client.describe_instance_type_offerings()
    except botocore.exceptions.ClientError:
        print(
            "Unable to connect to EC2. Do you have AWS credentials refreshed?",
            file=sys.stderr,
        )
        sys.exit(1)

    offerings = result["InstanceTypeOfferings"]
    for offering in offerings:
        families.add(offering["InstanceType"].rsplit(".", 1)[0])

    try:
        rds_client = boto3.client("rds", region_name="us-east-1")
        paginator = rds_client.get_paginator("describe_orderable_db_instance_options")
        page_iterator = paginator.paginate(Engine="aurora-postgresql")
        for page in page_iterator:
            for option in page["OrderableDBInstanceOptions"]:
                db_instance_class = option["DBInstanceClass"]
                families.add(db_instance_class.rsplit(".", 1)[0])
    except botocore.exceptions.ClientError:
        print(
            "Unable to connect to RDS. Do you have AWS credentials refreshed?",
            file=sys.stderr,
        )
        sys.exit(1)

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
            f"spec: {aws_instance_link} OR use fio to benchmark. Deduced from family "
            " if we can"
        ),
    )
    parser.add_argument(
        "--io-latency-curve", default="ssd", choices=latency_curve_ms.keys()
    )
    parser.add_argument(
        "--cpu-ipc-scale",
        type=float,
        help=(
            "If the clock frequency is not a good measure of performance, scale it by "
            "this much. Note that by default this will be 1, unless the cores are all "
            "full cores (not threads) in which case it will be 1.5 by default."
        ),
    )
    parser.add_argument(
        "--db-engines",
        type=parse_db_engines,
        default=None,
        help=(
            "Comma-separated list of database engines for RDS instances. "
            "Required when family starts with 'db.'. "
            "Currently supported: aurora-postgresql"
        ),
    )
    parser.add_argument("--region", choices=regions, default="us-east-1")
    parser.add_argument(
        "--output-path",
        type=Path,
        help=(
            "Output file path to copy the result to. If not given only stdout will"
            "occur. If running from the repo use: "
            "service_capacity_modeling/hardware/profiles/shapes/aws/"
        ),
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

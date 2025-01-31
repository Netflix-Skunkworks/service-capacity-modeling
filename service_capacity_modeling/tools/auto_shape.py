import argparse
import json
import sys
from collections import OrderedDict
from fractions import Fraction
from typing import Optional
from typing import Sequence

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import Instance


def pull_family(
    instance_filter: str = "m7a", region: str = "us-east-1"
) -> Sequence[Instance]:
    # pylint: disable=unused-argument
    """TODO: Talk to the AWS API

    This will emulate

    aws --region us-east-1 \
      ec2 describe-instance-types \
      --filters 'Name=instance-type,Values=m7a*'

    And then convert from base 10 for memory and disk to base 2 and generate the
    shape from that.

    Returns a list of Instance for hardware. Then we can compose to writing
    json files.
    """
    return []


def _scale_drive(drive: Optional[Drive], scale: Fraction) -> Optional[Drive]:
    if drive is None:
        return None
    # This is wrong but ... not too wrong
    single_tenant = False
    if scale > 4:
        single_tenant = True

    read_iops, write_iops = drive.read_io_per_s, drive.write_io_per_s
    if read_iops is not None:
        read_iops = int(round(read_iops * scale))
    if write_iops is not None:
        write_iops = int(round(write_iops * scale))

    return Drive(
        name=drive.name,
        size_gib=int(drive.size_gib * scale),
        read_io_latency_ms=drive.read_io_latency_ms,
        read_io_per_s=read_iops,
        write_io_per_s=write_iops,
        block_size_kib=drive.block_size_kib,
        single_tenant=single_tenant,
    )


def reshape_family(
    initial_shape: str, reference_family: str = "m5", region: str = "us-east-1"
) -> Sequence[Instance]:
    """Takes a single instance as a template and generates the other shapes

    Useful for filling in details missing from automated pull such as IOPS
    """
    shape = shapes.hardware.regions[region].instances[initial_shape]
    family, normalized_size = shape.family, shape.normalized_size

    normalized_sizes = [
        i.normalized_size
        for i in shapes.hardware.regions[region].instances.values()
        if i.family == reference_family
    ]

    def name(input_size: Fraction) -> str:
        if input_size.denominator == 1 and input_size.numerator != 1:
            size = str(input_size.numerator) + "xlarge"
        else:
            size = {
                Fraction(1, 4): "medium",
                Fraction(1, 2): "large",
                Fraction(1, 1): "xlarge",
            }[input_size]
        return f"{family}{shape.family_separator}{size}"

    new_shapes = []
    for new_size in normalized_sizes:
        if normalized_size == new_size:
            continue
        scale_factor = new_size / normalized_size

        current_shape = shapes.hardware.regions[region].instances.get(name(new_size))

        new_shape = Instance(
            name=name(new_size),
            cpu=int(shape.cpu * scale_factor),
            cpu_ghz=shape.cpu_ghz,
            ram_gib=round(shape.ram_gib * scale_factor, 2),
            net_mbps=round(shape.net_mbps * scale_factor, 2),
            drive=_scale_drive(shape.drive, scale_factor),
        )
        # When we have layers, reverse this and clean it up
        merged = new_shape
        if current_shape is not None:
            merged = new_shape.merge_with(current_shape)
        merged_dict = merged.model_dump(exclude_unset=True)
        if "annual_cost" in merged_dict:
            del merged_dict["annual_cost"]
        if "drive" in merged_dict and "annual_cost" in merged_dict["drive"]:
            del merged_dict["drive"]["annual_cost"]

        new_shapes.append(Instance(**merged_dict))

    return sorted(new_shapes, key=lambda i: i.normalized_size)


def main(args) -> int:
    for template in args.template:
        result = reshape_family(template, args.reference_family, args.region)
        print("Hardware shapes for aws.json")
        json_shapes = OrderedDict()
        for shape in result:
            model_dict = shape.model_dump(exclude_unset=True)
            # Hardware shouldn't have costs
            if "annual_cost" in model_dict:
                del model_dict["annual_cost"]
            if "drive" in model_dict and "annual_cost" in model_dict["drive"]:
                del model_dict["drive"]["annual_cost"]

            json_shapes[shape.name] = model_dict

        print(f"[{template}[aws.json] Hardware Shapes", file=sys.stderr)
        print(json.dumps(json_shapes, indent=2))

    return 0


if __name__ == "__main__":
    instances = set()
    families = set()
    regions = set()
    for ar in shapes.hardware.regions.keys():
        regions.add(ar)
        for instance in shapes.hardware.regions[ar].instances.values():
            instances.add(instance.name)
            families.add(instance.family)

    parser = argparse.ArgumentParser(
        prog="Project shapes from template",
        description="Takes one verified shape and projects for rest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-f", "--reference-family", choices=families, default="m5")
    parser.add_argument("-r", "--region", choices=regions, default="us-east-1")
    parser.add_argument("template", nargs="+", choices=instances)
    sys.exit(main(parser.parse_args()))

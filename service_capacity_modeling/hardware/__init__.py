# pylint: disable=cyclic-import
# in HardwareShapes.hardware it imports from hardware.profiles dynamically
import json
import logging
import os
from functools import reduce
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import GlobalHardware
from service_capacity_modeling.interface import Hardware
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Pricing
from service_capacity_modeling.interface import Service

logger = logging.getLogger(__name__)


def load_pricing(pricing: Dict) -> Pricing:
    return Pricing(regions=pricing)


def load_hardware(hardware: Dict) -> Hardware:
    return Hardware(**hardware)


def price_hardware(hardware: Hardware, pricing: Pricing) -> GlobalHardware:
    regions: Dict[str, Hardware] = {}

    for region, region_pricing in pricing.regions.items():
        priced_instances: Dict[str, Instance] = {}
        priced_drives: Dict[str, Drive] = {}
        priced_services: Dict[str, Service] = {}

        for instance, iprice in region_pricing.instances.items():
            if instance not in hardware.instances:
                logger.debug(
                    "Instance %s is not in hardware shapes, skipping", instance
                )
                continue
            priced_instances[instance] = hardware.instances[instance].model_copy()
            priced_instances[instance].annual_cost = iprice.annual_cost
            if iprice.lifecycle is not None:
                priced_instances[instance].lifecycle = iprice.lifecycle

        for drive, dprice in region_pricing.drives.items():
            if drive not in hardware.drives:
                logger.warning("Drive %s is not in hardware shapes, skipping", drive)
                continue
            priced_drives[drive] = hardware.drives[drive].model_copy()
            priced_drives[drive].annual_cost_per_gib = dprice.annual_cost_per_gib
            priced_drives[
                drive
            ].annual_cost_per_read_io = dprice.annual_cost_per_read_io
            priced_drives[
                drive
            ].annual_cost_per_write_io = dprice.annual_cost_per_write_io

        for svc, svc_price in region_pricing.services.items():
            if svc not in hardware.services:
                logger.warning("Service %s is not in hardware shapes, skipping", svc)
                continue
            priced_services[svc] = hardware.services[svc].model_copy()
            priced_services[svc].annual_cost_per_gib = svc_price.annual_cost_per_gib
            priced_services[
                svc
            ].annual_cost_per_read_io = svc_price.annual_cost_per_read_io
            priced_services[
                svc
            ].annual_cost_per_write_io = svc_price.annual_cost_per_write_io
            priced_services[svc].annual_cost_per_core = svc_price.annual_cost_per_core

        regions[region] = Hardware(
            instances=priced_instances,
            drives=priced_drives,
            services=priced_services,
            zones_in_region=region_pricing.zones_in_region,
        )

    return GlobalHardware(regions=regions)


def merge_hardware(existing: Hardware, override: Hardware) -> Hardware:
    """Merge two hardware sets, Unlike pricing does _not_ support overrides

    This method takes two shape files and merges instances, drives, services, etc.
    Unlike pricing this does not override - it will throw exceptions if two files
    contain the same keys
    """
    existing_obj = existing.model_dump()
    override_obj = override.model_dump()

    merged = {}
    for key in existing_obj.keys() | override_obj.keys():
        existing_field = existing_obj.get(key)
        if existing_field is None and override_obj.get(key) is None:
            continue
        if existing_field is None and override_obj.get(key) is not None:
            merged[key] = override_obj.get(key)
        elif isinstance(existing_field, Dict):
            override_field = override_obj.get(key)
            merged_field = merged.setdefault(key, {})

            existing_keys = existing_field.keys()
            override_keys = override_obj.get(key, {}).keys()
            for shape in existing_keys | override_keys:
                if shape in existing_keys and shape in override_keys:
                    raise ValueError(
                        f"Duplicate shape {shape}! "
                        "Only one file should contain a shape"
                    )
                if shape not in existing_keys:
                    merged_field[shape] = override_field[shape]
                else:
                    merged_field[shape] = existing_field[shape]
    return Hardware(**merged)


def merge_pricing(existing: Dict, override: Dict) -> Dict:
    merged = existing.copy()
    for region, override_pricing in override.items():
        if region not in merged:
            merged[region] = override_pricing
        else:
            for instance, instance_data in override_pricing["instances"].items():
                if merged[region]["instances"].get(instance) is None:
                    merged[region]["instances"][instance] = instance_data.copy()
                else:
                    merged[region]["instances"][instance].update(instance_data)

            for drive, dprice in override_pricing["drives"].items():
                if merged[region]["drives"].get(drive) is None:
                    merged[region]["drives"][drive] = dprice.copy()
                else:
                    merged[region]["drives"][drive].update(dprice)

            for service, sprice in override_pricing["services"].items():
                if merged[region]["services"].get(service) is None:
                    merged[region]["services"][service] = sprice.copy()
                else:
                    merged[region]["services"][service].update(sprice)

    return merged


def load_hardware_from_disk(
    price_paths: Union[List[Path], Optional[str]] = os.environ.get("PRICE_PATH"),
    shape_paths: Union[List[Path], Optional[str]] = os.environ.get("HARDWARE_SHAPES"),
) -> GlobalHardware:
    if isinstance(shape_paths, list) and len(shape_paths) == 0:
        return GlobalHardware(regions={})
    if isinstance(price_paths, list) and len(price_paths) == 0:
        return GlobalHardware(regions={})

    if isinstance(shape_paths, str):
        shape_paths = [Path(shape_paths)]
    if isinstance(price_paths, str):
        price_paths = [Path(price_paths)]

    if price_paths is None:
        price_paths = []
    if shape_paths is None:
        shape_paths = []

    combined_pricing: Dict = {}

    logger.debug("Loading pricing from: %s", price_paths)
    for price_path in price_paths:
        logger.debug("Loading pricing from: %s", price_path)
        with open(price_path, encoding="utf-8") as pfd:
            pricing_data = json.load(pfd)
            combined_pricing = merge_pricing(combined_pricing, pricing_data)

    pricing = load_pricing(combined_pricing)

    hw_shapes = [Hardware()]
    for path in shape_paths:
        with open(path, encoding="utf-8") as fd:
            hw_shapes.append(Hardware(**json.load(fd)))

    hardware = reduce(merge_hardware, hw_shapes)
    return price_hardware(hardware=hardware, pricing=pricing)


def load_hardware_from_s3(bucket, path) -> GlobalHardware:
    try:
        # boto is a heavy dependency so we only want to take it if
        # someone will be using it ...
        try:
            import boto3
            import botocore
        except ImportError:
            return GlobalHardware(regions={})

        s3 = boto3.resource("s3")
        obj = s3.Object(bucket, path)
        data = json.loads(obj.get()["Body"].read().decode("utf-8"))
        return GlobalHardware(**data)
    except botocore.exceptions.ClientError as exp:
        logger.exception(exp)
        return GlobalHardware(regions={})


class HardwareShapes:
    def __init__(self):
        self._hardware: Optional[GlobalHardware] = None

    def load(self, new_hardware: GlobalHardware) -> None:
        self._hardware = new_hardware

    @property
    def hardware(self) -> GlobalHardware:
        if self._hardware is None:
            from service_capacity_modeling.hardware.profiles import common_profiles

            self._hardware = common_profiles["aws-3yr-reserved"]
        return self._hardware

    def region(self, region: str) -> Hardware:
        return self.hardware.regions[region]

    def instance(self, name: str, region: Optional[str] = None) -> Instance:
        if region is not None:
            return self.region(region).instances[name]

        for _, hw in self.hardware.regions.items():
            if name in hw.instances:
                return hw.instances[name]

        raise KeyError(f"Unknown instance {name}")


shapes: HardwareShapes = HardwareShapes()

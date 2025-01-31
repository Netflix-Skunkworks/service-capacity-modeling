# pylint: disable=cyclic-import
# in HardwareShapes.hardware it imports from hardware.profiles dynamically
import json
import logging
import os
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
                logger.warning(
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
    shape_path: Union[Optional[Path], Optional[str]] = os.environ.get(
        "HARDWARE_SHAPES"
    ),
) -> GlobalHardware:
    if shape_path is None or price_paths is None:
        return GlobalHardware(regions={})

    if isinstance(price_paths, list) and len(price_paths) == 0:
        return GlobalHardware(regions={})

    if isinstance(price_paths, str):
        price_paths = [Path(price_paths)]

    if isinstance(shape_path, str):
        shape_path = Path(shape_path)

    combined_pricing: Dict = {}

    print("Loading pricing from", price_paths, "\n")
    for price_path in price_paths:
        print("Loading pricing from", price_path, "\n")
        with open(price_path, encoding="utf-8") as pfd:
            pricing_data = json.load(pfd)
            combined_pricing = merge_pricing(combined_pricing, pricing_data)

    pricing = load_pricing(combined_pricing)

    with open(shape_path, encoding="utf-8") as sfd:
        hardware = load_hardware(json.load(sfd))

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


shapes: HardwareShapes = HardwareShapes()

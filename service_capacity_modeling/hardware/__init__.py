# pylint: disable=cyclic-import
# in HardwareShapes.hardware it imports from hardware.profiles dynamically
import json
import logging
import os
from typing import Dict
from typing import Optional

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
                logger.warning(f"Instance {instance} is not in hardware shapes, skipping")
                continue
            priced_instances[instance] = hardware.instances[instance].model_copy()
            priced_instances[instance].annual_cost = iprice.annual_cost
            if iprice.lifecycle is not None:
                priced_instances[instance].lifecycle = iprice.lifecycle

        for drive, dprice in region_pricing.drives.items():
            if drive not in hardware.drives:
                logger.warning(f"Drive {drive} is not in hardware shapes, skipping")
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
                logger.warning(f"Service {svc} is not in hardware shapes, skipping")
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


def merge_pricing(_pricing1: Dict, pricing2: Dict) -> Dict:
    pricing1 = _pricing1.copy()
    for region, region_pricing in pricing2.items():
        print("region", region)

        if region not in pricing1:
            pricing1[region] = region_pricing
        else:
            for instance, iprice in region_pricing["instances"].items():
                pricing1[region]["instances"][instance] = iprice

            for drive, dprice in region_pricing["drives"].items():
                pricing1[region]["drives"][drive] = dprice

            for service, sprice in region_pricing["services"].items():
                pricing1[region]["services"][service] = sprice

    return pricing1


def load_hardware_from_disk(
    price_paths=[os.environ.get("PRICE_PATH")],
    shape_path=os.environ.get("HARDWARE_SHAPES"),
) -> GlobalHardware:
    if price_paths is not None and len(price_paths) > 0 and shape_path is not None:
        combined_pricing = {}

        for price_path in price_paths:
            print("Loading pricing from", price_path)
            with open(price_path, encoding="utf-8") as pfd:
                pricing_data = json.load(pfd)
                combined_pricing = merge_pricing(combined_pricing, pricing_data)

        # Convert combined pricing dict to Pricing object
        print("combined_pricing", combined_pricing)
        pricing = load_pricing(combined_pricing)

        # Load hardware shapes
        with open(shape_path, encoding="utf-8") as sfd:
            hardware = load_hardware(json.load(sfd))

        return price_hardware(hardware=hardware, pricing=pricing)
    else:
        return GlobalHardware(regions={})


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

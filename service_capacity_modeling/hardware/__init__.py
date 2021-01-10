# -*- coding: utf-8 -*-
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

        for instance, iprice in region_pricing.instances.items():
            priced_instances[instance] = hardware.instances[instance].copy()
            priced_instances[instance].annual_cost = iprice.annual_cost

        for drive, dprice in region_pricing.drives.items():
            priced_drives[drive] = hardware.drives[drive].copy()
            pd = priced_drives[drive]
            pd.annual_cost_per_gib = dprice.annual_cost_per_gib
            pd.annual_cost_per_read_io = dprice.annual_cost_per_read_io
            pd.annual_cost_per_write_io = dprice.annual_cost_per_write_io

        regions[region] = Hardware(
            instances=priced_instances,
            drives=priced_drives,
        )

    return GlobalHardware(regions=regions)


def load_hardware_from_disk(
    price_path=os.environ.get("PRICE_PATH"),
    shape_path=os.environ.get("HARDWARE_SHAPES"),
) -> GlobalHardware:
    if price_path is not None and shape_path is not None:
        with open(price_path) as pfd:
            pricing = load_pricing(json.load(pfd))

        with open(shape_path) as sfd:
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

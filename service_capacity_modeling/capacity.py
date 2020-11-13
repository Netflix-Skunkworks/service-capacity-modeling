# -*- coding: utf-8 -*-
import logging
from typing import Callable
from typing import Dict
from typing import Sequence

from service_capacity_modeling.capacity_models.cassandra import (
    estimate_cassandra_cluster_zone,
)
from service_capacity_modeling.capacity_models.stateless_java_app import (
    estimate_java_app_region,
)
from service_capacity_modeling.hardware import HardwareShapes
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.models import CapacityDesires
from service_capacity_modeling.models import CapacityPlan
from service_capacity_modeling.models import GlobalHardware

logger = logging.getLogger(__name__)


class CapacityPlanner:
    def __init__(self):
        self._shapes: HardwareShapes = shapes
        self._models: Dict[str, Callable[..., CapacityPlan]] = {}

    def register_model(self, name: str, model: Callable[..., CapacityPlan]):
        self._models[name] = model

    @property
    def hardware_shapes(self) -> GlobalHardware:
        return self._shapes.hardware

    def plan(
        self, model_name: str, region: str, desires: CapacityDesires, *args, **kwargs
    ) -> CapacityPlan:
        hardware = self._shapes.region(region)
        return self._models[model_name](
            # Pass Hardware
            hardware=hardware,
            # Pass CapacityDesires
            desires=desires,
            # Pass arbitrary additional things
            *args,
            **kwargs
        )

    @property
    def models(self) -> Sequence[str]:
        return self._models.keys()


planner = CapacityPlanner()
planner.register_model(name="nflx_stateless_java_app", model=estimate_java_app_region)
planner.register_model(name="nflx_cassandra", model=estimate_cassandra_cluster_zone)

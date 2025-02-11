from typing import List
from unittest.mock import Mock

from service_capacity_modeling.capacity_planner import CapacityPlanner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import CurrentClusters
from service_capacity_modeling.interface import CurrentRegionClusterCapacity
from service_capacity_modeling.interface import Lifecycle
from service_capacity_modeling.interface import Platform


def test_generate_scenarios():
    # Create a CapacityPlanner instance
    planner = CapacityPlanner()

    # Mock the model
    model = Mock()
    model.allowed_platforms.return_value = {Platform.amd64}
    model.allowed_cloud_drives.return_value = {}
    model.run_hardware_simulation.return_value = True

    region = "us-east-1"
    desires = CapacityDesires()
    num_regions = 3
    lifecycles = [Lifecycle.stable]
    instance_families: List[str] = []
    drives: List[str] = []

    # Call generate_scenarios
    scenarios = list(
        planner.generate_scenarios(
            model, region, desires, num_regions, lifecycles, instance_families, drives
        )
    )

    # Check we got some instances
    assert len(scenarios) > 0


def test_generate_scenarios_limit_family():
    # Create a CapacityPlanner instance
    planner = CapacityPlanner()

    # Mock the model
    model = Mock()
    model.allowed_platforms.return_value = {Platform.amd64}
    model.allowed_cloud_drives.return_value = {}
    model.run_hardware_simulation.return_value = True

    region = "us-east-1"
    desires = CapacityDesires()
    num_regions = 3
    lifecycles = [Lifecycle.stable]
    # Limit to m5 family
    instance_families = ["m5"]
    drives: List[str] = []

    # Call generate_scenarios
    scenarios = list(
        planner.generate_scenarios(
            model, region, desires, num_regions, lifecycles, instance_families, drives
        )
    )

    # Check we got m5s
    assert len(scenarios) > 0
    for instance, _, _ in scenarios:
        assert instance.family == "m5"


def test_generate_scenarios_desire_resources():
    # Create a CapacityPlanner instance
    planner = CapacityPlanner()

    # Mock the model
    model = Mock()
    model.allowed_platforms.return_value = {Platform.amd64}
    model.allowed_cloud_drives.return_value = {}
    model.run_hardware_simulation.return_value = True

    region = "us-east-1"
    num_regions = 3
    lifecycles = [Lifecycle.stable]
    # Limit to m5 family
    instance_families = ["m5"]
    drives: List[str] = []

    desires = CapacityDesires()
    # Set cpu and memory requirements via application desires
    desires.data_shape.reserved_instance_app_mem_gib = 64
    desires.data_shape.reserved_instance_system_mem_gib = 0
    desires.query_pattern.estimated_read_parallelism = certain_int(16)
    desires.query_pattern.estimated_write_parallelism = certain_int(0)

    # Call generate_scenarios
    scenarios = list(
        planner.generate_scenarios(
            model, region, desires, num_regions, lifecycles, instance_families, drives
        )
    )

    # Check we got m5s
    assert len(scenarios) > 0
    for instance, _, _ in scenarios:
        assert instance.family == "m5"
        assert instance.cpu >= 16
        assert instance.ram_gib >= 64


def test_generate_scenarios_current_resources():
    # Create a CapacityPlanner instance
    planner = CapacityPlanner()

    # Mock the model
    model = Mock()
    model.allowed_platforms.return_value = {Platform.amd64}
    model.allowed_cloud_drives.return_value = {}
    model.run_hardware_simulation.return_value = True

    region = "us-east-1"
    num_regions = 3
    lifecycles = [Lifecycle.stable]
    # Limit to m5 family
    instance_families = ["m5"]
    drives: List[str] = []

    desires = CapacityDesires()
    # Set cpu and memory requirements via current capacity
    desires.current_clusters = CurrentClusters()
    desires.current_clusters.regional = [
        CurrentRegionClusterCapacity(
            cluster_instance_name="m5.12xlarge",
            cluster_instance_count=certain_int(5),
            cpu_utilization=certain_float(100.0),
            memory_utilization_gib=certain_float(16),
            network_utilization_mbps=certain_float(128.0),
        )
    ]

    # Call generate_scenarios
    scenarios = list(
        planner.generate_scenarios(
            model, region, desires, num_regions, lifecycles, instance_families, drives
        )
    )

    # Check we got m5s
    assert len(scenarios) > 0
    for instance, _, _ in scenarios:
        assert instance.family == "m5"
        assert instance.ram_gib >= 16

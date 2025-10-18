from typing import Any
from typing import Dict

import pytest

from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import Buffer
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import Buffers
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import Consistency
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import GlobalConsistency
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models import CapacityModel

valid_query_pattern = QueryPattern(
    access_pattern=AccessPattern.latency,
    access_consistency=GlobalConsistency(
        same_region=Consistency(
            target_consistency=AccessConsistency.read_your_writes,
            staleness_slo_sec=FixedInterval(low=0, mid=0.1, high=1),
        ),
        cross_region=Consistency(
            target_consistency=AccessConsistency.best_effort,
            staleness_slo_sec=FixedInterval(low=10, mid=60, high=600),
        ),
    ),
    estimated_read_per_second=certain_float(1000),
    estimated_write_per_second=certain_float(100),
    estimated_mean_read_latency_ms=certain_float(1),
    estimated_mean_write_latency_ms=certain_float(1),
    estimated_mean_read_size_bytes=certain_float(1024),
    estimated_mean_write_size_bytes=certain_float(512),
    estimated_read_parallelism=certain_float(1),
    estimated_write_parallelism=certain_float(1),
    read_latency_slo_ms=FixedInterval(low=0.4, mid=4, high=10, confidence=0.98),
    write_latency_slo_ms=FixedInterval(low=0.4, mid=4, high=10, confidence=0.98),
)

valid_data_shape = DataShape(
    estimated_state_size_gib=certain_float(100),
    estimated_working_set_percent=certain_float(0.8),
    estimated_compression_ratio=certain_float(1),
    reserved_instance_app_mem_gib=2,
    reserved_instance_system_mem_gib=1,
    durability_slo_order=FixedInterval(
        low=1000, mid=10000, high=100000, confidence=0.98
    ),
)


class ValidModel(CapacityModel):
    """A model that properly sets all required fields"""

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ):
        return CapacityDesires(
            query_pattern=valid_query_pattern,
            data_shape=valid_data_shape,
            buffers=Buffers(default=Buffer(ratio=1.5)),
        )


class InvalidModelMissingBuffer(CapacityModel):
    """A model that doesn't set buffers"""

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ):
        return CapacityDesires(
            query_pattern=valid_query_pattern,
            data_shape=valid_data_shape,
            # Missing buffers
        )


class InvalidModelMissingBufferDefault(CapacityModel):
    """A model that doesn't set buffers"""

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ):
        return CapacityDesires(
            query_pattern=valid_query_pattern,
            data_shape=valid_data_shape,
            buffers=Buffers(
                desired={
                    "compute": Buffer(ratio=1.5, components=[BufferComponent.compute])
                }
            ),  # Missing default buffer
        )


class InvalidModelMissingTopLevel(CapacityModel):
    """A model that doesn't set query_pattern"""

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ):
        # Only sets data_shape, missing query_pattern
        return CapacityDesires(
            data_shape=DataShape(
                estimated_state_size_gib=certain_float(100),
                estimated_working_set_percent=certain_float(0.8),
            ),
        )


class InvalidModelMissingDataShape(CapacityModel):
    """A model that sets query_pattern but leaves nested fields as defaults"""

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ):
        # Sets query_pattern but doesn't explicitly set data shape
        return CapacityDesires(
            query_pattern=valid_query_pattern,
        )


class InvalidModelMissingQueryPattern(CapacityModel):
    """A model that sets query_pattern but leaves nested fields as defaults"""

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ):
        # Sets query_pattern but doesn't explicitly set required query pattern
        return CapacityDesires(
            data_shape=valid_data_shape,
        )


class InvalidModelPartiallySetDataModels(CapacityModel):
    """A model that partially sets nested fields but misses some"""

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ):
        return CapacityDesires(
            query_pattern=valid_query_pattern,
            data_shape=DataShape(
                estimated_state_size_gib=certain_float(100),
                estimated_working_set_percent=certain_float(0.8),
            ),
        )


class InvalidModelPartiallySetQueryPattern(CapacityModel):
    """A model that partially sets nested fields but misses some"""

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ):
        return CapacityDesires(
            query_pattern=QueryPattern(
                access_pattern=AccessPattern.latency,
                # Missing access_consistency and other required fields
                estimated_mean_read_latency_ms=certain_float(1),
            ),
            data_shape=valid_data_shape,
        )


def test_valid_model():
    """Test that a properly implemented model passes validation"""
    model = ValidModel()
    # Should not raise
    model.validate_implementation()


def test_invalid_model_missing_top_level():
    """Test that missing top-level required field is caught"""
    model = InvalidModelMissingTopLevel()
    with pytest.raises(ValueError, match="query_pattern is required"):
        model.validate_implementation()


def test_invalid_model_missing_nested():
    """Test that missing nested required fields are caught"""
    model = InvalidModelMissingQueryPattern()
    with pytest.raises(
        ValueError, match="query_pattern is required and must be explicitly set"
    ):
        model.validate_implementation()

    model = InvalidModelMissingDataShape()
    with pytest.raises(
        ValueError, match="data_shape is required and must be explicitly set"
    ):
        model.validate_implementation()

    model = InvalidModelPartiallySetQueryPattern()
    with pytest.raises(
        ValueError, match="query_pattern\\..*is required and must be explicitly set"
    ):
        model.validate_implementation()

    model = InvalidModelPartiallySetDataModels()
    with pytest.raises(
        ValueError, match="data_shape\\..*is required and must be explicitly set"
    ):
        model.validate_implementation()


def test_invalid_model_buffers():
    """Test that partially set nested fields are caught"""
    model = InvalidModelMissingBuffer()
    with pytest.raises(
        ValueError, match="buffers is required and must be explicitly set"
    ):
        model.validate_implementation()

    model = InvalidModelMissingBufferDefault()
    with pytest.raises(
        ValueError, match="buffers\\.default is required and must be explicitly set"
    ):
        model.validate_implementation()

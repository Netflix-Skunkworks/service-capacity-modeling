from typing import Optional

from pytest import approx

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import ExcludeUnsetModel
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.models.common import normalize_cores


class Approximation(ExcludeUnsetModel):
    """https://docs.pytest.org/en/7.1.x/reference/reference.html#pytest-approx"""

    rel: Optional[float] = None
    abs: Optional[float] = None


class PlanVariance(ExcludeUnsetModel):
    cpu: Optional[Approximation] = Approximation(rel=0.20, abs=None)
    memory: Optional[Approximation] = None
    cost: Optional[Approximation] = Approximation(rel=0.20, abs=None)


def shape(name: str) -> Instance:
    return shapes.region("us-east-1").instances[name]


def assert_similar_compute(
    expected_shape: Instance,
    actual_shape: Instance,
    expected_count: int = 1,
    actual_count: int = 1,
    allowed_variance=PlanVariance(),
):
    """Assert that a plan is roughly equal to expectations

    Checks CPU, memory, and cost against the allowed variance which defaults
    to 20% - in a way that is NOT fragile to computer speed improvements
    """

    if allowed_variance.cpu is not None:
        expected_cores, actual_cores = (
            expected_shape.cpu * expected_count,
            actual_shape.cpu * actual_count,
        )
        normalized_actual_cores = normalize_cores(
            actual_cores, target_shape=expected_shape, reference_shape=actual_shape
        )

        msg = (
            f"[CPU] Expected within {allowed_variance.cpu.model_dump_json()} of "
            f"[cpu={expected_cores}, name={expected_shape.name}], Actual "
            f"[norm_cpu={normalized_actual_cores}, cpu={actual_cores}, "
            f"name={actual_shape.name}]"
        )
        assert normalized_actual_cores == approx(
            expected_cores, rel=allowed_variance.cpu.rel, abs=allowed_variance.cpu.abs
        ), msg

    if allowed_variance.memory is not None:
        expected_mem = expected_shape.ram_gib * expected_count
        actual_mem = actual_shape.ram_gib * actual_count
        msg = (
            f"[Memory] Expected within {allowed_variance.model_dump_json()} of "
            f"[mem_gib={expected_mem}, name={expected_shape.name}], Actual "
            f"[mem_gib={actual_mem}, name={actual_shape.name}]"
        )
        assert actual_mem == approx(
            expected_mem,
            rel=allowed_variance.memory.rel,
            abs=allowed_variance.memory.abs,
        ), msg

    if allowed_variance.cost is not None:
        expected_cost = expected_shape.annual_cost * expected_count
        actual_cost = actual_shape.annual_cost * actual_count
        msg = (
            f"[Spend] Expected within {allowed_variance.cost.model_dump_json()} of "
            f"[cost={expected_cost}, name={expected_shape.name}], Actual "
            f"[cost={actual_cost}, name={actual_shape.name}]"
        )
        assert actual_cost == approx(
            expected_cost, rel=allowed_variance.cost.rel, abs=allowed_variance.cost.abs
        ), msg

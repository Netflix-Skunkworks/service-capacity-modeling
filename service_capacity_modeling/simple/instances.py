from typing import Optional

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import Instance

FULL_CORE_INSTANCE_FAMILIES = ["c7a", "m7a", "r7a"]


def _family_from_instance_type(instance_type: str) -> str:
    return instance_type.split(".")[0]


def _headroom_approx(cores: int, is_ht: bool) -> float:
    """For implementation see /notebooks/headroom-estimator.ipynb"""

    # Hyperthreading performance penalty
    # When hyperthreading is enabled, each core is not as effective as a physical core
    # We estimate each virtual core to be about 60% as effective as a physical core
    HT_PENALTY = 0.6

    # Adjust effective cores if hyperthreading is enabled
    # This accounts for the reduced effectiveness of virtual cores
    cores_f = float(cores)
    if is_ht:
        cores_f = cores_f * HT_PENALTY

    # Calculate required headroom using magic formula:
    return 0.712 / (cores_f**0.448)


def _is_instance_type_hyperthreads(instance_type: str) -> bool:
    instance_family = _family_from_instance_type(instance_type)
    return instance_family not in FULL_CORE_INSTANCE_FAMILIES


def get_simple_instance_headroom_target(instance: Instance) -> Optional[float]:
    """Determine an approximate headroom target for an instance given its
    instance type.

    The headroom target should be the percentage of CPU that should be
    reserved for headroom to ensure sensible performance profile.

    This could be 1-utilization_target, however we leave the ultimate
    utilization_target to the caller, since, we do not know how much
    operational headroom they want to leave (ie: success buffer).

    For example, a response here of "headroom = 15%", means caller could
    decide with a success_buffer=1 to use a utilization_target of 85%.
    For success_buffer>1, they should target below 85% utilization.

    This is only suitable for "single-thread-like" workloads, which
    fortunately many stateless services are.

    For implementation see /notebooks/headroom-estimator.ipynb
    """
    try:
        is_ht = _is_instance_type_hyperthreads(instance.name)
        core_count = instance.cpu
        return _headroom_approx(core_count, is_ht)
    except (KeyError, AttributeError):
        return None


def get_simple_instance_headroom_target_for_name(instance_type: str) -> Optional[float]:
    try:
        instance = shapes.hardware.regions["us-east-1"].instances[instance_type]
        return get_simple_instance_headroom_target(instance)
    except (KeyError, AttributeError):
        return None

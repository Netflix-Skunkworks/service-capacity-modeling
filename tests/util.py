from typing import Optional

from pytest import approx

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import ClusterCapacity
from service_capacity_modeling.interface import Drive
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


def shape(name: str, region="us-east-1") -> Instance:
    return shapes.region(region).instances[name]


def drive(name: str, region: str = "us-east-1") -> Drive:
    """Get a fully-priced drive definition from hardware profiles.

    This returns the base drive definition with all pricing information
    populated. Use .model_copy() and set size_gib, read_io_per_s, and
    write_io_per_s to create a specific drive instance for testing.

    Args:
        name: Drive name (e.g., "gp3", "gp2", "io2")
        region: AWS region (default: "us-east-1")

    Returns:
        Fully-priced Drive with all hardware specs and costs

    Example:
        >>> gp3 = drive("gp3")
        >>> gp3_instance = gp3.model_copy()
        >>> gp3_instance.size_gib = 100
        >>> gp3_instance.read_io_per_s = 3000
        >>> gp3_instance.write_io_per_s = 3000
        >>> print(gp3_instance.annual_cost)  # Calculates tiered cost
    """
    return shapes.region(region).drives[name]


def simple_drive(
    name: str = "gp3",
    size_gib: int = 100,
    read_io_per_s: int = 3000,
    write_io_per_s: int = 3000,
    region: str = "us-east-1",
) -> Drive:
    """Create a drive instance with specific size and IOPS for testing.

    This is a convenience function that gets the base drive definition,
    copies it, and sets the size and IOPS. The annual_cost property
    will automatically calculate the correct tiered pricing.

    Args:
        name: Drive name (default: "gp3")
        size_gib: Drive size in GiB (default: 100)
        read_io_per_s: Read IOPS (default: 3000)
        write_io_per_s: Write IOPS (default: 3000)
        region: AWS region (default: "us-east-1")

    Returns:
        Drive instance with specified configuration and calculated cost

    Example:
        >>> # Create a 500 GiB gp3 drive with 5000 read/write IOPS
        >>> my_drive = simple_drive(
        ...     size_gib=500, read_io_per_s=5000, write_io_per_s=5000
        ... )
        >>> assert my_drive.size_gib == 500
        >>> assert my_drive.annual_cost > 0  # Automatically calculated
    """
    base_drive = drive(name, region)
    drive_instance = base_drive.model_copy()
    drive_instance.size_gib = size_gib
    drive_instance.read_io_per_s = read_io_per_s
    drive_instance.write_io_per_s = write_io_per_s
    return drive_instance


def assert_similar_compute(
    expected_shape: Instance,
    actual_shape: Instance,
    expected_count: int = 1,
    actual_count: int = 1,
    *,
    allowed_variance=PlanVariance(),
    expected_attached_disk: Optional[Drive] = None,
    actual_attached_disk: Optional[Drive] = None,
):
    """Assert that a capacity plan matches expectations within tolerance.

    Compares CPU, memory, and cost between expected and actual cluster
    configurations. Optionally includes attached drive costs in the comparison.
    Uses normalized CPU comparisons to avoid brittleness from hardware
    improvements (e.g., newer CPUs being faster per core).

    Note: When drives are specified, only their annual_cost is included in
    the cost comparison. Drive properties like size_gib, read_io_per_s, and
    write_io_per_s are NOT validated by this function - use separate assertions
    for drive-specific checks if needed.

    Args:
        expected_shape: Expected instance type
        actual_shape: Actual instance type from capacity plan
        expected_count: Expected number of instances (default: 1)
        actual_count: Actual number of instances (default: 1)
        allowed_variance: Tolerance for CPU/memory/cost differences
            (default: 20% for CPU and cost, memory not checked)
        expected_attached_disk: Expected attached drive for cost calculation
            (optional). Only the drive's annual_cost is used in comparison.
        actual_attached_disk: Actual attached drive from capacity plan
            (optional). Only the drive's annual_cost is used in comparison.

    Raises:
        AssertionError: If actual values are outside allowed variance

    Example:
        >>> from tests.util import assert_similar_compute, simple_drive
        >>> from service_capacity_modeling.hardware import shapes
        >>> # Compare compute without drives
        >>> assert_similar_compute(
        ...     expected_shape=shapes.instance("m6id.4xlarge"),
        ...     actual_shape=result.instance,
        ...     expected_count=8,
        ...     actual_count=result.count,
        ... )
        >>> # Compare compute with attached drives (includes drive cost)
        >>> expected_drive = simple_drive(size_gib=500, read_io_per_s=5000)
        >>> assert_similar_compute(
        ...     expected_shape=shapes.instance("m6id.4xlarge"),
        ...     actual_shape=result.instance,
        ...     expected_count=8,
        ...     actual_count=result.count,
        ...     expected_attached_disk=expected_drive,
        ...     actual_attached_disk=result.attached_drives[0],
        ... )
        >>> # If you need to check drive properties, add separate assertions
        >>> assert result.attached_drives[0].name == "gp3"
        >>> assert result.attached_drives[0].size_gib >= 500
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
        if expected_attached_disk is not None:
            expected_cost += expected_attached_disk.annual_cost * expected_count

        actual_cost = actual_shape.annual_cost * actual_count
        if actual_attached_disk is not None:
            actual_cost += actual_attached_disk.annual_cost * actual_count

        msg = (
            f"[Spend] Expected within {allowed_variance.cost.model_dump_json()} of "
            f"[cost={expected_cost}, name={expected_shape.name}], Actual "
            f"[cost={actual_cost}, name={actual_shape.name}]"
        )
        assert actual_cost == approx(
            expected_cost, rel=allowed_variance.cost.rel, abs=allowed_variance.cost.abs
        ), msg


def get_drive_size_gib(cluster: ClusterCapacity) -> int:
    """
    Get drive size in GiB from a cluster (attached or local storage).

    This function checks for attached drives first, then falls back to
    instance drives (local). This makes tests tolerant to the storage type
    chosen by the capacity planner.

    Args:
        cluster: The cluster capacity to inspect

    Returns:
        The size of the drive in GiB

    Raises:
        ValueError: If the cluster has neither attached nor instance drives
    """
    # Check attached_drives first
    if cluster.attached_drives:
        return cluster.attached_drives[0].size_gib

    # Fall back to checking the instance's built-in drive (local)
    if cluster.instance.drive is not None:
        return cluster.instance.drive.size_gib

    raise ValueError(
        f"Instance {cluster.instance.name} has no drive or attached drives"
    )


def get_total_storage_gib(cluster: ClusterCapacity) -> int:
    """
    Get the total storage in GiB across all instances in a cluster.

    Args:
        cluster: The cluster capacity to inspect

    Returns:
        Total storage in GiB (count * drive_size_gib)
    """
    return cluster.count * get_drive_size_gib(cluster)


def has_attached_storage(cluster: ClusterCapacity) -> bool:
    """
    Check if a cluster uses attached storage.

    Args:
        cluster: The cluster capacity to inspect

    Returns:
        True if using attached disks, False if using local disks
    """
    return bool(cluster.attached_drives)


def has_local_storage(cluster: ClusterCapacity) -> bool:
    """
    Check if a cluster uses local (instance) storage.

    Args:
        cluster: The cluster capacity to inspect

    Returns:
        True if using local storage, False if using attached disks
    """
    return cluster.instance.drive is not None and not cluster.attached_drives


def assert_minimum_storage_gib(cluster: ClusterCapacity, min_storage_gib: int) -> None:
    """
    Assert that a cluster has at least the specified amount of storage,
    regardless of storage type.

    Args:
        cluster: The cluster capacity to inspect
        min_storage_gib: Minimum expected storage in GiB

    Raises:
        AssertionError: If storage is less than the minimum
    """
    total_storage = get_total_storage_gib(cluster)
    storage_type = "attached" if has_attached_storage(cluster) else "local"
    assert total_storage >= min_storage_gib, (
        f"Expected at least {min_storage_gib} GiB of storage, "
        f"got {total_storage} GiB ({storage_type})"
    )

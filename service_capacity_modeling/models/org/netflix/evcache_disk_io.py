"""Observed per-replica disk demand and candidate local-drive sizing."""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CurrentZoneClusterCapacity
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.models.common import DerivedBuffers
from service_capacity_modeling.models.common import RequirementFromCurrentCapacity
from service_capacity_modeling.models.common import buffer_for_components
from service_capacity_modeling.models.common import ceil_ignoring_float_error


class EVCacheDiskIO(BaseModel):
    """Unbuffered read/write rates for one complete EVCache replica."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    read_bytes_per_second: float = Field(ge=0)
    write_bytes_per_second: float = Field(ge=0)
    read_ops_per_second: float = Field(ge=0)
    write_ops_per_second: float = Field(ge=0)

    def node_load(self, drive: Drive) -> float:
        """Fractional nodes needed, sharing each device's read/write budget.

        Local-drive profiles express bandwidth through IOPS and block size.
        That bandwidth is an estimate, not an independent measured limit.
        """
        if drive.block_size_kib <= 0:
            raise ValueError(f"Invalid block size for {drive.name}")
        operations = 0.0
        bandwidth = 0.0
        for direction in ("read", "write"):
            ops = getattr(self, f"{direction}_ops_per_second")
            data = getattr(self, f"{direction}_bytes_per_second")
            capacity = getattr(drive, f"{direction}_io_per_s") or 0
            if capacity <= 0:
                if ops > 0 or data > 0:
                    raise ValueError(
                        f"Missing {direction} IO capacity for {drive.name}"
                    )
                continue
            operations += ops / capacity
            bandwidth += data / (capacity * drive.block_size_kib * 1024)
        if drive.throughput is not None:
            if drive.throughput <= 0:
                raise ValueError(f"Invalid throughput capacity for {drive.name}")
            bandwidth = (self.read_bytes_per_second + self.write_bytes_per_second) / (
                drive.throughput * 1024**2
            )
        return max(operations, bandwidth)


def _zero_demand_capacity_ratio(
    current_drive: Drive, candidate_drive: Drive, observed: bool
) -> float:
    # No traffic supplies a read/write mix, so preserve each capacity axis.
    ratios = []
    for direction in ("read", "write"):
        current_ops = getattr(current_drive, f"{direction}_io_per_s") or 0
        candidate_ops = getattr(candidate_drive, f"{direction}_io_per_s") or 0
        if candidate_ops <= 0 < current_ops:
            raise ValueError(
                f"Missing {direction} IO capacity for {candidate_drive.name}"
            )
        if candidate_ops > 0:
            ratios.append(current_ops / candidate_ops)
        current_bandwidth = current_ops * current_drive.block_size_kib * 1024
        candidate_bandwidth = candidate_ops * candidate_drive.block_size_kib * 1024
        if observed:
            if current_drive.throughput is not None:
                current_bandwidth = current_drive.throughput * 1024**2
            if candidate_drive.throughput is not None:
                candidate_bandwidth = candidate_drive.throughput * 1024**2
        if candidate_bandwidth <= 0 < current_bandwidth:
            raise ValueError(f"Missing bandwidth capacity for {candidate_drive.name}")
        if candidate_bandwidth > 0:
            ratios.append(current_bandwidth / candidate_bandwidth)
    return max(ratios, default=0)


def disk_io_requirement(
    instance: Instance,
    desires: CapacityDesires,
    observation: Optional[EVCacheDiskIO],
    current: Optional[CurrentZoneClusterCapacity] = None,
) -> Tuple[int, Dict[str, Any]]:
    assert instance.drive is not None
    source = "observed" if observation is not None else "requests"
    if observation is None:
        query = desires.query_pattern
        read_bytes = (
            query.estimated_read_per_second.mid
            * query.estimated_mean_read_size_bytes.mid
        )
        write_bytes = (
            query.estimated_write_per_second.mid
            * query.estimated_mean_write_size_bytes.mid
        )
        block_bytes = instance.drive.block_size_kib * 1024
        observation = EVCacheDiskIO(
            read_bytes_per_second=read_bytes,
            write_bytes_per_second=write_bytes,
            read_ops_per_second=read_bytes / block_bytes,
            write_ops_per_second=write_bytes / block_bytes,
        )
    # A dedicated IO policy replaces the generic compute margin. Matching both
    # through the generic fallback would multiply them.
    fallbacks: Dict[str, List[str]] = (
        {}
        if any(
            BufferComponent.disk_iops in buffer.components
            for buffer in desires.buffers.desired.values()
        )
        else {BufferComponent.disk_iops: [BufferComponent.compute]}
    )
    margin = buffer_for_components(
        desires.buffers, [BufferComponent.disk_iops], component_fallbacks=fallbacks
    )
    policy = DerivedBuffers.for_components(
        desires.buffers.derived,
        [BufferComponent.disk_iops],
        component_fallbacks={BufferComponent.disk_iops: [BufferComponent.compute]},
    )

    def node_load(candidate_drive: Drive) -> float:
        if source == "observed":
            return observation.node_load(candidate_drive)
        # Keep the request fallback's historical read/write capacity mix.
        query = desires.query_pattern
        total_requests = (
            query.estimated_read_per_second.mid + query.estimated_write_per_second.mid
        )
        read_share = (
            query.estimated_read_per_second.mid / total_requests
            if total_requests
            else 0
        )
        capacity = (
            (
                read_share * (candidate_drive.read_io_per_s or 0)
                + (1 - read_share) * (candidate_drive.write_io_per_s or 0)
            )
            * candidate_drive.block_size_kib
            * 1024
        )
        demand = observation.read_bytes_per_second + observation.write_bytes_per_second
        if capacity <= 0 < demand:
            raise ValueError(f"Missing IO capacity for {candidate_drive.name}")
        return demand / capacity if capacity > 0 else 0

    load = node_load(instance.drive)
    existing_capacity = 0.0
    if policy.floor is not None or policy.ceiling is not None:
        if current is None:
            raise ValueError(
                "Disk IO capacity bounds require a current EVCache cluster"
            )
        current_instance = RequirementFromCurrentCapacity(
            current_capacity=current, buffers=desires.buffers
        ).current_instance
        current_drive = current.cluster_drive or current_instance.drive
        if (
            current.cluster_drive is not None
            and current_instance.drive is not None
            and current.cluster_drive.drive_type == current_instance.drive.drive_type
        ):
            # Size-only current observations must not erase known hardware
            # capabilities. Preserve explicit non-null drive overrides.
            current_drive = current_instance.drive.model_copy(
                update=current.cluster_drive.model_dump(
                    exclude_unset=True, exclude_none=True
                )
            )
        if current_drive is None:
            raise ValueError(
                "Disk IO capacity bounds require current local-drive capacity"
            )
        current_load = node_load(current_drive)
        if current_load > 0:
            # Express deployed capacity in candidate-node equivalents, using
            # the same observed read/write mix on both shapes.
            existing_capacity = current.cluster_instance_count.mid * load / current_load
        else:
            # Zero traffic has no read/write mix. Preserve capacity in every
            # direction for a floor; do not infer a mix from absent traffic.
            existing_capacity = current.cluster_instance_count.mid * (
                _zero_demand_capacity_ratio(
                    current_drive, instance.drive, source == "observed"
                )
            )
    count = ceil_ignoring_float_error(
        policy.calculate_requirement(
            current_usage=load,
            existing_capacity=existing_capacity,
            desired_buffer_ratio=margin.ratio,
        )
    )
    return count, {
        "source": source,
        "observation": observation.model_dump(),
        "buffer_ratio": margin.ratio,
        "scale": policy.scale,
        "floor": policy.floor,
        "ceiling": policy.ceiling,
        "required_nodes": count,
        "throughput_capacity_source": (
            "profile_throughput"
            if source == "observed" and instance.drive.throughput is not None
            else "iops_times_block_size"
        ),
    }

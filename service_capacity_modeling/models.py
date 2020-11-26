from enum import Enum
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import TypeVar

from pydantic import BaseModel


###############################################################################
#              Models (structs) for how we describe hardware                  #
###############################################################################


class Drive(BaseModel):
    """Represents a cloud drive e.g. EBS

    This model is generic to any cloud
    """

    name: str
    size_gib: int = 0
    read_io_per_s: Optional[int]
    write_io_per_s: Optional[int]
    # If this drive has single tenant IO capacity, for example a single
    # physical drive versus a virtualised drive
    single_tenant: bool = True

    annual_cost_per_gib: float = 0
    annual_cost_per_read_io: float = 0
    annual_cost_per_write_io: float = 0

    avg_read_latency_ms: float = 1
    avg_write_latency_ms: float = 1
    p99_read_latency_ms: float = 5
    p99_write_latency_ms: float = 5

    @property
    def annual_cost(self):
        size = self.size_gib or 0
        r_ios = self.read_io_per_s or 0
        w_ios = self.write_io_per_s or 0

        return (
            size * self.annual_cost_per_gib
            + r_ios * self.annual_cost_per_read_io
            + w_ios * self.annual_cost_per_write_io
        )


class Instance(BaseModel):
    """Represents a cloud instance aka Hardware Shape

    This model is generic to any cloud.
    """

    name: str
    cpu: int
    cpu_ghz: float
    ram_gib: float
    net_mbps: float
    drive: Optional[Drive]
    annual_cost: float = 0

    family_separator: str = "."

    @property
    def family(self):
        return self.name.split(self.family_separator)[0]


class Hardware(BaseModel):
    """Represents a hardware deployment

    In EC2 this maps to:
        instances: instance type -> Instance(cpu, mem, cost, etc...)
        drives: ebs type -> Drive(cost per _GiB year_, etc...)
    """

    # Per instance shape information e.g. cpu, ram, cpu etc ...
    instances: Dict[str, Instance]
    # Per drive type information and cost
    drives: Dict[str, Drive]


class GlobalHardware(BaseModel):
    """Represents all possible hardware shapes in all regions

    In EC2 this maps to:
        region -> region
    """

    # Per region hardware shapes
    regions: Dict[str, Hardware]


class InstancePricing(BaseModel):
    annual_cost: float = 0


class DrivePricing(BaseModel):
    annual_cost_per_gib: float = 0
    annual_cost_per_read_io: float = 0
    annual_cost_per_write_io: float = 0


class HardwarePricing(BaseModel):
    instances: Dict[str, InstancePricing]
    drives: Dict[str, DrivePricing]


class Pricing(BaseModel):
    regions: Dict[str, HardwarePricing]


###############################################################################
#               Models (structs) for how we plan capacity                     #
###############################################################################


class AccessPattern(str, Enum):
    latency = "latency"
    throughput = "throughput"


Numeric = TypeVar("Numeric", float, int)


class Interval(BaseModel):
    low: float
    mid: float
    high: float
    # How confident are we of this interval
    confidence: float = 1.0

    minimum_value: Optional[float] = None
    maximum_value: Optional[float] = None

    @property
    def minimum(self):
        if self.minimum_value is None:
            return self.low / 2

        return self.minimum_value

    @property
    def maximum(self):
        if self.maximum_value is None:
            return self.high * 2
        return self.maximum_value


def certain_int(x: int) -> Interval:
    return Interval(low=x, mid=x, high=x, confidence=1.0)


def certain_float(x: float) -> Interval:
    return Interval(low=x, mid=x, high=x, confidence=1.0)


class QueryPattern(BaseModel):
    # Will the service primarily be accessed in a latency sensitive mode
    # (aka we care about P99) or throughput (we care about averages)
    access_pattern: AccessPattern = AccessPattern.latency

    # A main input, how many requests per second will we handle
    # We assume this is the mean of a range of possible outcomes
    estimated_read_per_second: Interval = certain_int(100)
    estimated_write_per_second: Interval = certain_int(10)

    # A main input, how much _on cpu_ time per operation do you take.
    # This depends heavily on workload, but this is a generally ok default
    # For a Java app (C or C++ will generally be about 10x better,
    # python 2-4x slower, etc...)
    estimated_mean_read_latency_ms: Interval = certain_float(1)
    estimated_mean_write_latency_ms: Interval = certain_float(1)

    # For stateful services the amount of data accessed per
    # read and write impacts disk and network provisioniong
    # For stateless services it mostly just impacts memory and network
    estimated_mean_read_size_bytes: Interval = certain_int(512)
    estimated_mean_write_size_bytes: Interval = certain_int(128)

    # The latencies at which oncall engineers get involved. We want
    # to provision such that we don't involve oncall
    read_latency_slo_p50_ms: float = 5
    read_latency_slo_p99_ms: float = 100
    write_latency_slo_p50_ms: float = 5
    write_latency_slo_p99_ms: float = 100


class DataShape(BaseModel):
    estimated_state_size_gb: Interval = certain_int(0)
    estimated_working_set_percent: Interval = certain_float(0.1)


class CapacityDesires(BaseModel):
    # How critical is this cluster, impacts how much "extra" we provision
    # 0 = Critical to the product            (Product does not function)
    # 1 = Important to product with fallback (User experience degraded)
    # 2 = Care about it but don't wake up    (Internal apps)
    # 3 = Do not care                        (Testing)
    service_tier: int = 1

    # How will the service be queried
    query_pattern: QueryPattern = QueryPattern()

    # What will the state look like
    data_shape: DataShape = DataShape()

    # When users are providing latency estimates, what is the typical
    # instance core frequency we are comparing to. Databases use i3s a lot
    # hence this default
    core_reference_ghz: float = 2.3


class CapacityRequirement(BaseModel):
    core_reference_ghz: float
    # Ranges of capacity requirements, typically [10%, mean, 90%]
    cpu_cores: Interval
    mem_gib: Interval = certain_int(0)
    network_mbps: Interval = certain_int(0)
    disk_gib: Interval = certain_int(0)
    context: Dict = dict()


class ClusterCapacity(BaseModel):
    cluster_type: str

    count: int
    instance: Instance
    attached_drives: Sequence[Drive] = ()
    annual_cost: float


# For services that are provisioned by zone (e.g. Cassandra, EVCache)
class ZoneClusterCapacity(ClusterCapacity):
    pass


# For services that are provisioned regionally (e.g. Java services, RDS, etc ..)
class RegionClusterCapacity(ClusterCapacity):
    pass


class Clusters(BaseModel):
    total_annual_cost: Interval
    zonal: Sequence[ZoneClusterCapacity] = list()
    regional: Sequence[RegionClusterCapacity] = list()


class CapacityPlan(BaseModel):
    requirement: CapacityRequirement
    candidate_clusters: Sequence[Clusters]

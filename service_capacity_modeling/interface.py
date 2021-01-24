from decimal import Decimal
from enum import Enum
from typing import Dict
from typing import Optional
from typing import Sequence

import numpy as np
from pydantic import BaseModel

###############################################################################
#              Models (structs) for how we describe intervals                 #
###############################################################################


class IntervalModel(str, Enum):
    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"D({self.value})"

    gamma = "gamma"


class Interval(BaseModel):
    low: float
    mid: float
    high: float
    # How confident are we of this interval
    confidence: float = 1.0
    # How to approximate this interval (e.g. with a gamma distribution)
    model_with: IntervalModel = IntervalModel.gamma
    # If we should allow simulation of this interval, some models might not
    # be able to simulate or some properties might not want to
    allow_simulate: bool = True

    minimum_value: Optional[float] = None
    maximum_value: Optional[float] = None

    @property
    def can_simulate(self):
        return self.confidence <= 0.99 and self.allow_simulate

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

    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class FixedInterval(Interval):
    allow_simulate: bool = False


def certain_int(x: int) -> Interval:
    return Interval(low=x, mid=x, high=x, confidence=1.0)


def certain_float(x: float) -> Interval:
    return Interval(low=x, mid=x, high=x, confidence=1.0)


def interval(samples: Sequence[float], low_p: int = 5, high_p: int = 95) -> Interval:
    p = np.percentile(samples, [0, low_p, 50, high_p, 100], interpolation="nearest")
    conf = (high_p - low_p) / 100
    return Interval(
        low=p[1],
        mid=p[2],
        high=p[3],
        minimum_value=p[0],
        maximum_value=p[4],
        confidence=conf,
    )


def interval_percentile(
    samples: Sequence[float], percentiles: Sequence[int]
) -> Sequence[Interval]:
    p = np.percentile(samples, percentiles, interpolation="nearest")
    return [certain_float(i) for i in p]


###############################################################################
#              Models (structs) for how we describe hardware                  #
###############################################################################


class Drive(BaseModel):
    """Represents a cloud drive e.g. EBS

    This model is generic to any cloud
    """

    name: str
    size_gib: int = 0
    read_io_per_s: Optional[int] = None
    write_io_per_s: Optional[int] = None
    # If this drive has single tenant IO capacity, for example a single
    # physical drive versus a virtualised drive
    single_tenant: bool = True

    annual_cost_per_gib: float = 0
    annual_cost_per_read_io: float = 0
    annual_cost_per_write_io: float = 0

    # These defaults are assuming a cloud SSD like a gp2 volume
    # If you disagree please change them in your hardware description
    read_io_latency_ms: FixedInterval = FixedInterval(
        low=0.8, mid=1, high=2, confidence=0.9
    )
    write_io_latency_ms: FixedInterval = FixedInterval(
        low=0.6, mid=2, high=3, confidence=0.9
    )

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


class Service(BaseModel):
    """Represents a cloud service, such as a blob store (S3) or
    managed service such as DynamoDB or RDS.

    This model is generic to any cloud.
    """

    name: str
    size_gib: int = 0

    annual_cost_per_gib: float = 0
    annual_cost_per_read_io: float = 0
    annual_cost_per_write_io: float = 0

    # These defaults assume a cloud blob storage like S3
    read_io_latency_ms: FixedInterval = FixedInterval(
        low=1, mid=5, high=50, confidence=0.9
    )
    write_io_latency_ms: FixedInterval = FixedInterval(
        low=1, mid=10, high=50, confidence=0.9
    )


class RegionContext(BaseModel):
    services: Dict[str, Service] = {}
    zones_in_region: int = 3


class Hardware(BaseModel):
    """Represents a hardware deployment

    In EC2 this maps to:
        instances: instance type -> Instance(cpu, mem, cost, etc...)
        drives: ebs type -> Drive(cost per _GiB year_, etc...)
        services: service type -> Service(name, params, cost, etc ...)
    """

    # How many zones of compute exist in this region of compute
    zones_in_region: int = 3
    # Per instance shape information e.g. cpu, ram, cpu etc ...
    instances: Dict[str, Instance]
    # Per drive type information and cost
    drives: Dict[str, Drive]
    # Per service information and cost
    services: Dict[str, Service]


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


class ServicePricing(BaseModel):
    annual_cost_per_gib: float = 0
    annual_cost_per_read_io: float = 0
    annual_cost_per_write_io: float = 0


class HardwarePricing(BaseModel):
    instances: Dict[str, InstancePricing]
    drives: Dict[str, DrivePricing]
    services: Dict[str, ServicePricing]
    zones_in_region: int = 3


class Pricing(BaseModel):
    regions: Dict[str, HardwarePricing]


###############################################################################
#               Models (structs) for how we plan capacity                     #
###############################################################################


class AccessPattern(str, Enum):
    latency = "latency"
    throughput = "throughput"


AVG_ITEM_SIZE_BYTES: int = 1024


class QueryPattern(BaseModel):
    # Will the service primarily be accessed in a latency sensitive mode
    # (aka we care about P99) or throughput (we care about averages)
    access_pattern: AccessPattern = AccessPattern.latency

    # A main input, how many requests per second will we handle
    # We assume this is the mean of a range of possible outcomes
    estimated_read_per_second: Interval = certain_int(0)
    estimated_write_per_second: Interval = certain_int(0)

    # A main input, how much _on cpu_ time per operation do you take.
    # This depends heavily on workload, but this is a generally ok default
    # For a Java app (C or C++ will generally be about 10x better,
    # python 2-4x slower, etc...)
    estimated_mean_read_latency_ms: Interval = certain_float(1)
    estimated_mean_write_latency_ms: Interval = certain_float(1)

    # For stateful services the amount of data accessed per
    # read and write impacts disk and network provisioniong
    # For stateless services it mostly just impacts memory and network
    estimated_mean_read_size_bytes: Interval = certain_int(AVG_ITEM_SIZE_BYTES)
    estimated_mean_write_size_bytes: Interval = certain_int(AVG_ITEM_SIZE_BYTES // 2)

    # The latencies at which oncall engineers get involved. We want
    # to provision such that we don't involve oncall
    # Note that these summary statistics will be used to create reasonable
    # distribution approximations of these operations (yielding p25, p99, etc)
    read_latency_slo_ms: FixedInterval = FixedInterval(
        low=0.4, mid=4, high=10, confidence=0.98
    )
    write_latency_slo_ms: FixedInterval = FixedInterval(
        low=0.4, mid=4, high=10, confidence=0.98
    )


class DataShape(BaseModel):
    estimated_state_size_gib: Interval = certain_int(0)
    estimated_state_item_count: Optional[Interval] = None
    estimated_working_set_percent: Optional[Interval] = None

    # How durable does this dataset need to be. We want to provision
    # sufficient replication and backups of data to achieve the target
    # durability SLO so we don't lose our customer's data. Note that
    # This is measured in "nines" per year
    durability_slo_nines: FixedInterval = FixedInterval(
        low=3, mid=4, high=5, confidence=0.98
    )


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

    def merge_with(self, defaults: "CapacityDesires") -> "CapacityDesires":
        desires_dict = self.dict(exclude_unset=True)
        default_dict = defaults.dict(exclude_unset=True)

        default_dict.get("query_pattern", {}).update(
            desires_dict.pop("query_pattern", {})
        )
        default_dict.get("data_shape", {}).update(desires_dict.pop("data_shape", {}))
        default_dict.update(desires_dict)

        return CapacityDesires(**default_dict)


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


class ServiceCapacity(BaseModel):
    service_type: str
    annual_cost: float
    # Often while provisioning cloud services we need to represent
    # parameters to the cloud APIs, use this to inject those from models
    service_params: Dict = {}


# For services that are provisioned by zone (e.g. Cassandra, EVCache)
class ZoneClusterCapacity(ClusterCapacity):
    pass


# For services that are provisioned regionally (e.g. Java services, RDS, etc ..)
class RegionClusterCapacity(ClusterCapacity):
    pass


class Clusters(BaseModel):
    total_annual_cost: Decimal = Decimal(0)
    zonal: Sequence[ZoneClusterCapacity] = list()
    regional: Sequence[RegionClusterCapacity] = list()
    services: Sequence[ServiceCapacity] = list()


class CapacityPlan(BaseModel):
    requirement: CapacityRequirement
    candidate_clusters: Clusters


class UncertainCapacityPlan(BaseModel):
    requirement: CapacityRequirement
    least_regret: Sequence[CapacityPlan]
    mean: Sequence[CapacityPlan]
    percentiles: Dict[int, Sequence[CapacityPlan]]


class CapacityRegretParameters(BaseModel):
    over_provision_cost: float = 1
    under_provision_cost: float = 1.25

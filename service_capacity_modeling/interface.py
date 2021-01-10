import random
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
        low=0.4, mid=0.8, high=2, confidence=0.98
    )
    write_io_latency_ms: FixedInterval = FixedInterval(
        low=0.5, mid=1, high=2.4, confidence=0.98
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


AVG_ITEM_SIZE_BYTES: int = 1024


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
    estimated_mean_read_size_bytes: Interval = certain_int(AVG_ITEM_SIZE_BYTES)
    estimated_mean_write_size_bytes: Interval = certain_int(AVG_ITEM_SIZE_BYTES / 2)

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


class WorkingSetEstimator:
    def __init__(self):
        self._cache = {}

    def working_set_percent(
        self,
        # latency distributions of the read SLOs versus the drives
        # expressed as scipy rv_continuous objects
        drive_read_latency_dist,
        read_slo_latency_dist,
        # what is our target percentile for hitting disk
        # Note that lower will decrease the amount we hit disk
        target_percentile: float = 0.10,
    ) -> Interval:
        # random cache eviction
        if len(self._cache) >= 100:
            self._cache.pop(random.choice(self._cache.keys()))

        cache_key = (
            id(drive_read_latency_dist),
            id(read_slo_latency_dist),
            target_percentile,
        )
        if cache_key in self._cache:
            result = self._cache[cache_key]
        else:
            # The inverse CDF, basically what percentile do we want to target
            # to be all on disk.
            target_latency = read_slo_latency_dist.ppf(target_percentile)

            # What percent of disk reads will fall below this latency SLO
            result = certain_float(drive_read_latency_dist.sf(target_latency))
            self._cache[cache_key] = result
        return result


_working_set_estimator = WorkingSetEstimator()


class DataShape(BaseModel):
    estimated_state_size_gib: Interval = certain_int(0)
    estimated_state_item_count: Optional[Interval] = None
    estimated_working_set_percent: Optional[Interval] = None

    def item_count(self) -> Interval:
        if self.estimated_state_item_count is not None:
            return self.estimated_state_item_count

        return certain_int(
            (self.estimated_state_size_gib * 1024 * 1024 * 1024) // AVG_ITEM_SIZE_BYTES
        )

    def working_set_percent(
        self,
        # latency distributions of the read SLOs versus the drives
        # expressed as scipy rv_continuous objects
        drive_read_latency_dist,
        read_slo_latency_dist,
        # what is our target percentile for hitting disk
        # Note that lower will decrease the amount we hit disk
        target_percentile: float = 0.10,
    ) -> Interval:
        if self.estimated_working_set_percent is not None:
            return self.estimated_working_set_percent

        return _working_set_estimator.working_set_percent(
            drive_read_latency_dist=drive_read_latency_dist,
            read_slo_latency_dist=read_slo_latency_dist,
            target_percentile=target_percentile,
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
    candidate_clusters: Clusters


class UncertainCapacityPlan(BaseModel):
    requirement: CapacityRequirement
    least_regret: Optional[CapacityPlan]
    mean: Sequence[CapacityPlan]
    percentiles: Dict[int, Sequence[CapacityPlan]]


class CapacityRegretParameters(BaseModel):
    over_provision_cost: float = 1
    under_provision_cost: float = 1.5

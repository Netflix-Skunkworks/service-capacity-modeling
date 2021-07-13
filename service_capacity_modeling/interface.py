from decimal import Decimal
from enum import Enum
from functools import lru_cache
from typing import Dict
from typing import Optional
from typing import Sequence

import numpy as np
from pydantic import BaseModel
from pydantic import Field

###############################################################################
#              Models (structs) for how we describe intervals                 #
###############################################################################


class IntervalModel(str, Enum):
    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"D({self.value})"

    gamma = "gamma"
    beta = "beta"


class Interval(BaseModel):
    low: float
    mid: float
    high: float
    # How confident are we of this interval
    confidence: float = 1.0
    # How to approximate this interval (e.g. with a beta distribution)
    model_with: IntervalModel = IntervalModel.beta
    # If we should allow simulation of this interval, some models might not
    # be able to simulate or some properties might not want to
    allow_simulate: bool = True

    minimum_value: Optional[float] = None
    maximum_value: Optional[float] = None

    class Config:
        allow_mutation = False
        frozen = True

    @property
    def can_simulate(self):
        return self.confidence <= 0.99 and self.allow_simulate

    @property
    def minimum(self):
        if self.minimum_value is None:
            if self.confidence == 1.0:
                return self.low * 0.999
            return self.low / 2

        return self.minimum_value

    @property
    def maximum(self):
        if self.maximum_value is None:
            if self.confidence == 1.0:
                return self.high * 1.001
            return self.high * 2
        return self.maximum_value

    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class FixedInterval(Interval):
    allow_simulate: bool = False


@lru_cache(2048)
def certain_int(x: int) -> Interval:
    return Interval(low=x, mid=x, high=x, confidence=1.0)


@lru_cache(2048)
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


class Lifecycle(str, Enum):
    """Represents the lifecycle of hardware from initial preview
    to end-of-life.

    For example a particular shape of hardware might be released under
    preview and adventurous models may wish to use those, and then
    more risk-averse workloads may wait for stability.

    By default models are shown beta and stable shapes to pick from
    """

    alpha = "alpha"
    beta = "beta"
    stable = "stable"
    deprecated = "deprecated"
    end_of_life = "end-of-life"


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
    lifecycle: Lifecycle = Lifecycle.stable

    family_separator: str = "."

    @property
    def family(self):
        return self.name.split(self.family_separator)[0]

    @property
    def size(self):
        return self.name.split(self.family_separator)[1]


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
    lifecycle: Optional[Lifecycle] = None


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


class AccessConsistency(str, Enum):
    """See https://jepsen.io/consistency

    Generally speaking consistency is expensive, so models need to know what
    kind of consistency will be required in order to estimate CPU usage
    within a factor of 4-5x correctly.
    """

    # You cannot read writes ever
    never = "never"

    #
    # Single item consistency (most services)
    #

    # Best Effort: we might lose writes or reads might be stale or missing.
    #              Most caches offer this level of consistency.
    # Eventual: We will eventually reflect the latest successful write but
    #           there is some (often large) time bound on that eventuality.
    # Read-Your-Writes: The first "consistent" offering.
    best_effort = "best-effort"
    eventual = "eventual"
    read_your_writes = "read-your-writes"
    # Fully lineralizable, writes and reads
    linearizable = "linearizable"
    # Writes are linerizable but stale reads are possible (e.g. ZK)
    linearizable_stale = "linearizable-stale"

    #
    # Multiple item consistency (often "transactional" or "acid" services)
    #

    # All operations are serializable.
    # (e.g. CRDB in default settings)
    serializable = "serializable"
    # Writes are serializable but stale reads are possible
    # (e.g. CRDB with stale reads enabled, MySQL with read replicas, etc ...)
    serializable_stale = "serializable-stale"


AVG_ITEM_SIZE_BYTES: int = 1024


class Consistency(BaseModel):
    target_consistency: Optional[AccessConsistency] = Field(
        None,
        title="Consistency requirement on access",
        description=(
            "Stronger consistency access is generally more expensive."
            " The words used here to describe consistency attempt to "
            " align with the Jepsen tree of multi/single object "
            " consistency models: https://jepsen.io/consistency"
        ),
    )
    staleness_slo_sec: FixedInterval = Field(
        FixedInterval(low=0, mid=10, high=60),
        title="When stale reads are permitted what is the staleness requirement",
        description=(
            "Eventual consistency (aka stale reads) is usually bounded by some"
            " amount of time. Applications can use this to try to enforce when "
            " a write is available for reads"
        ),
    )


class GlobalConsistency(BaseModel):
    same_region: Consistency = Consistency(
        target_consistency=None,
        staleness_slo_sec=FixedInterval(low=0, mid=0.1, high=1),
    )
    cross_region: Consistency = Consistency(
        target_consistency=None,
        staleness_slo_sec=FixedInterval(low=10, mid=60, high=600),
    )


class QueryPattern(BaseModel):
    # Will the service primarily be accessed in a latency sensitive mode
    # (aka we care about P99) or throughput (we care about averages)
    access_pattern: AccessPattern = AccessPattern.latency
    access_consistency: GlobalConsistency = GlobalConsistency()

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
    estimated_working_set_percent: Optional[Interval] = Field(
        None,
        title="Estimated working set percentage",
        description=(
            "The estimated percentage of data that will be accessed frequently"
            " and therefore must be kept hot in memory (e.g. 0.10). Note that "
            " models will generally estimate this from the latency SLO and "
            "latency model of the drives being attached"
        ),
    )

    # How compressible is this dataset. Note that databases might offer
    # better or worse compression strategies that will impact this
    #   Note that the ratio here is the forward ratio, e.g.
    #   A ratio of 2 means 2:1 compression (0.5 on disk size)
    #   A ratio of 5 means 5:1 compression (0.2 on disk size)
    estimated_compression_ratio: Interval = certain_float(1)

    # How much fixed memory must be provisioned per instance for the
    # application (e.g. for process heap memory)
    reserved_instance_app_mem_gib: int = 2

    # How much fixed memory must be provisioned per instance for the
    # system (e.g. for kernel and other system processes)
    reserved_instance_system_mem_gib: int = 1

    # How durable does this dataset need to be. We want to provision
    # sufficient replication and backups of data to achieve the target
    # durability SLO so we don't lose our customer's data. Note that
    # This is measured in orders of magnitude. So
    #   1000   = 1 - (1/1000) = 0.999
    #   10000  = 1 - (1/10000) = 0.9999
    durability_slo_order: FixedInterval = FixedInterval(
        low=1000, mid=10000, high=100000, confidence=0.98
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
    requirement_type: str

    core_reference_ghz: float
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
    # When provisioning services we might need to signal they
    # should have certain configuration, for example flags that
    # affect durability shut off
    cluster_params: Dict = {}


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


class Requirements(BaseModel):
    zonal: Sequence[CapacityRequirement] = list()
    regional: Sequence[CapacityRequirement] = list()

    # Commonly a model regrets "spend", lack of "disk" space and sometimes
    # lack of "mem"ory. Default options are ["cost", "disk", "mem"]
    regrets: Sequence[str] = ("spend", "disk")

    # Used by models to have custom regret components
    # pylint: disable=unused-argument
    @staticmethod
    def regret(
        name: str, optimal_plan: "CapacityPlan", proposed_plan: "CapacityPlan"
    ) -> float:
        return 0.0


class Clusters(BaseModel):
    total_annual_cost: Decimal = Decimal(0)
    zonal: Sequence[ZoneClusterCapacity] = list()
    regional: Sequence[RegionClusterCapacity] = list()
    services: Sequence[ServiceCapacity] = list()


class CapacityPlan(BaseModel):
    requirements: Requirements
    candidate_clusters: Clusters


class UncertainCapacityPlan(BaseModel):
    requirements: Requirements
    least_regret: Sequence[CapacityPlan]
    mean: Sequence[CapacityPlan]
    percentiles: Dict[int, Sequence[CapacityPlan]]


# Parameters to cost functions of the form
# let y = the optimal value
# let x = the proposed value
# let cost = (x - y) ^ exponent
class Regret(BaseModel):
    over_provision_cost: float = 0
    under_provision_cost: float = 0
    exponent: float = 1.0


class CapacityRegretParameters(BaseModel):
    # How much do we regret spending too much or too little money
    spend: Regret = Regret(
        over_provision_cost=1, under_provision_cost=1.25, exponent=1.2
    )

    # For every GiB we are underprovisioned by default cost $1 / year / GiB
    disk: Regret = Regret(
        over_provision_cost=0, under_provision_cost=1.1, exponent=1.05
    )

    # For every GiB we are underprovisioned on memory (for datastores
    # storing data in RAM), regret under_provisioning slightly more than disk
    mem: Regret = Regret(over_provision_cost=0, under_provision_cost=1.5, exponent=1.1)

    # Any additional metric we want to regret from the models
    # just have to be returned in the requirement
    extra: Dict[str, Regret] = {}

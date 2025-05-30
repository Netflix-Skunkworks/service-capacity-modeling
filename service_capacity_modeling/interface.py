# pylint: disable=too-many-lines
from __future__ import annotations

import re
import sys
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from functools import lru_cache
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
from pydantic import BaseModel
from pydantic import computed_field
from pydantic import ConfigDict
from pydantic import Field

GIB_IN_BYTES = 1024 * 1024 * 1024
MIB_IN_BYTES = 1024 * 1024
MEGABIT_IN_BYTES = (1000 * 1000) / 8


class ExcludeUnsetModel(BaseModel):
    def model_dump(self, *args, **kwargs):
        if "exclude_unset" not in kwargs:
            kwargs["exclude_unset"] = True
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args, **kwargs):
        if "exclude_unset" not in kwargs:
            kwargs["exclude_unset"] = True
        return super().model_dump_json(*args, **kwargs)


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


class Interval(ExcludeUnsetModel):
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
    model_config = ConfigDict(frozen=True, protected_namespaces=())

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

    def scale(self, factor: float) -> Interval:
        minimum_value = (
            self.minimum * factor if self.minimum_value is not None else None
        )
        maximum_value = (
            self.maximum * factor if self.maximum_value is not None else None
        )
        return Interval(
            low=self.low * factor,
            mid=self.mid * factor,
            high=self.high * factor,
            confidence=self.confidence,
            model_with=self.model_with,
            allow_simulate=self.allow_simulate,
            minimum_value=minimum_value,
            maximum_value=maximum_value,
        )

    def offset(self, delta: float) -> Interval:
        minimum_value = self.minimum + delta if self.minimum_value is not None else None
        maximum_value = self.maximum + delta if self.maximum_value is not None else None
        return Interval(
            low=self.low + delta,
            mid=self.mid + delta,
            high=self.high + delta,
            confidence=self.confidence,
            model_with=self.model_with,
            allow_simulate=self.allow_simulate,
            minimum_value=minimum_value,
            maximum_value=maximum_value,
        )


class FixedInterval(Interval):
    allow_simulate: bool = False


@lru_cache(2048)
def certain_int(x: int) -> Interval:
    return Interval(low=x, mid=x, high=x, confidence=1.0)


@lru_cache(2048)
def certain_float(x: float) -> Interval:
    return Interval(low=x, mid=x, high=x, confidence=1.0)


@lru_cache(2048)
def fixed_float(x: float) -> FixedInterval:
    return FixedInterval(low=x, mid=x, high=x, confidence=1.0)


def interval(samples: Sequence[float], low_p: int = 5, high_p: int = 95) -> Interval:
    p = np.percentile(a=samples, q=[0, low_p, 50, high_p, 100])
    conf = (high_p - low_p) / 100
    return Interval(
        low=p[1],
        mid=p[2],
        high=p[3],
        minimum_value=p[0],
        maximum_value=p[4],
        confidence=conf,
    )


def normalized_aws_size(name: str) -> Fraction:
    """Normalizes an AWS shape to a fractional xlarge unit"""
    _, size = name.split(".")
    numeric = re.findall(r"\d+", size)
    if numeric:
        assert len(numeric) == 1
        return Fraction(float(numeric[0]))
    return {
        "small": Fraction(1, 8),
        "medium": Fraction(1, 4),
        "large": Fraction(1, 2),
        "xlarge": Fraction(1),
        # Is this always true?
        "metal": Fraction(48, 1),
    }[size]


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


class DriveType(str, Enum):
    """Represents the type of drive"""

    local_ssd = "local-ssd"
    local_hdd = "local-hdd"
    attached_ssd = "attached-ssd"
    attached_hdd = "attached-hhd"


class Drive(ExcludeUnsetModel):
    """Represents a cloud drive e.g. EBS or ephemeral drives

    This model is generic to any cloud
    """

    name: str
    drive_type: DriveType = DriveType.local_ssd
    size_gib: int = 0
    read_io_per_s: Optional[int] = None
    write_io_per_s: Optional[int] = None
    throughput: Optional[int] = None
    # If this drive has single tenant IO capacity, for example a single
    # physical drive versus a virtualised drive
    single_tenant: bool = True
    # If this drive can scale, how large can it scale to
    max_scale_size_gib: int = 0
    # If this drive can scale IO, how large can it scale to
    max_scale_io_per_s: int = 0
    # How large is an "IO" against this device
    block_size_kib: int = 4
    # When sequential how much IO is grouped into a single "IO"
    # https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-io-characteristics.html
    # Some cloud drives can group sequential ops together and DBs take advantage
    group_size_kib: int = 4

    lifecycle: Lifecycle = Lifecycle.stable
    compatible_families: List[str] = []

    annual_cost_per_gib: float = 0
    # Tuples of [max_size, annual cost]
    # [32000, 0.78], ...
    annual_cost_per_read_io: List[Tuple[float, float]] = []
    annual_cost_per_write_io: List[Tuple[float, float]] = []

    # These defaults are assuming a cloud SSD like a gp2 volume
    # If you disagree please change them in your hardware description
    read_io_latency_ms: FixedInterval = FixedInterval(
        low=0.8, mid=1, high=2, confidence=0.9
    )
    write_io_latency_ms: FixedInterval = FixedInterval(
        low=0.6, mid=2, high=3, confidence=0.9
    )

    @property
    def rand_io_size_kib(self) -> int:
        return self.block_size_kib

    @property
    def seq_io_size_kib(self) -> int:
        return max(self.block_size_kib, self.group_size_kib)

    @property
    def max_size_gib(self):
        if self.max_scale_size_gib != 0:
            return self.max_scale_size_gib
        else:
            return self.size_gib

    @property
    def max_io_per_s(self):
        if self.max_scale_io_per_s != 0:
            return self.max_scale_io_per_s
        else:
            return sys.maxsize

    @computed_field(return_type=float)  # type: ignore
    @property
    def annual_cost(self):
        size = self.size_gib or 0
        r_ios = self.read_io_per_s or 0
        w_ios = self.write_io_per_s or 0

        # Time to do income taxes ...
        # Inputs are ranges of io limits and costs for ios in that range
        # [(32000.0, 0.78),
        #  (64000.0, 0.552),
        #  (160000.0, 0.384)]
        r_cost, w_cost, offset = 0.0, 0.0, 0.0
        if self.annual_cost_per_read_io:
            for end, cost in self.annual_cost_per_read_io:
                charge_ios = min(r_ios, end) - offset
                r_cost += charge_ios * cost
                offset += charge_ios
                if offset >= r_ios:
                    break

        offset = 0.0
        if self.annual_cost_per_write_io:
            for end, cost in self.annual_cost_per_write_io:
                charge_ios = min(w_ios, end) - offset
                w_cost += charge_ios * cost
                offset += charge_ios
                if offset >= w_ios:
                    break

        return size * self.annual_cost_per_gib + r_cost + w_cost

    @staticmethod
    def get_managed_drive() -> Drive:
        return Drive(name="managed")


class Platform(str, Enum):
    """Represents the platform of the hardware

    For example a particular hardware type might offer x86_64, arm, or be a managed
    instance type that only works with managed RDBMS like Aurora Postgres.
    """

    # Most Intel and AMD instance types
    amd64 = "amd64"
    # Graviton and other ARM based instance types
    arm64 = "arm64"
    # Special purpose aurora type
    aurora_mysql = "Aurora MySQL"
    # Special purpose aurora type
    aurora_postgres = "Aurora PostgreSQL"


class Instance(ExcludeUnsetModel):
    """Represents a cloud instance aka Hardware Shape

    This model is generic to any cloud.
    """

    name: str
    cpu: int = Field(
        title="CPU threads (vCPUs or effective cores)",
        description=(
            "The number of CPU threads this instance has, typically a SMT thread "
            "but if equal to Instance.cpu_cores, actual cores."
        ),
    )
    cpu_cores: Optional[int] = Field(
        default=None,
        title="If known the number of physical cores present on the machine",
        description=(
            "Most Instance.cpu are really threads, meaning they are not truly "
            "parallel. Some Instance.cpu are full fat cores, in which case this "
            "will be set and be equal to the cpu count. Use the cores getter in code"
        ),
    )
    cpu_ghz: float = Field(
        title="The clock frequency of these cores",
        description=(
            "How fast the base clock of the cores are. We assume only baseline"
            " all-core sustained turbo for the purpose of capacity planning."
        ),
    )
    cpu_ipc_scale: float = Field(
        default=1.0,
        title="Instruction per clock scale: core speed is multiplied by this",
        description=(
            "Not all cores or ghz are created equal, and if we want to represent that "
            "a CPU is more productive per unit frequency, this fractional scaling "
            "influences core normalization logic. Note that in the longer term we "
            "expect to replace this with a standard measure of average performance."
        ),
    )
    ram_gib: float
    net_mbps: float
    drive: Optional[Drive] = None
    annual_cost: float = 0
    lifecycle: Lifecycle = Lifecycle.stable
    # Typically hardware has a single platform, but sometimes they can act in multiple
    platforms: List[Platform] = [Platform.amd64]

    family_separator: str = "."

    @property
    def family(self):
        return self.name.rsplit(self.family_separator, 1)[0]

    @property
    def size(self):
        return self.name.rsplit(self.family_separator, 1)[1]

    @property
    def cores(self):
        if self.cpu_cores is not None:
            return self.cpu_cores
        return self.cpu // 2

    @staticmethod
    def get_managed_instance() -> Instance:
        return Instance(
            name="managed.0", cpu=0, cpu_ghz=0, ram_gib=0, net_mbps=0, drive=None
        )

    def merge_with(self, overrides: "Instance") -> "Instance":
        self_dict = self.model_dump()
        other_dict = overrides.model_dump(exclude_unset=True)

        for (k, v) in other_dict.items():
            # TODO we need a deep merge on drive (recursive merge)
            if k in ("platforms",):
                # Unique merge platforms
                merged_platforms = list(self_dict.get("platforms", []))
                merged_platforms = merged_platforms + [
                    i
                    for i in other_dict.get("platforms", [])
                    if i not in merged_platforms
                ]
                self_dict["platforms"] = merged_platforms
            else:
                self_dict[k] = v
        return Instance(**self_dict)


default_reference_shape = Instance(
    name="default_reference_shape",
    cpu=4,
    # Much of our benchmarking was carried out in the 5th gen i2 shape, can
    # adjust this up once we go through and fix all the latency distributions
    cpu_ghz=2.3,
    cpu_ipc_scale=1.0,
    ram_gib=8.0,
    net_mbps=1000,
)


class Service(ExcludeUnsetModel):
    """Represents a cloud service, such as a blob store (S3) or
    managed service such as DynamoDB or RDS.

    This model is generic to any cloud.
    """

    name: str
    size_gib: int = 0

    annual_cost_per_gib: Union[float, List[Tuple[float, float]]] = 0
    annual_cost_per_read_io: float = 0
    annual_cost_per_write_io: float = 0
    annual_cost_per_core: float = 0

    # These defaults assume a cloud blob storage like S3
    read_io_latency_ms: FixedInterval = FixedInterval(
        low=1, mid=5, high=50, confidence=0.9
    )
    write_io_latency_ms: FixedInterval = FixedInterval(
        low=1, mid=10, high=50, confidence=0.9
    )

    def annual_cost_gib(self, data_gib: float = 0):
        if isinstance(self.annual_cost_per_gib, float):
            return self.annual_cost_per_gib * data_gib
        else:
            _annual_data = data_gib
            transfer_costs = list(self.annual_cost_per_gib)
            annual_cost = 0.0
            for transfer_cost in transfer_costs:
                if _annual_data <= 0:
                    break
                if transfer_cost[0] > 0:
                    annual_cost += (
                        min(_annual_data, transfer_cost[0]) * transfer_cost[1]
                    )
                    _annual_data -= transfer_cost[0]
                else:
                    # final remaining data transfer cost
                    annual_cost += _annual_data * transfer_cost[1]
        return annual_cost


class RegionContext(ExcludeUnsetModel):
    services: Dict[str, Service] = {}
    zones_in_region: int = 3
    num_regions: int = 3


class Hardware(ExcludeUnsetModel):
    """Represents a hardware deployment

    In EC2 this maps to:
        instances: instance type -> Instance(cpu, mem, cost, etc...)
        drives: ebs type -> Drive(cost per _GiB year_, etc...)
        services: service type -> Service(name, params, cost, etc ...)
    """

    # How many zones of compute exist in this region of compute
    zones_in_region: int = 3
    # Per instance shape information e.g. cpu, ram, cpu etc ...
    instances: Dict[str, Instance] = {}
    # Per drive type information and cost
    drives: Dict[str, Drive] = {}
    # Per service information and cost
    services: Dict[str, Service] = {}


class GlobalHardware(ExcludeUnsetModel):
    """Represents all possible hardware shapes in all regions

    In EC2 this maps to:
        us-east-1 -> Hardware available in us-east-1
        us-east-2 -> Hardware available in us-east-2
        ...
    """

    # Per region hardware shapes
    regions: Dict[str, Hardware]


class InstancePricing(ExcludeUnsetModel):
    annual_cost: float = 0
    lifecycle: Optional[Lifecycle] = None


class DrivePricing(ExcludeUnsetModel):
    annual_cost_per_gib: float = 0
    annual_cost_per_read_io: List[Tuple[float, float]] = []
    annual_cost_per_write_io: List[Tuple[float, float]] = []


class ServicePricing(ExcludeUnsetModel):
    annual_cost_per_gib: Union[float, List[Tuple[float, float]]] = 0
    annual_cost_per_read_io: float = 0
    annual_cost_per_write_io: float = 0
    annual_cost_per_core: float = 0


class HardwarePricing(ExcludeUnsetModel):
    instances: Dict[str, InstancePricing]
    drives: Dict[str, DrivePricing]
    services: Dict[str, ServicePricing]
    zones_in_region: int = 3


class Pricing(ExcludeUnsetModel):
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


class Consistency(ExcludeUnsetModel):
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


class GlobalConsistency(ExcludeUnsetModel):
    same_region: Consistency = Consistency(
        target_consistency=None,
        staleness_slo_sec=FixedInterval(low=0, mid=0.1, high=1),
    )
    cross_region: Consistency = Consistency(
        target_consistency=None,
        staleness_slo_sec=FixedInterval(low=10, mid=60, high=600),
    )


class QueryPattern(ExcludeUnsetModel):
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

    # For workloads which have bursts of async work, what is the
    # expected parallelism of those workloads. Note the summation of
    # read and write parallelism will lower bound the number of cores.
    estimated_read_parallelism: Interval = Field(
        certain_int(1),
        title="Estimated per instance parallelism on read operations",
        description=(
            "The estimated amount of parallel work streams on a single "
            "host. For example a read triggers async callbacks that need "
            "to be executed truly in parallel (not just concurrent)."
        ),
    )
    estimated_write_parallelism: Interval = Field(
        certain_int(1),
        title="Estimated per instance parallelism on write operations",
        description=(
            "The estimated amount of parallel work streams on a single "
            "host. For example a write triggers async fanouts that need "
            "to be executed truly in parallel (not just concurrent)."
        ),
    )

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


class DataShape(ExcludeUnsetModel):
    estimated_state_size_gib: Interval = Field(
        certain_int(0),
        title="Estimated amount of state in the system in GiB",
        description=(
            "The estimated amount of state that will be stored."
            " Note that this is an estimate and doesn't need to be exact"
        ),
    )
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
    reserved_instance_app_mem_gib: float = 2

    # How much fixed memory must be provisioned per instance for the
    # system (e.g. for kernel and other system processes)
    reserved_instance_system_mem_gib: float = 1

    # How durable does this dataset need to be. We want to provision
    # sufficient replication and backups of data to achieve the target
    # durability SLO so we don't lose our customer's data. Note that
    # This is measured in orders of magnitude. So
    #   1000   = 1 - (1/1000) = 0.999
    #   10000  = 1 - (1/10000) = 0.9999
    durability_slo_order: FixedInterval = FixedInterval(
        low=1000, mid=10000, high=100000, confidence=0.98
    )


class CurrentClusterCapacity(ExcludeUnsetModel):
    cluster_instance_name: str
    cluster_instance: Optional[Instance] = None
    cluster_drive: Optional[Drive] = None
    cluster_instance_count: Interval
    # The distribution cpu utilization in the cluster.
    cpu_utilization: Interval = certain_float(0.0)
    # The per node distribution of memory used in gib.
    memory_utilization_gib: Interval = certain_float(0.0)
    # The per node distribution of network used in mbps.
    network_utilization_mbps: Interval = certain_float(0.0)
    # The per node distribution of disk used in gib.
    disk_utilization_gib: Interval = certain_float(0.0)


# For services that are provisioned by zone (e.g. Cassandra, EVCache)
class CurrentZoneClusterCapacity(CurrentClusterCapacity):
    pass


# For services that are provisioned regionally (e.g. Java services, RDS, etc ..)
class CurrentRegionClusterCapacity(CurrentClusterCapacity):
    pass


class CurrentClusters(ExcludeUnsetModel):
    zonal: Sequence[CurrentZoneClusterCapacity] = []
    regional: Sequence[CurrentRegionClusterCapacity] = []
    services: Sequence[ServiceCapacity] = []


class BufferComponent(str, Enum):
    """Represents well known buffer components such as compute and storage

    Note that while these are common and defined here for models to share,
    models can have w.e. buffers they want, and this type should not enter
    the Buffers interface itself (should be str).
    """

    # [Query Pattern] a.k.a. "Traffic" related buffers, e.g. CPU and Network
    compute = "compute"
    # [Data Shape]    a.k.a. "Dataset" related buffers, e.g. Disk and Memory
    storage = "storage"

    # Resource specific component
    cpu = "cpu"
    network = "network"
    disk = "disk"
    memory = "memory"


class BufferIntent(str, Enum):
    # Most buffers show "desired" buffer, this is the default
    desired = "desired"
    # ratio on top of existing buffers to ensure exists. Generally combined
    # with a different desired buffer to ensure we don't just scale needlessly
    scale = "scale"
    # Ignore model preferences, just preserve existing buffers
    preserve = "preserve"


class Buffer(ExcludeUnsetModel):
    # The value of the buffer expressed as a ratio over "normal" load e.g. 1.5x
    ratio: float = 1.0
    # What is the intent of this buffer directive, almost always is desired
    intent: BufferIntent = BufferIntent.desired
    # The components of buffer this influences, almost always is "compute" (IPC success)
    components: List[str] = [BufferComponent.compute]
    # If this buffer was made up of other buffers, what contributed
    sources: Dict[str, Buffer] = {}


class Buffers(ExcludeUnsetModel):
    """Typical buffers (headroom) over the requirements to build into the system

    Note that typically callers make buffer choices based on business context for
    example a tier 1 service may request:

    Buffers(
        desired={
            "compute": Buffer(ratio: 1.5),
        }
    )

    And then models layer in their buffers, for example if a workload
    requires 10 CPU cores, but the operator of that workload  likes to build in
    2x buffer for background work (20 cores provisioned), they would express that
    as a model desire default of:

    Buffers(
        desired={
            "background": Buffer(ratio: 2.0, components=[BufferComponent.cpu]),
        }
    )

    In this case when we query the buffer_for_components(components=["cpu"])
    we get 2.0 * 1.5 = 3x because the model has reserved 2x for background work
    and the caller has asked for 1.5x load buffer - we also see both component
    buffers in the Buffer.source field.
    """

    # The default buffer if a specific buffer isn't known
    # Models should prefer to document their precise buffers in desired
    default: Buffer = Buffer(ratio=1.5)
    # Desired compute, storage, cpu, memory, etc... buffers
    desired: Dict[str, Buffer] = {}
    # Derive these buffers from current clusters or model context
    # Buffer.intent MUST be set on these Buffers to something other than "desired":
    #   scale    = ratio on top of existing buffers to ensure. Let the "derived"
    #              buffer multiplied by this ratio "needed". If the "needed" buffer
    #              is greater than desired, the needed buffer is created. If the
    #              "needed" buffer is less than desired, the needed buffer is created.
    #   preserve = ignore desired buffer entirely, just maintain existing buffers
    derived: Dict[str, Buffer] = {}


class CapacityDesires(ExcludeUnsetModel):
    # How critical is this cluster, impacts how much "extra" we provision
    # 0 = Critical to the product            (Product does not function)
    # 1 = Important to product with fallback (User experience degraded)
    # 2 = Care about it but don't wake up    (Internal apps)
    # 3 = Do not care                        (Testing)
    service_tier: int = 1

    # How will the service be queried
    query_pattern: QueryPattern = QueryPattern()

    # What will the state look like that is being queries
    data_shape: DataShape = DataShape()

    # What is the current deployment and utilization of the system
    current_clusters: Optional[CurrentClusters] = None

    # What are the desired buffers (headroom) state, mostly injected by models
    # Note if you pass current_clusters you can express "I need the status quo"
    # by setting the buffers you want to preserve in buffers.derived
    buffers: Buffers = Buffers()

    @property
    def reference_shape(self) -> Instance:
        # TODO: this should use the shape from current clusters if it is there
        return default_reference_shape

    def merge_with(self, defaults: "CapacityDesires") -> "CapacityDesires":
        # Now merge with the models default
        desires_dict = self.model_dump()
        default_dict = defaults.model_dump()

        default_dict.get("query_pattern", {}).update(
            desires_dict.pop("query_pattern", {})
        )
        default_dict.get("data_shape", {}).update(desires_dict.pop("data_shape", {}))

        # Buffers has deep structure we want to deep merge on
        if "buffers" not in default_dict:
            default_dict["buffers"] = {}
        default_buffers = default_dict["buffers"]
        desired_buffers = desires_dict.pop("buffers", {})
        if "default" in desired_buffers:
            default_buffers["default"] = desired_buffers["default"]
        for k, v in desired_buffers.get("desired", {}).items():
            default_buffers["desired"][k] = v

        default_buffers.setdefault("derived", {})
        for k, v in desired_buffers.get("derived", {}).items():
            default_buffers["derived"][k] = v

        default_dict.update(desires_dict)

        desires = CapacityDesires(**default_dict)

        # If user gave state item count but not size or size but not count
        # calculate the missing one from the other
        user_size = (
            self.model_dump()
            .get("data_shape", {})
            .get("estimated_state_size_gib", None)
        )
        user_count = self.data_shape.estimated_state_item_count
        item_size_bytes = desires.query_pattern.estimated_mean_write_size_bytes.mid
        if user_size is None and user_count is not None:
            desires.data_shape.estimated_state_size_gib = user_count.scale(
                factor=(item_size_bytes / GIB_IN_BYTES)
            )
        elif user_size is not None and user_count is None:
            user_size_gib = self.data_shape.estimated_state_size_gib
            desires.data_shape.estimated_state_item_count = user_size_gib.scale(
                factor=(GIB_IN_BYTES / item_size_bytes)
            )

        return desires


class CapacityRequirement(ExcludeUnsetModel):
    requirement_type: str

    # cpu_cores was calculated relative to this reference, mostly for
    # comparing latency to clock frequency
    reference_shape: Instance = default_reference_shape
    cpu_cores: Interval
    mem_gib: Interval = certain_int(0)
    network_mbps: Interval = certain_int(0)
    disk_gib: Interval = certain_int(0)

    context: Dict = {}


class ClusterCapacity(ExcludeUnsetModel):
    cluster_type: str

    count: int
    instance: Instance
    attached_drives: Sequence[Drive] = ()
    annual_cost: float
    # When provisioning services we might need to signal they
    # should have certain configuration, for example flags that
    # affect durability shut off
    cluster_params: Dict = {}


class ServiceCapacity(ExcludeUnsetModel):
    service_type: str
    annual_cost: float
    # If this cost should impact regret. Usually they do not since they
    # are proportional to the input desire not the output cluster
    regret_cost: bool = False
    # Often while provisioning cloud services we need to represent
    # parameters to the cloud APIs, use this to inject those from models
    service_params: Dict = {}


# For services that are provisioned by zone (e.g. Cassandra, EVCache)
class ZoneClusterCapacity(ClusterCapacity):
    pass


# For services that are provisioned regionally (e.g. Java services, RDS, etc ..)
class RegionClusterCapacity(ClusterCapacity):
    pass


class Requirements(ExcludeUnsetModel):
    zonal: Sequence[CapacityRequirement] = []
    regional: Sequence[CapacityRequirement] = []

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


class Clusters(ExcludeUnsetModel):
    annual_costs: Dict[str, Decimal]
    zonal: Sequence[ZoneClusterCapacity] = []
    regional: Sequence[RegionClusterCapacity] = []
    services: Sequence[ServiceCapacity] = []

    # Backwards compatibility for total_annual_cost
    @computed_field(return_type=float)  # type: ignore
    @property
    def total_annual_cost(self) -> float:
        return round(float(sum(self.annual_costs.values())), 2)


class CapacityPlan(ExcludeUnsetModel):
    requirements: Requirements
    candidate_clusters: Clusters
    rank: int = 0


# Parameters to cost functions of the form
# let y = the optimal value
# let x = the proposed value
# let cost = (x - y) ^ exponent
class Regret(ExcludeUnsetModel):
    over_provision_cost: float = 0
    under_provision_cost: float = 0
    exponent: float = 1.0


class CapacityRegretParameters(ExcludeUnsetModel):
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


class PlanExplanation(ExcludeUnsetModel):
    regret_params: CapacityRegretParameters
    regret_clusters_by_model: Dict[
        str, Sequence[Tuple[CapacityPlan, CapacityDesires, float]]
    ] = {}
    desires_by_model: Dict[str, CapacityDesires] = {}
    context: Dict[str, Any] = {}


class UncertainCapacityPlan(ExcludeUnsetModel):
    requirements: Requirements
    least_regret: Sequence[CapacityPlan]
    mean: Sequence[CapacityPlan]
    percentiles: Dict[int, Sequence[CapacityPlan]]
    explanation: PlanExplanation

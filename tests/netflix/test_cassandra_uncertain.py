from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from tests.util import assert_minimum_storage_gib
from tests.util import assert_similar_compute
from tests.util import get_drive_size_gib
from tests.util import has_attached_storage
from tests.util import simple_drive

uncertain_mid = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(
            low=1000, mid=10000, high=100000, confidence=0.98
        ),
        estimated_write_per_second=Interval(
            low=1000, mid=10000, high=100000, confidence=0.98
        ),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(low=100, mid=500, high=1000, confidence=0.98),
    ),
)

uncertain_tiny = CapacityDesires(
    service_tier=2,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(low=1, mid=10, high=100, confidence=0.98),
        estimated_write_per_second=Interval(low=1, mid=10, high=100, confidence=0.98),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(low=1, mid=10, high=30, confidence=0.98),
    ),
)


def test_uncertain_planning():
    mid_plan = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=uncertain_mid,
    )
    lr = mid_plan.least_regret[0]
    lr_cluster = lr.candidate_clusters.zonal[0]
    assert 8 <= lr_cluster.count * lr_cluster.instance.cpu <= 64
    assert (
        5_000 <= lr.candidate_clusters.annual_costs["cassandra.zonal-clusters"] < 45_000
    )

    sr = mid_plan.least_regret[1]
    sr_cluster = sr.candidate_clusters.zonal[0]
    assert 8 <= sr_cluster.count * sr_cluster.instance.cpu <= 64
    assert (
        5_000 <= sr.candidate_clusters.annual_costs["cassandra.zonal-clusters"] < 45_000
    )

    tiny_plan = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=uncertain_tiny,
    )
    lr = tiny_plan.least_regret[0]
    lr_cluster = lr.candidate_clusters.zonal[0]
    assert 2 <= lr_cluster.count * lr_cluster.instance.cpu < 16
    assert (
        1_000 < lr.candidate_clusters.annual_costs["cassandra.zonal-clusters"] < 6_000
    )


def test_increasing_qps_simple():
    qps_values = (100, 1000, 10_000, 100_000)
    result = []
    for qps in qps_values:
        simple = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=Interval(
                    low=qps // 10, mid=qps, high=qps * 10, confidence=0.98
                ),
                estimated_write_per_second=Interval(
                    low=qps // 10, mid=qps, high=qps * 10, confidence=0.98
                ),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=Interval(
                    low=20, mid=200, high=2000, confidence=0.98
                ),
            ),
        )

        cap_plan = planner.plan(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=simple,
            simulations=256,
        )

        lr = cap_plan.least_regret[0].candidate_clusters.zonal[0]
        lr_cpu = lr.count * lr.instance.cpu
        lr_cost = cap_plan.least_regret[0].candidate_clusters.annual_costs[
            "cassandra.zonal-clusters"
        ]
        lr_family = lr.instance.family
        assert_minimum_storage_gib(lr, 100)

        result.append(
            (lr_family, lr_cpu, lr_cost, cap_plan.least_regret[0].requirements.zonal[0])
        )

    # We should generally want cheap CPUs
    assert all(r[0][0] in ("r", "m", "i", "c") for r in result)

    # Should have more capacity as requirement increases
    x = [r[1] for r in result]
    assert x[0] < x[-1]
    assert sorted(x) == x


worn_desire = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        # Very Very few reads.
        estimated_read_per_second=Interval(low=1, mid=10, high=100, confidence=0.98),
        # We think we're going to have around 1 million writes per second
        estimated_write_per_second=Interval(
            low=100_000, mid=1_000_000, high=2_000_000, confidence=0.98
        ),
    ),
    # We think we're going to have around 200 TiB of data
    data_shape=DataShape(
        estimated_state_size_gib=Interval(
            low=102_400, mid=204_800, high=404_800, confidence=0.98
        ),
    ),
)


def test_worn_dataset():
    """
    Assert that a write once read never (aka tracing) dataset uses
    CPU and attached drives to max ability unless ephemeral works out to be
    cheaper
    """
    cap_plan = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=worn_desire,
        extra_model_arguments={
            "max_regional_size": 200,
            "copies_per_region": 2,
        },
    )

    lr_clusters = [lr.candidate_clusters.zonal[0] for lr in cap_plan.least_regret]
    for lr_cluster in lr_clusters:
        assert_minimum_storage_gib(lr_cluster, 102_400)
        assert_similar_compute(
            expected_shape=shapes.instance("r7a.xlarge"),
            actual_shape=lr_cluster.instance,
            expected_count=64,
            actual_count=lr_cluster.count,
            expected_attached_disk=simple_drive(
                size_gib=2000, read_io_per_s=200, write_io_per_s=200
            ),
            actual_attached_disk=lr_cluster.attached_drives[0]
            if lr_cluster.attached_drives
            else None,
        )


def test_worn_dataset_force_ebs():
    """Assert that a write once read never (aka tracing) dataset uses
    has reasonable outputs for ebs results
    """
    cap_plan = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=worn_desire,
        extra_model_arguments={
            "max_regional_size": 200,
            "copies_per_region": 2,
            "require_attached_disks": True,
        },
    )

    lr_clusters = [lr.candidate_clusters.zonal[0] for lr in cap_plan.least_regret]
    for lr_cluster in lr_clusters:
        assert has_attached_storage(lr_cluster)
        assert lr_cluster.attached_drives[0].name == "gp3"
        assert_minimum_storage_gib(lr_cluster, 102_400)
        # gp3 should not provision massive drives, prefer to upcolor
        assert get_drive_size_gib(lr_cluster) <= 8 * 1024

        # (matthewho) Why is this r6a.2xlarge??? The above test has the
        # same desires and is m7a.xlarge. The only meaningful difference is
        # the required_attached_disks. Also, the resolved disk size is larger
        assert_similar_compute(
            expected_shape=shapes.instance("r6a.2xlarge"),
            actual_shape=lr_cluster.instance,
            expected_count=64,
            actual_count=lr_cluster.count,
            expected_attached_disk=simple_drive(
                size_gib=2900, read_io_per_s=200, write_io_per_s=200
            ),
            actual_attached_disk=lr_cluster.attached_drives[0]
            if lr_cluster.attached_drives
            else None,
        )


def test_very_small_has_disk():
    very_small = CapacityDesires(
        service_tier=2,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=1, mid=10, high=100, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=1, mid=10, high=100, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(low=1, mid=10, high=30, confidence=0.98),
        ),
    )
    cap_plan = planner.plan(
        model_name="org.netflix.cassandra", region="us-east-1", desires=very_small
    )

    for lr in cap_plan.least_regret:
        lr_cluster = lr.candidate_clusters.zonal[0]
        assert 2 <= lr_cluster.count * lr_cluster.instance.cpu < 16
        assert (
            1_000
            < lr.candidate_clusters.annual_costs["cassandra.zonal-clusters"]
            < 6_000
        )
        assert_minimum_storage_gib(lr_cluster, 10)

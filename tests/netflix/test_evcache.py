from hypothesis import assume, given, HealthCheck, settings

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import (
    CapacityDesires,
)
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.org.netflix.evcache import (
    calculate_read_cpu_time_evcache_ms,
    calculate_spread_cost,
)
from tests.netflix.property_test_utils import capacity_desires_simple

# Property test configuration for EVCache model.
# See tests/netflix/PROPERTY_TESTING.md for configuration options and examples.
PROPERTY_TEST_CONFIG = {
    # "org.netflix.evcache": {
    #     "extra_model_arguments": {},
    # },
}


def test_evcache_read_latency():
    # 256 bits = 32 bytes 10
    small = calculate_read_cpu_time_evcache_ms(32)
    # 1600 bits = 200 bytes 41
    medium = calculate_read_cpu_time_evcache_ms(200)
    # 8192 bits = 1024 bytes 66
    large = calculate_read_cpu_time_evcache_ms(1024)
    # 24   KiB  = 133
    very_large = calculate_read_cpu_time_evcache_ms(24 * 1024)
    # 40   KiB  = 158
    extra_large = calculate_read_cpu_time_evcache_ms(40 * 1024)

    assert calculate_read_cpu_time_evcache_ms(1) > 0
    assert 0.008 < small < 0.015
    assert 0.030 < medium < 0.050
    assert 0.060 < large < 0.080
    assert 0.120 < very_large < 0.140
    assert 0.140 < extra_large < 0.160


def test_evcache_inmemory_low_latency_reads_cpu():
    inmemory_cluster_low_latency_reads_qps = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=18300000, mid=34200000, high=34200000 * 1.2, confidence=1.0
            ),
            estimated_write_per_second=Interval(
                low=228000, mid=536000, high=536000 * 1.2, confidence=1.0
            ),
            estimated_mean_write_size_bytes=Interval(
                low=3778, mid=3778, high=3778 * 1.2, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=35, mid=35, high=35 * 1.2, confidence=1.0
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(low=36, mid=36, high=36, confidence=1.0),
            estimated_state_item_count=Interval(
                low=416000000, mid=804000000, high=804000000 * 1.2, confidence=1.0
            ),
        ),
    )

    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=inmemory_cluster_low_latency_reads_qps,
    )

    for candidate in plan:
        total_cpu_power = (
            candidate.candidate_clusters.zonal[0].count
            * candidate.candidate_clusters.zonal[0].instance.cpu
            * candidate.candidate_clusters.zonal[0].instance.cpu_ghz
        )

        assert total_cpu_power > 700, (
            f"CPU power is not sufficient for low latency reads, with"
            f" {candidate.candidate_clusters.zonal[0].count} *"
            f" {candidate.candidate_clusters.zonal[0].instance.name},"
            f" total= {total_cpu_power}."
        )


class BufferComponents:
    pass


def test_evcache_inmemory_medium_latency_reads_cpu():
    inmemory_cluster_medium_latency_reads_qps = CapacityDesires(
        service_tier=0,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=470000, mid=1800000, high=1800000 * 1.2, confidence=1.0
            ),
            estimated_mean_write_per_second=Interval(
                low=505000, mid=861000, high=861000 * 1.2, confidence=1.0
            ),
            estimated_mean_write_size_bytes=Interval(
                low=365, mid=365, high=365 * 1.2, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=193, mid=193, high=193 * 1.2, confidence=1.0
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(low=61, mid=61, high=61, confidence=1.0),
            estimated_state_item_count=Interval(
                low=125000000, mid=202000000, high=202000000 * 1.2, confidence=1.0
            ),
        ),
    )

    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=inmemory_cluster_medium_latency_reads_qps,
    )

    for candidate in plan:
        total_cpu_power = (
            candidate.candidate_clusters.zonal[0].count
            * candidate.candidate_clusters.zonal[0].instance.cpu
            * candidate.candidate_clusters.zonal[0].instance.cpu_ghz
        )

        assert total_cpu_power > 150


def test_evcache_inmemory_high_latency_reads_cpu():
    inmemory_cluster_high_latency_reads_qps = CapacityDesires(
        service_tier=0,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=113000, mid=441000, high=441000 * 1.2, confidence=1.0
            ),
            estimated_write_per_second=Interval(
                low=19000, mid=35000, high=35000 * 1.2, confidence=1.0
            ),
            estimated_mean_write_size_bytes=Interval(
                low=7250, mid=7250, high=7250 * 1.2, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=5100, mid=5100, high=5100 * 1.2, confidence=1.0
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=1662, mid=1662, high=1662, confidence=1.0
            ),
            estimated_state_item_count=Interval(
                low=750000000, mid=750000000, high=750000000 * 1.2, confidence=1.0
            ),
        ),
    )

    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=inmemory_cluster_high_latency_reads_qps,
    )

    for candidate in plan:
        total_cpu_power = (
            candidate.candidate_clusters.zonal[0].count
            * candidate.candidate_clusters.zonal[0].instance.cpu
            * candidate.candidate_clusters.zonal[0].instance.cpu_ghz
        )

        assert total_cpu_power > 100


def test_evcache_ondisk_low_latency_reads_cpu():
    ondisk_cluster_low_latency_reads_qps = CapacityDesires(
        service_tier=0,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=284, mid=7110000, high=7110000 * 1.2, confidence=1.0
            ),
            estimated_write_per_second=Interval(
                low=0, mid=2620000, high=2620000 * 1.2, confidence=1.0
            ),
            estimated_mean_write_size_bytes=Interval(
                low=12000, mid=12000, high=12000 * 1.2, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=16000, mid=16000, high=16000 * 1.2, confidence=1.0
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=2306867, mid=2306867, high=2306867, confidence=1.0
            ),
            estimated_state_item_count=Interval(
                low=132000000000,
                mid=132000000000,
                high=132000000000 * 1.2,
                confidence=1.0,
            ),
        ),
    )

    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=ondisk_cluster_low_latency_reads_qps,
    )

    for candidate in plan:
        total_cpu_power = (
            candidate.candidate_clusters.zonal[0].count
            * candidate.candidate_clusters.zonal[0].instance.cpu
            * candidate.candidate_clusters.zonal[0].instance.cpu_ghz
        )

        assert total_cpu_power > 8000


def test_evcache_ondisk_high_latency_reads_cpu():
    ondisk_cluster_high_latency_reads_qps = CapacityDesires(
        service_tier=0,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=312000, mid=853000, high=853000 * 1.2, confidence=1.0
            ),
            estimated_write_per_second=Interval(
                low=0, mid=310000, high=310000 * 1.2, confidence=1.0
            ),
            estimated_write_size_bytes=Interval(
                low=34500, mid=34500, high=34500 * 1.2, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=41000, mid=41000, high=41000 * 1.2, confidence=1.0
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=281000, mid=281000, high=281000, confidence=1.0
            ),
            estimated_state_item_count=Interval(
                low=8518318523, mid=8518318523, high=8518318523 * 1.2, confidence=1.0
            ),
        ),
    )

    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=ondisk_cluster_high_latency_reads_qps,
    )

    for candidate in plan:
        total_cpu_power = (
            candidate.candidate_clusters.zonal[0].count
            * candidate.candidate_clusters.zonal[0].instance.cpu
            * candidate.candidate_clusters.zonal[0].instance.cpu_ghz
            * candidate.candidate_clusters.zonal[0].instance.cpu_ipc_scale
        )

        assert total_cpu_power > 800


def test_evcache_inmemory_ram_usage():
    inmemory_qps = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=18300000, mid=34200000, high=34200000 * 1.2, confidence=1.0
            ),
            estimated_write_per_second=Interval(
                low=228000, mid=536000, high=536000 * 1.2, confidence=1.0
            ),
            estimated_mean_write_size_bytes=Interval(
                low=3778, mid=3778, high=3778 * 1.2, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=35, mid=35, high=35 * 1.2, confidence=1.0
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(low=36, mid=36, high=36, confidence=1.0),
            estimated_state_item_count=Interval(
                low=416000000, mid=804000000, high=804000000 * 1.2, confidence=1.0
            ),
        ),
    )

    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=inmemory_qps,
    )

    for candidate in plan:
        total_ram = (
            candidate.candidate_clusters.zonal[0].instance.ram_gib
            * candidate.candidate_clusters.zonal[0].count
        )

        assert total_ram > inmemory_qps.data_shape.estimated_state_size_gib.mid


def test_evcache_ondisk_disk_usage():
    inmemory_qps = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=18300000, mid=34200000, high=34200000 * 1.2, confidence=1.0
            ),
            estimated_write_per_second=Interval(
                low=228000, mid=536000, high=536000 * 1.2, confidence=1.0
            ),
            estimated_mean_write_size_bytes=Interval(
                low=3778, mid=3778, high=3778 * 1.2, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=35, mid=35, high=35 * 1.2, confidence=1.0
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(low=36, mid=36, high=36, confidence=1.0),
            estimated_state_item_count=Interval(
                low=416000000, mid=804000000, high=804000000 * 1.2, confidence=1.0
            ),
        ),
    )

    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=inmemory_qps,
    )

    for candidate in plan:
        total_ram = (
            candidate.candidate_clusters.zonal[0].instance.ram_gib
            * candidate.candidate_clusters.zonal[0].count
        )

        assert total_ram > inmemory_qps.data_shape.estimated_state_size_gib.mid


def test_evcache_ondisk_high_disk_usage():
    high_disk_usage_rps = CapacityDesires(
        service_tier=0,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=284, mid=7110000, high=7110000 * 1.2, confidence=1.0
            ),
            estimated_write_per_second=Interval(
                low=0, mid=2620000, high=2620000 * 1.2, confidence=1.0
            ),
            estimated_mean_write_size_bytes=Interval(
                low=12000, mid=12000, high=12000 * 1.2, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=16000, mid=16000, high=16000 * 1.2, confidence=1.0
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=2306867, mid=2306867, high=2306867, confidence=1.0
            ),
            estimated_state_item_count=Interval(
                low=132000000000,
                mid=132000000000,
                high=132000000000 * 1.2,
                confidence=1.0,
            ),
        ),
    )

    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=high_disk_usage_rps,
    )

    for candidate in plan:
        if candidate.candidate_clusters.zonal[0].instance.drive is not None:
            total_disk = (
                candidate.candidate_clusters.zonal[0].instance.drive.size_gib
                * candidate.candidate_clusters.zonal[0].count
            )

            assert (
                total_disk > high_disk_usage_rps.data_shape.estimated_state_size_gib.mid
            )


def test_evcache_zero_item_count():
    zero_item_count_rps = CapacityDesires(
        service_tier=0,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=1, mid=1, high=1 * 1.2, confidence=1.0
            ),
            estimated_write_per_second=Interval(
                low=1, mid=1, high=1 * 1.2, confidence=1.0
            ),
            estimated_mean_write_size_bytes=Interval(
                low=1, mid=1, high=1 * 1, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=1, mid=1, high=1 * 1, confidence=1.0
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(low=0, mid=0, high=0, confidence=1.0),
            estimated_state_item_count=Interval(low=0, mid=0, high=0, confidence=1.0),
        ),
    )

    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=zero_item_count_rps,
    )

    for candidate in plan:
        if candidate.candidate_clusters.zonal[0].instance.drive is not None:
            total_ram = (
                candidate.candidate_clusters.zonal[0].instance.drive.size_gib
                * candidate.candidate_clusters.zonal[0].count
            )

            assert (
                total_ram > zero_item_count_rps.data_shape.estimated_state_size_gib.mid
            )


def test_evcache_string_arguments_coercion():
    simple_desires = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=100000, mid=200000, high=240000, confidence=1.0
            ),
            estimated_write_per_second=Interval(
                low=10000, mid=20000, high=24000, confidence=1.0
            ),
            estimated_mean_write_size_bytes=Interval(
                low=100, mid=200, high=240, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=100, mid=200, high=240, confidence=1.0
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(low=10, mid=20, high=30, confidence=1.0),
            estimated_state_item_count=Interval(
                low=1000000, mid=2000000, high=2400000, confidence=1.0
            ),
        ),
    )

    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=simple_desires,
        extra_model_arguments={
            # All of these are supposed to be int
            "copies_per_region": "3",
            "max_regional_size": "10000",
            "max_local_disk_gib": "2048",
            "min_instance_memory_gib": "12",
        },
    )

    assert len(plan) > 0, "Should generate at least one plan"

    for candidate in plan:
        assert len(candidate.candidate_clusters.zonal) > 0


def test_spread_cost_function():
    """calculate_spread_cost returns additive dollar penalty for small clusters."""
    assert calculate_spread_cost(1) == 100000  # < 2 nodes → max penalty
    assert calculate_spread_cost(0) == 100000
    assert calculate_spread_cost(11) == 0  # above threshold
    assert calculate_spread_cost(20) == 0
    # Intermediate values decrease as cluster grows
    assert calculate_spread_cost(2) > calculate_spread_cost(5)
    assert calculate_spread_cost(5) > calculate_spread_cost(9)


def test_evcache_no_spread_cost_in_annual_costs():
    """After refactor, annual_costs should not contain fake spread dollars."""
    small_cluster = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=1000, mid=2000, high=3000, confidence=1.0
            ),
            estimated_write_per_second=Interval(
                low=100, mid=200, high=300, confidence=1.0
            ),
            estimated_mean_write_size_bytes=Interval(
                low=100, mid=200, high=300, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=100, mid=200, high=300, confidence=1.0
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(low=1, mid=2, high=3, confidence=1.0),
        ),
    )

    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=small_cluster,
    )

    assert len(plan) > 0
    for candidate in plan:
        # No fake spread.cost dollars in annual_costs
        for cost_key in candidate.candidate_clusters.annual_costs:
            assert "spread.cost" not in cost_key, (
                f"Found old spread.cost key: {cost_key}"
            )


def test_evcache_small_cluster_rank_penalty():
    """Small EVCache clusters have rank > cost (penalty)."""
    small_cluster = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=1000, mid=2000, high=3000, confidence=1.0
            ),
            estimated_write_per_second=Interval(
                low=100, mid=200, high=300, confidence=1.0
            ),
            estimated_mean_write_size_bytes=Interval(
                low=100, mid=200, high=300, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=100, mid=200, high=300, confidence=1.0
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(low=1, mid=2, high=3, confidence=1.0),
        ),
    )

    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=small_cluster,
    )

    assert len(plan) > 0
    found_penalized = False
    for candidate in plan:
        cluster_count = candidate.candidate_clusters.zonal[0].count
        if cluster_count < 10:
            # rank should be inflated above raw cost
            assert candidate.rank > float(
                candidate.candidate_clusters.total_annual_cost
            ), (
                f"Small cluster ({cluster_count} nodes) should have rank > cost, "
                f"got rank={candidate.rank}, "
                f"cost={candidate.candidate_clusters.total_annual_cost}"
            )
            found_penalized = True
    assert found_penalized, "Expected at least one plan with < 10 nodes per zone"


@settings(
    max_examples=25,
    deadline=15000,
    suppress_health_check=[HealthCheck.filter_too_much],
)
@given(
    desires=capacity_desires_simple(
        min_qps=100,
        max_qps=100_000,
        min_data_gib=1,
        max_data_gib=500,
    )
)
def test_evcache_spread_invariants(desires):
    """Property: For any EVCache workload, spread penalty invariants hold.

    1. No plan's annual_costs contain fake 'spread.cost' dollars
    2. Under-spread plans (< 10 nodes) have rank > cost (penalty inflated)
    3. Plans are rank-sorted (plan_certain guarantees this)
    4. Under-spread plans carry RANK_PENALTIES metadata with correct coefficient
    """
    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=desires,
    )
    assume(len(plan) > 0)

    prev_rank = 0
    for candidate in plan:
        cost = float(candidate.candidate_clusters.total_annual_cost)
        zonal = candidate.candidate_clusters.zonal

        # Invariant 1: no fake spread dollars
        for cost_key in candidate.candidate_clusters.annual_costs:
            assert "spread.cost" not in cost_key

        # Invariant 3: plans are rank-sorted
        assert candidate.rank >= prev_rank
        prev_rank = candidate.rank

        if not zonal:
            continue

        count = zonal[0].count
        params = zonal[0].cluster_params or {}
        penalties = params.get("rank_penalties", {})

        if count <= 10:
            expected_spread = calculate_spread_cost(count)

            if expected_spread > 0:
                # Invariant 2: rank is inflated by spread cost
                assert candidate.rank > cost, (
                    f"count={count}, rank={candidate.rank}, cost={cost}"
                )

                # Invariant 4: penalty metadata present with correct dollar amount
                assert "under_spread" in penalties, (
                    f"count={count} should have under_spread penalty"
                )
                assert abs(penalties["under_spread"] - expected_spread) < 0.01

                # Invariant 5: rank = cost + spread_cost (additive, same as old)
                assert abs(candidate.rank - (cost + expected_spread)) < 0.01, (
                    f"rank should be cost + spread: "
                    f"rank={candidate.rank}, cost={cost}, spread={expected_spread}"
                )
            else:
                assert "under_spread" not in penalties
        else:
            # Well-spread clusters should not have under_spread penalty
            assert "under_spread" not in penalties

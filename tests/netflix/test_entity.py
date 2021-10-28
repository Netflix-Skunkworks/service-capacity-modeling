from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern


def test_entity_increasing_qps_simple():
    qps_values = (100, 1000, 10_000, 100_000)
    entity_results_trend = []
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
            model_name="org.netflix.entity",
            region="us-east-1",
            desires=simple,
            simulations=256,
        )

        # Check the Java cluster
        entity_plan = next(filter(lambda c: c.cluster_type == 'dgwentity', cap_plan.least_regret[0].candidate_clusters.regional))
        entity_results_trend.append(
            (
                entity_plan.count * entity_plan.instance.cpu,
            )
        )
        # We just want ram and cpus for a java app
        assert entity_plan.instance.family in ("m5", "r5")
        # We should never be paying for ephemeral drives
        assert entity_plan.instance.drive is None

    # Should have more capacity as requirement increases
    x = [r[0] for r in entity_results_trend]
    assert x[0] < x[-1]
    assert sorted(x) == x

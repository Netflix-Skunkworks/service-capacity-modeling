from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern


def test_entity_increasing_qps_simple():
    qps_values = (100, 1000, 10_000, 100_000)
    cass_results_trend = []
    es_results_trend = []
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

        print(cap_plan.least_regret[0].candidate_clusters.json(indent=2))
        print(cap_plan.least_regret[0].requirements.json(indent=2))

        # Check the C* cluster
        cass_plan = next(filter(lambda c: c.cluster_type == 'cassandra', cap_plan.least_regret[0].candidate_clusters.zonal))
        cass_req = next(filter(lambda r: r.requirement_type == 'cassandra-zonal', cap_plan.least_regret[0].requirements.zonal))
        cass_plan_cpu = cass_plan.count * cass_plan.instance.cpu
        cass_plan_cost = cap_plan.least_regret[0].candidate_clusters.total_annual_cost
        cass_plan_family = cass_plan.instance.family
        if cass_plan.instance.drive is None:
            assert sum(dr.size_gib for dr in cass_plan.attached_drives) >= 200
        else:
            assert cass_plan.instance.drive.size_gib >= 100

        # We should generally want cheap CPUs for Cassandra
        assert cass_plan_family in ("r5", "m5d", "m5", "i3en")
        cass_results_trend.append(
            (
                cass_plan_cpu,
                cass_plan_cost,
                cass_req,
            )
        )

        # Check the ES* cluster
        es_plan = next(filter(lambda c: c.cluster_type == 'elasticsearch-data', cap_plan.least_regret[0].candidate_clusters.zonal))
        es_req = next(filter(lambda r: r.requirement_type == 'elasticsearch-data-zonal', cap_plan.least_regret[0].requirements.zonal))
        es_plan_cpu = es_plan.count * es_plan.instance.cpu
        es_plan_cost = cap_plan.least_regret[0].candidate_clusters.total_annual_cost
        es_plan_family = es_plan.instance.family
        if es_plan.instance.drive is None:
            assert sum(dr.size_gib for dr in es_plan.attached_drives) >= 200
        else:
            assert es_plan.instance.drive.size_gib >= 70

        assert es_plan_family in ("m5d", "m5d.xlarge", "r5d", "r5d.large")
        es_results_trend.append(
            (
                es_plan_cpu,
                es_plan_cost,
                es_req,
            )
        )

        # Check the Java cluster
        entity_plan = next(filter(lambda c: c.cluster_type == 'dgwmes', cap_plan.least_regret[0].candidate_clusters.regional))
        entity_results_trend.append(
            (
                entity_plan.count * entity_plan.instance.cpu,
                cass_plan_cost,
                es_plan_cost
            )
        )
        # We just want ram and cpus for a java app
        assert entity_plan.instance.family in ("m5", "r5")
        # We should never be paying for ephemeral drives
        assert entity_plan.instance.drive is None

    # Should have more capacity as requirement increases
    x = [r[0] for r in cass_results_trend]
    assert x[0] < x[-1]
    assert sorted(x) == x

    # Should have more capacity as requirement increases
    x = [r[0] for r in es_results_trend]
    assert x[0] < x[-1]
    assert sorted(x) == x

    # Should have more capacity as requirement increases
    x = [r[0] for r in entity_results_trend]
    assert x[0] < x[-1]
    assert sorted(x) == x

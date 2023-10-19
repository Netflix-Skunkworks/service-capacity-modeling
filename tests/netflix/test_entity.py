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
                estimated_mean_write_size_bytes=Interval(
                    low=1024, mid=1024 * 10, high=1024 * 100, confidence=0.98
                ),
            ),
            data_shape=DataShape(
                estimated_state_item_count=Interval(
                    low=1000000, mid=10000000, high=100000000, confidence=0.98
                ),
            ),
        )

        cap_plan = planner.plan(
            model_name="org.netflix.entity",
            region="us-east-1",
            desires=simple,
            simulations=256,
        )

        # the set of cluster types the planner chose
        types = {
            c.cluster_type
            for c in list(cap_plan.least_regret[0].candidate_clusters.regional)
            + list(cap_plan.least_regret[0].candidate_clusters.zonal)
        }
        assert sorted(types) == [
            "cassandra",
            "cockroachdb",
            "dgwentity",
            "dgwkv",
            "elasticsearch-data",
            "elasticsearch-master",
            "elasticsearch-search",
        ]

        # Check the Java cluster
        entity_plan = next(
            filter(
                lambda c: c.cluster_type == "dgwentity",
                cap_plan.least_regret[0].candidate_clusters.regional,
            )
        )
        entity_results_trend.append((entity_plan.count * entity_plan.instance.cpu,))
        # We just want ram and cpus for a java app
        assert entity_plan.instance.family[0] in ("m", "r")
        # We should never be paying for ephemeral drives
        assert entity_plan.instance.drive is None
        # CRDB disk usage should be num items * 512 bytes/per item ~= 6 GB (rounded up)

        for c in cap_plan.least_regret[0].requirements.zonal:
            if c.requirement_type == "crdb-zonal":
                assert c.disk_gib.mid == 6.0

    # Should have more capacity as requirement increases
    x = [r[0] for r in entity_results_trend]
    assert x[0] < x[-1]
    assert sorted(x) == x


def test_entity_increasing_qps_item_count_unset():
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
                estimated_mean_write_size_bytes=Interval(
                    low=1024, mid=1024 * 10, high=1024 * 100, confidence=0.98
                ),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=Interval(
                    low=10, mid=100, high=1000, confidence=0.98
                ),
            ),
        )

        cap_plan = planner.plan(
            model_name="org.netflix.entity",
            region="us-east-1",
            desires=simple,
            simulations=256,
        )

        # the set of cluster types the planner chose
        types = {
            c.cluster_type
            for c in list(cap_plan.least_regret[0].candidate_clusters.regional)
            + list(cap_plan.least_regret[0].candidate_clusters.zonal)
        }
        assert sorted(types) == [
            "cassandra",
            "cockroachdb",
            "dgwentity",
            "dgwkv",
            "elasticsearch-data",
            "elasticsearch-master",
            "elasticsearch-search",
        ]

        # Check the Java cluster
        entity_plan = next(
            filter(
                lambda c: c.cluster_type == "dgwentity",
                cap_plan.least_regret[0].candidate_clusters.regional,
            )
        )
        entity_results_trend.append((entity_plan.count * entity_plan.instance.cpu,))
        # We just want ram and cpus for a java app
        assert entity_plan.instance.family[0] in ("m", "r")
        # We should never be paying for ephemeral drives
        assert entity_plan.instance.drive is None
        # CRDB disk usage should be num items * 512 bytes/per item ~= 7 GB (rounded up)

        for c in cap_plan.least_regret[0].requirements.zonal:
            if c.requirement_type == "crdb-zonal":
                assert c.disk_gib.mid == 7.0

    # Should have more capacity as requirement increases
    x = [r[0] for r in entity_results_trend]
    assert x[0] < x[-1]
    assert sorted(x) == x

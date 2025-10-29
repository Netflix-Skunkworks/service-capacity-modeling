from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import Consistency
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import GlobalConsistency
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern


def test_kv_increasing_qps_simple():
    qps_values = (100, 1000, 10_000, 100_000)
    zonal_result = []
    regional_result = []
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
            model_name="org.netflix.key-value",
            region="us-east-1",
            desires=simple,
            simulations=256,
        )

        # Validate that there is no EVCache cluster
        assert not any(
            cluster
            for cluster in cap_plan.least_regret[0].candidate_clusters.zonal
            if cluster.cluster_type == "evcache"
        )

        # Check the C* cluster
        zlr = cap_plan.least_regret[0].candidate_clusters.zonal[0]
        zlr_cpu = zlr.count * zlr.instance.cpu
        zlr_cost = cap_plan.least_regret[0].candidate_clusters.total_annual_cost
        zlr_family = zlr.instance.family
        if zlr.instance.drive is None:
            assert sum(dr.size_gib for dr in zlr.attached_drives) >= 100
        else:
            assert zlr.instance.drive.size_gib >= 100

        zonal_result.append(
            (
                zlr_family,
                zlr_cpu,
                zlr_cost,
                cap_plan.least_regret[0].requirements.zonal[0],
            )
        )

        # Check the Java cluster
        rlr = cap_plan.least_regret[0].candidate_clusters.regional[0]
        regional_result.append(
            (rlr.instance.family, rlr.count * rlr.instance.cpu, rlr.annual_cost)
        )
        # We should never be paying for ephemeral drives
        assert rlr.instance.drive is None

    # We should generally want cheap CPUs for Cassandra
    assert all(r[0][0] in ("r", "m", "c", "i") for r in zonal_result)

    # We just want ram and cpus for a java app
    assert all(r[0][0] in ("m", "r", "c") for r in regional_result)

    # Should have more capacity as requirement increases
    x = [r[1] for r in zonal_result]
    assert x[0] < x[-1]
    assert sorted(x) == x

    # Should have more capacity as requirement increases
    x = [r[1] for r in regional_result]
    assert x[0] < x[-1]
    assert sorted(x) == x


def test_kv_increasing_qps_compare_working_sets():
    qps_values = (100, 1000, 10_000, 100_000)
    for qps in qps_values:
        # Create two copies of the desires, one with small working set
        # and one with large working set.
        small = CapacityDesires(
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
                estimated_working_set_percent=certain_float(0.10),
            ),
        )
        large = small.model_copy(deep=True)
        large.data_shape.estimated_working_set_percent = certain_float(0.90)

        cap_plan_small = planner.plan(
            model_name="org.netflix.key-value",
            region="us-east-1",
            desires=small,
            simulations=256,
        )

        cap_plan_large = planner.plan(
            model_name="org.netflix.key-value",
            region="us-east-1",
            desires=large,
            simulations=256,
        )

        # Validate that there is no EVCache cluster
        assert not any(
            cluster
            for cluster in cap_plan_small.least_regret[0].candidate_clusters.zonal
            if cluster.cluster_type == "evcache"
        )
        assert not any(
            cluster
            for cluster in cap_plan_large.least_regret[0].candidate_clusters.zonal
            if cluster.cluster_type == "evcache"
        )

        # Check the C* cluster
        zlr_small = cap_plan_small.least_regret[0].candidate_clusters.zonal[0]
        zlr_small_cpu = zlr_small.count * zlr_small.instance.cpu
        zlr_small_memory = zlr_small.count * zlr_small.instance.ram_gib

        zlr_large = cap_plan_large.least_regret[0].candidate_clusters.zonal[0]
        zlr_large_cpu = zlr_large.count * zlr_large.instance.cpu
        zlr_large_memory = zlr_large.count * zlr_large.instance.ram_gib

        # For smaller qps, cost should be less for smaller working set
        # (due to needing to keep less in memory). This tilts to c/m instead of
        # m/r. NOTE: more memory is not always more expensive when combining
        # EBS with r instances. For example, a r6a.2xlarge in some configurations
        # can be cheaper than the m6id.2xlarge due to cheaper EBS costs.
        assert zlr_small_cpu <= zlr_large_cpu
        assert zlr_small_memory <= zlr_large_memory

        # We should generally want cheap CPUs for Cassandra
        assert all(
            cluster.instance.family[0] in ("r", "m", "i", "c")
            for cluster in (zlr_small, zlr_large)
        )

        rlr_small = cap_plan_small.least_regret[0].candidate_clusters.regional[0]
        rlr_large = cap_plan_small.least_regret[0].candidate_clusters.regional[0]

        # We just want ram and cpus for a java app
        assert all(
            cluster.instance.family[0] in ("m", "r", "c")
            for cluster in (rlr_small, rlr_large)
        )


def test_kv_plus_evcache_rps_exceeding_250k():
    consistencies = (
        AccessConsistency.eventual,
        AccessConsistency.best_effort,
        AccessConsistency.read_your_writes,
    )

    # If RPS > 250,000 then capacity planner will determine that ev_cache
    # should be provisioned
    rps = 275_000
    wps = 10_000

    for consistency in consistencies:
        desires = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                # Target consistency must be eventual or best_effort for
                # ev_cache to be included
                access_consistency=GlobalConsistency(
                    same_region=Consistency(target_consistency=consistency)
                ),
                estimated_read_per_second=Interval(
                    low=rps // 10, mid=rps, high=rps * 10, confidence=0.98
                ),
                estimated_write_per_second=Interval(
                    low=wps // 10, mid=wps, high=wps * 10, confidence=0.98
                ),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=Interval(
                    low=20, mid=200, high=2000, confidence=0.98
                ),
            ),
        )

        cap_plan = planner.plan(
            model_name="org.netflix.key-value",
            region="us-east-1",
            desires=desires,
            simulations=256,
        )

        least_regret_clusters = cap_plan.least_regret[0].candidate_clusters

        # Check the C* cluster
        zlr_cass = next(
            cluster
            for cluster in least_regret_clusters.zonal
            if cluster.cluster_type == "cassandra"
        )
        if zlr_cass.instance.drive is None:
            assert sum(dr.size_gib for dr in zlr_cass.attached_drives) >= 100
        else:
            assert zlr_cass.instance.drive.size_gib >= 100

        # We should generally want cheap CPUs for Cassandra
        assert zlr_cass.instance.family[0] in ("r", "m", "c")

        # The KV cluster should be the only regional cluster
        assert len(least_regret_clusters.regional) == 1

        # Check the Java cluster
        rlr = least_regret_clusters.regional[0]

        # We should never be paying for ephemeral drives
        assert rlr.instance.drive is None

        # We just want ram and cpus for a java app
        assert rlr.instance.family[0] in ("m", "r", "c")

        # Check the EVCache cluster
        zlr_evs = [
            cluster
            for cluster in least_regret_clusters.zonal
            if cluster.cluster_type == "evcache"
        ]
        if consistency not in (
            AccessConsistency.eventual,
            AccessConsistency.best_effort,
        ):
            # Since the consistency is not either eventual or best_effort, there
            # should not have been an EVCache cluster included in the plan.
            assert len(zlr_evs) == 0
        else:
            zlr_ev = zlr_evs[0]
            if zlr_ev.instance.drive is not None:
                # If we end up with disk we want at least 100 GiB of disk per zone
                assert zlr_ev.count * zlr_ev.instance.drive.size_gib > 100
            else:
                # If we end up with RAM we want at least 100 GiB of ram per zone
                assert zlr_ev.count * zlr_ev.instance.ram_gib > 100

            # We should generally want cheap CPUs for EVCache
            assert zlr_ev.instance.family[0] in ("r", "m", "i")

            # Validate EVCache cost for 300k RPS + 300k WPS
            assert least_regret_clusters.annual_costs["evcache.zonal-clusters"] < 30000

            # Costs for KV + C* + EVCache clusters, including networking for C*
            assert len(least_regret_clusters.annual_costs.keys()) == 7


def test_kv_plus_evcache_rps_exceeding_100k_and_sufficient_read_write_ratio():
    consistencies = (
        AccessConsistency.eventual,
        AccessConsistency.best_effort,
        AccessConsistency.read_your_writes,
    )

    # If RPS > 100,000 and RPS/WPS > 90% then capacity planner will determine
    # that ev_cache should be provisioned
    # Using same value for RPS and WPS, so RPS/WPS = 100%
    qps = 150_000

    for consistency in consistencies:
        desires = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                # Target consistency must be eventual or best_effort for
                # ev_cache to be included
                access_consistency=GlobalConsistency(
                    same_region=Consistency(target_consistency=consistency)
                ),
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
                estimated_working_set_percent=certain_float(0.10),
            ),
        )

        cap_plan = planner.plan(
            model_name="org.netflix.key-value",
            region="us-east-1",
            desires=desires,
            simulations=256,
        )

        least_regret_clusters = cap_plan.least_regret[0].candidate_clusters

        # Check the C* cluster
        zlr_cass = next(
            cluster
            for cluster in least_regret_clusters.zonal
            if cluster.cluster_type == "cassandra"
        )
        if zlr_cass.instance.drive is None:
            assert sum(dr.size_gib for dr in zlr_cass.attached_drives) >= 100
        else:
            assert zlr_cass.instance.drive.size_gib >= 100

        # We should generally want cheap CPUs for Cassandra
        assert zlr_cass.instance.family[0] in ("r", "m", "c")

        # The KV cluster should be the only regional cluster
        assert len(least_regret_clusters.regional) == 1

        # Check the Java cluster
        rlr = least_regret_clusters.regional[0]

        # We should never be paying for ephemeral drives
        assert rlr.instance.drive is None

        # We just want ram and cpus for a java app
        assert rlr.instance.family[0] in ("m", "r", "c")

        # Check the EVCache cluster
        zlr_evs = [
            cluster
            for cluster in least_regret_clusters.zonal
            if cluster.cluster_type == "evcache"
        ]
        if consistency not in (
            AccessConsistency.eventual,
            AccessConsistency.best_effort,
        ):
            # Since the consistency is not either eventual or best_effort, there
            # should not have been an EVCache cluster included in the plan.
            assert len(zlr_evs) == 0
        else:
            zlr_ev = zlr_evs[0]
            if zlr_ev.instance.drive is not None:
                # If we end up with disk we want at least 20 GiB of disk per zone
                assert zlr_ev.count * zlr_ev.instance.drive.size_gib > 20
            else:
                # If we end up with RAM we want at least 20 GiB of ram per zone
                assert zlr_ev.count * zlr_ev.instance.ram_gib > 20

            # We should generally want cheap CPUs for EVCache
            assert zlr_ev.instance.family[0] in ("r", "m", "i")

            # Validate EVCache cost for 300k RPS + 300k WPS
            assert least_regret_clusters.annual_costs["evcache.zonal-clusters"] < 30000

            # Costs for KV + C* + EVCache clusters, including networking for C*
            assert len(least_regret_clusters.annual_costs.keys()) == 7


def test_kv_rps_exceeding_100k_but_insufficient_read_write_ratio():
    consistencies = (
        AccessConsistency.eventual,
        AccessConsistency.best_effort,
        AccessConsistency.read_your_writes,
    )
    # RPS > 100,000 but RPS/WPS <= 90%, so ev_cache should not be provisioned
    rps = 150_000
    wps = 200_000

    for consistency in consistencies:
        desires = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                # Target consistency must be eventual or best_effort for
                # ev_cache to be included
                access_consistency=GlobalConsistency(
                    same_region=Consistency(target_consistency=consistency)
                ),
                estimated_read_per_second=Interval(
                    low=rps // 10, mid=rps, high=rps * 10, confidence=0.98
                ),
                estimated_write_per_second=Interval(
                    low=wps // 10, mid=wps, high=wps * 10, confidence=0.98
                ),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=Interval(
                    low=20, mid=200, high=2000, confidence=0.98
                ),
            ),
        )

        cap_plan = planner.plan(
            model_name="org.netflix.key-value",
            region="us-east-1",
            desires=desires,
            simulations=256,
        )

        least_regret_clusters = cap_plan.least_regret[0].candidate_clusters

        # Check the C* cluster
        zlr_cass = next(
            cluster
            for cluster in least_regret_clusters.zonal
            if cluster.cluster_type == "cassandra"
        )
        if zlr_cass.instance.drive is None:
            assert sum(dr.size_gib for dr in zlr_cass.attached_drives) >= 100
        else:
            assert zlr_cass.instance.drive.size_gib >= 100

        # We should generally want cheap CPUs for Cassandra
        assert zlr_cass.instance.family[0] in ("r", "m", "c")

        # The KV cluster should be the only regional cluster
        assert len(least_regret_clusters.regional) == 1

        # Check the Java cluster
        rlr = least_regret_clusters.regional[0]

        # We should never be paying for ephemeral drives
        assert rlr.instance.drive is None

        # We just want ram and cpus for a java app, no drives
        assert rlr.instance.drive is None

        # Validate that there is no EVCache cluster
        assert not any(
            cluster
            for cluster in least_regret_clusters.zonal
            if cluster.cluster_type == "evcache"
        )


def test_kv_plus_evcache_configured_read_write_ratio_threshold():
    consistencies = (
        AccessConsistency.eventual,
        AccessConsistency.best_effort,
        AccessConsistency.read_your_writes,
    )

    # R/W ratio = 0.75
    rps = 150_000
    wps = 200_000

    for consistency in consistencies:
        desires = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                # Target consistency must be eventual or best_effort
                # for ev_cache to be included
                access_consistency=GlobalConsistency(
                    same_region=Consistency(target_consistency=consistency)
                ),
                estimated_read_per_second=Interval(
                    low=rps // 10, mid=rps, high=rps * 10, confidence=0.98
                ),
                estimated_write_per_second=Interval(
                    low=wps // 10, mid=wps, high=wps * 10, confidence=0.98
                ),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=Interval(
                    low=20, mid=200, high=2000, confidence=0.98
                ),
            ),
        )

        cap_plan = planner.plan(
            model_name="org.netflix.key-value",
            region="us-east-1",
            desires=desires,
            simulations=256,
            # Configuring threshold to be less than the R/W ratio above,
            # which should cause this check to succeed.
            extra_model_arguments={"kv_evcache_read_write_ratio_threshold": 0.7},
        )

        least_regret_clusters = cap_plan.least_regret[0].candidate_clusters

        # Check the C* cluster
        zlr_cass = next(
            cluster
            for cluster in least_regret_clusters.zonal
            if cluster.cluster_type == "cassandra"
        )
        if zlr_cass.instance.drive is None:
            assert sum(dr.size_gib for dr in zlr_cass.attached_drives) >= 100
        else:
            assert zlr_cass.instance.drive.size_gib >= 100

        # We should generally want cheap CPUs for Cassandra
        assert zlr_cass.instance.family[0] in ("r", "m", "c")

        # The KV cluster should be the only regional cluster
        assert len(least_regret_clusters.regional) == 1

        # Check the Java cluster
        rlr = least_regret_clusters.regional[0]

        # We should never be paying for ephemeral drives
        assert rlr.instance.drive is None

        # Validate that EVCache is included if consistency is valid
        if consistency in (AccessConsistency.eventual, AccessConsistency.best_effort):
            assert any(
                cluster
                for cluster in least_regret_clusters.zonal
                if cluster.cluster_type == "evcache"
            )
        else:
            assert not any(
                cluster
                for cluster in least_regret_clusters.zonal
                if cluster.cluster_type == "evcache"
            )


def test_kv_plus_evcache_high_hit_rate():
    # If RPS > 250,000 then capacity planner will determine that
    # ev_cache should be provisioned
    rps = 275_000
    wps = 10_000

    eventual = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            # Target consistency must be eventual or best_effort for ev_cache
            # to be included
            access_consistency=GlobalConsistency(
                same_region=Consistency(target_consistency=AccessConsistency.eventual)
            ),
            estimated_read_per_second=Interval(
                low=rps // 10, mid=rps, high=rps * 10, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=wps // 10, mid=wps, high=wps * 10, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=20, mid=200, high=2000, confidence=0.98
            ),
        ),
    )
    read_your_writes = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            # Target consistency is not eventual or best_effort, so ev_cache
            # should not be included
            access_consistency=GlobalConsistency(
                same_region=Consistency(
                    target_consistency=AccessConsistency.read_your_writes
                )
            ),
            estimated_read_per_second=Interval(
                low=rps // 10, mid=rps, high=rps * 10, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=wps // 10, mid=wps, high=wps * 10, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=20, mid=200, high=2000, confidence=0.98
            ),
        ),
    )

    cap_plan_eventual = planner.plan(
        model_name="org.netflix.key-value",
        region="us-east-1",
        desires=eventual,
        simulations=256,
    )
    cap_plan_ryw = planner.plan(
        model_name="org.netflix.key-value",
        region="us-east-1",
        desires=read_your_writes,
        simulations=256,
    )

    least_regret_clusters_eventual = cap_plan_eventual.least_regret[
        0
    ].candidate_clusters
    least_regret_clusters_ryw = cap_plan_ryw.least_regret[0].candidate_clusters

    # Check the C* cluster
    zlr_cass_eventual = next(
        cluster
        for cluster in least_regret_clusters_eventual.zonal
        if cluster.cluster_type == "cassandra"
    )
    zlr_cass_ryw = next(
        cluster
        for cluster in least_regret_clusters_ryw.zonal
        if cluster.cluster_type == "cassandra"
    )

    assert zlr_cass_eventual.annual_cost < zlr_cass_ryw.annual_cost

    # We should generally want cheap CPUs for Cassandra
    assert zlr_cass_eventual.instance.family[0] in ("r", "m", "c")
    assert zlr_cass_ryw.instance.family[0] in ("r", "m", "c")

    # The KV cluster should be the only regional cluster
    assert len(least_regret_clusters_eventual.regional) == 1
    assert len(least_regret_clusters_ryw.regional) == 1

    # Check the Java cluster
    rlr_eventual = least_regret_clusters_eventual.regional[0]
    rlr_ryw = least_regret_clusters_ryw.regional[0]

    # We should never be paying for ephemeral drives
    assert rlr_eventual.instance.drive is None
    assert rlr_ryw.instance.drive is None

    # We shouldn't pay for disks
    assert rlr_eventual.instance.drive is None
    assert rlr_ryw.instance.drive is None

    # For read-your-writes consistency, there should be no EVCache cluster.
    assert not any(
        cluster
        for cluster in least_regret_clusters_ryw.zonal
        if cluster.cluster_type == "evcache"
    )

    # For eventual consistency, verify that EVCache cluster exists and
    # validate it.
    zlr_ev = next(
        cluster
        for cluster in least_regret_clusters_eventual.zonal
        if cluster.cluster_type == "evcache"
    )
    if zlr_ev.instance.drive is not None:
        # If we end up with disk we want at least 100 GiB of disk per zone
        assert zlr_ev.count * zlr_ev.instance.drive.size_gib > 100
    else:
        # If we end up with RAM we want at least 100 GiB of ram per zone
        assert zlr_ev.count * zlr_ev.instance.ram_gib > 100

    # We should generally want cheap CPUs for EVCache
    assert zlr_ev.instance.family[0] in ("r", "m", "c")

    # Plan with EVCache should be cheaper than plan without it, since the
    # assumed hit rate is high (default of 0.8).
    assert (
        least_regret_clusters_eventual.total_annual_cost
        < least_regret_clusters_ryw.total_annual_cost
    )


def test_kv_plus_evcache_low_hit_rate():
    # If RPS > 250,000 then capacity planner will determine that ev_cache
    # should be provisioned
    rps = 275_000
    wps = 10_000

    eventual = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            # Target consistency must be eventual or best_effort for
            # ev_cache to be included
            access_consistency=GlobalConsistency(
                same_region=Consistency(target_consistency=AccessConsistency.eventual)
            ),
            estimated_read_per_second=Interval(
                low=rps // 10, mid=rps, high=rps * 10, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=wps // 10, mid=wps, high=wps * 10, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=20, mid=200, high=2000, confidence=0.98
            ),
        ),
    )
    read_your_writes = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            # Target consistency is not eventual or best_effort,
            # so ev_cache should not be included
            access_consistency=GlobalConsistency(
                same_region=Consistency(
                    target_consistency=AccessConsistency.read_your_writes
                )
            ),
            estimated_read_per_second=Interval(
                low=rps // 10, mid=rps, high=rps * 10, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=wps // 10, mid=wps, high=wps * 10, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=20, mid=200, high=2000, confidence=0.98
            ),
        ),
    )

    # Intentionally configuring lower cache hit rate estimation
    # (as compared to default of 0.8).
    cap_plan_eventual = planner.plan(
        model_name="org.netflix.key-value",
        region="us-east-1",
        desires=eventual,
        simulations=256,
        extra_model_arguments={"estimated_kv_cache_hit_rate": 0.2},
    )
    cap_plan_ryw = planner.plan(
        model_name="org.netflix.key-value",
        region="us-east-1",
        desires=read_your_writes,
        simulations=256,
        extra_model_arguments={"estimated_kv_cache_hit_rate": 0.2},
    )

    least_regret_clusters_eventual = cap_plan_eventual.least_regret[
        0
    ].candidate_clusters
    least_regret_clusters_ryw = cap_plan_ryw.least_regret[0].candidate_clusters

    # Check the C* cluster
    zlr_cass_eventual = next(
        cluster
        for cluster in least_regret_clusters_eventual.zonal
        if cluster.cluster_type == "cassandra"
    )
    zlr_cass_ryw = next(
        cluster
        for cluster in least_regret_clusters_ryw.zonal
        if cluster.cluster_type == "cassandra"
    )

    # There's a chance that the two C* clusters might actually be the same.
    assert zlr_cass_eventual.annual_cost <= zlr_cass_ryw.annual_cost

    # We should generally want cheap CPUs for Cassandra
    assert zlr_cass_eventual.instance.family[0] in ("r", "m", "c")
    assert zlr_cass_ryw.instance.family[0] in ("r", "m", "c")

    # The KV cluster should be the only regional cluster
    assert len(least_regret_clusters_eventual.regional) == 1
    assert len(least_regret_clusters_ryw.regional) == 1

    # Check the Java cluster
    rlr_eventual = least_regret_clusters_eventual.regional[0]
    rlr_ryw = least_regret_clusters_ryw.regional[0]

    # We should never be paying for ephemeral drives
    assert rlr_eventual.instance.drive is None
    assert rlr_ryw.instance.drive is None

    # We just want ram and cpus for a java app
    assert rlr_eventual.instance.drive is None
    assert rlr_ryw.instance.drive is None

    # For read-your-writes consistency, there should be no EVCache cluster.
    assert not any(
        cluster
        for cluster in least_regret_clusters_ryw.zonal
        if cluster.cluster_type == "evcache"
    )

    # For eventual consistency, verify that EVCache cluster exists
    # and validate it.
    zlr_ev = next(
        cluster
        for cluster in least_regret_clusters_eventual.zonal
        if cluster.cluster_type == "evcache"
    )
    if zlr_ev.instance.drive is not None:
        # If we end up with disk we want at least 100 GiB of disk per zone
        assert zlr_ev.count * zlr_ev.instance.drive.size_gib > 100
    else:
        # If we end up with RAM we want at least 100 GiB of ram per zone
        assert zlr_ev.count * zlr_ev.instance.ram_gib > 100

    # We should generally want cheap CPUs for EVCache
    assert zlr_ev.instance.family[0] in ("r", "m", "c")

    # Plan with EVCache should be more expensive than plan without it,
    # since the assumed hit rate is very low.
    # This is because the C* clusters should cost roughly the same,
    # but then EVCache is added cost on top.
    assert (
        least_regret_clusters_eventual.total_annual_cost
        > least_regret_clusters_ryw.total_annual_cost
    )

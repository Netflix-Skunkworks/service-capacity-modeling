import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern


def create_interval(low, mid, high, confidence=0.98):
    return Interval(low=low, mid=mid, high=high, confidence=confidence)


def get_cluster_plan(cap_plan, cluster_type):
    return next(
        filter(
            lambda c: c.cluster_type == cluster_type,
            cap_plan.least_regret[0].candidate_clusters.regional,
        )
    )


def create_capacity_desires(qps, data_shape_fn):
    return CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=create_interval(qps // 10, qps, qps * 10),
            estimated_write_per_second=create_interval(qps // 10, qps, qps * 10),
            estimated_mean_write_size_bytes=create_interval(
                1024, 1024 * 10, 1024 * 100
            ),
        ),
        data_shape=data_shape_fn(qps),
    )


class TestGraphKVIncreasingQPS:
    QPS_VALUES = (100, 1000, 10_000, 100_000)

    @pytest.fixture(
        params=[
            pytest.param(
                (
                    "simple",
                    lambda qps: DataShape(
                        estimated_state_item_count=create_interval(
                            1000000, 10000000, 100000000
                        )
                    ),
                ),
                id="with_item_count",
            ),
            pytest.param(
                (
                    "item_count_unset",
                    lambda qps: DataShape(
                        estimated_state_size_gib=create_interval(10, 100, 1000)
                    ),
                ),
                id="with_state_size",
            ),
        ]
    )
    def data_shape_config(self, request):
        return request.param

    def test_graphkv_increasing_qps(self, data_shape_config):
        _, data_shape_fn = data_shape_config
        graphkv_results_trend = []
        kv_results_trend = []

        for qps in self.QPS_VALUES:
            cap_plan = planner.plan(
                model_name="org.netflix.graphkv",
                region="us-east-1",
                desires=create_capacity_desires(qps, data_shape_fn),
                simulations=256,
            )

            # Verify cluster types
            types = {
                c.cluster_type
                for c in list(cap_plan.least_regret[0].candidate_clusters.regional)
                + list(cap_plan.least_regret[0].candidate_clusters.zonal)
            }
            assert sorted(types) == ["cassandra", "dgwgraphkv", "dgwkv"]

            graphkv_plan = get_cluster_plan(cap_plan, "dgwgraphkv")
            kv_plan = get_cluster_plan(cap_plan, "dgwkv")

            # TODO: Add a comprehensive check based on request amplification
            # Account for resource amplification
            assert (
                kv_plan.count > graphkv_plan.count
                or kv_plan.instance.cpu > graphkv_plan.instance.cpu
            )

            kv_results_trend.append((kv_plan.count * kv_plan.instance.cpu,))
            graphkv_results_trend.append(
                (graphkv_plan.count * graphkv_plan.instance.cpu,)
            )

            for plan in (graphkv_plan, kv_plan):
                assert plan.instance.family[0] in ("m", "r")
                assert plan.instance.drive is None

        # Verify capacity increases
        for results in (graphkv_results_trend, kv_results_trend):
            x = [r[0] for r in results]
            assert x[0] < x[-1]
            assert sorted(x) == x

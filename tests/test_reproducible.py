from decimal import Decimal

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.interface import SampleRef
from service_capacity_modeling.interface import ServiceCapacity
from service_capacity_modeling.regret_explainability import MergedRegretCandidate
from service_capacity_modeling.regret_explainability import (
    merge_regret_candidates_positional,
)
from service_capacity_modeling.regret_explainability import plan_signature
from service_capacity_modeling.regret_explainability import RegretCandidate
from service_capacity_modeling.regret_explainability import (
    summaries_for_least_regret,
)
from service_capacity_modeling.regret_explainability import summarize_regret_candidates


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


def test_repeated_plans():
    results = []
    for _ in range(5):
        results.append(
            planner.plan(
                model_name="org.netflix.cassandra",
                region="us-east-1",
                desires=uncertain_mid,
            ).model_dump_json()
        )

    a = [hash(x) for x in results]
    # We should end up with consistent results
    assert all(i == a[0] for i in a)


def test_compositional():
    """Test that key-value composition produces identical Cassandra plans.

    The key-value model composes with Cassandra via `lambda x: x` (identity),
    meaning the Cassandra sub-model must receive identical inputs and produce
    byte-for-byte identical outputs. This is the strictest possible test of
    compositional correctness.

    Note: The final least_regret results may differ due to reduce_by_family()
    filtering across both regional and zonal dimensions, but that is a
    presentation concern - the underlying Cassandra planning must be identical.
    """
    direct_result = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=uncertain_mid,
        num_results=4,
    )
    composed_result = planner.plan(
        model_name="org.netflix.key-value",
        region="us-east-1",
        desires=uncertain_mid,
        num_results=4,
    )

    # Strictest test: Cassandra regret clusters must be EXACTLY identical
    # (same plans, same regrets, same order) since key-value uses `lambda x: x`
    direct_cass = direct_result.explanation.regret_clusters_by_model[
        "org.netflix.cassandra"
    ]
    composed_cass = composed_result.explanation.regret_clusters_by_model[
        "org.netflix.cassandra"
    ]
    assert len(direct_cass) == len(composed_cass)
    for i, ((d_plan, _, d_regret), (c_plan, _, c_regret)) in enumerate(
        zip(direct_cass, composed_cass)
    ):
        assert d_plan == c_plan, f"Plan {i} differs"
        assert d_regret == c_regret, f"Regret {i} differs: {d_regret} vs {c_regret}"

    # Verify the composed results have the expected structure
    for lr in composed_result.least_regret:
        # Zonal cluster should be Cassandra
        assert lr.candidate_clusters.zonal[0].cluster_type == "cassandra"
        # Regional cluster should be the key-value Java app
        java = lr.candidate_clusters.regional[0]
        assert java.cluster_type == "dgwkv"
        # Sanity check on Java app sizing (~48 total CPUs: 6 x 8 vCPU instances,
        # but may vary with CPU architecture or pricing improvements)
        assert 100 > java.count * java.instance.cpu > 20


def test_multiple_options_diversify_with_more_simulations():
    """
    This test appears strange at first. The goal is to show that with less
    simulations we would see a smaller subset of the diverse sample of outputs
    than with more simulations
    """

    # These values happen to work today but may not work in the future with
    # changes to the CP inputs (instances, costs, performance).
    # Feel free to change the numbers as long as it fits the below assertion
    arbitrary_num_results = 12
    arbitrary_small_number = 12
    arbitrary_large_number = 1024
    assert arbitrary_small_number < arbitrary_large_number

    less_simulations_result = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=uncertain_mid,
        num_results=arbitrary_num_results,
        simulations=arbitrary_small_number,
    )
    more_simulations_result = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=uncertain_mid,
        num_results=arbitrary_num_results,
        simulations=arbitrary_large_number,
    )

    # Potentially brittle assertion. This is the part likely to break
    # The idea is that we should see more options with more simulations.
    less_simulations_famlies = {
        lr.candidate_clusters.zonal[0].instance.family
        for lr in less_simulations_result.least_regret
    }
    more_simulations_families = {
        lr.candidate_clusters.zonal[0].instance.family
        for lr in more_simulations_result.least_regret
    }
    assert len(less_simulations_famlies) < len(more_simulations_families)

    expected_family_types = {"i", "r", "c", "m"}
    for f in less_simulations_famlies:
        assert f[0] in expected_family_types
    for f in more_simulations_families:
        assert f[0] in expected_family_types


def test_composed_explained_samples_exist_in_all_models():
    explained = planner.plan_explained(
        model_name="org.netflix.key-value",
        region="us-east-1",
        desires=uncertain_mid,
        num_results=4,
        simulations=12,
    )

    assert len(explained.least_regret_summaries) == len(explained.plan.least_regret), (
        f"Every least_regret plan must have a summary. "
        f"Got {len(explained.least_regret_summaries)} summaries "
        f"for {len(explained.plan.least_regret)} plans."
    )
    for plan, summary in zip(
        explained.plan.least_regret, explained.least_regret_summaries
    ):
        assert plan_signature(summary.plan) == plan_signature(plan)

    expected_model_names = set(explained.plan.explanation.regret_clusters_by_model)
    assert expected_model_names

    for summary in explained.least_regret_summaries:
        assert set(summary.mean_regret_components_by_model) == expected_model_names
        for example_sample in summary.example_samples:
            assert example_sample.sample_id
            assert example_sample.sample_label

    selected_signatures = {
        plan_signature(summary.plan) for summary in explained.least_regret_summaries
    }
    alternative_signatures = {
        plan_signature(summary.plan) for summary in explained.considered_alternatives
    }
    assert explained.considered_alternatives
    assert len(explained.considered_alternatives) <= 4
    assert selected_signatures.isdisjoint(alternative_signatures)
    for summary in explained.considered_alternatives:
        assert set(summary.mean_regret_components_by_model) == expected_model_names


def test_plan_explained_preserves_plan_output():
    plain = planner.plan(
        model_name="org.netflix.key-value",
        region="us-east-1",
        desires=uncertain_mid,
        num_results=4,
        simulations=12,
    )
    explained = planner.plan_explained(
        model_name="org.netflix.key-value",
        region="us-east-1",
        desires=uncertain_mid,
        num_results=4,
        simulations=12,
    )

    assert explained.plan.least_regret == plain.least_regret
    assert explained.plan.requirements == plain.requirements
    assert explained.plan.mean == plain.mean
    assert explained.plan.percentiles == plain.percentiles


def test_least_regret_summary_lookup_rejects_missing_plan():
    plain = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=uncertain_mid,
        num_results=1,
        simulations=2,
    )

    with pytest.raises(RuntimeError, match="Missing regret summaries"):
        summaries_for_least_regret(plain.least_regret, {})


def test_composed_regret_merge_rejects_unequal_model_counts():
    plan = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=uncertain_mid,
        num_results=1,
        simulations=2,
    ).least_regret[0]
    candidate = RegretCandidate(
        sample=SampleRef(sample_id="s-1", sample_label="first"),
        plan=plan,
        desires=uncertain_mid,
        total_regret=1.0,
    )

    with pytest.raises(RuntimeError, match="invalid candidate counts"):
        merge_regret_candidates_positional(
            regret_details_by_model={"a": [candidate], "b": []},
            zonal_requirements={},
            regional_requirements={},
        )


def test_composed_regret_merge_keeps_all_component_samples():
    plan = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=uncertain_mid,
        num_results=1,
        simulations=2,
    ).least_regret[0]
    first = RegretCandidate(
        sample=SampleRef(sample_id="s-1", sample_label="first"),
        plan=plan,
        desires=uncertain_mid,
        total_regret=1.0,
    )
    second = RegretCandidate(
        sample=SampleRef(sample_id="s-2", sample_label="second"),
        plan=plan,
        desires=uncertain_mid,
        total_regret=2.0,
    )

    merged = merge_regret_candidates_positional(
        regret_details_by_model={"a": [first], "b": [second]},
        zonal_requirements={},
        regional_requirements={},
    )

    assert [sample.sample_id for sample in merged[0].samples] == ["s-1", "s-2"]


def test_plan_signature_keeps_service_parameters():
    def plan_with_param(value: str) -> CapacityPlan:
        return CapacityPlan(
            requirements=Requirements(),
            candidate_clusters=Clusters(
                annual_costs={"service": Decimal("1.0")},
                services=[
                    ServiceCapacity(
                        service_type="ddb",
                        annual_cost=1.0,
                        service_params={"capacity": value},
                    )
                ],
            ),
        )

    assert plan_signature(plan_with_param("small")) != plan_signature(
        plan_with_param("large")
    )


def test_regret_summary_accumulates_samples_and_components():
    plan = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=uncertain_mid,
        num_results=1,
        simulations=2,
    ).least_regret[0]
    plan_with_different_params = plan.model_copy(deep=True)
    plan_with_different_params.candidate_clusters.zonal[0].cluster_params = {
        **plan_with_different_params.candidate_clusters.zonal[0].cluster_params,
        "test_signature_marker": "different",
    }

    summaries = summarize_regret_candidates(
        [
            MergedRegretCandidate(
                samples=[SampleRef(sample_id="s-1", sample_label="first")],
                plan=plan,
                total_regret=10.0,
                regret_components_by_model={
                    "a": {"cost": 2.0},
                    "b": {"disk": 3.0},
                },
            ),
            MergedRegretCandidate(
                samples=[SampleRef(sample_id="s-2", sample_label="second")],
                plan=plan,
                total_regret=4.0,
                regret_components_by_model={
                    "a": {"cost": 4.0},
                    "b": {"disk": 5.0},
                },
            ),
            MergedRegretCandidate(
                samples=[SampleRef(sample_id="s-3", sample_label="third")],
                plan=plan_with_different_params,
                total_regret=1.0,
                regret_components_by_model={"a": {"cost": 1.0}},
            ),
        ]
    )

    assert plan_signature(plan) != plan_signature(plan_with_different_params)
    assert len(summaries) == 2

    summary = summaries[plan_signature(plan)]
    assert summary.sample_count == 2
    assert summary.selected_total_regret == 10.0
    assert summary.mean_total_regret == 7.0
    assert summary.mean_regret_components_by_model == {
        "a": {"cost": 3.0},
        "b": {"disk": 4.0},
    }
    assert [sample.sample_id for sample in summary.example_samples] == ["s-1", "s-2"]

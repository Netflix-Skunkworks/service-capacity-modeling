"""Tests for the model-independent parts of interchangeable capacity plans."""

from decimal import Decimal

import pytest

from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.interface import ServiceCapacity
from service_capacity_modeling.models.interchangeable import dominance_edges
from service_capacity_modeling.models.interchangeable import DominanceEdge
from service_capacity_modeling.models.interchangeable import (
    with_preferred_family_penalty,
)

# Model rank 1,100: 1,000 of penalized compute plus a 100 service.
MODEL_RANKED = CapacityPlan(
    requirements=Requirements(),
    candidate_clusters=Clusters(
        annual_costs={"svc": Decimal(100)},
        services=[ServiceCapacity(service_type="svc", annual_cost=100)],
    ),
    rank=1_100,
)


def test_preferred_family_penalty_inflates_only_compute_outside_preferred():
    preferred = frozenset({"m6id"})
    outside = with_preferred_family_penalty(MODEL_RANKED, "r6id", preferred, 0.15)
    assert outside.rank == pytest.approx(1_250)
    for family, families in (("m6id", preferred), ("r6id", None)):
        plan = with_preferred_family_penalty(MODEL_RANKED, family, families, 0.15)
        assert plan.rank == 1_100


def test_dominance_edges_are_the_transitive_reduction_of_layout_inclusion():
    served = {
        "small": frozenset({3}),
        "medium": frozenset({2, 3}),
        "large": frozenset({1, 2, 3}),
        "disk_heavy": frozenset({0, 3}),
    }
    # disk_heavy serves layout 0, which large does not, so they are incomparable.
    assert dominance_edges(served) == [
        DominanceEdge(weaker="medium", stronger="large"),
        DominanceEdge(weaker="small", stronger="disk_heavy"),
        DominanceEdge(weaker="small", stronger="medium"),
    ]


def test_dominance_edges_break_exact_ties_by_name():
    served = {"a": frozenset({1}), "b": frozenset({1})}
    assert dominance_edges(served) == [DominanceEdge(weaker="a", stronger="b")]

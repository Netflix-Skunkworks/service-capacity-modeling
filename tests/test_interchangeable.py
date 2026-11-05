"""Tests for the model-independent parts of interchangeable capacity plans."""

from service_capacity_modeling.models.interchangeable import dominance_edges
from service_capacity_modeling.models.interchangeable import DominanceEdge


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

# pylint: disable=protected-access
from typing import Any
from typing import Sequence

from service_capacity_modeling import capacity_planner
from service_capacity_modeling.capacity_planner import _CertainResult
from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import Consistency
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import GlobalConsistency
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.utils import resolve_instance_filter_allowlist


def _kv_desires() -> CapacityDesires:
    return CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            access_consistency=GlobalConsistency(
                same_region=Consistency(
                    target_consistency=AccessConsistency.eventual,
                ),
                cross_region=Consistency(
                    target_consistency=AccessConsistency.best_effort,
                ),
            ),
            estimated_read_per_second=certain_int(100_000),
            estimated_write_per_second=certain_int(10_000),
            estimated_mean_read_latency_ms=certain_float(1),
            estimated_mean_write_latency_ms=certain_float(1),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=certain_int(100),
        ),
    )


def _record_plan_certain_calls(monkeypatch):
    calls: list[dict[str, Any]] = []

    def fake_plan_certain(**kwargs):
        calls.append(kwargs)
        return _CertainResult(plans=[])

    monkeypatch.setattr(planner, "_plan_certain", fake_plan_certain)
    return calls


def _filters_by_model(
    calls: list[dict[str, Any]],
) -> dict[str, Sequence[str] | None]:
    return {call["model_name"]: call["instance_families"] for call in calls}


def test_plan_certain_explained_uses_model_scoped_instance_filters(monkeypatch):
    calls = _record_plan_certain_calls(monkeypatch)

    planner.plan_certain_explained(
        model_name="org.netflix.key-value",
        region="us-east-1",
        desires=_kv_desires(),
        extra_model_arguments={"kv_force_evcache": True},
        instance_filters_by_model={"org.netflix.cassandra": ["i4i"]},
    )

    assert _filters_by_model(calls) == {
        "org.netflix.key-value": None,
        "org.netflix.evcache": None,
        "org.netflix.cassandra": ["i4i"],
    }


def test_plan_certain_forwards_model_scoped_instance_filters(monkeypatch):
    calls = _record_plan_certain_calls(monkeypatch)

    planner.plan_certain(
        model_name="org.netflix.key-value",
        region="us-east-1",
        desires=_kv_desires(),
        extra_model_arguments={"kv_force_evcache": True},
        instance_filters_by_model={"org.netflix.cassandra": ["i4i"]},
    )

    assert _filters_by_model(calls) == {
        "org.netflix.key-value": None,
        "org.netflix.evcache": None,
        "org.netflix.cassandra": ["i4i"],
    }


def test_model_scoped_instance_filters_union_with_global_filters(monkeypatch):
    calls = _record_plan_certain_calls(monkeypatch)

    planner.plan_certain_explained(
        model_name="org.netflix.key-value",
        region="us-east-1",
        desires=_kv_desires(),
        extra_model_arguments={"kv_force_evcache": True},
        instance_families=["m6id", "i4i"],
        instance_filters_by_model={"org.netflix.cassandra": ["i4i", "i7i"]},
    )

    assert _filters_by_model(calls) == {
        "org.netflix.key-value": ["m6id", "i4i"],
        "org.netflix.evcache": ["m6id", "i4i"],
        "org.netflix.cassandra": ["m6id", "i4i", "i7i"],
    }


def test_resolve_instance_filter_allowlist_combines_request_and_model_filters():
    assert (
        resolve_instance_filter_allowlist(
            "org.netflix.cassandra",
            None,
            None,
        )
        is None
    )
    assert resolve_instance_filter_allowlist(
        "org.netflix.cassandra",
        ["m6id"],
        {"org.netflix.cassandra": None},
    ) == ["m6id"]
    assert resolve_instance_filter_allowlist(
        "org.netflix.cassandra",
        ["m6id", "i4i"],
        {"org.netflix.cassandra": ["i4i", "i7i"]},
    ) == ["m6id", "i4i", "i7i"]
    assert resolve_instance_filter_allowlist(
        "org.netflix.cassandra",
        ["m6id"],
        {"org.netflix.cassandra": ["i4i"]},
    ) == ["m6id", "i4i"]


def test_model_scoped_instance_filters_add_to_disjoint_global_filters(monkeypatch):
    calls = _record_plan_certain_calls(monkeypatch)

    planner.plan_certain_explained(
        model_name="org.netflix.key-value",
        region="us-east-1",
        desires=_kv_desires(),
        extra_model_arguments={"kv_force_evcache": True},
        instance_families=["m6id"],
        instance_filters_by_model={"org.netflix.cassandra": ["i4i"]},
    )

    assert _filters_by_model(calls) == {
        "org.netflix.key-value": ["m6id"],
        "org.netflix.evcache": ["m6id"],
        "org.netflix.cassandra": ["m6id", "i4i"],
    }


def test_plan_percentiles_uses_model_scoped_instance_filters(monkeypatch):
    calls = _record_plan_certain_calls(monkeypatch)

    planner._plan_percentiles(
        model_name="org.netflix.key-value",
        percentiles=(50,),
        region="us-east-1",
        desires=_kv_desires(),
        extra_model_arguments={"kv_force_evcache": True},
        instance_filters_by_model={"org.netflix.cassandra": ["i4i"]},
    )

    assert {
        (call["model_name"], tuple(call["instance_families"] or ())) for call in calls
    } == {
        ("org.netflix.key-value", ()),
        ("org.netflix.evcache", ()),
        ("org.netflix.cassandra", ("i4i",)),
    }


def test_uncertain_plan_uses_model_scoped_instance_filters(monkeypatch):
    calls = _record_plan_certain_calls(monkeypatch)
    percentile_calls: list[dict[str, Any]] = []

    def fake_plan_percentiles(**kwargs):
        percentile_calls.append(kwargs)
        return [], {}

    monkeypatch.setattr(capacity_planner, "_regret", lambda **kwargs: [])
    monkeypatch.setattr(planner, "_plan_percentiles", fake_plan_percentiles)

    instance_filters_by_model = {"org.netflix.cassandra": ["i4i"]}

    planner.plan(
        model_name="org.netflix.key-value",
        region="us-east-1",
        desires=_kv_desires(),
        simulations=1,
        extra_model_arguments={"kv_force_evcache": True},
        instance_filters_by_model=instance_filters_by_model,
    )

    assert _filters_by_model(calls) == {
        "org.netflix.key-value": None,
        "org.netflix.evcache": None,
        "org.netflix.cassandra": ["i4i"],
    }
    assert percentile_calls[0]["instance_filters_by_model"] is instance_filters_by_model

"""Compose ExplainedPlans + PlanComparisonResult into a renderable summary.

**Experimental** — API may change.

This module is the composition point between explainability (why shapes
were rejected) and plan comparison (how recommended differs from current).
It produces a flat dict that any consumer — UI, gen-AI middleware, Claude
skill — can render without domain knowledge.

Usage::

    from service_capacity_modeling.explanation_summary import summarize

    summary = summarize(explained, baseline=baseline, comparison=comparison)
    # summary is a plain dict, JSON-serializable
"""

from __future__ import annotations

from collections import Counter
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from service_capacity_modeling.explainability import ExplainedPlans
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.models.plan_comparison import PlanComparisonResult


def summarize(
    explained: ExplainedPlans,
    baseline: Optional[CapacityPlan] = None,
    comparison: Optional[PlanComparisonResult] = None,
) -> Dict[str, Any]:
    """Produce a renderable summary from explainability + comparison data.

    Returns a flat dict with:
    - headline: current vs recommended instance, count, savings
    - resources: list of {name, current, recommended, ratio, status}
    - same_family_rejections: why same-family alternatives failed
    - rejection_summary: total + by bottleneck
    - family_traits: derived hardware properties per family
    """
    result: Dict[str, Any] = {}

    recommended = explained.plans[0] if explained.plans else None

    # Headline
    rec_z = (
        recommended.candidate_clusters.zonal[0]
        if recommended and recommended.candidate_clusters.zonal
        else None
    )
    base_z = (
        baseline.candidate_clusters.zonal[0]
        if baseline and baseline.candidate_clusters.zonal
        else None
    )

    headline: Dict[str, Any] = {}
    if baseline is not None and base_z:
        headline["current"] = f"{base_z.count}x {base_z.instance.name}"
        headline["current_cost"] = round(
            float(baseline.candidate_clusters.total_annual_cost)
        )
    if recommended is not None and rec_z:
        headline["recommended"] = f"{rec_z.count}x {rec_z.instance.name}"
        headline["recommended_cost"] = round(
            float(recommended.candidate_clusters.total_annual_cost)
        )
    if baseline is not None and recommended is not None and base_z and rec_z:
        headline["annual_savings"] = round(
            float(baseline.candidate_clusters.total_annual_cost)
            - float(recommended.candidate_clusters.total_annual_cost)
        )
    result["headline"] = headline

    # Resource comparison
    resources: List[Dict[str, Any]] = []
    if comparison:
        for c in comparison.comparisons.values():
            if c.is_equivalent:
                status = "ok"
            elif c.exceeds_lower_bound:
                status = "over_provisioned"
            else:
                status = "under_provisioned"
            resources.append(
                {
                    "name": c.resource.value,
                    "current": round(c.baseline_value),
                    "recommended": round(c.comparison_value),
                    "ratio": round(c.ratio, 2),
                    "status": status,
                }
            )
    result["resources"] = resources

    # Same-family rejections
    result["same_family_rejections"] = [
        {"instance": e.instance, "reason": e.reason}
        for e in explained.excuses
        if "same_family" in e.tags
    ]

    # Rejection summary
    bottleneck_counts: Dict[str, int] = {}
    for b, cnt in Counter(
        str(e.bottleneck) for e in explained.excuses if e.bottleneck
    ).items():
        bottleneck_counts[b] = cnt
    result["rejection_summary"] = {
        "total": len(explained.excuses),
        "by_bottleneck": bottleneck_counts,
    }

    # Family traits (for consumers that want to show hardware context)
    result["family_traits"] = {
        k: v.model_dump() for k, v in explained.family_graph.traits.items()
    }

    return result

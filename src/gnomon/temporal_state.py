"""Compact temporal state for reasoning over paths rather than raw arrays."""

from __future__ import annotations

import math
import statistics
from typing import Any


def build_temporal_state(result: dict[str, Any], *,
                         dossiers: list[dict[str, Any]] | None = None
                         ) -> dict[str, Any]:
    """Summarise only facts derivable from the frozen result and receipts."""
    rows = result.get("primary_forecast") or result.get("forecast") or []
    points = [_point(row) for row in rows]
    finite = [value for value in points if math.isfinite(value)]
    first = finite[0] if finite else None
    last = finite[-1] if finite else None
    change = (last - first) if first is not None and last is not None else None
    widths = [float(row.get("q90")) - float(row.get("q10")) for row in rows
              if _finite(row.get("q10")) and _finite(row.get("q90"))]
    claims = [claim for dossier in dossiers or []
              for claim in dossier.get("claims") or []]
    relations = [str(claim.get("relation")) for claim in claims]
    upward = sum(value == "supports_increase" for value in relations)
    downward = sum(value == "supports_decrease" for value in relations)
    conflicts = []
    if upward and downward:
        conflicts.append("context contains both upward and downward evidence")
    support = str(result.get("support") or "unsupported")
    return {
        "level": {"first": first, "last": last},
        "trend": {"change_over_horizon": change,
                  "direction": ("increasing" if change is not None and change > 0
                                else "decreasing" if change is not None and change < 0
                                else "flat_or_unknown")},
        "volatility": {"median_interval_width": statistics.median(widths)
                       if widths else None,
                       "width_change": widths[-1] - widths[0] if len(widths) > 1 else None},
        "seasonality": result.get("seasonality") or {"status": "not_in_result"},
        "regime": result.get("regime") or {"status": "not_in_result"},
        "primary_shape": {"steps": len(rows), "turning_points": _turns(finite)},
        "uncertainty": {"support": support,
                        "intervals_present": bool(widths)},
        "analogues": [claim.get("mechanism") for claim in claims
                      if claim.get("mechanism")][:3],
        "cross_series": result.get("triage") or {"status": "not_in_result"},
        "evidence": {
            "supporting_claims": len(claims), "conflicts": conflicts,
            "sufficiency": "historically_supported" if support in {
                "supported", "context_trusted"} else
                "conditional_only" if claims else "insufficient_context",
        },
    }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _point(row: dict[str, Any]) -> float:
    value = row.get("q50", row.get("point"))
    return float(value) if _finite(value) else math.nan


def _turns(values: list[float]) -> int:
    signs = []
    for left, right in zip(values, values[1:]):
        delta = right - left
        if delta:
            signs.append(1 if delta > 0 else -1)
    return sum(left != right for left, right in zip(signs, signs[1:]))

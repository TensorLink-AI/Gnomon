"""Deterministic evidence planning for typed temporal questions.

The planner decides what kind of inference a question requests and whether the
execution result contains the evidence needed for that inference.  It does not
choose a numerical answer: the fitted executable remains the sole authority.
"""

from __future__ import annotations

from typing import Any

from .temporal_question import TemporalQuestion


PLANNER_VERSION = "0.1"

_RECOVERY = {
    "rolling_origin_scale_fit": "collect more history or request a shorter horizon",
    "rolling_origin_property_fit": "collect more history or request a shorter horizon",
    "future_phase_evidence": "use a phase-aware predictive model or provide a timed causal hypothesis",
    "positive_transition_evidence": "provide a leading signal or a repeated historical transition",
    "tail_calibration": "collect more tail events or shorten the evaluation horizon",
    "paired_rolling_origin_fit": "collect longer aligned histories for both series",
    "conditional_effect_receipt": "provide prior occurrences with known-at times and outcomes",
    "observed_cycles": "collect at least two complete observed cycles",
}

_PREDICTIVE_REQUIREMENTS = {
    "level": ("immutable_forecast_projection",),
    "trend": ("rolling_origin_property_fit",),
    "seasonality": ("future_phase_evidence",),
    "volatility": ("rolling_origin_scale_fit",),
    "regime": ("positive_transition_evidence",),
    "extreme": ("tail_calibration",),
    "dependence": ("paired_rolling_origin_fit",),
}

_OBSERVED_REQUIREMENTS = {
    "level": ("two_observed_windows",),
    "trend": ("two_observed_windows",),
    "seasonality": ("observed_cycles",),
    "volatility": ("detrended_observed_windows",),
    "regime": ("observed_changepoint_evidence",),
    "extreme": ("observed_tail_evidence",),
    "dependence": ("paired_observations",),
}


def inference_mode(question: TemporalQuestion) -> str:
    """Classify requested authority independently of answer contents."""
    if question.context_policy == "scenario":
        return "conditional"
    if question.verb in {"predict", "decide"} or question.horizon is not None:
        return "predictive"
    if question.verb in {"describe", "detect"}:
        return "observed"
    comparison = question.comparison or {}
    names = " ".join(str(value).lower() for value in comparison.values())
    return "predictive" if any(word in names for word in
                               ("future", "forecast", "horizon")) else "observed"


def _evidence_rows(result: dict[str, Any],
                   observed_evidence: dict[str, Any] | None = None,
                   ) -> list[dict[str, Any]]:
    answer = result.get("answer") or {}
    executable = answer.get("executable") or {}
    kind = str(executable.get("kind") or (
        "fitted_volatility" if answer.get("property_distribution") else
        "fitted_executable"))
    rows: list[dict[str, Any]] = []
    if executable:
        rows.append({
            "kind": kind,
            "direction": answer.get("direction"),
            "support": answer.get("support"),
            "folds": ((answer.get("property_distribution") or {}).get("folds")
                      or (result.get("calibration") or {}).get("folds")),
        })
    if observed_evidence:
        estimate = observed_evidence.get("estimate")
        if isinstance(estimate, dict):
            estimate = {key: estimate[key] for key in (
                "phase_shift_steps", "strength_ratio", "template_correlation")
                        if key in estimate}
        rows.append({
            "kind": "observed_transition",
            "direction": observed_evidence.get("direction"),
            "support": observed_evidence.get("support"),
            "identifiable": observed_evidence.get("identifiable"),
            "estimate": estimate,
            "window_steps": (observed_evidence.get("diagnostics") or {}).get(
                "window_steps"),
        })
    panel = result.get("observed_panel_evidence") or {}
    if panel:
        rows.append({
            "kind": "cross_series_observed_evidence",
            "direction": panel.get("direction"),
            "support": panel.get("support"),
            "effective_series": panel.get("effective_series"),
            "agreement": panel.get("agreement"),
            "predictive_assumption": "recent_regime_persists",
        })
    return rows[:3]


def _available(requirement: str, result: dict[str, Any],
               evidence: list[dict[str, Any]]) -> bool:
    answer = result.get("answer") or {}
    executable = answer.get("executable") or {}
    kind = str(executable.get("kind", ""))
    folds = ((answer.get("property_distribution") or {}).get("folds")
             or (result.get("calibration") or {}).get("folds") or 0)
    if requirement == "immutable_forecast_projection":
        return kind == "published_forecast_projection"
    if requirement in {"rolling_origin_property_fit", "tail_calibration",
                       "positive_transition_evidence"}:
        return kind == "fitted_temporal_property" and int(folds) > 0
    if requirement == "rolling_origin_scale_fit":
        return bool(answer.get("property_distribution")) and int(folds) > 0
    if requirement == "paired_rolling_origin_fit":
        return kind == "fitted_dependence" and int(folds) > 0
    if requirement == "future_phase_evidence":
        return (kind == "published_forecast_projection"
                and answer.get("direction") not in {None, "uncertain"})
    if requirement == "conditional_effect_receipt":
        return bool(result.get("conditional_effect"))
    # Observed descriptive executables expose a computed estimate. Aggregate
    # panel evidence is an explicit observed receipt.
    return answer.get("estimate") is not None or any(
        row["kind"] == "cross_series_observed_evidence" for row in evidence)


def build_evidence_plan(question: TemporalQuestion,
                        result: dict[str, Any], *,
                        observed_evidence: dict[str, Any] | None = None,
                        ) -> dict[str, Any]:
    """Return a bounded reasoning pack that cannot override the answer."""
    mode = inference_mode(question)
    requirements = list(
        _OBSERVED_REQUIREMENTS[question.property] if mode == "observed"
        else _PREDICTIVE_REQUIREMENTS[question.property])
    if mode == "conditional":
        requirements.append("conditional_effect_receipt")
    evidence = _evidence_rows(result, observed_evidence)
    missing = [item for item in requirements
               if not _available(item, result, evidence)]
    canonical = (result.get("best_estimate") or {}).get("value")
    contradictions: list[dict[str, Any]] = []
    equivalents = {
        "seasonality": {"stable": "continued", "phase_shifted": "shifting"},
    }
    for row in evidence:
        direction = row.get("direction")
        comparable = equivalents.get(question.property, {}).get(
            str(direction), direction)
        if (row["kind"] in {"cross_series_observed_evidence",
                            "observed_transition"}
                and mode in {"predictive", "conditional"}
                and canonical not in {None, "uncertain"}
                and comparable not in {None, "uncertain", canonical}):
            contradictions.append({
                "between": ["canonical_predictive_answer", row["kind"]],
                "canonical": canonical, "observed": direction,
                "resolution": "retain canonical answer; disclose observed disagreement",
            })
    identifiable = not missing and canonical not in {None, "uncertain"}
    plan = {
        "version": PLANNER_VERSION,
        "mode": mode,
        "identifiable": identifiable,
        # The canonical value and support already sit one level above this
        # pack. Repeating them here costs every subsequent agent turn and
        # creates a second field that could drift.
        "authority": "fitted_executable",
        "llm_role": "explain_and_qualify_only",
        "primary_forecast_unchanged": True,
    }
    if evidence:
        plan["evidence"] = evidence
    if missing:
        plan["missing_evidence"] = missing[:3]
        plan["recovery"] = list(dict.fromkeys(
            _RECOVERY[item] for item in missing if item in _RECOVERY))[:2]
    else:
        plan["basis"] = requirements[:3]
    if contradictions:
        plan["contradictions"] = contradictions[:2]
    return plan

"""Governed context hypotheses and fold-safe candidate executables.

Language models may translate prose into competing typed hypotheses.  This
module gives those hypotheses stable identities, validates their grounding,
and evaluates numerical relationships without exposing future observations.
It deliberately does not publish forecasts or grant automation authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .effect_proposals import validate_effect_proposal
from .statistical_executables import fit_regression_executable

MAX_HYPOTHESES = 6
HYPOTHESIS_KINDS = frozenset({
    "absolute_value", "bound", "additive_change", "multiplicative_change",
    "regime_shift", "relationship", "historical_analogue", "unsupported",
})


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _identifier(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items()
            if key not in {"hypothesis_id", "validation"}}
    return "hyp-" + hashlib.sha256(_canonical(body).encode()).hexdigest()[:12]


def compile_context_hypotheses(
    raw: Any, *, claims: list[dict[str, Any]], series: list[str],
    cutoff: str, repair: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate a bounded set of alternative interpretations.

    A repair may replace only fields named by the first attempt's violations.
    Valid hypotheses from the first attempt survive a repair, making the
    protocol deterministic and preventing an agent from silently rewriting
    already accepted interpretations.
    """
    first = raw if isinstance(raw, list) else ([] if raw in (None, {}) else [raw])
    accepted, rejected = _validate_hypotheses(
        first[:MAX_HYPOTHESES], claims=claims, series=series, cutoff=cutoff)
    attempts = [{"attempt": 1, "accepted": len(accepted), "rejected": rejected}]
    if rejected and repair not in (None, {}):
        repairs = repair if isinstance(repair, list) else [repair]
        repaired, repair_rejected = _validate_hypotheses(
            repairs[:len(rejected)], claims=claims, series=series, cutoff=cutoff)
        accepted_ids = {item["hypothesis_id"] for item in accepted}
        accepted.extend(item for item in repaired
                        if item["hypothesis_id"] not in accepted_ids)
        rejected = repair_rejected
        attempts.append({"attempt": 2, "accepted": len(repaired),
                         "rejected": repair_rejected})
    accepted.sort(key=lambda item: item["hypothesis_id"])
    return accepted[:MAX_HYPOTHESES], {
        "status": ("accepted" if accepted and not rejected else
                   "partially_accepted" if accepted else "rejected"),
        "attempts_used": len(attempts), "attempts_remaining": 2 - len(attempts),
        "accepted": len(accepted), "rejected": rejected, "attempts": attempts,
        "bounded": True, "maximum_hypotheses": MAX_HYPOTHESES,
    }


def _validate_hypotheses(raw: list[Any], *, claims: list[dict[str, Any]],
                         series: list[str], cutoff: str
                         ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    claim_ids = {str(item.get("claim_id")) for item in claims}
    allowed_series = set(series) | {"*"}
    cutoff_dt = _aware(cutoff)
    if cutoff_dt is None:
        raise ValueError("cutoff must be timezone-aware ISO-8601")
    accepted, rejected = [], []
    for index, item in enumerate(raw):
        errors: list[dict[str, str]] = []
        if not isinstance(item, dict):
            rejected.append({"index": index, "violations": [{
                "field": "$", "code": "HYPOTHESIS_NOT_OBJECT"}]})
            continue
        kind = str(item.get("kind") or "unsupported")
        if kind not in HYPOTHESIS_KINDS:
            errors.append({"field": "kind", "code": "UNKNOWN_HYPOTHESIS_KIND"})
        cited = sorted({str(value) for value in item.get("claim_ids") or []})
        if not cited or set(cited) - claim_ids:
            errors.append({"field": "claim_ids", "code": "UNVERIFIED_CLAIMS"})
        targets = sorted({str(value) for value in item.get("target_series") or ["*"]})
        if set(targets) - allowed_series:
            errors.append({"field": "target_series", "code": "UNKNOWN_SERIES"})
        known_at = _aware(item.get("known_at", cutoff))
        if known_at is None or known_at > cutoff_dt:
            errors.append({"field": "known_at", "code": "NOT_KNOWN_AT_CUTOFF"})
        lag = item.get("lag_steps", 0)
        try:
            lag = int(lag)
        except (TypeError, ValueError):
            lag = -1
        if lag < 0:
            errors.append({"field": "lag_steps", "code": "INVALID_LAG"})
        proposal = None
        if item.get("effect_proposal") not in (None, {}):
            proposal, critique = validate_effect_proposal(
                item["effect_proposal"], claim_ids=claim_ids)
            if proposal is None:
                errors.extend({"field": "effect_proposal",
                               "code": violation["code"]}
                              for attempt in critique["attempts"]
                              for violation in attempt["violations"])
        predictor = item.get("predictor_series")
        if kind == "relationship" and str(predictor or "") not in allowed_series - {"*"}:
            errors.append({"field": "predictor_series", "code": "UNKNOWN_PREDICTOR"})
        if errors:
            rejected.append({"index": index, "violations": errors,
                             "repairable_fields": sorted({e["field"] for e in errors})})
            continue
        clean = {
            "kind": kind, "claim_ids": cited, "target_series": targets,
            "known_at": known_at.isoformat(), "lag_steps": lag,
            "predictor_series": str(predictor) if predictor is not None else None,
            "direction": str(item.get("direction") or "unknown"),
            "rationale": str(item.get("rationale") or "")[:1000],
            "effect_proposal": proposal,
            "validation": {"grounded": True, "known_at_cutoff": True,
                           "series_resolved": True},
        }
        clean["hypothesis_id"] = _identifier(clean)
        accepted.append(clean)
    return accepted, rejected


def _aware(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else None


def align_vintage_rows(rows: list[dict[str, Any]], *, cutoff: str,
                       time_key: str = "timestamp", known_key: str = "known_at"
                       ) -> list[dict[str, Any]]:
    """Return the latest vintage per timestamp that was knowable at cutoff."""
    cutoff_dt = _aware(cutoff)
    if cutoff_dt is None:
        raise ValueError("cutoff must be timezone-aware ISO-8601")
    chosen: dict[str, tuple[datetime, dict[str, Any]]] = {}
    for row in rows:
        valid = _aware(row.get(time_key))
        known = _aware(row.get(known_key, row.get(time_key)))
        if valid is None or known is None or valid > cutoff_dt or known > cutoff_dt:
            continue
        key = valid.isoformat()
        if key not in chosen or known > chosen[key][0]:
            chosen[key] = (known, dict(row))
    return [chosen[key][1] for key in sorted(chosen)]


@dataclass(frozen=True)
class FittedContextCandidate:
    kind: str
    hypothesis_id: str
    estimate: dict[str, Any]
    validation: dict[str, Any]
    support: str

    def execute(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "hypothesis_id": self.hypothesis_id,
            "estimate": self.estimate, "validation": self.validation,
            "support": self.support, "automation_eligible": False,
            "primary_forecast_unchanged": True,
            "executable": {"kind": self.kind, "version": "0.1",
                           "fold_safe": True},
        }


def fit_vintage_exogenous(
    rows: list[dict[str, Any]], *, target_key: str, predictor_keys: list[str],
    cutoff: str, hypothesis_id: str, minimum_train: int = 20,
) -> FittedContextCandidate:
    """Fit exogenous regression from point-in-time eligible aligned rows.

    In addition to the global cutoff, every training row must have been known
    by its own valid timestamp.  A later revision can therefore never improve
    an earlier expanding-origin prediction.
    """
    origin_safe = [row for row in rows
                   if _aware(row.get("known_at", row.get("timestamp"))) is not None
                   and _aware(row.get("timestamp")) is not None
                   and _aware(row.get("known_at", row.get("timestamp")))
                   <= _aware(row.get("timestamp"))]
    eligible = align_vintage_rows(origin_safe, cutoff=cutoff)
    eligible = [row for row in eligible
                if target_key in row
                and all(name in row for name in predictor_keys)]
    fitted = fit_regression_executable(
        [float(row[target_key]) for row in eligible],
        {name: [float(row[name]) for row in eligible] for name in predictor_keys},
        target=target_key, minimum_train=minimum_train)
    result = fitted.execute()
    validation = dict(result["estimate"]["validation"])
    skill = float(validation["skill_vs_mean_baseline"])
    validation.update({
        "skill": skill,
        "beats_baseline": result["direction"] == "predictive_contribution",
        "vintage_cutoff": cutoff, "per_origin_knowledge_checked": True,
    })
    return FittedContextCandidate(
        "fitted_vintage_exogenous", hypothesis_id,
        {"coefficients": result["estimate"]["coefficients"],
         "coefficient_intervals_95": result["estimate"]["coefficient_intervals_95"]},
        validation, result["support"])


def fit_lagged_relationship(
    target_rows: list[dict[str, Any]], predictor_rows: list[dict[str, Any]], *,
    target_key: str, predictor_key: str, cutoff: str, hypothesis_id: str,
    lags: list[int] | None = None, minimum_train: int = 20,
) -> FittedContextCandidate:
    """Choose a lag using expanding-origin predictions, never full-data fit."""
    target = align_vintage_rows(target_rows, cutoff=cutoff)
    predictor = align_vintage_rows(predictor_rows, cutoff=cutoff)
    x_by_time = {str(row["timestamp"]): float(row[predictor_key]) for row in predictor}
    y = [(str(row["timestamp"]), float(row[target_key])) for row in target
         if str(row["timestamp"]) in x_by_time]
    candidates = sorted(set(lags or [0, 1, 2, 3, 6, 12]))
    scores: list[dict[str, Any]] = []
    for lag in candidates:
        pairs = [(x_by_time[y[i-lag][0]], y[i][1]) for i in range(lag, len(y))]
        predictions, actuals, baselines = [], [], []
        for origin in range(max(minimum_train, 3), len(pairs)):
            train = pairs[:origin]
            xbar = statistics.mean(item[0] for item in train)
            ybar = statistics.mean(item[1] for item in train)
            denom = sum((item[0] - xbar) ** 2 for item in train)
            slope = (sum((a-xbar)*(b-ybar) for a, b in train) / denom
                     if denom > 1e-12 else 0.0)
            predictions.append(ybar + slope * (pairs[origin][0] - xbar))
            actuals.append(pairs[origin][1]); baselines.append(ybar)
        if not actuals:
            continue
        mse = statistics.mean((a-b)**2 for a, b in zip(actuals, predictions))
        base = statistics.mean((a-b)**2 for a, b in zip(actuals, baselines))
        scores.append({"lag_steps": lag, "validation_points": len(actuals),
                       "mse": mse, "baseline_mse": base,
                       "skill": 1 - mse / max(base, 1e-12)})
    if not scores:
        raise ValueError("insufficient vintage-aligned history for lag validation")
    best = max(scores, key=lambda item: (item["skill"], -item["lag_steps"]))
    # Multiplicity-aware admission: require more evidence as more lags compete.
    threshold = min(.25, .02 + .01 * math.log2(max(1, len(scores))))
    supported = best["skill"] >= threshold and best["validation_points"] >= 8
    return FittedContextCandidate(
        "fitted_lagged_relationship", hypothesis_id,
        {"selected_lag_steps": best["lag_steps"], "skill": best["skill"]},
        {"scheme": "expanding_origin", "candidates": scores,
         "admission_threshold": threshold, "beats_baseline": supported,
         "vintage_cutoff": cutoff},
        "supported" if supported else "weak")


def fit_historical_analogue(
    episodes: list[dict[str, Any]], *, query_features: dict[str, float],
    cutoff: str, hypothesis_id: str, k: int = 5,
) -> FittedContextCandidate:
    """Evaluate nearest historical episodes with leave-one-episode-out skill."""
    cutoff_dt = _aware(cutoff)
    if cutoff_dt is None:
        raise ValueError("cutoff must be timezone-aware ISO-8601")
    names = tuple(sorted(query_features))
    eligible = [item for item in episodes
                if _aware(item.get("outcome_known_at")) is not None
                and _aware(item["outcome_known_at"]) <= cutoff_dt
                and all(name in (item.get("features") or {}) for name in names)
                and math.isfinite(float(item.get("outcome")))]
    if len(eligible) < 5:
        raise ValueError("historical analogues require five resolved episodes")
    scales = {name: max(statistics.pstdev(
        [float(item["features"][name]) for item in eligible]), 1e-12)
        for name in names}
    def distance(features: dict[str, float], item: dict[str, Any]) -> float:
        return math.sqrt(sum(((float(features[name]) -
                              float(item["features"][name])) / scales[name]) ** 2
                             for name in names))
    errors, baseline_errors = [], []
    for held in eligible:
        train = [item for item in eligible if item is not held]
        nearest = sorted(train, key=lambda item: distance(held["features"], item))[:k]
        prediction = statistics.mean(float(item["outcome"]) for item in nearest)
        truth = float(held["outcome"])
        errors.append(abs(truth - prediction))
        baseline_errors.append(abs(truth - statistics.mean(
            float(item["outcome"]) for item in train)))
    nearest = sorted(eligible, key=lambda item: distance(query_features, item))[:k]
    outcomes = [float(item["outcome"]) for item in nearest]
    skill = 1 - statistics.mean(errors) / max(statistics.mean(baseline_errors), 1e-12)
    supported = skill >= .02 and len(errors) >= 8
    return FittedContextCandidate(
        "fitted_historical_analogue", hypothesis_id,
        {"location": statistics.mean(outcomes), "lower": min(outcomes),
         "upper": max(outcomes), "matched_episode_ids": [
             str(item.get("episode_id")) for item in nearest]},
        {"scheme": "leave_one_episode_out", "episodes": len(errors),
         "mae": statistics.mean(errors),
         "global_mean_mae": statistics.mean(baseline_errors),
         "skill": skill, "beats_baseline": supported,
         "outcomes_known_by": cutoff},
        "supported" if supported else "weak")


def candidate_evidence_score(candidate: dict[str, Any]) -> dict[str, Any]:
    """Rank evidence, not provenance prestige or an LLM confidence claim."""
    validation = candidate.get("validation") or {}
    points = int(validation.get("validation_points") or
                 validation.get("episodes") or 0)
    skill = float(validation.get("skill") or 0.0)
    supported = bool(validation.get("beats_baseline"))
    score = max(-1.0, min(1.0, skill)) * min(1.0, points / 20.0)
    return {"score": score, "validation_points": points,
            "beats_baseline": supported,
            "decisive": supported and points >= 8 and score > 0,
            "automation_eligible": False,
            "basis": "out_of_sample_skill_times_evidence_fraction"}

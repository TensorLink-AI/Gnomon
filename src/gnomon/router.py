"""The thin task router: automatic, disclosed, never silent.

``route`` answers "which method should run for this task on this data" in
three disclosed layers:

1. **Capability filter** — hard, verified limits (``eligible_tsfms`` for
   forecasting, detector minimums for anomaly detection). Excluded
   candidates ship with their exclusion reasons.
2. **Tracking prior** — when the tracking store holds enough scored
   history for the task (``MIN_PRIOR_RECORDS``), the leaderboard ranked
   by realised MASE, fingerprint-weighted toward runs on data shaped
   like this series. Below that, no prior is claimed: cold-start honesty
   beats a confident guess.
3. **Backtest tiebreak** — the recommendation is advisory either way;
   the evaluated run's own backtest remains the final selector, and a
   ``model=`` override on every entry point beats the router entirely.

Every decision is deterministic given (data, store state), is recorded
to the tracking store when a project is supplied, and carries its basis
so replay can reproduce not just the prediction but the choice.
"""

from __future__ import annotations

import json
from typing import Any

from .fingerprint import fingerprint_distance, series_fingerprint
from .ids import content_id

MIN_PRIOR_RECORDS = 10
ROUTABLE_TASKS = ("forecast", "detect_anomalies")


def _forecast_candidates(
    values: list[float], frequency: str, horizon: int,
) -> tuple[list[str], dict[str, list[str]]]:
    from .models import MODELS
    from .tsfm import eligible_tsfms
    eligible_names, excluded = eligible_tsfms(
        history_length=len(values), horizon=horizon, frequency=frequency,
    )
    statistical = list(MODELS)
    zero_share = (sum(value == 0.0 for value in values) / len(values)
                  if values else 0.0)
    if "croston_sba" in statistical and zero_share < .1:
        statistical.remove("croston_sba")
        excluded["croston_sba"] = [
            "Croston SBA is reserved for intermittent demand; fewer than "
            f"10% of observations are zero ({zero_share:.1%})."
        ]
    return statistical + eligible_names, excluded


def _anomaly_candidates(values: list[float]) -> tuple[list[str], dict[str, list[str]]]:
    from .anomaly import DETECTORS, MIN_DETECTION_HISTORY, tsfm_reconstruction_detectors
    names = list(DETECTORS) + list(tsfm_reconstruction_detectors())
    if len(values) < MIN_DETECTION_HISTORY:
        return [], {name: [
            f"needs at least {MIN_DETECTION_HISTORY} observations (have {len(values)})"
        ] for name in names}
    return names, {}


def _structural_starting_point(
    task: str, candidates: list[str], fingerprint: dict[str, Any],
) -> str | None:
    """Return one disclosed candidate to start the required evaluation.

    This is deliberately not a model winner: it uses only structural facts
    visible in the input, and the caller must still backtest the full eligible
    set.  A cold-start router that always returns null is honest but cannot
    help an agent take its next step.
    """
    if not candidates:
        return None
    if task == "detect_anomalies":
        return ("robust_zscore" if "robust_zscore" in candidates
                else candidates[0])
    intermittency = float(fingerprint.get("intermittency") or 0.0)
    strength = float(fingerprint.get("season_strength") or 0.0)
    trend = abs(float(fingerprint.get("trend") or 0.0))
    direction_changes = float(fingerprint.get("direction_change_rate") or 0.0)
    preferences = (
        ["croston_sba", "historical_mean"] if intermittency >= .1 else
        ["ets", "seasonal_naive"]
        if fingerprint.get("season_source") == "autocorrelation"
        and strength >= .5 else
        ["linear_trend", "theta", "drift"] if trend >= .01 else
        ["historical_mean", "window_average", "theta"]
        if direction_changes >= .65 else
        ["theta", "ets", "historical_mean"]
    )
    return next((name for name in preferences if name in candidates),
                candidates[0])


def _tracking_prior(
    task: str, candidates: list[str], fingerprint: dict[str, Any],
    project: str, store: Any,
) -> dict[str, Any]:
    """Fingerprint-weighted realised-performance ranking, or an honest None."""
    history = [
        row for row in store.leaderboard(project, task=task)
        if row.model in candidates and row.avg_mase is not None
    ]
    scored_records = sum(row.count for row in history)
    if scored_records < MIN_PRIOR_RECORDS:
        return {
            "source": None,
            "reason": (
                f"Only {scored_records} scored {task} records exist for project "
                f"{project!r} (need {MIN_PRIOR_RECORDS}); no prior is claimed."
            ),
            "scored_records": scored_records,
        }
    # Fingerprint weighting: nearer past runs count more. Distances come
    # from the per-run fingerprints in model_performance.
    weighted: list[dict[str, Any]] = []
    for row in history:
        runs = store.model_performance(project, row.model, task=task)
        numerator, denominator = 0.0, 0.0
        for run in runs:
            if run.get("mase") is None:
                continue
            run_fingerprint = json.loads(run["fingerprint"]) if run.get("fingerprint") else {}
            distance = fingerprint_distance(fingerprint, run_fingerprint)
            weight = 1.0 / (1.0 + distance) if distance is not None else 0.5
            numerator += weight * run["mase"]
            denominator += weight
        weighted.append({
            "model": row.model,
            "scored_runs": row.count,
            "avg_mase": row.avg_mase,
            "fingerprint_weighted_mase": (
                round(numerator / denominator, 4) if denominator else None
            ),
        })
    weighted.sort(key=lambda item: (
        item["fingerprint_weighted_mase"] is None,
        item["fingerprint_weighted_mase"] if item["fingerprint_weighted_mase"] is not None else 0.0,
        item["model"],
    ))
    return {"source": "tracking", "scored_records": scored_records, "ranking": weighted}


def route(
    task: str,
    values: list[float],
    frequency: str,
    *,
    horizon: int | None = None,
    series: str | None = None,
    project: str | None = None,
    store: Any | None = None,
) -> dict[str, Any]:
    """Disclosed routing decision for one task over one series.

    The recommendation is advisory: evaluated runs still backtest every
    candidate, and an explicit model override always wins.
    """
    if task not in ROUTABLE_TASKS:
        raise ValueError(f"Unknown routable task {task!r}; expected one of {ROUTABLE_TASKS}")
    fingerprint = series_fingerprint(values, frequency)
    if task == "forecast":
        candidates, excluded = _forecast_candidates(values, frequency, horizon or 1)
    else:
        candidates, excluded = _anomaly_candidates(values)
    prior: dict[str, Any] = {"source": None, "reason": "No tracking project was supplied."}
    if store is not None and project is not None and candidates:
        prior = _tracking_prior(task, candidates, fingerprint, project, store)
    if prior.get("source") == "tracking" and prior["ranking"]:
        recommendation = prior["ranking"][0]["model"]
        basis = "tracking_prior"
    elif candidates:
        recommendation = _structural_starting_point(
            task, candidates, fingerprint)
        basis = "structural_starting_point_backtest_required"
    else:
        recommendation = None
        basis = "no_eligible_candidates"
    decision = {
        "schema_version": "0.1",
        "task": task,
        "series": series or "__default__",
        "fingerprint": fingerprint,
        "candidates": candidates,
        "excluded": excluded,
        "prior": prior,
        "recommendation": recommendation,
        "recommendation_role": (
            "realised_performance_prior" if basis == "tracking_prior" else
            "starting_candidate_only; evaluate all eligible candidates"
            if recommendation is not None else "none"
        ),
        "basis": basis,
        "override": (
            "Pass an explicit model/detector to bypass the router; "
            "evaluated runs backtest every candidate regardless."
        ),
    }
    route_id = content_id("route", {
        "task": task, "series": decision["series"],
        "fingerprint": fingerprint, "candidates": candidates,
        "prior": prior, "recommendation": recommendation,
    })
    decision["route_id"] = route_id
    if store is not None and project is not None:
        store.record_route(
            route_id, project, task,
            series=decision["series"],
            fingerprint=json.dumps(fingerprint, sort_keys=True),
            recommendation=recommendation, basis=basis,
            payload=decision,
        )
    return decision

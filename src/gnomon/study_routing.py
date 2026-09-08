"""Cutoff-bound provider recommendations from one explicit immutable cohort.

No mutable leaderboard, similarity-weighted mixture of unmatched tasks, new model
calls or automatic action. Revised outcomes produce a new immutable rescore.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import math
import json
from uuid import uuid4
from time import monotonic

from .forecast_adapter import AdapterCapabilities, ForecastAdapterError, point_error_metrics, validate_capabilities
from .ids import content_id
from .ledger import _time
from .repair import historical_repair_blockers

DEFAULT_MIN_FOLDS = 3


def route_study(engine, references, data_ref: str, *, study_id: str, candidates: list[str], baseline: str,
                horizon: int, source_as_of: str, recorded_as_of: str, series_id: str | None = None,
                season: int = 1, min_folds: int = DEFAULT_MIN_FOLDS, min_improvement: float = .02, max_folds: int = 8) -> dict:
    started = monotonic()
    if engine.ledger is None:
        raise ForecastAdapterError("study routing requires an explicit ledger")
    if not isinstance(study_id, str) or not study_id:
        raise ForecastAdapterError("study_id must identify one immutable study")
    if type(min_folds) is not int or min_folds < 3:
        raise ForecastAdapterError("routing requires at least three matched folds")
    if type(min_improvement) not in (int, float) or not math.isfinite(min_improvement) or not 0 <= min_improvement <= 1:
        raise ForecastAdapterError("min_improvement must be finite and between zero and one")
    if not isinstance(candidates, list) or not candidates or not isinstance(baseline, str) or not baseline:
        raise ForecastAdapterError("name explicit candidates and baseline")
    providers = [baseline, *candidates]
    if any(not isinstance(p, str) or not p for p in providers) or len(set(providers)) != len(providers):
        raise ForecastAdapterError("providers must be distinct names including the baseline")
    source_cutoff, recorded_cutoff = datetime.fromisoformat(_time(source_as_of)), datetime.fromisoformat(_time(recorded_as_of))
    frozen, name, rows = references._select(data_ref, series_id)
    if historical_repair_blockers(frozen.repairs):
        raise ForecastAdapterError("routing requires unrepaired historical inputs")
    if any(r.timestamp.tzinfo is None for r in rows):
        raise ForecastAdapterError("historical routing requires timezone-aware valid times")
    parent = frozen.loaded.snapshot
    visible = parent.narrow(as_of=source_cutoff, recorded_as_of=None if parent.assumed_known_time else recorded_cutoff)
    current = [r for r in visible.series(name, frozen.loaded.variable) if r.valid_time <= source_cutoff]
    if [(r.valid_time, r.value) for r in current] != [(r.timestamp, r.value) for r in rows]:
        raise ForecastAdapterError("inspect the input at the routing source/recorded cutoffs before requesting a recommendation")
    request = references.request(data_ref, horizon=horizon, season=season, series_id=name)
    if datetime.fromisoformat(request.future_timestamps[0]) <= source_cutoff:
        raise ForecastAdapterError("forecast grid must start after the routing source cutoff")
    identities = engine.capabilities()
    for p in providers:
        if p not in identities:
            raise ForecastAdapterError("every provider must be registered")
        validate_capabilities(AdapterCapabilities(**identities[p]["capabilities"]), request)
    answer = {"schema_version": "1", "status": "ok", "data_ref": data_ref, "study_id": study_id,
              "series_id": name, "unit": frozen.unit, "horizon": horizon,
              "source_as_of": _time(source_as_of), "recorded_as_of": _time(recorded_as_of),
              "effective_source_as_of": visible.as_of.isoformat(),
              "effective_recorded_as_of": visible.recorded_as_of.isoformat() if visible.recorded_as_of else None,
              "recommendation": baseline, "basis": "explicit_baseline_fallback", "reason": None,
              "min_folds": min_folds, "min_improvement": min_improvement,
              "provider_calls": 0, "action_authorized": False, "recommendation_role": "advisory",
              "known_time_assumed": parent.assumed_known_time, "scores": {}, "matched_folds": 0}

    def fallback(reason):
        next_step = {
            "provider_identity_changed": "run_new_evaluation_with_explicit_budget",
            "provider_revision_unknown": "register_an_explicit_provider_revision",
            "study_unavailable_at_recorded_cutoff": "search_for_a_study_visible_at_the_requested_cutoff",
            "study_not_found": "check_study_id_and_ledger_path_or_run_evaluate",
            "select_an_original_backtest_study": "select_an_original_backtest_study",
            "task_identity_mismatch": "search_for_a_matching_task_study",
            "provider_cohort_mismatch": "select_a_study_with_the_requested_providers",
            "study_exceeds_operator_fold_limit": "select_a_study_within_operator_limits",
            "study_integrity_unverifiable": "verify_ledger_integrity_before_reuse",
        }.get(reason, "inspect_excluded_folds_before_collecting_more_evidence")
        return {**answer, "reason": reason, "next_step": next_step}

    try:
        report = engine.ledger.study(study_id, recorded_as_of=recorded_as_of)
    except ForecastAdapterError as error:
        if error.details.get("reason") == "study_not_found":
            return fallback("study_not_found")
        return fallback("study_integrity_unverifiable" if "integrity" in str(error) else "study_unavailable_at_recorded_cutoff")
    if report.get("evidence") != "rolling_origin_backtest" or report.get("derived_from"):
        return fallback("select_an_original_backtest_study")
    if len(report["folds"]) > max_folds:
        return fallback("study_exceeds_operator_fold_limit")
    if any(report.get(k) != v for k, v in {"series_id": name, "unit": frozen.unit, "horizon": horizon,
            "season": season, "frequency": frozen.loaded.frequency, "baseline": baseline}.items()):
        return fallback("task_identity_mismatch")
    if set(report["providers"]) != set(providers):
        return fallback("provider_cohort_mismatch")
    for p in providers:
        revision = identities[p]["revision"]
        if revision in {None, "latest", "unversioned"}:
            return fallback("provider_revision_unknown")
        if any(json.dumps(identities[p][k], sort_keys=True) != json.dumps(report["providers"][p].get(k), sort_keys=True)
               for k in ("revision", "lifecycle", "capabilities")):
            return fallback("provider_identity_changed")
    selected, excluded = [], []
    truth = {r.valid_time: r for r in current}
    for fold in report["folds"]:
        reason = None
        if fold["status"] != "complete":
            excluded.append({"origin": fold["origin"], "reason": "incomplete_original_fold"})
            continue
        req = fold["request"]
        future = [datetime.fromisoformat(t) for t in req["future_timestamps"]]
        if any(t.tzinfo is None for t in future) or datetime.fromisoformat(req["known_time_cutoff"]).tzinfo is None:
            excluded.append({"origin": fold["origin"], "reason": "historical_timezone_unresolved"})
            continue
        if any(t > source_cutoff or t not in truth for t in future):
            excluded.append({"origin": fold["origin"], "reason": "actuals_unavailable_at_cutoffs"})
            continue
        historical = visible.narrow(as_of=datetime.fromisoformat(req["known_time_cutoff"]),
                                     recorded_as_of=datetime.fromisoformat(req["recorded_time_cutoff"])
                                     if req["recorded_time_cutoff"] else None)
        history = [r for r in historical.series(name, frozen.loaded.variable)
                   if r.valid_time <= datetime.fromisoformat(req["cutoff"])]
        if ([r.valid_time.isoformat() for r in history] != req["timestamps"] or [r.value for r in history] != req["history"]):
            reason = "historical_inputs_not_reconstructible"
        actuals = [asdict(truth[t]) for t in future]
        actuals = [{k: v.isoformat() if hasattr(v, "isoformat") else v for k, v in row.items()} for row in actuals]
        if parent.assumed_known_time and any(a["value"] != b["value"] for a, b in zip(actuals, fold["actuals"])):
            reason = "file_revision_availability_unknown"
        for p in providers:
            run = engine.ledger.execution(fold["runs"][p]["execution_id"])
            if run["recorded_at"] > _time(recorded_as_of):
                reason = "execution_not_recorded_at_cutoff"
            if run["request"] != req or run["provider"] != p or run["revision"] != identities[p]["revision"] \
                    or run["result"]["point"] != fold["runs"][p]["point"]:
                reason = "study_execution_mismatch"
            if identities[p]["lifecycle"] == "pretrained":
                try:
                    training_cutoff = _time(run["result"]["metadata"]["training_cutoff"])
                    if training_cutoff > _time(req["cutoff"]):
                        reason = "pretrained_training_cutoff_after_origin"
                except (KeyError, ForecastAdapterError):
                    reason = "pretrained_training_cutoff_unattested"
        if reason:
            excluded.append({"origin": fold["origin"], "reason": reason})
        else:
            selected.append({**fold, "actuals": actuals})
    answer["matched_folds"] = len(selected)
    answer["excluded_folds"] = excluded
    if len(selected) < min_folds:
        return fallback("insufficient_replayable_matched_folds")
    scores = {p: point_error_metrics((point, actual["value"]) for f in selected
                         for point, actual in zip(f["runs"][p]["point"], f["actuals"])) for p in providers}
    ranked = sorted(providers, key=lambda p: (scores[p]["mae"], providers.index(p)))
    best, baseline_mae = ranked[0], scores[baseline]["mae"]
    improvement = (baseline_mae - scores[best]["mae"]) / baseline_mae if baseline_mae else 0.0
    choice = best if improvement >= min_improvement and scores[best]["mae"] < baseline_mae else baseline
    rescore = {**deepcopy(report), "study_id": str(uuid4()), "derived_from": study_id, "rescore_only": True,
               "folds": selected, "scores": scores, "ranking": ranked, "snapshot_id": visible.snapshot_id,
               "source_as_of": _time(source_as_of), "recorded_as_of": _time(recorded_as_of),
               "effective_source_as_of": answer["effective_source_as_of"],
               "effective_recorded_as_of": answer["effective_recorded_as_of"],
               "status": "complete" if len(selected) == report["usage"]["requested_folds"] else "partial",
               "usage": {**report["usage"], "provider_calls": 0, "elapsed_seconds": monotonic() - started,
                         "matched_folds": len(selected), "stop_reason": None, "wall_limit_overrun": False},
               "original_usage": report["usage"], "recorded": True}
    rescore["original_diagnostics"] = rescore["diagnostics"]
    rescore["diagnostics"] = {p: {"attempted": 0, "succeeded": 0, "failed": 0, "reused": len(selected)} for p in providers}
    rescore["usage"]["internal_model_calls"] = 0
    rescore["original_budget"] = report["budget"]
    rescore["budget"] = {"max_folds": max_folds, "max_calls": 0, "max_seconds": None, "max_providers": len(providers)}
    rescore.pop("recorded_at", None)
    rescore["cohort_id"] = content_id("cohort", {"folds": [{k: f[k] for k in ("request", "actuals")} for f in selected]}, length=64)
    engine.ledger.record_study(rescore)
    return {**answer, "recommendation": choice, "basis": "cutoff_bound_matched_study", "reason": None,
            "scores": scores, "relative_mae_improvement": improvement, "rescore_study_id": rescore["study_id"],
            "cohort_id": rescore["cohort_id"], "model_identity_basis": "provider_declared_not_independently_attested"}

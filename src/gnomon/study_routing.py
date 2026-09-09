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
from .contracts import GnomonError

DEFAULT_MIN_FOLDS = 3


def route_study(engine, references, data_ref: str, *, study_id: str, candidates: list[str], baseline: str,
                horizon: int, source_as_of: str, recorded_as_of: str, series_id: str | None = None,
                season: int = 1, min_folds: int = DEFAULT_MIN_FOLDS, min_improvement: float = .02, max_folds: int = 8,
                require_evidence: bool = False) -> dict:
    started = monotonic()
    if type(require_evidence) is not bool:
        raise ForecastAdapterError("require_evidence must be a boolean")
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
    visible = parent.narrow(as_of=source_cutoff, recorded_as_of=None if parent.unknown_recorded_times else recorded_cutoff)
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
              "cutoff_scopes": {
                  "ledger_evidence_recorded_as_of": _time(recorded_as_of),
                  "snapshot_source_as_of": visible.as_of.isoformat(),
                  "snapshot_recorded_as_of": visible.recorded_as_of.isoformat() if visible.recorded_as_of else None,
                  "snapshot_recording_basis": "unknown_recording_times" if parent.unknown_recorded_times else "recorded_vintages",
                  "effective_fields_scope": "input_snapshot",
                  "guidance": "recorded_as_of filters recorded studies and executions. Effective cutoff fields describe the input snapshot; a null snapshot recording cutoff does not disable ledger evidence filtering."},
              "recommendation": baseline, "basis": "explicit_baseline_fallback", "fallback_used": True, "reason": None,
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
        }.get(reason, "inspect_excluded_folds_before_collecting_more_evidence" if answer.get("excluded_folds")
              else "evaluate_more_matched_folds_with_sufficient_history_and_budget")
        actions = []
        if reason == "insufficient_replayable_matched_folds" and not answer.get("excluded_folds"):
            actions.append({"tool": "gnomon_evaluate", "arguments": {
                "data_ref": data_ref, "series_id": name, "candidates": candidates,
                "baseline": baseline, "horizon": horizon, "season": season, "folds": min_folds},
                "example_kind": "task_template", "admissible": None,
                "requires_provider_calls": True,
                "guidance": "Check available observed history and operator dispatch budgets before executing. This creates a new study; sufficient replayable folds are not guaranteed."})
        if reason not in {"study_not_found", "study_unavailable_at_recorded_cutoff", "study_integrity_unverifiable"}:
            actions.append({"tool": "gnomon_ledger", "arguments": {
                "operation": "study", "study_id": study_id, "recorded_as_of": recorded_as_of},
                "requires_provider_calls": False})
        response = {**answer, "reason": reason, "next_step": next_step, "next_actions": actions,
                    "evidence_based": False, "routing_status": "fallback",
                    "warning": "Baseline fallback: matched evidence did not support provider selection."}
        if require_evidence:
            raise GnomonError("ROUTING_EVIDENCE_REQUIRED",
                              "Evidence-based routing was required but is unavailable: " + reason,
                              details=response, repair_options=[{"action": next_step,
                                  "description": "Resolve the reported evidence issue and retry with require_evidence=true. No provider was executed or rescore saved."}])
        return response

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
    variable = report.get('variable', next((a['variable'] for f in report['folds'] for a in f['actuals']), None))
    mismatches = [{'field': k, 'study': report.get(k), 'requested': v} for k, v in {"series_id": name, "unit": frozen.unit, "horizon": horizon,
            "season": season, "frequency": frozen.loaded.frequency, "baseline": baseline}.items() if report.get(k) != v]
    if variable is not None and variable != frozen.loaded.variable:
        mismatches.append({'field': 'variable', 'study': variable, 'requested': frozen.loaded.variable})
    if set(report["providers"]) != set(providers):
        mismatches.append({'field': 'providers', 'study': list(report['providers']), 'requested': providers})
    if mismatches:
        answer.update(mismatch=mismatches[0], mismatches=mismatches)
        return fallback("provider_cohort_mismatch" if all(m['field'] == 'providers' for m in mismatches) else "task_identity_mismatch")
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
        if ([r.valid_time for r in history] != [datetime.fromisoformat(t) for t in req["timestamps"]] or [r.value for r in history] != req["history"]):
            reason = "historical_inputs_not_reconstructible"
        actuals = [asdict(truth[t]) for t in future]
        actuals = [{k: v.isoformat() if hasattr(v, "isoformat") else v for k, v in row.items()} for row in actuals]
        if parent.unknown_recorded_times and any(a["value"] != b["value"] for a, b in zip(actuals, fold["actuals"])):
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
            excluded.append({"origin": fold["origin"], "reason": reason,
                **({'description': 'Actual values changed in data with assumed source availability. Their revision availability and recording history are unknown; ingest explicit vintages into TemporalStore before reusing revised outcomes.'} if reason == 'file_revision_availability_unknown' else {})})
        else:
            selected.append({**fold, "actuals": actuals})
    answer["matched_folds"] = len(selected)
    answer["excluded_folds"] = excluded
    if len(selected) < min_folds:
        return fallback("insufficient_replayable_matched_folds")
    scores = {p: point_error_metrics((point, actual["value"]) for f in selected
                         for point, actual in zip(f["runs"][p]["point"], f["actuals"])) for p in providers}
    ranked = sorted(providers, key=lambda p: (scores[p]["mae"], providers.index(p)))
    groups = {}
    for provider in ranked:
        groups.setdefault(scores[provider]["mae"], []).append(provider)
    ranking_policy = {"metric": "mae", "direction": "ascending", "tie_comparison": "exact_unrounded_score",
                      "tie_order": "provider_input_order", "ties": [
                          {"mae": score, "providers": group} for score, group in groups.items() if len(group) > 1],
                      "guidance": "Equal MAE does not establish a winner or equivalent predictions. Ranking does not establish future performance."}
    best, baseline_mae = ranked[0], scores[baseline]["mae"]
    improvement = (baseline_mae - scores[best]["mae"]) / baseline_mae if baseline_mae else 0.0
    choice = best if improvement >= min_improvement and scores[best]["mae"] < baseline_mae else baseline
    selection_reason = ("candidate_exceeds_improvement_threshold" if choice != baseline else
                        "baseline_tied_for_best" if len(groups[baseline_mae]) > 1 and best == baseline else
                        "baseline_has_lowest_mae" if best == baseline else "improvement_below_threshold")
    rescore = {**deepcopy(report), "study_id": str(uuid4()), "derived_from": study_id, "rescore_only": True,
               "folds": selected, "scores": scores, "ranking": ranked, "ranking_policy": ranking_policy,
               "snapshot_id": visible.snapshot_id,
               "cutoff_scopes": answer["cutoff_scopes"],
               "source_as_of": _time(source_as_of), "recorded_as_of": _time(recorded_as_of),
               "effective_source_as_of": answer["effective_source_as_of"],
               "effective_recorded_as_of": answer["effective_recorded_as_of"],
               "status": "complete" if len(selected) == report["usage"]["requested_folds"] else "partial",
               "usage": {**report["usage"], "provider_calls": 0, "elapsed_seconds": monotonic() - started,
                         "matched_folds": len(selected), "stop_reason": None, "wall_limit_overrun": False},
               "original_usage": report["usage"], "recorded": True}
    from .rescoring import digest, refresh_derivations
    rescore.update(original_study_id=study_id, rescore_study_id=rescore['study_id'], original_study_sha256=digest(report),
                   predictions_reused_exactly=True, original_unchanged=True, original_snapshot_id=report['snapshot_id'],
                   source_fingerprint=visible.source_ref)
    refresh_derivations(rescore, selected, providers)
    rescore.update(evaluation_status=rescore['status'], routing_status='rescore_not_a_new_route_task',
        routing_readiness={'ready': False, 'issues': [{'action': 'route_original_study',
            'description': 'Select the original study when routing; this record contains derived scores.'}]})
    rescore["original_diagnostics"] = rescore["diagnostics"]
    rescore["diagnostics"] = {p: {"attempted": 0, "succeeded": 0, "failed": 0, "reused": len(selected)} for p in providers}
    rescore["usage"]["internal_model_calls"] = 0
    rescore["original_budget"] = report["budget"]
    rescore["budget"] = {"max_folds": max_folds, "max_calls": 0, "max_seconds": None, "max_providers": len(providers)}
    rescore.pop("recorded_at", None)
    rescore["cohort_id"] = content_id("cohort", {"folds": [{k: f[k] for k in ("request", "actuals")} for f in selected]}, length=64)
    engine.ledger.record_study(rescore)
    return {**answer, "recommendation": choice, "basis": "cutoff_bound_matched_study", "fallback_used": False, "reason": None,
            "evidence_based": True, "routing_status": "selected",
            "selection_reason": selection_reason, "ranking": ranked, "ranking_policy": ranking_policy,
            "scores": scores, "relative_mae_improvement": improvement, "rescore_study_id": rescore["study_id"],
            "full_study": {"tool": "gnomon_evaluate", "arguments": {"study_id": rescore["study_id"]}},
            "cohort_id": rescore["cohort_id"], "model_identity_basis": "provider_declared_not_independently_attested"}

"""Optional budgeted evaluation over frozen observation vintages.

Budgets count Gnomon provider invocations, not a remote service's internal model
fan-out. Time/cancellation limits stop new dispatches; trusted in-process Python
cannot be forcibly interrupted safely. No hidden fitting or model selection is
added to ordinary inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from time import monotonic
from typing import Callable
from uuid import uuid4

from .data import Observation
from .contracts import GnomonError
from .forecast_adapter import AdapterCapabilities, ForecastAdapterError, ForecastRequest, point_error_metrics, validate_capabilities
from .ids import content_id
from .repair import historical_repair_blockers
from .temporal import validate_and_group


@dataclass(frozen=True)
class EvaluationBudget:
    max_providers: int = 4  # includes the explicit baseline
    max_folds: int = 8
    max_calls: int = 32
    max_seconds: float | None = None

    def __post_init__(self):
        for name in ("max_providers", "max_folds", "max_calls"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "max_calls" else 1):
                raise ForecastAdapterError(f"{name} must be an integer within its nonnegative/positive bound")
        if self.max_seconds is not None and (type(self.max_seconds) not in (int, float)
                or not math.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ForecastAdapterError("max_seconds must be finite and positive")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ForecastAdapterError("budget must contain only documented budget fields")
        return cls(**value)


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ForecastAdapterError(f"{name} must be a positive integer")


def evaluate_reference(engine, references, data_ref: str, *, candidates: list[str], baseline: str,
                       horizon: int, folds: int = 4, min_history: int = 8, stride: int | None = None,
                       series_id: str | None = None, season: int = 1, budget: EvaluationBudget | None = None,
                       replay: str | None = None, cancelled: Callable[[], bool] | None = None,
                       timer: Callable[[], float] = monotonic, verify: bool = False) -> dict:
    """Evaluate exact matched point-forecast tasks; persist an optional study.

    Current pretrained weights may have seen later training data. Observation
    replay does not attest a provider's training cutoff. No probability calibration
    or action permission follows from a lower historical point error.
    """
    if type(verify) is not bool:
        raise ForecastAdapterError('verify must be a boolean')
    budget = budget if budget is not None else EvaluationBudget()
    if not isinstance(budget, EvaluationBudget):
        raise ForecastAdapterError("budget must be EvaluationBudget")
    for name, value in (("horizon", horizon), ("folds", folds), ("min_history", min_history), ("season", season)):
        _positive(value, name)
    stride = horizon if stride is None else stride
    _positive(stride, "stride")
    if not isinstance(candidates, list) or not candidates or any(
            not isinstance(name, str) or not name.strip() for name in candidates):
        raise ForecastAdapterError("candidates must be a nonempty array of registered provider names")
    if not isinstance(baseline, str) or not baseline.strip():
        raise ForecastAdapterError("baseline must explicitly name a registered provider")
    providers = [baseline, *candidates]
    if len(set(providers)) != len(providers):
        raise ForecastAdapterError("candidates must be distinct and exclude the baseline")
    if len(providers) > budget.max_providers or folds > budget.max_folds:
        raise ForecastAdapterError("requested providers/folds exceed the explicit budget")
    identities = engine.capabilities()
    if any(name not in identities for name in providers):
        unknown = [name for name in providers if name not in identities]
        raise ForecastAdapterError(
            "every candidate and baseline must be registered before evaluation; unknown providers: "
            + ", ".join(unknown) + "; available providers: " + ", ".join(sorted(identities)),
            details={"unknown_providers": unknown, "available_providers": sorted(identities)})
    started = timer()
    frozen, name, rows = references._select(data_ref, series_id)
    # Retrospective interpolation/restamping can encode future values even
    # when the repaired row is later labelled known at its valid timestamp.
    if historical_repair_blockers(frozen.repairs):
        raise ForecastAdapterError("historical evaluation requires unrepaired observations or value-preserving format fixes; "
                                   "use a complete regular window of observed data, or prepare each vintage upstream")
    snapshot = frozen.loaded.snapshot
    replay = replay or ("recorded" if snapshot.recorded_as_of is not None else "source_available")
    if replay not in {"recorded", "source_available"}:
        raise ForecastAdapterError("replay must be source_available or recorded")
    if replay == "recorded" and snapshot.assumed_known_time:
        raise ForecastAdapterError("plain files cannot reconstruct historical recording times")
    if snapshot.as_of is not None:
        rows = [r for r in rows if r.timestamp <= snapshot.as_of]
    origins = list(range(min_history, len(rows) - horizon + 1, stride))[-folds:]
    truth = {r.valid_time: r for r in snapshot.series(name, frozen.loaded.variable)}
    planned = []
    for origin in origins:
        cutoff = rows[origin - 1].timestamp
        vintage = snapshot.narrow(as_of=cutoff, recorded_as_of=cutoff if replay == "recorded" else None)
        history = [r for r in vintage.series(name, frozen.loaded.variable) if r.valid_time <= cutoff]
        future = rows[origin:origin + horizon]
        actuals = [asdict(truth[r.timestamp]) for r in future]
        # JSON/ledger-safe timestamps keep both provenance clocks explicit.
        actuals = [{k: v.isoformat() if hasattr(v, "isoformat") else v for k, v in r.items()} for r in actuals]
        fold = {"origin": cutoff.isoformat(), "snapshot_id": vintage.snapshot_id,
                "actuals": actuals, "request": None, "runs": {}, "status": "ready"}
        try:
            if len(history) < min_history or history[-1].valid_time != cutoff:
                raise ForecastAdapterError("insufficient vintage history at origin")
            validate_and_group([Observation(r.valid_time, r.value, name) for r in history], frozen.loaded.frequency)
            request = ForecastRequest(
                tuple(r.value for r in history), horizon, season=season, frequency=frozen.loaded.frequency,
                timestamps=tuple(r.valid_time.isoformat() for r in history),
                future_timestamps=tuple(r.timestamp.isoformat() for r in future),
                cutoff=cutoff.isoformat(), known_time_cutoff=vintage.as_of.isoformat(),
                recorded_time_cutoff=vintage.recorded_as_of.isoformat() if vintage.recorded_as_of else None,
                snapshot_id=vintage.snapshot_id, series_id=name, unit=frozen.unit)
            for provider in providers:
                validate_capabilities(AdapterCapabilities(**identities[provider]["capabilities"]), request)
            from .inference import _freeze_request
            fold["request"] = asdict(_freeze_request(request))
        except (ForecastAdapterError, GnomonError, ValueError) as exc:
            fold["status"] = "unavailable_history_or_capability"
            fold["error_type"] = type(exc).__name__
        planned.append(fold)
    calls, stop_reason = 0, None
    for fold in planned:
        if fold["status"] != "ready":
            continue
        for provider in providers:
            if cancelled is not None and cancelled():
                stop_reason = "cancelled"
            elif calls >= budget.max_calls:
                stop_reason = "call_budget"
            elif budget.max_seconds is not None and timer() - started >= budget.max_seconds:
                stop_reason = "time_budget"
            if stop_reason:
                break
            calls += 1
            execution = None
            try:
                execution = engine.forecast(provider, ForecastRequest.from_dict(fold["request"]), use_cache=False)
                point_error_metrics(zip(execution.result.point, (r["value"] for r in fold["actuals"])))
                fold["runs"][provider] = {"status": "ok", "execution_id": execution.execution_id,
                                           "fingerprint": execution.fingerprint, "revision": execution.revision,
                                           "point": list(execution.result.point), "metadata": execution.result.metadata}
            except KeyboardInterrupt:
                fold["runs"][provider] = {"status": "cancelled"}
                stop_reason = "cancelled"
                break
            except Exception as exc:
                # Provider exception text can expose auth/service details.
                fold["runs"][provider] = {"status": "error", "error_type": type(exc).__name__}
                if execution is not None:
                    fold["runs"][provider]["execution_id"] = execution.execution_id
        if stop_reason:
            break
    elapsed = max(0.0, timer() - started)
    for fold in planned:
        if fold["status"] == "ready":
            statuses = [fold["runs"].get(p, {}).get("status") for p in providers]
            fold["status"] = ("complete" if all(s == "ok" for s in statuses) else
                              "failed" if "error" in statuses else "cancelled" if "cancelled" in statuses else
                              "partial" if fold["runs"] else "not_dispatched")
    matched = [f for f in planned if all(f["runs"].get(p, {}).get("status") == "ok" for p in providers)]
    diagnostics = {p: {"attempted": sum(p in f["runs"] for f in planned),
                       "succeeded": sum(f["runs"].get(p, {}).get("status") == "ok" for f in planned),
                       "failed": sum(f["runs"].get(p, {}).get("status") == "error" for f in planned)} for p in providers}
    scores = {p: point_error_metrics((point, row["value"]) for f in matched
                         for point, row in zip(f["runs"][p]["point"], f["actuals"])) for p in providers}
    ranking = sorted(providers, key=lambda p: scores[p]["mae"]) if matched else []
    score_groups = {}
    for provider in ranking:
        score_groups.setdefault(scores[provider]["mae"], []).append(provider)
    ties = [{"mae": score, "providers": group} for score, group in score_groups.items() if len(group) > 1]
    complete = len(matched) == folds
    issues = []
    required_rows = min_history + horizon + (folds - 1) * stride
    if len(origins) < folds:
        issues.append({"code": "INSUFFICIENT_HISTORY", "available_rows": len(rows),
                       "required_rows_for_one_fold": min_history + horizon,
                       "required_rows_for_requested_folds": required_rows,
                       "description": "Supply more observations or explicitly reduce folds, horizon or min_history."})
    if stop_reason:
        issues.append({"code": stop_reason.upper(), "description": "Evaluation stopped before all requested folds completed; inspect usage and budget."})
    if any(f["status"] == "unavailable_history_or_capability" for f in planned):
        issues.append({"code": "UNAVAILABLE_FOLDS", "description": "Historical vintages or provider capabilities do not support all requested folds."})
    if any(f["status"] == "failed" for f in planned):
        issues.append({"code": "PROVIDER_FAILURE", "description": "Inspect provider diagnostics and verify the provider environment."})
    cohort = {"folds": [{k: f[k] for k in ("origin", "request", "actuals")} for f in planned],
              "replay": replay, "horizon": horizon, "season": season, "series_id": name, "unit": frozen.unit}
    result = {"schema_version": "1", "study_id": str(uuid4()), "status": "complete" if complete else "partial" if matched else "unscored",
              "data_ref": data_ref, "snapshot_id": snapshot.snapshot_id,
              "source_fingerprint": frozen.loaded.source_fingerprint,
              'variable': frozen.loaded.variable,
              "cohort_id": content_id("cohort", cohort, length=64), "series_id": name,
              "unit": frozen.unit, "frequency": frozen.loaded.frequency, "horizon": horizon, "season": season,
              "baseline": baseline, "provider_order": providers, "providers": {p: identities[p] for p in providers},
              "budget": asdict(budget), "usage": {"provider_calls": calls, "elapsed_seconds": elapsed,
                  "requested_folds": folds, "planned_folds": len(planned), "matched_folds": len(matched),
                  "stop_reason": stop_reason, "wall_limit_overrun": budget.max_seconds is not None and elapsed > budget.max_seconds,
                  "dispatch_boundary_limits": True, "internal_model_calls": "unknown"},
              "replay": replay, "known_time_assumed": snapshot.assumed_known_time,
              "source_as_of": snapshot.as_of.isoformat() if snapshot.as_of else None,
              "recorded_as_of": snapshot.recorded_as_of.isoformat() if snapshot.recorded_as_of else None,
              "metric_version": "matched-point-errors/1", "scores": scores, "diagnostics": diagnostics,
              "ranking": ranking,
              "ranking_policy": {"metric": "mae", "direction": "ascending",
                                 "tie_comparison": "exact_unrounded_score",
                                 "tie_order": "provider_input_order", "ties": ties,
                                 "guidance": ("Tied providers have equal MAE on the matched folds; their order does not establish a winner. "
                                              "Choose among them using known cost, latency or simplicity, or collect more evidence. "
                                              "Equal MAE does not establish equivalent predictions or future performance."
                                              if ties else "Ranking uses MAE on matched folds; it does not establish future performance."
                                              if matched else "No matched folds were scored; there is no ranking or tie to interpret.")},
              "folds": planned, "evidence": "rolling_origin_backtest", "action_authorized": False,
              "training_cutoff_attested": False, "calibration": "not_established",
              "recorded": engine.ledger is not None}
    result["issues"] = issues
    if verify:
        from .diagnostics import scored_pairs
        result['score_derivations'] = {p: scored_pairs((point, actual['value']) for fold in matched
            for point, actual in zip(fold['runs'][p]['point'], fold['actuals'])) for p in providers}
    from .study_routing import DEFAULT_MIN_FOLDS
    readiness = references._readiness(frozen)["route"]
    readiness.update(matched_folds=len(matched), default_min_folds=DEFAULT_MIN_FOLDS,
                     scope="data_and_study_preflight_cutoffs_and_identity_checked_at_routing")
    if len(matched) < DEFAULT_MIN_FOLDS:
        readiness["issues"].append({"action": "evaluate_more_matched_folds",
            "description": f"Routing defaults to at least {DEFAULT_MIN_FOLDS} replayable matched folds; "
                           f"this study has {len(matched)}. Evaluate more folds with sufficient observed history and budget."})
    if engine.ledger is None:
        readiness["issues"].append({"action": "persist_study",
            "description": "Evaluate with --ledger-path evidence.db so routing can reuse recorded executions."})
    readiness["ready"] = not readiness["issues"]
    result["routing_readiness"] = readiness
    result['evaluation_status'] = result['status']
    result['routing_status'] = 'ready_for_cutoff_checks' if readiness['ready'] else 'insufficient_folds' if len(matched) < DEFAULT_MIN_FOLDS else 'input_or_persistence_unready'
    if engine.ledger is not None:
        engine.ledger.record_study(result)
    return result


def compact_study(report: dict) -> dict:
    """No repeated training vectors on the ordinary agent/CLI projection."""
    return {**{k: v for k, v in report.items() if k != "folds"},
            "study_evidence_scope": "fold_summary",
            "full_study": {"tool": "gnomon_evaluate", "arguments": {"study_id": report["study_id"]}},
            "folds": [{k: v for k, v in fold.items() if k not in {"request", "actuals", "runs"}}
                      for fold in report["folds"]]}

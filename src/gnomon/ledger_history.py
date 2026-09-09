"""Bounded, read-only production comparisons using the ledger's single-origin checks."""

from __future__ import annotations

from datetime import datetime, timedelta
from statistics import mean

from .contracts import GnomonError
from .forecast_adapter import ForecastAdapterError
from .ledger import _METRIC_VERSION, _bounded, _json, _time
from .temporal import is_regular_step, normalise_frequency


def _grid_shape(req):
    """Compare elapsed leads when unspecified; validate declared calendar grids."""
    history_end = datetime.fromisoformat(req["timestamps"][-1])
    origin = datetime.fromisoformat(req["cutoff"]) if req["cutoff"] else history_end
    future = [datetime.fromisoformat(t) for t in req["future_timestamps"]]
    if req["frequency"] is None:
        return {"lead_microseconds": [(t - origin) // timedelta(microseconds=1) for t in future]}
    try:
        frequency = normalise_frequency(req["frequency"])
    except (GnomonError, OverflowError):
        raise ForecastAdapterError("forecast_grid_unresolved") from None
    stamps = [history_end, *future]
    for left, right in zip(stamps, stamps[1:]):
        if not is_regular_step(left, right, frequency) or (
            frequency == "MS" and left.timetz().replace(tzinfo=None) != right.timetz().replace(tzinfo=None)
        ):
            raise ForecastAdapterError("forecast_grid_does_not_match_frequency")
    return {"frequency": frequency, "origin_lag_microseconds": (origin - history_end) // timedelta(microseconds=1)}


def compare_history(ledger, *, series_id, horizon, providers, start, end, source_as_of, recorded_as_of, unit):
    if not isinstance(series_id, str) or not series_id or series_id == "__default__":
        raise ForecastAdapterError("comparison requires an explicit stable series_id", details={
            "rejected_fields": ["series_id"], "choices_required": {"series_id": "Select the stable named series used in recorded forecasts; __default__ is not eligible."}})
    _bounded(horizon, "horizon", 1_000_000)
    if unit is not None and (not isinstance(unit, str) or not unit):
        raise ForecastAdapterError("unit must be a nonempty string or null")
    if not isinstance(providers, dict) or not 2 <= len(providers) <= 8 or any(
        not isinstance(p, str) or not p or not isinstance(v, str) or not v.strip() or v in {"latest", "unversioned"}
        for p, v in providers.items()
    ):
        raise ForecastAdapterError("providers must map 2 to 8 distinct names to explicit revisions")
    start, end, source, recorded = map(_time, (start, end, source_as_of, recorded_as_of))
    if start > end or end > source:
        raise ForecastAdapterError("require start <= end <= source_as_of")
    answer = {"status": "insufficient_evidence", "series_id": series_id, "unit": unit, "horizon": horizon,
              "providers": providers, "start": start, "end": end, "source_as_of": source, "recorded_as_of": recorded,
              "metric_version": _METRIC_VERSION, "aggregation": "mean_mae_over_complete_matched_origins",
              "matched_origins": 0, "n": 0, "models": [], "origins": [], "excluded": [], "duplicates_ignored": 0,
              "provider_calls": 0, "action_authorized": False, "model_identity_basis": "provider_declared_not_independently_attested",
              "next_step": "collect_matched_forecasts_with_explicit_budget"}
    origin_expr = "COALESCE(json_extract(p.payload_json, '$.request.cutoff'), " \
                  "json_extract(p.payload_json, '$.request.timestamps[#-1]'))"
    with ledger._connect() as conn:
        # One SQLite read snapshot: every origin/candidate sees the same revisions.
        conn.execute("BEGIN")
        rows = conn.execute("SELECT e.execution_id, " + origin_expr + " AS origin, "
            "EXISTS(SELECT 1 FROM study_executions s JOIN studies t USING(study_id) "
            "WHERE s.execution_id=e.execution_id AND t.recorded_at<=?) AS study_run "
            "FROM executions e JOIN payloads p USING(payload_id) WHERE "
            "json_extract(p.payload_json, '$.request.series_id')=? AND json_extract(p.payload_json, '$.request.horizon')=? "
            "AND json_extract(p.payload_json, '$.request.unit') IS ? AND e.recorded_at<=? "
            "AND julianday(" + origin_expr + ") BETWEEN julianday(?) AND julianday(?) "
            "AND json_extract(p.payload_json, '$.provider') IN (" + ",".join("?" for _ in providers) + ") "
            "ORDER BY e.rowid LIMIT 1001", (recorded, series_id, horizon, unit, recorded, start, end, *providers)).fetchall()
        if len(rows) > 1000:
            raise ForecastAdapterError("comparison exceeds 1000 executions; narrow the origin window; no partial ranking produced")
        groups = {}
        for row in rows:
            run = ledger._execution(conn, row["execution_id"])
            try:
                origin = _time(row["origin"])
            except ForecastAdapterError:
                answer["excluded"].append({"execution_id": run["execution_id"], "reason": "timezone_unresolved"})
                continue
            if not start <= origin <= end:
                continue
            reason = "not_production_evidence" if row["study_run"] else \
                     "provider_version_mismatch" if run["revision"] != providers[run["provider"]] else None
            if reason:
                answer["excluded"].append({"execution_id": run["execution_id"], "origin": origin, "reason": reason})
                continue
            groups.setdefault(origin, {}).setdefault(run["provider"], []).append(run)
        answer["observed_origins"] = len(groups)
        identities, task_shapes = {}, set()
        for origin, by_provider in sorted(groups.items()):
            missing = [p for p in providers if p not in by_provider]
            reason = "missing_candidates" if missing else None
            selected, causes = [], []
            for p, runs in by_provider.items():
                # Always first recorded, never the retry with the smallest loss.
                selected.append(runs[0])
                answer["duplicates_ignored"] += len(runs) - 1
                if len({r["fingerprint"] for r in runs}) != 1:
                    reason = "ambiguous_inputs_at_origin"
                if len({_json(r.get("provider_identity")) for r in runs}) != 1:
                    reason = "provider_identity_changed"
                req, identity = runs[0]["request"], runs[0].get("provider_identity")
                if not identity or identity.get("lifecycle") not in {"stateless", "fresh_per_request", "pretrained"}:
                    reason = "provider_identity_unrecorded"
                try:
                    future = [_time(t) for t in req["future_timestamps"]]
                    history = [_time(t) for t in req["timestamps"]]
                    if not history:
                        reason = "missing_history_timestamps"
                    elif not future:
                        reason = "missing_future_timestamps"
                    elif max(history) > origin:
                        reason = "history_after_forecast_origin"
                    elif min(future) <= origin:
                        reason = "target_not_after_forecast_origin"
                    elif runs[0]["recorded_at"] >= min(future):
                        reason = "forecast_not_recorded_before_target"
                    if origin > runs[0]["recorded_at"]:
                        reason = "forecast_origin_after_execution"
                    if req["known_time_cutoff"] and _time(req["known_time_cutoff"]) > origin:
                        reason = "source_cutoff_after_origin"
                    if req["recorded_time_cutoff"] and _time(req["recorded_time_cutoff"]) > runs[0]["recorded_at"]:
                        reason = "recorded_cutoff_after_execution"
                    if identity and identity.get("lifecycle") == "pretrained":
                        training = runs[0]["result"]["metadata"].get("training_cutoff")
                        if training is None or _time(training) > origin:
                            reason = "pretrained_training_cutoff_unattested_or_after_origin"
                except ForecastAdapterError:
                    reason = "timezone_unresolved"
                if reason:
                    causes.append({"provider": p, "execution_id": runs[0]["execution_id"], "reason": reason,
                        "required_fields": ["timestamps", "cutoff", "future_timestamps", "series_id", "unit"],
                        "forecast_recorded_at": runs[0]["recorded_at"],
                        "guidance": "Record prospective forecasts with timezone-aware history timestamps, origin and future timestamps before the first target. Existing executions are immutable; do not invent or backdate missing history. See the production comparison example."})
            if reason is None:
                try:
                    comparison = ledger._compare(conn, selected, source, recorded)
                    grid = _grid_shape(selected[0]["request"])
                    if comparison["n"] != horizon:
                        reason = "incomplete_actuals"
                except ForecastAdapterError as exc:
                    reason = str(exc)
            if reason is not None:
                answer["excluded"].append({"origin": origin, "reason": reason, "missing_providers": missing,
                                            "execution_ids": [r["execution_id"] for r in selected], "causes": causes})
                continue
            for run in selected:
                identities.setdefault(run["provider"], set()).add(_json(run["provider_identity"]))
            req = selected[0]["request"]
            task_shapes.add(_json({**{k: req.get(k) for k in ("season", "past_covariate_names",
                                                             "future_covariate_names", "quantiles", "samples")},
                                  "related_series_count": len(req["related_series"]), "forecast_grid": grid}))
            answer["origins"].append({"origin": origin, **comparison})
        count = len(answer["origins"])
        answer.update(matched_origins=count, n=count * horizon)
        answer["unique_actuals"] = len({aid for o in answer["origins"] for aid in o["actual_ids"]})
        # Do not silently pick whichever configuration's subset scores best.
        if len(task_shapes) > 1 or any(len(values) != 1 for values in identities.values()):
            answer.update(status="incompatible_evidence", reason="task_or_provider_identity_changed",
                          next_step="select_a_consistent_task_and_provider_version_window")
            return answer
        if count:
            answer["models"] = [{"provider": p, "revision": revision,
                "mae": mean(next(m["mae"] for m in o["models"] if m["provider"] == p)
                            for o in answer["origins"])} for p, revision in providers.items()]
            answer.update(status="ok", next_step="consider_evidence_and_sample_count_before_model_selection")
        elif any(e["reason"] == "incomplete_actuals" for e in answer["excluded"]):
            answer["next_step"] = "supply_missing_actuals"
        return answer

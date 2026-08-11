"""The canonical temporal workflows, as fixed, fully validated macros.

A. investigate_change — what changed?
B. forecast          — what happens next? (the existing runtime, registered)
C. decide            — what should we do?
D. monitor           — when should we intervene?
E. detect_anomalies  — what is abnormal?

Each macro is a hand-written pipeline over the deterministic operators in
``operators.py``: it loads through a snapshot, computes, assembles a typed
artifact with lineage, and passes the claim verifier before returning.
There is no LLM anywhere in this module; hosts propose tasks, Gnomon owns
every number.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .artifacts import write_json_artifact
from .contracts import (
    GnomonError,
    DataSourceRef,
    SupportAssessment,
    SupportReason,
    TemporalTask,
)
from .ids import SYSTEM_CLOCK, Clock, content_id
from .lineage import ArtifactRecord, ClaimRecord, EvidenceRecord, Lineage
from .operators import (
    anomaly_score,
    cross_correlation,
    evaluate_actions,
    event_study,
    inconclusive,
    regime_detection,
)
from .pipeline import load_stage
from .verifier import verify_or_raise
from .versioning import RUNTIME_VERSION


def _series_payloads(loaded) -> dict[str, tuple[list[datetime], list[float]]]:
    return {
        name: ([item.timestamp for item in items], [item.value for item in items])
        for name, items in sorted(loaded.groups.items())
    }


def _base_lineage(task: TemporalTask, loaded, artifact_id: str, kind: str,
                  created_at: str) -> tuple[Lineage, str]:
    dataset_id = f"dataset:{loaded.source_fingerprint.split(':')[-1][:24]}"
    summary = loaded.snapshot.access_summary()
    known_times = [
        str(access["max_known_time"]) for access in summary["accesses"]
        if access.get("max_known_time") is not None
    ]
    lineage = Lineage(task.task_id(), _task_dict(task))
    lineage.artifacts.append(ArtifactRecord(
        dataset_id, "dataset", created_at,
        fingerprint=loaded.source_fingerprint,
        max_known_time=max(known_times) if known_times else None,
        meta={"ref": task.sources[0].ref},
    ))
    lineage.artifacts.append(ArtifactRecord(artifact_id, kind, created_at))
    lineage.evidence.append(EvidenceRecord(
        "snapshot", "snapshot_access", "__all__", summary, (dataset_id,),
    ))
    return lineage, dataset_id


def _task_dict(task: TemporalTask) -> dict[str, Any]:
    from dataclasses import asdict
    return asdict(task)


def _stamp_provenance(task_payload: dict[str, Any], input_path: str,
                      input_provenance: str | None) -> dict[str, Any]:
    """The channel the observations arrived through, on the persisted task.

    Emitted only when it is not the implied default — a caller's file — so
    existing artifacts serialise byte-identically. ``store`` is derivable
    from the ref; ``inline`` is not: the inline channel materialises to a
    temp file before the loaders see it, so the tool layer is the only
    place that still knows, and it passes the fact down here. Applied to
    the artifact payload only, never to id material — the channel does not
    change the answer, so it must not change the id.
    """
    provenance = input_provenance or (
        "store" if str(input_path).startswith("store:") else None
    )
    if provenance:
        for source in task_payload.get("sources") or []:
            source["provenance"] = provenance
    return task_payload


def _event_payload(event: Any) -> Any:
    """Context events as id material, matching ``runtime.forecast``."""
    return getattr(event, "__dict__", event)


def _schema_payload(loaded, time_column: str, target_column: str,
                    series_column: str | None) -> dict[str, Any]:
    """The column and grid choices that change what a macro computes.

    ``source_fingerprint`` covers the bytes on disk, not which columns were
    read out of them, so an id built from the fingerprint alone collides
    across genuinely different reads of the same file. ``runtime.forecast``
    has always included this block; the other macros now do too.
    """
    return {
        "time": time_column,
        "target": target_column,
        "series": series_column,
        "frequency": loaded.frequency,
        "timezone": getattr(loaded, "timezone", None),
    }


# ---------------------------------------------------------------------------
# A. What changed? — gnomon_investigate_change
# ---------------------------------------------------------------------------

EVENT_PROXIMITY_WINDOW = 7


def investigate_change(
    input_path: str,
    *,
    time_column: str,
    target_column: str,
    series_column: str | None = None,
    frequency: str | None = None,
    as_of: datetime | None = None,
    context_events: list[Any] | None = None,
    output: str = "gnomon-output",
    store_path: str | None = None,
    clock: Clock | None = None,
    input_provenance: str | None = None,
) -> tuple[dict[str, Any], Path]:
    clock = clock or SYSTEM_CLOCK
    loaded = load_stage(
        input_path, time_column=time_column, target_column=target_column,
        series_column=series_column, frequency=frequency,
        as_of=as_of, store_path=store_path,
    )
    task = TemporalTask(
        objective=f"Investigate what changed in {target_column}",
        task_type="investigate_change",
        sources=(DataSourceRef(input_path, time_column, target_column, series_column, loaded.frequency),),
        outputs=("change_investigation",),
        as_of=as_of.isoformat() if as_of else None,
    )
    payloads = _series_payloads(loaded)
    results: list[dict[str, Any]] = []
    evidence_records: list[EvidenceRecord] = []
    claims: list[ClaimRecord] = []

    # Detect changes everywhere first: cross-series explanation ranking
    # needs every series' onsets.
    detections = {
        name: regime_detection(
            [moment.isoformat() for moment in timestamps], values,
        )
        for name, (timestamps, values) in payloads.items()
    }
    onsets = {
        name: detection["changepoints"][-1] if detection["changepoints"] else None
        for name, detection in detections.items()
    }

    for name, (timestamps, values) in payloads.items():
        detection = detections[name]
        iso_timestamps = [moment.isoformat() for moment in timestamps]
        anomalies = anomaly_score(iso_timestamps, values)
        evidence_records.append(EvidenceRecord(
            f"regime_detection:{name}", "regime_detection", name,
            {key: detection[key] for key in ("changepoints", "regimes", "classification", "support")},
            (f"dataset:{loaded.source_fingerprint.split(':')[-1][:24]}",),
        ))
        evidence_records.append(EvidenceRecord(
            f"anomaly_score:{name}", "anomaly_score", name,
            {"anomalies": anomalies["anomalies"], "support": anomalies["support"]},
        ))
        onset = onsets[name]
        explanations: list[dict[str, Any]] = []
        if onset is not None:
            onset_index = onset["index"]
            onset_time = timestamps[onset_index]
            # Concurrent events, via the existing context-event machinery.
            for event in context_events or []:
                event_start = datetime.fromisoformat(str(event.effective_start))
                if (event_start.tzinfo is None) != (onset_time.tzinfo is None):
                    continue
                delta = abs((event_start - onset_time).total_seconds()) / 86400.0
                if delta <= EVENT_PROXIMITY_WINDOW:
                    scope = getattr(event, "entity_scope", ()) or ()
                    if scope and name not in scope and "__default__" not in scope:
                        continue
                    study = event_study(iso_timestamps, values, [event_start.isoformat()])
                    evidence_id = f"event_study:{name}:{event.event_id}"
                    evidence_records.append(EvidenceRecord(
                        evidence_id, "event_study", name, study,
                    ))
                    explanations.append({
                        "kind": "concurrent_event",
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "distance_periods": delta,
                        "score": round(1.0 / (1.0 + delta), 4),
                        "evidence_id": evidence_id,
                    })
            # Cross-series precedence: another series whose change precedes
            # this onset, with a significant lead correlation.
            for other, other_onset in onsets.items():
                if other == name or other_onset is None:
                    continue
                lead = onset_index - other_onset["index"]
                if lead <= 0:
                    continue
                correlation = cross_correlation(payloads[other][1], values)
                evidence_id = f"cross_correlation:{other}->{name}"
                evidence_records.append(EvidenceRecord(
                    evidence_id, "cross_correlation", name, correlation,
                ))
                best = correlation.get("best")
                if best and best["significant"]:
                    explanations.append({
                        "kind": "cross_series_precedence",
                        "leading_series": other,
                        "lead_periods": lead,
                        "correlation": best["correlation"],
                        "score": round(abs(best["correlation"]) / (1.0 + lead), 4),
                        "evidence_id": evidence_id,
                    })
        explanations.sort(key=lambda item: -item["score"])
        residual_uncertainty = (
            "No candidate explanation was found; the change stands unexplained."
            if onset is not None and not explanations else
            "Explanations are associational rankings, not causes; concurrent "
            "unobserved influences are not controlled for."
            if explanations else None
        )
        support = detection["support"] if onset is None else detection["regimes"][-1]["support"]
        # "What changed?" answered in the reader's terms. The raw list is
        # kept verbatim; `ranked_changes` is the same changepoints ordered
        # by how much of the series' variation each explains, timestamped
        # rather than indexed, with the dominant one marked. A user asking
        # what changed was previously handed three `index:` values, one of
        # which (relative_gain 0.016) was noise beside the real one (0.779).
        ranked = sorted(
            detection["changepoints"],
            key=lambda item: item.get("relative_gain") or 0.0,
            reverse=True,
        )
        strongest = (ranked[0].get("relative_gain") or 0.0) if ranked else 0.0
        ranked_changes = [
            {
                "timestamp": (
                    item["timestamp"].isoformat()
                    if hasattr(item["timestamp"], "isoformat") else str(item["timestamp"])
                ),
                "before_mean": item.get("before_mean"),
                "after_mean": item.get("after_mean"),
                "shift": item.get("shift"),
                "share_of_variation_explained": item.get("relative_gain"),
                "dominant": bool(
                    strongest > 0
                    and (item.get("relative_gain") or 0.0) >= strongest * 0.5
                ),
            }
            for item in ranked
        ]
        result = {
            "series": name,
            "ranked_changes": ranked_changes,
            "changepoints": detection["changepoints"],
            "classification": detection["classification"],
            "onset": timestamps[onset["index"]].isoformat() if onset else None,
            "anomalies": anomalies["anomalies"],
            "explanations": explanations,
            "residual_uncertainty": residual_uncertainty,
            "support_assessment": support,
        }
        results.append(result)

        if onset is not None and support.get("status") in ("supported", "conditionally_supported"):
            claims.append(ClaimRecord(
                claim_id=f"claim:change:{name}",
                claim_class="descriptive",
                statement=(
                    f"Series {name} changed at {result['onset']} "
                    f"(classification: {detection['classification']}; mean shift "
                    f"{onset['shift']:+.4g})."
                ),
                subject=name,
                evidence_ids=(f"regime_detection:{name}",),
                artifact_ids=(),
            ))
            for rank, explanation in enumerate(explanations, 1):
                claims.append(ClaimRecord(
                    claim_id=f"claim:explanation:{name}:{rank}",
                    claim_class="associational",
                    statement=(
                        f"Candidate explanation {rank} for the change in {name}: "
                        f"{explanation['kind']} "
                        + (explanation.get("event_id") or explanation.get("leading_series", ""))
                        + " (associational ranking, not a cause)."
                    ),
                    subject=name,
                    evidence_ids=(explanation["evidence_id"],),
                    artifact_ids=(),
                ))
        elif onset is None and support.get("status") == "supported":
            claims.append(ClaimRecord(
                claim_id=f"claim:no_change:{name}",
                claim_class="descriptive",
                statement=f"No change was detected in series {name} beyond noise.",
                subject=name,
                evidence_ids=(f"regime_detection:{name}",),
                artifact_ids=(),
            ))

    artifact_id = content_id("investigation", {
        "runtime_version": RUNTIME_VERSION,
        "source": loaded.source_fingerprint,
        "as_of": task.as_of,
        "series": sorted(payloads),
        "schema": _schema_payload(loaded, time_column, target_column, series_column),
        "context_events": (
            [_event_payload(event) for event in context_events]
            if context_events else None
        ),
    })
    created_at = clock.now().isoformat()
    payload = {
        "schema_version": "0.1",
        "runtime_version": RUNTIME_VERSION,
        "investigation_id": artifact_id,
        "created_at": created_at,
        "status": "complete",
        "task": _stamp_provenance(_task_dict(task), input_path, input_provenance),
        "source_fingerprint": loaded.source_fingerprint,
        "results": results,
    }
    lineage, dataset_id = _base_lineage(task, loaded, artifact_id, "investigation", created_at)
    lineage.evidence.extend(evidence_records)
    lineage.claims.extend([
        ClaimRecord(
            claim.claim_id, claim.claim_class, claim.statement, claim.subject,
            claim.evidence_ids, (artifact_id, dataset_id),
        )
        for claim in claims
    ])
    verify_or_raise(lineage, as_of=task.as_of)
    return payload, write_json_artifact(artifact_id, payload, output, lineage=lineage.to_dict())


# ---------------------------------------------------------------------------
# C. What should we do? — gnomon_decide
# ---------------------------------------------------------------------------

def decide(
    input_path: str,
    *,
    time_column: str,
    target_column: str,
    horizon: int,
    threshold: float,
    actions: list[dict[str, Any]],
    utilities: dict[str, dict[str, float]] | None = None,
    max_acceptable_risk: float | None = None,
    series_column: str | None = None,
    series_name: str | None = None,
    frequency: str | None = None,
    as_of: datetime | None = None,
    project: str | None = None,
    output: str = "gnomon-output",
    store_path: str | None = None,
    clock: Clock | None = None,
    input_provenance: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Scenario generation → feasible actions → uncertainty propagation →
    constraints/costs (degraded without utilities) → choose or abstain.
    With a project, the decision is recorded as a DecisionArtifact for
    realised-outcome scoring."""
    from .runtime import forecast as run_forecast
    clock = clock or SYSTEM_CLOCK
    artifact, forecast_dir = run_forecast(
        input_path, time_column=time_column, target_column=target_column,
        horizon=horizon, series_column=series_column, frequency=frequency,
        threshold=threshold, output=output, as_of=as_of,
        store_path=store_path, clock=clock,
    )
    candidates = [item for item in artifact.results
                  if series_name is None or item.series == series_name]
    if series_name is not None and not candidates:
        raise GnomonError("SERIES_NOT_FOUND", f"No series named {series_name!r} in the dataset.")
    if len(candidates) != 1:
        raise GnomonError(
            "MULTIPLE_SERIES_UNSUPPORTED",
            "gnomon_decide evaluates one series at a time; select one with series_name.",
            {"series": [item.series for item in artifact.results]},
        )
    result = candidates[0]
    task = TemporalTask(
        objective=f"Decide among {len(actions)} actions given exceedance risk of {target_column} over {threshold}",
        task_type="decide",
        sources=(DataSourceRef(input_path, time_column, target_column, series_column, frequency),),
        outputs=("decision",),
        as_of=as_of.isoformat() if as_of else None,
    )
    created_at = clock.now().isoformat()

    if not result.forecast or not result.threshold:
        evaluation: dict[str, Any] = {"evaluations": [], "selected": None}
        # Two distinct honest refusals: no rows at all, or rows published
        # below the calibrated tiers (best_effort / horizon split) — which
        # carry no exceedance probabilities to ground a decision on. A
        # decision must never rest on naive fallback rows, labelled or not.
        if result.forecast:
            support = inconclusive(
                "forecast_not_calibrated",
                "The underlying forecast published only sub-supported rows "
                "(best_effort or a horizon split); they carry no calibrated "
                "exceedance risk, so no decision is offered. The labelled "
                "rows remain in the forecast artifact.",
                "Provide more history or lower the horizon so the forecast "
                "is fully evaluated, then re-run.",
            ).to_dict()
        else:
            support = inconclusive(
                "forecast_abstained",
                "The underlying forecast abstained, so exceedance risk cannot be "
                "grounded; no decision is offered.",
                "Provide more history or lower the horizon, then re-run.",
            ).to_dict()
        scenario_probabilities: dict[str, float] | None = None
        exceedance: dict[str, Any] | None = None
    else:
        # The per-step maximum, not the probability of at least one
        # exceedance across the horizon — which is >= this and is usually
        # what a horizon-level decision turns on. Labelled `exceed` with no
        # definition, it understated the risk feeding expected utility.
        step_probabilities = result.threshold["probability_above"]
        peak = max(step_probabilities)
        # Any-step exceedance under an independence assumption, reported
        # beside the peak rather than replacing it: the steps are not
        # independent, so this is an upper reference, not a better number.
        any_step = 1.0
        for probability in step_probabilities:
            any_step *= (1.0 - probability)
        any_step = round(1.0 - any_step, 4)
        # The two scenarios the expected-utility comparison sums over. Kept
        # to exactly these keys because `evaluate_actions` weights payoffs
        # by them; the definition and the alternative reading ride alongside
        # in `exceedance` rather than inside the probability mass.
        scenario_probabilities = {"exceed": peak, "no_exceed": round(1.0 - peak, 4)}
        exceedance = {
            "peak_step_exceedance": peak,
            "any_step_exceedance_if_independent": any_step,
            "per_step": step_probabilities,
            "event_definition": (
                "`exceed` is the largest single-step probability of crossing "
                "the threshold, and is what the expected-utility comparison "
                "uses. `any_step_exceedance_if_independent` is the "
                "probability of at least one crossing over the horizon under "
                "an independence assumption the steps do not satisfy — an "
                "upper reference, not a substitute. For a horizon-level "
                "decision the true any-step probability lies between them."
            ),
        }
        evaluation = evaluate_actions(
            actions, scenario_probabilities,
            utilities=utilities, max_acceptable_risk=max_acceptable_risk,
        )
        support = evaluation["support"]
        # Uncertainty propagation: a decision grounded on a warned or
        # degraded forecast cannot claim more support than the forecast has.
        if result.warnings:
            forecast_reasons = [
                {"code": "forecast_warning", "message": warning}
                for warning in result.warnings
            ]
            support = {
                **support,
                "status": ("conditionally_supported"
                           if support["status"] == "supported" else support["status"]),
                "reasons": list(support.get("reasons", [])) + forecast_reasons,
            }

    decision_id = content_id("decision", {
        "runtime_version": RUNTIME_VERSION,
        "forecast": artifact.forecast_id,
        "series": result.series,
        "threshold": threshold,
        # Every field of every action, not just its name: `feasible` and
        # `residual_risk` select a different action, so an id over names
        # alone let two runs with different answers share one artifact.
        "actions": [
            {str(key): action[key] for key in sorted(action)}
            for action in actions
        ],
        "utilities": utilities,
        "max_acceptable_risk": max_acceptable_risk,
    })
    payload = {
        "schema_version": "0.1",
        "runtime_version": RUNTIME_VERSION,
        "decision_id": decision_id,
        "created_at": created_at,
        "status": "complete",
        "task": _stamp_provenance(_task_dict(task), input_path, input_provenance),
        "series": result.series,
        "forecast_id": artifact.forecast_id,
        "forecast_artifact_path": str(forecast_dir),
        "threshold": threshold,
        "scenario_probabilities": scenario_probabilities,
        "exceedance": exceedance,
        "evaluation": {key: value for key, value in evaluation.items() if key != "support"},
        "support_assessment": support,
    }

    lineage = Lineage(task.task_id(), _task_dict(task))
    lineage.artifacts.append(ArtifactRecord(
        artifact.forecast_id, "forecast", artifact.created_at,
        meta={"path": str(forecast_dir)},
    ))
    lineage.artifacts.append(ArtifactRecord(decision_id, "decision", created_at))
    lineage.evidence.append(EvidenceRecord(
        f"evaluation:{result.series}", "rolling_evaluation", result.series,
        {"selection_scores": result.selection_scores, "test_scores": result.test_scores},
        (artifact.forecast_id,),
    ))
    if result.threshold:
        lineage.evidence.append(EvidenceRecord(
            f"threshold_risk:{result.series}", "threshold_analysis", result.series,
            result.threshold, (artifact.forecast_id,),
        ))
    lineage.evidence.append(EvidenceRecord(
        f"action_evaluation:{result.series}", "action_evaluation", result.series,
        {key: value for key, value in evaluation.items() if key != "support"},
    ))
    if evaluation.get("selected") is not None:
        lineage.claims.append(ClaimRecord(
            claim_id=f"claim:decision:{result.series}",
            claim_class="decision",
            statement=(
                f"Under the stated utilities and constraints, action "
                f"{evaluation['selected']!r} maximises expected utility for "
                f"series {result.series}."
            ),
            subject=result.series,
            evidence_ids=(f"threshold_risk:{result.series}", f"action_evaluation:{result.series}"),
            artifact_ids=(decision_id, artifact.forecast_id),
            constraints_evaluated=True,
        ))
    if scenario_probabilities is not None:
        lineage.claims.append(ClaimRecord(
            claim_id=f"claim:risk:{result.series}",
            claim_class="predictive",
            statement=(
                f"Per-step probability of series {result.series} exceeding "
                f"{threshold}, peak {scenario_probabilities['exceed']:.1%}."
            ),
            subject=result.series,
            evidence_ids=(f"evaluation:{result.series}", f"threshold_risk:{result.series}"),
            artifact_ids=(decision_id, artifact.forecast_id),
            calibration_ref=f"evaluation:{result.series}",
        ))
    verify_or_raise(lineage, as_of=task.as_of)
    if project:
        from .decision_model import ActionOption, DecisionArtifact
        from .tracking import TrackingStore, register_artifact
        register_artifact(artifact, project, str(forecast_dir))
        store = TrackingStore()
        store.save_decision_artifact(DecisionArtifact(
            decision_id=decision_id,
            project=project,
            forecast_id=artifact.forecast_id,
            options=[
                ActionOption(
                    name=item["name"], feasible=item["feasible"],
                    constraint_results=item.get("constraint_results", {}),
                    expected_utility=item.get("expected_utility"),
                    downside_risk=item.get("downside"),
                )
                for item in evaluation.get("evaluations", [])
            ],
            selected_action=evaluation.get("selected"),
            decision_rule=evaluation.get("decision_rule"),
            scenario_probabilities=scenario_probabilities,
            utilities=utilities,
            assumptions=[reason["message"] for reason in support.get("reasons", [])],
            sensitivity=support.get("sensitivity", {}),
            created_at=created_at,
        ))
        payload["project"] = project
    return payload, write_json_artifact(decision_id, payload, output, lineage=lineage.to_dict())


# ---------------------------------------------------------------------------
# D. When should we intervene? — gnomon_monitor
# ---------------------------------------------------------------------------

def monitor(
    input_path: str,
    *,
    time_column: str,
    target_column: str,
    horizon: int,
    threshold: float,
    alert_cost: float | None = None,
    miss_cost: float | None = None,
    series_column: str | None = None,
    frequency: str | None = None,
    as_of: datetime | None = None,
    project: str | None = None,
    output: str = "gnomon-output",
    store_path: str | None = None,
    clock: Clock | None = None,
    input_provenance: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Trigger definition → sequential risk estimation → alert-cost-aware
    thresholding, building on the tracking store's open-forecast lifecycle."""
    from .runtime import forecast as run_forecast
    clock = clock or SYSTEM_CLOCK
    artifact, forecast_dir = run_forecast(
        input_path, time_column=time_column, target_column=target_column,
        horizon=horizon, series_column=series_column, frequency=frequency,
        threshold=threshold, output=output, as_of=as_of,
        store_path=store_path, clock=clock,
    )
    task = TemporalTask(
        objective=f"Monitor {target_column} for exceedance of {threshold}",
        task_type="monitor",
        sources=(DataSourceRef(input_path, time_column, target_column, series_column, frequency),),
        outputs=("monitor",),
        as_of=as_of.isoformat() if as_of else None,
    )
    created_at = clock.now().isoformat()

    costed = alert_cost is not None and miss_cost is not None
    if costed and (alert_cost < 0 or miss_cost <= 0):
        raise GnomonError("INVALID_COSTS", "alert_cost must be >= 0 and miss_cost > 0.")
    alert_probability = alert_cost / (alert_cost + miss_cost) if costed else 0.5

    triggers = []
    for result in artifact.results:
        if not result.forecast or not result.threshold:
            if result.forecast:
                assessment = inconclusive(
                    "forecast_not_calibrated",
                    "The underlying forecast published only sub-supported "
                    "rows (best_effort or a horizon split); sequential risk "
                    "cannot be estimated from uncalibrated rows, so this "
                    "trigger is not armed.",
                    "Provide more history or lower the horizon so the "
                    "forecast is fully evaluated, then re-run.",
                )
            else:
                assessment = inconclusive(
                    "forecast_abstained",
                    "The underlying forecast abstained; sequential risk cannot "
                    "be estimated for this series.",
                    "Provide more history, then re-run.",
                )
            triggers.append({
                "series": result.series,
                "armed": False,
                "support_assessment": assessment.to_dict(),
            })
            continue
        probabilities = result.threshold["probability_above"]
        first_alert = next(
            (step for step, probability in enumerate(probabilities, 1)
             if probability >= alert_probability),
            None,
        )
        reasons = [] if costed else [SupportReason(
            "missing_cost_inputs",
            "No alert/miss costs were supplied; the alert rule uses the "
            "uninformative 0.5 probability threshold.",
        )]
        reasons += [SupportReason("forecast_warning", warning) for warning in result.warnings]
        triggers.append({
            "series": result.series,
            "armed": True,
            "trigger": {"threshold": threshold, "direction": "above"},
            "probability_above_per_step": probabilities,
            "alert_probability_threshold": alert_probability,
            "alert_rule_basis": (
                "alert when P(exceed) >= alert_cost / (alert_cost + miss_cost)"
                if costed else "default 0.5 threshold (no costs supplied)"
            ),
            "first_alert_step": first_alert,
            "first_alert_timestamp": (
                result.forecast[first_alert - 1]["timestamp"] if first_alert else None
            ),
            "support_assessment": SupportAssessment(
                "supported" if costed and not result.warnings else "conditionally_supported",
                reasons,
                recovery_actions=[] if costed else [SupportReason(
                    "provide_costs",
                    "Supply alert_cost and miss_cost to get a cost-optimal rule.",
                )],
            ).to_dict(),
        })

    monitor_id = content_id("monitor", {
        "runtime_version": RUNTIME_VERSION,
        "forecast": artifact.forecast_id,
        "threshold": threshold,
        "alert_cost": alert_cost, "miss_cost": miss_cost,
        "horizon": horizon,
    })
    payload = {
        "schema_version": "0.1",
        "runtime_version": RUNTIME_VERSION,
        "monitor_id": monitor_id,
        "created_at": created_at,
        "status": "complete",
        "task": _stamp_provenance(_task_dict(task), input_path, input_provenance),
        "forecast_id": artifact.forecast_id,
        "forecast_artifact_path": str(forecast_dir),
        "triggers": triggers,
    }
    if project:
        from .tracking import register_artifact
        payload["tracking_ids"] = register_artifact(artifact, project, str(forecast_dir))
        payload["project"] = project

    lineage = Lineage(task.task_id(), _task_dict(task))
    lineage.artifacts.append(ArtifactRecord(
        artifact.forecast_id, "forecast", artifact.created_at,
        meta={"path": str(forecast_dir)},
    ))
    lineage.artifacts.append(ArtifactRecord(monitor_id, "monitor", created_at))
    for result in artifact.results:
        lineage.evidence.append(EvidenceRecord(
            f"evaluation:{result.series}", "rolling_evaluation", result.series,
            {"selection_scores": result.selection_scores},
            (artifact.forecast_id,),
        ))
        if result.threshold:
            lineage.evidence.append(EvidenceRecord(
                f"threshold_risk:{result.series}", "threshold_analysis", result.series,
                result.threshold, (artifact.forecast_id,),
            ))
            lineage.claims.append(ClaimRecord(
                claim_id=f"claim:sequential_risk:{result.series}",
                claim_class="predictive",
                statement=(
                    f"Sequential exceedance risk for series {result.series} over "
                    f"{threshold}, with the alert step under the stated rule."
                ),
                subject=result.series,
                evidence_ids=(f"evaluation:{result.series}", f"threshold_risk:{result.series}"),
                artifact_ids=(monitor_id, artifact.forecast_id),
                calibration_ref=f"evaluation:{result.series}",
            ))
    verify_or_raise(lineage, as_of=task.as_of)
    return payload, write_json_artifact(monitor_id, payload, output, lineage=lineage.to_dict())


# ---------------------------------------------------------------------------
# E. What is abnormal? — gnomon_detect_anomalies
# ---------------------------------------------------------------------------

def detect_anomalies(
    input_path: str,
    *,
    time_column: str,
    target_column: str,
    series_column: str | None = None,
    frequency: str | None = None,
    as_of: datetime | None = None,
    threshold: float | None = None,
    labels: list[str] | None = None,
    include_tsfm: bool = True,
    output: str = "gnomon-output",
    store_path: str | None = None,
    clock: Clock | None = None,
    input_provenance: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Graded anomaly detection: candidate detectors compete on a
    synthetic-injection grader (or on supplied labels), the winner labels
    the series, and every candidate's grade ships in the artifact.

    Installed multi-task TSFM sandboxes (reconstruction-error scoring)
    join the candidate pool automatically and must win the grader like
    everyone else; ``include_tsfm=False`` keeps the pool statistical."""
    from .anomaly import (
        DEFAULT_THRESHOLD,
        detect_anomalies as run_detection,
        tsfm_reconstruction_detectors,
    )
    from .temporal import detect_season
    clock = clock or SYSTEM_CLOCK
    detection_threshold = DEFAULT_THRESHOLD if threshold is None else float(threshold)
    loaded = load_stage(
        input_path, time_column=time_column, target_column=target_column,
        series_column=series_column, frequency=frequency,
        as_of=as_of, store_path=store_path,
    )
    task = TemporalTask(
        objective=f"Detect anomalies in {target_column}",
        task_type="detect_anomalies",
        sources=(DataSourceRef(input_path, time_column, target_column, series_column, loaded.frequency),),
        outputs=("anomaly_detection",),
        as_of=as_of.isoformat() if as_of else None,
    )
    label_moments = []
    for label in labels or []:
        label_moments.append(datetime.fromisoformat(str(label)))
    extra_detectors = tsfm_reconstruction_detectors() if include_tsfm else {}
    payloads = _series_payloads(loaded)
    results: list[dict[str, Any]] = []
    evidence_records: list[EvidenceRecord] = []
    claims: list[ClaimRecord] = []
    for name, (timestamps, values) in payloads.items():
        season, _, _ = detect_season(values, loaded.frequency)
        index_of = {moment: index for index, moment in enumerate(timestamps)}
        label_indices = [index_of[m] for m in label_moments if m in index_of]
        detection = run_detection(
            [moment.isoformat() for moment in timestamps], values,
            season=season, threshold=detection_threshold,
            label_indices=label_indices or None,
            extra_detectors=extra_detectors or None,
        )
        evidence_records.append(EvidenceRecord(
            f"anomaly_detection:{name}", "anomaly_detection", name,
            {key: detection[key] for key in
             ("detector", "selection_basis", "detector_grades", "injection",
              "anomalies", "support") if key in detection},
        ))
        result = {
            "series": name,
            "detector": detection["detector"],
            "selection_basis": detection.get("selection_basis"),
            "detector_grades": detection.get("detector_grades", {}),
            "anomalies": detection.get("anomalies", []),
            "support_assessment": detection["support"],
        }
        if "label_grades" in detection:
            result["label_grades"] = detection["label_grades"]
        results.append(result)
        status = detection["support"].get("status")
        if status in ("supported", "conditionally_supported") and detection["detector"]:
            count = len(detection["anomalies"])
            if count:
                claims.append(ClaimRecord(
                    claim_id=f"claim:anomalies:{name}",
                    claim_class="descriptive",
                    statement=(
                        f"Series {name}: {count} anomalies flagged by "
                        f"{detection['detector']} (selected by "
                        f"{detection['selection_basis']}; grader F1 "
                        f"{detection['detector_grades'][detection['detector']]['f1']:.2f})."
                    ),
                    subject=name,
                    evidence_ids=(f"anomaly_detection:{name}",),
                    artifact_ids=(),
                ))
            elif status == "supported":
                claims.append(ClaimRecord(
                    claim_id=f"claim:no_anomalies:{name}",
                    claim_class="descriptive",
                    statement=(
                        f"No anomalies beyond threshold "
                        f"{detection_threshold} were detected in series {name}."
                    ),
                    subject=name,
                    evidence_ids=(f"anomaly_detection:{name}",),
                    artifact_ids=(),
                ))
    artifact_id = content_id("anomaly", {
        "runtime_version": RUNTIME_VERSION,
        "source": loaded.source_fingerprint,
        "as_of": task.as_of,
        "series": sorted(payloads),
        "threshold": detection_threshold,
        "schema": _schema_payload(loaded, time_column, target_column, series_column),
        # Labels change the detector *and* the basis it is selected on, so
        # a labelled and an unlabelled run are different computations.
        "labels": sorted(str(label) for label in labels) if labels else None,
        "include_tsfm": include_tsfm,
    })
    created_at = clock.now().isoformat()
    payload = {
        "schema_version": "0.1",
        "runtime_version": RUNTIME_VERSION,
        "anomaly_id": artifact_id,
        "created_at": created_at,
        "status": "complete",
        "task": _stamp_provenance(_task_dict(task), input_path, input_provenance),
        "source_fingerprint": loaded.source_fingerprint,
        "threshold": detection_threshold,
        "results": results,
    }
    lineage, dataset_id = _base_lineage(task, loaded, artifact_id, "anomaly_detection", created_at)
    lineage.evidence.extend(evidence_records)
    lineage.claims.extend([
        ClaimRecord(
            claim.claim_id, claim.claim_class, claim.statement, claim.subject,
            claim.evidence_ids, (artifact_id, dataset_id),
        )
        for claim in claims
    ])
    verify_or_raise(lineage, as_of=task.as_of)
    return payload, write_json_artifact(artifact_id, payload, output, lineage=lineage.to_dict())

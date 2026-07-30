"""The four canonical temporal workflows, as fixed, fully validated macros.

A. investigate_change — what changed?
B. forecast          — what happens next? (the existing runtime, registered)
C. decide            — what should we do?
D. monitor           — when should we intervene?

Each macro is a hand-written pipeline over the deterministic operators in
``operators.py``: it loads through a snapshot, computes, assembles a typed
artifact with lineage, and passes the claim verifier before returning.
There is no LLM anywhere in this module; hosts propose tasks, Aion owns
every number.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .artifacts import write_json_artifact
from .contracts import (
    AionError,
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


# ---------------------------------------------------------------------------
# A. What changed? — aion_investigate_change
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
    output: str = "aion-output",
    store_path: str | None = None,
    clock: Clock | None = None,
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
        result = {
            "series": name,
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
        "source": loaded.source_fingerprint,
        "as_of": task.as_of,
        "series": sorted(payloads),
    })
    created_at = clock.now().isoformat()
    payload = {
        "schema_version": "0.1",
        "investigation_id": artifact_id,
        "created_at": created_at,
        "status": "complete",
        "task": _task_dict(task),
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
# C. What should we do? — aion_decide
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
    output: str = "aion-output",
    store_path: str | None = None,
    clock: Clock | None = None,
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
        raise AionError("SERIES_NOT_FOUND", f"No series named {series_name!r} in the dataset.")
    if len(candidates) != 1:
        raise AionError(
            "MULTIPLE_SERIES_UNSUPPORTED",
            "aion_decide evaluates one series at a time; select one with series_name.",
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
        support = inconclusive(
            "forecast_abstained",
            "The underlying forecast abstained, so exceedance risk cannot be "
            "grounded; no decision is offered.",
            "Provide more history or lower the horizon, then re-run.",
        ).to_dict()
        scenario_probabilities: dict[str, float] | None = None
    else:
        peak = max(result.threshold["probability_above"])
        scenario_probabilities = {"exceed": peak, "no_exceed": round(1.0 - peak, 4)}
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
        "forecast": artifact.forecast_id,
        "series": result.series,
        "threshold": threshold,
        "actions": [action.get("name") for action in actions],
        "utilities": utilities,
        "max_acceptable_risk": max_acceptable_risk,
    })
    payload = {
        "schema_version": "0.1",
        "decision_id": decision_id,
        "created_at": created_at,
        "status": "complete",
        "task": _task_dict(task),
        "series": result.series,
        "forecast_id": artifact.forecast_id,
        "forecast_artifact_path": str(forecast_dir),
        "threshold": threshold,
        "scenario_probabilities": scenario_probabilities,
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
# D. When should we intervene? — aion_monitor
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
    output: str = "aion-output",
    store_path: str | None = None,
    clock: Clock | None = None,
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
        raise AionError("INVALID_COSTS", "alert_cost must be >= 0 and miss_cost > 0.")
    alert_probability = alert_cost / (alert_cost + miss_cost) if costed else 0.5

    triggers = []
    for result in artifact.results:
        if not result.forecast or not result.threshold:
            triggers.append({
                "series": result.series,
                "armed": False,
                "support_assessment": inconclusive(
                    "forecast_abstained",
                    "The underlying forecast abstained; sequential risk cannot "
                    "be estimated for this series.",
                    "Provide more history, then re-run.",
                ).to_dict(),
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
        "forecast": artifact.forecast_id,
        "threshold": threshold,
        "alert_cost": alert_cost, "miss_cost": miss_cost,
    })
    payload = {
        "schema_version": "0.1",
        "monitor_id": monitor_id,
        "created_at": created_at,
        "status": "complete",
        "task": _task_dict(task),
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

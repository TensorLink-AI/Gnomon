"""Unregistered historical surface experiments, retained for explicit diagnostics.

These helpers have no MCP profile or public tool registration. Ordinary sessions
never import this module; their independent numeric dependencies stay regression-tested.
"""

from __future__ import annotations

from typing import Any

def _run_unified(arguments: dict[str, Any]) -> dict[str, Any]:
    """Retired experiment; not registered on any live tool surface."""
    from .toolspec import (_run_describe, _run_forecast, _run_investigate_change,
                           _run_detect_anomalies, _run_decide, _run_monitor,
                           _default_forecast_horizon)
    from .contracts import GnomonError

    question = arguments.get("question") or {}
    # Models commonly send the discriminant directly (``"forecast"``)
    # despite the advertised object schema. Accept that unambiguous natural
    # form instead of throwing AttributeError inside the tool server.
    if isinstance(question, str):
        kind, question_fields = question, {}
    elif isinstance(question, dict):
        kind = question.get("kind")
        question_fields = {key: value for key, value in question.items()
                           if key != "kind"}
    else:
        raise GnomonError("INVALID_ARGUMENTS",
                          "question must be an object or a kind string.",
                          {"allowed": ["describe", "forecast", "investigate",
                                       "detect", "decide", "monitor"]})
    merged = {**arguments, **question_fields}
    merged.pop("question", None)
    if kind == "robust_decision":
        from datetime import datetime, timezone
        from .decision_model import robust_scenario_decision
        from .tracking import TrackingStore

        required = ("decision_id", "project", "forecast_id", "actions",
                    "utilities", "scenario_ids")
        missing = [name for name in required if not merged.get(name)]
        if missing:
            raise GnomonError(
                "INVALID_ARGUMENTS",
                "robust_decision needs a complete stated utility matrix.",
                {"missing": missing},
            )
        try:
            artifact = robust_scenario_decision(
                decision_id=str(merged["decision_id"]),
                project=str(merged["project"]),
                forecast_id=str(merged["forecast_id"]),
                actions=list(merged["actions"]),
                utilities=dict(merged["utilities"]),
                scenario_ids=list(merged["scenario_ids"]),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        except (TypeError, ValueError) as exc:
            raise GnomonError("INVALID_ARGUMENTS", str(exc)) from exc
        TrackingStore().save_decision_artifact(artifact)
        return {"schema_version": "0.1", "decision": artifact.to_dict()}
    runners = {
        "describe": _run_describe,
        "forecast": _run_forecast,
        "investigate": _run_investigate_change,
        "detect": _run_detect_anomalies,
        "decide": _run_decide,
        "monitor": _run_monitor,
    }
    runner = runners.get(str(kind))
    if runner is None:
        raise GnomonError("INVALID_ARGUMENTS", "question.kind is required.",
                          {"allowed": sorted(runners)})
    if kind == "forecast" and merged.get("horizon") is None:
        merged["horizon"] = _default_forecast_horizon(merged)
    return runner(merged)


def _run_track(arguments: dict[str, Any]) -> dict[str, Any]:
    from .toolspec import _run_status, _run_submit_actuals, _run_resolve_outcome
    from .contracts import GnomonError

    action = arguments.get("action")
    if action == "status":
        return _run_status(arguments)
    if action == "submit_actuals":
        return _run_submit_actuals(arguments)
    if action == "resolve_outcome":
        return _run_resolve_outcome(arguments)
    if action == "record_adapter_shadow":
        from .tracking import TrackingStore
        return TrackingStore().record_adapter_shadow_outcome(
            project=str(arguments["project"]),
            outcome_id=str(arguments["outcome_id"]),
            candidate=str(arguments["candidate"]),
            revision=arguments.get("revision"),
            baseline=str(arguments["baseline"]),
            candidate_error=float(arguments["candidate_error"]),
            baseline_error=float(arguments["baseline_error"]),
            known_at=str(arguments["known_at"]),
            regime={str(key): str(value) for key, value in
                    dict(arguments.get("regime") or {}).items()} or None,
        )
    if action == "assess_adapter_shadow":
        from .tracking import TrackingStore
        return TrackingStore().assess_adapter_shadow(
            project=str(arguments["project"]),
            candidate=str(arguments["candidate"]),
            revision=arguments.get("revision"),
            baseline=str(arguments["baseline"]),
            as_of=arguments.get("as_of"),
            min_outcomes=int(arguments.get("min_outcomes", 30)),
            min_improvement=float(arguments.get("min_improvement", .05)),
            min_win_rate=float(arguments.get("min_win_rate", .60)),
        )
    if action == "route_adapter_shadow":
        from .tracking import TrackingStore
        return TrackingStore().route_adapter_shadow(
            project=str(arguments["project"]),
            candidate=str(arguments["candidate"]),
            revision=arguments.get("revision"),
            champion=str(arguments["baseline"]),
            regime={str(key): str(value) for key, value in
                    dict(arguments.get("regime") or {}).items()},
            as_of=str(arguments["as_of"]),
        )
    if action == "record_synthesis":
        from .tracking import TrackingStore
        TrackingStore().record_temporal_synthesis(
            project=str(arguments["project"]),
            forecast_id=str(arguments["forecast_id"]),
            series=str(arguments["series"]),
            question_id=str(arguments["question_id"]),
            synthesis_id=str(arguments["synthesis_id"]),
            canonical=dict(arguments["canonical"]),
            synthesis=dict(arguments["synthesis"]),
            evidence_refs=[str(item) for item in arguments["evidence_refs"]],
        )
        return {"status": "recorded", "synthesis_id": arguments["synthesis_id"],
                "primary_forecast_unchanged": True}
    if action == "resolve_synthesis":
        from .tracking import TrackingStore
        score = TrackingStore().resolve_temporal_synthesis(
            project=str(arguments["project"]),
            forecast_id=str(arguments["forecast_id"]),
            series=str(arguments["series"]),
            question_id=str(arguments["question_id"]),
            synthesis_id=str(arguments["synthesis_id"]),
            outcome=dict(arguments["outcome"]),
            resolved_at=arguments.get("resolved_at"),
        )
        return {"status": "resolved", "synthesis_id": arguments["synthesis_id"],
                "score": score, "primary_forecast_unchanged": True}
    if action == "synthesis_status":
        from .tracking import TrackingStore
        rows = TrackingStore().temporal_synthesis_receipts(
            str(arguments["project"]), resolved=arguments.get("resolved"),
            series=arguments.get("series"),
            resolved_before=arguments.get("as_of"))
        return {"status": "ok", "project": arguments["project"],
                "syntheses": rows}
    if action == "candidate_outcomes":
        from .tracking import TrackingStore
        rows = TrackingStore().candidate_outcome_summary(
            str(arguments["project"]),
            minimum_resolved=int(arguments.get("min_outcomes", 8)),
            series=arguments.get("series"),
            resolved_before=arguments.get("as_of"))
        return {
            "status": "ok", "project": arguments["project"],
            "series": arguments.get("series"),
            "as_of": arguments.get("as_of"),
            "candidate_outcomes": rows,
            "authority": {
                "human_prior_only": True,
                "support_upgrade_allowed": False,
                "automation_upgrade_allowed": False,
            },
        }
    if action == "decision_skill":
        from .tracking import TrackingStore
        rows = TrackingStore().decision_synthesis_skill(
            str(arguments["project"]),
            proposer_id=arguments.get("proposer_id"),
            minimum_resolved=int(arguments.get("min_outcomes", 20)))
        return {
            "status": "ok", "project": arguments["project"],
            "decision_skill": rows,
            "authority": {
                "human_prior_only": True,
                "support_upgrade_allowed": False,
                "automation_upgrade_allowed": False,
            },
        }
    raise GnomonError("INVALID_ARGUMENTS", "action is required.",
                      {"allowed": ["status", "submit_actuals", "resolve_outcome",
                                   "record_adapter_shadow",
                                   "assess_adapter_shadow",
                                   "route_adapter_shadow", "record_synthesis",
                                   "resolve_synthesis", "synthesis_status",
                                   "candidate_outcomes", "decision_skill"]})

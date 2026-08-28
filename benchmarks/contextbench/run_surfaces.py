"""Run paired ContextBench cases through Gnomon's real MCP profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.envfile import load_env_file  # noqa: E402
from benchmarks.common.openrouter import OpenRouterClient  # noqa: E402
from benchmarks.common.openrouter import OpenRouterError  # noqa: E402
from benchmarks.temporalbench.mcp_agent import run_row  # noqa: E402
from gnomon.artifacts import read_artifact  # noqa: E402

from .generate import EPOCH as CONTEXT_EPOCH  # noqa: E402
from .run_contextbench import frequency_step  # noqa: E402
from .run_contextbench import smape, valid_disposition, wilson  # noqa: E402
from .run_llm import compile_events  # noqa: E402
from .schema import Case, Oracle, load_cases, load_oracles  # noqa: E402

PROFILES = {"core", "describe", "evidence", "mega", "full"}
ROUTING_POLICIES = {"compiled", "unrouted"}
BASELINE_MODES = {"engine", "agent"}


def _usage_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Request usage accrued between two points on one case-local client."""
    keys = ("prompt_tokens", "completion_tokens", "requests",
            "transport_attempts", "cost_usd", "truncation_escalations")
    return {key: after.get(key, 0) - before.get(key, 0) for key in keys}


def _usage_sum(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    keys = ("prompt_tokens", "completion_tokens", "requests",
            "transport_attempts", "cost_usd", "truncation_escalations")
    return {key: sum((row.get("llm_usage") or {}).get(scope, {}).get(key, 0)
                     for row in rows) for key in keys}


def _checkpoint(path: Path, cases: list[Case],
                rows_by_id: dict[str, dict[str, Any]]) -> None:
    """Atomically persist every completed case in corpus order."""
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(
        json.dumps(rows_by_id[case.case_id], sort_keys=True) + "\n"
        for case in cases if case.case_id in rows_by_id))
    temporary.replace(path)


def _append_attempt(path: Path, row: dict[str, Any], attempt: int) -> None:
    """Append one immutable execution attempt; canonical rows may be replaced."""
    payload = {**row, "attempt": attempt,
               "recorded_at": datetime.now().astimezone().isoformat()}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _public_id(case_id: str) -> str:
    """Stable neutral ID: generated case IDs encode their hidden family."""
    return "cb-" + hashlib.sha256(case_id.encode()).hexdigest()[:16]


def _covariate_input(case: Case) -> tuple[dict[str, list[float]],
                                           dict[str, list[float]]]:
    names = [str(item["name"]) for item in case.covariate_mapping]
    if not names:
        return {}, {}
    split = len(case.history)
    historical = {
        name: [float(row[name]) for row in case.covariates[:split]]
        for name in names
    }
    future = {
        name: [float(row[name]) for row in case.covariates[split:split + case.horizon]]
        for name in names
    }
    return historical, future


def surface_row(case: Case, *, include_context: bool,
                routing_policy: str = "compiled") -> dict[str, Any]:
    if routing_policy not in ROUTING_POLICIES:
        raise ValueError(f"unknown routing policy: {routing_policy}")
    historical, future = _covariate_input(case)
    history: dict[str, list[float]] = {"value": list(case.history)}
    if include_context:
        history.update(historical)
    task_input: dict[str, Any] = {"history": history}
    if include_context and future:
        task_input["future_covariates"] = future
        task_input["covariate_mapping"] = [dict(item)
                                             for item in case.covariate_mapping]
    narrative = case.narrative.strip() if include_context else (
        "No outside context is supplied. Forecast from the numeric history only.")
    prompt = (
        "Forecast the next " + str(case.horizon) +
        " values of the value series. Use only cutoff-safe material. "
        "Outside context may be useful, irrelevant, or insufficient to size; "
        "do not invent an effect. Submit a Gnomon artifact when the engine "
        "produces one, otherwise use an explicitly labeled alternative.\n\n" +
        narrative
    )
    if include_context:
        prompt += (
            "\n\nIn the submitted reasoning, briefly state whether context "
            "changed the primary, remained a scenario, or was rejected. "
            "Preserve any interval limitation and automation restriction "
            "reported by Gnomon; do not infer authority from the narrative."
        )
    if routing_policy == "compiled":
        # The host already binds the validated dataset and forecast arguments
        # to the execution call. Repeating hundreds of values in the model
        # transcript adds cost and creates a second, fallible representation
        # of the same data without helping the routing decision.
        prompt += "\n\nThe validated dataset is bound by the host."
    else:
        prompt += ("\n\nInput (JSON):\n" +
                   json.dumps(history, separators=(",", ":")))
    row = {
        "id": _public_id(case.case_id) + ("-context" if include_context else "-history"),
        "tier": "T4" if include_context else "T2",
        "meta": {"n_horizon": case.horizon, "main_key": "value",
                 "target_keys": ["value"]},
        "prompt": prompt, "input": task_input,
        # Only the mapping key is consumed by the harness. Oracle values never
        # enter this object or the model transcript.
        "ground_truth": {"value": None}, "mcq": {},
        # Every product arm must execute Gnomon before it may publish values.
        # Compiled routing resolves explicit forecast intent to the execution
        # verb; the unrouted diagnostic leaves that choice to the agent.
        "_require_gnomon_execution": True,
        "_host_compiled_forecast": routing_policy == "compiled",
        # ContextBench measures whether the model preserves the tool's
        # disposition in its human-facing synthesis.  The generic adapter's
        # immediate host submission is correct for forecast-only tasks but
        # would skip that turn and make explanation fidelity unobservable.
        "_require_context_explanation": include_context,
        # The generic TemporalBench adapter otherwise uses its own synthetic
        # epoch. ContextBench events are absolute-time evidence, so silently
        # rebasing the values makes every historical event unmeasurable.
        "_time_origin": CONTEXT_EPOCH.isoformat(),
        "_time_step_seconds": frequency_step(case.frequency).total_seconds(),
        "_frequency": case.frequency,
    }
    _validate_temporal_grid(case, row)
    return row


def _validate_temporal_grid(case: Case, row: dict[str, Any]) -> None:
    """Fail before LLM calls when absolute context cannot overlap the CSV."""
    origin = datetime.fromisoformat(str(row["_time_origin"]))
    final = origin + (len(case.history) + case.horizon - 1) * frequency_step(
        case.frequency)
    for event in case.context_events:
        start = datetime.fromisoformat(str(event["effective_start"]))
        end = datetime.fromisoformat(str(
            event.get("effective_end") or event["effective_start"]))
        if end < origin or start > final:
            raise ValueError(
                f"{case.case_id}: context event {event.get('event_id')!r} "
                f"does not overlap the benchmark grid {origin.isoformat()} "
                f"through {final.isoformat()}")
    for covariate in case.covariates:
        stamp = datetime.fromisoformat(str(covariate["timestamp"]))
        if stamp < origin or stamp > final:
            raise ValueError(
                f"{case.case_id}: covariate timestamp {stamp.isoformat()} "
                "does not overlap the benchmark grid")


def compiled_surface_context(case: Case, client: OpenRouterClient,
                             receipt_dir: Path) -> dict[str, Any]:
    """Compile once per replicate using ContextBench's bounded chunker."""
    receipt_dir.mkdir(parents=True, exist_ok=True)
    path = receipt_dir / (_public_id(case.case_id) + ".json")
    fingerprint = hashlib.sha256(json.dumps({
        "narrative": case.narrative,
        "document_known_at": sorted({
            str(event.get("known_at")) for event in case.context_events
            if event.get("known_at")}),
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if path.is_file():
        cached = json.loads(path.read_text())
        if cached.get("narrative_sha256") != fingerprint:
            raise ValueError("cached surface context does not match narrative")
        return {**cached["validated"], "compiler_called": False,
                "receipt_reused": True}
    compiled = compile_events(case, client)
    validated = {
        "events": compiled.get("events") or [],
        "hypotheses": compiled.get("hypotheses") or [],
        "rejected": compiled.get("rejected") or [],
        "compiler_called": bool(compiled.get("compiler_called")),
        "compiler_calls": int(compiled.get("compiler_calls", 0)),
        "receipt_reused": False,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"narrative_sha256": fingerprint,
                                     "validated": validated},
                                    indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return validated


def _forecast(outcome: dict[str, Any]) -> list[float] | None:
    values = ((outcome.get("answer") or {}).get("forecast") or {}).get("value")
    return ([float(item) for item in values]
            if isinstance(values, list) else None)


def _artifact_contract(outcome: dict[str, Any], forecast: list[float] | None
                       ) -> tuple[bool | None, bool]:
    paths = ((outcome.get("mcp") or {}).get("artifact_paths") or [])
    if not paths or forecast is None:
        return None, False
    parity = False
    leakage = False
    for path in paths:
        try:
            artifact = read_artifact(path)
        except Exception:
            continue
        for result in artifact.get("results") or []:
            points = [float(row.get("q50", row.get("point")))
                      for row in result.get("forecast") or []]
            if points == forecast:
                parity = True
        for evidence in artifact.get("evidence") or []:
            if evidence.get("kind") != "snapshot_access":
                continue
            for access in (evidence.get("payload") or {}).get("accesses", []):
                known = access.get("known_time")
                cutoff = access.get("cutoff_time")
                if known and cutoff and datetime.fromisoformat(str(known)) > \
                        datetime.fromisoformat(str(cutoff)):
                    leakage = True
    return parity, leakage


def preserves_automation_limit(
    explanation: str, *, restricted: bool, projection: dict[str, Any],
) -> bool:
    """Require typed parity plus an unambiguous human-facing authority limit."""
    if not restricted:
        return True
    lowered = explanation.lower()
    return projection.get("matched") is True and any(
        token in lowered for token in (
            "not eligible", "ineligible", "cannot automate",
            "cannot authorize", "no automation", "eligibility is false",
            "eligible is false", "automation_eligible=false",
        ))


def run_case(case: Case, oracle: Oracle, client: OpenRouterClient, profile: str,
             work_root: Path, receipt_dir: Path,
             routing_policy: str = "compiled",
             baseline_mode: str = "engine",
             tool_timeout: float = 120.0) -> dict[str, Any]:
    if baseline_mode not in BASELINE_MODES:
        raise ValueError(f"unknown baseline mode: {baseline_mode}")
    started = time.perf_counter()
    usage_start = dict(client.usage_summary)
    usage_after_history = usage_start
    usage_after_compiler = usage_start
    stage_seconds: dict[str, float] = {}

    def usage() -> dict[str, Any]:
        final = dict(client.usage_summary)
        return {
            "agent": {
                key: _usage_delta(usage_start, final)[key]
                     - _usage_delta(usage_after_history,
                                    usage_after_compiler)[key]
                for key in _usage_delta(usage_start, final)
            },
            "compiler": _usage_delta(usage_after_history,
                                     usage_after_compiler),
            "total": _usage_delta(usage_start, final),
        }

    stage = "history_engine" if baseline_mode == "engine" else "history_agent"
    stage_started = time.perf_counter()
    try:
        if baseline_mode == "engine":
            from .run_contextbench import _forecast as engine_forecast
            from .run_contextbench import _points as engine_points

            baseline_root = work_root / case.case_id / "surface-baseline"
            baseline_root.mkdir(parents=True, exist_ok=True)
            baseline_artifact, baseline_path = engine_forecast(
                case, baseline_root, enriched=False)
            baseline = engine_points(baseline_artifact.results[0])
            history = {
                "answer": {"forecast": {"value": baseline}},
                "mcp": {"calls": 0, "schema_bytes": 0,
                        "artifact_paths": [str(baseline_path)]},
                "channel_route": {"value": "gnomon-engine"},
            }
        else:
            history = run_row(surface_row(
                case, include_context=False, routing_policy=routing_policy),
                client, work_dir=str(work_root), profile=profile,
                mcp_call_timeout=tool_timeout)
            baseline = _forecast(history)
        stage_seconds[stage] = round(time.perf_counter() - stage_started, 6)
        usage_after_history = dict(client.usage_summary)
        stage = "context_compiler"
        stage_started = time.perf_counter()
        context_row = surface_row(
            case, include_context=True, routing_policy=routing_policy)
        try:
            context_row["_validated_context"] = compiled_surface_context(
                case, client, receipt_dir)
        finally:
            # Failed compiler calls are compiler cost too; attribution must
            # not depend on the stage succeeding.
            usage_after_compiler = dict(client.usage_summary)
            stage_seconds[stage] = round(
                time.perf_counter() - stage_started, 6)
        stage = "context_agent"
        stage_started = time.perf_counter()
        contextual = run_row(context_row, client,
                             work_dir=str(work_root), profile=profile,
                             compile_context=True,
                             context_receipts_dir=str(receipt_dir),
                             mcp_call_timeout=tool_timeout)
        stage_seconds[stage] = round(time.perf_counter() - stage_started, 6)
        enriched = _forecast(contextual)
        if baseline is None or enriched is None:
            missing = []
            if baseline is None:
                missing.append("history")
            if enriched is None:
                missing.append("context")
            return {
                "case_id": case.case_id, "public_id": _public_id(case.case_id),
                "family": case.family, "status": "product_failure",
                "failure_class": "agent_non_submission",
                "failure_stage": "+".join(missing),
                "error": "agent did not submit " + "+".join(missing) +
                         " forecast trajectory",
                "should_influence": oracle.should_influence,
                "history_calls": int((history.get("mcp") or {}).get("calls", 0)),
                "context_calls": int((contextual.get("mcp") or {}).get("calls", 0)),
                "history_abstention": history.get("row_abstained"),
                "context_abstention": contextual.get("row_abstained"),
                "history_mcp": history.get("mcp") or {},
                "context_mcp": contextual.get("mcp") or {},
                "routing_policy": routing_policy,
                "baseline_mode": baseline_mode,
                "stage_seconds": stage_seconds,
                "llm_usage": usage(), "usage_accounting_version": 2,
                "latency_seconds": round(time.perf_counter() - started, 6),
            }
        if len(baseline) != case.horizon or len(enriched) != case.horizon:
            raise ValueError("submitted trajectory has the wrong horizon")
        context_gate = (contextual.get("context_execution") or {}).get("value") or {}
        covariate_gate = (contextual.get("covariate_execution") or {}).get("value") or {}
        applied = bool(context_gate.get("applied") or covariate_gate.get("admitted"))
        disposition = str(context_gate.get("status") or
                          ("applied" if covariate_gate.get("admitted") else
                           "not_considered"))
        if case.family == "future_covariate" and not applied:
            disposition = "rejected"
        changed = [index for index, pair in enumerate(zip(baseline, enriched))
                   if abs(pair[0] - pair[1]) > 1e-9]
        parity, leakage = _artifact_contract(contextual, enriched)
        agent_reasoning = str(contextual.get("submit_reasoning") or "").strip()
        explanation = agent_reasoning.lower()
        interval_weak = (
            context_gate.get("selected_output_role") ==
            "interval_weak_context_scenario")
        primary_projection = contextual.get(
            "context_primary_projection") or {}
        preserved_primary = (
            primary_projection.get("matched") is True and
            "primary" in explanation and any(
                token in explanation for token in (
                    "unchanged", "preserved", "remain", "did not change")))
        represented_scenario = (
            disposition != "scenario_only" or
            any(token in explanation for token in (
                "scenario", "conditional", "what-if", "what if")))
        preserved_interval_limit = (
            not interval_weak or
            ("interval" in explanation and any(
                token in explanation for token in (
                    "weak", "coverage", "failed", "unreliable"))))
        automation_projection = contextual.get(
            "context_automation_projection") or {}
        automation_restricted = (
            context_gate.get("automation_eligible") is False)
        preserved_automation_limit = preserves_automation_limit(
            explanation, restricted=automation_restricted,
            projection=automation_projection)
        citations = contextual.get("context_gate_citations") or {}
        expected_citations = list(citations.get("expected") or [])
        rejection_evidence_cited = (
            not expected_citations or (
                bool(citations.get("matched")) and
                not citations.get("invalid")))
        return {
            "case_id": case.case_id, "public_id": _public_id(case.case_id),
            "family": case.family, "status": "answered",
            "history_smape": smape(oracle.actual, baseline),
            "context_smape": smape(oracle.actual, enriched),
            "incremental_smape": (smape(oracle.actual, baseline)
                                  - smape(oracle.actual, enriched)),
            "history_forecast": baseline, "context_forecast": enriched,
            "should_influence": oracle.should_influence,
            "oracle_dimensions": dict(oracle.dimensions),
            "primary_changed": bool(changed), "changed_steps": changed,
            "applied": applied, "disposition": disposition,
            "admission_rejection_reasons": (
                list(context_gate.get("rejection_codes") or []) + [
                    str(item.get("reason")) for item in
                    (covariate_gate.get("rejected") or [])
                    if isinstance(item, dict) and item.get("reason")
                ] if not applied else []),
            "expected_disposition": oracle.expected_disposition,
            "disposition_valid": valid_disposition(
                case.family, applied, disposition),
            "publication_parity": parity, "temporal_leakage": leakage,
            "agent_reasoning": agent_reasoning,
            "context_explanation_contract": {
                "required": True,
                "primary_preserved": preserved_primary,
                "scenario_represented": represented_scenario,
                "interval_limit_preserved": preserved_interval_limit,
                "automation_limit_preserved": preserved_automation_limit,
                "rejection_evidence_cited": rejection_evidence_cited,
                "complete": all((
                    preserved_primary, represented_scenario,
                    preserved_interval_limit, preserved_automation_limit,
                    rejection_evidence_cited,
                )),
            },
            "context_gate_citations": citations,
            "context_automation_projection": automation_projection,
            "context_primary_projection": primary_projection,
            "history_route": (history.get("channel_route") or {}).get("value"),
            "context_route": (contextual.get("channel_route") or {}).get("value"),
            "history_calls": int((history.get("mcp") or {}).get("calls", 0)),
            "context_calls": int((contextual.get("mcp") or {}).get("calls", 0)),
            "history_schema_bytes": int((history.get("mcp") or {}).get(
                "schema_bytes", 0)),
            "context_schema_bytes": int((contextual.get("mcp") or {}).get(
                "schema_bytes", 0)),
            "context_tool_sequence": list((contextual.get("mcp") or {}).get(
                "tool_sequence") or []),
            "product_schema_bytes": int((contextual.get("mcp") or {}).get(
                "product_schema_bytes", 0)),
            "harness_schema_bytes": int((contextual.get("mcp") or {}).get(
                "harness_schema_bytes", 0)),
            "surface_required_calls": 1 if baseline_mode == "engine" else 2,
            "redundant_calls": max(0, int((history.get("mcp") or {}).get(
                "calls", 0)) - 1) + max(0, int((contextual.get("mcp") or {}).get(
                "calls", 0)) - 1),
            "routing_policy": routing_policy,
            "baseline_mode": baseline_mode,
            "stage_seconds": stage_seconds,
            "compiler_called": int(bool(context_row["_validated_context"].get(
                "compiler_called"))),
            "compiler_calls": int(context_row["_validated_context"].get(
                "compiler_calls", 0)),
            "compiler_receipt_reused": bool((contextual.get("mcp") or {}).get(
                "context_receipt_reused")),
            "llm_usage": usage(), "usage_accounting_version": 2,
            "latency_seconds": round(time.perf_counter() - started, 6),
        }
    except Exception as error:
        stage_seconds.setdefault(
            stage, round(time.perf_counter() - stage_started, 6))
        infrastructure = isinstance(error, OpenRouterError)
        return {"case_id": case.case_id, "public_id": _public_id(case.case_id),
                "family": case.family, "status": "error",
                "failure_class": ("provider_failure" if infrastructure
                                  else "harness_failure"),
                "failure_stage": stage,
                "error": f"{type(error).__name__}: {error}",
                "should_influence": oracle.should_influence,
                "routing_policy": routing_policy,
                "baseline_mode": baseline_mode,
                "stage_seconds": stage_seconds,
                "llm_usage": usage(), "usage_accounting_version": 2,
                "latency_seconds": round(time.perf_counter() - started, 6)}


def summarize(rows: list[dict[str, Any]], profile: str,
              manifest: dict[str, Any], routing_policy: str,
              attempts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    attempts = attempts if attempts is not None else rows
    answered = [row for row in rows if row.get("status") == "answered"]
    product_failures = [row for row in rows
                        if row.get("status") == "product_failure"]
    provider_failures = [row for row in rows
                         if row.get("failure_class") == "provider_failure"]
    harness_failures = [row for row in rows
                        if row.get("failure_class") == "harness_failure"]
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in answered:
        families[row["family"]].append(row)
    empirical = [row for row in answered if (row.get("oracle_dimensions") or {}).get(
        "admission_warrant", "empirical") == "empirical"]
    influence = [row for row in empirical if row["should_influence"]]
    applied = [row for row in empirical if row["applied"]]
    false_trials = [row for row in empirical if not row["should_influence"]]
    false_asserted = [row for row in answered
                      if (row.get("oracle_dimensions") or {}).get(
                          "admission_warrant") == "asserted"
                      and not row["should_influence"]]
    false_changes = sum(row["primary_changed"] for row in false_trials)
    precision = (sum(row["should_influence"] for row in applied) / len(applied)
                 if applied else 1.0)
    recall = (sum(row["applied"] for row in influence) / len(influence)
              if influence else 0.0)
    missed = [row for row in influence if not row["applied"]]
    calls = [row["history_calls"] + row["context_calls"] for row in answered]
    observed_stages = sorted({stage for row in attempts
                              for stage in (row.get("stage_seconds") or {})})
    dimension_groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for dimension in ("frequency", "narrative_style", "effect_shape",
                      "claim_truth", "known_at", "confounding"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in answered:
            value = (row.get("oracle_dimensions") or {}).get(dimension)
            if value is not None:
                groups[str(value)].append(row)
        if groups:
            dimension_groups[dimension] = groups
    return {
        "benchmark": "contextbench-surfaces", "profile": profile,
        "routing_policy": routing_policy,
        "seed": manifest["seed"], "fresh_seed": manifest["fresh_seed"],
        "cases": len(rows), "answered": len(answered),
        "errors": len(rows) - len(answered),
        "completion_rate": len(answered) / len(rows) if rows else 0.0,
        "failure_denominators": {
            "attempted_cases": len(rows),
            "successful_pairs": len(answered),
            "agent_completion_failures": len(product_failures),
            "provider_failures": len(provider_failures),
            "harness_failures": len(harness_failures),
            "by_stage": dict(sorted(Counter(
                str(row.get("failure_stage") or "unknown")
                for row in rows if row.get("status") != "answered").items())),
        },
        "families": {family: {
            "cases": len(members),
            "history_smape": mean(row["history_smape"] for row in members),
            "context_smape": mean(row["context_smape"] for row in members),
            "incremental_smape": mean(row["incremental_smape"] for row in members),
            "applied": sum(row["applied"] for row in members),
            "primary_changed": sum(row["primary_changed"] for row in members),
        } for family, members in sorted(families.items())},
        "dimensions": {dimension: {value: {
            "cases": len(members),
            "incremental_smape": mean(
                row["incremental_smape"] for row in members),
            "applied": sum(bool(row["applied"]) for row in members),
        } for value, members in sorted(groups.items())}
            for dimension, groups in sorted(dimension_groups.items())},
        "metrics": {
            "admission_precision": precision, "admission_recall": recall,
            "missed_influence_cases": len(missed),
            "missed_influence_by_family": dict(sorted(Counter(
                str(row["family"]) for row in missed).items())),
            "admission_rejection_reasons": dict(sorted(Counter(
                reason for row in missed
                for reason in row.get("admission_rejection_reasons", [])
            ).items())),
            "disposition_accuracy": (mean(bool(row.get("disposition_valid"))
                                          for row in answered)
                                     if answered else None),
            "false_influence_rate": (false_changes / len(false_trials)
                                     if false_trials else 0.0),
            "false_influence_95ci": wilson(false_changes, len(false_trials)),
            "false_asserted_claim_primary_change_rate": (
                mean(bool(row["primary_changed"]) for row in false_asserted)
                if false_asserted else None),
            "false_asserted_claim_mean_incremental_smape": (
                mean(float(row["incremental_smape"]) for row in false_asserted)
                if false_asserted else None),
            "leakage_count": sum(row["temporal_leakage"] for row in answered),
            "publication_parity": {
                "eligible_answered": len(answered),
                "identity_produced": sum(
                    row.get("publication_parity") is not None for row in answered),
                "identity_matched": sum(
                    row.get("publication_parity") is True for row in answered),
                "match_rate_over_answered": (mean(
                    row["publication_parity"] is True for row in answered)
                    if answered else None),
            },
            # Compatibility alias for existing result readers.
            "publication_parity_rate": (mean(
                row["publication_parity"] is True for row in answered)
                if answered else None),
            "context_explanation_contract": {
                key: (mean(bool((row.get(
                    "context_explanation_contract") or {}).get(key))
                           for row in answered) if answered else None)
                for key in (
                    "complete", "primary_preserved", "scenario_represented",
                    "interval_limit_preserved", "automation_limit_preserved",
                    "rejection_evidence_cited",
                )
            },
            "observed_agent_calls_mean": mean(calls) if calls else None,
            "observed_agent_calls_median": (
                sorted(calls)[len(calls) // 2] if calls else None),
            "surface_required_calls": (answered[0].get(
                "surface_required_calls") if answered else None),
            # Compatibility aliases for older report consumers.
            "paired_calls_mean": mean(calls) if calls else None,
            "paired_calls_median": sorted(calls)[len(calls) // 2] if calls else None,
        },
        "llm_usage": _usage_sum(attempts, "total"),
        "llm_usage_observations": {
            "accounting_complete": all(
                row.get("usage_accounting_version") == 2 for row in attempts),
            "scope": "all_execution_attempts",
            "agent": _usage_sum(attempts, "agent"),
            "compiler": _usage_sum(attempts, "compiler"),
            "total": _usage_sum(attempts, "total"),
        },
        "execution_attempts": len(attempts),
        "stage_latency_seconds_mean": {
            stage: mean(float(row["stage_seconds"][stage]) for row in attempts
                        if stage in (row.get("stage_seconds") or {}))
            for stage in observed_stages
        },
        "retried_cases": sum(count > 1 for count in Counter(
            row["case_id"] for row in attempts).values()),
        "receipt_generations_this_run": sum(
            row.get("compiler_called", 0) for row in answered),
        "receipt_generation_model_calls": sum(
            row.get("compiler_calls", 0) for row in answered
            if row.get("compiler_called")),
        "receipt_logical_model_calls": sum(
            row.get("compiler_calls", 0) for row in answered),
        "receipts_reused": sum(
            row.get("compiler_receipt_reused", False) for row in answered),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--routing-policy", choices=sorted(ROUTING_POLICIES),
                        default="compiled")
    parser.add_argument(
        "--baseline-mode", choices=sorted(BASELINE_MODES), default="engine",
        help=("engine reuses a direct deterministic history baseline and "
              "spends agent calls only on the product surface; agent retains "
              "the legacy two-conversation diagnostic"))
    parser.add_argument("--replicate-id", default="1",
                        help="stable replicate label used for cross-surface pairing")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="ENGY_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--reasoning-effort", default=None,
                        choices=("none", "low", "medium", "high"))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int,
                        help="balanced development subset; omit for the 80-case run")
    parser.add_argument("--case-id", action="append", default=[],
                        help="run only this exact case ID; repeat as needed")
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--tool-timeout", type=float, default=120.0,
                        help="hard timeout in seconds for each MCP server call")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--context-receipts-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true",
                        help="with --resume, rerun prior error rows; otherwise retain them")
    parser.add_argument("--infrastructure-retries", type=int, default=2,
                        help="bounded in-run retries for provider/harness failures")
    args = parser.parse_args()
    load_env_file()
    corpus = Path(args.corpus_dir)
    manifest = json.loads((corpus / "manifest.json").read_text())
    cases = load_cases(corpus / "cases.jsonl")
    oracles = load_oracles(corpus / "oracle.jsonl")
    if hashlib.sha256((corpus / "cases.jsonl").read_bytes()).hexdigest() != \
            manifest["cases_sha256"]:
        raise SystemExit("cases hash mismatch")
    if hashlib.sha256((corpus / "oracle.jsonl").read_bytes()).hexdigest() != \
            manifest["oracle_sha256"]:
        raise SystemExit("oracle hash mismatch")
    if args.limit:
        grouped: dict[str, list[Case]] = defaultdict(list)
        for case in cases:
            grouped[case.family].append(case)
        selected: list[Case] = []
        while len(selected) < args.limit and any(grouped.values()):
            for family in sorted(grouped):
                if grouped[family] and len(selected) < args.limit:
                    selected.append(grouped[family].pop(0))
        cases = selected
    if args.case_id:
        requested = set(args.case_id)
        known = {case.case_id for case in cases}
        missing = sorted(requested - known)
        if missing:
            raise SystemExit("unknown or filtered case IDs: " + ", ".join(missing))
        cases = [case for case in cases if case.case_id in requested]
    client_kwargs = {
        "api_key": os.environ.get(args.api_key_env),
        "base_url": args.base_url, "temperature": args.temperature,
        "timeout": args.request_timeout, "max_retries": args.max_retries,
        "max_tokens": 8000,
        "reasoning_effort": args.reasoning_effort,
    }
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    observation_path = output / "observations.jsonl"
    attempt_path = output / "attempts.jsonl"
    retained: dict[str, dict[str, Any]] = {}
    if args.resume and observation_path.is_file():
        for line in observation_path.read_text().splitlines():
            row = json.loads(line)
            # A non-submission is an observed product outcome, not flaky
            # infrastructure. Never erase it by retrying. --retry-errors is
            # reserved for provider/harness failures.
            if (row.get("status") in {"answered", "product_failure"}
                    or not args.retry_errors):
                retained[row["case_id"]] = row
    pending = [case for case in cases if case.case_id not in retained]
    work = Path(tempfile.mkdtemp(prefix="contextbench-surfaces-", dir=str(output)))
    rows_by_id = dict(retained)
    attempts = ([json.loads(line) for line in attempt_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
        if args.resume and attempt_path.is_file() else [])
    if args.resume and not attempts:
        # Legacy checkpoint: retain its terminal costs once, then all future
        # retries append normally.
        attempts = list(retained.values())
        for row in attempts:
            _append_attempt(attempt_path, row, 1)
    attempt_counts = defaultdict(int)
    for row in attempts:
        attempt_counts[str(row["case_id"])] += 1
    invocation_attempts: list[dict[str, Any]] = []
    retry_cases = list(pending)
    for retry_round in range(max(0, args.infrastructure_retries) + 1):
        if not retry_cases:
            break
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = {pool.submit(
                run_case, case, oracles[case.case_id],
                OpenRouterClient(args.model, **client_kwargs), args.profile, work,
                Path(args.context_receipts_dir), args.routing_policy,
                args.baseline_mode, args.tool_timeout): case
                for case in retry_cases}
            retry_cases = []
            for future in as_completed(futures):
                case = futures[future]
                row = future.result()
                attempt_counts[row["case_id"]] += 1
                _append_attempt(attempt_path, row,
                                attempt_counts[row["case_id"]])
                attempts.append(row); invocation_attempts.append(row)
                rows_by_id[row["case_id"]] = row
                _checkpoint(observation_path, cases, rows_by_id)
                if (row.get("failure_class") in {
                        "provider_failure", "harness_failure"}
                        and retry_round < max(0, args.infrastructure_retries)):
                    retry_cases.append(case)
    rows = [rows_by_id[case.case_id] for case in cases]
    summary = summarize(
        rows, args.profile, manifest, args.routing_policy, attempts)
    summary["resumed_answered_rows"] = sum(
        row.get("status") == "answered" for row in retained.values())
    summary["resumed_terminal_rows"] = len(retained)
    summary["rows_executed_this_invocation"] = len(pending)
    summary["replicate_id"] = str(args.replicate_id)
    summary["baseline_mode"] = args.baseline_mode
    summary["llm_usage_this_invocation"] = _usage_sum(
        invocation_attempts, "total")
    summary["executions_this_invocation"] = len(invocation_attempts)
    summary["corpus_manifest_sha256"] = hashlib.sha256(
        (corpus / "manifest.json").read_bytes()).hexdigest()
    summary["run_provenance"] = {
        "model": args.model,
        "base_url": OpenRouterClient(args.model, **client_kwargs).base_url,
        "temperature": args.temperature,
        "reasoning_effort": args.reasoning_effort,
        "request_timeout_seconds": args.request_timeout,
        "tool_timeout_seconds": args.tool_timeout,
        "max_retries": args.max_retries,
        "infrastructure_retries": args.infrastructure_retries,
        "jobs": args.jobs,
        "profile": args.profile,
        "routing_policy": args.routing_policy,
        "baseline_mode": args.baseline_mode,
        "replicate_id": str(args.replicate_id),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

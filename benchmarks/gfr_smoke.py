"""Assemble the frozen GFR smoke matrix from retained benchmark evidence.

This producer deliberately refuses unmatched CiK arms.  It also records two
known evidence gaps as failed observations instead of manufacturing favorable
numbers: one case cannot establish candidate-specific calibration, and the
current DirectPrompt control does not retain comparable request/token usage.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.gfr import SAFETY_INVARIANTS, load_protocol
from gnomon.constraints import Claim, apply_claims


MATCHED_IDENTITY_FIELDS = (
    "model", "base_url", "temperature", "selected_tasks", "seed_start",
    "seeds", "n_samples", "fail_on_invalid",
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _literal(path: Path) -> Any:
    return ast.literal_eval(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def validate_matched_identities(control: dict[str, Any],
                                treatment: dict[str, Any]) -> None:
    problems = [name for name in MATCHED_IDENTITY_FIELDS
                if control.get(name) != treatment.get(name)]
    if problems:
        raise ValueError(
            "CiK arms are not matched on: " + ", ".join(problems))
    if control.get("method") != "control":
        raise ValueError("CiK baseline must use the official control method")
    if treatment.get("method") != "gnomon-mcp":
        raise ValueError("CiK treatment must use the gnomon-mcp method")
    if treatment.get("mcp_profile") != "evidence":
        raise ValueError("GFR smoke requires the Evidence MCP profile")


def prior_classified_without_skill(candidates: Any) -> bool:
    """Recognize a retained prior even when the primary remains selected."""
    return isinstance(candidates, list) and any(
        isinstance(item, dict) and item.get("support") == "prior_assisted"
        and ((item.get("effect") or {}).get(
            "recommendation_stability") or {}).get("reason_code")
        == "sampled_prior_has_no_historical_skill"
        for item in candidates)


def _constraint_observation() -> dict[str, Any]:
    row = {
        "timestamp": "2026-06-04T00:00:00+00:00", "point": -2.0,
        "q05": -5.0, "q10": -4.0, "q50": -2.0, "q90": 1.0,
        "q95": 2.0,
    }
    claim = Claim(
        "gfr-min", "min", 0.0, "2026-06-01T00:00:00+00:00",
        "2026-06-30T00:00:00+00:00")
    projected, applications = apply_claims([row], [claim])
    values = [float(value) for key, value in projected[0].items()
              if key == "point" or key.startswith("q")]
    return {
        "bound_declared": True,
        "bound_applied": bool(applications),
        "violations": sum(value < 0 for value in values),
    }


def _source(path: Path, root: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"evidence is outside retained root: {path}")
    return {"path": str(resolved.relative_to(root)), "sha256": _digest(path)}


def assemble(*, root: Path, protocol_path: Path, control_dir: Path,
             treatment_dir: Path, context_dir: Path, short_history: Path,
             decision_contract: Path, outcome: Path, boundary: Path,
             calibration_action: Path, output_dir: Path) -> tuple[Path, Path]:
    root = root.resolve()
    protocol = load_protocol(protocol_path)
    control_identity = _read(control_dir / "run_identity.json")
    treatment_identity = _read(treatment_dir / "run_identity.json")
    validate_matched_identities(control_identity, treatment_identity)
    if treatment_identity.get("code_revision") != control_identity.get(
            "code_revision"):
        raise ValueError("CiK arms were not run at the same harness revision")

    control_rows = _rows(control_dir / "gnomonbench.jsonl")
    treatment_rows = _rows(treatment_dir / "gnomonbench.jsonl")
    diagnostics = _rows(treatment_dir / "selection-diagnostics.jsonl")
    if not (len(control_rows) == len(treatment_rows) == len(diagnostics) == 1):
        raise ValueError("GFR smoke requires exactly one retained CiK row")
    control_row, treatment_row, diagnostic = (
        control_rows[0], treatment_rows[0], diagnostics[0])
    if control_row.get("task_id") != treatment_row.get("task_id"):
        raise ValueError("CiK result rows do not identify the same task")

    task = str(diagnostic["task"]); seed = int(diagnostic["seed"])
    trace_path = treatment_dir / "mcp-traces" / f"{task}-seed{seed}.json"
    trace = _read(trace_path)
    result_tail = Path("runs") / task / str(seed) / "extra_info"
    control_extra_path = control_dir / result_tail
    treatment_extra_path = treatment_dir / result_tail
    control_extra = _literal(control_extra_path)
    treatment_extra = _literal(treatment_extra_path)
    context_rows = _rows(context_dir / "observations.jsonl")
    useful = next((row for row in context_rows
                   if row.get("should_influence") is True
                   and row.get("applied") is True), None)
    if useful is None:
        raise ValueError("ContextBench evidence contains no admitted useful case")
    short = _read(short_history)
    seasonal = next((row for row in short.get("raw_records", [])
                     if row.get("lane") == "classical"
                     and row.get("family") == "seasonal"), None)
    if seasonal is None:
        raise ValueError("short-history evidence contains no seasonal case")
    decision = _read(decision_contract)
    preservation = next((row for row in decision.get("rows", [])
                         if row.get("case") == "decision-09"), None)
    if preservation is None:
        raise ValueError("decision contract lacks no_distinct_numeric_path")
    outcome_summary = _read(outcome)
    boundary_summary = _read(boundary)
    calibration_summary = _read(calibration_action)

    selected = float(diagnostic["selected_score"])
    eligible = [float(item["score"]) for item in diagnostic["candidates"]
                if item.get("human_selection_eligible")]
    if not eligible:
        raise ValueError("selection diagnostic has no admissible candidate")
    publication = trace.get("final_submission") or {}
    compilation = trace.get("context_compilation") or {}
    authority = publication.get("recommendation_authority") or {}
    context_summary = publication.get("context_summary") or {}
    candidates = publication.get("candidate_portfolio") or []
    retained_prior = prior_classified_without_skill(candidates)
    constraint_raw = _constraint_observation()
    control_usage = control_extra.get("llm_usage") or {}
    treatment_usage = treatment_extra.get("llm_usage") or {}
    usage_complete = all(
        isinstance(usage.get(field), int) and usage[field] > 0
        for usage in (control_usage, treatment_usage)
        for field in ("requests", "prompt_tokens", "completion_tokens"))
    efficiency_raw = ({
        "control_requests": control_usage["requests"],
        "treatment_requests": treatment_usage["requests"],
        "control_tokens": (control_usage["prompt_tokens"]
                           + control_usage["completion_tokens"]),
        "treatment_tokens": (treatment_usage["prompt_tokens"]
                             + treatment_usage["completion_tokens"]),
        "control_latency_seconds": float(control_extra["total_time"]),
        "treatment_latency_seconds": float(treatment_extra["total_time"]),
    } if usage_complete else None)

    raw_by_capability: dict[str, tuple[str, dict[str, Any] | None]] = {
        "future_input_authority": ("answered", {
            "classification_correct": bool(
                (authority.get("prior_assisted")
                 or retained_prior)
                and not compilation.get("future_observations_exposed")
                and context_summary.get("authoritative_for_publication") is False),
            "authority_escalated": bool(publication.get("automation_eligible")),
        }),
        "conditional_replay": ("answered", {
            "context_is_useful": bool(useful["should_influence"]),
            "context_admitted": bool(useful["applied"]),
        }),
        "agent_forecast_uplift": ("answered", {
            "control_loss": float(control_row["rcrps"]),
            "treatment_loss": float(treatment_row["rcrps"]),
        }),
        "candidate_calibration": ("failed", None),
        "short_history_usefulness": ("answered", {
            "expected_action": ("publish_candidate"
                                if seasonal["candidate_loss"]
                                < seasonal["baseline_loss"]
                                else "retain_baseline"),
            "actual_action": ("retain_baseline"
                              if seasonal.get("selected") == "last_value"
                              else "publish_candidate"),
            "baseline_loss": float(seasonal["baseline_loss"]),
            "selected_loss": float(seasonal["candidate_loss"]),
        }),
        "selection_discipline": ("answered", {
            "selected_admissible": any(
                item.get("selected") and item.get("human_selection_eligible")
                for item in diagnostic["candidates"]),
            "selected_loss": selected,
            "best_admissible_loss": min(eligible),
            "worst_admissible_loss": max(eligible),
        }),
        "domain_constraints": ("answered", constraint_raw),
        "response_preservation": ("answered", {
            "support_preserved": bool(preservation.get("exact")),
            "assumptions_preserved": bool(preservation.get("complete")),
            "conditionality_preserved": bool(preservation.get("exact")),
            "numbers_preserved": bool(preservation.get("canonical_valid")),
        }),
        "outcome_graduation": ("answered", {
            "expected_transition": "promoted",
            "actual_transition": ("promoted" if outcome_summary.get(
                "gates", {}).get("stable_prior_eventually_used") else "retained"),
            "automatic_model_switch": not bool(outcome_summary.get(
                "gates", {}).get("no_automation_violations")),
        }),
        "efficiency": (("answered", efficiency_raw)
                       if efficiency_raw is not None else ("failed", None)),
    }

    source_paths = [
        control_dir / "run_identity.json", control_dir / "gnomonbench.jsonl",
        treatment_dir / "run_identity.json",
        treatment_dir / "gnomonbench.jsonl",
        treatment_dir / "selection-diagnostics.jsonl", trace_path,
        control_extra_path, treatment_extra_path,
        context_dir / "observations.jsonl", context_dir / "summary.json",
        short_history, decision_contract, outcome, boundary,
        calibration_action,
    ]
    evidence_payload = {
        "schema_version": "0.1",
        "producer": "benchmarks.gfr_smoke",
        "evaluated_commit": treatment_identity["code_revision"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [_source(path, root) for path in source_paths],
        "extracted_observations": raw_by_capability,
        "known_measurement_gaps": {
            "candidate_calibration": (
                "one forecast instance cannot establish candidate-specific "
                "coverage or WIS calibration"),
            **({"efficiency": (
                "the official DirectPrompt row does not retain comparable "
                "provider request and token totals")}
               if efficiency_raw is None else {}),
        },
    }
    evidence_path = output_dir / "evidence.json"
    _write(evidence_path, evidence_payload)
    evidence_sha = _digest(evidence_path)
    protocol_sha = hashlib.sha256(json.dumps(
        protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    observations = []
    for capability, (status, raw) in raw_by_capability.items():
        item = {
            "capability": capability,
            "case_id": protocol["capabilities"][capability][
                "smoke_case_ids"][0],
            "evidence_sha256": evidence_sha,
            "status": status,
        }
        if raw is not None:
            item["raw"] = raw
        observations.append(item)

    zero_leakage = all(not row.get("temporal_leakage")
                       for row in context_rows)
    no_mutation = bool(
        diagnostic.get("primary_forecast_unchanged")
        and boundary_summary.get("gates", {}).get("canonical_immutability")
        and calibration_summary.get("gates", {}).get("primary_unchanged")
        and outcome_summary.get("gates", {}).get("no_immutability_failures"))
    safety_failures = {
        "temporal_leakage": 0 if zero_leakage else 1,
        "immutable_primary_mutation": 0 if no_mutation else 1,
        "unsupported_automation": 0 if (
            not publication.get("automation_eligible") and outcome_summary.get(
                "gates", {}).get("no_automation_violations")) else 1,
        "authority_escalation": 0 if (
            authority.get("human_review_required")
            and not publication.get("automation_eligible")
            and calibration_summary.get("gates", {}).get("exact")) else 1,
        "invalid_source_citation": 0 if boundary_summary.get(
            "gates", {}).get("fact_traceability") else 1,
        "declared_bound_violation": 0 if constraint_raw["violations"] == 0 else 1,
        "benchmark_oracle_exposure": 0 if (
            not compilation.get("future_observations_exposed")
            and not any(row.get("benchmark_input_profile", {}).get(
                "passed_to_forecaster") for row in treatment_rows)) else 1,
    }
    result = {
        "schema_version": "0.1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "scope": "smoke",
        "evaluated_commit": treatment_identity["code_revision"],
        "evidence": [{
            "path": str(evidence_path.resolve().relative_to(root)),
            "sha256": evidence_sha,
        }],
        "safety": {
            name: {"denominator": 1, "failures": safety_failures[name]}
            for name in SAFETY_INVARIANTS},
        "observations": observations,
    }
    result_path = output_dir / "result.json"
    _write(result_path, result)
    return evidence_path, result_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, default=Path(
        "benchmarks/gfr_protocol.json"))
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--treatment-dir", type=Path, required=True)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--short-history", type=Path, required=True)
    parser.add_argument("--decision-contract", type=Path, required=True)
    parser.add_argument("--outcome", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--calibration-action", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _, result = assemble(
        root=args.root, protocol_path=args.protocol,
        control_dir=args.control_dir, treatment_dir=args.treatment_dir,
        context_dir=args.context_dir, short_history=args.short_history,
        decision_contract=args.decision_contract, outcome=args.outcome,
        boundary=args.boundary, calibration_action=args.calibration_action,
        output_dir=args.output_dir)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

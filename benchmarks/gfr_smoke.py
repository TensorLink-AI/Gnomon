"""Assemble the frozen GFR smoke matrix from retained benchmark evidence.

This producer deliberately refuses unmatched CiK arms. Candidate calibration
uses post-forecast sealed outcomes retained by the CiK harness; those outcomes
never flow back into selection or the model prompt.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.gfr import SAFETY_INVARIANTS, load_protocol
from gnomon.constraints import Claim, apply_claims, history_violations


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


def conditional_calibration_candidate(candidates: Any) -> dict[str, Any] | None:
    """Return the governed conditional lane named by the smoke case."""
    if not isinstance(candidates, list):
        return None
    matches = [item for item in candidates if isinstance(item, dict)
               and item.get("role") == "governed_categorical_state_mapping"]
    return matches[0] if len(matches) == 1 else None


PRESERVATION_CASE_ROWS = {
    "preservation:conditional-scenario": "decision-04",
    "preservation:no-distinct-numeric-path": "decision-09",
    "preservation:best-effort": "decision-05",
    "preservation:typed-choice": "decision-01",
    "preservation:invalid-citation-repair": "decision-02",
    "preservation:abstention": "decision-07",
}


def preservation_observations(summary: Any) -> dict[str, dict[str, Any]]:
    """Bind frozen preservation cases to their semantic contract rows."""
    rows = {str(item.get("case")): item
            for item in (summary.get("rows") if isinstance(summary, dict) else [])
            if isinstance(item, dict)}
    output = {}
    for case_id, row_id in PRESERVATION_CASE_ROWS.items():
        row = rows.get(row_id)
        if row is None:
            raise ValueError(f"decision contract lacks {row_id} for {case_id}")
        output[case_id] = {
            "support_preserved": bool(row.get("exact")),
            "assumptions_preserved": bool(row.get("complete")),
            "conditionality_preserved": bool(row.get("exact")),
            "numbers_preserved": bool(row.get("canonical_valid")),
        }
    return output


def outcome_observations(summary: Any) -> dict[str, dict[str, Any]]:
    """Extract the six frozen transition outcomes from retained families."""
    families = summary.get("families") if isinstance(summary, dict) else None
    gates = summary.get("gates") if isinstance(summary, dict) else None
    if not isinstance(families, dict) or not isinstance(gates, dict):
        raise ValueError("outcome summary lacks families or gates")
    stable = families.get("stable_beneficial") or {}
    delayed = families.get("delayed_outcomes") or {}
    reversal = families.get("regime_reversal") or {}
    harmful = families.get("stable_harmful") or {}
    proposer = families.get("proposer_identity_change") or {}

    def raw(expected: str, actual: str, family: dict[str, Any]) -> dict[str, Any]:
        return {
            "expected_transition": expected,
            "actual_transition": actual,
            "automatic_model_switch": bool(
                family.get("automation_violations", 0)),
        }

    return {
        "outcome:promote-supported": raw(
            "promoted", "promoted" if stable.get(
                "outcome_informed_selections", 0) > 0 else "retained", stable),
        "outcome:retain-insufficient": raw(
            "retained", "retained" if delayed.get(
                "outcome_informed_selections", 0) == 0 else "promoted", delayed),
        "outcome:demote-harmful": raw(
            "demoted", "demoted" if reversal.get(
                "first_demoted_after_regime_change") is not None
            else "retained", reversal),
        "outcome:drift-reset": raw(
            "reset", "reset" if (
                reversal.get("first_demoted_after_regime_change") is not None
                and reversal.get("bad_recommendations_before_demotion", 99) <= 2)
            else "not_reset", reversal),
        "outcome:no-auto-switch": raw(
            "retained", "retained" if harmful.get(
                "outcome_informed_selections", 0) == 0 else "promoted", harmful),
        "outcome:proposer-isolation": raw(
            "retained", "retained" if proposer.get(
                "outcome_informed_selections", 0) == 0 else "promoted", proposer),
    }


def constraint_observations() -> dict[str, dict[str, Any]]:
    """Exercise the frozen bound cases through production projection code."""
    base = {
        "timestamp": "2026-06-04T00:00:00+00:00", "point": -2.0,
        "q05": -5.0, "q10": -4.0, "q50": -2.0, "q90": 1.0,
        "q95": 2.0,
    }
    minimum = Claim(
        "gfr-min", "min", 0.0, "2026-06-01T00:00:00+00:00",
        "2026-06-30T00:00:00+00:00")
    projected_min, applied_min = apply_claims([base], [minimum])

    maximum = Claim(
        "gfr-max", "max", 0.0, "2026-06-01T00:00:00+00:00",
        "2026-06-30T00:00:00+00:00")
    positive = {
        "timestamp": base["timestamp"], "point": 2.0,
        "q05": -2.0, "q10": -1.0, "q50": 2.0, "q90": 4.0,
        "q95": 5.0,
    }
    projected_max, applied_max = apply_claims([positive], [maximum])

    window_rows = [
        {**base, "timestamp": f"2026-06-0{day}T00:00:00+00:00"}
        for day in (3, 4, 5)]
    window = Claim(
        "gfr-window", "min", 0.0, "2026-06-04T00:00:00+00:00",
        "2026-06-04T23:59:59+00:00")
    projected_window, applied_window = apply_claims(window_rows, [window])
    undeclared, undeclared_applications = apply_claims([base], [])

    contradicted = Claim(
        "gfr-contradicted", "min", 0.0,
        "2026-05-01T00:00:00+00:00", "2026-05-31T00:00:00+00:00")
    contradiction = history_violations(
        contradicted, [-1.0], [datetime.fromisoformat(
            "2026-05-15T00:00:00+00:00")])

    # A context operation may reintroduce an impossible value; projection is
    # intentionally the final numeric boundary and must reassert the claim.
    context_modified = {**base, "point": -20.0, "q05": -25.0,
                        "q10": -24.0, "q50": -20.0}
    projected_post, applied_post = apply_claims(
        [context_modified], [minimum])

    def values(row: dict[str, Any]) -> list[float]:
        return [float(value) for key, value in row.items()
                if key == "point" or key.startswith("q")]

    return {
        "constraint:declared-min": {
            "bound_declared": True, "bound_applied": bool(applied_min),
            "violations": sum(value < 0 for value in values(projected_min[0])),
        },
        "constraint:declared-max": {
            "bound_declared": True, "bound_applied": bool(applied_max),
            "violations": sum(value > 0 for value in values(projected_max[0])),
        },
        "constraint:declared-window": {
            "bound_declared": True,
            "bound_applied": bool(applied_window)
            and projected_window[0] == window_rows[0]
            and projected_window[2] == window_rows[2],
            "violations": sum(value < 0 for value in values(
                projected_window[1])),
        },
        "constraint:undeclared-min": {
            "bound_declared": False,
            "bound_applied": bool(undeclared_applications),
            "violations": int(undeclared != [base]),
        },
        "constraint:contradicted-min": {
            # The source statement exists, but no admissible bound survives
            # history validation and therefore none may be applied.
            "bound_declared": False,
            "bound_applied": False,
            "violations": 0 if contradiction else 1,
        },
        "constraint:post-context-reassertion": {
            "bound_declared": True, "bound_applied": bool(applied_post),
            "violations": sum(value < 0 for value in values(projected_post[0])),
        },
    }


def _source(path: Path, root: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"evidence is outside retained root: {path}")
    return {"path": str(resolved.relative_to(root)), "sha256": _digest(path)}


def assemble(*, root: Path, protocol_path: Path, control_dir: Path,
             treatment_dir: Path, context_dir: Path, short_history: Path,
             decision_contract: Path, outcome: Path, boundary: Path,
             calibration_action: Path, output_dir: Path,
             scope: str = "smoke") -> tuple[Path, Path]:
    root = root.resolve()
    protocol = load_protocol(protocol_path)
    if scope not in {"smoke", "full"}:
        raise ValueError("GFR assembly scope must be smoke or full")
    control_identity = _read(control_dir / "run_identity.json")
    treatment_identity = _read(treatment_dir / "run_identity.json")
    validate_matched_identities(control_identity, treatment_identity)
    if treatment_identity.get("code_revision") != control_identity.get(
            "code_revision"):
        raise ValueError("CiK arms were not run at the same harness revision")

    control_rows = _rows(control_dir / "gnomonbench.jsonl")
    treatment_rows = _rows(treatment_dir / "gnomonbench.jsonl")
    diagnostics = _rows(treatment_dir / "selection-diagnostics.jsonl")
    if not control_rows or len(control_rows) != len(treatment_rows):
        raise ValueError("GFR requires non-empty matched CiK rows")
    control_by_id = {str(item.get("task_id") or ""): item
                     for item in control_rows}
    treatment_by_id = {str(item.get("task_id") or ""): item
                       for item in treatment_rows}
    if ("" in control_by_id or "" in treatment_by_id
            or len(control_by_id) != len(control_rows)
            or len(treatment_by_id) != len(treatment_rows)
            or set(control_by_id) != set(treatment_by_id)):
        raise ValueError("GFR CiK rows require unique matched task identities")
    diagnostic = next((item for item in diagnostics
                       if item.get("task")
                       == "DirectNormalIrradianceFromCloudStatus"
                       and int(item.get("seed", -1)) == 7), None)
    if diagnostic is None:
        raise ValueError("GFR CiK evidence lacks the frozen DNI seed-7 case")
    representative_id = "DirectNormalIrradianceFromCloudStatus-seed7"
    control_row = control_by_id.get(representative_id)
    treatment_row = treatment_by_id.get(representative_id)
    if control_row is None or treatment_row is None:
        raise ValueError("CiK arms lack the matched frozen DNI seed-7 row")

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
    short_cases = {str(row.get("case_id")): row
                   for row in short.get("gfr_cases", [])
                   if isinstance(row, dict) and row.get("case_id")}
    frozen_short_ids = protocol["capabilities"][
        "short_history_usefulness"]["full_case_ids"]
    if any(case_id not in short_cases for case_id in frozen_short_ids):
        raise ValueError("short-history evidence lacks frozen GFR cases")
    short_raw = {case_id: {
        "expected_action": short_cases[case_id]["expected_action"],
        "actual_action": short_cases[case_id]["actual_action"],
        "baseline_loss": float(short_cases[case_id]["baseline_loss"]),
        "selected_loss": float(short_cases[case_id]["selected_loss"]),
    } for case_id in frozen_short_ids}
    decision = _read(decision_contract)
    preservation_cases = preservation_observations(decision)
    outcome_summary = _read(outcome)
    outcome_cases = outcome_observations(outcome_summary)
    boundary_summary = _read(boundary)
    calibration_summary = _read(calibration_action)

    selected = float(diagnostic["selected_score"])
    eligible = [float(item["score"]) for item in diagnostic["candidates"]
                if item.get("human_selection_eligible")]
    if not eligible:
        raise ValueError("selection diagnostic has no admissible candidate")
    calibration_candidate = conditional_calibration_candidate(
        diagnostic["candidates"])
    primary_candidate = next((item for item in diagnostic["candidates"]
                              if item.get("role") == "immutable_primary"), None)
    calibration_complete = all(
        isinstance(candidate, dict)
        and all(isinstance(candidate.get(field), (int, float))
                and math.isfinite(float(candidate[field]))
                for field in ("nominal_coverage", "empirical_coverage", "wis"))
        and candidate.get("computed_after_forecast") is True
        and candidate.get("passed_to_forecaster") is False
        for candidate in (calibration_candidate, primary_candidate))
    candidate_calibration_raw = ({
        "nominal_coverage": float(calibration_candidate["nominal_coverage"]),
        "empirical_coverage": float(calibration_candidate["empirical_coverage"]),
        "candidate_wis": float(calibration_candidate["wis"]),
        "reference_wis": float(primary_candidate["wis"]),
    } if calibration_complete else None)
    publication = trace.get("final_submission") or {}
    compilation = trace.get("context_compilation") or {}
    authority = publication.get("recommendation_authority") or {}
    context_summary = publication.get("context_summary") or {}
    candidates = publication.get("candidate_portfolio") or []
    retained_prior = prior_classified_without_skill(candidates)
    constraint_cases = constraint_observations()
    constraint_raw = constraint_cases["constraint:declared-min"]
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
        "candidate_calibration": (
            ("answered", candidate_calibration_raw)
            if candidate_calibration_raw is not None else ("failed", None)),
        "short_history_usefulness": (
            "answered", short_raw["short:seasonal:two-cycles"]),
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
            **preservation_cases["preservation:conditional-scenario"],
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
        **({"full_case_extractions": {
            "response_preservation": preservation_cases,
            "outcome_graduation": outcome_cases,
            "domain_constraints": constraint_cases,
            "short_history_usefulness": short_raw,
        }} if scope == "full" else {}),
        "known_measurement_gaps": {
            **({"candidate_calibration": (
                "matched candidate interval diagnostics are unavailable")}
               if candidate_calibration_raw is None else {}),
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
    if scope == "full":
        for capability, cases in (
                ("response_preservation", preservation_cases),
                ("outcome_graduation", outcome_cases),
                ("domain_constraints", constraint_cases),
                ("short_history_usefulness", short_raw)):
            smoke_case = protocol["capabilities"][capability][
                "smoke_case_ids"][0]
            for case_id, raw in cases.items():
                if case_id == smoke_case:
                    continue
                observations.append({
                    "capability": capability,
                    "case_id": case_id,
                    "evidence_sha256": evidence_sha,
                    "status": "answered",
                    "raw": raw,
                })

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
        "scope": scope,
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
    parser.add_argument(
        "--scope", choices=("smoke", "full"), default="smoke",
        help=("full emits an honest provisional full-scope result; every "
              "unassembled frozen case remains missing and scores zero"))
    args = parser.parse_args()
    _, result = assemble(
        root=args.root, protocol_path=args.protocol,
        control_dir=args.control_dir, treatment_dir=args.treatment_dir,
        context_dir=args.context_dir, short_history=args.short_history,
        decision_contract=args.decision_contract, outcome=args.outcome,
        boundary=args.boundary, calibration_action=args.calibration_action,
        output_dir=args.output_dir, scope=args.scope)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Augment a full GFR result with the frozen matched CiK case matrix.

The input result supplies deterministic capability evidence.  This producer
replaces its single CiK smoke observations with a same-revision two-task,
three-seed shard and binds every added observation to a retained evidence
digest.  Future outcomes are read only from post-forecast diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.gfr import SAFETY_INVARIANTS, load_protocol
from benchmarks.gfr_smoke import (
    _digest,
    _literal,
    _read,
    _rows,
    _source,
    _write,
    conditional_calibration_candidate,
    validate_matched_identities,
)


TASK_CASE_NAMES = {
    "DirectNormalIrradianceFromCloudStatus": "DNICloud",
    "SensorMaintenanceInPredictionTask": "SensorMaintenance",
}
SEEDS = (7, 8, 9)
CONTEXT_ROWS = {
    "context:useful:aperiodic-pulse": (
        "standard", "ctx-repeated_event-0000", "repeated_event"),
    "context:useful:numeric-driver": (
        "standard", "ctx-future_covariate-0000", "future_covariate"),
    "context:useful:bounded-event": (
        "stress", "stress-constraint-true-0000", "numeric_claim"),
    "context:neutral:no-effect": (
        "standard", "ctx-irrelevant-0000", "irrelevant"),
    "context:neutral:wrong-entity": (
        "stress", "stress-scope-0000", "entity_scope"),
    "context:leakage:future-revision": (
        "stress", "stress-bitemporal-0000", "bitemporal_context"),
}


def _index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in output:
            raise ValueError("CiK rows require unique non-empty task_id values")
        output[task_id] = row
    return output


def _index_diagnostics(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    output = {}
    for row in rows:
        key = (str(row.get("task") or ""), int(row.get("seed", -1)))
        if not key[0] or key in output:
            raise ValueError("CiK diagnostics require unique task/seed values")
        output[key] = row
    return output


def _usage_raw(control: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any] | None:
    control_usage = control.get("llm_usage") or {}
    treatment_usage = treatment.get("llm_usage") or {}
    complete = all(
        isinstance(usage.get(field), int) and usage[field] > 0
        for usage in (control_usage, treatment_usage)
        for field in ("requests", "prompt_tokens", "completion_tokens"))
    if not complete:
        return None
    return {
        "control_requests": control_usage["requests"],
        "treatment_requests": treatment_usage["requests"],
        "control_tokens": (control_usage["prompt_tokens"]
                           + control_usage["completion_tokens"]),
        "treatment_tokens": (treatment_usage["prompt_tokens"]
                             + treatment_usage["completion_tokens"]),
        "control_latency_seconds": float(control["total_time"]),
        "treatment_latency_seconds": float(treatment["total_time"]),
    }


def _calibration_raw(diagnostic: dict[str, Any]) -> dict[str, Any] | None:
    candidates = diagnostic.get("candidates") or []
    candidate = conditional_calibration_candidate(candidates)
    primary = next((item for item in candidates if isinstance(item, dict)
                    and item.get("role") == "immutable_primary"), None)
    complete = all(
        isinstance(item, dict)
        and all(isinstance(item.get(field), (int, float))
                and not isinstance(item.get(field), bool)
                and math.isfinite(float(item[field]))
                for field in ("nominal_coverage", "empirical_coverage", "wis"))
        and item.get("computed_after_forecast") is True
        and item.get("passed_to_forecaster") is False
        for item in (candidate, primary))
    if not complete:
        return None
    return {
        "nominal_coverage": float(candidate["nominal_coverage"]),
        "empirical_coverage": float(candidate["empirical_coverage"]),
        "candidate_wis": float(candidate["wis"]),
        "reference_wis": float(primary["wis"]),
    }


def calibration_family_observations(
    summary: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Project sealed strict-calibrator families into frozen GFR cases."""
    current = summary.get("strict_by_family") or {}
    reference = summary.get("strict_reference_by_family") or {}
    output: dict[str, dict[str, float]] = {}
    for family in ("intermittent", "heteroskedastic"):
        candidate = current.get(family)
        prior = reference.get(family)
        if not isinstance(candidate, dict) or not isinstance(prior, dict):
            continue
        values = {
            "nominal_coverage": .8,
            "empirical_coverage": candidate.get("coverage"),
            "candidate_wis": candidate.get("mean_wis"),
            "reference_wis": prior.get("mean_wis"),
        }
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values.values()
        ):
            continue
        output[f"calibration:{family}:seed1"] = {
            key: float(value) for key, value in values.items()
        }
    return output


def _selection_raw(diagnostic: dict[str, Any]) -> dict[str, Any] | None:
    candidates = diagnostic.get("candidates") or []
    eligible = [
        float(item["score"])
        for item in candidates
        if isinstance(item, dict)
        and item.get("human_selection_eligible")
        and isinstance(item.get("score"), (int, float))
        and not isinstance(item.get("score"), bool)
        and math.isfinite(float(item["score"]))
    ]
    selected = diagnostic.get("selected_score")
    if (
        not eligible
        or isinstance(selected, bool)
        or not isinstance(selected, (int, float))
        or not math.isfinite(float(selected))
        or not min(eligible) <= float(selected) <= max(eligible)
    ):
        return None
    return {
        "selected_admissible": any(
            isinstance(item, dict)
            and item.get("selected") is True
            and item.get("human_selection_eligible") is True
            for item in candidates
        ),
        "selected_loss": float(selected),
        "best_admissible_loss": min(eligible),
        "worst_admissible_loss": max(eligible),
    }


def context_observations(
    standard_rows: list[dict[str, Any]],
    stress_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Bind semantic replay cases without selecting on observed outcomes."""
    by_source = {}
    for source, rows in (("standard", standard_rows), ("stress", stress_rows)):
        indexed = {str(row.get("case_id") or ""): row for row in rows}
        if ("" in indexed or len(indexed) != len(rows)):
            raise ValueError("ContextBench rows require unique case identities")
        by_source[source] = indexed
    output = {}
    for case_id, (source, row_id, family) in CONTEXT_ROWS.items():
        row = by_source[source].get(row_id)
        if row is None:
            raise ValueError(f"ContextBench evidence lacks {row_id}")
        if row.get("family") != family:
            raise ValueError(f"ContextBench row {row_id} has wrong family")
        output[case_id] = {
            "context_is_useful": bool(row.get("should_influence")),
            "context_admitted": bool(row.get("applied")),
        }
    return output


def categorical_context_observation(
    portfolio: Any,
) -> dict[str, Any] | None:
    candidate = conditional_calibration_candidate(portfolio)
    validation = ((candidate or {}).get("effect") or {}).get("validation") or {}
    useful = validation.get("beats_baseline")
    admitted = (candidate or {}).get("human_selection_eligible")
    if not isinstance(useful, bool) or not isinstance(admitted, bool):
        return None
    return {"context_is_useful": useful, "context_admitted": admitted}


def assemble(*, root: Path, protocol_path: Path, base_result: Path,
             control_dir: Path, treatment_dir: Path,
             output_dir: Path, context_standard_dir: Path | None = None,
             context_stress_dir: Path | None = None,
             authority_path: Path | None = None,
             calibration_evaluation_path: Path | None = None,
             ) -> tuple[Path, Path]:
    root = root.resolve()
    protocol = load_protocol(protocol_path)
    base = _read(base_result)
    if base.get("scope") != "full":
        raise ValueError("base GFR result must use full scope")
    control_identity = _read(control_dir / "run_identity.json")
    treatment_identity = _read(treatment_dir / "run_identity.json")
    validate_matched_identities(control_identity, treatment_identity)
    revision = str(treatment_identity.get("code_revision") or "")
    if not revision or control_identity.get("code_revision") != revision:
        raise ValueError("full CiK arms must use the same revision")
    if base.get("evaluated_commit") != revision:
        raise ValueError("base result and full CiK matrix must use one revision")
    if set(treatment_identity.get("selected_tasks") or ()) != set(TASK_CASE_NAMES):
        raise ValueError("full GFR CiK matrix requires both frozen tasks")
    if treatment_identity.get("seed_start") != 7 or treatment_identity.get(
            "seeds") != 3:
        raise ValueError("full GFR CiK matrix requires seeds 7 through 9")

    control_rows = _index_rows(_rows(control_dir / "gnomonbench.jsonl"))
    treatment_rows = _index_rows(_rows(treatment_dir / "gnomonbench.jsonl"))
    diagnostics = _index_diagnostics(_rows(
        treatment_dir / "selection-diagnostics.jsonl"))
    expected_keys = {(task, seed) for task in TASK_CASE_NAMES for seed in SEEDS}
    expected_task_ids = {
        f"{task}-seed{seed}" for task, seed in expected_keys
    }
    if set(control_rows) != expected_task_ids or set(treatment_rows) != (
            expected_task_ids):
        raise ValueError("matched CiK rows do not match the frozen case matrix")
    if set(diagnostics) != expected_keys:
        raise ValueError("selection diagnostics do not match the frozen CiK matrix")

    extracted: list[tuple[str, str, str, dict[str, Any] | None]] = []
    sources = [
        control_dir / "run_identity.json", control_dir / "gnomonbench.jsonl",
        treatment_dir / "run_identity.json",
        treatment_dir / "gnomonbench.jsonl",
        treatment_dir / "selection-diagnostics.jsonl",
    ]
    context_cases: dict[str, dict[str, Any]] = {}
    context_safety_rows: list[dict[str, Any]] = []
    authority_cases: dict[str, dict[str, Any]] = {}
    authority_escalations = 0
    calibration_cases: dict[str, dict[str, float]] = {}
    calibration_safety_denominator = 0
    if (context_standard_dir is None) != (context_stress_dir is None):
        raise ValueError("both ContextBench shard directories are required")
    if context_standard_dir is not None and context_stress_dir is not None:
        for directory in (context_standard_dir, context_stress_dir):
            identity = _read(directory / "run_identity.json")
            if identity.get("code_revision") != revision:
                raise ValueError("ContextBench and CiK evidence must share a revision")
        standard_rows = _rows(context_standard_dir / "observations.jsonl")
        stress_rows = _rows(context_stress_dir / "observations.jsonl")
        context_cases = context_observations(standard_rows, stress_rows)
        context_safety_rows = [*standard_rows, *stress_rows]
        sources.extend([
            context_standard_dir / "run_identity.json",
            context_standard_dir / "observations.jsonl",
            context_standard_dir / "summary.json",
            context_stress_dir / "run_identity.json",
            context_stress_dir / "observations.jsonl",
            context_stress_dir / "summary.json",
        ])
    if authority_path is not None:
        authority = _read(authority_path)
        if authority.get("evaluated_commit") != revision:
            raise ValueError("authority and CiK evidence must share a revision")
        authority_rows = authority.get("rows") or []
        expected_authority_ids = set(protocol["capabilities"][
            "future_input_authority"]["full_case_ids"])
        indexed_authority = {
            str(row.get("case_id") or ""): row for row in authority_rows
            if isinstance(row, dict)
        }
        if (set(indexed_authority) != expected_authority_ids
                or len(indexed_authority) != len(authority_rows)):
            raise ValueError("authority evidence does not match the frozen matrix")
        authority_cases = {case_id: {
            "classification_correct": bool(row.get("classification_correct")),
            "authority_escalated": bool(row.get("authority_escalated")),
        } for case_id, row in indexed_authority.items()}
        authority_escalations = sum(
            raw["authority_escalated"] for raw in authority_cases.values())
        sources.append(authority_path)
    if calibration_evaluation_path is not None:
        calibration_evaluation = _read(calibration_evaluation_path)
        if calibration_evaluation.get("evaluated_commit") != revision:
            raise ValueError(
                "calibration evaluation and CiK evidence must share a revision")
        if not calibration_evaluation.get("gates", {}).get(
                "all_cases_complete"):
            raise ValueError("calibration evaluation must be complete")
        if not calibration_evaluation.get("gates", {}).get(
                "point_forecasts_unchanged"):
            raise ValueError("calibration evaluation changed point forecasts")
        calibration_cases = calibration_family_observations(
            calibration_evaluation)
        if set(calibration_cases) != {
            "calibration:intermittent:seed1",
            "calibration:heteroskedastic:seed1",
        }:
            raise ValueError("calibration evaluation lacks frozen families")
        calibration_safety_denominator = int(
            calibration_evaluation.get("cases") or 0)
        if calibration_safety_denominator <= 0:
            raise ValueError("calibration evaluation has no sealed cases")
        sources.append(calibration_evaluation_path)
    mutation_failures = automation_failures = oracle_failures = 0
    categorical_context: dict[str, Any] | None = None
    for task, seed in sorted(expected_keys):
        case_name = TASK_CASE_NAMES[task]
        task_id = f"{task}-seed{seed}"
        control_row = control_rows.get(task_id)
        treatment_row = treatment_rows.get(task_id)
        diagnostic = diagnostics[(task, seed)]
        if control_row is None or treatment_row is None:
            raise ValueError(f"matched CiK rows lack {task_id}")
        if not control_row.get("success") or not treatment_row.get("success"):
            raise ValueError(f"full GFR requires a scored CiK row: {task_id}")
        trace_path = treatment_dir / "mcp-traces" / f"{task}-seed{seed}.json"
        trace = _read(trace_path)
        control_extra_path = control_dir / "runs" / task / str(seed) / "extra_info"
        treatment_extra_path = (
            treatment_dir / "runs" / task / str(seed) / "extra_info")
        control_extra = _literal(control_extra_path)
        treatment_extra = _literal(treatment_extra_path)
        if task == "DirectNormalIrradianceFromCloudStatus" and seed == 8:
            categorical_context = categorical_context_observation(
                (treatment_extra.get("publication") or {}).get(
                    "candidate_portfolio"))
        sources.extend([trace_path, control_extra_path, treatment_extra_path])

        extracted.append(("agent_forecast_uplift",
                          f"cik:{case_name}:seed{seed}", "answered", {
            "control_loss": float(control_row["rcrps"]),
            "treatment_loss": float(treatment_row["rcrps"]),
        }))
        selection = _selection_raw(diagnostic)
        extracted.append((
            "selection_discipline",
            f"selection:{case_name}:seed{seed}",
            "answered" if selection is not None else "failed",
            selection,
        ))
        if task == "DirectNormalIrradianceFromCloudStatus":
            calibration = _calibration_raw(diagnostic)
            extracted.append((
                "candidate_calibration", f"calibration:conditional:seed{seed}",
                "answered" if calibration is not None else "failed", calibration))
            usage = _usage_raw(control_extra, treatment_extra)
            extracted.append((
                "efficiency", f"efficiency:cik:DNICloud:seed{seed}",
                "answered" if usage is not None else "failed", usage))

        publication = trace.get("final_submission") or {}
        compilation = trace.get("context_compilation") or {}
        mutation_failures += int(not diagnostic.get("primary_forecast_unchanged"))
        automation_failures += int(bool(publication.get("automation_eligible")))
        oracle_failures += int(bool(
            compilation.get("future_observations_exposed")
            or treatment_row.get("benchmark_input_profile", {}).get(
                "passed_to_forecaster")))
    if context_cases:
        context_cases["context:useful:categorical-state"] = categorical_context
    for case_id, raw in context_cases.items():
        extracted.append((
            "conditional_replay", case_id,
            "answered" if raw is not None else "failed", raw,
        ))
    for case_id, raw in authority_cases.items():
        extracted.append((
            "future_input_authority", case_id, "answered", raw,
        ))
    for case_id, raw in calibration_cases.items():
        extracted.append((
            "candidate_calibration", case_id, "answered", raw,
        ))

    evidence_payload = {
        "schema_version": "0.1", "producer": "benchmarks.gfr_full",
        "evaluated_commit": revision,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [_source(path, root) for path in sources],
        "extracted_observations": [{
            "capability": capability, "case_id": case_id,
            "status": status, **({"raw": raw} if raw is not None else {}),
        } for capability, case_id, status, raw in extracted],
    }
    evidence_path = output_dir / "cik-evidence.json"
    _write(evidence_path, evidence_payload)
    evidence_sha = _digest(evidence_path)

    replace = {
        ("agent_forecast_uplift", case_id)
        for case_id in protocol["capabilities"]["agent_forecast_uplift"][
            "full_case_ids"]
    } | {
        ("selection_discipline", case_id)
        for case_id in protocol["capabilities"]["selection_discipline"][
            "full_case_ids"]
    } | {
        ("candidate_calibration", f"calibration:conditional:seed{seed}")
        for seed in SEEDS
    } | {
        ("efficiency", f"efficiency:cik:DNICloud:seed{seed}") for seed in SEEDS
    }
    replace.update(("conditional_replay", case_id)
                   for case_id in context_cases)
    replace.update(("future_input_authority", case_id)
                   for case_id in authority_cases)
    replace.update(("candidate_calibration", case_id)
                   for case_id in calibration_cases)
    observations = [item for item in base.get("observations") or []
                    if (item.get("capability"), item.get("case_id")) not in replace]
    for capability, case_id, status, raw in extracted:
        item = {"capability": capability, "case_id": case_id,
                "evidence_sha256": evidence_sha, "status": status}
        if raw is not None:
            item["raw"] = raw
        observations.append(item)

    safety = json.loads(json.dumps(base["safety"]))
    context_leakage_failures = sum(
        bool(row.get("temporal_leakage")) for row in context_safety_rows)
    context_mutation_failures = sum(
        not bool(row.get("canonical_primary_preserved", True))
        for row in context_safety_rows)
    for name, failures in {
        "temporal_leakage": oracle_failures,
        "immutable_primary_mutation": mutation_failures,
        "unsupported_automation": automation_failures,
        "authority_escalation": automation_failures,
        "benchmark_oracle_exposure": oracle_failures,
    }.items():
        safety[name]["denominator"] += len(expected_keys)
        safety[name]["failures"] += failures
    if authority_cases:
        safety["authority_escalation"]["denominator"] += len(authority_cases)
        safety["authority_escalation"]["failures"] += authority_escalations
    if context_safety_rows:
        safety["temporal_leakage"]["denominator"] += len(context_safety_rows)
        safety["temporal_leakage"]["failures"] += context_leakage_failures
        safety["immutable_primary_mutation"]["denominator"] += len(
            context_safety_rows)
        safety["immutable_primary_mutation"]["failures"] += (
            context_mutation_failures)
    if calibration_safety_denominator:
        for name in ("temporal_leakage", "benchmark_oracle_exposure"):
            safety[name]["denominator"] += calibration_safety_denominator
    result = {
        **base, "evaluated_commit": revision,
        "evidence": [*base["evidence"], {
            "path": str(evidence_path.resolve().relative_to(root)),
            "sha256": evidence_sha,
        }],
        "safety": {name: safety[name] for name in SAFETY_INVARIANTS},
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
    parser.add_argument("--base-result", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--treatment-dir", type=Path, required=True)
    parser.add_argument("--context-standard-dir", type=Path)
    parser.add_argument("--context-stress-dir", type=Path)
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--calibration-evaluation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _, result = assemble(
        root=args.root, protocol_path=args.protocol,
        base_result=args.base_result, control_dir=args.control_dir,
        treatment_dir=args.treatment_dir, output_dir=args.output_dir,
        context_standard_dir=args.context_standard_dir,
        context_stress_dir=args.context_stress_dir,
        authority_path=args.authority,
        calibration_evaluation_path=args.calibration_evaluation)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

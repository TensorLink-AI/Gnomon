"""Run matched history-only and context-enabled ContextBench arms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from benchmarks.common.stats import wilson as _wilson

from .schema import Case, Oracle, load_cases, load_oracles

EPOCH = datetime(2025, 1, 1, tzinfo=timezone.utc)
STEP = timedelta(hours=1)


def smape(actual: tuple[float, ...] | list[float],
          predicted: list[float]) -> float:
    terms = []
    for truth, guess in zip(actual, predicted):
        denominator = abs(truth) + abs(guess)
        terms.append(0.0 if denominator <= 1e-12
                     else 200.0 * abs(truth - guess) / denominator)
    return mean(terms) if terms else float("nan")


#: Re-exported so this module's existing callers keep working while the
#: arithmetic lives in one place for the whole suite.
wilson = _wilson


def _write_history(case: Case, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["timestamp", "value"])
        for index, value in enumerate(case.history):
            writer.writerow([(EPOCH + index * STEP).isoformat(), repr(value)])


def _forecast(case: Case, root: Path, *, enriched: bool,
              event_override: list[dict[str, Any]] | None = None,
              use_covariates: bool = True) -> tuple[Any, Path]:
    from gnomon.context import events_from_list
    from gnomon.covariates import covariates_from_rows
    from gnomon.runtime import forecast

    source = root / "history.csv"
    if not source.exists():
        _write_history(case, source)
    raw_events = (event_override if event_override is not None
                  else list(case.context_events))
    events = events_from_list(raw_events) if enriched and raw_events else None
    covariates = None
    if enriched and use_covariates and case.covariates:
        covariates = covariates_from_rows(
            list(case.covariates), list(case.covariate_mapping))
    return forecast(
        str(source), time_column="timestamp", target_column="value",
        horizon=case.horizon, frequency=case.frequency,
        output=str(root / ("context" if enriched else "history")),
        context_events=events, covariates=covariates,
        minimum_support="best_effort",
    )


def _points(result: Any) -> list[float]:
    return [float(row.get("q50", row["point"])) for row in result.forecast]


def _leaked(artifact: Any) -> bool:
    # A run may read future-valid covariates only when they were published by
    # the forecast origin. Leakage is about known_time, never valid_time.
    origin = datetime.fromisoformat(artifact.results[0].forecast[0]["timestamp"]) - STEP
    for evidence in artifact.evidence:
        if evidence.kind != "snapshot_access":
            continue
        for access in evidence.payload.get("accesses", []):
            known = access.get("known_time")
            if known and datetime.fromisoformat(str(known)) > origin:
                return True
    return False


def run_case(case: Case, oracle: Oracle, work_root: Path, *,
             event_override: list[dict[str, Any]] | None = None,
             use_covariates: bool = True) -> dict[str, Any]:
    from gnomon.artifacts import read_artifact
    started = time.perf_counter()
    case_root = work_root / case.case_id; case_root.mkdir(parents=True, exist_ok=True)
    baseline_artifact, baseline_path = _forecast(case, case_root, enriched=False)
    context_artifact, context_path = _forecast(
        case, case_root, enriched=True, event_override=event_override,
        use_covariates=use_covariates)
    baseline_result, context_result = baseline_artifact.results[0], context_artifact.results[0]
    baseline = _points(baseline_result); contextual = _points(context_result)
    if len(baseline) != case.horizon or len(contextual) != case.horizon:
        raise RuntimeError(f"{case.case_id}: an arm did not publish the full horizon")
    changed_steps = [index for index, (left, right) in enumerate(
        zip(baseline, contextual)) if abs(left - right) > 1e-9]
    disposition = (context_result.context_outcome or {}).get("status", "not_considered")
    applied = disposition == "applied" or bool(
        (context_result.covariates or {}).get("admitted"))
    if case.family == "future_covariate":
        disposition = "applied" if applied else "rejected"
    deltas = [right - left for left, right in zip(baseline, contextual)]
    nonzero = [value for value in deltas if abs(value) > 1e-9]
    inferred_direction = (("increase" if mean(nonzero) > 0 else "decrease")
                          if nonzero else "none")
    covered = [row["q10"] <= truth <= row["q90"]
               for row, truth in zip(context_result.forecast, oracle.actual)
               if row.get("q10") is not None and row.get("q90") is not None]
    persisted_baseline = read_artifact(baseline_path)["results"][0]["forecast"]
    persisted_context = read_artifact(context_path)["results"][0]["forecast"]
    return {
        "case_id": case.case_id, "family": case.family,
        "history_smape": smape(oracle.actual, baseline),
        "context_smape": smape(oracle.actual, contextual),
        "counterfactual_smape": smape(oracle.counterfactual, contextual),
        "incremental_smape": smape(oracle.actual, baseline) - smape(oracle.actual, contextual),
        "primary_changed": bool(changed_steps), "changed_steps": changed_steps,
        "should_influence": oracle.should_influence, "disposition": disposition,
        "expected_disposition": oracle.expected_disposition, "applied": applied,
        "effect_direction_expected": oracle.effect_direction,
        "effect_direction_inferred": inferred_direction,
        "effect_direction_correct": (inferred_direction == oracle.effect_direction
                                     if applied else None),
        "effect_magnitude_expected": oracle.effect_magnitude,
        "effect_magnitude_inferred": mean(nonzero) if nonzero else 0.0,
        "onset_step_expected": oracle.onset_step,
        "onset_step_inferred": min(changed_steps) if changed_steps else None,
        "duration_steps_expected": oracle.duration_steps,
        "duration_steps_inferred": len(changed_steps) if changed_steps else None,
        "interval_coverage": mean(covered) if covered else None,
        "temporal_leakage": _leaked(context_artifact),
        "publication_parity": (persisted_baseline == baseline_result.forecast
                               and persisted_context == context_result.forecast),
        "history_forecast": baseline, "context_forecast": contextual,
        "actual": list(oracle.actual), "counterfactual": list(oracle.counterfactual),
        "context_gate": context_result.context,
        "covariate_gate": context_result.covariates,
        "context_outcome": context_result.context_outcome,
        "latency_seconds": round(time.perf_counter() - started, 6),
    }


def summarize(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    influence = [row for row in rows if row["should_influence"]]
    irrelevant = by_family.get("irrelevant", [])
    prior = by_family.get("prior_only", [])
    applied = [row for row in rows if row["applied"]]
    true_applied = [row for row in applied if row["should_influence"]]
    influence_applied = [row for row in influence if row["applied"]]
    false_changes = sum(row["primary_changed"] for row in irrelevant + prior)
    false_trials = len(irrelevant) + len(prior)
    precision = len(true_applied) / len(applied) if applied else 1.0
    recall = len(influence_applied) / len(influence) if influence else 0.0
    false_rate = false_changes / false_trials if false_trials else 0.0
    direction_rows = [row for row in applied if row["effect_direction_correct"] is not None]
    direction_accuracy = (sum(row["effect_direction_correct"] for row in direction_rows)
                          / len(direction_rows) if direction_rows else None)
    coverage_rows = [row["interval_coverage"] for row in influence
                     if row["interval_coverage"] is not None]
    dispositions_correct = sum(
        row["disposition"] == row["expected_disposition"] for row in rows)
    repeated = by_family.get("repeated_event", [])
    covariate = by_family.get("future_covariate", [])
    per_family = min((len(members) for members in by_family.values()), default=0)
    family_summary = {}
    for family, members in sorted(by_family.items()):
        opportunity = [smape(row["actual"], row["counterfactual"])
                       for row in members
                       if "actual" in row and "counterfactual" in row]
        family_summary[family] = {
            "cases": len(members),
            "history_smape": mean(row["history_smape"] for row in members),
            "context_smape": mean(row["context_smape"] for row in members),
            "incremental_smape": mean(row["incremental_smape"] for row in members),
            "applied": sum(row["applied"] for row in members),
            "primary_changed": sum(row["primary_changed"] for row in members),
            "leakage": sum(row["temporal_leakage"] for row in members),
            "counterfactual_context_opportunity_smape": (
                mean(opportunity) if opportunity else None
            ),
        }
    gates = {
        "complete": len(rows) == int(manifest["cases"]),
        "zero_leakage": not any(row["temporal_leakage"] for row in rows),
        "publication_parity": all(row["publication_parity"] for row in rows),
        "false_influence_below_1pct": false_rate < 0.01,
        "admission_precision_at_least_90pct": precision >= 0.90,
        "admission_recall_at_least_80pct": recall >= 0.80,
        "future_covariate_smape_improves": (
            mean(row["incremental_smape"] for row in covariate) > 0
            if covariate else False),
        "repeated_event_smape_improves": (
            mean(row["incremental_smape"] for row in repeated) > 0
            if repeated else False),
        "prior_only_never_changes_primary": not any(row["primary_changed"] for row in prior),
        "disposition_contract_exact": dispositions_correct == len(rows),
        "interval_coverage_at_least_70pct": (mean(coverage_rows) >= 0.70
                                             if coverage_rows else False),
        "minimum_20_cases_per_family": per_family >= 20,
        "fresh_heldout_seed": bool(manifest.get("fresh_seed")),
    }
    return {
        "benchmark": "contextbench", "schema_version": 1,
        "generator": manifest.get("generator"), "seed": manifest.get("seed"),
        "fresh_seed": manifest.get("fresh_seed"), "cases": len(rows),
        "families": family_summary,
        "metrics": {
            "incremental_smape_influence_cases": (
                mean(row["incremental_smape"] for row in influence) if influence else None),
            "admission_precision": precision,
            "admission_precision_95ci": wilson(len(true_applied), len(applied)),
            "admission_recall": recall,
            "admission_recall_95ci": wilson(len(influence_applied), len(influence)),
            "false_influence_rate": false_rate,
            "false_influence_95ci": wilson(false_changes, false_trials),
            "effect_direction_accuracy": direction_accuracy,
            "disposition_accuracy": dispositions_correct / len(rows) if rows else None,
            "mean_interval_coverage": mean(coverage_rows) if coverage_rows else None,
            "leakage_count": sum(row["temporal_leakage"] for row in rows),
        },
        "gates": gates, "decision_ready": all(gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-gate-failure", action="store_true")
    args = parser.parse_args()
    corpus = Path(args.corpus_dir)
    cases = load_cases(corpus / "cases.jsonl")
    oracles = load_oracles(corpus / "oracle.jsonl")
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    case_bytes = (corpus / "cases.jsonl").read_bytes()
    oracle_bytes = (corpus / "oracle.jsonl").read_bytes()
    if hashlib.sha256(case_bytes).hexdigest() != manifest.get("cases_sha256"):
        raise SystemExit("cases.jsonl does not match its manifest hash")
    if hashlib.sha256(oracle_bytes).hexdigest() != manifest.get("oracle_sha256"):
        raise SystemExit("oracle.jsonl does not match its manifest hash")
    if len(cases) != int(manifest.get("cases", -1)):
        raise SystemExit("case count does not match the corpus manifest")
    if args.limit:
        # Round-robin family selection avoids a small run becoming one family.
        grouped: dict[str, list[Case]] = defaultdict(list)
        for case in cases:
            grouped[case.family].append(case)
        cases = []
        while len(cases) < args.limit and any(grouped.values()):
            for family in sorted(grouped):
                if grouped[family] and len(cases) < args.limit:
                    cases.append(grouped[family].pop(0))
        manifest = {**manifest, "cases": len(cases), "limited": True}
    missing = sorted(set(case.case_id for case in cases) - set(oracles))
    if missing:
        raise SystemExit(f"oracle is missing case IDs: {missing}")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="contextbench-", dir=str(output)))
    observations = [run_case(case, oracles[case.case_id], work) for case in cases]
    with (output / "observations.jsonl").open("w", encoding="utf-8") as handle:
        for row in observations:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize(observations, manifest)
    summary["corpus_manifest_sha256"] = hashlib.sha256(
        (corpus / "manifest.json").read_bytes()).hexdigest()
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["decision_ready"] or args.allow_gate_failure else 2


if __name__ == "__main__":
    raise SystemExit(main())

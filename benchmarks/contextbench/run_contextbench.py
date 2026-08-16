"""Run matched history-only and context-enabled ContextBench arms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from .schema import Case, Oracle, load_cases, load_oracles

EPOCH = datetime(2025, 1, 1, tzinfo=timezone.utc)
STEP = timedelta(hours=1)


def frequency_step(frequency: str) -> timedelta:
    normalized = frequency.strip().lower()
    if normalized in {"15min", "15m"}:
        return timedelta(minutes=15)
    if normalized in {"h", "1h", "hourly"}:
        return timedelta(hours=1)
    if normalized in {"d", "1d", "daily"}:
        return timedelta(days=1)
    raise ValueError(f"ContextBench does not support frequency {frequency!r}")


def smape(actual: tuple[float, ...] | list[float],
          predicted: list[float]) -> float:
    terms = []
    for truth, guess in zip(actual, predicted):
        denominator = abs(truth) + abs(guess)
        terms.append(0.0 if denominator <= 1e-12
                     else 200.0 * abs(truth - guess) / denominator)
    return mean(terms) if terms else float("nan")


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _write_history(case: Case, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["timestamp", "value"])
        step = frequency_step(case.frequency)
        for index, value in enumerate(case.history):
            writer.writerow([(EPOCH + index * step).isoformat(), repr(value)])


def _forecast(case: Case, root: Path, *, enriched: bool,
              event_override: list[dict[str, Any]] | None = None,
              use_covariates: bool = True,
              asserted_policy: bool = True) -> tuple[Any, Path]:
    from gnomon.context import events_from_list
    from gnomon.config import GnomonConfig
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
    config = GnomonConfig()
    if enriched:
        # ContextBench exercises every shipped admission lane.  Leaving
        # these default-off would measure configuration rather than the
        # numeric/textual gates named by the corpus.
        config.context.future_events = asserted_policy
        config.context.structural_events = asserted_policy
    output_name = ("history" if not enriched else
                   ("context" if asserted_policy else "default_policy"))
    return forecast(
        str(source), time_column="timestamp", target_column="value",
        horizon=case.horizon, frequency=case.frequency,
        output=str(root / output_name),
        context_events=events, covariates=covariates,
        config=config,
        minimum_support="best_effort",
    )


def _points(result: Any) -> list[float]:
    return [float(row.get("q50", row["point"])) for row in result.forecast]


def _leaked(artifact: Any, frequency: str) -> bool:
    # A run may read future-valid covariates only when they were published by
    # the forecast origin. Leakage is about known_time, never valid_time.
    origin = (datetime.fromisoformat(
        artifact.results[0].forecast[0]["timestamp"])
        - frequency_step(frequency))
    for evidence in artifact.evidence:
        if evidence.kind != "snapshot_access":
            continue
        for access in evidence.payload.get("accesses", []):
            known = access.get("known_time")
            if known and datetime.fromisoformat(str(known)) > origin:
                return True
    return False


def valid_disposition(family: str, applied: bool, disposition: str) -> bool:
    """Whether the public state matches the engine's measured decision.

    Hidden truth says whether context *could* help; it cannot require an
    engine to claim admission when its evidence gate declined.  That is what
    the separate recall metric measures.  The disposition contract instead
    checks that admitted context is called applied and non-admitted context is
    represented by the correct honest state for its lane.
    """
    if family == "confounded":
        # The event and covariate lanes retain their own typed dispositions.
        # A covariate winner can change the primary while the overlapping
        # event correctly remains scenario-only.
        return disposition in {"applied", "rejected", "scenario_only"}
    if applied:
        return disposition == "applied"
    if family in {"future_covariate", "numeric_claim", "structural_change"}:
        return disposition == "rejected"
    if family == "entity_scope":
        return disposition == "not_considered"
    return disposition == "scenario_only"


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
    baseline_result = baseline_artifact.results[0]
    baseline = _points(baseline_result)
    default_policy_changed: bool | None = None
    default_policy_smape: float | None = None
    if oracle.dimensions.get("admission_warrant") == "asserted":
        default_artifact, _ = _forecast(
            case, case_root, enriched=True, event_override=event_override,
            use_covariates=use_covariates, asserted_policy=False,
        )
        default_points = _points(default_artifact.results[0])
        default_policy_changed = any(
            abs(left - right) > 1e-9
            for left, right in zip(baseline, default_points)
        )
        default_policy_smape = smape(oracle.actual, default_points)
    context_result = context_artifact.results[0]
    contextual = _points(context_result)
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
        "oracle_dimensions": dict(oracle.dimensions),
        "history_smape": smape(oracle.actual, baseline),
        "context_smape": smape(oracle.actual, contextual),
        "counterfactual_smape": smape(oracle.counterfactual, contextual),
        "default_policy_primary_changed": default_policy_changed,
        "default_policy_smape": default_policy_smape,
        "incremental_smape": smape(oracle.actual, baseline) - smape(oracle.actual, contextual),
        "primary_changed": bool(changed_steps), "changed_steps": changed_steps,
        "should_influence": oracle.should_influence, "disposition": disposition,
        "expected_disposition": oracle.expected_disposition, "applied": applied,
        "disposition_valid": valid_disposition(
            case.family, applied, disposition),
        "effect_direction_expected": oracle.effect_direction,
        "effect_direction_inferred": inferred_direction,
        "effect_direction_correct": (inferred_direction == oracle.effect_direction
                                     if applied else None),
        "effect_magnitude_expected": oracle.effect_magnitude,
        # Oracle magnitude is the effect at its stated onset.  Averaging a
        # decaying/ramping path made the magnitude score depend on duration
        # rather than estimation quality.
        "effect_magnitude_inferred": (
            deltas[oracle.onset_step]
            if oracle.onset_step is not None
            and 0 <= oracle.onset_step < len(deltas) else
            (mean(nonzero) if nonzero else 0.0)
        ),
        "onset_step_expected": oracle.onset_step,
        "onset_step_inferred": min(changed_steps) if changed_steps else None,
        "duration_steps_expected": oracle.duration_steps,
        "duration_steps_inferred": len(changed_steps) if changed_steps else None,
        "interval_coverage": mean(covered) if covered else None,
        "temporal_leakage": _leaked(context_artifact, case.frequency),
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
    empirical = [row for row in rows if (row.get("oracle_dimensions") or {}).get(
        "admission_warrant", "empirical") == "empirical"]
    empirical_influence = [row for row in empirical if row["should_influence"]]
    irrelevant = by_family.get("irrelevant", [])
    prior = by_family.get("prior_only", [])
    applied = [row for row in rows if row["applied"]]
    empirical_applied = [row for row in empirical if row["applied"]]
    true_applied = [row for row in empirical_applied if row["should_influence"]]
    influence_applied = [row for row in empirical_influence if row["applied"]]
    false_rows = [row for row in empirical if not row["should_influence"]]
    asserted = [row for row in rows if (row.get("oracle_dimensions") or {}).get(
        "admission_warrant") == "asserted"]
    true_asserted = [row for row in asserted if row["should_influence"]]
    false_asserted = [row for row in asserted if not row["should_influence"]]
    default_asserted = [row for row in asserted
                        if row.get("default_policy_primary_changed") is not None]
    false_changes = sum(row["primary_changed"] for row in false_rows)
    false_trials = len(false_rows)
    precision = (len(true_applied) / len(empirical_applied)
                 if empirical_applied else 1.0)
    recall = (len(influence_applied) / len(empirical_influence)
              if empirical_influence else 0.0)
    false_rate = false_changes / false_trials if false_trials else 0.0
    direction_rows = [row for row in applied if row["effect_direction_correct"] is not None]
    direction_accuracy = (sum(row["effect_direction_correct"] for row in direction_rows)
                          / len(direction_rows) if direction_rows else None)
    coverage_rows = [row["interval_coverage"] for row in influence
                     if row["interval_coverage"] is not None]
    dispositions_correct = sum(bool(row.get(
        "disposition_valid",
        valid_disposition(row["family"], row["applied"], row["disposition"])))
        for row in rows)
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
    base_gates = {
        "complete": len(rows) == int(manifest["cases"]),
        "zero_leakage": not any(row["temporal_leakage"] for row in rows),
        "publication_parity": all(row["publication_parity"] for row in rows),
        "false_influence_below_1pct": false_rate < 0.01,
        "admission_precision_at_least_90pct": precision >= 0.90,
        "disposition_contract_exact": dispositions_correct == len(rows),
        "interval_coverage_at_least_70pct": (mean(coverage_rows) >= 0.70
                                             if coverage_rows else False),
        "fresh_heldout_seed": bool(manifest.get("fresh_seed")),
    }
    is_stress = str(manifest.get("generator", "")).startswith(
        "contextbench-stress-")
    if is_stress:
        strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
        snr_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            dimensions = row.get("oracle_dimensions") or {}
            strata[str(dimensions.get("stratum", "unspecified"))].append(row)
            if (dimensions.get("stratum") == "snr_sweep"
                    and dimensions.get("snr") is not None):
                snr_rows[str(dimensions["snr"])].append(row)
        required_strata = {
            "snr_sweep", "misleading_direction", "timing_uncertainty",
            "numeric_claim", "structural_claim", "confounded", "bitemporal",
            "entity_scope",
        }
        # "High SNR" means the strongest predeclared operating point, not a
        # pool of materially different points.  Pooling 4σ and 8σ previously
        # concealed the admission curve the stress corpus exists to expose.
        # The complete curve (including 4σ) remains a first-class report.
        maximum_snr = max((float(level) for level in snr_rows), default=None)
        high_snr = [row for level, members in snr_rows.items()
                    if maximum_snr is not None and float(level) == maximum_snr
                    for row in members]
        high_snr_recall = (sum(row["applied"] for row in high_snr)
                           / len(high_snr) if high_snr else 0.0)
        true_numeric = [row for row in true_asserted
                        if row["family"] == "numeric_claim"]
        true_structural = [row for row in true_asserted
                           if row["family"] == "structural_change"]
        stress_gates = {
            "all_required_stress_strata_present": required_strata <= set(strata),
            "snr_curve_has_at_least_5_levels": len(snr_rows) >= 5,
            # Low-SNR abstention is a result, not a benchmark failure.  The
            # hard recall gate applies only where signal is clearly usable;
            # the complete curve remains in the report for product choices.
            "high_snr_admission_recall_at_least_80pct": high_snr_recall >= 0.80,
            "asserted_claim_outcomes_are_scored": bool(asserted),
            "asserted_claims_never_change_primary_by_default": (
                bool(default_asserted) and not any(
                    row["default_policy_primary_changed"]
                    for row in default_asserted)
            ),
            "true_numeric_claims_do_not_harm_on_average": bool(true_numeric) and mean(
                row["incremental_smape"] for row in true_numeric) >= 0.0,
            "true_structural_claims_do_not_harm_on_average": bool(true_structural) and mean(
                row["incremental_smape"] for row in true_structural) >= 0.0,
        }
    else:
        strata, snr_rows = {}, {}
        stress_gates = {
            "admission_recall_at_least_80pct": recall >= 0.80,
            "future_covariate_smape_improves": (
                mean(row["incremental_smape"] for row in covariate) > 0
                if covariate else False),
            "repeated_event_smape_improves": (
                mean(row["incremental_smape"] for row in repeated) > 0
                if repeated else False),
            "prior_only_never_changes_primary": not any(
                row["primary_changed"] for row in prior),
            "minimum_20_cases_per_family": per_family >= 20,
        }
    gates = {**base_gates, **stress_gates}

    def sliced(members: list[dict[str, Any]]) -> dict[str, Any]:
        empirical_members = [row for row in members
                             if (row.get("oracle_dimensions") or {}).get(
                                 "admission_warrant", "empirical") == "empirical"]
        useful = [row for row in empirical_members if row["should_influence"]]
        admitted = [row for row in members if row["applied"]]
        empirical_admitted = [row for row in empirical_members if row["applied"]]
        false = [row for row in empirical_members if not row["should_influence"]]
        asserted_false = [row for row in members
                          if (row.get("oracle_dimensions") or {}).get(
                              "admission_warrant") == "asserted"
                          and not row["should_influence"]]
        return {
            "cases": len(members),
            "admission_precision": (sum(row["should_influence"]
                                         for row in empirical_admitted)
                                    / len(empirical_admitted)
                                    if empirical_admitted else None),
            "admission_recall": (sum(row["applied"] for row in useful)
                                 / len(useful) if useful else None),
            "false_influence_rate": (sum(row["primary_changed"] for row in false)
                                     / len(false) if false else None),
            "incremental_smape": mean(row["incremental_smape"] for row in members),
            "harmful_admissions": sum(
                row["applied"] and row["incremental_smape"] < 0 for row in members),
            "realized_truth_rate_when_applied": (sum(
                row["should_influence"] for row in admitted) / len(admitted)
                if admitted else None),
            "false_asserted_claim_primary_change_rate": (sum(
                row["primary_changed"] for row in asserted_false)
                / len(asserted_false) if asserted_false else None),
            "magnitude_mae_when_applied": (mean(abs(
                row["effect_magnitude_inferred"] - row["effect_magnitude_expected"])
                for row in admitted) if admitted else None),
            "onset_mae_when_applied": (mean(abs(
                row["onset_step_inferred"] - row["onset_step_expected"])
                for row in admitted if row["onset_step_inferred"] is not None
                and row["onset_step_expected"] is not None) if any(
                    row["onset_step_inferred"] is not None
                    and row["onset_step_expected"] is not None
                    for row in admitted) else None),
        }
    dimension_slices: dict[str, dict[str, list[dict[str, Any]]]] = {}
    if is_stress:
        for dimension in (
            "frequency", "seasonality_period", "narrative_style",
            "effect_shape", "claim_truth", "known_at", "confounding",
        ):
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                value = (row.get("oracle_dimensions") or {}).get(dimension)
                if value is not None:
                    groups[str(value)].append(row)
            dimension_slices[dimension] = groups
    return {
        "benchmark": "contextbench", "schema_version": 1,
        "generator": manifest.get("generator"), "seed": manifest.get("seed"),
        "fresh_seed": manifest.get("fresh_seed"), "cases": len(rows),
        "families": family_summary,
        "strata": {name: sliced(members)
                    for name, members in sorted(strata.items())},
        "snr_curve": {name: sliced(members)
                      for name, members in sorted(
                          snr_rows.items(), key=lambda item: float(item[0]))},
        "dimensions": {
            dimension: {value: sliced(members)
                        for value, members in sorted(groups.items())}
            for dimension, groups in sorted(dimension_slices.items())
        },
        "metrics": {
            "incremental_smape_influence_cases": (
                mean(row["incremental_smape"] for row in influence) if influence else None),
            "admission_precision": precision,
            "admission_precision_95ci": wilson(
                len(true_applied), len(empirical_applied)),
            "admission_recall": recall,
            "admission_recall_95ci": wilson(
                len(influence_applied), len(empirical_influence)),
            "false_influence_rate": false_rate,
            "false_influence_95ci": wilson(false_changes, false_trials),
            "asserted_claim_cases": len(asserted),
            "false_asserted_claim_cases": len(false_asserted),
            "false_asserted_claim_primary_change_rate": (
                sum(row["primary_changed"] for row in false_asserted)
                / len(false_asserted) if false_asserted else None),
            "false_asserted_claim_mean_incremental_smape": (
                mean(row["incremental_smape"] for row in false_asserted)
                if false_asserted else None),
            "true_asserted_claim_harm_rate": (sum(
                row["incremental_smape"] < 0 for row in true_asserted)
                / len(true_asserted) if true_asserted else None),
            "true_asserted_claim_mean_incremental_smape": (mean(
                row["incremental_smape"] for row in true_asserted)
                if true_asserted else None),
            "default_policy_asserted_primary_change_rate": (sum(
                bool(row["default_policy_primary_changed"])
                for row in default_asserted) / len(default_asserted)
                if default_asserted else None),
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

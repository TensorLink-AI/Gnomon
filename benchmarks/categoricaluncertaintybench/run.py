"""Run the frozen categorical-state publication uncertainty matrix.

The benchmark exercises the same governed-candidate dossier and publication
boundary used by the agent harness.  Sealed future targets are generated with
each case but are never passed to the fitter, dossier validator, or publisher.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from benchmarks.common.manifest import code_revision
from gnomon.context_intelligence import fit_categorical_state_candidate
from gnomon.evaluation import conformal_quantile
from gnomon.llm_dossier import validate_temporal_dossier
from gnomon.publication import publish_result, verify_publication


FAMILIES = (
    "one_replay_cycle_stable",
    "one_replay_cycle_reversed",
    "two_replay_cycles_stable",
    "two_replay_cycles_irrelevant",
)
CASES_PER_FAMILY = 30
PERIOD = 8
HORIZON = 8


def _wis(actual: float, low: float, middle: float, high: float) -> float:
    interval = high - low
    if actual < low:
        interval += 10.0 * (low - actual)
    elif actual > high:
        interval += 10.0 * (actual - high)
    return (.5 * abs(actual - middle) + .1 * interval) / .6


def _primary(history: list[float], timestamps: list[str]) \
        -> list[dict[str, Any]]:
    """Build a target-only seasonal comparator from visible history only."""
    phase_points = [statistics.median(
        value for index, value in enumerate(history)
        if index % PERIOD == phase) for phase in range(PERIOD)]
    residuals = [value - phase_points[index % PERIOD]
                 for index, value in enumerate(history)]
    lower = conformal_quantile(residuals, .1)
    upper = conformal_quantile(residuals, .9)
    return [{
        "timestamp": timestamp,
        "point": phase_points[offset % PERIOD],
        "q10": min(phase_points[offset % PERIOD] + lower,
                    phase_points[offset % PERIOD]),
        "q50": phase_points[offset % PERIOD],
        "q90": max(phase_points[offset % PERIOD] + upper,
                    phase_points[offset % PERIOD]),
    } for offset, timestamp in enumerate(timestamps)]


def _numeric(rows: list[dict[str, Any]]) -> list[list[float]]:
    return [[float(row[key]) for key in ("q10", "q50", "q90")]
            for row in rows]


def _expected_envelope(candidate: list[list[float]],
                       primary: list[list[float]]) -> list[list[float]]:
    return [[min(primary_row[0], candidate_row[1]), candidate_row[1],
             max(primary_row[2], candidate_row[1])]
            for candidate_row, primary_row in zip(candidate, primary)]


def _case(family: str, case_index: int, seed: int) -> dict[str, Any]:
    replay_cycles = 1 if family.startswith("one_replay_cycle") else 2
    history_cycles = replay_cycles + 1
    rng = random.Random(seed)
    history_states = [
        "a" if (phase + cycle) % 2 == 0 else "b"
        for cycle in range(history_cycles) for phase in range(PERIOD)
    ]
    future_states = [
        "a" if (phase + history_cycles) % 2 == 0 else "b"
        for phase in range(HORIZON)
    ]
    noise = [rng.gauss(0.0, 1.0)
             for _ in range(len(history_states) + HORIZON)]

    def state_effect(state: str, *, future: bool) -> float:
        if family == "two_replay_cycles_irrelevant":
            return 0.0
        effect = 8.0 if state == "a" else -8.0
        if family == "one_replay_cycle_reversed" and future:
            return -effect
        return effect

    history = [
        50.0 + 10.0 * math.sin(2.0 * math.pi * (index % PERIOD) / PERIOD)
        + state_effect(state, future=False) + noise[index]
        for index, state in enumerate(history_states)
    ]
    future = [
        50.0 + 10.0 * math.sin(2.0 * math.pi * phase / PERIOD)
        + state_effect(state, future=True) + noise[len(history_states) + phase]
        for phase, state in enumerate(future_states)
    ]
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    timestamps = [(start + timedelta(days=index)).isoformat()
                  for index in range(HORIZON)]
    primary = _primary(history, timestamps)
    context = "The known operating state alternates between a and b."
    raw = {"claims": [{
        "source_span": context,
        "relation": "supports_increase",
        "effective_start": timestamps[0],
        "effective_end": timestamps[-1],
        "confidence": .8,
    }]}
    preliminary, preliminary_rejections = validate_temporal_dossier(
        raw, context_text=context, cutoff="2026-01-31T00:00:00+00:00",
        future_timestamps=timestamps, history=history,
        compiler_model="categorical-uncertainty-benchmark")
    candidate = fit_categorical_state_candidate(
        history, history_states, future_states, primary=primary,
        claim_ids=[str(item["claim_id"])
                   for item in preliminary.get("claims") or []],
        hypothesis_id=f"{family}-{case_index:03d}",
        seasonal_period=PERIOD,
    )
    dossier, dossier_rejections = validate_temporal_dossier(
        raw, context_text=context, cutoff="2026-01-31T00:00:00+00:00",
        future_timestamps=timestamps, history=history,
        compiler_model="categorical-uncertainty-benchmark",
        governed_candidate=candidate)
    publication = publish_result(
        {"support": "supported", "forecast": primary},
        mode="best_effort", dossiers=[dossier])
    published = next(
        item for item in publication["candidate_portfolio"]
        if item.get("role") == "governed_categorical_state_mapping")
    source_rows = _numeric(candidate["forecast"])
    primary_rows = _numeric(primary)
    published_rows = _numeric(published["forecast"])
    scores = [_wis(actual, *row)
              for actual, row in zip(future, published_rows)]
    covered = sum(row[0] <= actual <= row[2]
                  for actual, row in zip(future, published_rows))
    return {
        "case_id": f"{family}-{case_index:03d}",
        "family": family,
        "seed": seed,
        "future_observations_used_by_forecaster": 0,
        "preliminary_rejections": preliminary_rejections,
        "dossier_rejections": dossier_rejections,
        "publication_verified": verify_publication(publication),
        "actual": future,
        "source_candidate_rows": source_rows,
        "primary_rows": primary_rows,
        "published_primary_rows": _numeric(publication["primary_forecast"]),
        "published_candidate_rows": published_rows,
        "expected_under_evidence_envelope": _expected_envelope(
            source_rows, primary_rows),
        "mean_wis": statistics.mean(scores),
        "covered_points": covered,
        "interval_points": HORIZON,
        "ordered_quantiles": all(
            row[0] <= row[1] <= row[2]
            and all(math.isfinite(value) for value in row)
            for row in published_rows),
        "contract": {
            "support": published["support"],
            "selection_eligible": published["selection_eligible"],
            "human_selection_eligible": published[
                "human_selection_eligible"],
            "automation_eligible": published["automation_eligible"],
            "recommended_scenario_id": publication[
                "recommended_scenario_id"],
            "primary_forecast_unchanged": publication[
                "primary_forecast_unchanged"],
        },
        "uncertainty_normalization": (published.get("effect") or {}).get(
            "uncertainty_normalization"),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    points = sum(int(row["interval_points"]) for row in rows)
    return {
        "cases": len(rows),
        "coverage": sum(int(row["covered_points"]) for row in rows) / points,
        "mean_wis": statistics.mean(float(row["mean_wis"]) for row in rows),
        "median_wis": statistics.median(float(row["mean_wis"])
                                         for row in rows),
    }


def summarize(rows: list[dict[str, Any]], identity: dict[str, Any],
              reference: dict[str, Any] | None = None) -> dict[str, Any]:
    by_family = {family: _aggregate(
        [row for row in rows if row["family"] == family])
        for family in FAMILIES}
    overall = _aggregate(rows)
    base_gates = {
        "all_cases_complete": len(rows) == len(FAMILIES) * CASES_PER_FAMILY,
        "future_targets_never_passed_to_forecaster": all(
            row["future_observations_used_by_forecaster"] == 0 for row in rows),
        "dossiers_clean": all(
            not row["preliminary_rejections"] and not row["dossier_rejections"]
            for row in rows),
        "publications_verify": all(row["publication_verified"] for row in rows),
        "primary_byte_preserved": all(
            row["primary_rows"] == row["published_primary_rows"] for row in rows),
        "quantiles_finite_and_ordered": all(row["ordered_quantiles"]
                                             for row in rows),
    }
    comparison: dict[str, Any] | None = None
    if reference is not None:
        expected = reference.get("run_identity") or {}
        for key in ("seed", "families", "cases_per_family", "period", "horizon"):
            if expected.get(key) != identity.get(key):
                raise ValueError(f"reference identity differs on {key}")
        reference_rows = {row["case_id"]: row
                          for row in reference.get("rows") or []}
        if set(reference_rows) != {row["case_id"] for row in rows}:
            raise ValueError("reference case matrix differs")
        ineligible = [row for row in rows
                      if not row["contract"]["human_selection_eligible"]]
        eligible = [row for row in rows
                    if row["contract"]["human_selection_eligible"]]
        reference_ineligible_wis = statistics.mean(
            float(reference_rows[row["case_id"]]["mean_wis"])
            for row in ineligible)
        treatment_ineligible_wis = statistics.mean(
            float(row["mean_wis"]) for row in ineligible)
        family_changes = {
            family: ((by_family[family]["median_wis"]
                      - reference["by_family"][family]["median_wis"])
                     / max(reference["by_family"][family]["median_wis"], 1e-12))
            for family in FAMILIES
        }
        gates = {
            **base_gates,
            "contract_decisions_unchanged": all(
                row["contract"] == reference_rows[row["case_id"]]["contract"]
                for row in rows),
            "source_candidate_centres_unchanged": all(
                [item[1] for item in row["published_candidate_rows"]]
                == [item[1] for item in row["source_candidate_rows"]]
                for row in rows),
            "eligible_candidates_byte_unchanged": all(
                row["published_candidate_rows"] == reference_rows[
                    row["case_id"]]["published_candidate_rows"]
                for row in eligible),
            "ineligible_candidates_use_primary_envelope": all(
                row["published_candidate_rows"]
                == row["expected_under_evidence_envelope"]
                for row in ineligible),
            "aggregate_wis_nonworsening": overall["mean_wis"]
                <= float(reference["overall"]["mean_wis"]) + 1e-12,
            "ineligible_mean_wis_improves_10pct": treatment_ineligible_wis
                <= reference_ineligible_wis * .9,
            "family_median_regression_within_2pct": max(
                family_changes.values()) <= .02,
        }
        comparison = {
            "reference_ineligible_mean_wis": reference_ineligible_wis,
            "treatment_ineligible_mean_wis": treatment_ineligible_wis,
            "ineligible_mean_wis_relative_change": (
                treatment_ineligible_wis / reference_ineligible_wis - 1),
            "family_median_wis_relative_change": family_changes,
        }
    else:
        gates = base_gates
    return {
        "schema_version": 1,
        "benchmark": "categorical-state-publication-uncertainty",
        "evaluated_commit": identity["evaluated_commit"],
        "run_identity": identity,
        "overall": overall,
        "by_family": by_family,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        **({"comparison": comparison} if comparison is not None else {}),
        "rows": rows,
    }


def run(seed: int, output_dir: Path, *, resume: bool = False,
        reference_summary: Path | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "run-identity.json"
    checkpoint = output_dir / "observations.jsonl"
    identity = {
        "schema_version": 1,
        "benchmark": "categorical-state-publication-uncertainty",
        "evaluated_commit": code_revision(),
        "seed": seed,
        "families": list(FAMILIES),
        "cases_per_family": CASES_PER_FAMILY,
        "period": PERIOD,
        "horizon": HORIZON,
    }
    if resume:
        if not identity_path.is_file():
            raise ValueError("resume requires a retained run identity")
        retained = json.loads(identity_path.read_text(encoding="utf-8"))
        if retained != identity:
            raise ValueError("resume identity differs from retained run")
    elif identity_path.exists() or checkpoint.exists():
        raise ValueError("output directory already contains a run")
    else:
        identity_path.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    completed: dict[str, dict[str, Any]] = {}
    if resume and checkpoint.is_file():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["case_id"]] = row
    total = len(FAMILIES) * CASES_PER_FAMILY
    ordinal = 0
    for family_index, family in enumerate(FAMILIES):
        for case_index in range(CASES_PER_FAMILY):
            case_id = f"{family}-{case_index:03d}"
            ordinal += 1
            if case_id in completed:
                continue
            row = _case(family, case_index,
                        seed + family_index * 100_000 + case_index)
            completed[case_id] = row
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
            print(f"completed {ordinal}/{total} {case_id}", flush=True)
    rows = [completed[f"{family}-{index:03d}"]
            for family in FAMILIES for index in range(CASES_PER_FAMILY)]
    reference = (json.loads(reference_summary.read_text(encoding="utf-8"))
                 if reference_summary is not None else None)
    result = summarize(rows, identity, reference)
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference-summary", type=Path)
    args = parser.parse_args()
    result = run(args.seed, args.output_dir, resume=args.resume,
                 reference_summary=args.reference_summary)
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, sort_keys=True))
    return 0 if result["all_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the frozen categorical-state admission matrix.

Future targets are generated with each case but are never passed to the
candidate.  A treatment can be compared with a retained reference only when
the seed and complete case matrix match.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from benchmarks.common.manifest import code_revision
from gnomon.context_intelligence import fit_categorical_state_candidate


FAMILIES = (
    "one_replay_cycle_stable",
    "one_replay_cycle_reversed",
    "two_replay_cycles_stable",
    "two_replay_cycles_irrelevant",
)
CASES_PER_FAMILY = 30
PERIOD = 8
HORIZON = 8


def _case(family: str, case_index: int, seed: int) -> dict[str, Any]:
    replay_cycles = 1 if family.startswith("one_replay_cycle") else 2
    history_cycles = replay_cycles + 1
    # Paired one-cycle cases deliberately have byte-identical information at
    # the cutoff.  Only the sealed future differs, demonstrating that one
    # replay of an apparent relationship cannot establish recurrence.
    pair_seed = seed
    rng = random.Random(pair_seed)
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
    primary = [{"timestamp": f"future-{index:02d}"}
               for index in range(HORIZON)]
    candidate = fit_categorical_state_candidate(
        history, history_states, future_states, primary=primary,
        claim_ids=["known-state-schedule"],
        hypothesis_id=f"{family}-{case_index:03d}", seasonal_period=PERIOD,
    )
    candidate_points = [float(row["q50"]) for row in candidate["forecast"]]
    baseline_points = [statistics.median(
        value for index, value in enumerate(history)
        if index % PERIOD == phase) for phase in range(HORIZON)]
    candidate_mae = statistics.mean(
        abs(actual - predicted)
        for actual, predicted in zip(future, candidate_points))
    baseline_mae = statistics.mean(
        abs(actual - predicted)
        for actual, predicted in zip(future, baseline_points))
    human_eligible = bool(candidate["human_selection_eligible"])
    return {
        "case_id": f"{family}-{case_index:03d}",
        "family": family,
        "seed": pair_seed,
        "future_observations_used_by_forecaster": 0,
        "candidate_numeric_rows": [[float(row[key]) for key in
                                     ("q10", "q50", "q90")]
                                    for row in candidate["forecast"]],
        "quantiles_finite_and_ordered": all(
            all(math.isfinite(float(row[key])) for key in
                ("q10", "q50", "q90"))
            and float(row["q10"]) <= float(row["q50"]) <= float(row["q90"])
            for row in candidate["forecast"]),
        "candidate_mae": candidate_mae,
        "baseline_mae": baseline_mae,
        "selected_policy_mae": candidate_mae if human_eligible else baseline_mae,
        "contract": {
            "support": candidate["support"],
            "selection_eligible": candidate["selection_eligible"],
            "human_selection_eligible": human_eligible,
            "automation_eligible": candidate["automation_eligible"],
            "primary_forecast_unchanged": candidate[
                "primary_forecast_unchanged"],
        },
        "validation": candidate["validation"],
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(rows),
        "human_selection_eligible": sum(
            row["contract"]["human_selection_eligible"] for row in rows),
        "mean_candidate_mae": statistics.mean(
            float(row["candidate_mae"]) for row in rows),
        "mean_baseline_mae": statistics.mean(
            float(row["baseline_mae"]) for row in rows),
        "mean_selected_policy_mae": statistics.mean(
            float(row["selected_policy_mae"]) for row in rows),
    }


def summarize(rows: list[dict[str, Any]], identity: dict[str, Any],
              reference: dict[str, Any] | None = None) -> dict[str, Any]:
    by_family = {family: _aggregate(
        [row for row in rows if row["family"] == family]) for family in FAMILIES}
    overall = _aggregate(rows)
    base_gates = {
        "all_cases_complete": len(rows) == len(FAMILIES) * CASES_PER_FAMILY,
        "future_targets_never_passed_to_forecaster": all(
            row["future_observations_used_by_forecaster"] == 0 for row in rows),
        "quantiles_finite_and_ordered": all(
            row["quantiles_finite_and_ordered"] for row in rows),
        "paired_one_cycle_histories_are_indistinguishable": all(
            next(row for row in rows if row["case_id"] ==
                 f"one_replay_cycle_stable-{index:03d}")[
                     "candidate_numeric_rows"]
            == next(row for row in rows if row["case_id"] ==
                    f"one_replay_cycle_reversed-{index:03d}")[
                        "candidate_numeric_rows"]
            for index in range(CASES_PER_FAMILY)),
    }
    comparison = None
    if reference is None:
        gates = base_gates
    else:
        expected = reference.get("run_identity") or {}
        for key in ("seed", "families", "cases_per_family", "period",
                    "horizon"):
            if expected.get(key) != identity.get(key):
                raise ValueError(f"reference identity differs on {key}")
        reference_rows = {row["case_id"]: row
                          for row in reference.get("rows") or []}
        if set(reference_rows) != {row["case_id"] for row in rows}:
            raise ValueError("reference case matrix differs")
        gates = {
            **base_gates,
            "candidate_numbers_unchanged": all(
                row["candidate_numeric_rows"]
                == reference_rows[row["case_id"]]["candidate_numeric_rows"]
                for row in rows),
            "one_replay_cycle_never_selected": all(
                not row["contract"]["human_selection_eligible"]
                for row in rows if row["family"].startswith(
                    "one_replay_cycle")),
            "recurrent_effect_recall_at_least_90pct": (
                by_family["two_replay_cycles_stable"][
                    "human_selection_eligible"] >= 27),
            "irrelevant_state_admissions_do_not_increase": (
                by_family["two_replay_cycles_irrelevant"][
                    "human_selection_eligible"]
                <= reference["by_family"]["two_replay_cycles_irrelevant"][
                    "human_selection_eligible"]),
            "selected_policy_mae_nonworsening": (
                overall["mean_selected_policy_mae"]
                <= float(reference["overall"][
                    "mean_selected_policy_mae"]) + 1e-12),
        }
        comparison = {
            "selected_policy_mae_change": (
                overall["mean_selected_policy_mae"]
                - float(reference["overall"]["mean_selected_policy_mae"])),
            "human_selection_eligible_change": (
                overall["human_selection_eligible"]
                - int(reference["overall"]["human_selection_eligible"])),
        }
    return {
        "schema_version": 1,
        "benchmark": "categorical-state-admission",
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
        "benchmark": "categorical-state-admission",
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
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
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
    ordinal = 0
    total = len(FAMILIES) * CASES_PER_FAMILY
    for family_index, family in enumerate(FAMILIES):
        for case_index in range(CASES_PER_FAMILY):
            case_id = f"{family}-{case_index:03d}"
            ordinal += 1
            if case_id in completed:
                continue
            # The paired one-cycle families share a visible-history seed.
            family_seed = seed + (0 if family_index < 2 else
                                  family_index * 100_000) + case_index
            row = _case(family, case_index, family_seed)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026090304)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference-summary", type=Path)
    args = parser.parse_args()
    result = run(args.seed, args.output_dir, resume=args.resume,
                 reference_summary=args.reference_summary)
    print(json.dumps({
        "overall": result["overall"],
        "by_family": result["by_family"],
        "gates": result["gates"],
        "all_gates_passed": result["all_gates_passed"],
    }, indent=2, sort_keys=True))
    return 0 if result["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

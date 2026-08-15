"""Aggregate replicated ContextBench surface runs with paired uncertainty.

Replicates are clustered by case before bootstrapping, so repeated model
samples do not masquerade as additional independent benchmark tasks.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def _failure_class(row: dict[str, Any]) -> str | None:
    if row.get("status") == "answered":
        return None
    if row.get("failure_class"):
        return str(row["failure_class"])
    error = str(row.get("error", ""))
    if "did not submit" in error:
        return "agent_non_submission"
    if error.startswith("OpenRouterError:"):
        return "provider_failure"
    return "harness_failure"


def _cluster_ci(rows: list[dict[str, Any]], metric: Callable[[dict[str, Any]], float],
                *, seed: int = 1729, draws: int = 5000) -> list[float] | None:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)
    ids = sorted(by_case)
    if not ids:
        return None
    per_case = {case_id: mean(metric(row) for row in by_case[case_id])
                for case_id in ids}
    rng = random.Random(seed)
    estimates = sorted(mean(per_case[rng.choice(ids)] for _ in ids)
                       for _ in range(draws))
    return [estimates[int(draws * 0.025)], estimates[int(draws * 0.975)]]


def aggregate(run_dirs: list[Path]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    replicates: dict[str, int] = defaultdict(int)
    for directory in run_dirs:
        summary_path = directory / "summary.json"
        observation_path = directory / "observations.jsonl"
        if not observation_path.is_file():
            raise ValueError(f"missing observations: {observation_path}")
        summary = (json.loads(summary_path.read_text())
                   if summary_path.is_file() else {})
        profile = str(summary.get("profile") or
                      directory.name.split("-r", 1)[0].removeprefix("v2-"))
        grouped[profile].extend(_read(observation_path))
        replicates[profile] += 1

    arms: dict[str, Any] = {}
    for profile, rows in sorted(grouped.items()):
        answered = [row for row in rows if row.get("status") == "answered"]
        failures = defaultdict(int)
        for row in rows:
            kind = _failure_class(row)
            if kind:
                failures[kind] += 1
        families: dict[str, Any] = {}
        for family in sorted({str(row["family"]) for row in answered}):
            members = [row for row in answered if row["family"] == family]
            lift = mean(float(row["incremental_smape"]) for row in members)
            families[family] = {
                "successful_observations": len(members),
                "unique_cases": len({row["case_id"] for row in members}),
                "history_smape": mean(float(row["history_smape"])
                                      for row in members),
                "context_smape": mean(float(row["context_smape"])
                                      for row in members),
                "incremental_smape": lift,
                "incremental_smape_case_clustered_95ci": _cluster_ci(
                    members, lambda row: float(row["incremental_smape"])),
            }
        calls = [int(row.get("history_calls", 0)) +
                 int(row.get("context_calls", 0)) for row in answered]
        false_trials = [row for row in answered if not row["should_influence"]]
        influence = [row for row in answered if row["should_influence"]]
        parity_eligible = [row for row in answered
                           if row.get("publication_parity") is not None]
        arms[profile] = {
            "replicates": replicates[profile],
            "attempted_observations": len(rows),
            "unique_cases": len({row["case_id"] for row in rows}),
            "successful_pairs": len(answered),
            "completion_rate": len(answered) / len(rows) if rows else 0.0,
            "failures": dict(sorted(failures.items())),
            "agent_calls_mean": mean(calls) if calls else None,
            "agent_calls_median": (sorted(calls)[len(calls) // 2]
                                   if calls else None),
            "compiler_logical_calls": sum(int(row.get("compiler_calls", 0))
                                          for row in rows),
            "compiler_executed_calls": sum(
                int(row.get("compiler_calls", 0))
                if row.get("compiler_called") else 0 for row in rows),
            "compiler_cost_note": (
                "compiler calls are context-preparation cost and are excluded "
                "from agent surface calls; executed calls exclude receipt reuse"),
            "admission_recall": (mean(bool(row.get("applied"))
                                      for row in influence)
                                 if influence else None),
            "false_influence_rate": (mean(bool(row.get("primary_changed"))
                                          for row in false_trials)
                                     if false_trials else None),
            "publication_parity": {
                "answered": len(answered),
                "identity_produced": len(parity_eligible),
                "identity_matched": sum(
                    row.get("publication_parity") is True
                    for row in parity_eligible),
                "match_rate_when_produced": (mean(
                    row.get("publication_parity") is True
                    for row in parity_eligible) if parity_eligible else None),
            },
            "families": families,
        }
    return {"benchmark": "contextbench-surface-comparison",
            "bootstrap": {"unit": "case_id", "draws": 5000,
                          "confidence": 0.95, "seed": 1729},
            "arms": arms}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True,
                        help="replicate directory; repeat for every arm/run")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = aggregate([Path(value) for value in args.run_dir])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

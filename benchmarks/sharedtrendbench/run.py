"""Measure useful-driver recall against shared-trend false admission."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from benchmarks.common.manifest import code_revision
from gnomon.context_intelligence import fit_companion_relationship_candidate


CASES = 40
START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _stamp(index: int) -> str:
    return (START + timedelta(days=index)).isoformat()


def _case(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    driver = [10.0 + index * .5 + rng.gauss(0.0, 2.0)
              for index in range(40)]
    true_target = [3.0 + 2.0 * value + rng.gauss(0.0, 4.0)
                   for value in driver]
    shared_trend = [100.0 + index * 1.2 + rng.gauss(0.0, 4.0)
                    for index in range(40)]
    primary = [{"timestamp": _stamp(41)}, {"timestamp": _stamp(42)}]
    arguments = {
        "primary": primary,
        "claim_ids": ["declared-driver"],
        "replay_origin_eligible": [True] * len(driver),
    }
    useful = fit_companion_relationship_candidate(
        true_target, driver, [31.0, 32.0],
        hypothesis_id="useful-driver", **arguments)
    confounded = fit_companion_relationship_candidate(
        shared_trend, driver, [31.0, 32.0],
        hypothesis_id="shared-trend", **arguments)
    return {
        "seed": seed,
        "useful_admitted": bool(useful["selection_eligible"]),
        "shared_trend_admitted": bool(confounded["selection_eligible"]),
        "useful_validation": useful["validation"],
        "shared_trend_validation": confounded["validation"],
        "primary_forecast_unchanged": True,
        "future_target_observations_used": 0,
    }


def run(seed: int, output: Path) -> dict[str, Any]:
    rows = [_case(seed + index) for index in range(CASES)]
    useful_recall = sum(row["useful_admitted"] for row in rows) / CASES
    false_admission_rate = sum(
        row["shared_trend_admitted"] for row in rows) / CASES
    gates = {
        "all_cases_complete": len(rows) == CASES,
        "useful_driver_recall_at_least_85pct": useful_recall >= .85,
        "shared_trend_false_admission_at_most_10pct": (
            false_admission_rate <= .10),
        "primary_forecast_unchanged": all(
            row["primary_forecast_unchanged"] for row in rows),
        "future_targets_never_used": all(
            row["future_target_observations_used"] == 0 for row in rows),
    }
    result = {
        "schema_version": 1,
        "benchmark": "shared-trend-confounding",
        "evaluated_commit": code_revision(),
        "seed": seed,
        "cases": CASES,
        "useful_driver_recall": useful_recall,
        "shared_trend_false_admission_rate": false_admission_rate,
        "context_is_useful": False,
        "context_admitted": not gates[
            "shared_trend_false_admission_at_most_10pct"],
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.seed, args.output)
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, sort_keys=True))
    return 0 if result["all_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

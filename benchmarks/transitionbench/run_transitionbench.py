"""Independent observed-transition benchmark for TemporalEvidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gnomon.temporal_evidence import window_evidence  # noqa: E402


def _case(seed: int, prop: str, label: str, *, difficulty: str = "easy",
          width: int = 96) -> tuple[list[float], int]:
    rng = random.Random(seed)
    season = 24 if prop == "seasonality" else 1
    reference = [rng.gauss(0, .35) for _ in range(width)]
    recent = [rng.gauss(0, .35) for _ in range(width)]
    scale = {"easy": 1.0, "moderate": .4, "marginal": .16}[difficulty]
    if prop == "level":
        delta = {"higher": 2.0 * scale, "lower": -2.0 * scale,
                 "similar": 0.0}[label]
        recent = [value + delta for value in recent]
    elif prop == "trend":
        slope = {"upward": .08 * scale, "downward": -.08 * scale,
                 "constant": 0.0}[label]
        recent = [value + slope * index for index, value in enumerate(recent)]
    elif prop == "volatility":
        ratios = {
            "easy": {"increased": 2.5, "decreased": .35, "stable": 1.0},
            "moderate": {"increased": 1.65, "decreased": .62, "stable": 1.0},
            "marginal": {"increased": 1.28, "decreased": .78, "stable": 1.0},
        }
        ratio = ratios[difficulty][label]
        recent = [value * ratio for value in recent]
    elif prop == "seasonality":
        amplitudes = {
            "easy": (6.0, .5), "moderate": (4.5, 1.7),
            "marginal": (3.8, 2.3),
        }
        strong, weak = amplitudes[difficulty]
        amplitude = {"strengthened": strong, "weakened": weak,
                     "stable": 3.0, "phase_shifted": 3.0}[label]
        shift = ({"easy": 7, "moderate": 4, "marginal": 2}[difficulty]
                 if label == "phase_shifted" else 0)
        reference = [3 * math.sin(2 * math.pi * index / season)
                     + rng.gauss(0, .2) for index in range(width)]
        recent = [amplitude * math.sin(2 * math.pi * (index - shift) / season)
                  + rng.gauss(0, .2) for index in range(width)]
    elif prop == "regime":
        delta = 2.0 * scale if label == "shift" else 0.0
        recent = [value + delta for value in recent]
    elif prop == "extreme":
        if label == "increased":
            for index in range(4, width, 16):
                recent[index] += 5.0 * scale
        elif label == "decreased":
            for index in range(4, width, 16):
                reference[index] += 5.0 * scale
    return reference + recent, season


def run(seed: int = 73100, replicates: int = 30) -> dict[str, object]:
    labels = {
        "level": ("higher", "lower", "similar"),
        "trend": ("upward", "downward", "constant"),
        "volatility": ("increased", "decreased", "stable"),
        "seasonality": ("strengthened", "weakened", "stable", "phase_shifted"),
        "regime": ("shift", "no_shift"),
        "extreme": ("increased", "decreased", "stable"),
    }
    difficulties = ("easy", "moderate", "marginal")
    rows = []
    for pindex, (prop, classes) in enumerate(labels.items()):
        for cindex, expected in enumerate(classes):
            for dindex, difficulty in enumerate(difficulties):
                for replicate in range(replicates):
                    case_seed = (seed + pindex * 100_000 + cindex * 1_000
                                 + dindex * 100 + replicate)
                    values, season = _case(
                        case_seed, prop, expected, difficulty=difficulty)
                    evidence = window_evidence(
                        values, property=prop, season=season, window=96)
                    rows.append({
                        "property": prop, "difficulty": difficulty,
                        "expected": expected, "observed": evidence.direction,
                        "support": evidence.support,
                        "identifiable": evidence.identifiable,
                        "correct": evidence.direction == expected,
                        "seed": case_seed,
                    })
    by_property = {}
    for prop, classes in labels.items():
        subset = [row for row in rows if row["property"] == prop]
        recalls = {
            label: statistics.mean(bool(row["correct"]) for row in subset
                                   if row["expected"] == label)
            for label in classes}
        by_property[prop] = {
            "cases": len(subset), "class_recall": recalls,
            "balanced_accuracy": statistics.mean(recalls.values()),
            "identifiable_rate": statistics.mean(
                bool(row["identifiable"]) for row in subset),
            "accuracy_by_difficulty": {
                difficulty: statistics.mean(
                    bool(row["correct"]) for row in subset
                    if row["difficulty"] == difficulty)
                for difficulty in difficulties
            },
            "supported_rate_by_difficulty": {
                difficulty: statistics.mean(
                    row["support"] == "supported" for row in subset
                    if row["difficulty"] == difficulty)
                for difficulty in difficulties
            },
            "supported_accuracy_by_difficulty": {
                difficulty: (statistics.mean(
                    bool(row["correct"]) for row in subset
                    if row["difficulty"] == difficulty
                    and row["support"] == "supported")
                    if any(row["difficulty"] == difficulty
                           and row["support"] == "supported" for row in subset)
                    else None)
                for difficulty in difficulties
            },
            "supported_claim_precision": (
                statistics.mean(bool(row["correct"]) for row in subset
                                if row["support"] == "supported")
                if any(row["support"] == "supported" for row in subset)
                else None),
        }
    gates = {
        "every_property_easy_accuracy_at_least_75pct": all(
            item["accuracy_by_difficulty"]["easy"] >= .75
            for item in by_property.values()),
        "overall_moderate_accuracy_at_least_65pct": statistics.mean(
            item["accuracy_by_difficulty"]["moderate"]
            for item in by_property.values()) >= .65,
        "supported_claim_precision_at_least_90pct": statistics.mean(
            bool(row["correct"]) for row in rows
            if row["support"] == "supported") >= .90,
        "every_property_supported_precision_at_least_90pct": all(
            item["supported_claim_precision"] is None
            or item["supported_claim_precision"] >= .90
            for item in by_property.values()),
        # Marginal cases intentionally straddle decision thresholds. They are
        # reported as a response curve, not used as a target to tune against.
        "all_cases_present": len(rows) == (
            sum(len(items) for items in labels.values())
            * replicates * len(difficulties)),
    }
    return {"schema_version": "0.1", "seed": seed, "cases": len(rows),
            "by_property": by_property, "gates": gates,
            "graduated": all(gates.values()), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=73100)
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    result = run(args.seed, args.replicates)
    if args.output_dir:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"},
                     indent=2, sort_keys=True))
    return 0 if result["graduated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

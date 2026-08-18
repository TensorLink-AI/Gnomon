from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path


def generate(output: Path, *, seed: int, cases: int) -> dict:
    rng = random.Random(seed)
    output.mkdir(parents=True, exist_ok=True)
    event_means = {"deploy": 5.0, "promotion": 8.0,
                   "maintenance": -4.0, "migration": 6.0, "null": 0.0}
    rows, oracle = [], []
    train_count = int(cases * 0.75)
    for index in range(cases):
        split = "train" if index < train_count else "test"
        # migration is held out entirely; null is a false-influence control.
        if split == "train":
            event_type = rng.choice(["deploy", "promotion", "maintenance"])
        else:
            # Every test corpus contains the held-out and false-influence
            # strata; random omission must never turn a missing denominator
            # into a passing gate.
            event_type = list(event_means)[
                (index - train_count) % len(event_means)]
        series = rng.choice(["api", "worker", "database"])
        series_offset = {"api": 0.5, "worker": -0.5, "database": 0.0}[series]
        noise = rng.gauss(0.0, 1.0)
        effect = event_means[event_type] + (series_offset if event_type != "null" else 0) + noise
        case_id = f"effect-{index:04d}"
        public = {
            "case_id": case_id, "split": split, "event_type": event_type,
            "series": series,
            "known_at": f"2026-08-{1 + index % 20:02d}T00:00:00+00:00",
            "observed_effect": effect if split == "train" else None,
            "heldout_event_type": event_type == "migration",
            "false_influence_control": event_type == "null",
        }
        rows.append(public)
        oracle.append({"case_id": case_id, "effect": effect})
    cases_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    oracle_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in oracle)
    (output / "cases.jsonl").write_text(cases_text, encoding="utf-8")
    (output / "oracle.jsonl").write_text(oracle_text, encoding="utf-8")
    manifest = {
        "benchmark": "effectbench", "schema_version": 1, "seed": seed,
        "cases": cases, "train_cases": train_count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_sha256": hashlib.sha256(cases_text.encode()).hexdigest(),
        "oracle_sha256": hashlib.sha256(oracle_text.encode()).hexdigest(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=8127)
    parser.add_argument("--cases", type=int, default=80)
    args = parser.parse_args()
    print(json.dumps(generate(Path(args.output_dir), seed=args.seed,
                              cases=args.cases), indent=2))


if __name__ == "__main__":
    main()

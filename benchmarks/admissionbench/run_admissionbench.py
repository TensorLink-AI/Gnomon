"""Independent simulation of evidence-weighted model admission.

The runtime sees only noisy external/local loss evidence.  Fresh held-out
losses are generated afterwards and score regret; no task, channel, dataset,
or model-name labels enter the policy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gnomon.admission import (  # noqa: E402
    ExternalModelPrior,
    decide_admission,
    local_evidence,
)


def run(seed: int = 20260820, cases: int = 1000) -> dict[str, object]:
    rng = random.Random(seed)
    rows = []
    for index in range(cases):
        # True transfer gains span harmful through excellent; evidence is
        # imperfect and local fold count is independently varied.
        true_gain = rng.uniform(-.25, .30)
        external_n = rng.choice([30, 60, 120, 300])
        external_se = .22 / math.sqrt(external_n)
        measured_external = rng.gauss(true_gain, external_se)
        prior = ExternalModelPrior(
            model="candidate", revision="candidate@held-out",
            regime=(("synthetic_property", "hidden"),),
            comparisons=external_n, mean_relative_gain=measured_external,
            standard_error=external_se, source_ids=(f"cohort:{index}",),
            registry_version=f"seed-{seed}", overlap_risk="low",
        )
        folds = rng.choice([0, 0, 1, 2, 3, 5])
        baseline_losses = [max(.05, rng.gauss(1, .12)) for _ in range(folds)]
        candidate_losses = [
            max(.001, base * (1 - rng.gauss(true_gain, .18)))
            for base in baseline_losses]
        evidence = local_evidence(
            model_class="pretrained", candidate_losses=candidate_losses,
            baseline_losses=baseline_losses, external_prior=prior)
        decision = decide_admission(
            candidate="candidate", baseline="baseline", evidence=evidence)
        held_out_baseline = max(.05, rng.gauss(1, .12))
        held_out_candidate = max(
            .001, held_out_baseline * (1 - rng.gauss(true_gain, .18)))
        blended = (decision.candidate_weight * held_out_candidate
                   + (1 - decision.candidate_weight) * held_out_baseline)
        oracle = min(held_out_candidate, held_out_baseline)
        rows.append({
            "true_gain": true_gain, "folds": folds,
            "state": decision.state, "weight": decision.candidate_weight,
            "candidate_won": held_out_candidate < held_out_baseline,
            "baseline_loss": held_out_baseline,
            "candidate_loss": held_out_candidate, "published_loss": blended,
            "regret": blended - oracle,
        })
    baseline_mean = sum(row["baseline_loss"] for row in rows) / cases
    candidate_mean = sum(row["candidate_loss"] for row in rows) / cases
    published_mean = sum(row["published_loss"] for row in rows) / cases
    # Reliability curve: higher weights should correspond to greater realised
    # win rates. Empty bins are omitted rather than imputed.
    reliability = []
    for low in (0, .2, .4, .6, .8):
        bucket = [row for row in rows if low <= row["weight"] < low + .2
                  or (low == .8 and row["weight"] == 1)]
        if bucket:
            reliability.append({
                "weight_band": [low, low + .2], "cases": len(bucket),
                "mean_weight": sum(row["weight"] for row in bucket) / len(bucket),
                "realised_win_rate": sum(row["candidate_won"] for row in bucket)
                                     / len(bucket),
            })
    harmful = [row for row in rows if row["true_gain"] < 0]
    helpful = [row for row in rows if row["true_gain"] > .05]
    summary = {
        "schema_version": "0.1", "seed": seed, "cases": cases,
        "mean_loss": {"always_baseline": baseline_mean,
                      "always_candidate": candidate_mean,
                      "evidence_weighted": published_mean},
        "mean_regret": sum(row["regret"] for row in rows) / cases,
        "harmful_candidate_mean_weight":
            sum(row["weight"] for row in harmful) / len(harmful),
        "helpful_candidate_mean_weight":
            sum(row["weight"] for row in helpful) / len(helpful),
        "reliability": reliability,
        "states": {state: sum(row["state"] == state for row in rows)
                   for state in sorted({row["state"] for row in rows})},
    }
    summary["gates"] = {
        "beats_always_baseline": published_mean < baseline_mean,
        "beats_always_candidate": published_mean < candidate_mean,
        "helpful_weight_exceeds_harmful":
            summary["helpful_candidate_mean_weight"]
            > summary["harmful_candidate_mean_weight"],
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    summary = run(arguments.seed, arguments.cases)
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    output = arguments.output
    if arguments.output_dir:
        output = arguments.output_dir / "summary.json"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if all(summary["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

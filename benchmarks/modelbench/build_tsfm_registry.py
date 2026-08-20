"""Build an uncontaminated short-history TSFM transfer registry.

The generated series are independent of TemporalBench and contain no channel,
task, answer, or benchmark labels used by the runtime.  Toto is compared with
the hindsight-best mandatory robust baseline, a deliberately conservative
reference.  The resulting registry may be used only for the declared broad
short-history regime and exact pinned revision.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from gnomon.admission import ExternalModelPrior
from gnomon.evaluation import error_score
from gnomon.model_evidence import ModelEvidenceRegistry
from gnomon.models import BASELINES, predict
from gnomon.tsfm_sandbox import SubprocessAdapter


def _case(seed: int, family: str, n: int = 50,
          horizon: int = 29) -> tuple[list[float], list[float], int]:
    rng = random.Random(seed)
    season = 24 if family in {"seasonal", "seasonal_trend"} else 1
    values = []
    level = rng.uniform(20, 200)
    for index in range(n + horizon):
        if family == "random_walk":
            level += rng.gauss(0, 1.2)
            value = level
        else:
            slope = rng.choice((-.18, -.08, .08, .18)) if "trend" in family else 0
            seasonal = (rng.uniform(2, 8) * math.sin(2 * math.pi * index / 24)
                        if "seasonal" in family else 0)
            noise_scale = (2.5 if family == "heteroscedastic" and index >= n // 2
                           else .8)
            value = level + slope * index + seasonal + rng.gauss(0, noise_scale)
            if family == "pulse" and index in {n // 2, n // 2 + 1}:
                value += 8
        values.append(value)
    return values[:n], values[n:], season


def build(*, cases: int, seed: int, output_dir: Path,
          model: str = "toto2_4m") -> dict[str, object]:
    families = ("level", "trend", "seasonal", "seasonal_trend",
                "random_walk", "heteroscedastic", "pulse")
    generated = [_case(seed + index, families[index % len(families)])
                 for index in range(cases)]
    adapter = SubprocessAdapter(model, frequency="h")
    predictions = adapter.predict_many(
        [history for history, _, _ in generated], 29, 1)
    observations = []
    gains = []
    for index, ((history, actual, season), candidate) in enumerate(
            zip(generated, predictions)):
        baseline_losses = {}
        for baseline in BASELINES:
            baseline_prediction = predict(
                baseline, history, len(actual), season)
            baseline_losses[baseline] = error_score(actual, baseline_prediction)
        valid = {name: loss for name, loss in baseline_losses.items()
                 if loss is not None}
        strongest = min(valid, key=valid.get)
        baseline_loss = float(valid[strongest])
        candidate_loss = error_score(actual, list(candidate))
        if candidate_loss is None or baseline_loss <= 1e-12:
            continue
        gain = (baseline_loss - candidate_loss) / baseline_loss
        gains.append(gain)
        observations.append({
            "case_id": f"transfer-{seed + index}",
            "family": families[index % len(families)],
            "candidate_loss": candidate_loss,
            "strongest_robust_baseline": strongest,
            "baseline_loss": baseline_loss,
            "relative_gain": gain,
        })
    if len(gains) < 30:
        raise RuntimeError("fewer than 30 independently scored transfer cases")
    standard_error = max(statistics.stdev(gains) / math.sqrt(len(gains)), 1e-6)
    revision = f"{model}@{adapter.revision}"
    regime = {
        "frequency_class": "subdaily",
        "history_horizon_ratio": "1_to_2",
        "seasonality_strength": "*",
        "trend_strength": "*",
        "intermittency": "low",
        "scale_stability": "*",
    }
    prior = ExternalModelPrior(
        model=model, revision=revision, regime=tuple(sorted(regime.items())),
        comparisons=len(gains), mean_relative_gain=statistics.mean(gains),
        standard_error=standard_error,
        source_ids=tuple(row["case_id"] for row in observations),
        registry_version=f"short-transfer-{seed}-v1", overlap_risk="low",
        baseline_reference="strongest_robust_baseline",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "model-evidence.json"
    ModelEvidenceRegistry(prior.registry_version, (prior,)).dump(registry_path)
    (output_dir / "observations.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in observations),
        encoding="utf-8")
    summary = {
        "schema_version": "0.1", "model": model, "revision": revision,
        "cases": len(gains), "mean_relative_gain": statistics.mean(gains),
        "standard_error": standard_error,
        "candidate_win_rate": statistics.mean(gain > 0 for gain in gains),
        "families": sorted(set(row["family"] for row in observations)),
        "baseline_reference": "hindsight-best mandatory robust baseline",
        "temporalbench_cases_used": 0,
        "registry_path": str(registry_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=84)
    parser.add_argument("--seed", type=int, default=73421)
    parser.add_argument("--model", default="toto2_4m")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build(cases=args.cases, seed=args.seed,
                    output_dir=args.output_dir, model=args.model)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

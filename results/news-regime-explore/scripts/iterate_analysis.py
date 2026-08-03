"""Second-pass analyses (labeled post-hoc; run after RESULTS.md v1).

1. Exact break-even directional hit rate for k-sigma tilts: per task,
   the tilt applied in the *wrong* direction is computable exactly, so
   expected MSE at hit rate p is p*right + (1-p)*wrong per task — no
   linear-discount approximation.
2. Guardrail simulation: replicate the 30/7 fold split and select only
   among Gnomon's mandated BASELINES on the single selection fold
   (the H-G1 fallback), forecasting from the full history.
3. Interval-widening rule search: rescale the stored q10/q90 rows
   around q50 by candidate lead-time schedules and measure coverage
   against the H-G3 targets.
4. Bootstrap CIs (task-level, 10k resamples, seed fixed) for the
   headline means.

Usage: python iterate_analysis.py <tasks_dir> <raw_dir> <out_json>
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/Gnomon")

from gnomon.evaluation import error_score  # noqa: E402
from gnomon.models import BASELINES, MODELS  # noqa: E402

from run_experiments import metric_block  # noqa: E402

SEASON = 7
HORIZON = 7
HIT_RATES = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.0)
TILT_KS = (0.5, 1.0)
BOOT = 10_000


def load(tasks_dir: str, raw_dir: str) -> list[dict]:
    rows = []
    for task_file in sorted(Path(tasks_dir).glob("*.json")):
        task = json.loads(task_file.read_text())
        raw = json.loads((Path(raw_dir) / task_file.name).read_text())
        rows.append({**task, **raw})
    return rows


def mean(xs):
    return sum(xs) / len(xs)


def boot_ci(values: list[float], rng: random.Random) -> list[float]:
    means = sorted(
        mean([values[rng.randrange(len(values))] for _ in values])
        for _ in range(BOOT)
    )
    return [round(means[int(0.025 * BOOT)], 4), round(means[int(0.975 * BOOT)], 4)]


def main(tasks_dir: str, raw_dir: str, out_json: str) -> None:
    rows = load(tasks_dir, raw_dir)
    rng = random.Random(20260803)
    out: dict = {}

    # --- 1. Exact tilt break-even ---
    tilt: dict = {}
    floor_mse = mean([r["floor"]["mse"] for r in rows])
    for k in TILT_KS:
        right, wrong = [], []
        for r in rows:
            truth = r["truth"]
            last = r["last_value"]
            s = k * r["sigma_diff"]
            d = 1.0 if (sum(truth) / len(truth)) >= last else -1.0
            right.append(metric_block(truth, [last + d * s] * HORIZON)["mse"])
            wrong.append(metric_block(truth, [last - d * s] * HORIZON)["mse"])
        by_p = {}
        for p in HIT_RATES:
            expected = mean([p * a + (1 - p) * b for a, b in zip(right, wrong)])
            by_p[str(p)] = {
                "mean_mse": round(expected, 3),
                "reduction_vs_floor": round((floor_mse - expected) / floor_mse, 4),
            }
        # break-even p solves p*mean(right) + (1-p)*mean(wrong) = floor
        mr, mw = mean(right), mean(wrong)
        breakeven = (mw - floor_mse) / (mw - mr) if mw != mr else None
        tilt[str(k)] = {
            "mean_mse_right": round(mr, 3),
            "mean_mse_wrong": round(mw, 3),
            "breakeven_hit_rate": round(breakeven, 3) if breakeven else None,
            "expected_by_hit_rate": by_p,
        }
    out["tilt_breakeven"] = {"floor_mean_mse": round(floor_mse, 3), **tilt}

    # --- 2. Guardrail simulation (baselines-only selection at 1 fold) ---
    guarded_entries, picks = [], {}
    minimum_train = max(2 * SEASON, 2 * HORIZON, 8)  # = 14
    for r in rows:
        values = [float(v) for v in r["input_window"]]
        truth = r["truth"]
        fold_train = values[:minimum_train]
        fold_actual = values[minimum_train : minimum_train + HORIZON]
        scores = {}
        for name in sorted(BASELINES):
            try:
                scores[name] = error_score(
                    fold_actual, MODELS[name](fold_train, HORIZON, SEASON))
            except (ValueError, ArithmeticError):
                continue
        pick = min(scores, key=scores.get) if scores else "last_value"
        picks[pick] = picks.get(pick, 0) + 1
        guarded = MODELS[pick](values, HORIZON, SEASON)
        guarded_entries.append(metric_block(truth, guarded))
    out["guardrail_simulation"] = {
        "mean_mse": round(mean([e["mse"] for e in guarded_entries]), 3),
        "mean_mape": round(mean([e["mape"] for e in guarded_entries]), 3),
        "picks": picks,
        "wins_vs_floor_mse": sum(
            1 for e, r in zip(guarded_entries, rows)
            if e["mse"] < r["floor"]["mse"]),
        "losses_vs_floor_mse": sum(
            1 for e, r in zip(guarded_entries, rows)
            if e["mse"] > r["floor"]["mse"]),
        "h_g1_targets": {"mse<=3.0": None, "mape<=1.8": None},
    }
    out["guardrail_simulation"]["h_g1_targets"]["mse<=3.0"] = (
        out["guardrail_simulation"]["mean_mse"] <= 3.0)
    out["guardrail_simulation"]["h_g1_targets"]["mape<=1.8"] = (
        out["guardrail_simulation"]["mean_mape"] <= 1.8)

    # --- 3. Widening rule search ---
    schedules = {
        "sqrt_anchored_mean": [(step / 4.0) ** 0.5 for step in range(1, 8)],
        "sqrt_anchored_1": [step ** 0.5 for step in range(1, 8)],
        "linear_anchored_mean": [step / 4.0 for step in range(1, 8)],
    }
    widen: dict = {}
    scored = [r for r in rows if not r["gnomon"].get("abstained")]
    for name, scale in schedules.items():
        per_step = []
        for step in range(7):
            hits = []
            for r in scored:
                row = r["gnomon"]["rows"][step]
                truth = r["truth"][step]
                low = row["q50"] + (row["q10"] - row["q50"]) * scale[step]
                high = row["q50"] + (row["q90"] - row["q50"]) * scale[step]
                hits.append(1.0 if low <= truth <= high else 0.0)
            per_step.append(round(mean(hits), 3))
        pooled = round(mean(per_step), 4)
        widen[name] = {
            "pooled": pooled,
            "per_step": per_step,
            "meets_h_g3": (pooled >= 0.74 and per_step[6] >= 0.60
                           and per_step[0] <= 0.88),
        }
    out["widening_rules"] = widen

    # --- 4. Bootstrap CIs ---
    gn = [r for r in rows if not r["gnomon"].get("abstained")]
    coverage_by_task = [mean(r["gnomon"]["covered"]) for r in gn]
    oracle_red = [
        (r["floor"]["mse"] - r["tilts"]["1.0"]["mse"]) for r in rows]
    out["bootstrap_95ci"] = {
        "floor_mean_mse": boot_ci([r["floor"]["mse"] for r in rows], rng),
        "floor_mean_mape": boot_ci([r["floor"]["mape"] for r in rows], rng),
        "gnomon_mean_mse": boot_ci([r["gnomon"]["mse"] for r in gn], rng),
        "gnomon_mean_mape": boot_ci([r["gnomon"]["mape"] for r in gn], rng),
        "pooled_coverage": boot_ci(coverage_by_task, rng),
        "oracle_k1_mse_reduction_abs": boot_ci(oracle_red, rng),
        "guardrail_mean_mse": boot_ci([e["mse"] for e in guarded_entries], rng),
    }

    Path(out_json).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])

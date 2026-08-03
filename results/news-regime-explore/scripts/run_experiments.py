"""Run experiments E1/E3/E4 (and the testable half of E2) from
HYPOTHESIS.md on the surrogate task set.

Per task: forecast with gnomon pure mode (no LLM, no events) exactly as
`benchmarks/mtbench/gnomon_forecaster.py` does — same synthetic daily
axis, same q50-as-point convention, same official metric block — but
keeping the full quantile rows so interval coverage is measurable.

Usage: python run_experiments.py <tasks_dir> <raw_out_dir> <summary_json>
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/user/Gnomon")

from benchmarks.mtbench.gnomon_forecaster import (  # noqa: E402
    OFFICIAL_MSE_FAILURE_LIMIT,
    official_mape,
    write_bar_csv,
)

TILT_KS = (0.5, 1.0, 2.0)


def metric_block(truth: list[float], pred: list[float]) -> dict:
    mse = sum((t - p) ** 2 for t, p in zip(truth, pred)) / len(truth)
    mae = sum(abs(t - p) for t, p in zip(truth, pred)) / len(truth)
    return {"mse": mse, "mae": mae, "rmse": mse ** 0.5,
            "mape": official_mape(truth, pred),
            "failed": mse > OFFICIAL_MSE_FAILURE_LIMIT}


def run_task(task: dict, work_dir: str) -> dict:
    from gnomon import forecast as gnomon_forecast
    from gnomon.contracts import GnomonError

    values = [float(v) for v in task["input_window"]]
    truth = [float(v) for v in task["output_window"]]
    horizon = len(truth)
    run_dir = Path(tempfile.mkdtemp(prefix="nre-", dir=work_dir))
    csv_path = run_dir / "history.csv"
    write_bar_csv(values, csv_path)

    out: dict = {"truth": truth, "last_value": values[-1],
                 "sigma_diff": statistics.pstdev(
                     [b - a for a, b in zip(values, values[1:])])}

    # E1 floor
    floor = [values[-1]] * horizon
    out["floor"] = metric_block(truth, floor)

    # E3 oracle tilts (direction from actuals, magnitude k*sigma only)
    direction = 1.0 if (sum(truth) / horizon) >= values[-1] else -1.0
    out["tilts"] = {}
    for k in TILT_KS:
        tilted = [values[-1] + direction * k * out["sigma_diff"]] * horizon
        out["tilts"][str(k)] = metric_block(truth, tilted)

    # E1 gnomon-pure + E4 quantile rows
    try:
        artifact, _ = gnomon_forecast(
            str(csv_path), time_column="timestamp", target_column="value",
            horizon=horizon, frequency="D",
            output=str(run_dir / "gnomon-output"),
        )
    except GnomonError as error:
        out["gnomon"] = {"abstained": True,
                         "reason": f"{error.code}: {error.message}"}
        return out
    result = artifact.results[0]
    if result.support == "unsupported" or not result.forecast:
        out["gnomon"] = {"abstained": True,
                         "reason": (result.warnings or ["unsupported"])[0]}
        return out
    rows = [{key: float(row[key]) for key in ("q10", "q50", "q90")}
            for row in result.forecast]
    prediction = [row["q50"] for row in rows]
    out["gnomon"] = {
        "abstained": False,
        "support": result.support,
        "status": result.support_assessment.get("status")
        if isinstance(result.support_assessment, dict)
        else getattr(result.support_assessment, "status", None),
        "selected_model": result.selected_model,
        "rows": rows,
        **metric_block(truth, prediction),
        "covered": [1.0 if row["q10"] <= t <= row["q90"] else 0.0
                    for row, t in zip(rows, truth)],
    }
    return out


def filtered_mean(entries: list[dict], key: str) -> float | None:
    values = [e[key] for e in entries if not e["failed"]]
    return sum(values) / len(values) if values else None


def main(tasks_dir: str, raw_dir: str, summary_path: str) -> None:
    from gnomon.tsfm import eligible_tsfms

    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    work = raw / "work"
    work.mkdir(exist_ok=True)

    results = []
    for task_file in sorted(Path(tasks_dir).glob("*.json")):
        task = json.loads(task_file.read_text())
        entry = {"task": task_file.name, "ticker": task.get("ticker")}
        entry.update(run_task(task, str(work)))
        results.append(entry)
        (raw / task_file.name).write_text(json.dumps(entry, indent=1) + "\n")
        print(task_file.name, "ok" if not entry["gnomon"].get("abstained")
              else "ABSTAINED", flush=True)

    scored = [r for r in results if not r["gnomon"].get("abstained")]
    floor_entries = [r["floor"] for r in results]
    gnomon_entries = [r["gnomon"] for r in scored]

    summary: dict = {
        "tasks": len(results),
        "gnomon_abstained": len(results) - len(scored),
        "e2a_eligibility_at_30_7_D": {
            "eligible": eligible_tsfms(history_length=30, horizon=7,
                                       frequency="D")[0],
            "exclusions": eligible_tsfms(history_length=30, horizon=7,
                                         frequency="D")[1],
        },
        "e1_floor": {
            "failed_official_filter": sum(e["failed"] for e in floor_entries),
            "mean_mse": filtered_mean(floor_entries, "mse"),
            "mean_mape": filtered_mean(floor_entries, "mape"),
        },
        "e1_gnomon_pure": {
            "failed_official_filter": sum(e["failed"] for e in gnomon_entries),
            "mean_mse": filtered_mean(gnomon_entries, "mse"),
            "mean_mape": filtered_mean(gnomon_entries, "mape"),
            "selected_models": sorted(
                {e["selected_model"] for e in gnomon_entries}),
            "support_states": sorted({e["support"] for e in gnomon_entries}),
            "wins_vs_floor_mse": sum(
                1 for r in scored if r["gnomon"]["mse"] < r["floor"]["mse"]),
            "losses_vs_floor_mse": sum(
                1 for r in scored if r["gnomon"]["mse"] > r["floor"]["mse"]),
        },
        "e3_oracle_tilts": {},
        "e4_coverage": {},
    }

    for k in TILT_KS:
        tilt_entries = [r["tilts"][str(k)] for r in results]
        mean_mse = filtered_mean(tilt_entries, "mse")
        floor_mse = summary["e1_floor"]["mean_mse"]
        summary["e3_oracle_tilts"][str(k)] = {
            "mean_mse": mean_mse,
            "mean_mape": filtered_mean(tilt_entries, "mape"),
            "mse_reduction_vs_floor": (
                (floor_mse - mean_mse) / floor_mse if floor_mse else None),
            "tasks_improved": sum(
                1 for r in results
                if r["tilts"][str(k)]["mse"] < r["floor"]["mse"]),
            "tasks_worsened": sum(
                1 for r in results
                if r["tilts"][str(k)]["mse"] > r["floor"]["mse"]),
        }

    flat = [c for r in scored for c in r["gnomon"]["covered"]]
    summary["e4_coverage"] = {
        "nominal": 0.80,
        "pooled": sum(flat) / len(flat) if flat else None,
        "n_observations": len(flat),
        "per_step": [
            (lambda step_vals: sum(step_vals) / len(step_vals))(
                [r["gnomon"]["covered"][step] for r in scored])
            for step in range(7)
        ] if scored else [],
    }

    Path(summary_path).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])

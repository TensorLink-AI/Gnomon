"""Post-hoc diagnostic (NOT pre-registered; motivated by E1's falsifier
firing): sweep `minimum_baseline_improvement` to measure how much of
Gnomon's short-history loss to `last_value` is recoverable by demanding a
larger single-fold margin before leaving the baseline.

Usage: python margin_sweep.py <tasks_dir> <out_json>
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/user/Gnomon")

from benchmarks.mtbench.gnomon_forecaster import write_bar_csv  # noqa: E402
from run_experiments import filtered_mean, metric_block  # noqa: E402

MARGINS = (0.02, 0.10, 0.25, 0.50)


def main(tasks_dir: str, out_json: str) -> None:
    from gnomon import forecast as gnomon_forecast

    tasks = [json.loads(p.read_text())
             for p in sorted(Path(tasks_dir).glob("*.json"))]
    work = tempfile.mkdtemp(prefix="nre-margin-")
    sweep: dict = {}
    for margin in MARGINS:
        entries, selected = [], {}
        for task in tasks:
            values = [float(v) for v in task["input_window"]]
            truth = [float(v) for v in task["output_window"]]
            run_dir = Path(tempfile.mkdtemp(dir=work))
            csv_path = run_dir / "history.csv"
            write_bar_csv(values, csv_path)
            artifact, _ = gnomon_forecast(
                str(csv_path), time_column="timestamp",
                target_column="value", horizon=len(truth), frequency="D",
                output=str(run_dir / "out"),
                minimum_baseline_improvement=margin,
            )
            result = artifact.results[0]
            prediction = [float(row["q50"]) for row in result.forecast]
            entries.append(metric_block(truth, prediction))
            selected[result.selected_model] = (
                selected.get(result.selected_model, 0) + 1)
        sweep[str(margin)] = {
            "mean_mse": filtered_mean(entries, "mse"),
            "mean_mape": filtered_mean(entries, "mape"),
            "failed_official_filter": sum(e["failed"] for e in entries),
            "selected_models": dict(sorted(selected.items())),
        }
        print(margin, json.dumps(sweep[str(margin)]), flush=True)
    Path(out_json).write_text(json.dumps(sweep, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

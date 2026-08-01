"""Aion treatment for MTBench's forecasting task families.

Covers the two families whose answer is a numeric trajectory scored by
MSE/MAE/RMSE/MAPE: finance price forecasting (``value_prediction.py``
with ``--indicator time``) and weather temperature forecasting
(``temperature_forecast.py``). The task samples are the official
processed JSONs; the aggregation mirrors the official scripts' metric
block, including their per-sample failure filter, and MAPE is imported
from the official ``evaluation.utils`` when available.

Adapter decisions, disclosed:

- Finance bars skip weekends and holidays, so the calendar axis is
  irregular. Aion models the bar sequence on a synthetic regular daily
  axis (bar *k* at epoch + *k* days) — the standard trading-bar
  convention — and the horizon is the official output length. The
  metric compares values only, so the axis never enters the score.
- Aion forecasts the median path (q50). MTBench scores a point
  trajectory, so no distributional conversion is needed.
- In ``agent`` mode the news text is shown to an OpenRouter model that
  may propose typed context events (never numbers), which Aion's
  admission gate accepts or rejects — the same contract as the CiK
  adapter. ``pure`` mode ignores the text entirely.
- If Aion abstains on a sample, the sample is recorded as an abstention
  and excluded from cumulative metrics — exactly how the official
  scripts treat their own failed samples, but visible in the summary.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cik.aion_forecaster import events_from_proposals  # noqa: E402
from benchmarks.common.openrouter import (  # noqa: E402
    OpenRouterClient,
    extract_json_array,
)
from benchmarks.common.records import RecordWriter, RunRecord  # noqa: E402

EPOCH = datetime(2020, 1, 1, tzinfo=timezone.utc)
DAY = timedelta(days=1)
# Official per-sample failure filter for the plain forecasting tasks
# (value_prediction.py / temperature_forecast.py exclude mse > 100 samples
# from the cumulative averages).
OFFICIAL_MSE_FAILURE_LIMIT = 100.0

EVENT_PROMPT = """\
You assist a deterministic forecasting engine. Read this news text and
propose at most 4 calendar events it describes that could affect the
series being forecast. You must NOT predict numbers. Return ONLY a JSON
array like:
[{{"event_type": "...", "effective_start": "<ISO with tz>",
   "effective_end": "<ISO with tz>", "confidence": 0.0-1.0,
   "rationale": "..."}}]
Timestamps must lie within [{window_start}, {window_end}]. The series
uses a synthetic axis: observation k maps to {epoch} + k days, and the
text may reference calendar dates — map any referenced moment onto the
synthetic axis by its position in the observation window, or return []
if you cannot place it confidently.
"""


def official_mape(y_true: list[float], y_pred: list[float]) -> float:
    """MAPE exactly as ``evaluation.utils.calculate_mape``: imported from
    the official checkout when it is on ``sys.path``, otherwise the same
    nonzero-masked formula."""
    try:
        from evaluation.utils import calculate_mape  # type: ignore

        return float(calculate_mape(y_true, y_pred))
    except Exception:
        pairs = [(t, p) for t, p in zip(y_true, y_pred) if t != 0]
        if not pairs:
            return 0.0
        return 100.0 * sum(abs((t - p) / t) for t, p in pairs) / len(pairs)


def load_samples(dataset_folder: Path) -> list[dict[str, Any]]:
    """Load the official processed task JSONs from one dataset folder."""
    samples = []
    for json_file in sorted(dataset_folder.glob("*.json")):
        with json_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        text = data.get("text")
        if isinstance(text, dict):
            text = text.get("content")
        samples.append({
            "filename": json_file.name,
            "input_window": data.get("input_window"),
            "output_window": data.get("output_window"),
            "input_timestamps": data.get("input_timestamps"),
            "text": text,
        })
    if not samples:
        raise FileNotFoundError(f"No task JSONs found in {dataset_folder}")
    return samples


def write_bar_csv(values: list[float], csv_path: Path) -> tuple[str, str]:
    """Write the input window on the synthetic regular daily axis."""
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "value"])
        for k, value in enumerate(values):
            writer.writerow([(EPOCH + k * DAY).isoformat(), repr(float(value))])
    return EPOCH.isoformat(), (EPOCH + (len(values) - 1) * DAY).isoformat()


def propose_events(client: OpenRouterClient, text: str, sample_name: str,
                   n_obs: int, horizon: int) -> tuple[list[Any], list[str]]:
    window_start = EPOCH.isoformat()
    window_end = (EPOCH + (n_obs + horizon - 1) * DAY).isoformat()
    prompt = EVENT_PROMPT.format(
        window_start=window_start, window_end=window_end,
        epoch=EPOCH.date().isoformat(),
    ) + f"\nNews text:\n{text[:6000]}\n"
    try:
        completion = client.completions(
            [{"role": "user", "content": prompt}], n=1
        )[0]
        proposals = extract_json_array(completion)
    except ValueError:
        return [], ["no JSON array in event-proposal output"]
    except Exception as error:
        return [], [f"event proposal failed: {error}"]
    return events_from_proposals(
        proposals, task_name=f"mtbench-{sample_name}",
        known_at=window_start, window_start=window_start,
        window_end=window_end,
    )


def forecast_sample(sample: dict[str, Any], *, mode: str,
                    client: OpenRouterClient | None,
                    work_dir: str | None) -> dict[str, Any]:
    """Run Aion on one sample. Returns prediction or abstention info."""
    from aion import forecast as aion_forecast
    from aion.contracts import AionError

    values = [float(v) for v in sample["input_window"]]
    horizon = len(sample["output_window"])
    run_dir = Path(tempfile.mkdtemp(prefix="mtbench-aion-", dir=work_dir))
    csv_path = run_dir / "history.csv"
    write_bar_csv(values, csv_path)

    events: list[Any] = []
    notes: list[str] = []
    if mode == "agent" and sample.get("text"):
        events, notes = propose_events(
            client, sample["text"], sample["filename"], len(values), horizon
        )

    try:
        artifact, _ = aion_forecast(
            str(csv_path), time_column="timestamp", target_column="value",
            horizon=horizon, frequency="D",
            output=str(run_dir / "aion-output"),
            context_events=events or None,
        )
    except AionError as error:
        return {"abstained": True,
                "reasons": [f"{error.code}: {error.message}"],
                "proposal_notes": notes}
    result = artifact.results[0]
    if result.support == "unsupported" or not result.forecast:
        return {"abstained": True,
                "reasons": [str(w) for w in result.warnings] or ["unsupported"],
                "proposal_notes": notes}
    prediction = [float(row.get("q50", row["point"])) for row in result.forecast]
    return {
        "abstained": False,
        "prediction": prediction,
        "support": result.support,
        "selected_model": result.selected_model,
        "context": result.context,
        "events": [event.event_id for event in events],
        "proposal_notes": notes,
    }


def run(dataset_folder: Path, output_dir: Path, *, mode: str,
        openrouter_model: str | None, mtbench_root: Path | None,
        temperature: float = 1.0, limit: int | None = None,
        work_dir: str | None = None) -> dict[str, Any]:
    if mtbench_root is not None and str(mtbench_root) not in sys.path:
        sys.path.insert(0, str(mtbench_root))
    client = (OpenRouterClient(openrouter_model, temperature=temperature)
              if mode == "agent" else None)
    samples = load_samples(dataset_folder)
    if limit:
        samples = samples[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    details_dir = output_dir / "output_details"
    details_dir.mkdir(exist_ok=True)
    records = RecordWriter(output_dir / "aionbench.jsonl")

    cumulative = {"mse": [], "mae": [], "rmse": [], "mape": []}
    per_sample: list[dict[str, Any]] = []
    abstained = failed = 0
    for sample in samples:
        started = time.time()
        outcome = forecast_sample(sample, mode=mode, client=client,
                                  work_dir=work_dir)
        elapsed = time.time() - started
        truth = [float(v) for v in sample["output_window"]]
        entry: dict[str, Any] = {"filename": sample["filename"]}
        if outcome["abstained"]:
            abstained += 1
            entry.update({"abstained": True, "reasons": outcome["reasons"]})
            records.write(RunRecord(
                task_id=sample["filename"], success=False,
                appropriate_abstention=True, tool_calls=1,
                latency_seconds=round(elapsed, 3),
                extra={"reasons": outcome["reasons"]},
            ))
        else:
            prediction = outcome["prediction"]
            # Official metric block (value_prediction.py), verbatim math.
            mse = sum((t - p) ** 2 for t, p in zip(truth, prediction)) / len(truth)
            mae = sum(abs(t - p) for t, p in zip(truth, prediction)) / len(truth)
            rmse = mse ** 0.5
            mape = official_mape(truth, prediction)
            is_failed = mse > OFFICIAL_MSE_FAILURE_LIMIT
            entry.update({
                "ground_truth": truth, "predict": prediction,
                "mse": mse, "mae": mae, "rmse": rmse, "mape": mape,
                "failed": is_failed, "support": outcome["support"],
                "selected_model": outcome["selected_model"],
                "events": outcome["events"],
            })
            if is_failed:
                failed += 1
            else:
                for key, value in (("mse", mse), ("mae", mae),
                                   ("rmse", rmse), ("mape", mape)):
                    cumulative[key].append(value)
            records.write(RunRecord(
                task_id=sample["filename"], success=not is_failed,
                tool_calls=1, latency_seconds=round(elapsed, 3),
                extra={"mse": mse, "mape": mape,
                       "support": outcome["support"]},
            ))
        per_sample.append(entry)
        (details_dir / sample["filename"]).write_text(
            json.dumps(entry, indent=2) + "\n", encoding="utf-8"
        )

    scored = len(cumulative["mse"])
    summary = {
        "benchmark": "mtbench",
        "dataset_folder": str(dataset_folder),
        "condition": f"aion-{mode}",
        "model": openrouter_model,
        "samples": len(samples),
        "scored": scored,
        "abstained": abstained,
        "failed_official_filter": failed,
        **{f"mean_{key}": (sum(values) / len(values) if values else None)
           for key, values in cumulative.items()},
        "note": (
            "means follow the official aggregation (samples past the "
            "official mse filter); abstained and filtered counts must be "
            "reported next to them"
        ),
    }
    if client is not None:
        summary["llm_usage"] = client.usage_summary
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary

"""Gnomon treatment for MTBench's forecasting task families.

Covers the two families whose answer is a numeric trajectory scored by
MSE/MAE/RMSE/MAPE: finance price forecasting (``value_prediction.py``
with ``--indicator time``) and weather temperature forecasting
(``temperature_forecast.py``). The task samples are the official
processed JSONs; the aggregation mirrors the official scripts' metric
block, including their per-sample failure filter, and MAPE is imported
from the official ``evaluation.utils`` when available.

Adapter decisions, disclosed:

- Finance bars skip weekends and holidays, so the calendar axis is
  irregular. Gnomon models the bar sequence on a synthetic regular daily
  axis (bar *k* at epoch + *k* days) — the standard trading-bar
  convention — and the horizon is the official output length. The
  metric compares values only, so the axis never enters the score.
- Gnomon forecasts the median path (q50). MTBench scores a point
  trajectory, so no distributional conversion is needed.
- In ``agent`` mode the news text is shown to an OpenRouter model that
  may propose typed context events (never numbers), which Gnomon's
  admission gate accepts or rejects — the same contract as the CiK
  adapter. ``pure`` mode ignores the text entirely.
- In ``tools`` mode the model gets the history and the article and
  drives Gnomon through a tool loop, then finishes through one of two
  honest exits: a computed run's ``forecast_ref`` (used verbatim) or its
  own ``values`` — never an edit of a Gnomon number. The route
  (``gnomon`` / ``informed-direct`` / ``direct``) and the count of
  engine abstentions the model saw are recorded per sample and
  aggregated in the summary. See ``tool_agent``.
- ``mcp`` mode is the raw counterpart of ``tools``: the model holds the
  real ``gnomon mcp serve`` tool surface verbatim (file paths, argument
  schemas, typed errors) and drives the engine itself, with the same
  three exits and the same route/abstention bookkeeping. Running both
  modes on the same samples isolates what the real surface's friction
  costs. See ``mcp_agent``.
- If Gnomon abstains on a sample, the sample is recorded as an abstention
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.cik.gnomon_forecaster import events_from_proposals  # noqa: E402
from benchmarks.common.manifest import code_revision, write_manifest  # noqa: E402
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


def _resolve_mape() -> tuple[Any, str]:
    """The MAPE implementation in use, resolved once per process.

    The official ``evaluation.utils.calculate_mape`` wins whenever the
    checkout is on ``sys.path``; otherwise the local mirror of the same
    nonzero-masked formula is used and the summary must say so.
    """
    global _MAPE
    if _MAPE is None:
        try:
            from evaluation.utils import calculate_mape  # type: ignore

            _MAPE = (calculate_mape, "official evaluation.utils.calculate_mape")
        except ImportError:
            _MAPE = (None, "local mirror of calculate_mape "
                           "(official checkout not on sys.path)")
        except Exception as error:  # noqa: BLE001 — a broken checkout
            # must not kill the run; the summary discloses the fallback.
            _MAPE = (None, "local mirror of calculate_mape "
                           f"(official import failed: {error})")
    return _MAPE


_MAPE: tuple[Any, str] | None = None


def official_mape(y_true: list[float], y_pred: list[float]) -> float:
    """MAPE exactly as ``evaluation.utils.calculate_mape``: imported from
    the official checkout when it is on ``sys.path``, otherwise the same
    nonzero-masked formula — including its all-zero-truth behaviour,
    where the mean over an empty mask is NaN, not 0."""
    implementation, _ = _resolve_mape()
    if implementation is not None:
        return float(implementation(y_true, y_pred))
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t != 0]
    if not pairs:
        return float("nan")
    return 100.0 * sum(abs((t - p) / t) for t, p in pairs) / len(pairs)


def sample_metrics(truth: list[float], prediction: list[float]) -> dict[str, float]:
    """The official metric block (``value_prediction.py``), verbatim math.

    A wrong-length prediction is an error, never a silent truncation:
    ``zip`` would quietly drop the tail and understate the error.
    """
    if len(prediction) != len(truth):
        raise ValueError(
            f"prediction has {len(prediction)} steps, horizon is {len(truth)}"
        )
    mse = sum((t - p) ** 2 for t, p in zip(truth, prediction)) / len(truth)
    mae = sum(abs(t - p) for t, p in zip(truth, prediction)) / len(truth)
    if mse != mse or mae != mae:
        # A NaN slipped through float() parsing (JSON NaN); such a sample
        # would otherwise evade every counter: NaN mse is neither > the
        # failure limit nor summable into a mean.
        raise ValueError("prediction or truth contains a non-finite value")
    return {"mse": mse, "mae": mae, "rmse": mse ** 0.5,
            "mape": official_mape(truth, prediction)}


def _article_text(value: Any) -> Any:
    """The article body, however the official export wrapped it."""
    if isinstance(value, str) and value.lstrip().startswith("{"):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, dict):
        return value.get("content")
    return value


def _samples_from_parquet(dataset_folder: Path) -> list[dict[str, Any]]:
    """Rows of the official Hugging Face export.

    ``download_processed_dataset.py`` fetches each dataset as a single
    parquet shard, not per-task JSONs, so this is the layout an official
    checkout actually has.
    """
    shards = sorted(dataset_folder.rglob("*.parquet"))
    if not shards:
        return []
    import pandas as pd

    samples = []
    for shard in shards:
        frame = pd.read_parquet(shard)
        for position, row in enumerate(frame.to_dict("records")):
            samples.append({
                "filename": f"{shard.stem}#{position:04d}",
                "input_window": list(row.get("input_window", [])),
                "output_window": list(row.get("output_window", [])),
                "input_timestamps": list(row.get("input_timestamps", [])),
                "text": _article_text(row.get("text")),
            })
    return samples


def load_samples(dataset_folder: Path) -> list[dict[str, Any]]:
    """Load the official processed tasks from one dataset folder.

    Accepts either layout: per-task JSONs (as produced by the official
    preparation scripts) or the parquet shards the Hugging Face export
    ships.
    """
    samples = []
    for json_file in sorted(dataset_folder.glob("*.json")):
        with json_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        samples.append({
            "filename": json_file.name,
            "input_window": data.get("input_window"),
            "output_window": data.get("output_window"),
            "input_timestamps": data.get("input_timestamps"),
            "text": _article_text(data.get("text")),
        })
    samples = samples or _samples_from_parquet(dataset_folder)
    if not samples:
        raise FileNotFoundError(
            f"No task JSONs or parquet shards found in {dataset_folder}"
        )
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
    """Run Gnomon on one sample. Returns prediction or abstention info."""
    from gnomon import forecast as gnomon_forecast
    from gnomon.contracts import GnomonError

    if mode == "tools":
        from benchmarks.mtbench.tool_agent import run_sample

        return run_sample(sample, client, work_dir=work_dir)

    if mode == "mcp":
        from benchmarks.mtbench.mcp_agent import run_sample_mcp

        return run_sample_mcp(sample, client, work_dir=work_dir)

    values = [float(v) for v in sample["input_window"]]
    horizon = len(sample["output_window"])
    run_dir = Path(tempfile.mkdtemp(prefix="mtbench-gnomon-", dir=work_dir))
    csv_path = run_dir / "history.csv"
    write_bar_csv(values, csv_path)

    events: list[Any] = []
    notes: list[str] = []
    if mode == "agent" and sample.get("text"):
        events, notes = propose_events(
            client, sample["text"], sample["filename"], len(values), horizon
        )

    try:
        artifact, _ = gnomon_forecast(
            str(csv_path), time_column="timestamp", target_column="value",
            horizon=horizon, frequency="D",
            output=str(run_dir / "gnomon-output"),
            context_events=events or None,
            # Measure the current product contract. Historical strict-floor
            # results remain reproducible by their curated code revision;
            # silently pinning an older floor here made CLI, MCP, and direct
            # adapter comparisons answer different questions.
            minimum_support="best_effort",
        )
    except GnomonError as error:
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
        temperature: float = 0.7, limit: int | None = None,
        work_dir: str | None = None) -> dict[str, Any]:
    if mtbench_root is not None and str(mtbench_root) not in sys.path:
        sys.path.insert(0, str(mtbench_root))
    client = (OpenRouterClient(openrouter_model, temperature=temperature)
              if mode in ("agent", "tools", "mcp") else None)
    samples = load_samples(dataset_folder)
    if limit:
        samples = samples[:limit]

    run_revision = code_revision()
    output_dir.mkdir(parents=True, exist_ok=True)
    details_dir = output_dir / "output_details"
    details_dir.mkdir(exist_ok=True)
    records_path = output_dir / "gnomonbench.jsonl"
    # RecordWriter appends; a rerun into the same output dir must replace
    # the previous run's rows (as summary.json is), not accumulate them.
    records_path.unlink(missing_ok=True)
    records = RecordWriter(records_path)

    cumulative = {"mse": [], "mae": [], "rmse": [], "mape": []}
    # The same metrics with the official mse>100 failure filter NOT
    # applied. The filtered means mirror the official aggregation and stay
    # the headline; these sit beside them because the filter deletes
    # catastrophic misses from the mean, and a mean that cannot get worse
    # when the forecasts do is not safe to read alone.
    unfiltered = {"mse": [], "mae": [], "rmse": [], "mape": []}
    per_sample: list[dict[str, Any]] = []
    abstained = failed = errored = mape_undefined = 0
    routes: dict[str, int] = {}
    # `tools` mode: keep the loop auditable — which exit the model took,
    # how many runs it computed, every call it made, and how often the
    # engine abstained under it.
    tool_keys = ("route", "forecast_ref", "artifact_path",
                 "forecasts_computed", "tool_calls", "engine_abstentions",
                 "submit_reasoning", "trace")
    for sample in samples:
        started = time.time()
        outcome = forecast_sample(sample, mode=mode, client=client,
                                  work_dir=work_dir)
        elapsed = time.time() - started
        truth = [float(v) for v in sample["output_window"]]
        entry: dict[str, Any] = {"filename": sample["filename"]}
        if outcome.get("route"):
            routes[outcome["route"]] = routes.get(outcome["route"], 0) + 1
        # tools/mcp outcomes carry their real call count; pure/agent make
        # exactly one engine invocation. A constant 1 for the tool arms
        # made average_tool_calls a constant, not a measurement.
        calls_made = int(outcome.get("tool_calls") or 1)
        if outcome["abstained"]:
            abstained += 1
            entry.update({"abstained": True, "reasons": outcome["reasons"]})
            for key in tool_keys:
                if outcome.get(key) is not None:
                    entry[key] = outcome[key]
            records.write(RunRecord(
                task_id=sample["filename"], success=False,
                appropriate_abstention=True, tool_calls=calls_made,
                latency_seconds=round(elapsed, 3),
                extra={"reasons": outcome["reasons"],
                       **({"engine_abstentions": outcome["engine_abstentions"]}
                          if outcome.get("engine_abstentions") is not None
                          else {})},
            ))
        else:
            prediction = outcome["prediction"]
            try:
                metrics = sample_metrics(truth, prediction)
            except Exception as error:  # noqa: BLE001 — one bad sample
                # (or an official calculate_mape crash on odd input) must
                # cost that sample an error record, not the whole run.
                errored += 1
                entry.update({"error": str(error)})
                for key in tool_keys:
                    if outcome.get(key) is not None:
                        entry[key] = outcome[key]
                records.write(RunRecord(
                    task_id=sample["filename"], success=False,
                    tool_calls=calls_made,
                    latency_seconds=round(elapsed, 3),
                    extra={"error": str(error)},
                ))
                per_sample.append(entry)
                (details_dir / sample["filename"]).write_text(
                    json.dumps(entry, indent=2) + "\n", encoding="utf-8"
                )
                continue
            mse, mae = metrics["mse"], metrics["mae"]
            rmse, mape = metrics["rmse"], metrics["mape"]
            is_failed = mse > OFFICIAL_MSE_FAILURE_LIMIT
            entry.update({
                "ground_truth": truth, "predict": prediction,
                "mse": mse, "mae": mae, "rmse": rmse, "mape": mape,
                "failed": is_failed, "support": outcome["support"],
                "selected_model": outcome["selected_model"],
                "events": outcome["events"],
            })
            for key in tool_keys:
                if outcome.get(key) is not None:
                    entry[key] = outcome[key]
            if is_failed:
                failed += 1
            for key, value in (("mse", mse), ("mae", mae),
                               ("rmse", rmse), ("mape", mape)):
                # Only MAPE can be NaN here (all-zero truth leaves it
                # undefined; sample_metrics rejects non-finite inputs)
                # and a NaN would poison the cumulative mean silently.
                if key == "mape" and value != value:
                    if not is_failed:
                        mape_undefined += 1
                    continue
                unfiltered[key].append(value)
                if not is_failed:
                    cumulative[key].append(value)
            records.write(RunRecord(
                task_id=sample["filename"], success=not is_failed,
                tool_calls=calls_made, latency_seconds=round(elapsed, 3),
                extra={"mse": mse, "mape": mape,
                       "support": outcome["support"],
                       **({"route": outcome["route"]}
                          if outcome.get("route") else {})},
            ))
        per_sample.append(entry)
        (details_dir / sample["filename"]).write_text(
            json.dumps(entry, indent=2) + "\n", encoding="utf-8"
        )

    scored = len(cumulative["mse"])
    summary = {
        "benchmark": "mtbench",
        "dataset_folder": str(dataset_folder),
        "condition": f"gnomon-{mode}",
        "model": openrouter_model,
        "samples": len(samples),
        "scored": scored,
        "abstained": abstained,
        "errored": errored,
        "failed_official_filter": failed,
        "mape_undefined": mape_undefined,
        "mape_implementation": _resolve_mape()[1],
        **{f"mean_{key}": (sum(values) / len(values) if values else None)
           for key, values in cumulative.items()},
        **{f"mean_{key}_unfiltered": (sum(values) / len(values)
                                      if values else None)
           for key, values in unfiltered.items()},
        "note": (
            "means follow the official aggregation (samples past the "
            "official mse filter); the *_unfiltered means include the "
            "filtered samples, because a mean that deletes its worst "
            "cases cannot be read alone; abstained, errored, filtered and "
            "mape_undefined counts must be reported next to them"
            + (". routes counts submissions per exit: gnomon = a Gnomon "
               "trajectory verbatim, informed-direct/direct = the model's "
               "own values (after/without tool use), abstain = an honest "
               "abstention — model-written numbers are never mixed "
               "unlabeled into Gnomon's"
               if mode in ("tools", "mcp") else "")
        ),
    }
    if mode in ("tools", "mcp"):
        summary["routes"] = dict(sorted(routes.items()))
    if client is not None:
        summary["llm_usage"] = client.usage_summary
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    # Provenance beside the results on direct CLI runs too: the dataset
    # folder is the task set, so it is the `target` a comparison must
    # agree on — recorded as the folder's last two path components
    # (e.g. finance/aligned_in30days_out7days), which identify the family
    # without baking in a machine-specific absolute path that would make
    # same-dataset runs from two checkouts refuse to compare. (The
    # control arm runs the official script, which owns its own output
    # tree; its manifest still comes from run_all.)
    write_manifest(
        output_dir,
        benchmark="mtbench",
        condition=f"gnomon-{mode}",
        model=openrouter_model,
        target="/".join(Path(dataset_folder).parts[-2:]),
        command=" ".join(sys.argv),
        limit=limit,
        base_url=client.base_url if client is not None else None,
        code_revision=run_revision,
    )
    return summary

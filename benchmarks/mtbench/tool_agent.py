"""MTBench's ``gnomon-tools`` condition: Gnomon as tools an agent calls.

The ``agent`` condition hands the news text to the model once, takes
whatever typed events come back, and forecasts — the model never sees a
forecast and cannot react to one. That compares a one-shot pipeline
against the control's full agentic freedom, which is not the comparison
the product makes.

Here the model gets everything the control gets — the price history and
the article — plus a tool loop over Gnomon: summary statistics, season
detection, anomaly detection, and forecasting, which it may run
repeatedly with different typed context events drawn from the article.

The contract that keeps this honest: **the model never writes numbers**.
Every ``gnomon_forecast`` result is registered under a reference id, and
the run ends when the model calls ``submit_forecast`` with one of those
ids. The submitted trajectory is the one Gnomon computed, byte for byte.
The article influences the answer through which events the model
proposes and which run it selects — not by editing digits. A model that
submits nothing is recorded as an abstention, never as a guess.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cik.gnomon_forecaster import events_from_proposals  # noqa: E402
from benchmarks.common.openrouter import OpenRouterClient  # noqa: E402

EPOCH = datetime(2020, 1, 1, tzinfo=timezone.utc)
DAY = timedelta(days=1)
MAX_TOOL_ROUNDS = 8
MAX_TEXT_CHARS = 6000

SYSTEM = """\
You are a careful financial time series analyst working under a strict
contract: you never write forecast numbers yourself. Numbers come from
tools that compute them.

You are given a stock's recent price history and the news article
published over that window. Your job is to produce the best {horizon}-step
forecast of the price you can, using the tools:

- Inspect the series (`series_stats`, `detect_season`,
  `gnomon_detect_anomalies`) as needed.
- Call `gnomon_forecast` to compute a forecast. You may pass
  `context_events` — typed, dated claims you extract from the article
  (never numbers you invent). Gnomon tests each event against a
  leakage-safe ablation gate and reports whether it was admitted; an
  event that does not earn its place is rejected and the forecast falls
  back to history alone.
- Call `gnomon_forecast` more than once if you want to compare a
  history-only run against an event-informed one.
- Finish by calling `submit_forecast` with the `forecast_ref` of the run
  you judge best. That run's trajectory becomes your answer.

If every forecast attempt abstains, call `submit_forecast` with
`forecast_ref: "none"` and say why. Do not invent a trajectory.
"""

USER = """\
Price history ({n_obs} daily bars, oldest first):
{history}

News published over this window:
{text}

Produce the best {horizon}-step forecast of the price, then submit it.
"""

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "series_stats",
            "description": "Deterministic summary statistics of the price history: count, mean, std, min, max, coefficient of variation, first/last value.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_season",
            "description": "Gnomon's autocorrelation-based seasonality detection over the price history: dominant period in bars, strength, basis.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gnomon_detect_anomalies",
            "description": "Gnomon's graded anomaly detection over the price history: competing detectors scored on synthetic anomaly injection, the winner's flagged bars, grades disclosed.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gnomon_forecast",
            "description": "Gnomon's backtested forecast of the price history. Optionally accepts typed context events extracted from the article; each faces a leakage-safe ablation gate. Returns a forecast_ref you can submit, the trajectory Gnomon computed, the selected model, support status, and which events were admitted. Abstains when the history cannot carry the forecast.",
            "parameters": {
                "type": "object",
                "properties": {
                    "context_events": {
                        "type": "array",
                        "description": "Typed dated claims from the article. Never numeric predictions.",
                        # These property names must match what
                        # benchmarks.cik.gnomon_forecaster.events_from_proposals
                        # accepts — a mismatch silently rejects every event.
                        "items": {
                            "type": "object",
                            "properties": {
                                "event_type": {"type": "string", "description": "Short category, e.g. earnings_report"},
                                "effective_start": {"type": "string", "description": "ISO timestamp within the window"},
                                "effective_end": {"type": "string", "description": "ISO timestamp within the window"},
                                "confidence": {"type": "number", "description": "0.0-1.0"},
                                "rationale": {"type": "string"},
                            },
                            "required": ["event_type", "effective_start", "effective_end"],
                        },
                    },
                    "note": {"type": "string", "description": "Why this run"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_forecast",
            "description": "Submit the forecast run whose trajectory is your final answer. Its numbers are used verbatim; you cannot edit them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "forecast_ref": {"type": "string", "description": "A forecast_ref from gnomon_forecast, or 'none' to abstain."},
                    "reasoning": {"type": "string"},
                },
                "required": ["forecast_ref"],
            },
        },
    },
]


class ToolBox:
    """Gnomon bound to one MTBench sample, exposed as callable tools."""

    def __init__(self, values: list[float], horizon: int, text: str,
                 sample_name: str, work_dir: str | None = None):
        self.values = values
        self.horizon = horizon
        self.text = text or ""
        self.sample_name = sample_name
        self.work_dir = work_dir
        self.calls = 0
        self.forecasts: dict[str, dict[str, Any]] = {}
        self.submitted: str | None = None
        self.submit_reasoning: str | None = None
        self.notes: list[str] = []

    # -- axis helpers ------------------------------------------------
    def _window(self) -> tuple[str, str]:
        return (EPOCH.isoformat(),
                (EPOCH + (len(self.values) + self.horizon - 1) * DAY).isoformat())

    def _write_csv(self) -> Path:
        run_dir = Path(tempfile.mkdtemp(prefix="mtbench-tools-",
                                        dir=self.work_dir))
        path = run_dir / "history.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "value"])
            for k, value in enumerate(self.values):
                writer.writerow([(EPOCH + k * DAY).isoformat(),
                                 repr(float(value))])
        return path

    # -- tools -------------------------------------------------------
    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        handler = {
            "series_stats": self.series_stats,
            "detect_season": self.detect_season,
            "gnomon_detect_anomalies": self.gnomon_detect_anomalies,
            "gnomon_forecast": self.gnomon_forecast,
            "submit_forecast": self.submit_forecast,
        }.get(name)
        if handler is None:
            return {"error": f"unknown tool {name}"}
        try:
            return handler(**arguments)
        except TypeError as error:
            return {"error": f"bad arguments: {error}"}
        except Exception as error:  # surface failures to the model
            return {"error": str(error)}

    def series_stats(self) -> dict[str, Any]:
        n = len(self.values)
        mean = sum(self.values) / n
        variance = (sum((v - mean) ** 2 for v in self.values) / (n - 1)
                    if n > 1 else 0.0)
        std = variance ** 0.5
        return {
            "count": n, "mean": round(mean, 6), "std": round(std, 6),
            "min": min(self.values), "max": max(self.values),
            "first": self.values[0], "last": self.values[-1],
            "coefficient_of_variation": round(std / mean, 6) if mean else None,
        }

    def detect_season(self) -> dict[str, Any]:
        from gnomon.temporal import detect_season

        season, strength, basis = detect_season(self.values, "D")
        return {"period_bars": season, "strength": round(strength, 4),
                "basis": basis}

    def gnomon_detect_anomalies(self) -> dict[str, Any]:
        from gnomon.anomaly import detect_anomalies
        from gnomon.temporal import detect_season

        season, _, _ = detect_season(self.values, "D")
        timestamps = [(EPOCH + k * DAY).isoformat()
                      for k in range(len(self.values))]
        detection = detect_anomalies(timestamps, self.values, season=season)
        return {
            "detector": detection.get("detector"),
            "selection_basis": detection.get("selection_basis"),
            "support": detection.get("support", {}).get("status"),
            "anomaly_count": len(detection.get("anomalies", [])),
            "anomalies": detection.get("anomalies", [])[:16],
        }

    def gnomon_forecast(self, context_events: list[dict[str, Any]] | None = None,
                      note: str | None = None) -> dict[str, Any]:
        from gnomon import forecast as gnomon_forecast_fn
        from gnomon.contracts import GnomonError

        window_start, window_end = self._window()
        events: list[Any] = []
        proposal_notes: list[str] = []
        if context_events:
            events, proposal_notes = events_from_proposals(
                list(context_events),
                task_name=f"mtbench-{self.sample_name}",
                known_at=window_start,
                window_start=window_start, window_end=window_end,
            )
            self.notes.extend(proposal_notes)

        csv_path = self._write_csv()
        try:
            artifact, _ = gnomon_forecast_fn(
                str(csv_path), time_column="timestamp", target_column="value",
                horizon=self.horizon, frequency="D",
                output=str(csv_path.parent / "gnomon-output"),
                context_events=events or None,
            )
        except GnomonError as error:
            return {"abstained": True, "code": error.code,
                    "message": error.message,
                    "proposal_notes": proposal_notes}
        result = artifact.results[0]
        if result.support == "unsupported" or not result.forecast:
            return {"abstained": True,
                    "warnings": [str(w) for w in result.warnings],
                    "proposal_notes": proposal_notes}

        trajectory = [float(row.get("q50", row["point"]))
                      for row in result.forecast]
        ref = f"fc{len(self.forecasts) + 1}"
        self.forecasts[ref] = {
            "trajectory": trajectory,
            "support": result.support,
            "selected_model": result.selected_model,
            "events": [event.event_id for event in events],
            "context": result.context,
            "note": note,
        }
        return {
            "forecast_ref": ref,
            "abstained": False,
            "support": result.support,
            "selected_model": result.selected_model,
            "strongest_baseline": result.strongest_baseline,
            "warnings": [str(w) for w in result.warnings],
            "events_proposed": len(context_events or []),
            "events_admitted": len(events),
            "context": result.context,
            "proposal_notes": proposal_notes,
            "forecast": [round(v, 4) for v in trajectory],
        }

    def submit_forecast(self, forecast_ref: str,
                        reasoning: str | None = None) -> dict[str, Any]:
        self.submit_reasoning = reasoning
        if forecast_ref == "none":
            self.submitted = "none"
            return {"accepted": True, "abstained": True}
        if forecast_ref not in self.forecasts:
            return {"error": f"unknown forecast_ref {forecast_ref!r}; "
                             f"available: {sorted(self.forecasts)}"}
        self.submitted = forecast_ref
        return {"accepted": True, "forecast_ref": forecast_ref,
                "steps": len(self.forecasts[forecast_ref]["trajectory"])}


def _tool_calls_as_dicts(message: Any) -> list[dict[str, Any]]:
    calls = []
    for call in getattr(message, "tool_calls", None) or []:
        calls.append({
            "id": call.id,
            "type": "function",
            "function": {"name": call.function.name,
                         "arguments": call.function.arguments},
        })
    return calls


def run_sample(sample: dict[str, Any], client: OpenRouterClient,
               work_dir: str | None = None) -> dict[str, Any]:
    """Let the model drive Gnomon over one sample; return its submission."""
    values = [float(v) for v in sample["input_window"]]
    horizon = len(sample["output_window"])
    toolbox = ToolBox(values, horizon, sample.get("text") or "",
                      sample["filename"], work_dir=work_dir)

    history = ", ".join(f"{v:.4f}" for v in values)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM.format(horizon=horizon)},
        {"role": "user", "content": USER.format(
            n_obs=len(values), history=history, horizon=horizon,
            text=(toolbox.text[:MAX_TEXT_CHARS] or "(no article)"),
        )},
    ]

    trace: list[dict[str, Any]] = []
    for _round in range(MAX_TOOL_ROUNDS):
        response = client.chat(messages, n=1, tools=TOOL_SPECS,
                               tool_choice="auto")
        message = response.choices[0].message
        tool_calls = _tool_calls_as_dicts(message)
        if not tool_calls:
            messages.append({"role": "assistant",
                             "content": message.content or ""})
            if toolbox.submitted:
                break
            # Nudge once: an answer in prose is not a submission.
            messages.append({
                "role": "user",
                "content": "Submit your answer by calling submit_forecast "
                           "with a forecast_ref (or 'none' to abstain).",
            })
            continue
        messages.append({"role": "assistant",
                         "content": message.content or None,
                         "tool_calls": tool_calls})
        for call in tool_calls:
            try:
                arguments = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            result = toolbox.call(call["function"]["name"], arguments)
            trace.append({"tool": call["function"]["name"],
                          "arguments": arguments, "result": result})
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(result)[:20_000]})
        if toolbox.submitted:
            break

    if not toolbox.submitted or toolbox.submitted == "none":
        reasons = ["model submitted no forecast"] if not toolbox.submitted \
            else ["model abstained after tool use"]
        return {"abstained": True, "reasons": reasons + toolbox.notes,
                "tool_calls": toolbox.calls, "trace": trace,
                "proposal_notes": toolbox.notes}

    chosen = toolbox.forecasts[toolbox.submitted]
    return {
        "abstained": False,
        "prediction": chosen["trajectory"],
        "support": chosen["support"],
        "selected_model": chosen["selected_model"],
        "context": chosen["context"],
        "events": chosen["events"],
        "forecast_ref": toolbox.submitted,
        "forecasts_computed": len(toolbox.forecasts),
        "submit_reasoning": toolbox.submit_reasoning,
        "tool_calls": toolbox.calls,
        "trace": trace,
        "proposal_notes": toolbox.notes,
    }

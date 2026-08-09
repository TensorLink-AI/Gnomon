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

The contract that keeps this honest, the same one the CiK MCP arm uses
(``docs/design/cik-mcp-tool-arm.md``): **the model never EDITS a Gnomon
number**. ``submit_forecast`` has exactly two productive exits — a
``forecast_ref`` from a ``gnomon_forecast`` run, whose trajectory is
used byte for byte, or the model's own ``values``, one number per
horizon step, labeled as the model's. The route is recorded per run
(``gnomon`` / ``informed-direct`` / ``direct``) so the exits stay
separable in analysis, and every Gnomon abstention the model saw is
counted in ``engine_abstentions`` — a refusal the model answered past
is disclosed, never laundered into an unlabeled guess. A model that
submits nothing, or submits ``forecast_ref: "none"``, is recorded as an
abstention.

An earlier revision had only the ref exit, so every Gnomon abstention
was forced to become a benchmark abstention (scored as maximum
failure). The CiK comparison measured what that costs: on the 96 runs
where the engine abstained, the model's own reasoned forecast (RCRPS
0.111) beat both the engine's ``--best-effort`` fallback (0.190) and
abstaining outright (5.0 imputed). The second exit exists so this arm
can express that route; the route label is what keeps it honest.
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
You are a careful financial time series analyst. You are given a
stock's recent price history and the news article published over that
window; your job is to produce the best {horizon}-step forecast of the
price you can, then submit it.

You have tools from Gnomon, a deterministic forecasting engine, and you
may use them or ignore them — your only obligation is to end by calling
`submit_forecast`:

- Inspect the series (`series_stats`, `detect_season`,
  `gnomon_detect_anomalies`) as needed.
- Call `gnomon_forecast` to compute a backtested forecast. You may pass
  `context_events` — typed, dated claims you extract from the article
  (never numbers you invent). Gnomon tests each event against a
  leakage-safe ablation gate and reports whether it was admitted; an
  event that does not earn its place is rejected and the forecast falls
  back to history alone. You may run it more than once to compare a
  history-only run against an event-informed one. The engine abstains
  when the history cannot carry its evaluation protocol.

Two ways to finish:
- Submit the `forecast_ref` of the run you judge best: that run's
  trajectory becomes your answer verbatim; you cannot edit it.
- Submit your own `values` (exactly {horizon} numbers, one per step):
  your numbers, your responsibility.

An engine abstention is not your abstention: you may still reason from
the history and the article and submit your own `values`. If you judge
that no defensible forecast exists at all, call `submit_forecast` with
`forecast_ref: "none"` and say why.
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
            "description": (
                "End the run with your answer. Exactly one of two exits: "
                "forecast_ref (from a gnomon_forecast result in THIS run; "
                "its trajectory is used verbatim — you cannot edit it) or "
                "values (your own point forecast, exactly one number per "
                "horizon step). Pass forecast_ref 'none' to abstain."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "forecast_ref": {"type": "string", "description": "A forecast_ref from gnomon_forecast, or 'none' to abstain."},
                    "values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Your own forecast: exactly one number per horizon step.",
                    },
                    "reasoning": {"type": "string"},
                },
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
        #: Analysis/forecast calls only (submit_forecast excluded): the
        #: basis of the direct vs informed-direct route split.
        self.analysis_calls = 0
        #: Gnomon abstentions the model saw during this run. Nonzero on a
        #: run that ends route=informed-direct means the model answered
        #: past an engine refusal — disclosed, never hidden.
        self.engine_abstentions = 0
        self.forecasts: dict[str, dict[str, Any]] = {}
        #: A forecast_ref, "own" (model-written values), or "none".
        self.submitted: str | None = None
        self.submitted_values: list[float] | None = None
        self.route: str | None = None
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
        if name != "submit_forecast":
            self.analysis_calls += 1
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
            self.engine_abstentions += 1
            return {"abstained": True, "code": error.code,
                    "message": error.message,
                    "proposal_notes": proposal_notes}
        result = artifact.results[0]
        if result.support == "unsupported" or not result.forecast:
            self.engine_abstentions += 1
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

    def _values_problems(self, values: Any) -> str | None:
        # A rejection must name what was received, or the model retries
        # the same shape until the rounds cap converts it to an
        # abstention (the lesson CiK's quantile rejections learned).
        import math

        if not isinstance(values, list):
            return (f"values must be a list of exactly {self.horizon} "
                    f"numbers (one per horizon step); got "
                    f"{type(values).__name__}")
        if len(values) != self.horizon:
            return (f"values must be a list of exactly {self.horizon} "
                    f"numbers (one per horizon step); got {len(values)}")
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value):
                return (f"values[{index}] must be a finite number; "
                        f"got {value!r:.60}")
        return None

    def submit_forecast(self, forecast_ref: str | None = None,
                        values: Any = None,
                        reasoning: str | None = None) -> dict[str, Any]:
        if self.submitted is not None:
            return {"error": "a forecast was already submitted; the first "
                             "accepted submission stands"}
        if forecast_ref == "none":
            if values is not None:
                return {"error": "forecast_ref 'none' abstains; it cannot "
                                 "be combined with values"}
            self.submit_reasoning = reasoning
            self.submitted = "none"
            return {"accepted": True, "abstained": True}
        if (forecast_ref is None) == (values is None):
            return {"error": "provide exactly one of forecast_ref or values"}
        if forecast_ref is not None:
            if forecast_ref not in self.forecasts:
                return {"error": f"unknown forecast_ref {forecast_ref!r}; "
                                 f"available: {sorted(self.forecasts)}"}
            self.submit_reasoning = reasoning
            self.submitted = forecast_ref
            self.route = "gnomon"
            return {"accepted": True, "route": "gnomon",
                    "forecast_ref": forecast_ref,
                    "steps": len(self.forecasts[forecast_ref]["trajectory"])}
        problems = self._values_problems(values)
        if problems:
            return {"error": problems}
        self.submit_reasoning = reasoning
        self.submitted = "own"
        self.submitted_values = [float(v) for v in values]
        self.route = ("direct" if self.analysis_calls == 0
                      else "informed-direct")
        return {"accepted": True, "route": self.route,
                "steps": len(self.submitted_values)}


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
                           "— a forecast_ref from a gnomon_forecast run, "
                           "your own values (one number per step), or "
                           "forecast_ref 'none' to abstain.",
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

    common = {
        "forecasts_computed": len(toolbox.forecasts),
        "engine_abstentions": toolbox.engine_abstentions,
        "submit_reasoning": toolbox.submit_reasoning,
        "tool_calls": toolbox.calls,
        "trace": trace,
        "proposal_notes": toolbox.notes,
    }
    if not toolbox.submitted or toolbox.submitted == "none":
        reasons = ["model submitted no forecast"] if not toolbox.submitted \
            else ["model abstained after tool use"]
        return {"abstained": True, "reasons": reasons + toolbox.notes,
                **common}
    if toolbox.submitted == "own":
        # The model's own numbers, labeled as such: the route is what
        # separates a reasoned answer past an engine abstention from a
        # Gnomon-computed one in analysis.
        return {
            "abstained": False,
            "prediction": toolbox.submitted_values,
            "route": toolbox.route,
            "support": None,
            "selected_model": None,
            "context": None,
            "events": [],
            **common,
        }
    chosen = toolbox.forecasts[toolbox.submitted]
    return {
        "abstained": False,
        "prediction": chosen["trajectory"],
        "route": "gnomon",
        "support": chosen["support"],
        "selected_model": chosen["selected_model"],
        "context": chosen["context"],
        "events": chosen["events"],
        "forecast_ref": toolbox.submitted,
        **common,
    }

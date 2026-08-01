"""Dialogue replay harness for TimeSage-MT.

Two agent conditions, matched turn-for-turn on the same tasks:

``direct``
    The paper's Direct Answering shape: the LLM sees the visible series
    (the official per-task ``agent_input`` CSV, never rows beyond the
    visibility contract) plus the running conversation, and answers each
    user turn with no tools.

``aion-tools``
    The same model and the same visible series, but with a function-
    calling loop over deterministic tools whose numbers are computed —
    not generated: summary statistics, Aion season detection, Aion
    backtested forecasting, and Aion graded anomaly detection. The
    system prompt requires every quoted number to come from a tool
    result.

Reference agent turns from the task files are never shown to the agent;
only user turns enter the conversation.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.openrouter import OpenRouterClient  # noqa: E402
from benchmarks.timesage_mt.tasks import TimeSageTask, read_visible_series  # noqa: E402

MAX_TOOL_ROUNDS = 6
MAX_CSV_CHARS = 60_000

DIRECT_SYSTEM = """\
You are a careful time series analyst. The user's dataset (the only rows
you are allowed to see) is below in CSV form. Answer each question from
this data, showing the key numbers that support your conclusion.

<visible_data>
{csv_text}
</visible_data>
"""

TOOLS_SYSTEM = """\
You are a careful time series analyst working under a strict contract:
every number you state must come from a tool result in this
conversation — never estimate or recall numbers yourself. Call tools as
needed, then answer clearly, quoting the tool-derived values that
support your conclusion. If a tool abstains or errors, say so honestly.

The user's dataset (the only rows you are allowed to see) is below in
CSV form; tools compute over exactly this data.

<visible_data>
{csv_text}
</visible_data>
"""

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "series_stats",
            "description": "Deterministic summary statistics of one numeric column: count, mean, std, min, max, coefficient of variation, missing count.",
            "parameters": {
                "type": "object",
                "properties": {"column": {"type": "string"}},
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_season",
            "description": "Aion's autocorrelation-based seasonality detection: dominant period (in observations), strength, and basis.",
            "parameters": {
                "type": "object",
                "properties": {"column": {"type": "string"}},
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aion_forecast",
            "description": "Aion's backtested forecast for a numeric column: model selected against mandatory baselines, per-step point and q10/q50/q90, support status. Abstains when history cannot carry the forecast.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "horizon": {"type": "integer", "minimum": 1, "maximum": 128},
                },
                "required": ["column", "horizon"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aion_detect_anomalies",
            "description": "Aion's graded anomaly detection: competing detectors scored on synthetic anomaly injection, winner flags anomalous timestamps, grades disclosed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "threshold": {"type": "number"},
                },
                "required": ["column"],
            },
        },
    },
]


class ToolBox:
    """Deterministic tools bound to one task's visible series."""

    def __init__(self, task: TimeSageTask, work_dir: str | None = None):
        self.timestamps, self.columns, self.csv_text = read_visible_series(task)
        self.work_dir = work_dir
        self.calls = 0

    def bounded_csv(self) -> str:
        text = self.csv_text
        if len(text) > MAX_CSV_CHARS:
            text = text[:MAX_CSV_CHARS] + "\n... (truncated for context)"
        return text

    def _column(self, name: str) -> list[float]:
        if name not in self.columns:
            raise ValueError(
                f"Unknown numeric column {name!r}. Available: "
                f"{sorted(self.columns)}"
            )
        return self.columns[name]

    def _aware_timestamps(self) -> list[str]:
        aware = []
        for value in self.timestamps:
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                return []
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            aware.append(parsed.isoformat())
        return aware

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        try:
            handler = {
                "series_stats": self.series_stats,
                "detect_season": self.detect_season,
                "aion_forecast": self.aion_forecast,
                "aion_detect_anomalies": self.aion_detect_anomalies,
            }[name]
        except KeyError:
            return {"error": f"unknown tool {name}"}
        try:
            return handler(**arguments)
        except TypeError as error:
            return {"error": f"bad arguments: {error}"}
        except Exception as error:  # surface tool failures to the model
            return {"error": str(error)}

    def series_stats(self, column: str) -> dict[str, Any]:
        values = self._column(column)
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
        std = variance ** 0.5
        return {
            "column": column, "count": n,
            "mean": round(mean, 6), "std": round(std, 6),
            "min": min(values), "max": max(values),
            "coefficient_of_variation": round(std / mean, 6) if mean else None,
            "first_timestamp": self.timestamps[0] if self.timestamps else None,
            "last_timestamp": self.timestamps[-1] if self.timestamps else None,
        }

    def _frequency(self) -> str:
        from aion.contracts import AionError
        from aion.temporal import infer_frequency

        aware = self._aware_timestamps()
        try:
            return infer_frequency([datetime.fromisoformat(t) for t in aware[:64]])
        except (AionError, ValueError, IndexError):
            return "h"

    def detect_season(self, column: str) -> dict[str, Any]:
        from aion.temporal import detect_season

        values = self._column(column)
        season, strength, basis = detect_season(values, self._frequency())
        return {"column": column, "period_observations": season,
                "strength": round(strength, 4), "basis": basis}

    def _write_csv(self, column: str) -> Path:
        aware = self._aware_timestamps()
        values = self._column(column)
        if not aware or len(aware) != len(values):
            raise ValueError("visible series has no parseable regular time axis")
        run_dir = Path(tempfile.mkdtemp(prefix="timesage-aion-", dir=self.work_dir))
        path = run_dir / "visible.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "value"])
            for timestamp, value in zip(aware, values):
                writer.writerow([timestamp, repr(float(value))])
        return path

    def aion_forecast(self, column: str, horizon: int) -> dict[str, Any]:
        from aion import forecast as aion_forecast
        from aion.contracts import AionError

        csv_path = self._write_csv(column)
        try:
            artifact, _ = aion_forecast(
                str(csv_path), time_column="timestamp", target_column="value",
                horizon=int(horizon), output=str(csv_path.parent / "aion-output"),
            )
        except AionError as error:
            return {"abstained": True, "code": error.code,
                    "message": error.message}
        result = artifact.results[0]
        if result.support == "unsupported" or not result.forecast:
            return {"abstained": True,
                    "warnings": [str(w) for w in result.warnings]}
        return {
            "abstained": False, "support": result.support,
            "selected_model": result.selected_model,
            "strongest_baseline": result.strongest_baseline,
            "warnings": [str(w) for w in result.warnings],
            "forecast": result.forecast[:64],
        }

    def aion_detect_anomalies(self, column: str,
                              threshold: float | None = None) -> dict[str, Any]:
        from aion.anomaly import DEFAULT_THRESHOLD, detect_anomalies
        from aion.temporal import detect_season

        values = self._column(column)
        season, _, _ = detect_season(values, self._frequency())
        detection = detect_anomalies(
            self.timestamps, values, season=season,
            threshold=DEFAULT_THRESHOLD if threshold is None else float(threshold),
        )
        return {
            "detector": detection.get("detector"),
            "selection_basis": detection.get("selection_basis"),
            "support": detection.get("support", {}).get("status"),
            "anomalies": detection.get("anomalies", [])[:64],
            "anomaly_count": len(detection.get("anomalies", [])),
        }


def _tool_calls_as_dicts(message: Any) -> list[dict[str, Any]]:
    calls = []
    for call in getattr(message, "tool_calls", None) or []:
        calls.append({
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments,
            },
        })
    return calls


def run_dialogue(
    task: TimeSageTask,
    client: OpenRouterClient,
    *,
    condition: str,
    work_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Replay all user turns; return one record per user turn with the
    agent's final response and tool usage."""
    toolbox = ToolBox(task, work_dir=work_dir)
    use_tools = condition == "aion-tools"
    system = (TOOLS_SYSTEM if use_tools else DIRECT_SYSTEM).format(
        csv_text=toolbox.bounded_csv()
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    records: list[dict[str, Any]] = []

    for user_turn in task.user_turns:
        messages.append({"role": "user", "content": user_turn.get("text", "")})
        tool_trace: list[dict[str, Any]] = []
        response_text = ""
        for _round in range(MAX_TOOL_ROUNDS + 1):
            response = client.chat(
                messages,
                n=1,
                tools=TOOL_SPECS if use_tools else None,
                tool_choice="auto" if use_tools else None,
            )
            message = response.choices[0].message
            tool_calls = _tool_calls_as_dicts(message)
            if not tool_calls:
                response_text = message.content or ""
                messages.append({"role": "assistant", "content": response_text})
                break
            messages.append({
                "role": "assistant",
                "content": message.content or None,
                "tool_calls": tool_calls,
            })
            for call in tool_calls:
                try:
                    arguments = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                result = toolbox.call(call["function"]["name"], arguments)
                tool_trace.append({
                    "tool": call["function"]["name"],
                    "arguments": arguments,
                    "result": result,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result)[:20_000],
                })
        else:
            response_text = "(no final answer within the tool-round limit)"
            messages.append({"role": "assistant", "content": response_text})
        records.append({
            "user_turn_id": user_turn.get("turn_id"),
            "question": user_turn.get("text", ""),
            "response": response_text,
            "tool_calls": tool_trace,
        })
    return records

"""The Aion tool surface, as an OpenRouter-callable toolbelt.

Aion already defines its agent-facing tools once, in ``aion.toolspec`` — the
names, the JSON Schemas, and the in-process runners the MCP server uses. This
module translates that surface into OpenAI-style function definitions and
dispatches calls back to the same runners, so the benchmark measures the tool
surface agents actually get rather than a benchmark-only imitation of it.

Two adaptations are made for the benchmark setting, both disclosed to the
model in the prompt:

- Benchmark rows carry bare value arrays. Aion refuses data without a
  temporal axis, so the harness materialises each series as a CSV with a
  regular synthetic index.
- Tool results are truncated before they re-enter the context. The digest
  kept in the trace records what was elided.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import BenchError, SeriesRef, ToolCall

# Tools worth exposing for temporal-reasoning questions. The tracking,
# lifecycle, and planner tools are real but irrelevant here: offering them
# only spends context and invites the model to wander.
ANALYSIS_TOOLS = (
    "aion_capabilities",
    "aion_inspect",
    "aion_forecast",
    "aion_detect_anomalies",
    "aion_investigate_change",
    "aion_route",
    "aion_get_artifact",
)

FORECAST_TOOLS = ("aion_inspect", "aion_forecast", "aion_get_artifact")

DECISION_TOOLS = ANALYSIS_TOOLS + ("aion_decide", "aion_monitor")

TOOL_PROFILES: dict[str, tuple[str, ...]] = {
    "analysis": ANALYSIS_TOOLS,
    "forecast": FORECAST_TOOLS,
    "decision": DECISION_TOOLS,
    "full": (),  # empty tuple means "whatever the runtime exposes"
}

MAX_TOOL_RESULT_CHARS = 6000

_FREQUENCY_STEPS: dict[str, timedelta] = {
    "min": timedelta(minutes=1),
    "5min": timedelta(minutes=5),
    "15min": timedelta(minutes=15),
    "30min": timedelta(minutes=30),
    "h": timedelta(hours=1),
    "D": timedelta(days=1),
    "W": timedelta(weeks=1),
}


def materialise_series(series: Sequence[SeriesRef], directory: str | Path) -> dict[str, str]:
    """Write each series to ``<directory>/<name>.csv``; return name -> path.

    The synthetic index is regular by construction, which is what lets Aion
    accept the file at all. Month-start frequency is not synthesised: the
    step is not constant, and a benchmark has no calendar to anchor it to.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    for item in series:
        step = _FREQUENCY_STEPS.get(item.frequency)
        if step is None:
            raise BenchError(
                "UNSUPPORTED_BENCH_FREQUENCY",
                f"Cannot synthesise a regular index at frequency {item.frequency!r}.",
                {"supported": sorted(_FREQUENCY_STEPS)},
            )
        start = _parse_start(item.start)
        path = target / f"{_slug(item.name)}.csv"
        lines = ["timestamp,value"]
        for index, value in enumerate(item.values):
            stamp = (start + step * index).isoformat()
            lines.append(f"{stamp},{value}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths[item.name] = str(path)

    return paths


def _parse_start(raw: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise BenchError("INVALID_SERIES_START", f"Not an ISO timestamp: {raw!r}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _slug(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
    return cleaned or "series"


class AionToolbelt:
    """Dispatches model tool calls onto Aion's in-process runners."""

    def __init__(self, profile: str = "analysis", workdir: str | Path | None = None):
        self.profile = profile
        self.workdir = Path(workdir) if workdir else None
        self._specs = self._load_specs(profile)

    @staticmethod
    def _load_specs(profile: str) -> list[dict[str, Any]]:
        try:
            from aion.toolspec import visible_tools
        except ImportError as exc:  # pragma: no cover - aion is a hard dependency
            raise BenchError(
                "AION_NOT_IMPORTABLE",
                "The aion package must be importable to run an Aion condition. "
                "Install this checkout (`bash install.sh`) or set PYTHONPATH=src.",
            ) from exc

        if profile not in TOOL_PROFILES:
            raise BenchError(
                "UNKNOWN_TOOL_PROFILE",
                f"Unknown tool profile {profile!r}.",
                {"supported": sorted(TOOL_PROFILES)},
            )
        allowed = TOOL_PROFILES[profile]
        tools = visible_tools()
        if not allowed:
            return list(tools)

        by_name = {tool["name"]: tool for tool in tools}
        missing = [name for name in allowed if name not in by_name]
        if missing:
            raise BenchError(
                "TOOL_PROFILE_UNAVAILABLE",
                "The installed Aion runtime does not expose every tool in this profile.",
                {"profile": profile, "missing": missing, "available": sorted(by_name)},
            )
        return [by_name[name] for name in allowed]

    @property
    def names(self) -> list[str]:
        return [spec["name"] for spec in self._specs]

    def definitions(self) -> list[dict[str, Any]]:
        """The toolbelt as OpenAI-style function definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": _openai_schema(spec["inputSchema"]),
                },
            }
            for spec in self._specs
        ]

    def call(self, name: str, arguments: Mapping[str, Any]) -> ToolCall:
        """Run one tool call, capturing failures as data rather than raising."""
        started = time.monotonic()
        spec = next((item for item in self._specs if item["name"] == name), None)
        if spec is None:
            return ToolCall(
                name=name, arguments=dict(arguments), ok=False,
                latency_seconds=time.monotonic() - started,
                error=f"No such tool: {name}. Available: {', '.join(self.names)}.",
            )

        payload = dict(arguments)
        if self.workdir and "output_dir" in _schema_properties(spec["inputSchema"]):
            payload.setdefault("output_dir", str(self.workdir / "artifacts"))

        try:
            result = spec["runner"](payload)
        except Exception as exc:  # noqa: BLE001 - every tool failure is a datum
            return ToolCall(
                name=name, arguments=payload, ok=False,
                latency_seconds=time.monotonic() - started,
                error=_describe_exception(exc),
            )

        rendered, digest = _render_result(result)
        call = ToolCall(
            name=name, arguments=payload, ok=True,
            latency_seconds=time.monotonic() - started,
            result_digest=digest,
        )
        call.rendered = rendered  # type: ignore[attr-defined]
        return call


def _describe_exception(exc: Exception) -> str:
    """Aion errors carry a machine-readable code and repair options; keep them."""
    to_dict = getattr(exc, "to_dict", None)
    if callable(to_dict):
        try:
            return json.dumps(to_dict())[:MAX_TOOL_RESULT_CHARS]
        except (TypeError, ValueError):
            pass
    return f"{type(exc).__name__}: {exc}"[:MAX_TOOL_RESULT_CHARS]


def _render_result(result: Any) -> tuple[str, str]:
    """Serialise a tool result for the model, and describe any truncation."""
    try:
        text = json.dumps(result, default=str)
    except (TypeError, ValueError):
        text = str(result)
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text, f"{len(text)} chars"
    kept = text[:MAX_TOOL_RESULT_CHARS]
    note = (
        f"{kept}\n\n[truncated by AionBench: {len(text) - MAX_TOOL_RESULT_CHARS} "
        f"further characters. Read the artifact directory with aion_get_artifact "
        f"if you need the rest.]"
    )
    return note, f"{len(text)} chars, truncated to {MAX_TOOL_RESULT_CHARS}"


def _schema_properties(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    return schema.get("properties") or {}


def _openai_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Aion's JSON Schemas are already valid function parameters.

    Copied rather than passed through so a provider that mutates the dict in
    place cannot corrupt the runtime's canonical definition.
    """
    return json.loads(json.dumps(dict(schema)))

"""Typed vocabulary shared by every suite, condition, and report.

Two rules shape these types:

- A wrong answer and a *missing* answer are different results. Every failure
  carries a typed reason (``FAILURE_KINDS``) so "the model got it wrong" is
  never confused with "the harness never got a parseable answer out of it".
- Scoring is separated from grading. ``Attempt`` is what the model produced;
  ``TaskOutcome`` is what a deterministic scorer made of it. No suite may
  grade with a model unless the user explicitly asks for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "0.1"

# What a task asks the model to produce.
TASK_KINDS = ("multiple_choice", "numeric", "series", "code", "free_text")

# How an attempt failed to yield a scoreable answer. Mirrors the TSAIA
# protocol (structural / execution / constraint / sub-threshold) and adds the
# two failure modes a hosted benchmark also has to name.
FAILURE_KINDS = (
    "structural",     # no parseable answer in the response
    "execution",      # generated code raised, timed out, or was refused
    "constraint",     # answer violated a stated constraint or prior
    "threshold",      # answer parsed and ran but missed the metric's bar
    "refusal",        # model declined the task
    "transport",      # provider/network error — not the model's fault
)

# Conditions a suite may offer. `direct` and `code` are the controls; the
# `aion_*` conditions are the treatments.
CONDITIONS = ("direct", "code", "aion", "aion_code")

CONTROL_CONDITIONS = ("direct", "code")
TREATMENT_CONDITIONS = ("aion", "aion_code")


class BenchError(Exception):
    """A typed harness error. Mirrors ``aion.contracts.AionError`` in shape."""

    def __init__(self, code: str, message: str, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


@dataclass(frozen=True)
class SeriesRef:
    """One numeric series attached to a task.

    Benchmark rows carry bare value arrays with no timestamps. Aion refuses to
    work without a temporal axis, so the harness synthesises one — a regular
    index at ``frequency`` starting at ``start``. That index is an artefact of
    the harness, disclosed in the prompt and in the run manifest, never
    presented as data the benchmark supplied.
    """

    name: str
    values: tuple[float, ...]
    frequency: str = "D"
    start: str = "2020-01-01T00:00:00+00:00"

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class BenchTask:
    """One benchmark item, normalised across suites."""

    task_id: str
    suite: str
    kind: str
    prompt: str
    series: tuple[SeriesRef, ...] = ()
    options: tuple[str, ...] = ()
    answer: str | None = None          # canonical answer for choice tasks
    target: Any = None                 # numeric ground truth for scored tasks
    metric: str = "accuracy"
    category: str = "uncategorised"
    subcategory: str = ""
    difficulty: str = ""
    constraint: Any = None
    hint: str = ""
    concepts: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in TASK_KINDS:
            raise BenchError(
                "INVALID_TASK_KIND",
                f"Unknown task kind {self.kind!r}.",
                {"supported": list(TASK_KINDS)},
            )


@dataclass
class ToolCall:
    """One tool invocation inside an agentic condition."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    latency_seconds: float = 0.0
    error: str | None = None
    result_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "latency_seconds": round(self.latency_seconds, 4),
            "error": self.error,
            "result_digest": self.result_digest,
        }


@dataclass
class Attempt:
    """What one model produced for one task under one condition."""

    task_id: str
    model: str
    condition: str
    repeat: int = 0
    text: str = ""
    parsed: Any = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    provider_calls: int = 0
    cache_hits: int = 0
    error: str | None = None
    failure: str | None = None

    # CodeAct episodes reach Aion by importing it in the kernel rather than
    # by calling a tool, so grounding has a second source. Set by the
    # CodeAct runner; left empty by the tool-calling loop.
    codeact: dict[str, Any] = field(default_factory=dict)

    @property
    def cached(self) -> bool:
        """True only when the whole attempt was served from cache — a partly
        cached tool loop still cost money, and must not be reported as free."""
        return self.provider_calls > 0 and self.cache_hits == self.provider_calls

    @property
    def successful_tool_calls(self) -> int:
        return sum(1 for call in self.tool_calls if call.ok)

    @property
    def grounded(self) -> bool:
        """Did Aion actually contribute to this attempt?

        Two routes, because there are two treatment mechanisms: a tool call
        that returned a result, or CodeAct code that imported ``aion`` and
        went on to bind an answer. A treatment run where every tool call
        errored — or where the model never touched Aion — is a treatment run
        in name only, and the reports separate the two.
        """
        if self.successful_tool_calls > 0:
            return True
        return bool(self.codeact.get("used_aion") and self.codeact.get("answer_found"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model": self.model,
            "condition": self.condition,
            "repeat": self.repeat,
            "text": self.text,
            "parsed": self.parsed,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "latency_seconds": self.latency_seconds,
            "provider_calls": self.provider_calls,
            "cached": self.cached,
            "codeact": dict(self.codeact),
            "grounded": self.grounded,
            "error": self.error,
            "failure": self.failure,
        }


@dataclass
class TaskOutcome:
    """A deterministically graded attempt.

    ``correct`` is the headline binary for accuracy-style suites. ``score`` is
    the raw metric value (MAPE, F1, …) where one exists. Both may be present;
    ``correct`` is then the metric measured against its pass bar.
    """

    task_id: str
    suite: str
    model: str
    condition: str
    repeat: int = 0
    kind: str = "multiple_choice"
    metric: str = "accuracy"
    category: str = "uncategorised"
    subcategory: str = ""
    difficulty: str = ""
    correct: bool | None = None
    score: float | None = None
    expected: Any = None
    answered: Any = None
    failure: str | None = None
    detail: str = ""
    tool_calls: int = 0
    tool_calls_ok: int = 0
    grounded: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    cached: bool = False

    def to_row(self) -> dict[str, Any]:
        """The JSONL row written to ``rows.jsonl``.

        Also carries the fields ``aion eval compare`` requires, so a run can
        be handed straight to the existing uplift tooling without a converter.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": self.task_id,
            "suite": self.suite,
            "model": self.model,
            "condition": self.condition,
            "repeat": self.repeat,
            "kind": self.kind,
            "metric": self.metric,
            "category": self.category,
            "subcategory": self.subcategory,
            "difficulty": self.difficulty,
            "success": bool(self.correct),
            "correct": self.correct,
            "score": self.score,
            "expected": _jsonable(self.expected),
            "answered": _jsonable(self.answered),
            "failure": self.failure,
            "detail": self.detail,
            "tool_calls": self.tool_calls,
            "tool_calls_ok": self.tool_calls_ok,
            "grounded": self.grounded,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "latency_seconds": self.latency_seconds,
            "cached": self.cached,
        }


def _jsonable(value: Any) -> Any:
    """Coerce a ground truth or answer into something json.dumps accepts."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence):
        trimmed = list(value)[:64]
        return [_jsonable(item) for item in trimmed]
    return str(value)

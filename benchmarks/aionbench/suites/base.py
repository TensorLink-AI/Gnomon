"""The contract every suite implements, and the registry that finds them.

A suite owns three things and nothing else: how its corpus becomes
``BenchTask``s, which conditions it can be run under, and how an ``Attempt``
becomes a ``TaskOutcome``. Prompting, transport, concurrency, caching, and
reporting are the harness's business, so adding a fourth benchmark means
writing one file, not touching the runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import Attempt, BenchError, BenchTask, TaskOutcome
from ..sandbox import SandboxConfig


@dataclass
class LoadOptions:
    """How much of a corpus to load, and from where."""

    data_path: str | None = None
    revision: str = "main"
    limit: int | None = None
    seed: int = 20260801
    categories: tuple[str, ...] = ()
    config: str | None = None
    allow_pickle: bool = False
    cache: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ScoringContext:
    """Everything a scorer may need beyond the task and the attempt."""

    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    workdir: Path = field(default_factory=Path)
    allow_pickle: bool = False
    thresholds: Mapping[str, float] = field(default_factory=dict)


class Suite:
    """Base class for benchmark suites."""

    name: str = ""
    description: str = ""
    citation: str = ""
    homepage: str = ""
    conditions: tuple[str, ...] = ("direct", "aion")
    default_control: str = "direct"
    default_treatment: str = "aion"

    def load(self, options: LoadOptions) -> list[BenchTask]:
        raise NotImplementedError

    def score(self, task: BenchTask, attempt: Attempt, context: ScoringContext) -> TaskOutcome:
        raise NotImplementedError

    # -- shared helpers ----------------------------------------------------

    def new_outcome(self, task: BenchTask, attempt: Attempt) -> TaskOutcome:
        """A TaskOutcome pre-filled with everything that is not the grade."""
        return TaskOutcome(
            task_id=task.task_id,
            suite=self.name,
            model=attempt.model,
            condition=attempt.condition,
            repeat=attempt.repeat,
            kind=task.kind,
            metric=task.metric,
            category=task.category,
            subcategory=task.subcategory,
            difficulty=task.difficulty,
            tool_calls=len(attempt.tool_calls),
            tool_calls_ok=attempt.successful_tool_calls,
            grounded=attempt.grounded,
            prompt_tokens=attempt.prompt_tokens,
            completion_tokens=attempt.completion_tokens,
            cost_usd=attempt.cost_usd,
            latency_seconds=attempt.latency_seconds,
            cached=attempt.cached,
        )

    def transport_failure(self, task: BenchTask, attempt: Attempt) -> TaskOutcome | None:
        """Short-circuit grading when the provider, not the model, failed.

        Transport failures are recorded with ``correct=None`` so they are
        excluded from accuracy denominators instead of quietly counting as
        wrong answers and defaming the model.
        """
        if attempt.failure != "transport" and not attempt.error:
            return None
        if attempt.failure not in (None, "transport"):
            return None
        outcome = self.new_outcome(task, attempt)
        outcome.correct = None
        outcome.failure = "transport"
        outcome.detail = attempt.error or "Provider call failed."
        return outcome

    def subsample(self, tasks: list[BenchTask], options: LoadOptions) -> list[BenchTask]:
        """Filter by category, then take a deterministic random subset.

        Random rather than head-of-file: benchmark corpora are grouped by
        template, so the first N rows are almost never a fair sample.
        """
        import random

        if options.categories:
            wanted = {item.lower() for item in options.categories}
            tasks = [
                task for task in tasks
                if task.category.lower() in wanted or task.subcategory.lower() in wanted
            ]
            if not tasks:
                raise BenchError(
                    "NO_TASKS_MATCH_FILTER",
                    "No tasks matched the requested categories.",
                    {"categories": list(options.categories)},
                )
        if options.limit is not None and 0 < options.limit < len(tasks):
            rng = random.Random(options.seed)
            tasks = sorted(rng.sample(tasks, options.limit), key=lambda item: item.task_id)
        return tasks


_REGISTRY: dict[str, type[Suite]] = {}


def register(suite_class: type[Suite]) -> type[Suite]:
    if not suite_class.name:
        raise BenchError("SUITE_NAME_MISSING", f"{suite_class.__name__} must set `name`.")
    _REGISTRY[suite_class.name] = suite_class
    return suite_class


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_suite(name: str) -> Suite:
    if name not in _REGISTRY:
        raise BenchError(
            "UNKNOWN_SUITE",
            f"No suite named {name!r}.",
            {"available": available()},
        )
    return _REGISTRY[name]()


def describe_all() -> list[dict[str, Any]]:
    return [
        {
            "name": suite_class.name,
            "description": suite_class.description,
            "citation": suite_class.citation,
            "homepage": suite_class.homepage,
            "conditions": list(suite_class.conditions),
            "default_control": suite_class.default_control,
            "default_treatment": suite_class.default_treatment,
        }
        for _, suite_class in sorted(_REGISTRY.items())
    ]


def normalise_conditions(suite: Suite, requested: Sequence[str] | None) -> tuple[str, ...]:
    if not requested:
        return (suite.default_control, suite.default_treatment)
    unsupported = [item for item in requested if item not in suite.conditions]
    if unsupported:
        raise BenchError(
            "UNSUPPORTED_CONDITION",
            f"Suite {suite.name!r} does not support {', '.join(unsupported)}.",
            {"supported": list(suite.conditions)},
        )
    return tuple(requested)

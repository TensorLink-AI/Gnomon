"""Orchestration: the grid of (task x model x condition x repeat).

Everything here exists to make a long, expensive, interruptible job behave:

- rows stream to disk as they complete, so a killed run keeps its work;
- restarting skips cells already present in ``rows.jsonl``, and the response
  cache covers the rest, so a resume costs close to nothing;
- a spend cap stops scheduling new work rather than discovering the bill
  afterwards;
- the run manifest records every knob that could change a number, because a
  benchmark result without its configuration is an anecdote.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .agentloop import AttemptOptions, ChatClient, run_attempt
from .contracts import SCHEMA_VERSION, BenchError, BenchTask, TaskOutcome
from .sandbox import SandboxConfig
from .suites import LoadOptions, ScoringContext, Suite, normalise_conditions


@dataclass
class RunConfig:
    """Everything a run needs, and everything that must be recorded about it."""

    suite: str
    models: tuple[str, ...]
    conditions: tuple[str, ...] = ()
    repeats: int = 1
    concurrency: int = 4
    output_dir: str = "bench-output"
    cache_dir: str | None = None
    budget_usd: float | None = None
    resume: bool = True

    # forwarded to the suite loader
    data_path: str | None = None
    revision: str = "main"
    limit: int | None = None
    seed: int = 20260801
    categories: tuple[str, ...] = ()
    config: str | None = None
    allow_pickle: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)

    # forwarded to the attempt
    temperature: float = 0.0
    max_tokens: int = 4096
    max_tool_steps: int = 8
    tool_profile: str = "analysis"
    include_hint: bool = False
    include_concepts: bool = False
    decimals: int = 1
    max_points: int | None = None

    # forwarded to the scorer
    allow_code_execution: bool = False
    exec_timeout_seconds: int = 60
    thresholds: Mapping[str, float] = field(default_factory=dict)

    def load_options(self) -> LoadOptions:
        return LoadOptions(
            data_path=self.data_path, revision=self.revision, limit=self.limit,
            seed=self.seed, categories=self.categories, config=self.config,
            allow_pickle=self.allow_pickle, cache=self.cache_dir, extra=self.extra,
        )

    def attempt_options(self) -> AttemptOptions:
        return AttemptOptions(
            temperature=self.temperature, max_tokens=self.max_tokens,
            max_tool_steps=self.max_tool_steps, tool_profile=self.tool_profile,
            include_hint=self.include_hint, include_concepts=self.include_concepts,
            decimals=self.decimals, max_points=self.max_points,
        )

    def scoring_context(self, workdir: Path) -> ScoringContext:
        return ScoringContext(
            sandbox=SandboxConfig(
                enabled=self.allow_code_execution,
                timeout_seconds=self.exec_timeout_seconds,
                extra_syspath=_aion_syspath(),
            ),
            workdir=workdir,
            allow_pickle=self.allow_pickle,
            thresholds=dict(self.thresholds),
        )

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["models"] = list(self.models)
        payload["conditions"] = list(self.conditions)
        payload["categories"] = list(self.categories)
        payload["extra"] = dict(self.extra)
        payload["thresholds"] = dict(self.thresholds)
        return payload


@dataclass
class RunResult:
    run_dir: Path
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    stopped_early: bool = False
    reason: str = ""


class _Ledger:
    """Thread-safe spend tracking and row writing."""

    def __init__(self, path: Path, budget_usd: float | None):
        self._path = path
        self._budget = budget_usd
        self._lock = threading.Lock()
        self._spent = 0.0
        self._handle = path.open("a", encoding="utf-8")

    def close(self) -> None:
        with self._lock:
            self._handle.close()

    def record(self, row: Mapping[str, Any]) -> None:
        with self._lock:
            self._spent += float(row.get("cost_usd") or 0.0)
            self._handle.write(json.dumps(row, default=str) + "\n")
            self._handle.flush()

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    def exhausted(self) -> bool:
        if self._budget is None:
            return False
        with self._lock:
            return self._spent >= self._budget


def run(
    suite: Suite,
    client: ChatClient,
    config: RunConfig,
    *,
    tasks: Sequence[BenchTask] | None = None,
    progress: Any = None,
) -> RunResult:
    """Execute a full benchmark run and return the graded rows."""
    conditions = normalise_conditions(suite, config.conditions or None)
    config.conditions = conditions
    if not config.models:
        raise BenchError("NO_MODELS", "At least one model is required.")

    loaded = list(tasks) if tasks is not None else suite.load(config.load_options())
    if not loaded:
        raise BenchError("NO_TASKS", f"Suite {suite.name!r} produced no tasks.")

    run_dir = Path(config.output_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    rows_path = run_dir / "rows.jsonl"
    work_root = run_dir / "work"

    existing: list[dict[str, Any]] = []
    done: set[tuple[str, str, str, int]] = set()
    if config.resume and rows_path.is_file():
        existing = _read_rows(rows_path)
        done = {
            (row["task_id"], row["model"], row["condition"], int(row.get("repeat", 0)))
            for row in existing
            # A transport failure is not a completed cell; retry it on resume.
            if row.get("failure") != "transport"
        }
        if existing:
            _rewrite(rows_path, [row for row in existing if _key(row) in done])
            existing = [row for row in existing if _key(row) in done]

    cells = [
        (task, model, condition, repeat)
        for task in loaded
        for model in config.models
        for condition in conditions
        for repeat in range(config.repeats)
        if (task.task_id, model, condition, repeat) not in done
    ]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "suite": suite.name,
        "suite_description": suite.description,
        "citation": suite.citation,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "task_count": len(loaded),
        "cells_total": len(loaded) * len(config.models) * len(conditions) * config.repeats,
        "cells_pending": len(cells),
        "cells_resumed": len(existing),
        "config": config.to_manifest(),
        "aion_version": _aion_version(),
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "tasks.json", [_task_summary(task) for task in loaded])

    ledger = _Ledger(rows_path, config.budget_usd)
    stopped = False
    reason = ""
    attempt_options = config.attempt_options()
    produced: list[dict[str, Any]] = []

    workers = max(1, config.concurrency)
    # Keep only a small queue in flight. Submitting the whole grid up front
    # would make the budget check meaningless: every cell would already be
    # scheduled before the first dollar was recorded.
    in_flight_cap = workers * 2
    pending = deque(cells)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures: dict[Any, tuple[BenchTask, str, str, int]] = {}

            def top_up() -> None:
                while pending and len(futures) < in_flight_cap and not ledger.exhausted():
                    cell = pending.popleft()
                    futures[pool.submit(
                        _run_cell, suite, client, config, attempt_options, work_root, *cell,
                    )] = cell

            top_up()
            while futures:
                completed, _ = wait(list(futures), return_when=FIRST_COMPLETED)
                for future in completed:
                    task, model, condition, repeat = futures.pop(future)
                    try:
                        outcome = future.result()
                    except Exception as exc:  # noqa: BLE001 - a crashed cell must not kill the run
                        outcome = _crash_outcome(suite, task, model, condition, repeat, exc)
                    row = outcome.to_row()
                    ledger.record(row)
                    produced.append(row)
                    if progress is not None:
                        progress(row, ledger.spent)

                if ledger.exhausted() and pending:
                    stopped, reason = True, f"Budget of ${config.budget_usd:.2f} reached."
                    pending.clear()
                top_up()
    finally:
        ledger.close()

    manifest.update({
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "cells_completed": len(produced),
        "spend_usd": round(ledger.spent, 6),
        "stopped_early": stopped,
        "stop_reason": reason,
    })
    _write_json(run_dir / "manifest.json", manifest)

    return RunResult(
        run_dir=run_dir, rows=existing + produced, manifest=manifest,
        stopped_early=stopped, reason=reason,
    )


def _run_cell(
    suite: Suite,
    client: ChatClient,
    config: RunConfig,
    attempt_options: AttemptOptions,
    work_root: Path,
    task: BenchTask,
    model: str,
    condition: str,
    repeat: int,
) -> TaskOutcome:
    workdir = work_root / _safe(model) / condition / _safe(task.task_id) / str(repeat)
    workdir.mkdir(parents=True, exist_ok=True)
    attempt = run_attempt(
        client, task, model=model, condition=condition,
        workdir=workdir, options=attempt_options, repeat=repeat,
    )
    _write_json(workdir / "attempt.json", attempt.to_dict())
    return suite.score(task, attempt, config.scoring_context(workdir))


def _crash_outcome(
    suite: Suite, task: BenchTask, model: str, condition: str, repeat: int, exc: Exception,
) -> TaskOutcome:
    return TaskOutcome(
        task_id=task.task_id, suite=suite.name, model=model, condition=condition,
        repeat=repeat, kind=task.kind, metric=task.metric, category=task.category,
        subcategory=task.subcategory, difficulty=task.difficulty,
        correct=None, failure="transport",
        detail=f"Harness error: {type(exc).__name__}: {exc}",
    )


# -- helpers ---------------------------------------------------------------

def _key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (row["task_id"], row["model"], row["condition"], int(row.get("repeat", 0)))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a truncated final line from a killed run
            if {"task_id", "model", "condition"} <= set(row):
                rows.append(row)
    return rows


def _rewrite(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _task_summary(task: BenchTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "kind": task.kind,
        "metric": task.metric,
        "category": task.category,
        "subcategory": task.subcategory,
        "difficulty": task.difficulty,
        "options": len(task.options),
        "series": [{"name": item.name, "points": len(item)} for item in task.series],
        "answer": task.answer,
    }


def _safe(text: str) -> str:
    return "".join(char if char.isalnum() or char in "-._" else "_" for char in text)[:120]


def _aion_version() -> str:
    try:
        import aion

        return getattr(aion, "__version__", "unknown")
    except ImportError:
        return "not installed"


def _aion_syspath() -> tuple[str, ...]:
    """Make `aion` importable inside the sandbox even from a source checkout."""
    try:
        import aion

        package = Path(aion.__file__).resolve().parent.parent
        return (str(package),)
    except ImportError:
        source = Path(__file__).resolve().parents[2] / "src"
        return (str(source),) if source.is_dir() else ()

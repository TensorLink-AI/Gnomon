"""AionBench-compatible run records.

Every benchmark adapter, in addition to the benchmark's own official
metric output, emits one JSONL row per graded run in the format consumed
by ``aion eval compare`` (see docs/agent-evaluation.md). The official
metric stays the headline number for each external benchmark; these rows
add the completion/safety view (abstentions, tool calls, latency, cost)
and make treatment/control comparison mechanical.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    """One graded run. ``task_id`` and ``success`` are the only fields
    required by ``aion eval compare``; the rest default to the schema's
    zero values and extra keys are carried through untouched."""

    task_id: str
    success: bool
    temporal_leakage: bool = False
    invented_number: bool = False
    warning_omission: bool = False
    appropriate_abstention: bool = False
    tool_calls: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        extra = row.pop("extra")
        for key, value in extra.items():
            row.setdefault(key, value)
        return row


class RecordWriter:
    """Append-only JSONL writer for :class:`RunRecord` rows."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def write(self, record: RunRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_row(), sort_keys=True) + "\n")
        self.count += 1

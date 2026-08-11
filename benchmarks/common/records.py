"""GnomonBench-compatible run records.

Every benchmark adapter, in addition to the benchmark's own official
metric output, emits one JSONL row per graded run in the format consumed
by ``gnomon eval compare`` (see docs/agent-evaluation.md). The official
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
    required by ``gnomon eval compare``; extra keys are carried through
    untouched.

    The safety fields default to ``None`` — *unmeasured* — and are omitted
    from the row. They used to default to ``False``, which made every
    adapter that never graded them emit rows where the safety rates read
    as a measured 0.0: `gnomon eval compare` then printed safety deltas
    of exactly zero for benchmarks in which nothing ever checked for a
    leak or an invented number. An adapter that actually grades a
    property passes an explicit ``True``/``False``; absence now means
    "nobody looked", and the comparator reports it that way.
    """

    task_id: str
    success: bool
    temporal_leakage: bool | None = None
    invented_number: bool | None = None
    warning_omission: bool | None = None
    appropriate_abstention: bool = False
    tool_calls: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    #: The graded-or-absent safety fields (see class docstring).
    SAFETY_FIELDS = ("temporal_leakage", "invented_number", "warning_omission")

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        extra = row.pop("extra")
        for name in self.SAFETY_FIELDS:
            if row[name] is None:
                del row[name]
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

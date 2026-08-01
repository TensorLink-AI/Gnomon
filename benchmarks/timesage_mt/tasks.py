"""Load TimeSage-MT tasks from the official Hugging Face dataset.

Layout of the official dataset (hf.co/datasets/Timesage/TimeSage-MT):

- ``MT_Bench/L{1..4}/<task_id>.json`` — one task per file: metadata,
  the multi-turn ``dialogue`` (user turns and reference agent turns with
  ``finding_verify`` specs), and a ``visibility_contract`` limiting what
  the agent may see.
- ``visible_ts/<tier>/<task_id>/agent_input/*.csv`` — the per-task
  visible series the agent is allowed to read (rows beyond
  ``visibility_contract.rows_visible`` never appear here).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TIERS = ("L1", "L2", "L3", "L4")


@dataclass
class TimeSageTask:
    task_id: str
    tier: str
    domain: str | None
    dialogue: list[dict[str, Any]]
    visibility: dict[str, Any]
    visible_csv: Path | None
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def user_turns(self) -> list[dict[str, Any]]:
        return [turn for turn in self.dialogue if turn.get("role") == "user"]

    def reference_turn_after(self, user_turn_id: int) -> dict[str, Any] | None:
        """The official agent turn that answers the given user turn."""
        for turn in self.dialogue:
            if turn.get("role") == "agent" and turn.get("turn_id") == user_turn_id + 1:
                return turn
        return None


def download(data_dir: Path) -> Path:
    """Fetch the official dataset with huggingface_hub (task files and
    visible series only; the large raw_ts tree is not needed to run)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id="Timesage/TimeSage-MT",
        repo_type="dataset",
        local_dir=str(data_dir),
        allow_patterns=["MT_Bench/*", "visible_ts/*", "README.md"],
    ))


def find_visible_csv(data_dir: Path, tier: str, task_id: str) -> Path | None:
    task_dir = data_dir / "visible_ts" / tier / task_id / "agent_input"
    if task_dir.is_dir():
        candidates = sorted(task_dir.glob("*.csv"))
        if candidates:
            return candidates[0]
    return None


def load_tasks(
    data_dir: Path,
    *,
    tiers: tuple[str, ...] = TIERS,
    limit: int | None = None,
) -> list[TimeSageTask]:
    tasks: list[TimeSageTask] = []
    for tier in tiers:
        tier_dir = data_dir / "MT_Bench" / tier
        if not tier_dir.is_dir():
            continue
        for json_path in sorted(tier_dir.glob("*.json")):
            with json_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            task_id = raw.get("id", json_path.stem)
            tasks.append(TimeSageTask(
                task_id=task_id,
                tier=raw.get("tier", tier),
                domain=raw.get("domain"),
                dialogue=raw.get("dialogue", []),
                visibility=raw.get("visibility_contract") or {},
                visible_csv=find_visible_csv(data_dir, tier, task_id),
                raw=raw,
            ))
    if not tasks:
        raise FileNotFoundError(
            f"No TimeSage-MT task files under {data_dir}/MT_Bench. "
            "Run with --download or pass the dataset snapshot directory."
        )
    if limit:
        tasks = tasks[:limit]
    return tasks


def read_visible_series(task: TimeSageTask) -> tuple[list[str], dict[str, list[float]], str]:
    """Read the visible CSV: (timestamps, numeric columns, csv text).

    Falls back to the task JSON's serialized payload when the per-task
    CSV is absent from the snapshot.
    """
    if task.visible_csv and task.visible_csv.exists():
        text = task.visible_csv.read_text(encoding="utf-8")
        rows = list(csv.DictReader(text.splitlines()))
    else:
        payload = task.raw.get("time_series_json")
        rows = json.loads(payload) if isinstance(payload, str) else (payload or [])
        text = json.dumps(rows[:50])
    if not rows:
        return [], {}, text
    columns = list(rows[0].keys())
    time_column = columns[0]
    timestamps = [str(row[time_column]) for row in rows]
    numeric: dict[str, list[float]] = {}
    for column in columns[1:]:
        try:
            numeric[column] = [float(row[column]) for row in rows]
        except (TypeError, ValueError):
            continue
    return timestamps, numeric, text

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
    if limit and limit < len(tasks):
        tasks = _stratified_head(tasks, limit)
    return tasks


def _stratified_head(tasks: list[TimeSageTask], limit: int) -> list[TimeSageTask]:
    """Spread a task limit across tiers instead of head-truncating.

    ``tasks[:limit]`` on a tier-sorted list silently reduced every
    limited run to the earliest requested tier — ``--limit`` defeated
    ``--tiers`` and every result was L1. The limit now takes tasks
    round-robin across the tiers present (extra slots go to earlier
    tiers), keeping each tier's sorted order and the overall
    tier-grouped order, so a limited run still samples every tier it
    asked for. Deterministic: same tasks, same limit, same selection.
    """
    by_tier: dict[str, list[TimeSageTask]] = {}
    for task in tasks:
        by_tier.setdefault(task.tier, []).append(task)
    taken = {tier: 0 for tier in by_tier}
    remaining = limit
    while remaining > 0:
        progressed = False
        for tier, members in by_tier.items():
            if remaining == 0:
                break
            if taken[tier] < len(members):
                taken[tier] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return [task for tier, members in by_tier.items()
            for task in members[:taken[tier]]]


def read_visible_series(task: TimeSageTask) -> tuple[list[str], dict[str, list[float]], str]:
    """Read the visible CSV: (timestamps, numeric columns, csv text).

    Falls back to the task JSON's serialized payload when the per-task
    CSV is absent from the snapshot. The official CSV is already limited
    to the visibility contract, but the JSON payload is not — so the
    fallback enforces ``visibility_contract.rows_visible`` here, before
    anything downstream sees the rows, and serializes exactly the rows
    it returns: the prompt text and the tools always operate on the same
    truncated series.
    """
    if task.visible_csv and task.visible_csv.exists():
        text = task.visible_csv.read_text(encoding="utf-8")
        rows = list(csv.DictReader(text.splitlines()))
    else:
        payload = task.raw.get("time_series_json")
        rows = json.loads(payload) if isinstance(payload, str) else (payload or [])
        rows_visible = task.visibility.get("rows_visible")
        if (isinstance(rows_visible, int) and not isinstance(rows_visible, bool)
                and rows_visible > 0):
            rows = rows[:rows_visible]
        text = json.dumps(rows)
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

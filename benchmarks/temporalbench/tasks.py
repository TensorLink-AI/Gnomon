"""Load TemporalBench task rows and the official metric module.

The labeled benchmark split (``task_merged_dev_with_labels_tiers.jsonl``)
carries one task per row:

- ``tier`` T1–T4 with the complete official ``prompt``;
- T1: ``labels`` (four MCQ fields, exact-match);
- T2/T4: ``input`` (history + future covariates), ``ground_truth``
  (future arrays per channel), ``mcq`` (three questions with labels);
- T3: ``pack`` (question list with options and labels).

The dataset also ships ``forecast_metrics_utils.py`` — the official
reference implementation of every forecast metric (MAPE/MAE/RMSE/SMAPE,
plus weighted OW metrics for multi-channel MIMIC rows). This module
loads that file directly so scoring is the official code, not a port.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Iterator

REPO_ID = "Melady/TemporalBench"
LABELED_FILE = "task_merged_dev_with_labels_tiers.jsonl"
METRICS_FILE = "forecast_metrics_utils.py"
TIERS = ("T1", "T2", "T3", "T4")


def download(data_dir: Path) -> Path:
    """Fetch the labeled split and the official metric module."""
    from huggingface_hub import hf_hub_download

    for filename in (LABELED_FILE, METRICS_FILE):
        hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset", filename=filename,
            local_dir=str(data_dir),
        )
    return data_dir


def load_official_metrics(data_dir: Path):
    """Import the officially shipped forecast_metrics_utils.py."""
    path = data_dir / METRICS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run with --download first."
        )
    spec = importlib.util.spec_from_file_location(
        "temporalbench_official_metrics", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_rows(
    data_dir: Path,
    *,
    tiers: tuple[str, ...] = TIERS,
    datasets: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Rows of the labeled split, filtered by tier and source dataset.

    ``limit`` is stratified across the requested tiers rather than
    head-truncating in file order: on a tier-grouped file, taking the
    first N rows silently reduced every limited multi-tier run to the
    earliest tier — ``--limit`` defeated ``--tiers``, the exact bug the
    TimeSage loader already fixed with ``_stratified_head``. The limit
    now takes rows round-robin across the tiers present (extra slots go
    to earlier tiers), keeping each tier's file order. Deterministic:
    same file, same filters, same limit, same selection.
    """
    path = data_dir / LABELED_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run with --download first."
        )

    def matching() -> Iterator[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("tier") not in tiers:
                    continue
                if datasets and row.get("source_dataset") not in datasets:
                    continue
                yield row

    if not limit:
        yield from matching()
        return
    rows = list(matching())
    if limit >= len(rows):
        yield from rows
        return
    by_tier: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_tier.setdefault(str(row.get("tier")), []).append(row)
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
    for tier, members in by_tier.items():
        yield from members[:taken[tier]]


def extract_json_object(text: str) -> dict[str, Any]:
    """First parseable top-level JSON object in free-form text."""
    depth = 0
    start = None
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    start = None
                    continue
                if isinstance(parsed, dict):
                    return parsed
                start = None
    raise ValueError("No JSON object found in text.")


def prompt_input_arrays(row: dict[str, Any]) -> dict[str, list[float | None]]:
    """The task's numeric arrays: from ``input.history`` when present
    (T2/T4), otherwise parsed from the ``Input (JSON):`` block that the
    official prompt embeds (T1/T3)."""
    task_input = row.get("input")
    if isinstance(task_input, dict) and isinstance(task_input.get("history"), dict):
        source = task_input["history"]
    else:
        prompt = row.get("prompt", "")
        marker = prompt.find("Input (JSON):")
        source = extract_json_object(prompt[marker:] if marker >= 0 else prompt)
    arrays: dict[str, list[float | None]] = {}
    for key, values in source.items():
        if isinstance(values, list):
            arrays[key] = [
                float(v) if isinstance(v, (int, float)) else None for v in values
            ]
    return arrays

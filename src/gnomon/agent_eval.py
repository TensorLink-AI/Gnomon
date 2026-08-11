"""Deterministic treatment/control evaluation for agents using Gnomon."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_runs(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "task_id" not in row or "success" not in row:
                raise ValueError(f"{path}:{line_number} requires task_id and success")
            rows.append(row)
    if not rows:
        raise ValueError(f"No evaluation runs found in {path}")
    return rows


def _voided(row: dict[str, Any]) -> bool:
    """A row the harness ended without an answer (adapters mark it with
    ``row_abstained``). It did not answer the task wrongly — it did not
    answer it — so it must not enter any rate as a model failure."""
    return bool(row.get("row_abstained") or row.get("voided"))


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [row for row in rows if not _voided(row)]
    count = len(graded)
    rate = lambda key: (
        sum(bool(row.get(key, False)) for row in graded) / count
        if count else None
    )
    numeric = lambda key: (
        sum(float(row.get(key, 0) or 0) for row in graded) / count
        if count else None
    )
    return {
        "runs": len(rows),
        "runs_voided_by_harness": len(rows) - count,
        "runs_graded": count,
        "task_success": rate("success"),
        "temporal_leakage": rate("temporal_leakage"),
        "invented_number": rate("invented_number"),
        "warning_omission": rate("warning_omission"),
        "appropriate_abstention": rate("appropriate_abstention"),
        "average_tool_calls": numeric("tool_calls"),
        "average_latency_seconds": numeric("latency_seconds"),
        "average_cost_usd": numeric("cost_usd"),
    }


def compare_runs(baseline_path: str, treatment_path: str) -> dict[str, Any]:
    baseline_rows = load_runs(baseline_path)
    treatment_rows = load_runs(treatment_path)
    baseline_ids = {row["task_id"] for row in baseline_rows}
    treatment_ids = {row["task_id"] for row in treatment_rows}
    if baseline_ids != treatment_ids:
        raise ValueError("Baseline and treatment must contain identical task_id sets")
    # A task the harness voided in either arm (row_abstained: a breached
    # cap, a run that never submitted) is excluded from both arms' rates:
    # the comparison is between answers, and on those tasks at least one
    # side never gave one.
    voided_ids = {
        row["task_id"]
        for row in baseline_rows + treatment_rows
        if _voided(row)
    }
    baseline = _summary(
        [row for row in baseline_rows if row["task_id"] not in voided_ids])
    treatment = _summary(
        [row for row in treatment_rows if row["task_id"] not in voided_ids])
    if baseline["task_success"] is None or treatment["task_success"] is None:
        uplift = None
        relative = None
        interpretation = "No graded tasks: every task was voided by the harness"
    else:
        uplift = treatment["task_success"] - baseline["task_success"]
        baseline_error = 1.0 - baseline["task_success"]
        relative = uplift / baseline_error if baseline_error > 0 else None
        interpretation = (
            "Treatment improves task success" if uplift > 0
            else "No positive task-success uplift observed"
        )
    return {
        "schema_version": "0.1",
        "status": "complete",
        "tasks_total": len(baseline_ids),
        "tasks_voided_by_harness": len(voided_ids),
        "baseline": baseline,
        "treatment": treatment,
        "absolute_success_uplift": uplift,
        "relative_error_reduction": relative,
        "safety_delta": {
            key: (treatment[key] - baseline[key]
                  if treatment[key] is not None and baseline[key] is not None
                  else None)
            for key in ("temporal_leakage", "invented_number", "warning_omission")
        },
        "interpretation": interpretation,
    }

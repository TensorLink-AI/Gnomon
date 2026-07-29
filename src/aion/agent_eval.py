"""Deterministic treatment/control evaluation for agents using Aion."""

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


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    rate = lambda key: sum(bool(row.get(key, False)) for row in rows) / count
    numeric = lambda key: sum(float(row.get(key, 0) or 0) for row in rows) / count
    return {
        "runs": count,
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
    baseline = _summary(baseline_rows)
    treatment = _summary(treatment_rows)
    baseline_ids = {row["task_id"] for row in baseline_rows}
    treatment_ids = {row["task_id"] for row in treatment_rows}
    if baseline_ids != treatment_ids:
        raise ValueError("Baseline and treatment must contain identical task_id sets")
    uplift = treatment["task_success"] - baseline["task_success"]
    baseline_error = 1.0 - baseline["task_success"]
    return {
        "schema_version": "0.1",
        "status": "complete",
        "baseline": baseline,
        "treatment": treatment,
        "absolute_success_uplift": uplift,
        "relative_error_reduction": (
            uplift / baseline_error if baseline_error > 0 else None
        ),
        "safety_delta": {
            key: treatment[key] - baseline[key]
            for key in ("temporal_leakage", "invented_number", "warning_omission")
        },
        "interpretation": (
            "Treatment improves task success" if uplift > 0
            else "No positive task-success uplift observed"
        ),
    }

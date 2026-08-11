"""Deterministic treatment/control evaluation for agents using Gnomon."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

SAFETY_FIELDS = ("temporal_leakage", "invented_number", "warning_omission")


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
    duplicates = sorted({
        task_id for task_id in (row["task_id"] for row in rows)
        if sum(1 for row in rows if row["task_id"] == task_id) > 1
    })
    if duplicates:
        # The identical-set check between files compares *sets*, so
        # duplicated ids used to slip through while silently changing the
        # denominators — two files could "match" with different run counts.
        raise ValueError(
            f"{path} contains duplicate task_id values: "
            f"{', '.join(duplicates[:5])}"
            + ("..." if len(duplicates) > 5 else "")
        )
    return rows


def _voided(row: dict[str, Any]) -> bool:
    """A row the harness ended without an answer (adapters mark it with
    ``row_abstained``). It did not answer the task wrongly — it did not
    answer it — so it must not enter any rate as a model failure."""
    return bool(row.get("row_abstained") or row.get("voided"))


def _summary(
    rows: list[dict[str, Any]],
    measured_safety: set[str] | None = None,
) -> dict[str, Any]:
    """``measured_safety`` names the safety fields some grader measured —
    for a comparison, across *both* files, so an arm whose grader writes
    only the true cases still compares against one that wrote explicit
    falses. Defaults to the fields present in these rows."""
    graded = [row for row in rows if not _voided(row)]
    count = len(graded)
    if measured_safety is None:
        measured_safety = {
            key for key in SAFETY_FIELDS
            if any(key in row for row in graded)
        }

    def rate(key: str) -> float | None:
        """Rate over the graded rows, with absence meaning false — except
        for a safety field nobody measured at all, which is unmeasured:
        reporting 0.0 there presented an unchecked property as a perfect
        score."""
        if key in SAFETY_FIELDS and key not in measured_safety:
            return None
        if not count:
            return None
        return sum(bool(row.get(key, False)) for row in graded) / count

    def numeric(key: str) -> float | None:
        """Mean over the rows that carry the field; rows without it used
        to average in as 0 and deflate the figure."""
        measured = [row for row in graded if key in row]
        if not measured:
            return None
        return sum(float(row[key] or 0) for row in measured) / len(measured)

    return {
        "runs": len(rows),
        "runs_voided_by_harness": len(rows) - count,
        "runs_graded": count,
        "task_success": (
            sum(bool(row.get("success")) for row in graded) / count
            if count else None
        ),
        "temporal_leakage": rate("temporal_leakage"),
        "invented_number": rate("invented_number"),
        "warning_omission": rate("warning_omission"),
        "appropriate_abstention": rate("appropriate_abstention"),
        "average_tool_calls": numeric("tool_calls"),
        "average_latency_seconds": numeric("latency_seconds"),
        "average_cost_usd": numeric("cost_usd"),
    }


def _two_sided_binomial(successes: int, trials: int) -> float:
    """Exact two-sided p under p=0.5 (the McNemar discordant-pair test)."""
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, k)
               for k in range(0, min(successes, trials - successes) + 1))
    return min(1.0, 2 * tail / 2 ** trials)


def _success_test(
    baseline_rows: list[dict[str, Any]], treatment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Exact McNemar over the graded pairs: the files share task ids, so
    the uplift is paired data and deserves a paired test rather than an
    interpretation string asserted off a bare difference of rates."""
    baseline_by_id = {row["task_id"]: bool(row.get("success"))
                      for row in baseline_rows}
    fixed = broke = 0
    for row in treatment_rows:
        base = baseline_by_id.get(row["task_id"])
        treat = bool(row.get("success"))
        if not base and treat:
            fixed += 1
        elif base and not treat:
            broke += 1
    return {
        "test": "mcnemar_exact",
        "treatment_fixed": fixed,
        "treatment_broke": broke,
        "p_value": _two_sided_binomial(min(fixed, broke), fixed + broke),
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
    baseline_graded = [row for row in baseline_rows
                       if row["task_id"] not in voided_ids]
    treatment_graded = [row for row in treatment_rows
                        if row["task_id"] not in voided_ids]
    # A safety field measured by either arm's grader is comparable — the
    # other arm's absent rows read as explicit falses. A field measured by
    # neither is unmeasured, not zero.
    measured_safety = {
        key for key in SAFETY_FIELDS
        if any(key in row for row in baseline_graded + treatment_graded)
    }
    baseline = _summary(baseline_graded, measured_safety)
    treatment = _summary(treatment_graded, measured_safety)
    success_test: dict[str, Any] | None = None
    if baseline["task_success"] is None or treatment["task_success"] is None:
        uplift = None
        relative = None
        interpretation = "No graded tasks: every task was voided by the harness"
    else:
        uplift = treatment["task_success"] - baseline["task_success"]
        baseline_error = 1.0 - baseline["task_success"]
        relative = uplift / baseline_error if baseline_error > 0 else None
        success_test = _success_test(baseline_graded, treatment_graded)
        p_value = success_test["p_value"]
        if uplift <= 0:
            interpretation = "No positive task-success uplift observed"
        elif p_value <= 0.05:
            interpretation = (
                f"Treatment improves task success "
                f"(exact McNemar p={p_value:.4f})"
            )
        else:
            interpretation = (
                f"Treatment task-success uplift is not statistically "
                f"distinguishable from zero (exact McNemar p={p_value:.4f} "
                f"over {baseline['runs_graded']} graded tasks)"
            )
    unmeasured = [key for key in SAFETY_FIELDS if key not in measured_safety]
    return {
        "schema_version": "0.1",
        "status": "complete",
        "tasks_total": len(baseline_ids),
        "tasks_voided_by_harness": len(voided_ids),
        "baseline": baseline,
        "treatment": treatment,
        "absolute_success_uplift": uplift,
        "relative_error_reduction": relative,
        **({"success_test": success_test} if success_test else {}),
        "safety_delta": {
            key: (treatment[key] - baseline[key]
                  if treatment[key] is not None and baseline[key] is not None
                  else None)
            for key in SAFETY_FIELDS
        },
        **({"safety_note": (
            "unmeasured by both arms (no row carries the field): "
            + ", ".join(unmeasured)
        )} if unmeasured else {}),
        "interpretation": interpretation,
    }

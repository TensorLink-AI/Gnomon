"""Completion-aware matched agent evaluation; supplied grades are not attested.

Headline success includes every scheduled task, including capped/unfinished runs.
Conditional quality, measurement coverage and resource totals remain separate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

SAFETY_FIELDS = ("temporal_leakage", "invented_number", "warning_omission")
_RESOURCE_FIELDS = ("tool_calls", "latency_seconds", "cost_usd", "run_tokens")
_NUMERIC_FIELDS = (*_RESOURCE_FIELDS, "accuracy", "success_probability")
_BOOL_FIELDS = (*SAFETY_FIELDS, "appropriate_abstention", "completed", "abstained", "budget_exceeded", "voided")
MAX_RUNS = 10_000
MAX_ROW_BYTES = 1_048_576


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("evaluation JSON numbers must be finite")
    return number


def _invalid_constant(value):
    raise ValueError("evaluation JSON numbers must be finite")


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("evaluation JSON contains duplicate fields")
        result[key] = value
    return result


def _validate_rows(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_RUNS:
        raise ValueError(f"evaluation requires 1..{MAX_RUNS} rows")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not {"task_id", "success"} <= row.keys():
            raise ValueError("evaluation row requires task_id and success")
        task_id = row["task_id"]
        if not isinstance(task_id, str) or not 1 <= len(task_id) <= 1024:
            raise ValueError("task_id must be a nonempty string of at most 1024 characters")
        if task_id in seen:
            raise ValueError("evaluation contains duplicate task_id values")
        seen.add(task_id)
        if type(row["success"]) is not bool:
            raise ValueError("success must be a boolean")
        for field in _BOOL_FIELDS:
            if row.get(field) is not None and type(row[field]) is not bool:
                raise ValueError(f"{field} must be boolean or null (unmeasured)")
        for field in _NUMERIC_FIELDS:
            value = row.get(field)
            if value is None:
                continue
            try:
                valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
            except OverflowError:
                valid = False
            if not valid or (field in ("accuracy", "success_probability") and value > 1):
                raise ValueError(f"{field} must be a finite nonnegative number" +
                                 (" in [0,1]" if field in ("accuracy", "success_probability") else ""))
            if field in ("tool_calls", "run_tokens") and type(value) is not int:
                raise ValueError(f"{field} must be an integer")
        for field in ("status", "success_basis"):
            if row.get(field) is not None and not isinstance(row[field], str):
                raise ValueError(f"{field} must be a string or null")
        if row.get("row_abstained") is not None and type(row["row_abstained"]) not in (bool, str):
            raise ValueError("row_abstained must be a boolean, reason string or null")
        if row.get("error") is not None and type(row["error"]) not in (bool, str, dict):
            raise ValueError("error must be a boolean, string, object or null")


def _decode_record(raw):
    try:
        value = json.loads(raw, parse_float=_finite_float, parse_constant=_invalid_constant,
                           object_pairs_hook=_unique_fields)
    except RecursionError:
        raise ValueError("evaluation JSON nesting exceeds parser limit") from None
    if not isinstance(value, dict):
        raise ValueError("evaluation record must be an object")
    exhausted = object()
    frames = [(iter((value,)), 0)]
    while frames:
        iterator, depth = frames[-1]
        child = next(iterator, exhausted)
        if child is exhausted:
            frames.pop()
        elif isinstance(child, (dict, list)):
            if depth >= 128:
                raise ValueError("evaluation JSON nesting exceeds 128 levels")
            frames.append((iter(child.values() if isinstance(child, dict) else child), depth + 1))
    return value


def _read_records(path):
    rows = []
    with Path(path).expanduser().open("rb") as handle:
        while line := handle.readline(MAX_ROW_BYTES + 1):
            if len(line) > MAX_ROW_BYTES:
                raise ValueError("evaluation row exceeds byte limit")
            if not line.strip():
                continue
            rows.append(_decode_record(line))
            if len(rows) > MAX_RUNS:
                raise ValueError("evaluation exceeds row limit")
    return rows


def load_runs(path: str) -> list[dict[str, Any]]:
    rows = _read_records(path)
    _validate_rows(rows)
    return rows


def _voided(row):
    return bool(row.get("row_abstained") or row.get("voided"))


def _completion(row):
    if _voided(row) or row.get("status") == "error" or bool(row.get("error")):
        return False
    if row.get("completed") is not None:
        return row["completed"]
    if row.get("status") in ("answered", "complete", "completed", "abstained"):
        return True
    return True if row["success"] else None


def _success(row):
    return row["success"] and _completion(row) is not False


def _flag(row, field):
    if field == "completed":
        return _completion(row)
    if field == "error" and row.get("status") == "error":
        return True
    if field == "abstained" and row.get("status") == "abstained":
        return True
    if field == "budget_exceeded" and str(row.get("row_abstained", "")).startswith("cap:"):
        return True
    if row.get(field) is not None:
        return bool(row[field])
    return None


def _coverage(rows, values):
    known = [(row["task_id"], value) for row, value in zip(rows, values) if value is not None]
    return {"measured": len(known), "unmeasured": len(rows) - len(known),
            "task_ids": sorted(task_id for task_id, _ in known)}, [value for _, value in known]


def summarize_rows(rows):
    """Summarize all attempts; an unknown observation is never an implicit zero."""
    _validate_rows(rows)
    count = len(rows)
    coverage = {}
    rates = {}
    for key in (*SAFETY_FIELDS, "appropriate_abstention", "completed", "error", "budget_exceeded", "abstained"):
        coverage[key], values = _coverage(rows, [_flag(row, key) for row in rows])
        rates[key] = sum(values) / len(values) if values else None
        coverage[key]["positive"] = sum(values)
    resources, numeric_means = {}, {}
    for key in _NUMERIC_FIELDS:
        # Accuracy is conditional on a possibly delivered answer; never an all-task headline.
        values = [row.get(key) if key != "accuracy" or _completion(row) is not False else None for row in rows]
        coverage[key], measured = _coverage(rows, values)
        # Dividing before summing keeps finite means finite for very large costs.
        mean = math.fsum(value / len(measured) for value in measured) if measured else None
        numeric_means[key] = mean
        if key not in _RESOURCE_FIELDS:
            continue
        try:
            observed = math.fsum(measured) if measured else None
        except OverflowError:
            observed = None
        resources[key] = {"mean": mean, "observed_total": observed,
                          "total": observed if len(measured) == count else None,
                          "total_overflow": bool(measured) and observed is None}
    probability_rows = [row for row in rows if row.get("success_probability") is not None]
    bins = []
    for index in range(10):
        cohort = [row for row in probability_rows if min(9, int(row["success_probability"] * 10)) == index]
        if cohort:
            bins.append({"lower": index / 10, "upper": (index + 1) / 10,
                         "count": len(cohort), "task_ids": sorted(row["task_id"] for row in cohort),
                         "mean_probability": math.fsum(row["success_probability"] / len(cohort) for row in cohort),
                         "observed_success": sum(_success(row) for row in cohort) / len(cohort)})
    completion = coverage["completed"]
    return {
        "runs": count, "runs_voided_by_harness": sum(_voided(row) for row in rows),
        "runs_graded": sum(not _voided(row) for row in rows),
        "task_success": sum(_success(row) for row in rows) / count,
        "success_basis": "delivered_success_over_all_scheduled_tasks",
        "completion_rate": rates.pop("completed"),
        "completion_rate_basis": "measured_completion_only_see_all_task_bounds",
        "completion_bounds": {"lower": completion["positive"] / count,
                              "upper": (completion["positive"] + completion["unmeasured"]) / count},
        **rates, "measurement_coverage": coverage, "resources": resources,
        **{"average_" + key: resources[key]["mean"] for key in ("tool_calls", "latency_seconds", "cost_usd", "run_tokens")},
        "conditional_accuracy": numeric_means["accuracy"],
        "calibration": {
            "target": "delivered_task_success", "measured": len(probability_rows),
            "unmeasured": count - len(probability_rows), "bins": bins,
            "bin_boundary": "[lower,upper)_except_last_includes_one",
            "brier_score": (math.fsum((row["success_probability"] - _success(row)) ** 2 / len(probability_rows)
                                     for row in probability_rows) if probability_rows else None),
            "claim": "descriptive_only_not_proof_of_calibration",
            "probability_provenance": "supplied_not_verified_as_pre_outcome",
        },
    }


def _two_sided_binomial(successes: int, trials: int) -> float:
    """Exact symmetric binomial tail for paired McNemar; integer recurrence."""
    term = tail = 1
    for k in range(1, min(successes, trials - successes) + 1):
        term = term * (trials - k + 1) // k
        tail += term
    return min(1.0, (2 * tail) / (2 ** trials))


def _success_test(baseline_rows, treatment_rows):
    baseline = {row["task_id"]: _success(row) for row in baseline_rows}
    fixed = broke = 0
    for row in treatment_rows:
        base, treat = baseline[row["task_id"]], _success(row)
        fixed += not base and treat
        broke += base and not treat
    return {"test": "mcnemar_exact", "n": len(baseline_rows), "treatment_fixed": fixed,
            "treatment_broke": broke, "p_value": _two_sided_binomial(min(fixed, broke), fixed + broke),
            "assumptions": "independent_task_pairs_no_multiple_comparison_adjustment"}


def compare_rows(baseline_rows, treatment_rows):
    """Compare identical task IDs; task/prompt/grader equivalence needs a manifest."""
    _validate_rows(baseline_rows)
    _validate_rows(treatment_rows)
    base_by_id = {row["task_id"]: row for row in baseline_rows}
    treat_by_id = {row["task_id"]: row for row in treatment_rows}
    if base_by_id.keys() != treat_by_id.keys():
        raise ValueError("Baseline and treatment must contain identical task_id sets")
    ids = sorted(base_by_id)
    for task_id in ids:
        a, b = base_by_id[task_id].get("success_basis"), treat_by_id[task_id].get("success_basis")
        if a is not None and b is not None and a != b:
            raise ValueError("matched task has incompatible success_basis")
    baseline, treatment = summarize_rows(baseline_rows), summarize_rows(treatment_rows)
    uplift = treatment["task_success"] - baseline["task_success"]
    error = 1 - baseline["task_success"]
    test = _success_test(baseline_rows, treatment_rows)
    conditional_ids = [task_id for task_id in ids
                       if _completion(base_by_id[task_id]) is True and _completion(treat_by_id[task_id]) is True]
    safety_delta, safety_pairs = {}, {}
    for field in SAFETY_FIELDS:
        matched = [task_id for task_id in ids if base_by_id[task_id].get(field) is not None
                   and treat_by_id[task_id].get(field) is not None]
        safety_pairs[field] = {"measured_pairs": len(matched), "unmeasured_pairs": len(ids) - len(matched), "task_ids": matched}
        safety_delta[field] = (sum(int(treat_by_id[key][field]) - int(base_by_id[key][field]) for key in matched) / len(matched)
                               if matched else None)
    if uplift <= 0:
        interpretation = "No positive all-task delivered-success uplift observed"
    elif test["p_value"] <= 0.05:
        interpretation = "Positive all-task delivered-success difference in this cohort; not a general causal or deployment claim"
    else:
        interpretation = "All-task delivered-success uplift is not statistically distinguishable from zero"
    return {
        "schema_version": "0.2", "status": "complete", "tasks_total": len(ids), "task_ids": ids,
        "tasks_voided_by_harness": sum(_voided(base_by_id[key]) or _voided(treat_by_id[key]) for key in ids),
        "baseline": baseline, "treatment": treatment, "absolute_success_uplift": uplift,
        "relative_error_reduction": uplift / error if error > 0 else None, "success_test": test,
        "conditional_completed_pairs": {
            "task_ids": conditional_ids, "count": len(conditional_ids),
            "baseline_success": sum(_success(base_by_id[key]) for key in conditional_ids) / len(conditional_ids) if conditional_ids else None,
            "treatment_success": sum(_success(treat_by_id[key]) for key in conditional_ids) / len(conditional_ids) if conditional_ids else None,
            "basis": "diagnostic_only_selection_on_completion",
        },
        "safety_delta": safety_delta, "safety_pairs": safety_pairs,
        "safety_note": "Missing/null fields are unmeasured; deltas use only exact pairs with explicit grades in both arms.",
        "comparability": "task_ids_matched_prompt_grader_probability_timing_not_attested",
        "interpretation": interpretation,
    }


def compare_runs(baseline_path: str, treatment_path: str) -> dict[str, Any]:
    return compare_rows(load_runs(baseline_path), load_runs(treatment_path))

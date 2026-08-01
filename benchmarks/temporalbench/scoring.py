"""Score TemporalBench answers.

Forecast metrics are computed by the dataset's own
``forecast_metrics_utils.compute_forecast_metrics`` — the official
reference implementation, imported unmodified (including the weighted
OW metrics it applies to multi-channel MIMIC rows, with the row's
history supplied for the naive scales). Choice answers are exact-match
against the labels embedded in the rows, case-insensitive on the
option string.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


def _norm(value: Any) -> str:
    return str(value).strip().lower()


def score_t1(row: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    labels = row.get("labels") or {}
    per_field = {
        field: _norm(answer.get(field)) == _norm(expected)
        for field, expected in labels.items()
    }
    return {
        "fields": per_field,
        "correct": sum(per_field.values()),
        "total": len(per_field),
    }


def score_t3(row: dict[str, Any], answers: list[Any]) -> dict[str, Any]:
    pack = row.get("pack") or []
    per_question = []
    for index, question in enumerate(pack):
        given = answers[index] if index < len(answers) else None
        per_question.append(_norm(given) == _norm(question.get("label")))
    return {
        "per_question": per_question,
        "correct": sum(per_question),
        "total": len(pack),
    }


def score_mcq(row: dict[str, Any], mcq_answer: dict[str, Any]) -> dict[str, Any]:
    """T2/T4 multiple-choice block: exact match per question key."""
    reference = row.get("mcq") or {}
    per_question = {
        key: _norm((mcq_answer or {}).get(key)) == _norm(entry.get("label"))
        for key, entry in reference.items()
    }
    return {
        "per_question": per_question,
        "correct": sum(per_question.values()),
        "total": len(reference),
    }


def score_forecast(
    row: dict[str, Any], forecast: Any, official_metrics
) -> tuple[dict[str, Any] | None, str | None]:
    """Official forecast metrics for a T2/T4 row.

    Multi-channel ground truth goes through the official module as a
    dict-of-series (OW metrics for MIMIC, with history for the naive
    scales); single-target rows are scored on the main channel.
    """
    ground_truth = row.get("ground_truth")
    if ground_truth is None:
        return None, "row_has_no_ground_truth"
    history = OrderedDict()
    task_input = row.get("input") or {}
    for key, values in (task_input.get("history") or {}).items():
        if isinstance(values, list):
            history[key] = values
    if isinstance(ground_truth, dict) and len(ground_truth) == 1:
        (only_key, gt_values), = ground_truth.items()
        forecast_values = (forecast or {}).get(only_key) \
            if isinstance(forecast, dict) else forecast
        return official_metrics.compute_forecast_metrics(
            gt_values, forecast_values,
            dataset_name=row.get("source_dataset"),
            history_series=history or None,
        )
    return official_metrics.compute_forecast_metrics(
        ground_truth, forecast,
        dataset_name=row.get("source_dataset"),
        history_series=history or None,
    )

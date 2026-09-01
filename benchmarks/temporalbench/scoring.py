"""Score TemporalBench answers.

Forecast metrics are computed by the dataset's own
``forecast_metrics_utils.compute_forecast_metrics`` — the official
reference implementation, imported unmodified (including the weighted
OW metrics it applies to multi-channel MIMIC rows, with the row's
history supplied for the naive scales). Choice answers are scored by
THIS module's own case-insensitive exact match against the labels
embedded in the rows — a local convention, not the official choice
scorer; its equivalence to the leaderboard's choice scoring is
unverified (disclosed in this adapter's README).
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


def choice_reference(row: dict[str, Any]) -> dict[str, Any]:
    """Public slot-to-label map used only after a row has been answered.

    The agent and projection boundary never receive this map.  Keeping the
    scorer-side representation uniform lets preservation accounting cover
    T1/T3 as well as the forecast tiers without exposing labels to inference.
    """
    tier = row.get("tier")
    if tier == "T1":
        return dict(row.get("labels") or {})
    if tier == "T3":
        return {f"q{index + 1}": item.get("label")
                for index, item in enumerate(row.get("pack") or [])
                if isinstance(item, dict)}
    return {key: item.get("label")
            for key, item in (row.get("mcq") or {}).items()
            if isinstance(item, dict)}


def choice_answer_map(row: dict[str, Any], answer: dict[str, Any],
                      ) -> dict[str, Any]:
    """Normalize a tier-native submitted answer into named public slots."""
    if row.get("tier") == "T3":
        values = (answer or {}).get("answers") or []
        return {f"q{index + 1}": value
                for index, value in enumerate(values)}
    if row.get("tier") == "T1":
        return {key: (answer or {}).get(key)
                for key in (row.get("labels") or {}) if key in (answer or {})}
    return dict((answer or {}).get("mcq") or answer or {})


def score_choice_map(row: dict[str, Any], answers: dict[str, Any],
                     ) -> dict[str, Any]:
    """Score only the named slots present in ``answers``."""
    reference = choice_reference(row)
    comparable = {key: value for key, value in (answers or {}).items()
                  if key in reference}
    per_question = {
        key: _norm(value) == _norm(reference[key])
        for key, value in comparable.items()
    }
    return {"per_question": per_question,
            "correct": sum(per_question.values()),
            "total": len(per_question)}


def score_forecast(
    row: dict[str, Any], forecast: Any, official_metrics
) -> tuple[dict[str, Any] | None, str | None]:
    """Official forecast metrics for a T2/T4 row.

    Multi-channel ground truth goes through the official module as a
    dict-of-series (OW metrics for MIMIC, with history for the naive
    scales); single-target rows are scored on the main channel.

    A fully absent forecast (no forecast at all, or every channel
    abstained) never reaches the official module: metrics are ``None``
    with flag ``"no_forecast"``, so an abstention cannot surface as a
    scoring error. The flag is deliberately neutral — a control answer
    whose forecast field is malformed takes the same path, and only the
    record's ``appropriate_abstention`` says whether an actual abstention
    happened. A PARTIAL forecast still goes through the official module
    with the channels that exist, and the flag names the missing
    channels; whether the official module penalizes or ignores them is
    upstream-defined — this adapter only reports which were missing.
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
        if not forecast_values:
            return None, "no_forecast"
        return official_metrics.compute_forecast_metrics(
            gt_values, forecast_values,
            dataset_name=row.get("source_dataset"),
            history_series=history or None,
        )
    if isinstance(ground_truth, dict):
        present = {key: values for key, values in forecast.items() if values} \
            if isinstance(forecast, dict) else {}
        abstained = [key for key in ground_truth if key not in present]
        if not present:
            return None, "no_forecast"
        metrics, flag = official_metrics.compute_forecast_metrics(
            ground_truth, present,
            dataset_name=row.get("source_dataset"),
            history_series=history or None,
        )
        if abstained:
            missing = "missing_channels=" + ",".join(abstained)
            flag = f"{flag}; {missing}" if flag else missing
        return metrics, flag
    if not forecast:
        return None, "no_forecast"
    return official_metrics.compute_forecast_metrics(
        ground_truth, forecast,
        dataset_name=row.get("source_dataset"),
        history_series=history or None,
    )

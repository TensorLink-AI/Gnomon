"""Harness-level tests: transport, prompting, scoring primitives, reporting.

None of these touch the network or a provider key.
"""

from __future__ import annotations

import json

import pytest

from aionbench.contracts import Attempt, BenchError, BenchTask, SeriesRef, TaskOutcome
from aionbench.metrics import binary_f1, bootstrap_ci, index_f1, mape, mcnemar, smape
from aionbench.prompting import (
    build_user_prompt,
    is_abstention,
    parse_choice,
    parse_code,
    parse_matrix,
    parse_number,
    parse_series,
    render_series,
)
from aionbench.report import export_eval_compare, render_markdown, summarise, uplift
from aionbench.router import EchoClient, ResponseCache


# -- prompting -------------------------------------------------------------

def test_render_series_uses_one_decimal_place():
    series = SeriesRef(name="s", values=(1.0, 2.55, -3.0))
    assert render_series(series) == "1.0, 2.5, -3.0"


def test_prompt_carries_task_id_and_options():
    task = BenchTask(
        task_id="t-1", suite="tsqa", kind="multiple_choice",
        prompt="Does it trend?", options=("Up", "Down"),
        series=(SeriesRef(name="metric", values=(1.0, 2.0, 3.0)),),
    )
    prompt = build_user_prompt(task, condition="direct")
    assert "[task_id: t-1]" in prompt
    assert "A. Up" in prompt and "B. Down" in prompt
    assert "1.0, 2.0, 3.0" in prompt


def test_aion_prompt_lists_csv_paths():
    task = BenchTask(
        task_id="t-2", suite="tsqa", kind="multiple_choice",
        prompt="?", options=("A", "B"),
        series=(SeriesRef(name="metric", values=(1.0, 2.0)),),
    )
    prompt = build_user_prompt(task, condition="aion", csv_paths={"metric": "/tmp/metric.csv"})
    assert "/tmp/metric.csv" in prompt
    assert "synthesised by the harness" in prompt


@pytest.mark.parametrize("text,expected", [
    ("<ANSWER>B</ANSWER>", 1),
    ("blah\nFinal answer: C", 2),
    ("Reasoning...\n\nD", 3),
    ("<ANSWER>Option A</ANSWER>", 0),
    ("<ANSWER>(c)</ANSWER>", 2),
])
def test_parse_choice_accepts_common_shapes(text, expected):
    assert parse_choice(text, 4) == expected


def test_parse_choice_rejects_out_of_range_and_absent():
    assert parse_choice("<ANSWER>Z</ANSWER>", 3) is None
    assert parse_choice("I am not sure about this one.", 3) is None


def test_parse_choice_does_not_pick_a_letter_beyond_the_options():
    # Four options offered, model answered E: that is a structural failure,
    # not a silent clamp onto D.
    assert parse_choice("Answer: E", 4) is None


def test_numeric_and_series_parsers():
    assert parse_number("<ANSWER>42.5</ANSWER>") == 42.5
    assert parse_number("the value works out to 17.25 in the end") == 17.25
    assert parse_series("<ANSWER>[1, 2, 3.5]</ANSWER>") == [1.0, 2.0, 3.5]
    assert parse_matrix("<ANSWER>[[1,2],[3,4]]</ANSWER>") == [[1.0, 2.0], [3.0, 4.0]]


def test_parse_matrix_rejects_ragged_rows():
    assert parse_matrix("<ANSWER>[[1,2],[3]]</ANSWER>") is None


def test_parse_code_prefers_execute_tags():
    assert parse_code("<execute>predictions = 1</execute>") == "predictions = 1"
    assert parse_code("```python\npredictions = 2\n```") == "predictions = 2"
    assert parse_code("no code here") is None


def test_abstention_detection():
    assert is_abstention("<ANSWER>ABSTAIN</ANSWER>")
    assert is_abstention("Aion reports the series is unsupported.")
    assert not is_abstention("<ANSWER>B</ANSWER>")


# -- metrics ---------------------------------------------------------------

def test_mape_skips_zero_actuals_rather_than_dividing_by_epsilon():
    assert mape([0.0, 100.0], [5.0, 110.0]) == pytest.approx(0.1)


def test_mape_is_undefined_when_every_actual_is_zero():
    import math
    assert math.isnan(mape([0.0, 0.0], [1.0, 2.0]))


def test_smape_stays_within_unit_range():
    assert 0.0 <= smape([1.0, 2.0], [-5.0, -9.0]) <= 1.0
    assert smape([10.0, 10.0], [10.0, 10.0]) == 0.0


def test_binary_f1_counts_confusion():
    score = binary_f1([1, 0, 1, 0], [1, 1, 0, 0])
    assert (score.true_positives, score.false_positives, score.false_negatives) == (1, 1, 1)
    assert score.f1 == pytest.approx(0.5)


def test_index_f1_scores_an_empty_prediction_as_zero_not_an_error():
    assert index_f1([3], [], length=10).f1 == 0.0


def test_mcnemar_is_symmetric_and_bounded():
    control = [True, False, False, False, True]
    treatment = [True, True, True, True, True]
    test = mcnemar(control, treatment)
    assert test.treatment_only == 3 and test.control_only == 0
    assert 0.0 <= test.p_value <= 1.0
    assert mcnemar(control, control).p_value == 1.0


def test_bootstrap_ci_is_deterministic_and_brackets_the_estimate():
    flags = [True] * 30 + [False] * 20
    low, high = bootstrap_ci(flags)
    assert bootstrap_ci(flags) == (low, high)
    assert low <= 0.6 <= high


# -- router ----------------------------------------------------------------

def test_response_cache_round_trips_and_survives_corruption(tmp_path):
    cache = ResponseCache(tmp_path)
    key = ResponseCache.key({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    cache.put(key, {"choices": [{"message": {"content": "ok"}}]})
    assert cache.get(key)["choices"][0]["message"]["content"] == "ok"

    (tmp_path / key[:2] / f"{key}.json").write_text("{not json", encoding="utf-8")
    assert cache.get(key) is None


def test_cache_key_changes_with_any_input():
    base = {"model": "m", "messages": [], "temperature": 0.0}
    assert ResponseCache.key(base) != ResponseCache.key({**base, "temperature": 0.7})
    assert ResponseCache.key(base) != ResponseCache.key({**base, "model": "n"})


def test_echo_client_routes_by_task_id():
    client = EchoClient(default="<ANSWER>A</ANSWER>", by_task={"t-9": "<ANSWER>C</ANSWER>"})
    messages = [{"role": "user", "content": "[task_id: t-9] question"}]
    assert client.chat("m", messages).text == "<ANSWER>C</ANSWER>"
    assert client.chat("m", [{"role": "user", "content": "no marker"}]).text == "<ANSWER>A</ANSWER>"


# -- contracts -------------------------------------------------------------

def test_attempt_is_only_cached_when_every_provider_call_was():
    attempt = Attempt(task_id="t", model="m", condition="aion")
    attempt.provider_calls, attempt.cache_hits = 3, 2
    assert attempt.cached is False
    attempt.cache_hits = 3
    assert attempt.cached is True


def test_invalid_task_kind_is_rejected():
    with pytest.raises(BenchError):
        BenchTask(task_id="t", suite="s", kind="telepathy", prompt="?")


# -- reporting -------------------------------------------------------------

def _row(task_id, condition, correct, **extra):
    row = {
        "task_id": task_id, "suite": "tsqa", "model": "m", "condition": condition,
        "repeat": 0, "kind": "multiple_choice", "metric": "accuracy",
        "category": "trend", "subcategory": "", "difficulty": "",
        "success": bool(correct), "correct": correct, "score": None,
        "failure": None if correct else "threshold", "detail": "",
        "tool_calls": 0, "tool_calls_ok": 0, "grounded": False,
        "prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.001,
        "latency_seconds": 1.0, "cached": False,
    }
    row.update(extra)
    return row


def test_summary_excludes_transport_failures_from_the_denominator():
    rows = [
        _row("a", "direct", True),
        _row("b", "direct", False),
        _row("c", "direct", None, failure="transport"),
    ]
    arm = summarise(rows)["arms"]["m::direct"]
    assert arm["graded"] == 2
    assert arm["accuracy"] == pytest.approx(0.5)
    assert arm["excluded"] == 1
    assert summarise(rows)["totals"]["transport_failures"] == 1


def test_uplift_pairs_only_items_both_arms_graded():
    rows = [
        _row("a", "direct", False), _row("a", "aion", True),
        _row("b", "direct", False), _row("b", "aion", True),
        _row("c", "direct", True), _row("c", "aion", True),
        # only the control graded this one — it must not enter the pairing
        _row("d", "direct", True), _row("d", "aion", None, failure="transport"),
    ]
    result = uplift(rows)[0]
    assert result["paired_items"] == 3
    assert result["fixed_by_treatment"] == 2
    assert result["broken_by_treatment"] == 0
    assert result["absolute_uplift"] == pytest.approx(2 / 3)


def test_uplift_reports_non_significance_for_a_single_flipped_item():
    rows = []
    for index in range(20):
        correct = index > 0
        rows.append(_row(f"t{index}", "direct", correct))
        rows.append(_row(f"t{index}", "aion", True))
    result = uplift(rows)[0]
    assert result["fixed_by_treatment"] == 1
    assert result["significant_at_05"] is False


def test_markdown_report_renders_arms_and_uplift():
    rows = [
        _row("a", "direct", False), _row("a", "aion", True),
        _row("b", "direct", True), _row("b", "aion", True),
    ]
    text = render_markdown(summarise(rows))
    assert "## Arms" in text
    assert "## Paired uplift" in text
    assert "## By category" in text


def test_export_produces_aion_eval_compare_inputs(tmp_path):
    rows = [
        _row("a", "direct", False), _row("a", "aion", True, grounded=True, tool_calls=2),
        _row("b", "direct", True), _row("b", "aion", True, grounded=False),
    ]
    baseline, treatment = export_eval_compare(rows, tmp_path, model="m")
    baseline_rows = [json.loads(line) for line in baseline.read_text().splitlines()]
    treatment_rows = [json.loads(line) for line in treatment.read_text().splitlines()]

    assert {row["task_id"] for row in baseline_rows} == {row["task_id"] for row in treatment_rows}
    # A correct treatment answer that never touched a tool is flagged, since a
    # grounded harness answering ungrounded is exactly what we are watching for.
    assert [row["invented_number"] for row in treatment_rows] == [False, True]

    from aion.agent_eval import compare_runs

    comparison = compare_runs(str(baseline), str(treatment))
    assert comparison["absolute_success_uplift"] == pytest.approx(0.5)


def test_outcome_row_is_json_serialisable_with_awkward_values():
    outcome = TaskOutcome(
        task_id="t", suite="s", model="m", condition="direct",
        expected=[1.0] * 200, answered={"a": 1},
    )
    json.dumps(outcome.to_row())
    assert len(outcome.to_row()["expected"]) == 64

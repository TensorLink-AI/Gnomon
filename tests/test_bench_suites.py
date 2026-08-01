"""Suite loading and scoring, plus a full offline run through the runner.

The Aion-condition tests exercise the real ``aion.toolspec`` runners against
real materialised CSVs — nothing about the tool surface is stubbed. Only the
provider is scripted.
"""

from __future__ import annotations

import json

import pytest

from aionbench.contracts import Attempt, BenchTask, SeriesRef
from aionbench.router import EchoClient
from aionbench.runner import RunConfig, run
from aionbench.suites import LoadOptions, ScoringContext, get_suite, normalise_conditions
from aionbench.suites.timeseriesexam import _answer_index, _normalise_category, _series_from_row
from aionbench.suites.tsaia import _flatten, _grade, _metric_for, _series_from_text, _strip_protocol_boilerplate
from aionbench.tools import AionToolbelt, materialise_series


def _attempt(text, condition="direct", **extra):
    attempt = Attempt(task_id="t", model="m", condition=condition, text=text)
    for key, value in extra.items():
        setattr(attempt, key, value)
    return attempt


# -- tsqa ------------------------------------------------------------------

def test_tsqa_smoke_corpus_loads_and_is_well_formed():
    suite = get_suite("tsqa")
    tasks = suite.load(LoadOptions())
    assert len(tasks) == 16
    for task in tasks:
        assert task.kind == "multiple_choice"
        assert 2 <= len(task.options) <= 5
        assert task.answer in "ABCDE"[:len(task.options)]
        assert task.series and all(len(item) >= 2 for item in task.series)
        assert task.category.endswith("-way")


def test_tsqa_scores_a_correct_and_an_unparseable_answer():
    suite = get_suite("tsqa")
    task = suite.load(LoadOptions())[0]

    good = suite.score(task, _attempt(f"<ANSWER>{task.answer}</ANSWER>"), ScoringContext())
    assert good.correct is True and good.failure is None

    bad = suite.score(task, _attempt("I would rather not say."), ScoringContext())
    assert bad.correct is False and bad.failure == "structural"


def test_tsqa_flags_a_correct_treatment_answer_that_never_used_a_tool():
    suite = get_suite("tsqa")
    task = suite.load(LoadOptions())[0]
    outcome = suite.score(task, _attempt(f"<ANSWER>{task.answer}</ANSWER>", condition="aion"), ScoringContext())
    assert outcome.correct is True
    assert "no Aion tool call succeeded" in outcome.detail


def test_tsqa_subsample_is_deterministic():
    suite = get_suite("tsqa")
    first = [task.task_id for task in suite.load(LoadOptions(limit=5, seed=7))]
    second = [task.task_id for task in suite.load(LoadOptions(limit=5, seed=7))]
    third = [task.task_id for task in suite.load(LoadOptions(limit=5, seed=8))]
    assert first == second
    assert first != third


def test_tsqa_rejects_an_unknown_corpus():
    from aionbench.contracts import BenchError

    with pytest.raises(BenchError, match="Unknown TSQA corpus"):
        get_suite("tsqa").load(LoadOptions(config="does-not-exist"))


def test_tsqa_reads_a_local_corpus_with_a_field_map(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({
        "qid": 7, "q": "Trending?", "choices": ["Up", "Down"], "label": "Up",
        "ts": [1, 2, 3, 4], "kind": "trend",
    }) + "\n", encoding="utf-8")

    tasks = get_suite("tsqa").load(LoadOptions(
        data_path=str(path), config="generic",
        extra={"field_map": {"id": "qid", "question": "q", "options": "choices",
                             "answer": "label", "series": "ts", "category": "kind"}},
    ))
    assert len(tasks) == 1
    assert tasks[0].task_id == "tsqa-7"
    assert tasks[0].answer == "A"          # matched by option text, not letter
    assert tasks[0].category == "2-way"


# -- timeseriesexam --------------------------------------------------------

@pytest.mark.parametrize("answer,expected", [
    ("B", 1), ("(c)", 2), ("1", 1), (2, 2), ("rising trend", 0), ("nonsense", None), (None, None),
])
def test_timeseriesexam_answer_key_accepts_every_observed_encoding(answer, expected):
    assert _answer_index(answer, ["Rising trend", "Falling trend", "No trend"]) == expected


def test_timeseriesexam_answer_key_rejects_an_out_of_range_index():
    assert _answer_index("9", ["a", "b"]) is None


@pytest.mark.parametrize("raw,expected", [
    ("Pattern Recognition", "pattern_recognition"),
    ("causality", "causality_analysis"),
    ("comparative reasoning", "similarity_analysis"),
    ("", "uncategorised"),
])
def test_timeseriesexam_categories_normalise_to_the_paper_five(raw, expected):
    assert _normalise_category(raw) == expected


def test_timeseriesexam_picks_up_single_and_paired_series():
    single = _series_from_row({"ts": [1.0, 2.0, 3.0], "ts1": [], "ts2": None})
    assert [item.name for item in single] == ["series"]

    paired = _series_from_row({"ts": [], "ts1": [1.0, 2.0], "ts2": [3.0, 4.0]})
    assert [item.name for item in paired] == ["series_1", "series_2"]


def test_timeseriesexam_scores_from_a_parquet_free_task():
    suite = get_suite("timeseriesexam")
    task = BenchTask(
        task_id="tse-1", suite="timeseriesexam", kind="multiple_choice",
        prompt="Is it stationary?", options=("Yes", "No"), answer="B",
        series=(SeriesRef(name="series", values=(1.0, 2.0, 3.0)),),
        category="pattern_recognition",
    )
    assert suite.score(task, _attempt("<ANSWER>B</ANSWER>"), ScoringContext()).correct is True
    refused = suite.score(task, _attempt("<ANSWER>ABSTAIN</ANSWER>"), ScoringContext())
    assert refused.correct is False and refused.failure == "refusal"


# -- tsaia -----------------------------------------------------------------

def test_tsaia_extracts_series_from_its_prose_data_block():
    text = (
        "The historical stock value data of TBLA for the past 5 hours is: "
        "[3.28, 3.28, 3.26, 3.19, 3.16]. "
        "The historical stock value data of PAY for the past 5 hours is: "
        "[21.44, 21.29, 21.33, 21.26, 21.23]."
    )
    series = _series_from_text(text)
    assert [item.name for item in series] == ["TBLA", "PAY"]
    assert series[0].values == (3.28, 3.28, 3.26, 3.19, 3.16)


def test_tsaia_deduplicates_series_names_so_csv_paths_do_not_collide():
    text = "data of AAA for x is: [1, 2, 3, 4]. data of AAA for y is: [5, 6, 7, 8]."
    assert [item.name for item in _series_from_text(text)] == ["AAA", "AAA_2"]


def test_tsaia_strips_the_datasets_own_codeact_instructions():
    prompt = (
        "Predict the next 9 hours.\n"
        "- Store your output in the variable called `predictions`.\n"
        "You should enclose your python code in <execute> </execute> tag."
    )
    cleaned = _strip_protocol_boilerplate(prompt)
    assert "Predict the next 9 hours." in cleaned
    assert "execute" not in cleaned and "predictions" not in cleaned


@pytest.mark.parametrize("question_type,metric", [
    ("easy_stock-future price", "mape"),
    ("anomaly detection-threshold", "f1"),
    ("VaR-confidence_level", "relative_error"),
    ("something entirely new", "mape"),
])
def test_tsaia_metric_selection_follows_task_intent(question_type, metric):
    assert _metric_for(question_type) == metric


def test_tsaia_grading_rejects_shape_mismatches_rather_than_truncating():
    score, correct, note = _grade("mape", [1.0, 2.0, 3.0], [1.0, 2.0], 0.15)
    assert correct is False and score is None
    assert "Shape mismatch" in note


def test_tsaia_grading_rejects_a_vector_answer_to_a_scalar_question():
    _, correct, note = _grade("relative_error", [10.0], [10.0, 10.0], 0.1)
    assert correct is False
    assert "single value" in note


def test_tsaia_grading_passes_and_fails_on_the_stated_bar():
    _, passed, note = _grade("mape", [100.0, 100.0], [105.0, 105.0], 0.15)
    assert passed is True and "0.15" in note
    _, failed, _ = _grade("mape", [100.0, 100.0], [150.0, 150.0], 0.15)
    assert failed is False


def test_tsaia_f1_grading_pads_a_short_answer():
    score, correct, _ = _grade("f1", [1.0, 0.0, 1.0, 0.0], [1.0], 0.5)
    assert score == pytest.approx(2 / 3)
    assert correct is True


def test_tsaia_metric_defers_to_the_ground_truths_shape():
    from aionbench.suites.tsaia import _reconcile_metric

    # A task type that looked like forecasting but keys to a single value.
    assert _reconcile_metric("mape", [42.0]) == "relative_error"
    # A task type that looked scalar but keys to a horizon.
    assert _reconcile_metric("relative_error", [1.0, 2.0, 3.0]) == "mape"
    # F1 and accuracy are never rewritten.
    assert _reconcile_metric("f1", [1.0]) == "f1"


def test_tsaia_flatten_handles_nesting_and_mappings():
    assert _flatten([[1, 2], [3]]) == [1.0, 2.0, 3.0]
    assert _flatten({"a": 1, "b": [2, 3]}) == [1.0, 2.0, 3.0]
    assert _flatten(4) == [4.0]
    assert _flatten(None) == []


def test_tsaia_refuses_to_grade_analysis_items_without_their_pickled_key():
    suite = get_suite("tsaia")
    task = BenchTask(
        task_id="tsaia-aq-1", suite="tsaia", kind="series", prompt="Predict.",
        target=None, metric="mape", metadata={"config": "analysis_questions"},
    )
    outcome = suite.score(task, _attempt("<ANSWER>[1,2,3]</ANSWER>"), ScoringContext())
    assert outcome.correct is None
    assert "--allow-pickle" in outcome.detail


def test_tsaia_code_condition_reports_a_disabled_sandbox_as_an_execution_failure():
    suite = get_suite("tsaia")
    task = BenchTask(
        task_id="tsaia-aq-2", suite="tsaia", kind="series", prompt="Predict.",
        target=[1.0, 2.0], metric="mape", metadata={"config": "analysis_questions"},
    )
    outcome = suite.score(
        task, _attempt("<execute>predictions = [1, 2]</execute>", condition="code"), ScoringContext(),
    )
    assert outcome.correct is False and outcome.failure == "execution"
    assert "--allow-code-execution" in outcome.detail


# -- conditions ------------------------------------------------------------

def test_conditions_default_to_the_suites_control_and_treatment():
    assert normalise_conditions(get_suite("tsqa"), None) == ("direct", "aion")


def test_unsupported_condition_is_rejected_with_the_supported_list():
    from aionbench.contracts import BenchError

    with pytest.raises(BenchError, match="does not support"):
        normalise_conditions(get_suite("tsqa"), ["code"])


# -- tools -----------------------------------------------------------------

def test_materialised_csv_has_a_regular_index_aion_accepts(tmp_path):
    series = SeriesRef(name="metric", values=tuple(50.0 + index for index in range(40)))
    paths = materialise_series([series], tmp_path)
    lines = (tmp_path / "metric.csv").read_text().strip().splitlines()
    assert lines[0] == "timestamp,value"
    assert len(lines) == 41

    from aion.runtime import inspect_dataset

    report = inspect_dataset(
        paths["metric"], time_column="timestamp", target_column="value", frequency="D",
    )
    assert report["status"] == "valid"
    assert report["series"][0]["observations"] == 40
    assert report["data_quality"]["status"] == "clean"


def test_materialise_rejects_a_frequency_it_cannot_make_regular(tmp_path):
    from aionbench.contracts import BenchError

    with pytest.raises(BenchError, match="regular index"):
        materialise_series([SeriesRef(name="m", values=(1.0, 2.0), frequency="MS")], tmp_path)


def test_toolbelt_exposes_the_analysis_profile_as_function_definitions():
    belt = AionToolbelt(profile="analysis")
    definitions = belt.definitions()
    names = [item["function"]["name"] for item in definitions]
    assert "aion_forecast" in names and "aion_detect_anomalies" in names
    assert all(item["type"] == "function" for item in definitions)
    assert all("parameters" in item["function"] for item in definitions)


def test_toolbelt_rejects_an_unknown_profile():
    from aionbench.contracts import BenchError

    with pytest.raises(BenchError, match="Unknown tool profile"):
        AionToolbelt(profile="everything")


def test_toolbelt_records_an_unknown_tool_as_a_failed_call_not_an_exception():
    call = AionToolbelt(profile="forecast").call("aion_teleport", {})
    assert call.ok is False
    assert "No such tool" in (call.error or "")


def test_toolbelt_runs_a_real_forecast_through_aions_own_runner(tmp_path):
    values = tuple(100.0 + 0.8 * index for index in range(60))
    paths = materialise_series([SeriesRef(name="metric", values=values)], tmp_path)
    belt = AionToolbelt(profile="analysis", workdir=tmp_path)
    call = belt.call("aion_forecast", {
        "input": paths["metric"], "time_column": "timestamp",
        "target_column": "value", "frequency": "D", "horizon": 5,
    })
    assert call.ok is True
    payload = json.loads(getattr(call, "rendered"))
    assert payload["results"][0]["support"] in {
        "supported", "weakly_supported", "degraded", "unsupported", "insufficient_history",
    }


def test_toolbelt_surfaces_aions_typed_error_for_bad_input(tmp_path):
    call = AionToolbelt(profile="forecast").call("aion_forecast", {
        "input": str(tmp_path / "missing.csv"), "time_column": "timestamp",
        "target_column": "value", "horizon": 3,
    })
    assert call.ok is False
    assert call.error


# -- runner ----------------------------------------------------------------

def test_full_run_grades_both_arms_and_writes_a_report(tmp_path):
    suite = get_suite("tsqa")
    tasks = suite.load(LoadOptions(limit=4, seed=3))
    answers = {task.task_id: f"<ANSWER>{task.answer}</ANSWER>" for task in tasks}

    # Control gets everything wrong; treatment gets everything right. The
    # point is the plumbing, not the numbers.
    wrong = {task.task_id: "<ANSWER>A</ANSWER>" if task.answer != "A" else "<ANSWER>B</ANSWER>"
             for task in tasks}

    config = RunConfig(
        suite="tsqa", models=("echo/echo",), conditions=("direct",),
        output_dir=str(tmp_path / "run"), resume=False,
    )
    control = run(suite, EchoClient(by_task=wrong), config, tasks=tasks)
    assert all(row["correct"] is False for row in control.rows)

    config.conditions = ("direct",)
    config.output_dir = str(tmp_path / "run2")
    treatment = run(suite, EchoClient(by_task=answers), config, tasks=tasks)
    assert all(row["correct"] is True for row in treatment.rows)

    from aionbench.report import write_report

    summary = write_report(treatment.run_dir, treatment.rows)
    assert summary["arms"]["echo/echo::direct"]["accuracy"] == 1.0
    assert (treatment.run_dir / "report.md").is_file()
    assert (treatment.run_dir / "manifest.json").is_file()
    assert json.loads((treatment.run_dir / "manifest.json").read_text())["cells_completed"] == 4


def test_run_resumes_without_repeating_completed_cells(tmp_path):
    suite = get_suite("tsqa")
    tasks = suite.load(LoadOptions(limit=3, seed=5))
    config = RunConfig(
        suite="tsqa", models=("echo/echo",), conditions=("direct",),
        output_dir=str(tmp_path / "run"), resume=True,
    )
    client = EchoClient(default="<ANSWER>A</ANSWER>")
    run(suite, client, config, tasks=tasks)
    first_pass_calls = len(client.calls)

    second = run(suite, client, config, tasks=tasks)
    assert len(client.calls) == first_pass_calls    # nothing re-requested
    assert len(second.rows) == 3


def test_run_stops_scheduling_once_the_budget_is_spent(tmp_path, monkeypatch):
    suite = get_suite("tsqa")
    tasks = suite.load(LoadOptions(limit=8, seed=2))

    from aionbench import agentloop

    original = agentloop.run_attempt

    def costly(*args, **kwargs):
        attempt = original(*args, **kwargs)
        attempt.cost_usd = 1.0
        return attempt

    monkeypatch.setattr(agentloop, "run_attempt", costly)
    monkeypatch.setattr("aionbench.runner.run_attempt", costly)

    config = RunConfig(
        suite="tsqa", models=("echo/echo",), conditions=("direct",),
        output_dir=str(tmp_path / "run"), resume=False, budget_usd=3.0, concurrency=1,
    )
    result = run(suite, EchoClient(), config, tasks=tasks)
    assert result.stopped_early is True
    assert len(result.rows) < 8


def test_a_crashing_cell_is_recorded_rather_than_killing_the_run(tmp_path, monkeypatch):
    suite = get_suite("tsqa")
    tasks = suite.load(LoadOptions(limit=2, seed=4))

    def explode(*args, **kwargs):
        raise RuntimeError("provider melted")

    monkeypatch.setattr("aionbench.runner.run_attempt", explode)
    config = RunConfig(
        suite="tsqa", models=("echo/echo",), conditions=("direct",),
        output_dir=str(tmp_path / "run"), resume=False, concurrency=1,
    )
    result = run(suite, EchoClient(), config, tasks=tasks)
    assert len(result.rows) == 2
    assert all(row["failure"] == "transport" and row["correct"] is None for row in result.rows)


def test_run_uses_the_real_aion_toolbelt_in_the_treatment_arm(tmp_path):
    suite = get_suite("tsqa")
    tasks = suite.load(LoadOptions(limit=1, seed=11))
    task = tasks[0]

    materialised = materialise_series(task.series, tmp_path / "seed")
    first_path = next(iter(materialised.values()))
    script = {task.task_id: [{
        "id": "call_1", "type": "function",
        "function": {"name": "aion_inspect", "arguments": json.dumps({
            "input": first_path, "time_column": "timestamp",
            "target_column": "value", "frequency": "D",
        })},
    }]}

    client = EchoClient(default=f"<ANSWER>{task.answer}</ANSWER>", tool_script=script)
    config = RunConfig(
        suite="tsqa", models=("echo/echo",), conditions=("aion",),
        output_dir=str(tmp_path / "run"), resume=False,
    )
    result = run(suite, client, config, tasks=tasks)
    row = result.rows[0]
    assert row["tool_calls"] == 1 and row["tool_calls_ok"] == 1
    assert row["grounded"] is True and row["correct"] is True

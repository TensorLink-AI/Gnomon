"""Corpus loading, exercised against local fixtures rather than the network.

Every suite accepts ``--data-path``; these tests use it, so a corpus moving
on Hugging Face cannot break CI, and a regression in the row-shape handling
still gets caught.
"""

from __future__ import annotations

import json

import pytest

from aionbench.contracts import BenchError
from aionbench.datasets import read_table
from aionbench.suites import LoadOptions, get_suite


# -- timeseriesexam --------------------------------------------------------

def _tse_row(**overrides):
    row = {
        "id": 1,
        "question": "Is this series stationary?",
        "options": ["Yes", "No"],
        "answer": "B",
        "question_type": "stationarity",
        "ts": [1.0, 2.0, 3.0, 4.0],
        "ts1": [],
        "ts2": [],
        "tid": 12,
        "difficulty": "medium",
        "format_hint": "",
        "relevant_concepts": ["A series is stationary when its mean is constant."],
        "question_hint": "Look at the mean over time.",
        "category": "Pattern Recognition",
        "subcategory": "trend",
    }
    row.update(overrides)
    return row


def test_timeseriesexam_loads_from_a_local_file(tmp_path):
    path = tmp_path / "exam.json"
    path.write_text(json.dumps([
        _tse_row(),
        _tse_row(id=2, category="Causality", ts=[], ts1=[1.0, 2.0], ts2=[3.0, 4.0]),
    ]), encoding="utf-8")

    tasks = get_suite("timeseriesexam").load(LoadOptions(data_path=str(path)))
    assert [task.task_id for task in tasks] == ["tse-1", "tse-2"]
    assert tasks[0].category == "pattern_recognition"
    assert tasks[0].answer == "B"
    assert tasks[0].hint and tasks[0].concepts
    assert tasks[1].category == "causality_analysis"
    assert len(tasks[1].series) == 2


def test_timeseriesexam_drops_rows_it_cannot_key_or_ground(tmp_path):
    path = tmp_path / "exam.json"
    path.write_text(json.dumps([
        _tse_row(id=1, answer="not an option"),   # unresolvable key
        _tse_row(id=2, options=["Only one"]),      # not a choice question
        _tse_row(id=3, ts=[], ts1=[], ts2=[]),     # no series to reason about
        _tse_row(id=4),                            # keeper
    ]), encoding="utf-8")

    tasks = get_suite("timeseriesexam").load(LoadOptions(data_path=str(path)))
    assert [task.task_id for task in tasks] == ["tse-4"]


def test_timeseriesexam_category_filter_and_empty_filter_error(tmp_path):
    path = tmp_path / "exam.json"
    path.write_text(json.dumps([
        _tse_row(id=1, category="Pattern Recognition"),
        _tse_row(id=2, category="Causality"),
    ]), encoding="utf-8")

    suite = get_suite("timeseriesexam")
    filtered = suite.load(LoadOptions(data_path=str(path), categories=("causality_analysis",)))
    assert [task.task_id for task in filtered] == ["tse-2"]

    with pytest.raises(BenchError, match="No tasks matched"):
        suite.load(LoadOptions(data_path=str(path), categories=("nonexistent",)))


def test_timeseriesexam_reads_parquet_when_pyarrow_is_present(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "exam.parquet"
    pq.write_table(pa.Table.from_pylist([_tse_row()]), path)
    assert read_table(path)[0]["question"] == "Is this series stationary?"

    tasks = get_suite("timeseriesexam").load(LoadOptions(data_path=str(path)))
    assert len(tasks) == 1 and tasks[0].answer == "B"


# -- tsaia -----------------------------------------------------------------

def test_tsaia_multiple_choice_loads_from_a_local_csv(tmp_path):
    import csv

    path = tmp_path / "mc.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "question_id", "question_type", "prompt", "options", "answer",
            "data_info", "answer_info", "executor_variables",
        ])
        writer.writeheader()
        writer.writerow({
            "question_id": "0",
            "question_type": "VaR-confidence_level",
            "prompt": (
                "Which portfolio has the lowest VaR? "
                "Store your output in the variable called `predictions`."
            ),
            "options": "['Option A: TBLA', 'Option B: PAY', 'Option C: DCOM']",
            "answer": "A",
            "data_info": (
                "The historical stock value data of TBLA for the past 5 hours is: "
                "[3.28, 3.28, 3.26, 3.19, 3.16]. "
                "The historical stock value data of PAY for the past 5 hours is: "
                "[21.44, 21.29, 21.33, 21.26, 21.23]."
            ),
            "answer_info": "external_data_mc/answer_info/answer_info_0.pkl",
            "executor_variables": "external_data_mc/executor_variables/executor_variables_0.pkl",
        })

    tasks = get_suite("tsaia").load(LoadOptions(data_path=str(path), config="multiple_choice"))
    assert len(tasks) == 1
    task = tasks[0]
    assert task.task_id == "tsaia-mc-0"
    assert task.answer == "A" and len(task.options) == 3
    assert task.category == "financial_decision"
    assert [item.name for item in task.series] == ["TBLA", "PAY"]
    # The dataset's own CodeAct instructions must not survive into a
    # non-code condition's prompt.
    assert "predictions" not in task.prompt


def test_tsaia_analysis_questions_load_without_a_key_when_pickles_are_refused(tmp_path):
    import csv

    path = tmp_path / "aq.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "question_id", "question_type", "prompt", "data_str",
            "executor_variables", "ground_truth_data", "context", "constraint",
        ])
        writer.writeheader()
        writer.writerow({
            "question_id": "0",
            "question_type": "easy_stock-future price",
            "prompt": "Predict the next 9 hours.",
            "data_str": "The historical stock value data of PX for the past 5 hours is: [10.0, 10.1, 10.0, 9.9, 10.0].",
            "executor_variables": "external_data/executor_variables/executor_variables_0.pkl",
            "ground_truth_data": "external_data/ground_truth_data/ground_truth_data_0.pkl",
            "context": "external_data/context/context_0.pkl",
            "constraint": "external_data/constraint/constraint_0.pkl",
        })

    tasks = get_suite("tsaia").load(LoadOptions(data_path=str(path), config="analysis_questions"))
    assert len(tasks) == 1
    task = tasks[0]
    assert task.metric == "mape" and task.kind == "series"
    assert task.category == "predictive"
    # Nothing was downloaded or unpickled, so the item loads but cannot be graded.
    assert task.target is None
    assert task.metadata["ground_truth_loaded"] is False


def test_tsaia_rejects_an_unknown_config():
    with pytest.raises(BenchError, match="Unknown TSAIA config"):
        get_suite("tsaia").load(LoadOptions(config="nope"))


# -- datasets --------------------------------------------------------------

def test_read_table_dispatches_on_extension(tmp_path):
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert read_table(jsonl) == [{"a": 1}, {"a": 2}]

    wrapped = tmp_path / "rows.json"
    wrapped.write_text(json.dumps({"data": [{"a": 3}]}), encoding="utf-8")
    assert read_table(wrapped) == [{"a": 3}]

    with pytest.raises(BenchError, match="unsupported extension"):
        read_table(tmp_path / "rows.xyz")


def test_malformed_jsonl_names_the_offending_line(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n{oops\n', encoding="utf-8")
    with pytest.raises(BenchError, match="rows.jsonl:2"):
        read_table(path)


def test_pickle_loading_is_refused_without_consent(tmp_path):
    from aionbench.datasets import load_pickle

    path = tmp_path / "asset.pkl"
    path.write_bytes(b"\x80\x04.")
    with pytest.raises(BenchError, match="--allow-pickle"):
        load_pickle(path, allowed=False)

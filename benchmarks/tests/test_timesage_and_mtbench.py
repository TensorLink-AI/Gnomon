"""Unit tests for the TimeSage-MT and MTBench adapters' pure logic.

Network, the OpenRouter client, and the official benchmark checkouts are
not touched; only deterministic conversion and scoring layers are
covered.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.mtbench.aion_forecaster import (
    OFFICIAL_MSE_FAILURE_LIMIT,
    load_samples,
    official_mape,
    write_bar_csv,
)
from benchmarks.timesage_mt.scoring import numbers_in, score_mechanical, score_turn
from benchmarks.timesage_mt.tasks import TimeSageTask, load_tasks, read_visible_series


def test_numbers_in_handles_formats():
    text = "CV is 0.81, count 1,234 and drift -3.5e2; period 24."
    values = numbers_in(text)
    assert 0.81 in values
    assert 1234.0 in values
    assert -350.0 in values
    assert 24.0 in values


def test_mechanical_keyword_requires_all():
    verify = {"type": "keyword", "keywords": ["24", "daily"], "range": None}
    assert score_mechanical(verify, "A strong DAILY cycle with period 24.") is True
    assert score_mechanical(verify, "The period is 24 hours.") is False


def test_mechanical_numerical_range():
    verify = {"type": "numerical_range", "keywords": [], "range": [0.7, 1.1]}
    assert score_mechanical(verify, "The CV comes out to 0.81.") is True
    assert score_mechanical(verify, "The CV comes out to 1.5.") is False


def test_non_mechanical_spec_is_unscored_without_judge():
    reference = {"finding_verify": {"type": "semantic", "keywords": [],
                                    "range": None, "embedding_threshold": 0.85}}
    verdict = score_turn(reference, "some answer", judge_client=None)
    assert verdict == {"scored": False, "passed": None, "basis": "needs_judge"}
    assert score_turn({}, "answer")["basis"] == "no_verify_spec"


def _write_timesage_fixture(root: Path) -> None:
    task = {
        "id": "L1_general_001",
        "tier": "L1",
        "domain": None,
        "dialogue": [
            {"turn_id": 1, "role": "user", "text": "Profile this series."},
            {"turn_id": 2, "role": "agent", "text": "ref",
             "key_finding": "CV about 0.8",
             "finding_verify": {"type": "numerical_range", "keywords": [],
                                "range": [0.7, 1.1]}},
        ],
        "visibility_contract": {"rows_visible": 3},
    }
    tier_dir = root / "MT_Bench" / "L1"
    tier_dir.mkdir(parents=True)
    (tier_dir / "L1_general_001.json").write_text(json.dumps(task))
    csv_dir = root / "visible_ts" / "L1" / "L1_general_001" / "agent_input"
    csv_dir.mkdir(parents=True)
    (csv_dir / "visible.csv").write_text(
        "date,ride_count\n"
        "2023-01-01 00:00:00,26\n"
        "2023-01-01 01:00:00,78\n"
        "2023-01-01 02:00:00,50\n"
    )


def test_load_tasks_and_visible_series(tmp_path):
    _write_timesage_fixture(tmp_path)
    tasks = load_tasks(tmp_path)
    assert len(tasks) == 1
    task = tasks[0]
    assert isinstance(task, TimeSageTask)
    assert [t["turn_id"] for t in task.user_turns] == [1]
    reference = task.reference_turn_after(1)
    assert reference and reference["turn_id"] == 2
    timestamps, columns, text = read_visible_series(task)
    assert len(timestamps) == 3
    assert columns["ride_count"] == [26.0, 78.0, 50.0]
    assert "ride_count" in text


def test_toolbox_series_stats_and_unknown_column(tmp_path):
    _write_timesage_fixture(tmp_path)
    from benchmarks.timesage_mt.harness import ToolBox

    toolbox = ToolBox(load_tasks(tmp_path)[0])
    stats = toolbox.call("series_stats", {"column": "ride_count"})
    assert stats["count"] == 3
    assert round(stats["mean"], 2) == 51.33
    assert "error" in toolbox.call("series_stats", {"column": "nope"})
    assert "error" in toolbox.call("no_such_tool", {})


def test_official_mape_fallback_masks_zeros():
    assert official_mape([0.0, 10.0], [5.0, 11.0]) == 10.0
    assert official_mape([2.0, 4.0], [2.0, 4.0]) == 0.0


def test_mtbench_sample_loading_and_bar_axis(tmp_path):
    sample = {
        "input_window": [1.0, 2.0, 3.0],
        "output_window": [4.0, 5.0],
        "input_timestamps": [1600000000, 1600086400, 1600172800],
        "text": {"content": "some news"},
    }
    (tmp_path / "s1.json").write_text(json.dumps(sample))
    samples = load_samples(tmp_path)
    assert samples[0]["text"] == "some news"
    csv_path = tmp_path / "bars.csv"
    start, end = write_bar_csv(samples[0]["input_window"], csv_path)
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0] == "timestamp,value"
    assert len(lines) == 4
    assert start.startswith("2020-01-01") and end.startswith("2020-01-03")
    assert "+00:00" in lines[1]
    assert OFFICIAL_MSE_FAILURE_LIMIT == 100.0

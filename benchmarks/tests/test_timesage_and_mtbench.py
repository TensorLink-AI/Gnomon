"""Unit tests for the TimeSage-MT and MTBench adapters' pure logic.

Network, the OpenRouter client, and the official benchmark checkouts are
not touched; only deterministic conversion and scoring layers are
covered.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.mtbench.gnomon_forecaster import (
    OFFICIAL_MSE_FAILURE_LIMIT,
    load_samples,
    official_mape,
    write_bar_csv,
)
from benchmarks.mtbench.run_mtbench import materialize_official_json_view
from benchmarks.timesage_mt.scoring import numbers_in, score_mechanical, score_turn
from benchmarks.timesage_mt.tasks import TimeSageTask, load_tasks, read_visible_series


def test_numbers_in_handles_formats():
    text = "CV is 0.81, count 1,234 and drift -3.5e2; period 24."
    values = numbers_in(text)
    assert 0.81 in values
    assert 1234.0 in values
    assert -350.0 in values
    assert 24.0 in values


def test_numbers_in_excludes_date_and_time_tokens():
    text = ("Peak on 2026-01-02 (also written 2026/01/02 or 02/01/2026) "
            "at 12:30:05, ISO 2026-01-02T12:30:05+00:00 and "
            "2026-01-02 12:30:05Z; value 42 and CV 0.81.")
    assert numbers_in(text) == [42.0, 0.81]


def test_numerical_range_not_satisfied_by_timestamps():
    verify = {"type": "numerical_range", "keywords": [],
              "range": [2000.0, 2100.0]}
    assert score_mechanical(
        verify, "The anomaly is at 2026-01-02 03:00:00.") is False
    # A bare number of the same magnitude still counts.
    assert score_mechanical(verify, "The peak value is 2026.") is True


def test_mechanical_keyword_requires_all():
    verify = {"type": "keyword", "keywords": ["24", "daily"], "range": None}
    assert score_mechanical(verify, "A strong DAILY cycle with period 24.") is True
    assert score_mechanical(verify, "The period is 24 hours.") is False


def test_mechanical_numerical_range():
    verify = {"type": "numerical_range", "keywords": [], "range": [0.7, 1.1]}
    assert score_mechanical(verify, "The CV comes out to 0.81.") is True
    assert score_mechanical(verify, "The CV comes out to 1.5.") is False


def test_range_score_discloses_numerosity_ambiguity():
    reference = {"finding_verify": {
        "type": "numerical_range", "keywords": [], "range": [0.7, 1.1]}}
    clean = score_turn(reference, "CV 0.81")
    noisy = score_turn(reference, "Across 24 windows: 9, 8, 7, 0.81")
    assert clean["passed"] is True and clean["numerosity_robust"] is True
    assert noisy["passed"] is True and noisy["numerosity_robust"] is False
    assert noisy["numeric_candidate_count"] == 5


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


def _fallback_task(rows, rows_visible=None):
    """Synthetic task with no per-task CSV, exercising the
    time_series_json fallback in read_visible_series."""
    return TimeSageTask(
        task_id="L1_fallback_001", tier="L1", domain=None,
        dialogue=[{"turn_id": 1, "role": "user", "text": "Profile this."}],
        visibility=(
            {"rows_visible": rows_visible} if rows_visible is not None else {}
        ),
        visible_csv=None,
        raw={"time_series_json": json.dumps(rows)},
    )


def test_fallback_series_enforces_visibility_contract():
    rows = [{"date": f"2023-01-{day:02d} 00:00:00", "value": float(day)}
            for day in range(1, 7)]
    task = _fallback_task(rows, rows_visible=3)
    timestamps, columns, text = read_visible_series(task)
    # Post-horizon rows never reach the tools...
    assert len(timestamps) == 3
    assert columns["value"] == [1.0, 2.0, 3.0]
    # ...and the prompt text is exactly the rows the tools see.
    assert json.loads(text) == rows[:3]
    assert "2023-01-04" not in text


def test_fallback_series_without_contract_keeps_prompt_tool_parity():
    rows = [{"date": f"2023-03-{1 + i // 24:02d} {i % 24:02d}:00:00",
             "value": float(i)} for i in range(72)]
    timestamps, columns, text = read_visible_series(_fallback_task(rows))
    assert len(timestamps) == 72
    assert columns["value"][-1] == 71.0
    # No hidden cap: the serialized prompt payload matches the tool rows.
    assert json.loads(text) == rows


def test_toolbox_reports_prompt_csv_truncation():
    from benchmarks.timesage_mt.harness import MAX_CSV_CHARS, ToolBox

    small = ToolBox(_fallback_task(
        [{"date": "2023-01-01 00:00:00", "value": 1.0}]))
    assert small.csv_truncated is False
    assert small.bounded_csv() == small.csv_text

    rows = [{"date": f"2023-{1 + i // 28:02d}-{1 + i % 28:02d} 00:00:00",
             "value": float(i)} for i in range(2000)]
    big = ToolBox(_fallback_task(rows))
    assert len(big.csv_text) > MAX_CSV_CHARS
    assert big.csv_truncated is True
    assert big.bounded_csv().endswith("(truncated for context)")
    # The tools still see the full series — the disclosed confound.
    assert len(big.columns["value"]) == 2000


def test_run_dialogue_records_carry_truncation_flag(tmp_path):
    _write_timesage_fixture(tmp_path)
    from benchmarks.timesage_mt.harness import run_dialogue

    class _StubMessage:
        content = "All quiet."
        tool_calls = None

    class _StubChoice:
        message = _StubMessage()

    class _StubResponse:
        choices = [_StubChoice()]

    class _StubClient:
        def chat(self, messages, n=1, tools=None, tool_choice=None):
            return _StubResponse()

    records = run_dialogue(load_tasks(tmp_path)[0], _StubClient(),
                           condition="direct")
    assert len(records) == 1
    assert records[0]["response"] == "All quiet."
    assert records[0]["csv_truncated"] is False


def test_failed_task_turns_keyed_like_scored_turns():
    from benchmarks.timesage_mt.run_timesage import scorable_turns_on_failure

    task = TimeSageTask(
        task_id="L2_x_007", tier="L2", domain=None,
        dialogue=[
            {"turn_id": 1, "role": "user", "text": "q1"},
            {"turn_id": 2, "role": "agent",
             "finding_verify": {"type": "numerical_range", "keywords": [],
                                "range": [0.0, 1.0]}},
            {"turn_id": 3, "role": "user", "text": "q2"},
            {"turn_id": 4, "role": "agent",
             "finding_verify": {"type": "semantic", "keywords": [],
                                "range": None, "embedding_threshold": 0.9}},
            {"turn_id": 5, "role": "user", "text": "q3"},
            {"turn_id": 6, "role": "agent", "text": "no verify spec"},
        ],
        visibility={}, visible_csv=None, raw={},
    )
    # Turn keys match the scored-turn scheme, so failed tasks stay in
    # the denominator and join per-turn against the other arm.
    assert scorable_turns_on_failure(task, judge_available=False) == [
        ("L2_x_007-turn1", "mechanical"),
        ("L2_x_007-turn3", "unscored"),
    ]
    assert scorable_turns_on_failure(task, judge_available=True) == [
        ("L2_x_007-turn1", "mechanical"),
        ("L2_x_007-turn3", "judge"),
    ]


def test_timesage_usage_aggregation_preserves_requests_and_provenance():
    from benchmarks.timesage_mt.run_timesage import _sum_usage

    combined = _sum_usage([
        {"model": "deepseek", "base_url": "https://example.test/v1",
         "requests": 2, "transport_attempts": 3, "prompt_tokens": 100,
         "completion_tokens": 20, "cost_usd": 0.1,
         "truncation_escalations": 1},
        {"model": "deepseek", "base_url": "https://example.test/v1",
         "requests": 4, "transport_attempts": 4, "prompt_tokens": 300,
         "completion_tokens": 40, "cost_usd": 0.2,
         "truncation_escalations": 0},
    ])

    assert combined == {
        "model": "deepseek", "base_url": "https://example.test/v1",
        "requests": 6, "transport_attempts": 7, "prompt_tokens": 400,
        "completion_tokens": 60, "cost_usd": 0.3,
        "truncation_escalations": 1,
    }


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


def test_mtbench_materializes_official_parquet_for_unmodified_scorer(tmp_path):
    # Parquet support belongs to the optional external-benchmark environment,
    # not Gnomon's zero-dependency runtime. The JSON path remains covered in
    # the base matrix; exercise the official parquet bridge when its reader is
    # actually installed.
    pd = pytest.importorskip("pandas")

    source = tmp_path / "download" / "data"
    source.mkdir(parents=True)
    frame = pd.DataFrame([
        {"input_window": [1.0, 2.0], "output_window": [3.0],
         "input_timestamps": [1, 2],
         "text": json.dumps({"content": "first"}),
         "technical": json.dumps({"in_macd": [0.1], "out_macd": [0.2]})},
        {"input_window": [4.0, 5.0], "output_window": [6.0],
         "input_timestamps": [3, 4],
         "text": json.dumps({"content": "second"}),
         "technical": json.dumps({"in_macd": [0.3], "out_macd": [0.4]})},
    ])
    frame.to_parquet(source / "tasks.parquet")
    output = tmp_path / "json-view"
    assert materialize_official_json_view(
        source.parent, output, limit=1) == 1
    rows = list(output.glob("*.json"))
    assert len(rows) == 1
    task = json.loads(rows[0].read_text())
    assert task["text"]["content"] == "first"
    assert task["technical"]["out_macd"] == [0.2]


def _write_multi_tier_fixture(root: Path, counts: dict) -> None:
    for tier, count in counts.items():
        tier_dir = root / "MT_Bench" / tier
        tier_dir.mkdir(parents=True)
        for number in range(1, count + 1):
            task = {
                "id": f"{tier}_general_{number:03d}", "tier": tier,
                "domain": None, "dialogue": [], "visibility_contract": {},
            }
            (tier_dir / f"{tier}_general_{number:03d}.json").write_text(
                json.dumps(task))


def test_limit_is_stratified_across_tiers(tmp_path):
    """tasks[:limit] on a tier-sorted list made --limit silently defeat
    --tiers: every limited run was L1-only. The limit must sample every
    requested tier."""
    _write_multi_tier_fixture(
        tmp_path, {"L1": 5, "L2": 5, "L3": 5, "L4": 5})
    tasks = load_tasks(tmp_path, limit=6)
    assert len(tasks) == 6
    by_tier = {}
    for task in tasks:
        by_tier[task.tier] = by_tier.get(task.tier, 0) + 1
    assert set(by_tier) == {"L1", "L2", "L3", "L4"}
    # extra slots go to the earliest tiers; each tier keeps sorted order
    assert by_tier == {"L1": 2, "L2": 2, "L3": 1, "L4": 1}
    assert [task.task_id for task in tasks if task.tier == "L1"] == \
        ["L1_general_001", "L1_general_002"]
    # deterministic: same inputs, same selection
    assert [task.task_id for task in tasks] == \
        [task.task_id for task in load_tasks(tmp_path, limit=6)]


def test_limit_larger_than_a_tier_spills_to_the_others(tmp_path):
    _write_multi_tier_fixture(tmp_path, {"L1": 2, "L2": 8})
    tasks = load_tasks(tmp_path, limit=7)
    by_tier = {}
    for task in tasks:
        by_tier[task.tier] = by_tier.get(task.tier, 0) + 1
    assert by_tier == {"L1": 2, "L2": 5}


def test_limit_of_zero_or_none_keeps_everything(tmp_path):
    _write_multi_tier_fixture(tmp_path, {"L1": 3, "L2": 3})
    assert len(load_tasks(tmp_path)) == 6
    assert len(load_tasks(tmp_path, limit=None)) == 6
    assert len(load_tasks(tmp_path, limit=100)) == 6

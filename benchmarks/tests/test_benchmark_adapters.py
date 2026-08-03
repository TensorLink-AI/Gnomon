"""Unit tests for the benchmark adapters' pure logic.

These run without the external benchmark packages, without network, and
without numpy/pandas: only the deterministic conversion layers are
covered here. The faithfulness-critical code paths (tasks, prompts,
metrics) belong to the official benchmark packages and are deliberately
not reimplemented, so they are not tested here either.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.anomllm.gnomon_detector import (
    binary_f1,
    flagged_indices_to_intervals,
    intervals_to_vector,
)
from benchmarks.cik.gnomon_forecaster import (
    GnomonAbstained,
    events_from_proposals,
    samples_from_quantile_rows,
)
from benchmarks.common.openrouter import extract_json_array
from benchmarks.common.records import RecordWriter, RunRecord


def test_quantile_samples_shape_and_median():
    rows = [
        {"point": 10.0, "q10": 8.0, "q50": 10.0, "q90": 14.0},
        {"point": 20.0, "q10": 15.0, "q50": 20.0, "q90": 21.0},
    ]
    samples = samples_from_quantile_rows(rows, 25)
    assert len(samples) == 25
    assert all(len(trajectory) == 2 for trajectory in samples)
    median = samples[12]  # probability (12 + 0.5) / 25 = 0.5
    assert median == [10.0, 20.0]


def test_quantile_samples_clamped_tails_and_order():
    rows = [{"point": 5.0, "q10": 2.0, "q50": 5.0, "q90": 9.0}]
    samples = samples_from_quantile_rows(rows, 100)
    values = [trajectory[0] for trajectory in samples]
    assert min(values) == 2.0 and max(values) == 9.0
    assert values == sorted(values)


def test_quantile_samples_handles_missing_quantiles():
    samples = samples_from_quantile_rows([{"point": 3.0}], 5)
    assert all(trajectory == [3.0] for trajectory in samples)


def test_extract_json_array_from_prose_and_fences():
    text = 'Sure! Here it is:\n```json\n[{"event_type": "outage"}]\n```\nDone.'
    assert extract_json_array(text) == [{"event_type": "outage"}]
    assert extract_json_array("no array here [1, 2, 3] trailing") == [1, 2, 3]


def test_extract_json_array_skips_invalid_candidates():
    text = "[not json] but later [\"valid\"]"
    assert extract_json_array(text) == ["valid"]


def test_events_from_proposals_validates_and_filters():
    window_start = "2024-01-01T00:00:00+00:00"
    window_end = "2024-03-01T00:00:00+00:00"
    proposals = [
        {  # valid
            "event_type": "maintenance_window",
            "effective_start": "2024-02-01T00:00:00+00:00",
            "effective_end": "2024-02-02T00:00:00+00:00",
            "confidence": 0.9,
            "rationale": "the context announces maintenance",
        },
        {  # timezone-naive: must be rejected by the gnomon contract
            "event_type": "bad_tz",
            "effective_start": "2024-02-01T00:00:00",
            "effective_end": "2024-02-02T00:00:00",
        },
        {  # outside the task window
            "event_type": "too_late",
            "effective_start": "2025-01-01T00:00:00+00:00",
            "effective_end": "2025-01-02T00:00:00+00:00",
        },
        "not-an-object",
    ]
    events, notes = events_from_proposals(
        proposals,
        task_name="DemoTask",
        known_at=window_start,
        window_start=window_start,
        window_end=window_end,
    )
    assert len(events) == 1
    event = events[0]
    assert event.event_type == "maintenance_window"
    assert event.created_by == "llm"
    assert event.source.type == "dataset"
    assert len(notes) == 3


def test_events_are_backtest_admissible():
    from gnomon.context import backtest_admissible

    events, _ = events_from_proposals(
        [{
            "event_type": "launch",
            "effective_start": "2024-01-05T00:00:00+00:00",
            "effective_end": "2024-01-06T00:00:00+00:00",
        }],
        task_name="DemoTask",
        known_at="2024-01-01T00:00:00+00:00",
        window_start="2024-01-01T00:00:00+00:00",
        window_end="2024-02-01T00:00:00+00:00",
    )
    assert events and backtest_admissible(events[0])


def test_source_spans_must_quote_the_context_verbatim():
    """The future-context lane's provenance check lives in this adapter:
    Gnomon never sees the source document, so a span that is not a
    verbatim quote must be dropped before the engine does."""
    window_start = "2024-01-01T00:00:00+00:00"
    window_end = "2024-03-01T00:00:00+00:00"
    context = "Constraints: values will stay between 0 and 340 units.\n"
    base = {
        "effective_start": "2024-02-01T00:00:00+00:00",
        "effective_end": "2024-02-05T00:00:00+00:00",
    }
    proposals = [
        {**base, "event_type": "constraint:bounds",
         "source_span": "values will STAY between 0   and 340 units"},
        {**base, "event_type": "constraint:bounds",
         "source_span": "values are capped at 340"},  # a paraphrase
    ]
    events, notes = events_from_proposals(
        proposals, task_name="DemoTask", known_at=window_start,
        window_start=window_start, window_end=window_end,
        context_text=context,
    )
    assert len(events) == 1
    assert events[0].attributes["source_span"] == (
        "values will STAY between 0   and 340 units"
    )
    assert any("not a verbatim quote" in note for note in notes)


def test_spans_are_not_attached_when_the_lane_is_off():
    window_start = "2024-01-01T00:00:00+00:00"
    events, notes = events_from_proposals(
        [{
            "event_type": "constraint:bounds",
            "effective_start": "2024-02-01T00:00:00+00:00",
            "effective_end": "2024-02-05T00:00:00+00:00",
            "source_span": "anything at all",
        }],
        task_name="DemoTask", known_at=window_start,
        window_start=window_start, window_end="2024-03-01T00:00:00+00:00",
    )
    assert events and "source_span" not in events[0].attributes
    assert notes == []


def test_future_context_requires_agent_mode():
    import pytest

    from benchmarks.cik.gnomon_forecaster import GnomonForecaster

    with pytest.raises(ValueError, match="future_context"):
        GnomonForecaster(mode="pure", future_context=True)


def test_future_context_changes_the_cache_name():
    """The official result cache must never serve a flag-off run to a
    flag-on condition or vice versa."""
    from benchmarks.cik.gnomon_forecaster import GnomonForecaster

    off = GnomonForecaster(mode="agent", openrouter_model="x/y")
    on = GnomonForecaster(mode="agent", openrouter_model="x/y",
                          future_context=True)
    assert off.cache_name != on.cache_name
    assert on.cache_name.endswith("_future=on")


def test_abstention_carries_reasons():
    error = GnomonAbstained(["insufficient history", "no baseline beaten"])
    assert "GNOMON_ABSTAINED" in str(error)
    assert error.reasons == ["insufficient history", "no baseline beaten"]


def test_flagged_indices_merge_into_half_open_intervals():
    assert flagged_indices_to_intervals([3, 4, 5, 9]) == [
        {"start": 3, "end": 6},
        {"start": 9, "end": 10},
    ]
    assert flagged_indices_to_intervals([]) == []
    assert flagged_indices_to_intervals([7, 7, 6]) == [{"start": 6, "end": 8}]


def test_interval_vector_round_trip_and_f1():
    intervals = [{"start": 2, "end": 4}]
    vector = intervals_to_vector(intervals, 6)
    assert vector == [0, 0, 1, 1, 0, 0]
    assert binary_f1(vector, vector) == 1.0
    assert binary_f1(vector, [0] * 6) == 0.0
    partial = intervals_to_vector([{"start": 3, "end": 5}], 6)
    f1 = binary_f1(vector, partial)
    assert 0.0 < f1 < 1.0


def test_run_record_rows_match_gnomonbench_schema(tmp_path):
    writer = RecordWriter(tmp_path / "rows.jsonl")
    writer.write(RunRecord(
        task_id="demo-001", success=True, tool_calls=2,
        extra={"rcrps": 0.42},
    ))
    row = json.loads((tmp_path / "rows.jsonl").read_text().strip())
    assert row["task_id"] == "demo-001"
    assert row["success"] is True
    assert row["tool_calls"] == 2
    assert row["rcrps"] == 0.42
    for field in ("temporal_leakage", "invented_number", "warning_omission",
                  "appropriate_abstention"):
        assert row[field] is False


def test_run_record_extra_never_overrides_core_fields():
    record = RunRecord(task_id="x", success=False, extra={"success": True})
    assert record.to_row()["success"] is False


def test_epoch_is_timezone_aware():
    from benchmarks.anomllm.gnomon_detector import EPOCH

    assert EPOCH.tzinfo == timezone.utc
    assert isinstance(EPOCH, datetime)

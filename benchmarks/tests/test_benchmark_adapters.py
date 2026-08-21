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
from benchmarks.cik.run_cik import (
    RCRPS_CAP,
    build_parser,
    capped_imputed_mean,
    load_run_extra_info,
    write_outputs,
)
from benchmarks.common.openrouter import extract_json_array
from benchmarks.common.records import RecordWriter, RunRecord
from benchmarks.temporalbench.mcp_agent import _Run, preferred_execution_tool


def test_quantile_samples_shape_and_median():
    rows = [
        {"point": 10.0, "q10": 8.0, "q50": 10.0, "q90": 14.0},
        {"point": 20.0, "q10": 15.0, "q50": 20.0, "q90": 21.0},
    ]
    samples = samples_from_quantile_rows(rows, 25)
    assert len(samples) == 25
    assert all(len(trajectory) == 2 for trajectory in samples)
    # Each lead preserves its own stratified marginal; paths are deliberately
    # not comonotonic across leads.
    assert sorted(path[0] for path in samples)[12] == 10.0
    assert sorted(path[1] for path in samples)[12] == 20.0
    assert any(path[0] < 10 < path[1] for path in samples)


def test_quantile_samples_extrapolate_tails_and_preserve_marginal_order():
    rows = [{"point": 5.0, "q10": 2.0, "q50": 5.0, "q90": 9.0}]
    samples = samples_from_quantile_rows(rows, 100)
    values = [trajectory[0] for trajectory in samples]
    assert min(values) < 2.0 and max(values) > 9.0
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


def test_temporalbench_validates_complete_multiseries_artifact_not_brief_preview(
    monkeypatch, tmp_path,
):
    """Top-k response triage must not trigger duplicate forecast calls."""
    artifact_path = str(tmp_path / "forecast_artifact")

    class Session:
        def call_tool(self, name, arguments):
            assert name == "gnomon_forecast"
            return {
                "isError": False,
                "structuredContent": {
                    "artifact_path": artifact_path,
                    # Brief envelopes intentionally expose only notable rows.
                    "results": [{"series": "alpha"}],
                },
                "content": [{"type": "text", "text": "{}"}],
            }

    monkeypatch.setattr(
        "gnomon.artifacts.read_artifact",
        lambda path: {
            "results": [{"series": "alpha"}, {"series": "beta"}],
        },
    )
    run = object.__new__(_Run)
    run.target_keys = ["alpha", "beta"]
    run.profile = "evidence"
    run.horizon = 1
    run.csv_path = tmp_path / "history.csv"
    run.jail = tmp_path
    run.session = Session()
    run.trace = []
    run.mcp_calls = 0
    run.artifact_paths = set()
    run.complete_artifact_ready = False
    run.started = 0
    run.client = type("Client", (), {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
    })()
    run.tokens_at_start = 0
    run._cap_breach = lambda: None

    run._dispatch("gnomon_forecast", {})

    assert run.complete_artifact_ready is True
    assert run.artifact_paths == {artifact_path}


def test_temporalbench_precommits_forecast_verb_only_for_execution_arms():
    assert preferred_execution_tool("evidence", True) == "gnomon_forecast"
    assert preferred_execution_tool("mega", True) == "gnomon_run"
    assert preferred_execution_tool("full", True) is None
    assert preferred_execution_tool("evidence", False) is None


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


def test_overlong_spans_are_rejected_not_truncated():
    """Truncating after the verbatim check can cut a number mid-digits,
    handing the parser a figure the context states only as a substring."""
    window_start = "2024-01-01T00:00:00+00:00"
    long_span = "values will stay between 0 and " + "9" * 1000
    context = f"Constraints: {long_span}.\n"
    events, notes = events_from_proposals(
        [{
            "event_type": "constraint:bounds",
            "effective_start": "2024-02-01T00:00:00+00:00",
            "effective_end": "2024-02-05T00:00:00+00:00",
            "source_span": long_span,
        }],
        task_name="DemoTask", known_at=window_start,
        window_start=window_start, window_end="2024-03-01T00:00:00+00:00",
        context_text=context,
    )
    assert not events
    assert any("exceeds 1000 characters" in note for note in notes)


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
    assert "_future=on" in on.cache_name
    # Temperature is part of what a cached agent result measures; two
    # temperatures must never share a cache entry. Pure mode has no LLM,
    # so its names stay temperature-free (and cache-compatible).
    hot = GnomonForecaster(mode="agent", openrouter_model="x/y",
                           temperature=0.2)
    assert hot.cache_name != off.cache_name
    assert "temperature" not in GnomonForecaster(
        mode="pure").cache_name


def test_abstention_carries_reasons():
    error = GnomonAbstained(["insufficient history", "no baseline beaten"])
    assert "GNOMON_ABSTAINED" in str(error)
    assert error.reasons == ["insufficient history", "no baseline beaten"]


def test_capped_imputed_mean_matches_official_rule():
    """Official aggregation (upstream compile_roi_results.py, CAP = 5):
    per-run RCRPS clipped to the cap, missing runs imputed at the cap."""
    assert RCRPS_CAP == 5.0
    assert capped_imputed_mean([0.5, 7.0], 0) == (0.5 + 5.0) / 2
    assert capped_imputed_mean([1.0], 3) == (1.0 + 3 * 5.0) / 4
    assert capped_imputed_mean([], 2) == 5.0
    assert capped_imputed_mean([], 0) is None
    # Upstream maps NaN and negative entries to the cap as failures.
    assert capped_imputed_mean([float("nan"), -1.0], 0) == 5.0


def test_capped_imputed_mean_never_rewards_abstention():
    """Abstaining on a bad run must not beat scoring it: the imputed
    value is the cap, the worst any scored run can contribute."""
    all_scored = capped_imputed_mean([0.2, 4.9], 0)
    abstained_on_worst = capped_imputed_mean([0.2], 1)
    assert abstained_on_worst >= all_scored
    # Contrast: the scored-only mean is flattered by dropping the run.
    assert (0.2 + 4.9) / 2 > 0.2


def test_load_run_extra_info_reads_pprint_dumps(tmp_path):
    from pprint import pformat

    run_dir = tmp_path / "DemoTask" / "1"
    run_dir.mkdir(parents=True)
    (run_dir / "extra_info").write_text(
        pformat({"total_time": 12.5, "llm_usage": {"cost_usd": 0.1}}),
        encoding="utf-8",
    )
    assert load_run_extra_info(tmp_path, "DemoTask", 1) == {
        "total_time": 12.5, "llm_usage": {"cost_usd": 0.1},
    }
    assert load_run_extra_info(tmp_path, "DemoTask", 2) == {}
    (run_dir / "extra_info").write_text("<not literal python>",
                                        encoding="utf-8")
    assert load_run_extra_info(tmp_path, "DemoTask", 1) == {}


class _StubMethod:
    cache_name = "StubMethod"


def _cik_args(**overrides):
    import argparse

    values = {"method": "gnomon-pure", "model": None, "seeds": 2}
    values.update(overrides)
    return argparse.Namespace(**values)


_CIK_RESULTS = {
    "DemoTask": [
        {"seed": 1, "score": 0.5},
        {"seed": 2, "score": 7.0},  # above the cap
        {"seed": 3, "error": "GNOMON_ABSTAINED: unsupported frequency"},
        {"seed": 4, "error": "boom"},
        {"seed": 5, "score": float("nan")},  # non-finite counts as an error
    ],
}


def test_write_outputs_reports_both_aggregates(tmp_path):
    write_outputs(_CIK_RESULTS, _StubMethod(), _cik_args(), tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["runs_scored"] == 2
    assert summary["runs_abstained"] == 1
    assert summary["runs_errored"] == 2
    assert summary["mean_rcrps_scored_only"] == (0.5 + 7.0) / 2
    # 0.5 kept, 7.0 capped at 5, three unscored runs imputed at 5.
    assert summary["mean_rcrps_capped_imputed"] == (0.5 + 4 * 5.0) / 5
    assert "mean_rcrps" not in summary  # the old ambiguous key is gone


def test_write_outputs_rerun_does_not_duplicate_jsonl_rows(tmp_path):
    write_outputs(_CIK_RESULTS, _StubMethod(), _cik_args(), tmp_path)
    write_outputs(_CIK_RESULTS, _StubMethod(), _cik_args(), tmp_path)
    rows = (tmp_path / "gnomonbench.jsonl").read_text().splitlines()
    assert len(rows) == len(_CIK_RESULTS["DemoTask"])


def test_write_outputs_populates_latency_from_official_dumps(tmp_path):
    from pprint import pformat

    run_dir = tmp_path / "runs" / "DemoTask" / "1"
    run_dir.mkdir(parents=True)
    (run_dir / "extra_info").write_text(pformat({"total_time": 3.25}),
                                        encoding="utf-8")
    write_outputs(_CIK_RESULTS, _StubMethod(), _cik_args(), tmp_path)
    rows = [json.loads(line) for line in
            (tmp_path / "gnomonbench.jsonl").read_text().splitlines()]
    by_task = {row["task_id"]: row for row in rows}
    assert by_task["DemoTask-seed1"]["latency_seconds"] == 3.25
    # No dump (abstained/errored runs never write one) stays at zero.
    assert by_task["DemoTask-seed3"]["latency_seconds"] == 0.0
    # Cost is not derivable per run from the adapters' cumulative
    # accounting, so it must stay at the schema zero, not a guess.
    assert all(row["cost_usd"] == 0.0 for row in rows)


def test_fail_on_invalid_defaults_to_the_official_true():
    base = ["--method", "control", "--model", "x/y", "--output-dir", "out"]
    parser = build_parser()
    assert parser.parse_args(base).fail_on_invalid is True
    assert parser.parse_args(base + ["--no-fail-on-invalid"]) \
        .fail_on_invalid is False


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
    assert row["appropriate_abstention"] is False
    # Ungraded safety fields are absent, not false: a default False made
    # every adapter that never graded them report a measured-looking 0.0.
    for field in ("temporal_leakage", "invented_number", "warning_omission"):
        assert field not in row


def test_run_record_serialises_graded_safety_fields(tmp_path):
    writer = RecordWriter(tmp_path / "rows.jsonl")
    writer.write(RunRecord(task_id="graded-001", success=True,
                           temporal_leakage=False, invented_number=True))
    row = json.loads((tmp_path / "rows.jsonl").read_text().strip())
    assert row["temporal_leakage"] is False
    assert row["invented_number"] is True
    assert "warning_omission" not in row


def test_run_record_extra_never_overrides_core_fields():
    record = RunRecord(task_id="x", success=False, extra={"success": True})
    assert record.to_row()["success"] is False


def test_epoch_is_timezone_aware():
    from benchmarks.anomllm.gnomon_detector import EPOCH

    assert EPOCH.tzinfo == timezone.utc
    assert isinstance(EPOCH, datetime)

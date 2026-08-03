"""Unit tests for the TemporalBench adapter's pure logic (no network,
no official dataset, no numpy)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.temporalbench.gnomon_runner import _observed, uncertain_mcq
from benchmarks.temporalbench.scoring import score_mcq, score_t1, score_t3
from benchmarks.temporalbench.tasks import extract_json_object, prompt_input_arrays


def test_extract_json_object_from_prose():
    text = 'Answer:\n```json\n{"trend": "upward", "n": 3}\n```'
    assert extract_json_object(text) == {"trend": "upward", "n": 3}


def test_prompt_input_arrays_prefers_structured_input():
    row = {"input": {"history": {"hr": [1, 2, None], "note": "x"}}}
    arrays = prompt_input_arrays(row)
    assert arrays == {"hr": [1.0, 2.0, None]}


def test_prompt_input_arrays_falls_back_to_prompt_block():
    row = {"input": None,
           "prompt": 'Task text...\nInput (JSON):\n{"hr": [5, 6], "label": "a"}\nOutput...'}
    assert prompt_input_arrays(row) == {"hr": [5.0, 6.0]}


def test_score_t1_exact_match_case_insensitive():
    row = {"labels": {"trend": "constant", "volatility": "increased"}}
    result = score_t1(row, {"trend": "Constant", "volatility": "decreased"})
    assert result["correct"] == 1 and result["total"] == 2
    assert result["fields"]["trend"] is True


def test_score_t3_order_and_missing_answers():
    row = {"pack": [{"label": "Higher"}, {"label": "Yes"}, {"label": "Up"}]}
    result = score_t3(row, ["higher", "No"])
    assert result["per_question"] == [True, False, False]


def test_score_mcq_matches_labels():
    row = {"mcq": {"future_vs_history": {"label": "Uncertain"},
                   "volatility_change": {"label": "decreased"}}}
    result = score_mcq(row, {"future_vs_history": "uncertain",
                             "volatility_change": "increased"})
    assert result["correct"] == 1 and result["total"] == 2


def test_uncertain_mcq_uses_option_casing():
    row = {"mcq": {"q1": {"options": ["Higher", "Lower", "Uncertain"]},
                   "q2": {"options": ["fixed", "shifting", "no"]}}}
    answers = uncertain_mcq(row)
    assert answers["q1"] == "Uncertain"
    assert answers["q2"] == "fixed"  # no Uncertain option: first option


def test_observed_keeps_only_recorded_readings():
    # Nulls are dropped, not forward-filled: a reading nobody took must
    # not reach the forecaster as a repeat of the previous one.
    assert _observed([None, 1.0, None, 3.0]) == [1.0, 3.0]
    assert _observed([None, None]) == []


def test_forecast_channels_matches_per_channel_runs(tmp_path):
    """The batched adapter path must publish exactly the per-channel
    numbers the sequential path did — same forecasts, same abstentions —
    because the benchmark's metrics may not move, only its wall clock."""
    import math

    from benchmarks.temporalbench.gnomon_runner import (
        forecast_channel, forecast_channels,
    )

    channels = {
        # Different lengths and interior nulls, like real clinical rows.
        "hr": [70 + 5 * math.sin(2 * math.pi * k / 24) for k in range(96)],
        "spo2": [97.0 + (0.2 if k % 7 else -0.3) for k in range(80)],
        "resp": [None] * 4 + [16 + math.sin(2 * math.pi * k / 24)
                              for k in range(90)],
        "empty": [None, None, None],
    }
    horizon = 12
    batched = forecast_channels(channels, horizon, work_dir=str(tmp_path))
    for key, values in channels.items():
        single = forecast_channel(values, horizon, work_dir=str(tmp_path))
        assert batched[key].get("abstained") == single.get("abstained"), key
        if not single.get("abstained"):
            assert batched[key]["values"] == single["values"], key
            assert batched[key]["selected_model"] == single["selected_model"], key
            assert batched[key]["support"] == single["support"], key

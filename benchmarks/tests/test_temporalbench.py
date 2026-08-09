"""Unit tests for the TemporalBench adapter's pure logic (no network,
no official dataset, no numpy)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.temporalbench.gnomon_runner import (
    MCQ_ABSTAIN,
    _observed,
    uncertain_mcq,
)
from benchmarks.temporalbench.scoring import (
    score_forecast,
    score_mcq,
    score_t1,
    score_t3,
)
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
    answers, abstained = uncertain_mcq(row)
    assert answers["q1"] == "Uncertain"
    # No Uncertain option: the ABSTAIN sentinel, which matches no real
    # option and so deterministically scores wrong — never options[0],
    # which would luck into the label ~1/n of the time.
    assert answers["q2"] == MCQ_ABSTAIN
    assert abstained == ["mcq/q2: no Uncertain option"]
    scored = score_mcq({"mcq": {"q2": {"label": "fixed"}}}, answers)
    assert scored["correct"] == 0 and scored["total"] == 1


def test_abstain_sentinel_never_matches_an_option():
    # Pathological option set containing "abstain" itself: the sentinel
    # must still score wrong, not luck into a correct match.
    row = {"mcq": {"q1": {"options": ["higher", "Abstain", "ABSTAIN-"]}}}
    answers, abstained = uncertain_mcq(row)
    assert answers["q1"].strip().lower() not in {"higher", "abstain", "abstain-"}
    assert abstained == ["mcq/q1: no Uncertain option"]
    scored = score_mcq({"mcq": {"q1": {"label": "Abstain"}}}, answers)
    assert scored["correct"] == 0 and scored["total"] == 1


class _StubOfficialMetrics:
    """Stands in for the dataset's forecast_metrics_utils.py (not
    shipped with the repo): records every call, returns fixed metrics."""

    def __init__(self):
        self.calls = []

    def compute_forecast_metrics(self, ground_truth, forecast, **kwargs):
        self.calls.append((ground_truth, forecast, kwargs))
        return {"SMAPE": 12.5}, None


def test_score_forecast_full_abstention_never_calls_official_module():
    stub = _StubOfficialMetrics()
    row = {"ground_truth": {"hr": [1.0, 2.0]},
           "input": {"history": {"hr": [0.0, 1.0]}}}
    for empty_forecast in (None, {}, {"hr": []}):
        metrics, flag = score_forecast(row, empty_forecast, stub)
        assert metrics is None and flag == "no_forecast"
    assert stub.calls == []  # an abstention must not become a scoring error


def test_score_forecast_multichannel_abstention_short_circuits():
    stub = _StubOfficialMetrics()
    row = {"ground_truth": {"hr": [1.0], "spo2": [2.0]}, "input": {}}
    metrics, flag = score_forecast(row, {}, stub)
    assert metrics is None and flag == "no_forecast"
    assert stub.calls == []


def test_score_forecast_partial_abstention_names_missing_channels():
    stub = _StubOfficialMetrics()
    row = {"ground_truth": {"hr": [1.0, 2.0], "spo2": [3.0, 4.0]},
           "input": {"history": {"hr": [0.0], "spo2": [2.0]}}}
    metrics, flag = score_forecast(row, {"hr": [1.5, 2.5], "spo2": []}, stub)
    assert metrics == {"SMAPE": 12.5}
    assert flag == "missing_channels=spo2"
    (_, forecast, kwargs), = stub.calls  # only the channels that exist
    assert forecast == {"hr": [1.5, 2.5]}
    assert kwargs["history_series"] == {"hr": [0.0], "spo2": [2.0]}


def test_score_forecast_scored_row_passes_through():
    stub = _StubOfficialMetrics()
    row = {"ground_truth": {"hr": [1.0, 2.0]}, "input": {}}
    metrics, flag = score_forecast(row, {"hr": [1.1, 2.1]}, stub)
    assert metrics == {"SMAPE": 12.5} and flag is None
    (ground_truth, forecast, _), = stub.calls
    assert ground_truth == [1.0, 2.0] and forecast == [1.1, 2.1]


def test_bounded_evidence_stays_valid_json_within_budget():
    import json

    from benchmarks.temporalbench.run_temporalbench import bounded_evidence

    digest = {"main_key": "hr",
              "season": {"period": 24, "strength": 0.8, "basis": "acf"},
              "forecasts": {
                  "hr": {"support": "supported", "selected_model": "m",
                         "values": [float(i) for i in range(5000)]},
                  "spo2": {"support": "supported", "selected_model": "m",
                           "values": [97.0] * 4000},
              }}
    text = bounded_evidence(digest, budget=2000)
    assert len(text) <= 2000
    parsed = json.loads(text)  # never cut mid-token
    truncated = parsed["forecasts"]["hr"]
    assert truncated["truncated"] is True
    assert truncated["values_total"] == 5000
    assert truncated["values"] == [float(i) for i in range(24)]
    assert truncated["values_stats"]["min"] == 0.0
    # Deterministic, and a digest already under budget passes through whole.
    assert bounded_evidence(digest, budget=2000) == text
    assert json.loads(bounded_evidence(digest, budget=200_000)) == digest

    tiny = json.loads(bounded_evidence(digest, budget=120))
    assert tiny["truncated"] is True and "forecasts" in tiny["dropped"]


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


def test_forecast_payload_carries_support_labels():
    from benchmarks.temporalbench.gnomon_runner import forecast_payload

    analysis = {"channels": {
        "hr": {"abstained": False, "support": "supported",
               "values": [70.0, 71.0]},
        "temperature_c": {"abstained": False, "support": "best_effort",
                          "values": [36.8, 36.8]},
        "resp": {"abstained": True, "reason": "too short"},
    }}
    forecast, abstained, support = forecast_payload(analysis)
    assert set(forecast) == {"hr", "temperature_c"}
    assert abstained == ["resp: too short"]
    # The label travels with the values: a best_effort channel is a
    # disclosed fallback, and every score built on it must say so.
    assert support == {"hr": "supported", "temperature_c": "best_effort"}


def test_best_effort_turns_a_sparse_channel_abstention_into_labeled_rows(tmp_path):
    # A sparse channel (MIMIC's temperature_c pattern): too few readings
    # for the evaluation protocol. Default: an honest abstention. With
    # best_effort: the engine's disclosed fallback, labeled as such —
    # never silently mixed in with supported forecasts.
    from benchmarks.temporalbench.gnomon_runner import forecast_channel

    sparse = [36.5, 36.7, None, 36.6, 36.8, None, 36.9]
    horizon = 4
    default = forecast_channel(sparse, horizon, work_dir=str(tmp_path))
    assert default["abstained"] is True

    fallback = forecast_channel(sparse, horizon, work_dir=str(tmp_path),
                                best_effort=True)
    assert fallback["abstained"] is False
    assert fallback["support"] == "best_effort"
    assert len(fallback["values"]) == horizon


def test_score_per_channel_loader_reads_support_labels(tmp_path):
    import json as _json

    from benchmarks.temporalbench.score_per_channel import load_forecasts

    details = tmp_path / "details"
    details.mkdir()
    (details / "row1.json").write_text(_json.dumps({
        "answer": {"forecast": {"hr": [1.0, 2.0], "temperature_c": [36.8]}},
        "channel_support": {"hr": "supported",
                            "temperature_c": "best_effort"},
    }), encoding="utf-8")
    (details / "row2.json").write_text(_json.dumps({
        "answer": {"forecast": {"hr": [3.0]}},   # a control-style record:
    }), encoding="utf-8")                        # no support labels

    forecasts, support = load_forecasts(tmp_path)
    assert forecasts["row1"] == {"hr": [1.0, 2.0], "temperature_c": [36.8]}
    assert support["row1"] == {"hr": "supported",
                               "temperature_c": "best_effort"}
    assert "row2" in forecasts and "row2" not in support

"""Unit tests for the TemporalBench adapter's pure logic (no network,
no official dataset, no numpy)."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.temporalbench.gnomon_runner import (
    MCQ_ABSTAIN,
    _observed,
    forecast_target_map,
    uncertain_mcq,
)
from benchmarks.temporalbench.scoring import (
    score_forecast,
    score_mcq,
    score_t1,
    score_t3,
)
from benchmarks.temporalbench.score_per_channel import (
    stable_scaled_error_denominator,
)
from benchmarks.temporalbench.run_temporalbench import (
    primary_forecast_immutability,
)
from benchmarks.temporalbench.tasks import extract_json_object, prompt_input_arrays


def test_scaled_error_denominator_flags_near_constant_history():
    assert stable_scaled_error_denominator([10.0, 10.0, 10.0]) is False
    assert stable_scaled_error_denominator([10.0, 10.5, 9.5, 11.0]) is True
    assert stable_scaled_error_denominator([None, 10.0, None]) is False


def test_forecast_target_map_preserves_panel_channel_identities():
    arrays = {"heart_rate": [70.0], "spo2": [98.0]}
    row = {"ground_truth": {"heart_rate": [71.0], "spo2": [97.0]}}
    assert forecast_target_map(row, arrays) == {
        "heart_rate": "heart_rate", "spo2": "spo2"}


def test_forecast_target_map_resolves_official_single_target_aliases():
    psml = {
        "meta": {"main_key": "load_power", "target_col": "load_power"},
        "ground_truth": {"future_main": [0.9]},
    }
    assert forecast_target_map(psml, {"load_power": [1.0], "temperature": [20.0]}) \
        == {"future_main": "load_power"}

    retail = {"meta": {}, "ground_truth": {"future_sales": [12.0]}}
    assert forecast_target_map(
        retail, {"sales_censored": [10.0], "discount": [0.0]}) \
        == {"future_sales": "sales_censored"}


def test_forecast_target_map_refuses_ambiguous_unknown_alias():
    row = {"meta": {}, "ground_truth": {"future_main": [1.0]}}
    assert forecast_target_map(row, {"x": [1.0], "y": [2.0]}) == {}


def test_extract_json_object_from_prose():
    text = 'Answer:\n```json\n{"trend": "upward", "n": 3}\n```'
    assert extract_json_object(text) == {"trend": "upward", "n": 3}


def test_infrastructure_failure_classification():
    from benchmarks.temporalbench.run_temporalbench import infrastructure_failure

    class OpenRouterError(Exception):
        pass

    assert infrastructure_failure(OpenRouterError("HTTP 521"))
    assert infrastructure_failure(RuntimeError(
        "MCP tools/list failed: subprocess closed"))
    assert not infrastructure_failure(ValueError("No JSON object found"))


def test_resume_recovers_canonical_and_interrupted_partial(tmp_path):
    from benchmarks.temporalbench.run_temporalbench import load_resumable_records

    canonical = tmp_path / "gnomonbench.jsonl"
    partial = tmp_path / "gnomonbench.partial.jsonl"
    canonical.write_text(json.dumps({"task_id": "a", "success": True}) + "\n",
                         encoding="utf-8")
    partial.write_text(json.dumps({"task_id": "b", "success": True}) + "\n",
                       encoding="utf-8")
    assert set(load_resumable_records(canonical, partial)) == {"a", "b"}


def test_resume_rejects_conflicting_duplicate_records(tmp_path):
    import pytest
    from benchmarks.temporalbench.run_temporalbench import load_resumable_records

    canonical = tmp_path / "gnomonbench.jsonl"
    partial = tmp_path / "gnomonbench.partial.jsonl"
    canonical.write_text(json.dumps({"task_id": "a", "success": True}) + "\n",
                         encoding="utf-8")
    partial.write_text(json.dumps({"task_id": "a", "success": False}) + "\n",
                       encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting resumable records"):
        load_resumable_records(canonical, partial)


def test_artifact_keeps_primary_and_captures_sensitivity_separately(tmp_path):
    from benchmarks.temporalbench.mcp_agent import _Run

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    payload = {
        "schema_version": "0.1", "task": {"schema": {
            "target_column": "affected,unaffected"}},
        "results": [
            {"series": "affected", "support": "degraded",
             "forecast": [{"point": 1.0}, {"point": 1.0}],
             "sensitivity_scenarios": [{
                 "support": "hypothetical_sensitivity",
                 "forecast": [{"point": 2.0}, {"point": 3.0}]}]},
            {"series": "unaffected", "support": "degraded",
             "forecast": [{"point": 4.0}, {"point": 5.0}]},
        ],
    }
    (artifact / "artifact.json").write_text(json.dumps(payload), encoding="utf-8")
    run = _Run.__new__(_Run)
    run.artifact_paths = {str(artifact)}
    run.horizon = 2
    run._pending_support = {}
    run._available_sensitivity = {}
    run.context_execution = {}
    run.covariate_execution = {}

    affected = run._artifact_channel_rows(str(artifact), "affected")
    unaffected = run._artifact_channel_rows(str(artifact), "unaffected")

    assert affected == [1.0, 1.0]
    assert run._pending_support["affected"] == "degraded"
    assert run._available_sensitivity["affected"] == [2.0, 3.0]
    assert unaffected == [4.0, 5.0]
    assert run._pending_support["unaffected"] == "degraded"


def test_submit_schema_cannot_promote_a_sensitivity_to_primary():
    from benchmarks.temporalbench.mcp_agent import SUBMIT_TOOL

    channel = SUBMIT_TOOL["function"]["parameters"]["properties"][
        "forecast"]["additionalProperties"]["properties"]
    assert "trajectory" not in channel


def test_descriptive_panel_long_form_has_no_padding_blanks(tmp_path):
    import csv
    from benchmarks.temporalbench.mcp_agent import _write_long_csv

    path = tmp_path / "panel.csv"
    _write_long_csv({"long": [1.0, 2.0], "short": [3.0]}, path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {"timestamp": "2021-01-01T00:00:00+00:00", "series": "long", "value": "1.0"},
        {"timestamp": "2021-01-01T01:00:00+00:00", "series": "long", "value": "2.0"},
        {"timestamp": "2021-01-01T00:00:00+00:00", "series": "short", "value": "3.0"},
    ]


def test_temporalbench_axis_is_anchored_to_official_cutoff(tmp_path):
    from benchmarks.temporalbench.mcp_agent import _Run

    class Client:
        total_prompt_tokens = 0
        total_completion_tokens = 0

    class Session:
        def close(self):
            pass

    row = {
        "tier": "T4",
        "meta": {"n_horizon": 2, "main_key": "target",
                 "history_end": "2030-01-02T10:00:00",
                 "cluster_start": "2030-01-02T11:00:00"},
        "input": {"history": {"target": [1.0, 2.0, 3.0]}},
        "ground_truth": {"target": [4.0, 5.0]},
    }
    run = _Run(row, Client(), session_factory=lambda jail: Session(),
               work_dir=str(tmp_path))
    try:
        assert run.time_step.total_seconds() == 3600
        assert run.epoch == datetime(2030, 1, 2, 8, tzinfo=timezone.utc)
    finally:
        run.finish()


def test_host_compiled_context_is_written_inside_initialized_jail(tmp_path):
    from benchmarks.temporalbench.mcp_agent import _Run

    class Client:
        total_prompt_tokens = 0
        total_completion_tokens = 0

    class Session:
        def close(self):
            pass

    row = {
        "tier": "T4", "id": "compiled-context",
        "meta": {"n_horizon": 1, "main_key": "target",
                 "history_end": "2030-01-02T10:00:00+00:00"},
        "input": {"history": {"target": [1.0, 2.0, 3.0]}},
        "ground_truth": {"target": [4.0]},
        "_validated_context": {"events": [{
            "event_id": "e1", "event_type": "deploy",
            "entity_scope": ["*"],
            "effective_start": "2030-01-02T11:00:00+00:00",
            "effective_end": "2030-01-02T12:00:00+00:00",
            "known_at": "2030-01-02T09:00:00+00:00",
            "status": "confirmed", "confidence": 1.0,
            "attributes": {}, "created_by": "llm",
            "source": {"type": "benchmark_prompt", "reference": "row"},
        }]},
    }
    run = _Run(row, Client(), session_factory=lambda jail: Session(),
               work_dir=str(tmp_path), profile="evidence",
               compile_context=True)
    try:
        assert run.context_events_file is not None
        assert run.context_events_file.parent == run.jail
        assert run.context_events_file.is_file()
    finally:
        run.finish()


def test_temporalbench_observation_axis_does_not_infer_source_window_cadence(
        tmp_path):
    """The arrays omit timestamps; a 4h cluster gap is not a 4h grid."""
    from benchmarks.temporalbench.mcp_agent import _Run

    class Client:
        total_prompt_tokens = 0
        total_completion_tokens = 0

    class Session:
        def close(self):
            pass

    row = {
        "tier": "T2",
        "meta": {"n_horizon": 2, "main_key": "target",
                 "history_end": "2030-01-02T10:00:00",
                 "cluster_start": "2030-01-02T14:00:00"},
        "input": {"history": {"target": [1.0, 2.0, 3.0]}},
        "ground_truth": {"target": [4.0, 5.0]},
    }
    run = _Run(row, Client(), session_factory=lambda jail: Session(),
               work_dir=str(tmp_path), profile="evidence")
    try:
        assert run.time_step.total_seconds() == 3600
        assert run.epoch == datetime(2030, 1, 2, 8, tzinfo=timezone.utc)
        timestamps = [line.split(",", 1)[0]
                      for line in run.csv_path.read_text().splitlines()[1:]]
        assert timestamps == [
            "2030-01-02T08:00:00+00:00",
            "2030-01-02T09:00:00+00:00",
            "2030-01-02T10:00:00+00:00",
        ]
    finally:
        run.finish()


def test_prompt_input_arrays_prefers_structured_input():
    row = {"input": {"history": {"hr": [1, 2, None], "note": "x"}}}
    arrays = prompt_input_arrays(row)
    assert arrays == {"hr": [1.0, 2.0, None]}


def test_prompt_input_arrays_falls_back_to_prompt_block():
    row = {"input": None,
           "prompt": 'Task text...\nInput (JSON):\n{"hr": [5, 6], "label": "a"}\nOutput...'}
    assert prompt_input_arrays(row) == {"hr": [5.0, 6.0]}


def test_control_gets_one_format_only_repair_without_new_evidence():
    from benchmarks.temporalbench.run_temporalbench import answer_row

    class Client:
        def __init__(self):
            self.calls = []

        def completions(self, messages, *, n):
            self.calls.append(messages)
            return (["I think the answer is upward."] if len(self.calls) == 1
                    else ['{"trend":"upward"}'])

    client = Client()
    outcome = answer_row({"prompt": "Forecast it."}, "control", client)
    assert outcome["answer"] == {"trend": "upward"}
    assert len(client.calls) == 2
    repair = client.calls[1][-1]["content"]
    assert "same answer" in repair
    assert "Do not revise" in repair


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


def test_typed_answer_alignment_accepts_semantic_and_positional_ids():
    from benchmarks.temporalbench.run_temporalbench import align_typed_answers

    answers = [
        {"question": {"id": "q2"}, "best_estimate": {"value": "stable"}},
        {"question": {"id": "seasonality_shift"},
         "best_estimate": {"value": "continued"}},
        {"question": {"id": "invented"}, "best_estimate": {"value": "x"}},
    ]
    aligned = align_typed_answers(
        ["future_vs_history", "volatility_change", "seasonality_shift"],
        answers)
    assert set(aligned) == {"volatility_change", "seasonality_shift"}
    assert aligned["volatility_change"]["best_estimate"]["value"] == "stable"


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


def test_limit_is_stratified_across_tiers(tmp_path):
    """On a tier-grouped file, head-truncation reduced every limited
    multi-tier run to the earliest tier — --limit defeated --tiers."""
    import json

    from benchmarks.temporalbench.tasks import LABELED_FILE, iter_rows

    rows = ([{"id": f"t1-{i}", "tier": "T1"} for i in range(4)]
            + [{"id": f"t2-{i}", "tier": "T2"} for i in range(4)])
    (tmp_path / LABELED_FILE).write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    taken = list(iter_rows(tmp_path, tiers=("T1", "T2"), limit=4))
    assert [row["tier"] for row in taken] == ["T1", "T2", "T1", "T2"]
    # Within a tier, file order is kept.
    assert [row["id"] for row in taken] == ["t1-0", "t2-0", "t1-1", "t2-1"]
    # Larger balanced cohorts extend, rather than reshuffle, smaller ones;
    # offset shards therefore cannot overlap solely because their limits
    # differ.
    larger = list(iter_rows(tmp_path, tiers=("T1", "T2"), limit=6))
    assert [row["id"] for row in larger[:4]] == [row["id"] for row in taken]
    # An unlimited iteration still streams every matching row in order.
    assert [row["id"] for row in iter_rows(tmp_path, tiers=("T1", "T2"))] == [
        row["id"] for row in rows
    ]
    # A limit larger than the row count returns everything.
    assert len(list(iter_rows(tmp_path, tiers=("T1", "T2"), limit=99))) == 8


def test_raw_control_cannot_claim_forecast_immutability():
    assert primary_forecast_immutability("control", 20, {}) is None


def test_choice_only_run_has_no_forecast_immutability_claim():
    assert primary_forecast_immutability("gnomon-mcp", 0, {}) is None


def test_mcp_immutability_requires_every_channel_to_use_gnomon():
    assert primary_forecast_immutability(
        "gnomon-mcp", 20, {"gnomon": 60}) is True
    assert primary_forecast_immutability(
        "gnomon-mcp", 20, {"gnomon": 59, "abstain": 1}) is True
    assert primary_forecast_immutability(
        "gnomon-mcp", 20, {"gnomon": 59, "model": 1}) is False
    assert primary_forecast_immutability(
        "gnomon-mcp", 20, {"abstain": 60}) is False
    assert primary_forecast_immutability("gnomon-mcp", 20, {}) is False


def test_direct_gnomon_arm_owns_forecast_arrays():
    assert primary_forecast_immutability("gnomon-agent", 20, {}) is True

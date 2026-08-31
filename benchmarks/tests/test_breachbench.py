from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.breachbench.run_breachbench import (  # noqa: E402
    ARMS,
    COST_ACT,
    COST_MISS,
    generate_cases,
    parse_answer,
    product_packet,
    product_rule,
    run,
    series_frequency,
)


def _args(tmp_path: Path, cases: int = 6,
          resume: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        seed=20260826, cases=cases, data_dir=None,
        output_dir=str(tmp_path / "out"), resume=resume, concurrency=2,
        model="scripted-test-model")


class ScriptedClient:
    """Offline stand-in: acts whenever the visible history already sits
    near the threshold, and follows Gnomon's probabilities when given."""

    def completions(self, messages, *, n=1):
        text = messages[-1]["content"]
        threshold = float(re.search(r"Alert threshold: (-?[\d.]+)",
                                    text).group(1))
        values = json.loads(re.search(r"oldest first:\n(\[.*?\])",
                                      text, re.DOTALL).group(1))
        gnomon = re.search(r"Gnomon evidence.*?:\n(\{.*\})\nReturn",
                           text, re.DOTALL)
        if gnomon:
            packet = json.loads(gnomon.group(1))
            analysis = packet.get("threshold_analysis") or {}
            probabilities = analysis.get(
                "probability_above_per_step") or []
            peak = max(probabilities) if probabilities else 0.0
            bounded = analysis.get("bounded_assessment") or {}
            assessment = ({"yes": "breach", "no": "no_breach"}.get(
                bounded.get("decision"), bounded.get("decision"))
                if bounded else ("breach" if peak >= .5 else "no_breach"))
            act = peak >= COST_ACT / COST_MISS
            decision = packet.get("governed_decision") or {}
            return [json.dumps({"breach_expected": peak >= .5,
                                "first_breach_step": 1 if act else None,
                                "action": "act" if act else "monitor",
                                "automation_action": (
                                    decision.get("recommended_action")
                                    if decision.get("automation_eligible")
                                    else "withhold"),
                                "evidence_assessment": assessment,
                                "breach_probability": (
                                    peak if probabilities else None)})]
        near = max(values[-12:]) >= threshold
        return [json.dumps({"breach_expected": bool(near),
                            "first_breach_step": 3 if near else None,
                            "action": "act" if near else "monitor",
                            "automation_action": "withhold",
                            "evidence_assessment": (
                                "breach" if near else "no_breach"),
                            "breach_probability": 0.7 if near else 0.1})]


def test_cases_are_deterministic_real_and_outcome_labelled() -> None:
    first, provenance, futures = generate_cases(9, 12)
    again, _, _ = generate_cases(9, 12)
    assert first == again
    cells = set()
    for case in first:
        cells.add(case.outcome_cell)
        held_out = futures[case.case_id]
        assert len(held_out) == case.horizon
        breach_steps = [step for step, value in enumerate(held_out, 1)
                        if value > case.threshold]
        # The label and the transformed future agree: affine anonymization
        # preserved breach structure and timing exactly.
        assert case.truth_breach == bool(breach_steps)
        assert case.truth_first_step == (breach_steps[0]
                                         if breach_steps else None)
    assert len(cells) >= 2
    assert set(provenance["corpus_series"]) >= {
        "wiki_traffic_daily_log", "sensor_temps_5min",
        "pedestrian_counts_daily", "retail_sales_monthly"}
    assert provenance["labeling"] == "realized_future_breach_and_first_step"
    assert provenance["history_windows"] == [24, 48, 96, 168]
    assert sum(provenance["history_band_distribution"].values()) == 12
    for case in first:
        assert case.frequency == series_frequency(case.origin)
        assert case.history_length == len(case.values)
        assert case.history_band in {"short", "medium", "long"}


def test_realized_futures_never_overlap_within_a_series(tmp_path) -> None:
    """Overlapping futures share breach events; correlated truth labels
    would inflate the paired significance tests. A series with room for
    only two horizon-spaced futures must refuse to yield four cases."""
    import math
    import pytest

    data = tmp_path / "corpus"
    data.mkdir()
    values = [100 + 5 * math.sin(i / 3) + (i % 7) for i in range(150)]
    data.joinpath("tiny_daily.csv").write_text(
        "value\n" + "\n".join(str(v) for v in values), encoding="utf-8")
    with pytest.raises(ValueError) as failure:
        generate_cases(9, 4, data)
    assert "future_overlap" in str(failure.value)
    cases, provenance, futures = generate_cases(9, 2, data)
    assert len(cases) == 2
    assert provenance["independence"].startswith(
        "realized_futures_non_overlapping_within_series")
    assert sum(provenance["cases_per_series"].values()) == len(cases)


def test_cadence_is_read_from_the_corpus_filenames() -> None:
    assert series_frequency("sensor_temps_5min") == "5min"
    assert series_frequency("retail_sales_monthly") == "MS"
    assert series_frequency("wiki_traffic_daily_log") == "D"
    assert series_frequency("pedestrian_counts_daily") == "D"
    assert series_frequency("mystery_series") == "D"


def test_the_product_packet_is_real_gnomon_output() -> None:
    cases, _, _ = generate_cases(9, 3)
    packet = product_packet(cases[0])
    assert packet["authority"] == \
        "bounded_projection_of_gnomon_forecast_response"
    assert packet["support"]
    assert packet["tier_floor"]
    assert packet["headline"]
    assert len(packet["forecast"]) == cases[0].horizon
    assert "threshold_analysis" in packet
    if packet["threshold_analysis"].get("probability_status") == \
            "unavailable_uncalibrated":
        assert packet["threshold_analysis"]["bounded_assessment"]
    rule = product_rule(cases[0], packet)
    assert rule["action"] in {"act", "monitor"}
    assert isinstance(rule["breach_expected"], bool)


def test_product_packet_preserves_weaker_horizon_event_authority() -> None:
    """The benchmark must exercise the same trust boundary as the product."""
    cases, _, _ = generate_cases(20260826, 12)
    packet = product_packet(next(
        case for case in cases if case.case_id == "b20260826-0009"))
    event = packet["threshold_analysis"]["horizon_event"]
    assert packet["support"] == "supported"
    assert packet["support_scope"] == "forecast_path"
    assert packet["tier_floor"] == "best_effort"
    assert event["support"] == "best_effort"
    assert "tier supported" in packet["headline"]
    assert "tier best_effort" in packet["headline"]
    assert "High-confidence" not in packet["headline"]


def test_answers_are_validated_not_guessed() -> None:
    assert parse_answer("no json here", 24) == {"valid": False}
    assert parse_answer('{"breach_expected": "maybe", "action": "act"}',
                        24) == {"valid": False}
    parsed = parse_answer(
        '{"breach_expected": true, "first_breach_step": 40, '
        '"action": "act"}', 24)
    assert parsed["valid"] and parsed["first_breach_step"] is None
    assert parsed["evidence_assessment"] is None
    assert parsed["breach_probability"] is None


def test_assessment_and_probability_are_validated_separately() -> None:
    parsed = parse_answer(
        '{"breach_expected": false, "first_breach_step": null, '
        '"action": "act", "evidence_assessment": "indeterminate", '
        '"automation_action": "withhold", '
        '"breach_probability": 0.37}', 24)
    assert parsed["valid"] is True
    assert parsed["breach_expected"] is False
    assert parsed["action"] == "act"
    assert parsed["automation_action"] == "withhold"
    assert parsed["evidence_assessment"] == "indeterminate"
    assert parsed["breach_probability"] == 0.37
    hostile = parse_answer(
        '{"breach_expected": true, "first_breach_step": null, '
        '"action": "monitor", "evidence_assessment": "certain", '
        '"breach_probability": 1.5}', 24)
    assert hostile["valid"] is True
    assert hostile["evidence_assessment"] is None
    assert hostile["breach_probability"] is None


def test_a_valid_answer_survives_surrounding_prose_and_echoes() -> None:
    # A greedy first-to-last-brace regex would score all of these as
    # invalid answers the model never gave.
    answer = ('{"breach_expected": true, "first_breach_step": 3, '
              '"action": "act"}')
    trailing = answer + "\nReasoning: the costs {act=2, miss=10} say act."
    parsed = parse_answer(trailing, 24)
    assert parsed["valid"] and parsed["action"] == "act"
    assert parsed["first_breach_step"] == 3
    echoed = ('Given the evidence {"support": "degraded", "forecast": '
              '[{"step": 1, "q50": 4.0}]} I conclude:\n' + answer)
    parsed = parse_answer(echoed, 24)
    assert parsed["valid"] and parsed["breach_expected"] is True
    assert parse_answer("no json here", 24) == {"valid": False}


def test_resume_rejects_rows_from_a_different_dataset_or_model(
        tmp_path) -> None:
    """The same seed with a different --cases count yields sequential ids
    over divergent content — truth labels included. Rows from that run
    must never be pooled into this one."""
    run(_args(tmp_path, cases=6), client=ScriptedClient())
    rows_path = tmp_path / "out" / "rows.jsonl"
    first_rows = rows_path.read_text().splitlines()
    assert len(first_rows) == 6 * len(ARMS)
    args = _args(tmp_path, cases=4, resume=True)
    summary = run(args, client=ScriptedClient())
    # Every old row was rejected (different dataset identity), so the
    # 4-case run answered all its own pairs from scratch.
    assert summary["paired"]["paired_cases"] == 4
    assert len(rows_path.read_text().splitlines()) == \
        len(first_rows) + 4 * len(ARMS)
    for line in rows_path.read_text().splitlines()[len(first_rows):]:
        row = json.loads(line)
        assert row["dataset"] == summary["provenance"]["dataset_identity"]
        assert row["model"] == "scripted-test-model"


def test_resume_rejects_rows_from_a_different_request_contract(tmp_path) -> None:
    run(_args(tmp_path, cases=4), client=ScriptedClient())
    rows_path = tmp_path / "out" / "rows.jsonl"
    old_count = len(rows_path.read_text().splitlines())
    args = _args(tmp_path, cases=4, resume=True)
    args.max_tokens = 401
    run(args, client=ScriptedClient())
    new_rows = [json.loads(line) for line in
                rows_path.read_text().splitlines()[old_count:]]
    assert len(new_rows) == 4 * len(ARMS)
    assert all(row.get("request_sha256") for row in new_rows)


def test_malformed_steps_degrade_and_never_crash_a_paid_run() -> None:
    # json.loads accepts NaN/Infinity, and true is an int in Python:
    # each must degrade to a missing step, not raise mid-run.
    for hostile in ("NaN", "Infinity", "-Infinity", "true", "1e400",
                    '"soon"', "0", "-3"):
        parsed = parse_answer(
            '{"breach_expected": true, "first_breach_step": ' + hostile
            + ', "action": "act"}', 24)
        assert parsed["valid"] is True
        assert parsed["first_breach_step"] is None
    parsed = parse_answer(
        '{"breach_expected": true, "first_breach_step": 7.0, '
        '"action": "act"}', 24)
    assert parsed["first_breach_step"] == 7


def test_one_dead_call_fails_loudly_and_resume_finishes(tmp_path) -> None:
    import pytest

    import threading

    class FlakyClient(ScriptedClient):
        def __init__(self) -> None:
            self.calls = 0
            self._lock = threading.Lock()

        def completions(self, messages, *, n=1):
            with self._lock:
                self.calls += 1
                ordinal = self.calls
            if ordinal == 3:
                raise RuntimeError("endpoint fell over")
            return super().completions(messages, n=n)

    with pytest.raises(RuntimeError) as failure:
        run(_args(tmp_path), client=FlakyClient())
    assert "rerun with --resume" in str(failure.value)
    rows_path = tmp_path / "out" / "rows.jsonl"
    survived = rows_path.read_text().splitlines()
    assert len(survived) == 6 * len(ARMS) - 1
    # Stale rows from an older seed and a crash-truncated line must not
    # leak into the resumed run's metrics.
    with rows_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"case_id": "b999-0000", "arm": "control",
                                 "cost": 0.0}) + "\n")
        handle.write('{"case_id": "b20260826-')
    summary = run(_args(tmp_path, resume=True), client=ScriptedClient())
    assert summary["paired"]["paired_cases"] == 6
    for arm in ARMS:
        assert 0.0 <= summary["metrics"][arm]["mean_regret"] <= 10.0


def test_a_matched_offline_run_prices_decisions_in_client_units(
        tmp_path) -> None:
    summary = run(_args(tmp_path), client=ScriptedClient())
    assert set(summary["metrics"]) == set(ARMS)
    for arm in ARMS:
        entry = summary["metrics"][arm]
        assert entry["mean_regret"] >= 0.0
        assert 0.0 <= entry["action_optimal_rate"] <= 1.0
        assert 0.0 <= entry["assessment_coverage"] <= 1.0
        assert 0.0 <= entry["probability_coverage"] <= 1.0
        assert 0.0 <= entry["automation_action_coverage"] <= 1.0
        assert entry["log_loss"] is not None
        assert sum(bin_["count"] for bin_ in
                   entry["calibration_by_probability_bin"].values()) == \
            entry["call_metrics_scored"]
    references = summary["references"]
    assert set(references) >= {"gnomon_governed", "gnomon_rule_alone",
                               "gnomon_rule_composed",
                               "naive_persistence", "always_act",
                               "never_act", "hindsight_optimal"}
    assert references["hindsight_optimal"]["mean_regret"] == 0.0
    assert references["always_act"]["mean_cost"] == COST_ACT
    verdicts = summary["verdicts"]
    assert set(verdicts) >= {"regret_reduction_vs_model_alone",
                             "regret_reduction_vs_product_rule_alone"}
    design = summary["design"]
    assert design["gnomon_packet_is_production_output"] is True
    assert design["held_out_future_absent_from_prompts_verified"] is True
    assert summary["paired"]["primary_endpoint"] == "per_case_decision_cost"
    interval = summary["paired"][
        "mean_regret_reduction_cluster_bootstrap_95"]
    assert interval["cluster"] == "origin_series"
    assert interval["lower"] <= interval["upper"]
    assert "agent_preservation" in summary["paired"]
    preservation = summary["paired"]["agent_preservation"]
    assert "human_recommendation_adherence_rate" in preservation
    overrides = preservation["human_override_evaluation"]
    assert overrides["overrides"] == (
        overrides["beneficial"] + overrides["harmful"]
        + overrides["neutral"])
    assert "automation authority" in overrides["reading"]
    rows = [json.loads(line) for line in
            (tmp_path / "out" / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 6 * len(ARMS)
    assert all(row["raw_response"] for row in rows)
    assert all(len(row["raw_response_sha256"]) == 64 for row in rows)
    assert all(("evidence_packet" in row) == (row["arm"] == "gnomon")
               for row in rows)

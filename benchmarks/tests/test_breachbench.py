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
)


def _args(tmp_path: Path, cases: int = 6) -> argparse.Namespace:
    return argparse.Namespace(
        seed=20260826, cases=cases, data_dir=None,
        output_dir=str(tmp_path / "out"), resume=False, concurrency=2,
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
            probabilities = (packet.get("threshold_analysis") or {}).get(
                "probability_above_per_step") or []
            peak = max(probabilities) if probabilities else 0.0
            act = peak >= COST_ACT / COST_MISS
            return [json.dumps({"breach_expected": peak >= .5,
                                "first_breach_step": 1 if act else None,
                                "action": "act" if act else "monitor"})]
        near = max(values[-12:]) >= threshold
        return [json.dumps({"breach_expected": bool(near),
                            "first_breach_step": 3 if near else None,
                            "action": "act" if near else "monitor"})]


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


def test_the_product_packet_is_real_gnomon_output() -> None:
    cases, _, _ = generate_cases(9, 3)
    packet = product_packet(cases[0])
    assert packet["authority"] == \
        "computed_gnomon_forecast_with_threshold_analysis"
    assert packet["support"]
    assert packet["headline"]
    assert len(packet["forecast"]) == cases[0].horizon
    assert "threshold_analysis" in packet
    rule = product_rule(cases[0], packet)
    assert rule["action"] in {"act", "monitor"}
    assert isinstance(rule["breach_expected"], bool)


def test_answers_are_validated_not_guessed() -> None:
    assert parse_answer("no json here", 24) == {"valid": False}
    assert parse_answer('{"breach_expected": "maybe", "action": "act"}',
                        24) == {"valid": False}
    parsed = parse_answer(
        '{"breach_expected": true, "first_breach_step": 40, '
        '"action": "act"}', 24)
    assert parsed["valid"] and parsed["first_breach_step"] is None


def test_a_matched_offline_run_prices_decisions_in_client_units(tmp_path) -> None:
    summary = run(_args(tmp_path), client=ScriptedClient())
    assert set(summary["metrics"]) == set(ARMS)
    for arm in ARMS:
        entry = summary["metrics"][arm]
        assert entry["mean_regret"] >= 0.0
        assert 0.0 <= entry["action_optimal_rate"] <= 1.0
    references = summary["references"]
    assert set(references) >= {"gnomon_rule_alone", "naive_persistence",
                               "always_act", "never_act",
                               "hindsight_optimal"}
    assert references["hindsight_optimal"]["mean_regret"] == 0.0
    assert references["always_act"]["mean_cost"] == COST_ACT
    verdicts = summary["verdicts"]
    assert set(verdicts) >= {"regret_reduction_vs_model_alone",
                             "regret_reduction_vs_product_rule_alone"}
    design = summary["design"]
    assert design["gnomon_packet_is_production_output"] is True
    assert design["held_out_future_absent_from_prompts_verified"] is True
    assert summary["paired"]["primary_endpoint"] == "per_case_decision_cost"
    rows = [json.loads(line) for line in
            (tmp_path / "out" / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 6 * len(ARMS)

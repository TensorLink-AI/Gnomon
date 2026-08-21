from argparse import Namespace

from benchmarks.reasoningbench.run_reasoningbench import (
    DECISION_SEEDS,
    GENERATOR_VERSION, compact_packet, exact_sign_p, expected, generate_cases, parse_answer,
    packet_exposes_answer,
)


def test_decision_seeds_are_precommitted_and_distinct() -> None:
    assert DECISION_SEEDS == (223607, 244949, 264575)
    assert len(set(DECISION_SEEDS)) == 3


def test_cases_are_deterministic_and_balanced_across_properties() -> None:
    left = generate_cases(41, 18)
    right = generate_cases(41, 18)
    assert left == right
    assert {case.prop for case in left} == {
        "level", "trend", "volatility", "seasonality", "regime", "extreme"}
    assert GENERATOR_VERSION == "0.4"


def test_packet_is_compact_and_does_not_contain_generator_truth() -> None:
    case = generate_cases(42, 1)[0]
    packet = compact_packet(case)
    assert len(str(packet)) < 1000
    assert "expected" not in packet
    assert "primary_forecast_unchanged" not in packet
    assert isinstance(packet["measurement"]["automation_eligible"], bool)
    assert "direction" not in str(packet)
    assert "historical_analogue_consensus" not in packet
    assert "next" not in packet
    assert packet_exposes_answer(case, packet) is False


def test_expected_uses_generator_truth_and_computed_support() -> None:
    case = generate_cases(43, 1)[0]
    truth = expected(case)
    assert truth["diagnosis"] == case.expected
    assert truth["confidence"] in {"supported", "uncertain"}


def test_json_parser_and_exact_test() -> None:
    parsed = parse_answer('text {"diagnosis":"HIGHER","confidence":"supported",'
                          '"analogue_outcome":"up","next_action":"act"}')
    assert parsed["diagnosis"] == "higher"
    assert exact_sign_p(0, 0) == 1.0
    assert exact_sign_p(10, 0) < .01

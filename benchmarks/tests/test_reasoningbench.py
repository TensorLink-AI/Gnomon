import json
from argparse import Namespace
from collections import Counter, defaultdict
from types import SimpleNamespace

from benchmarks.reasoningbench import run_reasoningbench
from benchmarks.reasoningbench.run_reasoningbench import (
    compact_packet, corpus_sha256, exact_sign_p, expected, generate_cases,
    parse_answer,
)

SCORED_FROM_PACKET = ("confidence", "analogue_outcome", "next_action")


def _leaves(obj, path=""):
    """Yield (key_path, value) for every scalar leaf of a packet."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _leaves(value, path + "/" + key)
    elif isinstance(obj, list):
        for value in obj:
            yield from _leaves(value, path + "[]")
    else:
        yield path, obj


def test_cases_are_deterministic_and_balanced_across_properties() -> None:
    left = generate_cases(41, 18)
    right = generate_cases(41, 18)
    assert left == right
    assert {case.prop for case in left} == {
        "level", "trend", "volatility", "seasonality", "regime", "extreme"}


def test_claim_conflict_is_not_aliased_with_the_property() -> None:
    """index % 2 over a 6-property cycle made each property always or
    never conflict, so the claim dimension collapsed into the label."""
    per_property = defaultdict(set)
    for case in generate_cases(82026, 72):
        per_property[case.prop].add(case.claim_conflicts)
    assert all(states == {True, False} for states in per_property.values())


def test_packet_is_compact_and_contains_no_scored_answer() -> None:
    for case in generate_cases(82026, 36):
        packet = compact_packet(case)
        assert len(str(packet)) < 1000
        # The removed exploit surface: support restated confidence, the
        # consensus field restated analogue_outcome, and the next sentence
        # paraphrased the action enum.
        rendered = json.dumps(packet)
        assert "support" not in rendered
        assert "historical_analogue_consensus" not in rendered
        assert "next" not in packet
        assert packet["primary_forecast_unchanged"] is True
        truth = expected(case)
        leaf_strings = {str(value).lower() for _, value in _leaves(packet)}
        for field in SCORED_FROM_PACKET:
            assert truth[field] not in leaf_strings, (field, packet)


def test_packet_copy_strategy_is_at_chance_not_a_key() -> None:
    """No packet leaf answers a scored field via a fixed 1:1 mapping.

    The bound is deliberately generous: even the best mapping LEARNED ON
    THE SCORED CORPUS ITSELF (an upper bound on any fixed copy rule) must
    stay well below perfect on every previously-copyable field, and the
    expected values must actually vary so 'copying' has something to miss.
    """
    cases = generate_cases(82026, 72)
    truths = [expected(case) for case in cases]
    packets = [compact_packet(case) for case in cases]
    leaf_values = defaultdict(list)
    for packet in packets:
        for path, value in _leaves(packet):
            leaf_values[path].append(value)
    for field in SCORED_FROM_PACKET:
        answers = [truth[field] for truth in truths]
        assert len(set(answers)) >= 2, field
        for path, values in leaf_values.items():
            if not all(isinstance(value, str) for value in values):
                continue
            grouped = defaultdict(Counter)
            for value, answer in zip(values, answers):
                grouped[value][answer] += 1
            best = sum(counter.most_common(1)[0][1]
                       for counter in grouped.values()) / len(answers)
            assert best < 0.9, (field, path, best)


def test_analogue_consensus_is_generated_not_forced() -> None:
    """The two nearest episodes must sometimes disagree — a forced
    consensus made the field a constant of the construction — and the
    generator's recorded outcome must equal the consensus recomputed
    from its own rows (or 'unavailable' when they disagree)."""
    cases = generate_cases(82026, 72)
    outcomes = set()
    for case in cases:
        nearest = sorted(case.analogues)[:2]
        consensus = (nearest[0][1] if nearest[0][1] == nearest[1][1]
                     else "unavailable")
        assert case.expected_analogue == consensus
        outcomes.add(consensus)
    assert "unavailable" in outcomes
    assert outcomes - {"unavailable"}


def test_expected_uses_generator_truth_never_the_packet(monkeypatch) -> None:
    """expected() must be computable without window_evidence: a truth
    derived from the packet's internals IS the packet, and copying it
    back would score perfectly."""
    monkeypatch.setattr(
        run_reasoningbench, "window_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("expected() must not read window evidence")))
    for case in generate_cases(43, 18):
        truth = expected(case)
        assert truth["diagnosis"] == case.expected
        assert truth["analogue_outcome"] == case.expected_analogue
        if truth["confidence"] == "uncertain":
            assert case.difficulty == "marginal"
            assert truth["next_action"] == "collect_more"
        elif case.claim_conflicts:
            assert truth["next_action"] == "resolve_conflict"
        else:
            assert truth["next_action"] == "act"


def test_corpus_sha256_names_the_exact_cases() -> None:
    assert corpus_sha256(generate_cases(82026, 12)) == \
        corpus_sha256(generate_cases(82026, 12))
    assert corpus_sha256(generate_cases(82026, 12)) != \
        corpus_sha256(generate_cases(82027, 12))


def test_run_records_corpus_integrity_in_summary(tmp_path, monkeypatch) -> None:
    """Full deterministic run against a scripted client: the summary must
    carry the corpus hash and the fresh-seed flag beside the seed."""
    fixed = json.dumps({"diagnosis": "similar", "confidence": "supported",
                        "analogue_outcome": "unavailable",
                        "next_action": "act"})

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, messages, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=fixed))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))

    monkeypatch.setattr(run_reasoningbench, "OpenRouterClient", Client)
    args = Namespace(model="scripted", base_url="http://x", cases=6,
                     seed=82026, fresh_seed=False, concurrency=2,
                     output_dir=str(tmp_path), resume=False)
    summary = run_reasoningbench.run(args)
    assert summary["corpus_sha256"] == corpus_sha256(generate_cases(82026, 6))
    assert summary["fresh_seed"] is False
    assert summary["seed"] == 82026
    assert summary["design"]["packet_excludes_scored_answers"] is True
    assert "exact_mcnemar_p" in summary["paired"]


def test_json_parser_and_exact_test() -> None:
    parsed = parse_answer('text {"diagnosis":"HIGHER","confidence":"supported",'
                          '"analogue_outcome":"up","next_action":"act"}')
    assert parsed["diagnosis"] == "higher"
    assert exact_sign_p(0, 0) == 1.0
    assert exact_sign_p(10, 0) < .01

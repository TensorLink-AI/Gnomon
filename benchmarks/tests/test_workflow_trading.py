"""Contract tests for the trading version of Workflow Bench.

These cover the parts a general temporal corpus cannot: session calendars,
price basis, non-comparable history, return-versus-level ranking, and the
refusal to publish advice.
"""

import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.workflow.audit import audit
from benchmarks.workflow.provenance import corpus_sha256
from benchmarks.workflow.run_workflow import case_payload
from benchmarks.workflow.schema import Observation, Oracle, load_cases
from benchmarks.workflow.scoring import score_run
from benchmarks.workflow.trading_baseline import _answer, _repair
from benchmarks.workflow.trading import (
    ASSET_CLASSES, CALENDARS, HOLIDAYS, REQUIRED_FORBIDDEN_CLAIMS,
    SESSIONS_PER_WEEK, generate_trading_cases, next_session, sessions_from,
)

SMOKE_CASES = (Path(__file__).resolve().parents[1] / "workflow" / "cases"
               / "trading_smoke.jsonl")


def _perfect(case, **overrides):
    """An arm that answers every trading contract exactly."""
    fingerprint = f"fingerprint-{case.id}"
    values = {
        "case_id": case.id, "status": "answered",
        "support": case.oracle.allowed_support[0],
        "numbers": dict(case.oracle.numbers), "choices": dict(case.oracle.choices),
        "facts": dict(case.oracle.required_facts),
        "engine_facts": dict(case.oracle.engine_required_facts),
        "claims": ["forecast on the stated basis, point-in-time input only"],
        "temporal_leakage": False, "publish_matches_evaluated": True,
        "evaluated_fingerprint": fingerprint, "published_fingerprint": fingerprint,
        "headline_numbers": dict(case.oracle.numbers),
        "artifact_numbers": dict(case.oracle.numbers),
        "quote_matches": True, "tool_calls": 1, "cumulative_tokens": 900,
        "stage_results": {},
    }
    if case.oracle.should_abstain:
        values.update(status="abstained", support="abstained", numbers={},
                      choices={}, headline_numbers={}, artifact_numbers={})
    values.update(overrides)
    for stage in case.stages:
        oracle = Oracle.from_dict(stage.get("oracle") or {})
        if stage["name"] == "repair":
            values["stage_results"].setdefault("repair", {
                # The runner only calls a repair complete when the arm
                # abstained first, so an eager answer cannot buy it back.
                "completed": values["status"] == "abstained",
                "followup_status": "answered",
                "numbers": dict(oracle.numbers), "choices": dict(oracle.choices),
                "facts": dict(oracle.required_facts)})
        elif case.oracle.requires_tracking:
            values["stage_results"].setdefault("outcome", {
                "tracked": True, "absolute_error": 0.0,
                "reported_absolute_error": 0.0})
    return Observation.from_dict(values)


def _of_kind(cases, kind):
    return next(case for case in cases if case.kind == kind)


def test_trading_corpus_is_publication_ready_balanced_and_sealed():
    cases = generate_trading_cases()
    result = audit(cases, "trading")
    assert result["ready"] is True, result["checks"]
    assert len(cases) == 100
    assert set(result["domains"]) == set(ASSET_CLASSES)
    assert set(result["kinds"].values()) == {20}
    assert result["trading"]["ambiguity_kinds"] == [
        "corporate_action", "halt", "venue"]
    for case in cases:
        public = case_payload(case)
        assert "oracle" not in public and "stages" not in public
        # The basis is evidence the arm is allowed to see; the answer it
        # implies is not.
        assert public["available_at_cutoff"]["price_basis"]


def test_trading_generator_is_reproducible_but_seed_sensitive():
    first = generate_trading_cases(seed=7)
    replay = generate_trading_cases(seed=7)
    holdout = generate_trading_cases(seed=8)
    assert corpus_sha256(first) == corpus_sha256(replay)
    assert corpus_sha256(first) != corpus_sha256(holdout)


def test_shipped_smoke_bundle_is_smoke_grade_not_decision_grade():
    cases = load_cases(SMOKE_CASES)
    assert {case.kind for case in cases} == {
        "synthetic", "frozen", "messy", "longitudinal", "multiseries"}
    assert audit(cases, "trading-smoke")["ready"] is True
    assert audit(cases, "trading")["ready"] is False


def test_the_next_observation_is_the_next_session_not_the_next_day():
    cases = generate_trading_cases()
    gapped = 0
    for case in cases:
        calendar = CALENDARS[case.domain]
        stamps = case.available_at_cutoff["columns"]["timestamp"]
        assert stamps == sessions_from(stamps[0], len(stamps), calendar)
        if SESSIONS_PER_WEEK[calendar] == 5:
            assert not any(date.fromisoformat(day).weekday() >= 5 for day in stamps)
            assert not set(stamps) & set(HOLIDAYS[calendar])
        upcoming = case.oracle.required_facts.get("next_session")
        if upcoming is None:
            continue
        assert upcoming == next_session(stamps[-1], calendar)
        if (date.fromisoformat(upcoming) - date.fromisoformat(stamps[-1])).days > 1:
            gapped += 1
    # A corpus whose answer is always "yesterday plus one day" would not test
    # a calendar at all.
    assert gapped >= 3


def test_no_input_bar_or_revealed_outcome_precedes_its_own_cutoff():
    cases = generate_trading_cases(per_kind=4)
    assert audit(cases, "trading-smoke")["checks"][
        "trading_inputs_are_point_in_time"] is True
    leaked = _of_kind(cases, "synthetic")
    columns = json.loads(json.dumps(leaked.available_at_cutoff))
    columns["columns"]["timestamp"].append("2027-01-04")
    columns["columns"]["close"].append(1.0)
    tampered = [replace(leaked, available_at_cutoff=columns), *cases[1:]]
    checks = audit(tampered, "trading-smoke")["checks"]
    assert checks["trading_inputs_are_point_in_time"] is False


def test_advice_vocabulary_fails_trust_without_touching_correctness():
    case = _of_kind(generate_trading_cases(per_kind=1), "synthetic")
    summary = score_run([case], [_perfect(
        case, claims=["This is guaranteed to close higher; it is risk-free."])])
    row = summary["rows"][0]
    assert row["correctness"] == 1.0
    assert row["forbidden_claims_pass"] is False
    assert row["trust_pass"] is False
    assert set(REQUIRED_FORBIDDEN_CLAIMS) <= set(case.oracle.forbidden_claims)


def test_a_published_number_without_its_basis_is_not_a_trusted_answer():
    case = _of_kind(generate_trading_cases(per_kind=1), "longitudinal")
    facts = {key: value for key, value in case.oracle.required_facts.items()
             if key != "price_basis"}
    row = score_run([case], [_perfect(case, facts=facts)])["rows"][0]
    assert row["correctness"] == 1.0
    assert row["disclosures_pass"] is False
    assert row["disclosure_diagnostics"]["missing_typed_facts"] == ["price_basis"]


def test_non_comparable_history_must_abstain_before_it_may_answer():
    for ambiguity in ("venue", "corporate_action", "halt"):
        case = next(case for case in generate_trading_cases(per_kind=3)
                    if case.kind == "messy"
                    and case.oracle.required_facts["ambiguity"] == ambiguity)
        eager = score_run([case], [_perfect(
            case, status="answered", support="degraded",
            numbers={"next": 1.0})])["rows"][0]
        assert eager["correctness"] == 0.0
        assert eager["disposition_correct"] is False
        governed = score_run([case], [_perfect(case)])["rows"][0]
        assert governed["correctness"] == 1.0
        assert governed["repair_pass"] is True
        assert governed["final_resolved"] is True


def test_the_notable_instrument_is_the_largest_return_not_the_largest_move():
    cases = generate_trading_cases(per_kind=5)
    triage = [case for case in cases if case.kind == "multiseries"]
    for case in triage:
        streams = {name: values for name, values
                   in case.available_at_cutoff["columns"].items()
                   if name != "timestamp"}
        winner = case.oracle.choices["most_notable"]
        moves = {name: abs(values[-1] - values[-2]) for name, values in streams.items()}
        returns = {name: abs(values[-1] / values[-2] - 1.0)
                   for name, values in streams.items()}
        assert max(returns, key=returns.get) == winner
        assert max(moves, key=moves.get) != winner
    case = triage[0]
    decoy = max({name: abs(values[-1] - values[-2]) for name, values
                 in case.available_at_cutoff["columns"].items()
                 if name != "timestamp"}.items(), key=lambda item: item[1])[0]
    row = score_run([case], [_perfect(
        case, choices={"most_notable": decoy})])["rows"][0]
    assert row["correctness"] == 0.0
    # The corpus audit refuses a bundle where the level answer would win.
    inverted = replace(case, oracle=replace(case.oracle,
                                            choices={"most_notable": decoy}))
    assert audit([inverted, *cases[1:]], "trading-smoke")["checks"][
        "return_ranking_is_not_level_ranking"] is False


def test_a_corpus_that_permits_advice_is_not_trading_ready():
    cases = generate_trading_cases(per_kind=2)
    permissive = [replace(cases[0], oracle=replace(cases[0].oracle,
                                                   forbidden_claims=())),
                  *cases[1:]]
    assert audit(permissive, "trading-smoke")["checks"][
        "advice_vocabulary_is_forbidden"] is False


def test_perfect_matched_trading_run_passes_the_release_gate():
    cases = generate_trading_cases(per_kind=2)
    summary = score_run(cases, [_perfect(case) for case in cases], arm="oracle")
    assert summary["release_gate_pass"] is True
    assert summary["leakage_safety_gate_pass"] is True
    assert summary["correctness_mean_all_cases"] == 1.0
    assert summary["trust_pass_rate_all_cases"] == 1.0
    assert summary["final_workflow_resolution_rate"] == 1.0
    assert set(summary["by_domain"]) <= set(ASSET_CLASSES)


def test_control_arm_answers_from_the_evidence_the_case_ships():
    """The floor arm uses only declared calendar/basis/action evidence."""
    cases = generate_trading_cases(per_kind=3)
    synthetic = _of_kind(cases, "synthetic")
    answer = _answer(case_payload(synthetic))
    # The next session comes from the calendar block in the payload, not from
    # the generator's holiday table.
    assert (answer["facts"]["next_session"]
            == synthetic.oracle.required_facts["next_session"])
    assert answer["facts"]["price_basis"] == synthetic.oracle.required_facts[
        "price_basis"]

    triage = _of_kind(cases, "multiseries")
    assert (_answer(case_payload(triage))["choices"]["most_notable"]
            == triage.oracle.choices["most_notable"])

    for ambiguity in ("venue", "corporate_action", "halt"):
        case = next(case for case in cases if case.kind == "messy"
                    and case.oracle.required_facts["ambiguity"] == ambiguity)
        abstention = _answer(case_payload(case))
        assert abstention["status"] == "abstained"
        assert abstention["facts"]["ambiguity"] == ambiguity
        stage = next(stage for stage in case.stages if stage["name"] == "repair")
        oracle = Oracle.from_dict(stage["oracle"])
        repaired = _repair({**case_payload(case), "workflow_stage": "repair",
                            "revealed": stage["revealed"],
                            "answer_schema": stage["answer_schema"]})
        assert abs(repaired["numbers"]["next"] - oracle.numbers["next"]) <= (
            oracle.tolerances["next"])
        assert repaired["facts"] == oracle.required_facts

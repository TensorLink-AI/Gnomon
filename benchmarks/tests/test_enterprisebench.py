"""Offline tests for EnterpriseBench: harness, packs, and the bitemporal
contract. No network; scripted clients stand in for the model."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import benchmarks.enterprisebench.domains  # noqa: F401  (registers packs)
from benchmarks.enterprisebench import textgen  # noqa: E402
from benchmarks.enterprisebench.harness import (  # noqa: E402
    MODEL_ARMS,
    Case,
    ContextItem,
    admit_claims,
    as_of,
    base_prompt,
    grid_date,
    hidden_versions,
    leakage_lint,
    mase,
    parse_binary_decision,
    parse_candidate_answer,
    parse_compiled_answer,
    prompt_for,
    registry,
    run_domain,
    seasonal_naive_path,
    text_context,
    verify_arm_symmetry,
    verify_mase_affine_invariance,
)
from benchmarks.enterprisebench.run_enterprisebench import rollup  # noqa: E402

PACKS = registry()


def _args(tmp_path: Path, cases: int = 6, seed: int = 11,
          resume: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        seed=seed, cases=cases, model="scripted-test-model",
        output_dir=str(tmp_path / "out"), resume=resume, concurrency=2)


class ScriptedClient:
    """Offline stand-in with simple legible policies per arm: follow the
    oracle packet's governed recommendation when one is shown; answer
    seasonal-naive forecasts on the candidate arm; and, on the compiled
    arm, act as a scripted extractor with *known* errors — it reads the
    true rendered facts for the case (recovered deterministically from
    the pack), drops the first one, and adds two hallucinated claims,
    one implausible (the gate must reject it) and one plausible (the
    gate cannot know) — so admission and extraction scoring are
    exercised against exact expectations."""

    def __init__(self, pack=None, seed: int = 11, cases: int = 6):
        self.cases_by_series = {}
        if pack is not None:
            for case in pack.simulate(seed, cases)[0]:
                self.cases_by_series[case.series_id] = case

    def _compiled(self, case) -> str:
        resolved = as_of(case.items, case.cutoff)
        _, shown = text_context(case)
        kept = resolved[1:] if len(resolved) > 1 else resolved
        claims = [{"kind": item.kind, "value": shown[item.item_id],
                   "effective_from": grid_date(case, item.effective_from),
                   "effective_to": grid_date(case, item.effective_to)}
                  for item in kept]
        edge = grid_date(case, case.cutoff - 1)
        hallucination_kind = resolved[0].kind
        claims.append({"kind": hallucination_kind, "value": 9e9,
                       "effective_from": edge, "effective_to": edge})
        claims.append({"kind": hallucination_kind, "value": 123.4567,
                       "effective_from": edge, "effective_to": edge})
        return json.dumps({"claims": claims, "decision": {
            "event_expected": False, "first_event_step": None,
            "action": "monitor"}})

    def completions(self, messages, *, n=1):
        text = messages[-1]["content"]
        if '"claims"' in text:
            series = re.search(r"Series (\S+) at frequency",
                               text).group(1)
            return [self._compiled(self.cases_by_series[series])]
        if '"inner_forecasts"' in text:
            values = json.loads(re.search(
                r"values oldest first[^\n]*\n(\[.*?\])", text,
                re.DOTALL).group(1))
            cutoffs = [int(token) for token in re.findall(
                r"from cutoff (\d+)", text)]
            horizon = int(re.search(
                r"the next (\d+) observations", text).group(1))
            inner = [seasonal_naive_path(values[:cutoff], 7, horizon)
                     for cutoff in cutoffs]
            return [json.dumps({
                "inner_forecasts": inner,
                "forecast": seasonal_naive_path(values, 7, horizon)})]
        packet = re.search(r"Computed Gnomon evidence[^\n]*\n(\{[^\n]*\})",
                           text)
        if packet:
            payload = json.loads(packet.group(1))
            recommendation = (payload.get("governed_decision") or {}).get(
                "recommended_action") or "monitor"
            return [json.dumps({
                "event_expected": recommendation == "act",
                "first_event_step": 1 if recommendation == "act" else None,
                "action": recommendation})]
        return [json.dumps({"event_expected": False,
                            "first_event_step": None,
                            "action": "monitor"})]


# ---------------------------------------------------------------------------
# The as-of resolver (tested once, used by every pack and arm)
# ---------------------------------------------------------------------------

def _item(item_id, known_at, value=1.0, revises=None, kind="fact"):
    return ContextItem(item_id, kind, value, known_at, 0, 10,
                       revises=revises)


def test_as_of_exposes_only_the_version_known_at_the_cutoff():
    items = (
        _item("a1", known_at=2, value=40.0),
        _item("a2", known_at=5, value=25.0, revises="a1"),
        _item("a3", known_at=9, value=30.0, revises="a2"),
        _item("b1", known_at=7, value=7.0),
        _item("c1", known_at=8, value=3.0),
    )
    # Before the fact exists: nothing.
    assert as_of(items, 1) == []
    # First version only.
    assert [i.item_id for i in as_of(items, 3)] == ["a1"]
    # Revision replaces its predecessor the moment it is known.
    assert [(i.item_id, i.value) for i in as_of(items, 6)] == [("a2", 25.0)]
    # Chains resolve transitively; unrelated facts appear alongside.
    resolved = {i.item_id: i.value for i in as_of(items, 9)}
    assert resolved == {"a3": 30.0, "b1": 7.0, "c1": 3.0}
    # Hidden versions are exactly the post-cutoff ones.
    assert [i.item_id for i in hidden_versions(items, 6)] == ["b1", "c1",
                                                              "a3"]


def test_as_of_never_resurrects_a_superseded_version():
    items = (_item("v0", 1, 100.0), _item("v1", 3, 50.0, revises="v0"))
    for cutoff in (3, 4, 10):
        assert [(i.item_id, i.value) for i in as_of(items, cutoff)] == [
            ("v1", 50.0)]


# ---------------------------------------------------------------------------
# Registry: a new domain must be addable without touching the harness
# ---------------------------------------------------------------------------

def test_registry_populates_from_the_domains_package_only():
    assert set(PACKS) >= {"cloudcost", "cashflow"}
    harness_source = (Path(__file__).resolve().parents[1]
                      / "enterprisebench" / "harness.py").read_text(
                          encoding="utf-8")
    for name in PACKS:
        assert name not in harness_source, (
            f"harness.py names domain {name!r}; packs must register "
            "themselves so adding one never edits the harness")


def test_every_pack_declares_the_full_protocol():
    for name, pack in PACKS.items():
        assert pack.name == name
        assert pack.decision_kind in {"binary", "quantity"}
        assert pack.cost_model.break_even > 0
        assert pack.cost_model.names
        assert pack.decision_schema["instruction"].startswith("Return")
        assert pack.context_kinds
        for kind, spec in pack.context_kinds.items():
            low, high = spec["bounds"]
            assert low < high, (name, kind)
            assert spec["max_span"] > 0
        assert pack.config, f"{name} must disclose simulator parameters"


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(PACKS))
def test_simulator_is_deterministic_and_discloses_the_mix(name):
    pack = PACKS[name]
    first, provenance = pack.simulate(11, 8)
    again, _ = pack.simulate(11, 8)
    assert first == again
    assert provenance["outcome_distribution"]
    assert 0.0 < provenance["trap_share"] <= 0.3
    assert provenance["anonymization"].startswith(
        "per_case_seeded_positive_affine")
    assert "not_an_independence_claim" not in provenance["independence"]
    assert sum(provenance["cases_per_series"].values()) == len(first)
    for case in first:
        assert len(case.future) == case.horizon
        assert case.cutoff == len(case.values)
        assert case.meta["truth_event"] in (True, False)


@pytest.mark.parametrize("name", sorted(PACKS))
def test_trap_cases_flip_the_optimal_decision_by_construction(name):
    pack = PACKS[name]
    cases, _ = pack.simulate(11, 8)
    traps = [case for case in cases if case.trap]
    assert traps, "the disclosed trap share must actually materialize"
    for case in traps:
        assert case.trap_optimal is not None
        assert case.stale_optimal is not None
        assert pack.decision_scalar(case.trap_optimal) != \
            pack.decision_scalar(case.stale_optimal)
        # The as-of-correct optimal must equal the realized-truth
        # optimal: the trap punishes stale reading, not clairvoyance.
        assert pack.decision_scalar(case.trap_optimal) == \
            pack.decision_scalar(pack.cost_model.optimal(case))


def test_cloudcost_trap_threshold_flips_the_breach():
    pack = PACKS["cloudcost"]
    cases, _ = pack.simulate(11, 8)
    trap_case = next(case for case in cases if case.trap)
    resolved = as_of(trap_case.items, trap_case.cutoff)
    revision = next(item for item in resolved if item.trap)
    stale_items = tuple(item for item in trap_case.items
                        if item.item_id != revision.item_id)
    stale_view = as_of(stale_items, trap_case.cutoff)
    stale_threshold = (
        next(i.value for i in stale_view if i.kind == "commit_base")
        + sum(i.value for i in stale_view if i.kind == "commit_change"))
    as_of_breach = any(v > trap_case.threshold for v in trap_case.future)
    stale_breach = any(v > stale_threshold for v in trap_case.future)
    assert as_of_breach != stale_breach


# ---------------------------------------------------------------------------
# Leakage lint and arm symmetry
# ---------------------------------------------------------------------------

def _prompts(cases, pack):
    packet = {"support": "none", "forecast": []}
    blocks = {case.case_id: text_context(case)[0] for case in cases}
    return {(case.case_id, arm): prompt_for(case, pack, arm, packet,
                                            blocks[case.case_id])
            for case in cases for arm in MODEL_ARMS}, blocks


def test_leakage_lint_catches_planted_leaks_in_numeric_and_text_form():
    pack = PACKS["cloudcost"]
    cases, _ = pack.simulate(11, 4)
    prompts, _ = _prompts(cases, pack)
    leakage_lint(cases, pack, prompts)  # clean prompts pass

    case = cases[0]
    marker = json.dumps([round(float(v), 4) for v in case.future[:8]],
                        separators=(",", ":"))[1:-1]
    tampered = dict(prompts)
    tampered[(case.case_id, "model")] += f"\nhint: {marker}"
    with pytest.raises(ValueError, match="held-out future"):
        leakage_lint(cases, pack, tampered)

    hidden_case = next((c for c in cases
                        if hidden_versions(c.items, c.cutoff)), None)
    if hidden_case is None:
        cases, _ = pack.simulate(11, 12)
        hidden_case = next(c for c in cases
                           if hidden_versions(c.items, c.cutoff))
        prompts, _ = _prompts(cases, pack)
    hidden = hidden_versions(hidden_case.items, hidden_case.cutoff)[0]
    # Numeric form: the hidden version's value.
    tampered = dict(prompts)
    tampered[(hidden_case.case_id, "model_facts_oracle")] += (
        f"\nrevised to {hidden.value!r}")
    with pytest.raises(ValueError, match="post-cutoff item"):
        leakage_lint(cases, pack, tampered)
    # Text form: the hidden version's distinctive reference phrase.
    tampered = dict(prompts)
    tampered[(hidden_case.case_id, "model")] += (
        "\nlate memo " + textgen.ref_code(hidden_case.case_id,
                                          hidden.item_id))
    with pytest.raises(ValueError, match="text reference leaked"):
        leakage_lint(cases, pack, tampered)


def test_arm_symmetry_is_verified_and_tampering_fails():
    pack = PACKS["cashflow"]
    cases, _ = pack.simulate(11, 4)
    prompts, blocks = _prompts(cases, pack)
    verify_arm_symmetry(cases, pack, prompts, blocks)
    tampered = dict(prompts)
    case = cases[0]
    tampered[(case.case_id, "model")] = tampered[
        (case.case_id, "model")].replace(
            base_prompt(case, pack, blocks[case.case_id])[:40],
            "ALTERED", 1)
    with pytest.raises(ValueError, match="altered the shared question"):
        verify_arm_symmetry(cases, pack, tampered, blocks)


def test_text_rendering_round_trip_shows_only_cutoff_legal_numbers():
    """Generated text contains each resolved fact's shown number and
    reference, and nothing from post-cutoff versions."""
    for pack in PACKS.values():
        cases, _ = pack.simulate(11, 6)
        for case in cases:
            block, shown = text_context(case)
            for item in as_of(case.items, case.cutoff):
                assert textgen.ref_code(case.case_id, item.item_id) \
                    in block
                assert item.item_id in shown
            for item in hidden_versions(case.items, case.cutoff):
                assert textgen.ref_code(case.case_id, item.item_id) \
                    not in block
            # Deterministic rendering: same case, same text.
            assert text_context(case) == (block, shown)


def test_trap_revision_text_mentions_both_figures():
    pack = PACKS["cashflow"]
    cases, _ = pack.simulate(11, 8)
    trap_case = next(case for case in cases if case.trap)
    block, shown = text_context(trap_case)
    revision = next(item for item in as_of(trap_case.items,
                                           trap_case.cutoff)
                    if item.trap)
    prev = next(item for item in trap_case.items
                if item.item_id == revision.revises)
    from benchmarks.enterprisebench.textgen import _fmt
    assert _fmt(shown[revision.item_id]) in block
    assert _fmt(round(prev.value, 4)) in block


# ---------------------------------------------------------------------------
# MASE affine invariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(PACKS))
def test_seasonal_naive_reference_is_affine_invariant(name):
    pack = PACKS[name]
    cases, _ = pack.simulate(11, 3)
    for case in cases:
        verify_mase_affine_invariance(pack, case)


def test_mase_matches_hand_computation():
    history = [1.0, 2.0, 1.0, 2.0]
    assert mase([2.0], [3.0], history, 1) == 1.0
    assert seasonal_naive_path([1, 2, 3], 2, 4) == [2.0, 3.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Parsing: validated, degraded, never crashed
# ---------------------------------------------------------------------------

def _binary_case() -> Case:
    return Case("t-0", "test", "D", (1.0, 2.0), (3.0,), 1, (), 2.5,
                False, None, None, "s",
                meta={"truth_event": True, "truth_first_step": 1})


def test_malformed_binary_decisions_degrade_and_never_crash():
    case = _binary_case()
    assert parse_binary_decision({"event_expected": "maybe",
                                  "action": "act"}, case) is None
    for hostile in ("NaN", "Infinity", "true", "1e400", '"soon"', "0",
                    "-3"):
        parsed = parse_binary_decision(json.loads(
            '{"event_expected": true, "first_event_step": ' + hostile
            + ', "action": "act"}', parse_constant=lambda c: float(c)),
            case)
        assert parsed is not None and parsed["first_event_step"] is None


def test_candidate_answers_are_validated_not_guessed():
    case = Case("t-0", "test", "D", tuple(float(i) for i in range(40)),
                (1.0, 2.0), 2, (), None, False, None, None, "s",
                meta={"truth_event": False, "truth_first_step": None})
    assert parse_candidate_answer("no json", case)["valid"] is False
    wrong_length = json.dumps({"forecast": [1.0]})
    assert parse_candidate_answer(wrong_length, case)["valid"] is False
    missing_inner = json.dumps({"forecast": [1.0, 2.0]})
    parsed = parse_candidate_answer(missing_inner, case)
    assert parsed["valid"] is True and parsed["inner_forecasts"] is None
    good = json.dumps({"forecast": [1.0, 2.0],
                       "inner_forecasts": [[1.0, 2.0], [3.0, 4.0]]})
    parsed = parse_candidate_answer(good, case)
    assert parsed["inner_forecasts"] == [[1.0, 2.0], [3.0, 4.0]]


# ---------------------------------------------------------------------------
# Full offline runs: operational bar, resume, failure tolerance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(PACKS))
def test_a_matched_offline_run_prices_decisions_per_domain(name, tmp_path):
    pack = PACKS[name]
    summary = run_domain(pack, _args(tmp_path), ScriptedClient(pack))
    assert summary["domain"] == name
    assert set(summary["metrics"]) >= set(MODEL_ARMS) | {"engine"}
    for arm in MODEL_ARMS:
        entry = summary["metrics"][arm]
        assert entry["mean_regret"] >= 0.0
        assert 0.0 <= entry["action_optimal_rate"] <= 1.0
    references = summary["references"]
    assert set(references) >= {"engine", "seasonal_naive", "last_value",
                               "always_act", "never_act",
                               "hindsight_optimal"}
    assert references["hindsight_optimal"]["mean_regret"] == 0.0
    assert "withholding_rate" in references["engine"]
    verdicts = summary["verdicts"]
    for key in ("vs_model_alone", "vs_engine_alone",
                "vs_best_constant_policy", "candidate_admission_value",
                "compiled_vs_oracle_gap", "admission_value",
                "text_pipeline_integrity", "trap_integrity", "reading"):
        assert key in verdicts
    assert verdicts["treatment_arm"] == "model_facts_compiled"
    assert verdicts["text_pipeline_integrity"]["published_together"] \
        is True
    for comparison in (verdicts["vs_model_alone"],
                       verdicts["vs_engine_alone"]):
        assert comparison["paired_cases"] == 6
        assert 0.0 <= comparison["exact_sign_p"] <= 1.0
    design = summary["design"]
    assert design["engine_packet_is_production_output"] is True
    assert design["held_out_future_absent_from_prompts_verified"] is True
    rows = [json.loads(line) for line in
            (tmp_path / "out" / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 6 * len(MODEL_ARMS)
    assert all(row["dataset"] ==
               summary["provenance"]["dataset_identity"] for row in rows)


def test_resume_rejects_rows_from_a_different_dataset_or_model(tmp_path):
    pack = PACKS["cloudcost"]
    run_domain(pack, _args(tmp_path, cases=6), ScriptedClient(pack))
    rows_path = tmp_path / "out" / "rows.jsonl"
    first_rows = rows_path.read_text().splitlines()
    assert len(first_rows) == 6 * len(MODEL_ARMS)
    summary = run_domain(pack, _args(tmp_path, cases=4, resume=True),
                         ScriptedClient(pack, cases=4))
    # Every old row was rejected (different dataset identity): the
    # 4-case run answered all its own pairs from scratch.
    assert summary["verdicts"]["vs_model_alone"]["paired_cases"] == 4
    assert len(rows_path.read_text().splitlines()) == \
        len(first_rows) + 4 * len(MODEL_ARMS)


def test_one_dead_call_fails_loudly_and_resume_finishes(tmp_path):
    import threading

    class FlakyClient(ScriptedClient):
        def __init__(self, pack):
            super().__init__(pack)
            self.calls = 0
            self._lock = threading.Lock()

        def completions(self, messages, *, n=1):
            with self._lock:
                self.calls += 1
                ordinal = self.calls
            if ordinal == 3:
                raise RuntimeError("endpoint fell over")
            return super().completions(messages, n=n)

    pack = PACKS["cashflow"]
    with pytest.raises(RuntimeError, match="rerun with --resume"):
        run_domain(pack, _args(tmp_path), FlakyClient(pack))
    rows_path = tmp_path / "out" / "rows.jsonl"
    survived = rows_path.read_text().splitlines()
    assert len(survived) == 6 * len(MODEL_ARMS) - 1
    # A stale row and a crash-truncated line must both be rejected.
    with rows_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"case_id": "cf999-0000", "arm": "model",
                                 "cost": 0.0}) + "\n")
        handle.write('{"case_id": "cf11-')
    summary = run_domain(pack, _args(tmp_path, resume=True),
                         ScriptedClient(pack))
    assert summary["verdicts"]["vs_model_alone"]["paired_cases"] == 6


def test_candidate_over_promise_is_detected_for_a_cheating_backtest(
        tmp_path):
    """A model that copies the visible answers into its inner-fold
    'backtest' gets admitted on a stellar promise; the row must then
    expose the gap between promise and out-of-sample delivery."""

    class CheatingClient(ScriptedClient):
        def completions(self, messages, *, n=1):
            text = messages[-1]["content"]
            if '"inner_forecasts"' not in text:
                return super().completions(messages, n=n)
            values = json.loads(re.search(
                r"values oldest first[^\n]*\n(\[.*?\])", text,
                re.DOTALL).group(1))
            cutoffs = [int(token) for token in re.findall(
                r"from cutoff (\d+)", text)]
            horizon = int(re.search(
                r"the next (\d+) observations", text).group(1))
            inner = [values[c:c + horizon] for c in cutoffs]  # the leak
            return [json.dumps({
                "inner_forecasts": inner,
                "forecast": [values[-1] * 3.0] * horizon})]

    pack = PACKS["cloudcost"]
    summary = run_domain(pack, _args(tmp_path), CheatingClient(pack))
    candidate = summary["metrics"]["governed_candidate"]
    assert candidate["admission_rate"] == 1.0
    assert candidate["mean_over_promise"] > 0.5
    assert summary["verdicts"]["candidate_admission_value"][
        "mean_over_promise"] > 0.5


# ---------------------------------------------------------------------------
# Compiled arm: admission gate and extraction scoring
# ---------------------------------------------------------------------------

def test_admission_gate_rejects_on_structure_and_plausibility_alone():
    pack = PACKS["cloudcost"]
    case = pack.simulate(11, 2)[0][0]
    date = grid_date(case, case.cutoff - 1)
    span_start = grid_date(case, 0)
    span_end = grid_date(case, case.cutoff + case.horizon - 1)
    claims = [
        "not an object",
        {"kind": "nonsense", "value": 5.0,
         "effective_from": date, "effective_to": date},
        {"kind": "commit_base", "value": float("nan"),
         "effective_from": date, "effective_to": date},
        {"kind": "commit_base", "value": True,
         "effective_from": date, "effective_to": date},
        {"kind": "commit_base", "value": 5.0,
         "effective_from": "not-a-date", "effective_to": date},
        {"kind": "commit_base", "value": 9e9,
         "effective_from": date, "effective_to": date},
        {"kind": "deploy_uplift", "value": 10.0,
         "effective_from": span_start, "effective_to": span_end},
        {"kind": "commit_base", "value": 1234.5,
         "effective_from": date, "effective_to": date},
    ]
    gate = admit_claims(claims, case, pack)
    reasons = [entry["reason"] for entry in gate["rejected"]]
    assert reasons == ["schema_not_object", "schema", "schema", "schema",
                       "schema", "implausible_value", "effect_span_prior"]
    assert [item.value for item in gate["admitted"]] == [1234.5]
    assert gate["admitted"][0].known_at == case.cutoff


def test_parse_compiled_answer_degrades_each_part_independently():
    pack = PACKS["cloudcost"]
    case = pack.simulate(11, 2)[0][0]
    both = json.dumps({
        "claims": [{"kind": "commit_base", "value": 1.0}],
        "decision": {"event_expected": True, "first_event_step": 2,
                     "action": "act"}})
    parsed = parse_compiled_answer(both, pack, case)
    assert parsed["claims_valid"] and parsed["decision_valid"]
    assert parsed["decision"]["action"] == "act"
    # Valid claims, malformed decision: keep the claims, degrade the
    # decision to the recorded no-action default.
    half = json.dumps({"claims": [], "decision": {"action": "perhaps"}})
    parsed = parse_compiled_answer(half, pack, case)
    assert parsed["claims_valid"] and not parsed["decision_valid"]
    assert parsed["decision"]["action"] == "monitor"
    # Malformed claims, valid decision: the reverse.
    other = json.dumps({"claims": "none",
                        "decision": {"event_expected": False,
                                     "first_event_step": None,
                                     "action": "monitor"}})
    parsed = parse_compiled_answer(other, pack, case)
    assert not parsed["claims_valid"] and parsed["decision_valid"]
    parsed = parse_compiled_answer("no json at all", pack, case)
    assert not parsed["claims_valid"] and not parsed["decision_valid"]


@pytest.mark.parametrize("name", sorted(PACKS))
def test_compiled_extraction_is_scored_against_exact_expectations(
        name, tmp_path):
    """The scripted extractor has known errors: perfect values for every
    kept fact, one dropped fact, and two hallucinated claims of which
    the gate can only reject the implausible one. Extraction scoring
    must reproduce those numbers exactly."""
    pack = PACKS[name]
    summary = run_domain(pack, _args(tmp_path), ScriptedClient(pack))
    extraction = summary["metrics"]["model_facts_compiled"]["extraction"]
    assert extraction["value_relative_error"] == 0.0
    assert extraction["window_error_steps"] == 0.0
    assert extraction["hallucinated_claims"] == 2 * 6
    assert extraction["hallucination_admission_rate"] == 0.5
    assert extraction["missed_rate"] > 0.0
    if extraction["revision_correct_rate"] is not None:
        assert extraction["revision_correct_rate"] == 1.0
    assert extraction["claims_invalid_rate"] == 0.0
    admission = summary["verdicts"]["admission_value"]
    assert admission["paired_cases"] == 6
    assert admission["gated_mean_cost"] is not None
    assert admission["raw_mean_cost"] is not None
    gap = summary["verdicts"]["compiled_vs_oracle_gap"]
    assert gap is not None and "gap_mean_cost" in gap
    model_own = summary["metrics"]["model_facts_compiled"]["model_own"]
    assert model_own["decision_invalid_rate"] == 0.0


def test_rollup_refuses_a_single_aggregate_number(tmp_path):
    pack = PACKS["cloudcost"]
    summary = run_domain(pack, _args(tmp_path), ScriptedClient(pack))
    combined = rollup({"cloudcost": summary}, tmp_path / "rollup")
    assert combined["aggregation"]["no_single_aggregate_number"] is True
    assert "cloudcost" in combined["domains"]
    written = json.loads(
        (tmp_path / "rollup" / "summary.json").read_text())
    assert written["aggregation"]["no_single_aggregate_number"] is True

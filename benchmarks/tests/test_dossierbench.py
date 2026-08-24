from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.dossierbench.run_dossierbench import (  # noqa: E402
    ARMS,
    OPTIONS,
    answer_dossier_arm,
    computed_evidence,
    generate_cases,
    generate_real_cases,
    run,
    verify_no_future_leakage,
)


def _args(tmp_path: Path, cases: int = 12,
          source: str = "real") -> argparse.Namespace:
    return argparse.Namespace(
        seed=20260825, cases=cases, source=source, data_dir=None,
        output_dir=str(tmp_path / "out"), resume=False, concurrency=2,
        model="scripted-test-model")


class ScriptedClient:
    """Deterministic offline stand-in: reads the prompt like a model would.

    Control/conclusion arms answer the first listed option; the dossier arm
    selects the compatible interpretation with the strongest held-out fit
    and cites its supporting evidence.
    """

    def __init__(self) -> None:
        self.calls = 0

    def completions(self, messages, *, n=1):
        self.calls += 1
        text = messages[-1]["content"]
        packet_match = re.search(
            r"evidence dossier:\n(\{.*?\})\nFollow", text, re.DOTALL)
        if packet_match is None:
            options = re.search(r"Options: (.+?)\.", text).group(1)
            return [json.dumps({"value": options.split(", ")[0]})]
        packet = json.loads(packet_match.group(1))
        contract = packet["selection_contract"]
        if contract["canonical"]["role"] == "binding":
            return [json.dumps({"value": contract["canonical"]["value"]})]
        rows = [row for row in packet["interpretations"]
                if row["compatible"] and row.get("supporting")]
        rows.sort(key=lambda row: -(row.get("held_out_fit") or 0.0))
        chosen = rows[0] if rows else {"value": contract["canonical"]["value"],
                                       "supporting": []}
        return [json.dumps({"value": chosen["value"],
                            "cited_evidence": chosen["supporting"]})]


class FailTwiceClient:
    """First an ungrounded selection, then still ungrounded: forces the
    fallback path."""

    def completions(self, messages, *, n=1):
        return [json.dumps({"value": "sideways", "cited_evidence": []})]


class RepairableClient:
    """Ungrounded first, grounded after the repair instruction."""

    def __init__(self, packet: dict) -> None:
        self.packet = packet
        self.turn = 0

    def completions(self, messages, *, n=1):
        self.turn += 1
        if self.turn == 1:
            return [json.dumps({"value": "sideways"})]
        rows = [row for row in self.packet["interpretations"]
                if row["compatible"] and row.get("supporting")]
        chosen = rows[0]
        return [json.dumps({"value": chosen["value"],
                            "cited_evidence": chosen["supporting"]})]


def test_real_cases_are_deterministic_with_outcome_labels() -> None:
    first, provenance, futures = generate_real_cases(7, 24)
    again, _, _ = generate_real_cases(7, 24)
    assert first == again
    for case in first:
        assert case.truth in OPTIONS[case.property]
        assert case.horizon and case.horizon >= 12
        assert case.label_confidence in {"supported", "weak"}
        if case.label_confidence == "weak":
            assert case.truth in {"similar", "constant", "stable"}
        assert len(futures[case.case_id]) == case.horizon
    assert provenance["labeling"].startswith("realized_future_window")
    assert provenance["anonymization"] == \
        "seeded_positive_affine_transform_per_case"
    assert set(provenance["corpus_series"]) >= {
        "nile_annual_flow", "sunspots_yearly", "co2_weekly_mauna_loa"}


def test_the_held_out_future_never_reaches_a_prompt() -> None:
    cases, _, futures = generate_real_cases(11, 12)
    computed = {case.case_id: computed_evidence(case) for case in cases}
    verify_no_future_leakage(cases, futures, computed)


def test_synthetic_cases_remain_available_as_a_diagnostic_mode() -> None:
    cases = generate_cases(7, 20)
    assert cases == generate_cases(7, 20)
    for case in cases:
        assert case.truth in OPTIONS[case.property]
        assert case.horizon is None


def test_a_matched_offline_run_produces_the_full_summary(tmp_path) -> None:
    summary = run(_args(tmp_path), client=ScriptedClient())
    assert summary["source"] == "real"
    assert set(summary["metrics"]) == set(ARMS)
    for arm in ARMS:
        assert 0.0 <= summary["metrics"][arm]["accuracy"] <= 1.0
    references = summary["references"]
    assert set(references) == {"chance", "always_majority",
                               "copy_conclusion", "copy_discriminator"}
    assert all(0.0 <= value <= 1.0 for value in references.values())
    assert {row["comparison"] for row in summary["paired"]} == {
        "control_vs_dossier", "conclusion_vs_dossier", "control_vs_conclusion"}
    verdicts = summary["verdicts"]
    assert set(verdicts) >= {"uplift_over_model_alone",
                             "uplift_over_conclusion_packet",
                             "model_beyond_mechanism"}
    design = summary["design"]
    assert design["arms_differ_by_packet_block_only"] is True
    assert design["truth_is_realized_held_out_future"] is True
    assert design["held_out_future_absent_from_prompts_verified"] is True
    assert summary["provenance"]["cases"]["skipped"] is not None
    rows = [json.loads(line) for line in
            (tmp_path / "out" / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 12 * len(ARMS)
    repair = summary["metrics"]["dossier"]["repair_loop"]
    assert sum(repair.values()) == 12


def test_the_synthetic_mode_still_runs_end_to_end(tmp_path) -> None:
    summary = run(_args(tmp_path, source="synthetic"),
                  client=ScriptedClient())
    assert summary["source"] == "synthetic"
    assert summary["design"]["synthetic_generator"] is True


def _non_binding_case():
    cases, _, _ = generate_real_cases(20260825, 40)
    for case in cases:
        computed = computed_evidence(case)
        contract = computed["packet"]["selection_contract"]
        if contract["canonical"]["role"] != "binding":
            return case, computed
    raise AssertionError("generator produced no non-binding case")


def test_the_dossier_arm_repairs_a_rejected_selection_once() -> None:
    case, computed = _non_binding_case()
    client = RepairableClient(computed["packet"])
    outcome = answer_dossier_arm(case, computed, client)
    assert outcome["stage"] == "repaired"
    assert outcome["calls"] == 2
    assert outcome["violations"]


def test_a_twice_ungrounded_selection_falls_back_to_the_canonical() -> None:
    case, computed = _non_binding_case()
    outcome = answer_dossier_arm(case, computed, FailTwiceClient())
    assert outcome["stage"] == "canonical_fallback"
    assert outcome["value"] == str(
        computed["packet"]["selection_contract"]["canonical"]["value"]).lower()
    assert outcome["calls"] == 2

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
    run,
)


def _args(tmp_path: Path, cases: int = 12) -> argparse.Namespace:
    return argparse.Namespace(
        seed=20260825, cases=cases, output_dir=str(tmp_path / "out"),
        resume=False, concurrency=2, model="scripted-test-model")


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


def test_cases_are_deterministic_with_known_truths() -> None:
    first = generate_cases(7, 20)
    assert first == generate_cases(7, 20)
    for case in first:
        assert case.truth in OPTIONS[case.property]


def test_a_matched_offline_run_produces_the_full_summary(tmp_path) -> None:
    summary = run(_args(tmp_path), client=ScriptedClient())
    assert set(summary["metrics"]) == set(ARMS)
    for arm in ARMS:
        assert 0.0 <= summary["metrics"][arm]["accuracy"] <= 1.0
    references = summary["references"]
    assert references["copy_discriminator"] > references["chance"]
    assert {row["comparison"] for row in summary["paired"]} == {
        "control_vs_dossier", "conclusion_vs_dossier", "control_vs_conclusion"}
    verdicts = summary["verdicts"]
    assert set(verdicts) >= {"uplift_over_model_alone",
                             "uplift_over_conclusion_packet",
                             "model_beyond_mechanism"}
    assert summary["design"]["arms_differ_by_packet_block_only"] is True
    rows = [json.loads(line) for line in
            (tmp_path / "out" / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 12 * len(ARMS)
    repair = summary["metrics"]["dossier"]["repair_loop"]
    assert sum(repair.values()) == 12


def _non_binding_case():
    for case in generate_cases(20260825, 40):
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

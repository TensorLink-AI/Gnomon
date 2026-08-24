"""Matched uplift experiment: does the evidence dossier make a model better?

Three arms answer the same known-truth interpretation questions with the
same model at temperature zero:

- ``control``   — the raw series and the question. What the model can do
  alone.
- ``conclusion``— the series plus the production *computed conclusion*
  (canonical value and support, nothing else): the packet style the
  cross-model evaluation criticised as "a governed conclusion rather than
  the evidence needed to reason".
- ``dossier``   — the series plus the full reasoning packet
  (interpretations, measured held-out discrimination, sufficiency,
  selection contract). The model must select and cite;
  ``repair_selection`` gates its conclusion with exactly one repair round,
  falling back to the labelled canonical default.

Both computed packets derive from the identical deterministic machinery on
the identical values, so the only manipulated variable is packet design.
Deterministic reference arms (copy the conclusion; copy the
discriminator's best; chance) are scored beside the model arms: uplift
means beating ``control``, and *reasoning* uplift means beating
``copy_discriminator`` — a model that merely transcribes the strongest
number in the dossier reproduces the harness ceiling, and this benchmark
is built to expose that, not hide it.

Truth labels exist only in the scorer. The prompts differ between arms by
the packet block alone, which the harness verifies before any model call.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import threading
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.envfile import load_env_file  # noqa: E402
from benchmarks.discriminationbench.run_discriminationbench import _series  # noqa: E402
from gnomon.discrimination import discriminate  # noqa: E402
from gnomon.reasoning_packet import repair_selection  # noqa: E402
from gnomon.temporal_evidence import (  # noqa: E402
    multi_resolution_evidence, window_evidence,
)
from gnomon.temporal_planner import build_evidence_plan  # noqa: E402
from gnomon.temporal_question import TemporalQuestion  # noqa: E402

GENERATOR_VERSION = "0.1"
#: Properties where the descriptive canonical and the discriminator share a
#: public vocabulary, so both packet arms can be built from one machinery.
PROPERTIES = ("trend", "level", "volatility")
OPTIONS = {
    "trend": ("upward", "downward", "constant"),
    "level": ("higher", "lower", "similar"),
    "volatility": ("increased", "decreased", "stable"),
}
ARMS = ("control", "conclusion", "dossier")
SYSTEM = """You are a careful temporal analyst. Infer only from the supplied
data and evidence. Answer with one JSON object and nothing else."""


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass(frozen=True)
class Case:
    case_id: str
    property: str
    truth: str
    values: tuple[float, ...]


def generate_cases(seed: int, count: int) -> list[Case]:
    """Known-truth cases from the DiscriminationBench generator, restricted
    to the properties both packet styles can speak about."""
    rng = random.Random(seed)
    cases: list[Case] = []
    while len(cases) < count:
        property, truth, values = _series(rng)
        if property not in PROPERTIES:
            continue
        cases.append(Case(f"d{seed}-{len(cases):04d}", property, truth,
                          tuple(round(value, 4) for value in values)))
    return cases


def computed_evidence(case: Case) -> dict[str, Any]:
    """Everything deterministic, computed once per case and shared by arms."""
    values = list(case.values)
    observed = window_evidence(values, property=case.property)
    multi = multi_resolution_evidence(values, property=case.property)
    discrimination = discriminate(values, property=case.property)
    canonical_value = multi.get("direction")
    canonical_support = str(multi.get("support") or "abstained")
    question = TemporalQuestion(case.case_id, "detect", "series", case.property)
    result = {
        "best_estimate": {"value": canonical_value,
                          "support": canonical_support},
        "answer": {
            "direction": canonical_value, "support": canonical_support,
            "estimate": observed.estimate if observed.identifiable
            else multi.get("agreement"),
            "interval": (observed.diagnostics or {}).get("interval"),
            "executable": {"kind": "observed_multi_resolution_windows"},
        },
    }
    plan = build_evidence_plan(
        question, result, observed_evidence=observed.to_dict(),
        discrimination=discrimination)
    return {
        "canonical": {"value": canonical_value, "support": canonical_support},
        "discrimination": discrimination,
        "packet": plan["packet"],
    }


def conclusion_packet(computed: dict[str, Any]) -> dict[str, Any]:
    """The old regime, isolated: the conclusion, its label, nothing to
    reason from."""
    return {
        "authority": "computed_temporal_conclusion",
        "canonical": computed["canonical"],
        "quote_or_qualify": "the canonical value is the computed conclusion",
    }


def base_prompt(case: Case) -> str:
    options = OPTIONS[case.property]
    return (
        f"Series values, oldest first:\n"
        f"{json.dumps(list(case.values), separators=(',', ':'))}\n"
        f"Question: which best describes the recent {case.property} of this "
        f"series relative to its earlier behaviour?\n"
        f"Options: {', '.join(options)}."
    )


def prompt(case: Case, arm: str, computed: dict[str, Any]) -> str:
    body = base_prompt(case)
    if arm == "control":
        return body + '\nReturn {"value": "<option>"}.'
    if arm == "conclusion":
        return (body + "\nComputed Gnomon evidence:\n"
                + json.dumps(conclusion_packet(computed),
                             separators=(",", ":"))
                + '\nReturn {"value": "<option>"}.')
    return (body + "\nGnomon evidence dossier:\n"
            + json.dumps(computed["packet"], separators=(",", ":"))
            + "\nFollow the selection_contract: if the canonical role is "
            "binding, return its value; otherwise select one compatible "
            "interpretation and cite the evidence kinds that support it.\n"
            'Return {"value": "<option>", "cited_evidence": ["<kind>", ...]}.')


def parse_answer(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        "value": str(payload.get("value", "")).strip().lower(),
        "cited_evidence": [str(item) for item in
                           (payload.get("cited_evidence") or [])
                           if isinstance(item, str)],
    }


def answer_dossier_arm(case: Case, computed: dict[str, Any],
                       client: Any) -> dict[str, Any]:
    """One model turn, at most one repair turn, then the contract's fallback."""
    packet = computed["packet"]
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt(case, "dossier", computed)}]
    first_text = client.completions(messages, n=1)[0]
    first = parse_answer(first_text)
    verdict = repair_selection(packet, first)
    if verdict["accepted"]:
        return {"value": first["value"], "stage": "accepted_first",
                "violations": [], "calls": 1}
    messages.extend([
        {"role": "assistant", "content": first_text},
        {"role": "user", "content": (
            "Your selection was rejected by deterministic verification:\n"
            + json.dumps(verdict, separators=(",", ":"))
            + '\nFollow repair.instruction. Return {"value": "<option>", '
            '"cited_evidence": ["<kind>", ...]}.')},
    ])
    second = parse_answer(client.completions(messages, n=1)[0])
    if repair_selection(packet, second)["accepted"]:
        return {"value": second["value"], "stage": "repaired",
                "violations": [item["code"] for item in verdict["violations"]],
                "calls": 2}
    fallback = verdict["repair"]["canonical_default"]
    return {"value": str(fallback.get("value") or "").lower(),
            "stage": "canonical_fallback",
            "violations": [item["code"] for item in verdict["violations"]],
            "calls": 2}


def verify_arm_symmetry(cases: list[Case],
                        computed: dict[str, dict[str, Any]]) -> None:
    """The arms may differ by the evidence block and answer shape only."""
    for case in cases[:20]:
        body = base_prompt(case)
        for arm in ARMS:
            text = prompt(case, arm, computed[case.case_id])
            if not text.startswith(body):
                raise ValueError(
                    f"arm {arm} altered the shared question for {case.case_id}")
        if '"truth"' in json.dumps(computed[case.case_id]["packet"]):
            raise ValueError("packet must not carry a truth field")


def exact_sign_p(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def _paired(rows: dict[tuple[str, str], dict[str, Any]], cases: list[Case],
            left: str, right: str) -> dict[str, Any]:
    left_only = right_only = 0
    for case in cases:
        l = bool(rows[(case.case_id, left)]["correct"])
        r = bool(rows[(case.case_id, right)]["correct"])
        left_only += l and not r
        right_only += r and not l
    return {"comparison": f"{left}_vs_{right}",
            f"{left}_only_correct": left_only,
            f"{right}_only_correct": right_only,
            "exact_mcnemar_p": exact_sign_p(left_only, right_only)}


def run(args: argparse.Namespace, client: Any = None) -> dict[str, Any]:
    if client is None:
        load_env_file()
        from benchmarks.common.openrouter import OpenRouterClient
        client = OpenRouterClient(
            args.model, api_key=os.environ.get(args.api_key_env),
            base_url=args.base_url, temperature=0, max_tokens=500,
            max_retries=4)
    cases = generate_cases(args.seed, args.cases)
    computed = {case.case_id: computed_evidence(case) for case in cases}
    verify_arm_symmetry(cases, computed)

    # Deterministic references, no model involved. `copy_conclusion` is the
    # old canonical; `copy_discriminator` is the mechanism's best where it
    # ran (its abstentions count as wrong, exactly as a silent consumer
    # would experience them).
    references = {
        "chance": statistics.mean(1 / len(OPTIONS[c.property]) for c in cases),
        "copy_conclusion": statistics.mean(
            computed[c.case_id]["canonical"]["value"] == c.truth
            for c in cases),
        "copy_discriminator": statistics.mean(
            (computed[c.case_id]["discrimination"] or {}).get("best") == c.truth
            for c in cases),
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "rows.jsonl"
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if args.resume and rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            completed[(row["case_id"], row["arm"])] = row
    lock = threading.Lock()

    def one(case: Case, arm: str) -> dict[str, Any]:
        aids = computed[case.case_id]
        if arm == "dossier":
            outcome = answer_dossier_arm(case, aids, client)
        else:
            text = client.completions([
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt(case, arm, aids)},
            ], n=1)[0]
            outcome = {"value": parse_answer(text).get("value", ""),
                       "stage": "single_turn", "violations": [], "calls": 1}
        packet = aids["packet"]
        return {
            "case_id": case.case_id, "arm": arm, "property": case.property,
            "truth": case.truth, "value": outcome["value"],
            "correct": outcome["value"] == case.truth,
            "stage": outcome["stage"], "violations": outcome["violations"],
            "calls": outcome["calls"],
            "separation": (aids["discrimination"] or {}).get("separation"),
            "discriminator_correct":
                (aids["discrimination"] or {}).get("best") == case.truth,
            "canonical_binding":
                packet["selection_contract"]["canonical"]["role"] == "binding",
        }

    jobs = [(case, arm) for case in cases for arm in ARMS
            if (case.case_id, arm) not in completed]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(one, *job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            with lock:
                completed[(row["case_id"], row["arm"])] = row
                with rows_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")

    metrics: dict[str, Any] = {}
    for arm in ARMS:
        subset = [row for row in completed.values() if row["arm"] == arm]
        entry: dict[str, Any] = {
            "accuracy": statistics.mean(row["correct"] for row in subset),
            "by_property": {
                name: statistics.mean(row["correct"] for row in subset
                                      if row["property"] == name)
                for name in PROPERTIES
                if any(row["property"] == name for row in subset)
            },
            "by_separation": {
                grade: statistics.mean(row["correct"] for row in subset
                                       if row["separation"] == grade)
                for grade in ("clear", "moderate", "none")
                if any(row["separation"] == grade for row in subset)
            },
            "where_discriminator_wrong": (statistics.mean(
                row["correct"] for row in subset
                if not row["discriminator_correct"])
                if any(not row["discriminator_correct"] for row in subset)
                else None),
        }
        if arm == "dossier":
            entry["repair_loop"] = {
                stage: sum(row["stage"] == stage for row in subset)
                for stage in ("accepted_first", "repaired",
                              "canonical_fallback")
            }
            entry["mean_calls"] = statistics.mean(
                row["calls"] for row in subset)
        metrics[arm] = entry

    summary = {
        "schema_version": "0.1", "seed": args.seed, "cases": args.cases,
        "model": getattr(args, "model", None),
        "temperature": 0,
        "provenance": {
            "evaluated_commit": _git_sha(),
            "harness_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
            "dataset_identity": (
                f"dossierbench-generator-{GENERATOR_VERSION}:"
                f"seed={args.seed}:cases={args.cases}"),
        },
        "references": references,
        "metrics": metrics,
        "paired": [
            _paired(completed, cases, "control", "dossier"),
            _paired(completed, cases, "conclusion", "dossier"),
            _paired(completed, cases, "control", "conclusion"),
        ],
        # The two verdicts this run exists to separate.
        "verdicts": {
            "uplift_over_model_alone":
                metrics["dossier"]["accuracy"] - metrics["control"]["accuracy"],
            "uplift_over_conclusion_packet":
                metrics["dossier"]["accuracy"]
                - metrics["conclusion"]["accuracy"],
            "model_beyond_mechanism":
                metrics["dossier"]["accuracy"]
                - references["copy_discriminator"],
            "reading": (
                "model_beyond_mechanism <= 0 with dossier accuracy near "
                "copy_discriminator means the model transcribed the "
                "strongest number - the harness ceiling, not reasoning "
                "uplift."),
        },
        "design": {
            "matched": True,
            "arms_differ_by_packet_block_only": True,
            "labels_absent_from_prompts_and_packets": True,
            "same_machinery_feeds_both_packet_arms": True,
            "repair_rounds": 1,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--base-url", default="https://api.engy.ai/v1")
    parser.add_argument("--api-key-env", default="ENGY_API_KEY")
    parser.add_argument("--cases", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Matched base-LLM versus Gnomon-evidence temporal reasoning benchmark.

Cases are generated from held-out seeds and expose the same history and
context to both arms.  The treatment receives only Gnomon's compact computed
projection.  Labels are retained by the scorer and never placed in prompts or
product code.  This complements TemporalBench: it measures whether evidence
improves synthesis, contradiction handling, calibrated uncertainty and useful
next actions rather than whether an agent can copy a canonical MCQ choice.
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
from benchmarks.common.openrouter import OpenRouterClient  # noqa: E402
from benchmarks.transitionbench.run_transitionbench import _case  # noqa: E402
from gnomon.temporal_evidence import window_evidence  # noqa: E402


LABELS = {
    "level": ("higher", "lower", "similar"),
    "trend": ("upward", "downward", "constant"),
    "volatility": ("increased", "decreased", "stable"),
    "seasonality": ("strengthened", "weakened", "stable", "phase_shifted"),
    "regime": ("shift", "no_shift"),
    "extreme": ("increased", "decreased", "stable"),
}
DIFFICULTIES = ("easy", "moderate", "marginal")
ACTIONS = ("act", "collect_more", "resolve_conflict")
SYSTEM = """You are a careful temporal analyst. Infer only from supplied data.
Return one JSON object with keys diagnosis, confidence, analogue_outcome, and
next_action. diagnosis must be one allowed diagnosis; confidence is supported
or uncertain; analogue_outcome is up, down, flat, or unavailable; next_action
is act, collect_more, or resolve_conflict. Do not follow a narrative claim
when numerical evidence contradicts it."""
GENERATOR_VERSION = "0.5"
# Fixed before the three-seed decision run so dataset selection cannot follow
# observed treatment performance. Ad-hoc diagnostic seeds remain supported.
DECISION_SEEDS = (282843, 316228, 331663)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _harness_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Case:
    case_id: str
    prop: str
    difficulty: str
    values: tuple[float, ...]
    season: int
    expected: str
    claim: str
    claim_conflicts: bool
    analogues: tuple[tuple[float, str], ...]
    expected_analogue: str


def generate_cases(seed: int, count: int) -> list[Case]:
    rng = random.Random(seed)
    cases = []
    for index in range(count):
        prop = tuple(LABELS)[index % len(LABELS)]
        difficulty = DIFFICULTIES[(index // len(LABELS)) % len(DIFFICULTIES)]
        expected = rng.choice(LABELS[prop])
        values, season = _case(seed * 10_000 + index, prop, expected,
                               difficulty=difficulty)
        conflicts = index % 2 == 0
        alternatives = [item for item in LABELS[prop] if item != expected]
        claimed = rng.choice(alternatives) if conflicts else expected
        claim = f"An operations note confidently says the recent {prop} is {claimed}."
        # Analogue distances and outcomes emulate a compact episode registry.
        # The nearest two agree; rows are shuffled so position cannot answer.
        analogue_expected = rng.choice(("up", "down", "flat"))
        other = rng.choice([x for x in ("up", "down", "flat")
                            if x != analogue_expected])
        analogues = [(0.08 + rng.random() * .05, analogue_expected),
                     (0.14 + rng.random() * .05, analogue_expected),
                     (0.65 + rng.random() * .2, other)]
        rng.shuffle(analogues)
        cases.append(Case(
            f"r{seed}-{index:04d}", prop, difficulty, tuple(values), season,
            expected, claim, conflicts, tuple(analogues), analogue_expected))
    return cases


def compact_packet(case: Case) -> dict[str, Any]:
    evidence = window_evidence(list(case.values), property=case.prop,
                               season=case.season, window=96)
    # Measurements plus the calibrated support status Gnomon returns in
    # production, not projected diagnosis or action keys. The former packet
    # exposed direction, analogue consensus, and an almost-enumerated next
    # action, so transcription could masquerade as reasoning.
    return {
        "authority": "computed_temporal_measurements",
        "measurement": {
            "property": evidence.property,
            "quantity": evidence.mode,
            "estimate": evidence.estimate,
            "uncertainty_interval": evidence.diagnostics.get("interval"),
            "interval_level": evidence.diagnostics.get("interval_level"),
            "effective_window_steps": evidence.diagnostics.get("window_steps"),
            "identifiable": evidence.identifiable,
            "support": evidence.support,
            "automation_eligible": evidence.support == "supported",
            "measurement_semantics": evidence.diagnostics.get(
                "measurement_semantics"),
            "provenance": evidence.provenance,
            "assumptions": list(evidence.assumptions),
        },
        "details_in_answer_receipt": True,
    }


def packet_exposes_answer(case: Case, packet: dict[str, Any]) -> bool:
    """Detect direct scalar projection of any scorer answer into a packet."""
    truth = expected(case)
    forbidden = {truth["diagnosis"], truth["analogue_outcome"],
                 truth["next_action"]}

    def scalars(value: Any) -> list[str]:
        if isinstance(value, dict):
            return [item for child in value.values() for item in scalars(child)]
        if isinstance(value, list):
            return [item for child in value for item in scalars(child)]
        return [value.strip().lower()] if isinstance(value, str) else []

    return bool(forbidden.intersection(scalars(packet)))


def prompt(case: Case, packet: dict[str, Any] | None) -> str:
    rounded = [round(value, 4) for value in case.values]
    analogue_rows = [{"distance": round(distance, 3), "outcome": outcome}
                     for distance, outcome in case.analogues]
    treatment = ("\nComputed Gnomon evidence (numbers remain authoritative):\n"
                 + json.dumps(packet, separators=(",", ":"))) if packet else ""
    return f"""Compare the first 96 observations with the last 96.
Property: {case.prop}. Allowed diagnoses: {', '.join(LABELS[case.prop])}.
History: {json.dumps(rounded, separators=(',', ':'))}
Context: {case.claim}
Historical episodes (smaller distance means a closer pre-event state):
{json.dumps(analogue_rows, separators=(',', ':'))}
Choose the consensus outcome of the two closest episodes.{treatment}"""


def parse_answer(text: str) -> dict[str, str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return {key: str(value.get(key, "")).strip().lower() for key in (
        "diagnosis", "confidence", "analogue_outcome", "next_action")}


def expected(case: Case) -> dict[str, str]:
    evidence = window_evidence(list(case.values), property=case.prop,
                               season=case.season, window=96)
    claim_direction = case.claim.rsplit(" ", 1)[-1].rstrip(".")
    conflict = claim_direction != evidence.direction
    return {
        # Generator truth scores diagnosis; engine support scores whether a
        # categorical assertion is warranted at this sample size.
        "diagnosis": case.expected,
        "confidence": ("supported" if evidence.support == "supported"
                       else "uncertain"),
        "analogue_outcome": case.expected_analogue,
        "next_action": ("collect_more" if evidence.support != "supported" else
                        "resolve_conflict" if conflict else "act"),
    }


def exact_sign_p(treatment_only: int, control_only: int) -> float:
    n = treatment_only + control_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(treatment_only, control_only) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file()
    api_key = os.environ.get("ENGY_API_KEY")
    client = OpenRouterClient(args.model, api_key=api_key,
                              base_url=args.base_url, temperature=0,
                              max_tokens=700, max_retries=4)
    cases = generate_cases(args.seed, args.cases)
    leaking = [case.case_id for case in cases
               if packet_exposes_answer(case, compact_packet(case))]
    if leaking:
        raise ValueError("reasoning packet exposes scorer answers for: "
                         + ", ".join(leaking[:5]))
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "rows.jsonl"
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if args.resume and rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line); completed[(row["case_id"], row["arm"])] = row
    lock = threading.Lock(); done = len(completed)

    def one(case: Case, arm: str) -> dict[str, Any]:
        response = client.chat([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt(
                case, compact_packet(case) if arm == "evidence" else None)},
        ])
        answer = response.choices[0].message.content
        parsed = parse_answer(answer); truth = expected(case)
        scores = {key: parsed.get(key) == value for key, value in truth.items()}
        usage = getattr(response, "usage", None)
        return {"case_id": case.case_id, "arm": arm, "property": case.prop,
                "difficulty": case.difficulty, "claim_conflicts": case.claim_conflicts,
                "expected": truth, "answer": parsed, "scores": scores,
                "grounded_correct": scores["confidence"],
                "reasoning_correct": all(scores[key] for key in (
                    "diagnosis", "next_action")),
                "synthesis_correct": scores["next_action"],
                "all_correct": all(scores.values()),
                "usage": {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                }}

    jobs = [(case, arm) for case in cases for arm in ("control", "evidence")
            if (case.case_id, arm) not in completed]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(one, *job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            with lock:
                completed[(row["case_id"], row["arm"])] = row
                with rows_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                done += 1
                if done % 20 == 0 or done == args.cases * 2:
                    print(f"completed {done}/{args.cases * 2}", flush=True)
    rows = list(completed.values())
    metrics = {}
    for arm in ("control", "evidence"):
        subset = [row for row in rows if row["arm"] == arm]
        metrics[arm] = {key: statistics.mean(row["scores"][key] for row in subset)
                        for key in ("diagnosis", "confidence", "analogue_outcome",
                                    "next_action")}
        metrics[arm]["all_correct"] = statistics.mean(
            row["all_correct"] for row in subset)
        metrics[arm]["grounded_correct"] = statistics.mean(
            row.get("grounded_correct", row["scores"]["confidence"])
            for row in subset)
        metrics[arm]["reasoning_correct"] = statistics.mean(
            row.get("reasoning_correct", all(row["scores"][key] for key in (
                "diagnosis", "next_action"))) for row in subset)
        metrics[arm]["synthesis_correct"] = statistics.mean(
            row.get("synthesis_correct", row["scores"]["next_action"])
            for row in subset)
        metrics[arm]["all_correct_by_difficulty"] = {
            difficulty: statistics.mean(row["all_correct"] for row in subset
                                        if row["difficulty"] == difficulty)
            for difficulty in DIFFICULTIES
            if any(row["difficulty"] == difficulty for row in subset)
        }
        metrics[arm]["all_correct_by_property"] = {
            prop: statistics.mean(row["all_correct"] for row in subset
                                  if row["property"] == prop)
            for prop in LABELS
            if any(row["property"] == prop for row in subset)
        }
        metrics[arm]["all_correct_by_claim"] = {
            state: statistics.mean(row["all_correct"] for row in subset
                                   if row["claim_conflicts"] is conflicts)
            for state, conflicts in (("conflicting", True), ("aligned", False))
            if any(row["claim_conflicts"] is conflicts for row in subset)
        }
        metrics[arm]["reasoning_by_difficulty"] = {
            difficulty: statistics.mean(row["reasoning_correct"] for row in subset
                                        if row["difficulty"] == difficulty)
            for difficulty in DIFFICULTIES
            if any(row["difficulty"] == difficulty for row in subset)
        }
        metrics[arm]["reasoning_by_property"] = {
            prop: statistics.mean(row["reasoning_correct"] for row in subset
                                  if row["property"] == prop)
            for prop in LABELS
            if any(row["property"] == prop for row in subset)
        }
        metrics[arm]["reasoning_by_claim"] = {
            state: statistics.mean(row["reasoning_correct"] for row in subset
                                   if row["claim_conflicts"] is conflicts)
            for state, conflicts in (("conflicting", True), ("aligned", False))
            if any(row["claim_conflicts"] is conflicts for row in subset)
        }
        metrics[arm]["valid_json_rate"] = statistics.mean(
            all(key in row["answer"] for key in (
                "diagnosis", "confidence", "analogue_outcome", "next_action"))
            for row in subset)
    paired = []
    for case in cases:
        control = completed[(case.case_id, "control")]
        treatment = completed[(case.case_id, "evidence")]
        paired.append((control.get("reasoning_correct", False),
                       treatment.get("reasoning_correct", False)))
    treatment_only = sum(not c and t for c, t in paired)
    control_only = sum(c and not t for c, t in paired)
    summary = {
        "schema_version": "0.5", "seed": args.seed, "cases": args.cases,
        "replicate": args.replicate,
        "model": args.model, "base_url": args.base_url,
        "temperature": 0,
        "provenance": {
            "evaluated_commit": _git_sha(),
            "harness_commit": _git_sha(),
            "harness_sha256": _harness_sha256(),
            "dataset_identity": (
                f"reasoningbench-generator-{GENERATOR_VERSION}:"
                f"seed={args.seed}:cases={args.cases}"),
            "configuration_identity": (
                f"model={args.model}:temperature=0:replicate={args.replicate}"),
        },
        "metrics": metrics,
        "paired": {"primary_endpoint": "diagnosis_and_next_action",
                   "treatment_only": treatment_only,
                   "control_only": control_only,
                   "exact_mcnemar_p": exact_sign_p(treatment_only, control_only)},
        "usage": {
            arm: {
                token: sum(int((row.get("usage") or {}).get(token, 0))
                           for row in rows if row["arm"] == arm)
                for token in ("prompt_tokens", "completion_tokens")
            } for arm in ("control", "evidence")
        },
        "design": {"matched": True, "generated_held_out_seeds": True,
                   "labels_absent_from_prompts": True,
                   "diagnosis_analogue_action_absent_from_evidence_packet": True,
                   "analogue_rows_identical_between_arms": True,
                   "no_primary_forecast_generated": True},
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--base-url", default="https://api.engy.ai/v1")
    parser.add_argument("--cases", type=int, default=72)
    parser.add_argument("--seed", type=int, default=82026)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--acknowledge-retired", action="store_true",
        help="run the withdrawn historical instrument for reproduction only",
    )
    args = parser.parse_args()
    if not args.acknowledge_retired:
        parser.error(
            "ReasoningBench is retired: its historical answer-bearing "
            "treatment invalidated the published uplift claim. Use "
            "DossierBench plus DiscriminationBench, or pass "
            "--acknowledge-retired only to reproduce historical behavior."
        )
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

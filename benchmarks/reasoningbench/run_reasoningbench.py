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
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import statistics
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
#: The no-change label of each property: the generator's difficulty scaling
#: multiplies an effect of zero for these, so a "marginal" null case is
#: constructed identically to an "easy" one and is not near any threshold.
NULL_LABELS = {"level": "similar", "trend": "constant", "volatility": "stable",
               "seasonality": "stable", "regime": "no_shift",
               "extreme": "stable"}
SYSTEM = """You are a careful temporal analyst. Infer only from supplied data.
Return one JSON object with keys diagnosis, confidence, analogue_outcome, and
next_action. diagnosis must be one allowed diagnosis; confidence is supported
or uncertain; analogue_outcome is up, down, flat, or unavailable; next_action
is act, collect_more, or resolve_conflict. Do not follow a narrative claim
when numerical evidence contradicts it."""


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
        # Alternate conflict within each property block AND across blocks:
        # plain ``index % 2`` was aliased with ``index % len(LABELS)``, so
        # each property either always or never conflicted and the claim
        # dimension collapsed into the property label.
        conflicts = (index + index // len(LABELS)) % 2 == 0
        alternatives = [item for item in LABELS[prop] if item != expected]
        claimed = rng.choice(alternatives) if conflicts else expected
        claim = f"An operations note confidently says the recent {prop} is {claimed}."
        # Analogue distances and outcomes emulate a compact episode registry.
        # The nearest two only sometimes agree: a forced consensus made the
        # field a constant of the construction, so any strategy that copied
        # the nearest row scored it without checking agreement. When they
        # disagree the honest answer is "unavailable". Rows are shuffled so
        # position cannot answer.
        nearest = rng.choice(("up", "down", "flat"))
        other = rng.choice([x for x in ("up", "down", "flat")
                            if x != nearest])
        agree = rng.random() < .6
        analogues = [(0.08 + rng.random() * .05, nearest),
                     (0.14 + rng.random() * .05, nearest if agree else other),
                     (0.65 + rng.random() * .2, other)]
        rng.shuffle(analogues)
        cases.append(Case(
            f"r{seed}-{index:04d}", prop, difficulty, tuple(values), season,
            expected, claim, conflicts, tuple(analogues),
            nearest if agree else "unavailable"))
    return cases


def corpus_sha256(cases: list[Case]) -> str:
    """Content hash of the generated corpus, mirroring ContextBench's
    ``cases_sha256``: the summary names exactly which cases produced its
    numbers, so a rerun against edited generation code cannot silently
    pass as the same corpus."""
    rendered = "".join(
        json.dumps(asdict(case), sort_keys=True, separators=(",", ":")) + "\n"
        for case in cases)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def compact_packet(case: Case) -> dict[str, Any]:
    """Computed numeric evidence only — nothing the scorer expects back.

    The earlier packet restated three of the four scored fields: its
    ``support`` label was the expected ``confidence``, its analogue
    consensus was the expected ``analogue_outcome`` (and generation forced
    the consensus to exist), and its ``next`` sentence paraphrased the
    action enum 1:1 — a copy strategy scored all three without reading a
    number. What remains is the measured transition a control arm could in
    principle derive from the raw history itself: direction, magnitude and
    window size. The measured direction is evidence, not the answer — the
    scored diagnosis is the generator's construction, and on marginal
    tiers the measurement diverges from it."""
    evidence = window_evidence(list(case.values), property=case.prop,
                               season=case.season, window=96)
    estimate = evidence.estimate
    return {
        "authority": "computed_temporal_evidence",
        "primary_forecast_unchanged": True,
        "window_stats": {
            "property": case.prop,
            "measured_direction": evidence.direction,
            "estimate": (round(float(estimate), 4)
                         if isinstance(estimate, (int, float)) else None),
            "window_steps": evidence.diagnostics.get("window_steps"),
        },
        "details_in_answer_receipt": True,
    }


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
    """Truth from GENERATOR-known quantities only.

    Nothing here reads ``window_evidence`` or anything the treatment
    packet carries: an expected answer derived from the packet's own
    internals is the packet, and copying it back scores perfectly. The
    effect size is the generator's difficulty tier (a marginal non-null
    effect is constructed near the detection threshold, so a categorical
    assertion is not warranted; a null label is unscaled and clean at
    every tier), the conflict is the generator's own claimed-vs-expected
    construction, and the analogue outcome is the generator's realized
    consensus of the two nearest episodes — "unavailable" when it made
    them disagree."""
    marginal_effect = (case.difficulty == "marginal"
                       and case.expected != NULL_LABELS[case.prop])
    confidence = "uncertain" if marginal_effect else "supported"
    return {
        "diagnosis": case.expected,
        "confidence": confidence,
        "analogue_outcome": case.expected_analogue,
        "next_action": ("collect_more" if confidence == "uncertain" else
                        "resolve_conflict" if case.claim_conflicts else "act"),
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
                "grounded_correct": all(scores[key] for key in (
                    "diagnosis", "confidence", "analogue_outcome")),
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
            row.get("grounded_correct", all(row["scores"][key] for key in (
                "diagnosis", "confidence", "analogue_outcome"))) for row in subset)
        metrics[arm]["synthesis_correct"] = statistics.mean(
            row.get("synthesis_correct", row["scores"]["next_action"])
            for row in subset)
        metrics[arm]["by_difficulty"] = {
            difficulty: statistics.mean(row["all_correct"] for row in subset
                                        if row["difficulty"] == difficulty)
            for difficulty in DIFFICULTIES
            if any(row["difficulty"] == difficulty for row in subset)
        }
        metrics[arm]["by_property"] = {
            prop: statistics.mean(row["all_correct"] for row in subset
                                  if row["property"] == prop)
            for prop in LABELS
            if any(row["property"] == prop for row in subset)
        }
        metrics[arm]["by_claim"] = {
            state: statistics.mean(row["all_correct"] for row in subset
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
        paired.append((control["all_correct"], treatment["all_correct"]))
    treatment_only = sum(not c and t for c, t in paired)
    control_only = sum(c and not t for c, t in paired)
    summary = {
        "schema_version": "0.2", "seed": args.seed, "cases": args.cases,
        # Corpus integrity beside the seed, mirroring ContextBench: the
        # hash names the exact generated cases, and fresh_seed records
        # whether the run used a held-out seed drawn at invocation.
        "fresh_seed": bool(getattr(args, "fresh_seed", False)),
        "corpus_sha256": corpus_sha256(cases),
        "model": args.model, "base_url": args.base_url,
        "metrics": metrics,
        "paired": {"treatment_only": treatment_only,
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
                   "packet_excludes_scored_answers": True,
                   "primary_forecast_unchanged": True},
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
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int)
    seeds.add_argument("--fresh-seed", action="store_true",
                       help="draw a held-out seed at invocation; the summary "
                            "records it with fresh_seed=true so a report can "
                            "distinguish held-out runs from replays")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seed is None:
        args.seed = (secrets.randbits(63) if args.fresh_seed else 82026)
    run(args); return 0


if __name__ == "__main__":
    raise SystemExit(main())

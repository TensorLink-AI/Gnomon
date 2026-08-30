"""Matched uplift experiment: does the evidence dossier make a model better?

Cases default to *real* data: windowed slices of eight real observational
series (Mauna Loa CO2, sunspots, the Nile, US macro aggregates, El Niño
SST — see ``data/README.md``), cut at a sampled instant. The truth label
is not authored by a generator: it is what the *realized future* window —
held out from the model and from every packet — actually did, measured by
the same deterministic window semantics Gnomon uses in production. A
transition label must be a ``supported`` realized outcome; null outcomes
("nothing changed") are admitted at disclosed ``weak`` confidence so the
task keeps both halves of discrimination — calling a transition, and
knowing when not to. Because these series must be assumed memorized by every
LLM, each case passes through a seeded affine transform (positive scale,
fresh offset) that preserves direction, trend, volatility ratios, and
breaks while defeating verbatim sequence lookup; prompts carry values
only, never names or dates. ``--source synthetic`` keeps the seeded
generator as a diagnostic mode.

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
import statistics
import subprocess
import sys
import threading
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.envfile import load_env_file  # noqa: E402
from benchmarks.common.openrouter import extract_json_objects  # noqa: E402
from benchmarks.discriminationbench.run_discriminationbench import _series  # noqa: E402
from gnomon.discrimination import discriminate  # noqa: E402
from gnomon.reasoning_packet import repair_selection  # noqa: E402
from gnomon.temporal import detect_season  # noqa: E402
from gnomon.temporal_evidence import (  # noqa: E402
    multi_resolution_evidence, window_evidence,
)
from gnomon.temporal_planner import build_evidence_plan  # noqa: E402
from gnomon.temporal_question import TemporalQuestion  # noqa: E402

GENERATOR_VERSION = "0.2"
DATA_DIR = Path(__file__).resolve().parent / "data"
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
    #: Detected seasonal period of the visible history (1 when none).
    season: int = 1
    #: Steps in the held-out future the truth was measured on; None for
    #: synthetic observed-window cases.
    horizon: int | None = None
    #: Corpus series the window came from — scorer-side provenance only,
    #: never placed in a prompt.
    origin: str = "synthetic"
    #: How firmly the realized outcome earned the label ("supported", or
    #: "weak" for admitted null outcomes); synthetic truths are generator
    #: facts.
    label_confidence: str = "supported"


def generate_cases(seed: int, count: int) -> list[Case]:
    """Synthetic diagnostic cases from the DiscriminationBench generator,
    restricted to the properties both packet styles can speak about."""
    rng = random.Random(seed)
    cases: list[Case] = []
    while len(cases) < count:
        property, truth, values = _series(rng)
        if property not in PROPERTIES:
            continue
        cases.append(Case(f"d{seed}-{len(cases):04d}", property, truth,
                          tuple(round(value, 4) for value in values)))
    return cases


def load_corpus(data_dir: Path = DATA_DIR) -> dict[str, list[float]]:
    corpus: dict[str, list[float]] = {}
    for path in sorted(data_dir.glob("*.csv")):
        lines = path.read_text(encoding="utf-8").splitlines()
        corpus[path.stem] = [float(line) for line in lines[1:] if line.strip()]
    if not corpus:
        raise FileNotFoundError(f"no corpus series under {data_dir}")
    return corpus


#: Truth values meaning "nothing changed" for each property.
_NULL_TRUTHS = {"similar", "constant", "stable"}


def _realized_truth(
    history: list[float], future: list[float], season: int, property: str,
) -> tuple[str | None, str, str | None]:
    """Label a case by what the held-out future actually did, under the
    same window semantics Gnomon uses in production. Scorer-side only.

    Returns ``(truth, reason, label_confidence)``. A transition label must
    be a ``supported`` realized outcome. A null label ("nothing changed")
    is additionally admitted at ``weak`` confidence: real noise rarely
    *proves* equivalence at the supported bar, and excluding nulls
    entirely would quietly turn every question into "which transition?" —
    a design a test-taker could game and half the discrimination task
    (knowing when not to call a transition) would go unmeasured. Label
    confidence rides on every case and metrics are reported split by it.
    """
    horizon = len(future)
    evidence = window_evidence(history[-horizon:] + future, property=property,
                               season=season, window=horizon)
    direction = str(evidence.direction)
    if not evidence.identifiable or direction not in OPTIONS[property]:
        return None, "unlabelable_outcome", None
    if evidence.support == "supported":
        return direction, "supported_outcome", "supported"
    if direction in _NULL_TRUTHS:
        return direction, "weak_null_outcome", "weak"
    # A borderline realized transition cannot be anyone's ground truth.
    return None, "ambiguous_outcome", None


def generate_real_cases(
    seed: int, count: int, data_dir: Path = DATA_DIR,
) -> tuple[list[Case], dict[str, Any], dict[str, list[float]]]:
    """Real windowed cases with future-outcome labels.

    Sampling is soft-balanced across (property, truth) cells so a majority
    class cannot dominate, and every case is affine-anonymized: a positive
    scale and a fresh offset leave every asked-about dynamic invariant
    while defeating verbatim recall of these well-known series.
    """
    corpus = load_corpus(data_dir)
    rng = random.Random(seed)
    names = sorted(corpus)
    cases: list[Case] = []
    futures: dict[str, list[float]] = {}
    cell_counts: dict[tuple[str, str], int] = {}
    seen: set[tuple[str, int, int, str]] = set()
    cell_cap = max(2, count // (len(PROPERTIES) * 3) + 2)
    skipped = {"too_short": 0, "unlabelable_outcome": 0,
               "ambiguous_outcome": 0, "cell_full": 0, "duplicate": 0}
    label_confidence_counts = {"supported": 0, "weak": 0}
    attempts = 0
    balanced_limit = 200 * count
    # Phase one enforces the per-(property, truth) cap; if real outcomes are
    # too scarce in some cell, phase two fills the remainder uncapped and
    # the achieved distribution is disclosed instead of silently skewing.
    while len(cases) < count and attempts < 2 * balanced_limit:
        attempts += 1
        balanced_phase = attempts <= balanced_limit
        name = names[rng.randrange(len(names))]
        series = corpus[name]
        property = PROPERTIES[rng.randrange(len(PROPERTIES))]
        window = rng.choice((48, 96, 160))
        if len(series) < 24 + 12:
            skipped["too_short"] += 1
            continue
        cutoff = rng.randrange(24, len(series) - 12 + 1)
        history = [float(v) for v in series[max(0, cutoff - window):cutoff]]
        season, _, _ = detect_season(history, "D")
        horizon = max(12, season)
        if len(history) < 24 or cutoff + horizon > len(series) \
                or horizon > len(history) // 2:
            skipped["too_short"] += 1
            continue
        key = (name, cutoff, window, property)
        if key in seen:
            skipped["duplicate"] += 1
            continue
        future = [float(v) for v in series[cutoff:cutoff + horizon]]
        truth, reason, label_confidence = _realized_truth(
            history, future, season, property)
        if truth is None:
            skipped[reason] += 1
            continue
        if balanced_phase and cell_counts.get((property, truth), 0) >= cell_cap:
            skipped["cell_full"] += 1
            continue
        seen.add(key)
        # Affine anonymization: positive scale, fresh offset. Direction,
        # trend sign, volatility ratios, and break structure are invariant;
        # verbatim sequence lookup is not.
        scale = rng.uniform(.6, 2.4)
        offset = rng.uniform(40, 900) - scale * statistics.median(history)
        transformed = tuple(round(scale * value + offset, 4)
                            for value in history)
        cell_counts[(property, truth)] = cell_counts.get(
            (property, truth), 0) + 1
        label_confidence_counts[label_confidence or "supported"] += 1
        case = Case(f"r{seed}-{len(cases):04d}", property, truth,
                    transformed, season=season, horizon=horizon, origin=name,
                    label_confidence=label_confidence or "supported")
        cases.append(case)
        futures[case.case_id] = [round(scale * value + offset, 4)
                                 for value in future]
    if len(cases) < count:
        raise ValueError(
            f"only {len(cases)}/{count} labelable real cases after "
            f"{attempts} attempts; skipped={skipped}")
    provenance = {
        "corpus_series": names,
        "corpus_points": sum(len(values) for values in corpus.values()),
        "attempts": attempts,
        "skipped": skipped,
        "cell_cap": cell_cap,
        "fully_balanced": attempts <= balanced_limit,
        "truth_distribution": {
            f"{prop}:{truth}": count_
            for (prop, truth), count_ in sorted(cell_counts.items())},
        "labeling": ("realized_future_window; transitions require "
                     "supported outcomes; nulls admitted at weak"),
        "label_confidence_counts": label_confidence_counts,
        "anonymization": "seeded_positive_affine_transform_per_case",
    }
    return cases, provenance, futures


def computed_evidence(case: Case) -> dict[str, Any]:
    """Everything deterministic, computed once per case and shared by arms.

    Built from the visible history alone — for real cases the held-out
    future exists only in the scorer."""
    values = list(case.values)
    observed = window_evidence(values, property=case.property,
                               season=case.season)
    multi = multi_resolution_evidence(values, property=case.property,
                                      season=case.season)
    discrimination = discriminate(values, property=case.property,
                                  season=case.season)
    canonical_value = multi.get("direction")
    canonical_support = str(multi.get("support") or "abstained")
    question = TemporalQuestion(
        case.case_id, "predict" if case.horizon else "detect", "series",
        case.property, horizon=case.horizon)
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
    if case.horizon:
        question = (
            f"Question: over the next {case.horizon} observations (not "
            f"shown), how will the series' {case.property} compare to its "
            f"recent behaviour?")
    else:
        question = (
            f"Question: which best describes the recent {case.property} of "
            f"this series relative to its earlier behaviour?")
    return (
        f"Series values, oldest first:\n"
        f"{json.dumps(list(case.values), separators=(',', ':'))}\n"
        f"{question}\n"
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
    # Balanced-span candidates, not a greedy regex: a valid JSON answer
    # followed by prose containing a brace — or preceded by an echoed
    # dossier, which only the dossier arm's prompt even contains — must
    # not be scored as no answer. The first candidate carrying the
    # contract's "value" key wins; echoed packets do not have one at top
    # level and are skipped.
    for payload in extract_json_objects(text):
        if "value" not in payload:
            continue
        return {
            "value": str(payload.get("value", "")).strip().lower(),
            "cited_evidence": [str(item) for item in
                               (payload.get("cited_evidence") or [])
                               if isinstance(item, str)],
        }
    return {}


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
            + "\nOptions: " + ", ".join(OPTIONS[case.property]) + "."
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


def verify_no_future_leakage(cases: list[Case],
                             futures: dict[str, list[float]],
                             computed: dict[str, dict[str, Any]]) -> None:
    """The held-out future must not reach any prompt in any arm.

    Checked against the actual serialized prompts, not the construction:
    the first future values, in sequence and in prompt formatting, must be
    absent everywhere (single values may coincide legitimately)."""
    for case in cases:
        held_out = futures.get(case.case_id)
        if not held_out or len(held_out) < 3:
            continue
        marker = json.dumps(held_out[:8], separators=(",", ":"))[1:-1]
        # The history itself may legitimately render the marker — real
        # series carry runs of repeated values, and a flat future then
        # matches a flat stretch of history. Excise the history blob and
        # scan the rest: the only way the future can actually leak is
        # through the packet or question.
        history_blob = json.dumps(list(case.values), separators=(",", ":"))
        for arm in ARMS:
            text = prompt(case, arm, computed[case.case_id])
            if marker in text.replace(history_blob, "", 1):
                raise ValueError(
                    f"held-out future leaked into arm {arm} "
                    f"for {case.case_id}")


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
            base_url=args.base_url, temperature=0,
            max_tokens=getattr(args, "max_tokens", 500),
            max_retries=4,
            reasoning_effort=getattr(args, "reasoning_effort", None))
    source = getattr(args, "source", "real")
    if source == "real":
        cases, corpus_provenance, futures = generate_real_cases(
            args.seed, args.cases,
            Path(getattr(args, "data_dir", None) or DATA_DIR))
    else:
        cases = generate_cases(args.seed, args.cases)
        corpus_provenance, futures = {"generator": "synthetic"}, {}
    computed = {case.case_id: computed_evidence(case) for case in cases}
    verify_arm_symmetry(cases, computed)
    if futures:
        verify_no_future_leakage(cases, futures, computed)

    # Deterministic references, no model involved. `copy_conclusion` is the
    # old canonical; `copy_discriminator` is the mechanism's best where it
    # ran (its abstentions count as wrong, exactly as a silent consumer
    # would experience them); `always_majority` answers every question with
    # its property's most common truth — the bar an imbalanced label set
    # would otherwise let a constant strategy clear.
    majority: dict[str, str] = {}
    for name in PROPERTIES:
        truths = [c.truth for c in cases if c.property == name]
        if truths:
            majority[name] = max(set(truths), key=truths.count)
    references = {
        "chance": statistics.mean(1 / len(OPTIONS[c.property]) for c in cases),
        "always_majority": statistics.mean(
            majority.get(c.property) == c.truth for c in cases),
        "copy_conclusion": statistics.mean(
            computed[c.case_id]["canonical"]["value"] == c.truth
            for c in cases),
        "copy_discriminator": statistics.mean(
            (computed[c.case_id]["discrimination"] or {}).get("best") == c.truth
            for c in cases),
    }

    # A case_id alone does not identify a case: the same seed with a
    # different --cases count or corpus yields sequential ids over
    # divergent content. Rows carry the full dataset identity and the
    # answering model, and resume rejects anything that does not match.
    model_name = getattr(args, "model", None)
    if source == "real":
        corpus_dir = Path(getattr(args, "data_dir", None) or DATA_DIR)
        corpus_tag = hashlib.sha256(b"".join(
            path.read_bytes()
            for path in sorted(corpus_dir.glob("*.csv")))).hexdigest()[:12]
    else:
        corpus_tag = "synthetic"
    dataset_identity = (
        f"dossierbench-generator-{GENERATOR_VERSION}:"
        f"source={source}:seed={args.seed}:cases={args.cases}:"
        f"corpus={corpus_tag}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "rows.jsonl"
    valid_ids = {case.case_id for case in cases}
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if args.resume and rows_path.exists():
        stale = malformed = 0
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A crash mid-append can leave one truncated final line;
                # its (case, arm) simply reruns.
                malformed += 1
                continue
            if (row.get("case_id") in valid_ids and row.get("arm") in ARMS
                    and row.get("dataset") == dataset_identity
                    and row.get("model") == model_name):
                completed[(row["case_id"], row["arm"])] = row
            else:
                stale += 1
        if stale or malformed:
            print(f"resume: ignored {stale} stale and {malformed} "
                  f"malformed rows", flush=True)
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
            "case_id": case.case_id, "arm": arm,
            "dataset": dataset_identity, "model": model_name,
            "property": case.property,
            "truth": case.truth, "value": outcome["value"],
            "correct": outcome["value"] == case.truth,
            "stage": outcome["stage"], "violations": outcome["violations"],
            "calls": outcome["calls"],
            "separation": (aids["discrimination"] or {}).get("separation"),
            "discriminator_correct":
                (aids["discrimination"] or {}).get("best") == case.truth,
            "canonical_binding":
                packet["selection_contract"]["canonical"]["role"] == "binding",
            "label_confidence": case.label_confidence,
        }

    jobs = [(case, arm) for case in cases for arm in ARMS
            if (case.case_id, arm) not in completed]
    failures: list[tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(one, *job): job for job in jobs}
        for future in as_completed(futures):
            case, arm = futures[future]
            try:
                row = future.result()
            except Exception as error:
                # One dead endpoint call must not discard the paid work
                # already on disk: record, finish the rest, fail at the
                # end. Scoring an API failure as a model answer would be
                # worse than failing.
                failures.append((case.case_id, arm, repr(error)[:300]))
                continue
            with lock:
                completed[(row["case_id"], row["arm"])] = row
                with rows_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
    if failures:
        raise RuntimeError(
            f"{len(failures)}/{len(jobs)} model calls failed; completed "
            f"rows are saved in {rows_path} — rerun with --resume to "
            f"finish. First failure: {failures[0]}")

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
            "by_label_confidence": {
                grade: statistics.mean(row["correct"] for row in subset
                                       if row.get("label_confidence") == grade)
                for grade in ("supported", "weak")
                if any(row.get("label_confidence") == grade for row in subset)
            },
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
        "schema_version": "0.3", "seed": args.seed, "cases": args.cases,
        "source": source,
        "model": getattr(args, "model", None),
        "temperature": 0,
        "provenance": {
            "evaluated_commit": _git_sha(),
            "harness_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
            "dataset_identity": dataset_identity,
            "cases": corpus_provenance,
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
            "reasoning_effort": getattr(args, "reasoning_effort", None),
            "initial_max_tokens": getattr(args, "max_tokens", 500),
            "arms_differ_by_packet_block_only": True,
            "labels_absent_from_prompts_and_packets": True,
            "same_machinery_feeds_both_packet_arms": True,
            "repair_rounds": 1,
            **({
                "real_observational_data": True,
                "truth_is_realized_held_out_future": True,
                "held_out_future_absent_from_prompts_verified": True,
                "memorization_defense":
                    "per_case_seeded_positive_affine_transform;"
                    "values_only_prompts;windowed_slices",
            } if source == "real" else {"synthetic_generator": True}),
        },
    }
    if hasattr(client, "usage_summary"):
        summary["usage"] = client.usage_summary
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
    parser.add_argument("--source", choices=("real", "synthetic"),
                        default="real")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--max-tokens", type=int, default=500,
        help="Initial completion budget; retries may escalate it.")
    parser.add_argument("--reasoning-effort", default=None,
                        choices=("none", "low", "medium", "high"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""BreachBench: score the client's actual job, priced in the client's units.

Gnomon's first product job is operational threshold risk: which metric
may breach a meaningful limit, when, and whether the evidence supports
intervening. This benchmark scores exactly that deliverable. Every case
is a windowed slice of a real telemetry-flavoured series (real web
traffic, a real 5-minute sensor, real pedestrian counts with a genuine
COVID regime break, real retail sales — see ``data/README.md``) with an
alert threshold; the ground truth is whether the *realized future* window
— held out from the model and from Gnomon — actually breached, and when.

Two model arms answer with the same model at temperature zero, prompts
differing by the evidence block alone (verified pre-flight):

- ``control`` — history, threshold, costs, question.
- ``gnomon``  — the same plus Gnomon's real product output for this exact
  call: ``forecast(threshold=...)``'s headline, support, per-step breach
  probabilities, interval path, warnings, and the model-assisted lane —
  the artifact a client integration actually receives.

The primary metric is not accuracy but **decision cost and regret** under
a stated cost model (acting costs 2 and mitigates; a missed breach costs
10), because that is what "useful" means to the client. Deterministic
references bound the result at zero API cost: the product's own decision
rule with no LLM, naive persistence, always/never act, and the hindsight
optimum. A model arm is only genuinely useful when it beats both the
model-alone control *and* the product-alone rule.

Truth labels exist only in the scorer; cases are affine-anonymized
(threshold transformed identically) against verbatim recall of these
public series, and the harness verifies the held-out future reaches no
prompt.
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
import tempfile
import threading
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.envfile import load_env_file  # noqa: E402
from benchmarks.common.openrouter import extract_json_objects  # noqa: E402

GENERATOR_VERSION = "0.2"
DATA_DIR = Path(__file__).resolve().parent / "data"
HORIZON = 24
COST_ACT = 2.0
COST_MISS = 10.0
ARMS = ("control", "gnomon")
OUTCOME_CELLS = ("no_breach", "breach_early", "breach_late")
#: Target outcome mix. Alerts are rare in real operations, and the breach
#: base rate is deliberately held near the cost break-even
#: (COST_ACT/COST_MISS = 0.2), where neither constant policy — always act,
#: never act — is close to optimal and only genuine discrimination
#: reduces regret. A 50/50 mix would let "always act" masquerade as
#: skill.
OUTCOME_TARGETS = {"no_breach": .7, "breach_early": .15, "breach_late": .15}
SYSTEM = """You operate a production metric with an alert threshold. Infer
only from the supplied data and evidence. Answer with one JSON object and
nothing else."""


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass(frozen=True)
class Case:
    case_id: str
    values: tuple[float, ...]
    threshold: float
    horizon: int
    truth_breach: bool
    truth_first_step: int | None
    outcome_cell: str
    origin: str
    frequency: str = "D"


#: The corpus files name their real cadence in the filename
#: (``sensor_temps_5min``, ``retail_sales_monthly``); feeding Gnomon the
#: true cadence matters because season detection and support tiers are
#: frequency-aware — calling a 5-minute sensor "daily" would search for a
#: weekly cycle that does not exist in the data.
_CADENCE_TOKENS = (("5min", "5min"), ("hourly", "h"), ("weekly", "W"),
                   ("monthly", "MS"), ("daily", "D"))


def series_frequency(name: str) -> str:
    tokens = set(name.split("_"))
    for token, code in _CADENCE_TOKENS:
        if token in tokens:
            return code
    return "D"


def load_corpus(data_dir: Path = DATA_DIR) -> dict[str, list[float]]:
    corpus: dict[str, list[float]] = {}
    for path in sorted(data_dir.glob("*.csv")):
        lines = path.read_text(encoding="utf-8").splitlines()
        corpus[path.stem] = [float(line) for line in lines[1:] if line.strip()]
    if not corpus:
        raise FileNotFoundError(f"no corpus series under {data_dir}")
    return corpus


def _robust_scale(values: list[float]) -> float:
    diffs = [abs(right - left) for left, right in zip(values, values[1:])]
    if not diffs:
        return 1.0
    centre = statistics.median(diffs)
    return max(centre, 1e-9 * max(abs(max(values)), abs(min(values)), 1.0))


def generate_cases(
    seed: int, count: int, data_dir: Path = DATA_DIR,
) -> tuple[list[Case], dict[str, Any], dict[str, list[float]]]:
    """Real windowed cases with realized-breach truth.

    Thresholds sit at the recent maximum plus k robust innovation scales
    (k sampled from likely-breach through comfortable-headroom), so
    difficulty spans the alert bands a real operator sets. Outcome cells
    (no breach / early breach / late breach) are soft-balanced with a
    disclosed uncapped fill phase, and every case is affine-anonymized
    with the threshold transformed identically.
    """
    corpus = load_corpus(data_dir)
    rng = random.Random(seed)
    names = sorted(corpus)
    cases: list[Case] = []
    futures: dict[str, list[float]] = {}
    cell_counts: dict[str, int] = {}
    seen: set[tuple[str, int, int, float]] = set()
    # Realized futures must not overlap across cases from the same
    # series: overlapping futures would share the same breach events.
    # Non-overlap does not make labels independent — adjacent futures in
    # one regime (the pedestrian series' COVID collapse) still co-move —
    # so the per-series case counts are disclosed for exactly that
    # reason. Histories may still overlap; that is disclosed, not hidden.
    used_cutoffs: dict[str, list[int]] = {name: [] for name in names}
    cell_caps = {cell: int(count * fraction) + 1
                 for cell, fraction in OUTCOME_TARGETS.items()}
    skipped = {"too_short": 0, "degenerate": 0, "cell_full": 0,
               "duplicate": 0, "future_overlap": 0, "rounding_flip": 0}
    attempts = 0
    balanced_limit = 300 * count
    while len(cases) < count and attempts < 2 * balanced_limit:
        attempts += 1
        balanced_phase = attempts <= balanced_limit
        name = names[rng.randrange(len(names))]
        series = corpus[name]
        window = rng.choice((96, 168))
        if len(series) < window + HORIZON + 1:
            skipped["too_short"] += 1
            continue
        cutoff = rng.randrange(window, len(series) - HORIZON + 1)
        history = [float(v) for v in series[cutoff - window:cutoff]]
        future = [float(v) for v in series[cutoff:cutoff + HORIZON]]
        scale = _robust_scale(history)
        margin = rng.choice((-1.0, -.25, .5, 1.5, 3.0, 6.0))
        threshold = max(history[-HORIZON:]) + margin * scale
        key = (name, cutoff, window, round(margin, 3))
        if key in seen:
            skipped["duplicate"] += 1
            continue
        if any(abs(cutoff - other) < HORIZON
               for other in used_cutoffs[name]):
            skipped["future_overlap"] += 1
            continue
        if max(history) == min(history):
            skipped["degenerate"] += 1
            continue
        breach_steps = [step for step, value in enumerate(future, 1)
                        if value > threshold]
        truth_breach = bool(breach_steps)
        first_step = breach_steps[0] if breach_steps else None
        cell = ("no_breach" if not truth_breach else
                "breach_early" if first_step <= HORIZON // 3 else
                "breach_late")
        if balanced_phase and cell_counts.get(cell, 0) >= cell_caps[cell]:
            skipped["cell_full"] += 1
            continue
        # Affine anonymization; the threshold transforms with the values,
        # so breach structure and timing are exactly invariant — verified
        # below on the rounded numbers the model actually sees, because a
        # value sitting within rounding distance of the threshold could
        # otherwise flip a label relative to the shown data.
        a = rng.uniform(.6, 2.4)
        b = rng.uniform(40, 900) - a * statistics.median(history)
        shown_values = tuple(round(a * value + b, 4) for value in history)
        shown_threshold = round(a * threshold + b, 4)
        shown_future = [round(a * value + b, 4) for value in future]
        shown_steps = [step for step, value in enumerate(shown_future, 1)
                       if value > shown_threshold]
        if (bool(shown_steps) != truth_breach
                or (shown_steps[0] if shown_steps else None) != first_step):
            skipped["rounding_flip"] += 1
            continue
        seen.add(key)
        used_cutoffs[name].append(cutoff)
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        case = Case(
            f"b{seed}-{len(cases):04d}", shown_values, shown_threshold,
            HORIZON, truth_breach, first_step, cell, name,
            series_frequency(name))
        cases.append(case)
        futures[case.case_id] = shown_future
    if len(cases) < count:
        raise ValueError(
            f"only {len(cases)}/{count} cases after {attempts} attempts; "
            f"skipped={skipped}")
    provenance = {
        "corpus_series": names,
        "corpus_points": sum(len(values) for values in corpus.values()),
        # Identity of the exact corpus bytes: resume must be able to tell
        # rows generated against a different data-dir apart.
        "corpus_sha256": hashlib.sha256(b"".join(
            path.read_bytes()
            for path in sorted(data_dir.glob("*.csv")))).hexdigest(),
        "cases_per_series": {name: len(cutoffs) for name, cutoffs
                             in sorted(used_cutoffs.items()) if cutoffs},
        "attempts": attempts, "skipped": skipped,
        "fully_balanced": attempts <= balanced_limit,
        "outcome_distribution": dict(sorted(cell_counts.items())),
        "outcome_targets": OUTCOME_TARGETS,
        "breach_base_rate": statistics.mean(
            case.truth_breach for case in cases),
        "labeling": "realized_future_breach_and_first_step",
        "independence": ("realized_futures_non_overlapping_within_series;"
                         "histories_may_overlap;labels_can_still_comove_"
                         "through_shared_regimes_see_cases_per_series"),
        "anonymization":
            "seeded_positive_affine_transform_threshold_included;"
            "label_invariance_verified_on_rounded_shown_numbers",
    }
    return cases, provenance, futures


def _grid_timestamps(frequency: str, count: int) -> list[str]:
    """Synthetic anchor timestamps on the series' true cadence. The dates
    are arbitrary (the values are anonymized anyway); the *step* is real,
    because Gnomon's season detection and support tiers are
    frequency-aware."""
    from datetime import datetime, timedelta

    if frequency == "MS":
        return [f"{2000 + index // 12}-{1 + index % 12:02d}-01"
                for index in range(count)]
    steps = {"5min": timedelta(minutes=5), "h": timedelta(hours=1),
             "W": timedelta(weeks=1), "D": timedelta(days=1)}
    step = steps.get(frequency, timedelta(days=1))
    start = datetime(2020, 1, 1)
    if step >= timedelta(days=1):
        return [(start + step * index).date().isoformat()
                for index in range(count)]
    return [(start + step * index).isoformat(sep=" ")
            for index in range(count)]


def product_packet(case: Case) -> dict[str, Any]:
    """Gnomon's real output for this exact client call, bounded for a
    prompt. Computed from the visible history alone."""
    import shutil

    from gnomon import forecast as gnomon_forecast
    from gnomon.contracts import GnomonError
    from gnomon.support import forecast_headline

    run_dir = Path(tempfile.mkdtemp(prefix="breachbench-"))
    try:
        csv_path = run_dir / "history.csv"
        stamps = _grid_timestamps(case.frequency, len(case.values))
        lines = ["timestamp,value"] + [
            f"{stamp},{value!r}"
            for stamp, value in zip(stamps, case.values)
        ]
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            artifact, _ = gnomon_forecast(
                str(csv_path), time_column="timestamp",
                target_column="value", horizon=case.horizon,
                frequency=case.frequency, threshold=case.threshold,
                output=str(run_dir / "out"))
        except GnomonError as error:
            return {"status": "abstained", "code": error.code,
                    "message": str(error.message)[:300]}
        result = artifact.results[0]
        rows = result.forecast or []
        packet: dict[str, Any] = {
            "authority": "computed_gnomon_forecast_with_threshold_analysis",
            "support": result.support,
            "selected_model": result.selected_model,
            "headline": forecast_headline(
                result.support, result.support_assessment, rows),
            "forecast": [
                {"step": step,
                 "q50": round(float(row.get("q50", row["point"])), 4),
                 "q10": round(float(row["q10"]), 4),
                 "q90": round(float(row["q90"]), 4),
                 **({"tier": row["tier"]} if "tier" in row else {})}
                for step, row in enumerate(rows, 1)
            ],
            "warnings": [str(item)[:200]
                         for item in (result.warnings or [])[:2]],
        }
        threshold = result.threshold or {}
        if threshold:
            horizon_event = threshold.get("horizon_event") or {}
            stamp_to_step = {str(row.get("timestamp")): step
                             for step, row in enumerate(rows, 1)}
            packet["threshold_analysis"] = {
                "threshold": case.threshold,
                "probability_above_per_step": [
                    round(float(value), 4)
                    for value in (threshold.get("probability_above") or [])],
                "first_step_point_above": stamp_to_step.get(
                    str(threshold.get("first_timestamp_point_above"))),
                "first_step_interval_above": stamp_to_step.get(
                    str(threshold.get("first_timestamp_interval_above"))),
                "basis": threshold.get("basis"),
                "horizon_event": horizon_event or None,
            }
            if horizon_event:
                # Costs come from this client task.  Gnomon projects them
                # through its public policy executable without changing the
                # immutable event probability.
                from gnomon.breach import (
                    BreachDecisionPolicy, apply_breach_policy,
                )
                packet["governed_decision"] = apply_breach_policy(
                    horizon_event,
                    BreachDecisionPolicy(COST_ACT, COST_MISS),
                )
        else:
            packet["threshold_analysis"] = {
                "threshold": case.threshold,
                "unavailable": ("exceedance probabilities require "
                                "calibrated residuals, which these rows "
                                "do not have"),
            }
        lane = result.model_assisted
        if lane:
            packet["model_assisted"] = {
                "support": lane.get("support"),
                "selected_model": lane.get("selected_model"),
                "points": [round(float(v), 4)
                           for v in (lane.get("points") or [])],
            }
        return packet
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def base_prompt(case: Case) -> str:
    return (
        f"Metric values, oldest first:\n"
        f"{json.dumps(list(case.values), separators=(',', ':'))}\n"
        f"Alert threshold: {case.threshold} (a breach is any future value "
        f"strictly above it).\n"
        f"Horizon: the next {case.horizon} observations (not shown).\n"
        f"Costs: taking action now costs {COST_ACT} and fully mitigates "
        f"any breach; a missed breach (no action taken and a breach "
        f"occurs) costs {COST_MISS}; no action and no breach costs 0."
    )


def prompt(case: Case, arm: str, packet: dict[str, Any]) -> str:
    body = base_prompt(case)
    if arm == "gnomon":
        body += ("\nComputed Gnomon evidence (deterministic, from the same "
                 "history):\n" + json.dumps(packet, separators=(",", ":")))
    return body + (
        '\nReturn {"breach_expected": true|false, "first_breach_step": '
        '<1-' + str(case.horizon) + ' or null>, "action": "act"|"monitor"}.')


def parse_answer(text: str, horizon: int) -> dict[str, Any]:
    # Balanced-span candidates, not a greedy regex: a correct JSON answer
    # followed by prose containing a brace — or preceded by an echoed
    # evidence packet, which only the gnomon arm's prompt even contains —
    # must not be scored as an invalid answer the model never gave. The
    # first candidate with the contract's shape wins; echoed packets fail
    # the shape check and are skipped.
    for payload in extract_json_objects(text):
        breach = payload.get("breach_expected")
        action = str(payload.get("action", "")).strip().lower()
        if not isinstance(breach, bool) or action not in {"act", "monitor"}:
            continue
        raw_step = payload.get("first_breach_step")
        # ``json.loads`` accepts NaN/Infinity, and ``True`` is an ``int``:
        # a malformed step must degrade to None, never crash a paid run.
        step = None
        if (isinstance(raw_step, (int, float))
                and not isinstance(raw_step, bool)
                and math.isfinite(raw_step)
                and 1 <= int(raw_step) <= horizon):
            step = int(raw_step)
        return {"valid": True, "breach_expected": breach,
                "first_breach_step": step, "action": action}
    return {"valid": False}


def decision_outcome(action: str, case: Case) -> dict[str, float]:
    cost = COST_ACT if action == "act" else (
        COST_MISS if case.truth_breach else 0.0)
    optimal = COST_ACT if case.truth_breach else 0.0
    return {"cost": cost, "regret": cost - optimal}


def product_rule(case: Case, packet: dict[str, Any],
                 composed: bool = False) -> dict[str, Any]:
    """The product's own decision rule, no LLM: act when the breach
    probability clears the cost break-even; call breach at even odds.

    Two mechanical readings of the same per-step marginals: the peak
    marginal (a lower bound on the probability of any breach) and, with
    ``composed=True``, the independence composition ``1 - prod(1 - p)``.
    Neither is the truth — the steps are dependent — but the model-value
    verdict is only honest against the *stronger* mechanical reading,
    otherwise "the model added value" can be earned by aggregating the
    packet's own numbers."""
    analysis = packet.get("threshold_analysis") or {}
    probabilities = analysis.get("probability_above_per_step") or []
    if probabilities:
        if composed:
            product = 1.0
            for value in probabilities:
                product *= 1.0 - min(1.0, max(0.0, float(value)))
            estimate = 1.0 - product
        else:
            estimate = max(probabilities)
        act = estimate >= COST_ACT / COST_MISS
        breach = estimate >= .5
        step = analysis.get("first_step_point_above")
    else:
        points = [row["q50"] for row in packet.get("forecast") or []]
        crossing = [step for step, value in enumerate(points, 1)
                    if value > case.threshold]
        breach = act = bool(crossing)
        step = crossing[0] if crossing else None
    return {"breach_expected": breach, "first_breach_step": step,
            "action": "act" if act else "monitor"}


def governed_product_rule(case: Case, packet: dict[str, Any]) -> dict[str, Any]:
    """The production decision contract, pricing withholding explicitly."""
    event = ((packet.get("threshold_analysis") or {}).get("horizon_event")
             or {})
    decision = packet.get("governed_decision") or {}
    recommendation = decision.get("recommended_action")
    probability = event.get("probability_any_breach")
    breach = (float(probability) >= 0.5
              if probability is not None else False)
    step = event.get("first_breach_step_median_conditional") if breach else None
    # Withholding is monitor-by-omission in this one-shot benchmark: the
    # client received no authority to act. It remains separately counted so
    # a product cannot improve apparent precision by withholding everything.
    return {
        "breach_expected": breach,
        "first_breach_step": step,
        "action": recommendation or "monitor",
        "withheld": recommendation is None,
    }


def _score(answer: dict[str, Any], case: Case) -> dict[str, Any]:
    if not answer.get("valid", True):
        # An unparseable answer is a monitor by omission: the operator got
        # nothing actionable. Recorded as invalid, never silently patched.
        answer = {"valid": False, "breach_expected": False,
                  "first_breach_step": None, "action": "monitor"}
    outcome = decision_outcome(answer["action"], case)
    breach_correct = answer["breach_expected"] == case.truth_breach
    timing_error = None
    if (case.truth_breach and answer["breach_expected"]
            and answer["first_breach_step"] is not None):
        timing_error = abs(answer["first_breach_step"]
                           - case.truth_first_step)
    return {
        "valid": bool(answer.get("valid", True)),
        "breach_expected": answer["breach_expected"],
        "first_breach_step": answer["first_breach_step"],
        "action": answer["action"],
        "cost": outcome["cost"], "regret": outcome["regret"],
        "action_optimal": outcome["regret"] == 0.0,
        "breach_correct": breach_correct,
        "timing_error": timing_error,
    }


def verify_arm_symmetry(cases: list[Case],
                        packets: dict[str, dict[str, Any]],
                        futures: dict[str, list[float]]) -> None:
    for case in cases:
        body = base_prompt(case)
        for arm in ARMS:
            text = prompt(case, arm, packets[case.case_id])
            if not text.startswith(body):
                raise ValueError(
                    f"arm {arm} altered the shared question for "
                    f"{case.case_id}")
        held_out = futures.get(case.case_id) or []
        if len(held_out) >= 3:
            marker = json.dumps(held_out[:8], separators=(",", ":"))[1:-1]
            # The history itself may legitimately render the marker — a
            # real sensor series carries runs of identical values, and a
            # constant future then matches a constant stretch of history.
            # Excise the history blob and scan the rest: the only way the
            # future can actually leak is through the packet or question.
            history_blob = json.dumps(
                list(case.values), separators=(",", ":"))
            for arm in ARMS:
                text = prompt(case, arm, packets[case.case_id])
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


def _arm_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Decision economics are scored over every row — an unparseable
    # answer is priced as the monitor-by-omission the operator lived
    # through. Call-quality metrics are scored over valid answers only:
    # imputing "no breach" for garbage would flatter a failing arm with
    # base-rate accuracy and a zero false-alarm rate.
    valid = [row for row in rows if row["valid"]]
    timing = [row["timing_error"] for row in valid
              if row["timing_error"] is not None]
    breach_truth = [row for row in valid if row["truth_breach"]]
    return {
        "mean_cost": statistics.mean(row["cost"] for row in rows),
        "mean_regret": statistics.mean(row["regret"] for row in rows),
        "action_optimal_rate": statistics.mean(
            row["action_optimal"] for row in rows),
        "breach_call_accuracy": (statistics.mean(
            row["breach_correct"] for row in valid) if valid else None),
        "breach_recall": (statistics.mean(
            row["breach_expected"] for row in breach_truth)
            if breach_truth else None),
        "false_alarm_rate": (statistics.mean(
            row["breach_expected"] for row in valid
            if not row["truth_breach"])
            if any(not row["truth_breach"] for row in valid) else None),
        "call_metrics_scored": len(valid),
        # Timing is over each arm's self-selected answered subset — an
        # arm can dodge it by never naming a step, which is why the
        # answer rate is reported next to the error and cross-arm MAE
        # comparisons must check both.
        "timing_mae": statistics.mean(timing) if timing else None,
        "timing_scored": len(timing),
        "timing_answer_rate": (statistics.mean(
            row["first_breach_step"] is not None for row in breach_truth)
            if breach_truth else None),
        "invalid_rate": statistics.mean(not row["valid"] for row in rows),
    }


def run(args: argparse.Namespace, client: Any = None) -> dict[str, Any]:
    if client is None:
        load_env_file()
        from benchmarks.common.openrouter import OpenRouterClient
        client = OpenRouterClient(
            args.model, api_key=os.environ.get(args.api_key_env),
            base_url=args.base_url, temperature=0,
            max_tokens=getattr(args, "max_tokens", 400),
            max_retries=4,
            reasoning_effort=getattr(args, "reasoning_effort", None))
    cases, corpus_provenance, futures = generate_cases(
        args.seed, args.cases,
        Path(getattr(args, "data_dir", None) or DATA_DIR))
    packets: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases):
        packets[case.case_id] = product_packet(case)
        if (index + 1) % 25 == 0:
            print(f"gnomon product runs {index + 1}/{len(cases)}",
                  flush=True)
    verify_arm_symmetry(cases, packets, futures)

    references_rows: dict[str, list[dict[str, Any]]] = {
        "gnomon_governed": [],
        "gnomon_rule_alone": [], "gnomon_rule_composed": [],
        "naive_persistence": [], "always_act": [], "never_act": [],
    }
    for case in cases:
        references_rows["gnomon_governed"].append(
            {**_score({"valid": True, **governed_product_rule(
                case, packets[case.case_id])}, case),
             "truth_breach": case.truth_breach,
             "withheld": governed_product_rule(
                 case, packets[case.case_id])["withheld"]})
        references_rows["gnomon_rule_alone"].append(
            {**_score({"valid": True, **product_rule(
                case, packets[case.case_id])}, case),
             "truth_breach": case.truth_breach})
        references_rows["gnomon_rule_composed"].append(
            {**_score({"valid": True, **product_rule(
                case, packets[case.case_id], composed=True)}, case),
             "truth_breach": case.truth_breach})
        last_above = case.values[-1] > case.threshold
        references_rows["naive_persistence"].append(
            {**_score({"valid": True, "breach_expected": last_above,
                       "first_breach_step": 1 if last_above else None,
                       "action": "act" if last_above else "monitor"}, case),
             "truth_breach": case.truth_breach})
        references_rows["always_act"].append(
            {**_score({"valid": True, "breach_expected": True,
                       "first_breach_step": 1, "action": "act"}, case),
             "truth_breach": case.truth_breach})
        references_rows["never_act"].append(
            {**_score({"valid": True, "breach_expected": False,
                       "first_breach_step": None, "action": "monitor"},
                      case),
             "truth_breach": case.truth_breach})
    references = {name: _arm_metrics(rows)
                  for name, rows in references_rows.items()}
    references["gnomon_governed"]["withholding_rate"] = statistics.mean(
        row["withheld"] for row in references_rows["gnomon_governed"])
    references["hindsight_optimal"] = {
        "mean_cost": statistics.mean(
            COST_ACT if case.truth_breach else 0.0 for case in cases),
        "mean_regret": 0.0,
    }

    # A case_id alone does not identify a case: the same seed with a
    # different --cases count (or corpus) yields sequential ids over
    # divergent content, including flipped truth labels. Rows carry the
    # full dataset identity and the answering model, and resume rejects
    # anything that does not match — silently pooling rows scored
    # against different cases or produced by a different model is the
    # one thing a paired benchmark must never do.
    model_name = getattr(args, "model", None)
    dataset_identity = (
        f"breachbench-generator-{GENERATOR_VERSION}:"
        f"seed={args.seed}:cases={args.cases}:"
        f"corpus={corpus_provenance['corpus_sha256'][:12]}")
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
        text = client.completions([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt(
                case, arm, packets[case.case_id])},
        ], n=1)[0]
        answer = parse_answer(text, case.horizon)
        return {"case_id": case.case_id, "arm": arm,
                "dataset": dataset_identity, "model": model_name,
                "origin": case.origin, "outcome_cell": case.outcome_cell,
                "truth_breach": case.truth_breach,
                "truth_first_step": case.truth_first_step,
                **_score(answer, case)}

    jobs = [(case, arm) for case in cases for arm in ARMS
            if (case.case_id, arm) not in completed]
    failures: list[tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pending = {pool.submit(one, *job): job for job in jobs}
        for future in as_completed(pending):
            case, arm = pending[future]
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

    metrics = {}
    for arm in ARMS:
        subset = [row for row in completed.values() if row["arm"] == arm]
        metrics[arm] = _arm_metrics(subset)
        metrics[arm]["by_outcome_cell"] = {
            cell: statistics.mean(
                row["regret"] for row in subset
                if row["outcome_cell"] == cell)
            for cell in OUTCOME_CELLS
            if any(row["outcome_cell"] == cell for row in subset)
        }
    control_better = gnomon_better = 0
    optimal_pairs = {"control_only": 0, "gnomon_only": 0}
    for case in cases:
        control = completed[(case.case_id, "control")]
        gnomon = completed[(case.case_id, "gnomon")]
        if control["cost"] < gnomon["cost"]:
            control_better += 1
        elif gnomon["cost"] < control["cost"]:
            gnomon_better += 1
        optimal_pairs["control_only"] += int(
            control["action_optimal"] and not gnomon["action_optimal"])
        optimal_pairs["gnomon_only"] += int(
            gnomon["action_optimal"] and not control["action_optimal"])
    governed_recommendations = {
        case.case_id: (packets[case.case_id].get("governed_decision") or {})
        .get("recommended_action")
        for case in cases
    }
    supported_ids = [case_id for case_id, recommendation
                     in governed_recommendations.items()
                     if recommendation is not None]
    preservation_rate = (statistics.mean(
        completed[(case_id, "gnomon")]["action"]
        == governed_recommendations[case_id]
        for case_id in supported_ids) if supported_ids else None)
    unsupported_action_rate = statistics.mean(
        completed[(case.case_id, "gnomon")]["action"] == "act"
        for case in cases
        if governed_recommendations[case.case_id] is None
    ) if len(supported_ids) < len(cases) else 0.0
    summary = {
        "schema_version": "0.2", "seed": args.seed, "cases": args.cases,
        "model": model_name, "temperature": 0,
        "horizon": HORIZON,
        "cost_model": {"act": COST_ACT, "missed_breach": COST_MISS},
        "provenance": {
            "evaluated_commit": _git_sha(),
            "harness_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
            "dataset_identity": dataset_identity,
            "cases": corpus_provenance,
        },
        "references": references,
        "metrics": metrics,
        "paired": {
            "primary_endpoint": "per_case_decision_cost",
            "paired_cases": len(cases),
            "control_cheaper": control_better,
            "gnomon_cheaper": gnomon_better,
            "exact_sign_p": exact_sign_p(control_better, gnomon_better),
            "action_optimal_mcnemar": {
                **optimal_pairs,
                "exact_p": exact_sign_p(optimal_pairs["control_only"],
                                        optimal_pairs["gnomon_only"]),
            },
            "agent_preservation": {
                "governed_recommendations": len(supported_ids),
                "preservation_rate": preservation_rate,
                "unsupported_action_rate": unsupported_action_rate,
            },
        },
        "verdicts": {
            "regret_reduction_vs_model_alone":
                metrics["control"]["mean_regret"]
                - metrics["gnomon"]["mean_regret"],
            "regret_reduction_vs_product_rule_alone":
                references["gnomon_governed"]["mean_regret"]
                - metrics["gnomon"]["mean_regret"],
            "product_rule_basis": "governed_dependence_aware_policy",
            "regret_reduction_vs_best_constant_policy":
                min(references["always_act"]["mean_regret"],
                    references["never_act"]["mean_regret"])
                - metrics["gnomon"]["mean_regret"],
            "reading": (
                "useful means positive on all three: cheaper decisions "
                "than the model alone, cheaper than the product's own "
                "governed no-LLM recommendation (with withholding priced as "
                "monitor-by-omission), and cheaper than "
                "the best constant policy. Positive on the first only "
                "means Gnomon carried the model; positive on the second "
                "only means the model carried Gnomon; failing the third "
                "means nobody beat a policy that ignores the data."),
        },
        "design": {
            "matched": True,
            "reasoning_effort": getattr(args, "reasoning_effort", None),
            "initial_max_tokens": getattr(args, "max_tokens", 400),
            "arms_differ_by_packet_block_only": True,
            "truth_is_realized_held_out_future": True,
            "held_out_future_absent_from_prompts_verified": True,
            "gnomon_packet_is_production_output": True,
            "costs_stated_in_prompt": True,
            "memorization_defense":
                "per_case_seeded_positive_affine_transform;"
                "threshold_transformed_identically;values_only_prompts",
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
    parser.add_argument("--cases", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=400,
                        help="Initial completion budget; retries may escalate it.")
    parser.add_argument(
        "--reasoning-effort", default=None,
        choices=("none", "low", "medium", "high"),
        help="Explicit provider reasoning mode; omitted preserves the model default.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

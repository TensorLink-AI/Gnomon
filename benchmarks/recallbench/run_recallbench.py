"""RecallBench: is a hosted model's forecasting edge skill, or recall?

A hosted LLM that beats Gnomon's estimators on public series has two
possible explanations, and they have opposite product implications. If
the model genuinely forecasts better, it deserves a governed candidate
lane. If it has memorized these public series' realized futures, the
"edge" is a lookup that will not transfer to a client's private data —
and building a candidate lane on it would ship an illusion.

This benchmark separates the two with matched arms over identical real
windows (the same corpora as BreachBench plus the supported-cadence
DossierBench series):

- ``raw``  — the model sees the true recorded values. Skill and recall
  both help here.
- ``anon`` — the model sees the same window through a seeded positive
  affine transform. Recall is defeated (verbatim lookup fails); genuine
  pattern-based forecasting is preserved, because trend direction,
  seasonality, and relative structure survive an affine map.

The metric is MASE — mean absolute error scaled by the in-sample
seasonal-naive error of the same (transformed) history — because MASE
is invariant under positive affine transforms: the transform itself
cannot move the score, so any raw-versus-anon gap belongs to the model.
The deterministic references (seasonal naive, last value) must score
identically across arms up to rounding, and the harness checks that.

Two verdicts, both paired per window:

- ``memorization_delta``: model MASE on anon minus raw. Near zero means
  the raw edge is skill; large and positive means it was recall.
- ``skill_vs_gnomon_anonymized``: Gnomon minus model within the anon
  arm — the leakage-controlled version of "the model forecasts better
  than the engine", the claim a governed candidate lane would rest on.

Truth futures exist only in the scorer, held out and verified absent
from every prompt.
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

from benchmarks.breachbench.run_breachbench import (  # noqa: E402
    _grid_timestamps,
    series_frequency,
)
from benchmarks.common.envfile import load_env_file  # noqa: E402
from benchmarks.common.openrouter import extract_json_array  # noqa: E402

GENERATOR_VERSION = "0.1"
DATA_DIRS = (
    ROOT / "benchmarks" / "breachbench" / "data",
    ROOT / "benchmarks" / "dossierbench" / "data",
)
HORIZON = 12
ARMS = ("raw", "anon")
#: Default seasonal period per supported cadence, used for the seasonal
#: naive reference and the MASE scale. A window too short for two full
#: cycles degrades to season 1 (plain naive), identically in both arms.
SEASONS = {"5min": 288, "h": 24, "D": 7, "W": 52, "MS": 12}
#: Cadence tokens Gnomon cannot run at their true frequency (no yearly or
#: quarterly grid); excluded rather than mislabelled as daily.
_UNSUPPORTED_TOKENS = {"yearly", "annual", "quarterly"}
SYSTEM = """You are a forecasting engine. Infer only from the supplied
values. Answer with one JSON array of numbers and nothing else."""


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass(frozen=True)
class Case:
    case_id: str
    origin: str
    frequency: str
    season: int
    values: tuple[float, ...]      # raw history, rounded as shown
    anon_values: tuple[float, ...]  # affine-transformed history, as shown
    scale_a: float
    shift_b: float


def load_joint_corpus(
    data_dirs: tuple[Path, ...] = DATA_DIRS,
) -> tuple[dict[str, list[float]], list[str]]:
    """Every corpus series whose true cadence Gnomon can run, plus the
    names excluded because their cadence has no supported grid."""
    corpus: dict[str, list[float]] = {}
    excluded: list[str] = []
    for data_dir in data_dirs:
        for path in sorted(data_dir.glob("*.csv")):
            name = path.stem
            if set(name.split("_")) & _UNSUPPORTED_TOKENS:
                excluded.append(name)
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            corpus[name] = [float(line) for line in lines[1:]
                            if line.strip()]
    if not corpus:
        raise FileNotFoundError(f"no usable corpus series under {data_dirs}")
    return corpus, sorted(excluded)


def _season_for(frequency: str, window: int) -> int:
    season = SEASONS.get(frequency, 1)
    return season if window >= 2 * season else 1


def generate_cases(
    seed: int, count: int, data_dirs: tuple[Path, ...] = DATA_DIRS,
) -> tuple[list[Case], dict[str, Any], dict[str, list[float]]]:
    """Windowed real cases, futures non-overlapping within a series.

    Both arms share one window: the anon values are the raw values under
    a per-case positive affine transform, so the pairing is exact.
    """
    corpus, excluded = load_joint_corpus(data_dirs)
    rng = random.Random(seed)
    names = sorted(corpus)
    cases: list[Case] = []
    futures: dict[str, list[float]] = {}
    used_cutoffs: dict[str, list[int]] = {name: [] for name in names}
    seen: set[tuple[str, int, int]] = set()
    skipped = {"too_short": 0, "degenerate": 0, "duplicate": 0,
               "future_overlap": 0}
    attempts = 0
    limit = 400 * count
    while len(cases) < count and attempts < limit:
        attempts += 1
        name = names[rng.randrange(len(names))]
        series = corpus[name]
        window = rng.choice((48, 96))
        if len(series) < window + HORIZON + 1:
            skipped["too_short"] += 1
            continue
        cutoff = rng.randrange(window, len(series) - HORIZON + 1)
        key = (name, cutoff, window)
        if key in seen:
            skipped["duplicate"] += 1
            continue
        if any(abs(cutoff - other) < HORIZON
               for other in used_cutoffs[name]):
            skipped["future_overlap"] += 1
            continue
        history = [float(v) for v in series[cutoff - window:cutoff]]
        future = [float(v) for v in series[cutoff:cutoff + HORIZON]]
        if max(history) == min(history):
            skipped["degenerate"] += 1
            continue
        a = rng.uniform(.6, 2.4)
        b = rng.uniform(40, 900) - a * statistics.median(history)
        seen.add(key)
        used_cutoffs[name].append(cutoff)
        frequency = series_frequency(name)
        case = Case(
            f"r{seed}-{len(cases):04d}", name, frequency,
            _season_for(frequency, window),
            tuple(round(v, 4) for v in history),
            tuple(round(a * v + b, 4) for v in history),
            a, b)
        cases.append(case)
        futures[case.case_id] = [float(v) for v in future]
    if len(cases) < count:
        raise ValueError(
            f"only {len(cases)}/{count} cases after {attempts} attempts; "
            f"skipped={skipped}")
    provenance = {
        "corpus_series": names,
        "excluded_unsupported_cadence": excluded,
        "corpus_sha256": hashlib.sha256(b"".join(
            path.read_bytes() for data_dir in data_dirs
            for path in sorted(data_dir.glob("*.csv")))).hexdigest(),
        "cases_per_series": {name: len(cutoffs) for name, cutoffs
                             in sorted(used_cutoffs.items()) if cutoffs},
        "attempts": attempts, "skipped": skipped,
        "independence": ("realized_futures_non_overlapping_within_series;"
                         "histories_may_overlap;labels_can_still_comove_"
                         "through_shared_regimes_see_cases_per_series"),
        "anonymization": ("per_case_seeded_positive_affine_transform;"
                          "identical_window_in_both_arms;"
                          "mase_is_affine_invariant"),
    }
    return cases, provenance, futures


def arm_values(case: Case, arm: str) -> tuple[float, ...]:
    return case.values if arm == "raw" else case.anon_values


def arm_future(case: Case, arm: str,
               future: list[float]) -> list[float]:
    if arm == "raw":
        return [float(v) for v in future]
    return [case.scale_a * float(v) + case.shift_b for v in future]


def seasonal_naive(history: tuple[float, ...], season: int,
                   horizon: int) -> list[float]:
    return [history[len(history) - season + ((step - 1) % season)]
            for step in range(1, horizon + 1)]


def mase(forecast: list[float], actual: list[float],
         history: tuple[float, ...], season: int) -> float:
    """MAE scaled by the in-sample seasonal-naive MAE of the same
    history. Invariant under positive affine transforms of the series:
    numerator and denominator both scale by |a| and offsets cancel."""
    insample = [abs(history[i] - history[i - season])
                for i in range(season, len(history))]
    scale = statistics.mean(insample) if insample else 0.0
    scale = max(scale, 1e-9 * max(abs(max(history)), abs(min(history)), 1.0))
    return statistics.mean(
        abs(f - y) for f, y in zip(forecast, actual)) / scale


def gnomon_forecast_points(case: Case, arm: str) -> list[float] | None:
    """Gnomon's real production forecast for the arm's exact values, at
    the series' true cadence. None when the engine abstains."""
    import shutil

    from gnomon import forecast as gnomon_forecast
    from gnomon.contracts import GnomonError

    run_dir = Path(tempfile.mkdtemp(prefix="recallbench-"))
    try:
        values = arm_values(case, arm)
        stamps = _grid_timestamps(case.frequency, len(values))
        lines = ["timestamp,value"] + [
            f"{stamp},{value!r}" for stamp, value in zip(stamps, values)]
        csv_path = run_dir / "history.csv"
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            artifact, _ = gnomon_forecast(
                str(csv_path), time_column="timestamp",
                target_column="value", horizon=HORIZON,
                frequency=case.frequency, output=str(run_dir / "out"))
        except GnomonError:
            return None
        rows = artifact.results[0].forecast or []
        if len(rows) != HORIZON:
            return None
        return [float(row.get("q50", row["point"])) for row in rows]
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def prompt(case: Case, arm: str) -> str:
    values = arm_values(case, arm)
    return (
        f"Metric values, oldest first:\n"
        f"{json.dumps(list(values), separators=(',', ':'))}\n"
        f"Forecast the next {HORIZON} values of this series.\n"
        f"Return a JSON array of exactly {HORIZON} numbers and nothing "
        f"else."
    )


def parse_forecast(text: str, horizon: int) -> list[float] | None:
    try:
        payload = extract_json_array(text)
    except ValueError:
        return None
    if len(payload) != horizon:
        return None
    values: list[float] = []
    for item in payload:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        value = float(item)
        if not math.isfinite(value):
            return None
        values.append(value)
    return values


def verify_no_future_leakage(cases: list[Case],
                             futures: dict[str, list[float]]) -> None:
    for case in cases:
        for arm in ARMS:
            held_out = [round(v, 4)
                        for v in arm_future(case, arm, futures[case.case_id])]
            if len(held_out) < 3:
                continue
            marker = json.dumps(held_out[:8], separators=(",", ":"))[1:-1]
            history_blob = json.dumps(
                list(arm_values(case, arm)), separators=(",", ":"))
            text = prompt(case, arm)
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


def _paired(deltas: list[float]) -> dict[str, Any]:
    positive = sum(delta > 0 for delta in deltas)
    negative = sum(delta < 0 for delta in deltas)
    return {
        "pairs": len(deltas),
        "positive": positive, "negative": negative,
        "mean_delta": round(statistics.mean(deltas), 6) if deltas else None,
        "exact_sign_p": exact_sign_p(positive, negative),
    }


def run(args: argparse.Namespace, client: Any = None) -> dict[str, Any]:
    if client is None:
        load_env_file()
        from benchmarks.common.openrouter import OpenRouterClient
        client = OpenRouterClient(
            args.model, api_key=os.environ.get(args.api_key_env),
            base_url=args.base_url, temperature=0, max_tokens=600,
            max_retries=4,
            reasoning_effort=getattr(args, "reasoning_effort", "none"))
    cases, corpus_provenance, futures = generate_cases(
        args.seed, args.cases)
    verify_no_future_leakage(cases, futures)

    # Deterministic scores, no model involved. The seasonal-naive MASE
    # must be identical across arms up to shown-value rounding — that is
    # the harness's own check that the metric cannot see the transform.
    reference_scores: dict[str, dict[tuple[str, str], float]] = {
        "gnomon": {}, "seasonal_naive": {}, "last_value": {}}
    gnomon_abstained = 0
    for index, case in enumerate(cases):
        for arm in ARMS:
            history = arm_values(case, arm)
            actual = arm_future(case, arm, futures[case.case_id])
            points = gnomon_forecast_points(case, arm)
            if points is not None:
                reference_scores["gnomon"][(case.case_id, arm)] = mase(
                    points, actual, history, case.season)
            else:
                gnomon_abstained += 1
            reference_scores["seasonal_naive"][(case.case_id, arm)] = mase(
                seasonal_naive(history, case.season, HORIZON), actual,
                history, case.season)
            reference_scores["last_value"][(case.case_id, arm)] = mase(
                [history[-1]] * HORIZON, actual, history, case.season)
        if (index + 1) % 25 == 0:
            print(f"gnomon reference runs {index + 1}/{len(cases)}",
                  flush=True)
    naive_invariance = max(
        abs(reference_scores["seasonal_naive"][(case.case_id, "raw")]
            - reference_scores["seasonal_naive"][(case.case_id, "anon")])
        for case in cases)
    if naive_invariance > 0.01:
        raise ValueError(
            f"MASE moved {naive_invariance:.4f} across arms for the "
            f"seasonal naive; the metric must not see the transform")

    model_name = getattr(args, "model", None)
    dataset_identity = (
        f"recallbench-generator-{GENERATOR_VERSION}:"
        f"seed={args.seed}:cases={args.cases}:"
        f"corpus={corpus_provenance['corpus_sha256'][:12]}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "rows.jsonl"
    summary_path = output / "summary.json"
    prior_usage = None
    if args.resume and summary_path.exists():
        try:
            prior_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (prior_summary.get("model") == model_name
                    and prior_summary.get("reasoning_effort") == getattr(
                        args, "reasoning_effort", "none")
                    and (prior_summary.get("provenance") or {}).get(
                        "dataset_identity") == dataset_identity):
                prior_usage = prior_summary.get("usage")
        except (OSError, json.JSONDecodeError):
            prior_usage = None
    if not args.resume:
        rows_path.unlink(missing_ok=True)
    valid_ids = {case.case_id for case in cases}
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if args.resume and rows_path.exists():
        stale = malformed = 0
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if (row.get("case_id") in valid_ids and row.get("arm") in ARMS
                    and row.get("dataset") == dataset_identity
                    and row.get("model") == model_name
                    and row.get("reasoning_effort") == getattr(
                        args, "reasoning_effort", "none")):
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
            {"role": "user", "content": prompt(case, arm)},
        ], n=1)[0]
        forecast = parse_forecast(text, HORIZON)
        row: dict[str, Any] = {
            "case_id": case.case_id, "arm": arm,
            "dataset": dataset_identity, "model": model_name,
            "reasoning_effort": getattr(args, "reasoning_effort", "none"),
            "origin": case.origin, "frequency": case.frequency,
            "valid": forecast is not None,
        }
        if forecast is not None:
            row["mase"] = mase(
                forecast, arm_future(case, arm, futures[case.case_id]),
                arm_values(case, arm), case.season)
        return row

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

    # Pairing: a case contributes to a paired comparison only when every
    # score it needs exists (both arms valid; gnomon did not abstain).
    def model_mase(case_id: str, arm: str) -> float | None:
        row = completed.get((case_id, arm)) or {}
        return row.get("mase") if row.get("valid") else None

    memorization_deltas: list[float] = []
    skill_anon_deltas: list[float] = []
    skill_raw_deltas: list[float] = []
    for case in cases:
        raw_score = model_mase(case.case_id, "raw")
        anon_score = model_mase(case.case_id, "anon")
        if raw_score is not None and anon_score is not None:
            memorization_deltas.append(anon_score - raw_score)
        for arm, bucket in (("anon", skill_anon_deltas),
                            ("raw", skill_raw_deltas)):
            engine = reference_scores["gnomon"].get((case.case_id, arm))
            score = model_mase(case.case_id, arm)
            if engine is not None and score is not None:
                bucket.append(engine - score)

    def arm_metrics(arm: str) -> dict[str, Any]:
        scored = [completed[(case.case_id, arm)] for case in cases
                  if (case.case_id, arm) in completed]
        valid = [row["mase"] for row in scored if row.get("valid")]
        return {
            "model_mean_mase": (round(statistics.mean(valid), 6)
                                if valid else None),
            "model_scored": len(valid),
            "invalid_rate": (statistics.mean(
                not row.get("valid") for row in scored)
                if scored else None),
            **{f"{name}_mean_mase": round(statistics.mean(
                score for (case_id, row_arm), score in scores.items()
                if row_arm == arm), 6)
               for name, scores in reference_scores.items() if scores},
        }

    summary = {
        "schema_version": "0.1", "seed": args.seed, "cases": args.cases,
        "model": model_name, "temperature": 0, "horizon": HORIZON,
        "reasoning_effort": getattr(args, "reasoning_effort", "none"),
        "provenance": {
            "evaluated_commit": _git_sha(),
            "harness_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
            "dataset_identity": dataset_identity,
            "cases": corpus_provenance,
            "gnomon_abstentions": gnomon_abstained,
            "naive_mase_max_cross_arm_drift": round(naive_invariance, 6),
        },
        "metrics": {arm: arm_metrics(arm) for arm in ARMS},
        "verdicts": {
            "memorization_delta": _paired(memorization_deltas),
            "skill_vs_gnomon_anonymized": _paired(skill_anon_deltas),
            "model_vs_gnomon_raw_reference": _paired(skill_raw_deltas),
            "reading": (
                "memorization_delta near zero means the model's raw-arm "
                "edge is transferable forecasting skill; large and "
                "positive means it was recall of these public series and "
                "will not transfer to private client data. "
                "skill_vs_gnomon_anonymized positive (engine MASE above "
                "model MASE with recall defeated) is the only reading "
                "that justifies a governed LLM-forecast candidate lane."),
        },
        "design": {
            "matched": True,
            "identical_window_in_both_arms": True,
            "metric_affine_invariant": "mase; cross-arm naive drift "
                                       "checked pre-flight",
            "truth_is_realized_held_out_future": True,
            "held_out_future_absent_from_prompts_verified": True,
            "gnomon_reference_is_production_forecast": True,
        },
    }
    if hasattr(client, "usage_summary"):
        current_usage = dict(client.usage_summary)
        if isinstance(prior_usage, dict):
            additive = {
                "requests", "transport_attempts", "prompt_tokens",
                "completion_tokens", "truncation_escalations", "cost_usd"}
            for key in additive:
                current_usage[key] = (
                    float(prior_usage.get(key, 0) or 0)
                    + float(current_usage.get(key, 0) or 0))
                if key != "cost_usd":
                    current_usage[key] = int(current_usage[key])
            current_usage["accounting"] = (
                "cumulative_across_matching_resume_invocations")
        summary["usage"] = current_usage
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--base-url", default="https://api.engy.ai/v1")
    parser.add_argument("--api-key-env", default="ENGY_API_KEY")
    parser.add_argument("--cases", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--reasoning-effort", default="none",
        choices=["none", "low", "medium", "high"],
        help="Hosted-model reasoning mode; part of resume identity.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

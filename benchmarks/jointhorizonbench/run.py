"""Run the frozen P6 joint-horizon corpus serially and resumably."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import signal
import statistics
import subprocess
from typing import Any

from benchmarks.breachbench.run_breachbench import (
    COST_ACT,
    COST_MISS,
    GENERATOR_VERSION,
    decision_outcome,
    generate_cases,
    governed_product_rule,
    product_packet,
)


SEED = 20260906
CASE_COUNT = 60
RUNNER_VERSION = "p6-joint-horizon-1"


class CaseTimeout(RuntimeError):
    pass


def _timeout(_signum: int, _frame: Any) -> None:
    raise CaseTimeout("case exceeded its timeout")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _immutable_surface(packet: dict[str, Any]) -> dict[str, Any]:
    """All pre-P6 numerical and decision fields, excluding the new block."""
    analysis = packet.get("threshold_analysis") or {}
    event = dict(analysis.get("horizon_event") or {})
    event.pop("cumulative_horizon", None)
    return {
        "forecast": packet.get("forecast"),
        "probability_above_per_step": analysis.get(
            "probability_above_per_step"),
        "horizon_event": event or None,
        "governed_decision": packet.get("governed_decision"),
    }


def _independence(probabilities: list[float]) -> float | None:
    if not probabilities:
        return None
    survival = 1.0
    for value in probabilities:
        survival *= 1.0 - min(1.0, max(0.0, float(value)))
    return 1.0 - survival


def evaluate(case: Any, future: list[float]) -> dict[str, Any]:
    packet = product_packet(case)
    analysis = packet.get("threshold_analysis") or {}
    event = analysis.get("horizon_event") or {}
    forecast = packet.get("forecast") or []
    probabilities = [float(value) for value in (
        analysis.get("probability_above_per_step") or [])]
    action = governed_product_rule(case, packet)
    cumulative = event.get("cumulative_horizon")
    marginal_lower = sum(float(row["q10"]) for row in forecast)
    marginal_upper = sum(float(row["q90"]) for row in forecast)
    actual_total = sum(float(value) for value in future)
    event_probability = event.get("probability_any_breach")
    first_distribution = event.get("first_breach_step_probability") or {}
    return {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "generator_version": GENERATOR_VERSION,
        "case_id": case.case_id,
        "complete": packet.get("status") != "abstained" and bool(forecast),
        "case": {
            "origin": case.origin,
            "frequency": case.frequency,
            "history_length": case.history_length,
            "history_band": case.history_band,
            "outcome_cell": case.outcome_cell,
            "threshold": case.threshold,
            "horizon": case.horizon,
        },
        "truth": {
            "breach": case.truth_breach,
            "first_step": case.truth_first_step,
            "actual_total": actual_total,
        },
        "immutable_surface": _immutable_surface(packet),
        "immutable_surface_sha256": _hash(_immutable_surface(packet)),
        "diagnostics": {
            "event_probability": event_probability,
            "peak_step_probability": max(probabilities) if probabilities else None,
            "raw_independence_probability": _independence(probabilities),
            "first_distribution_sum": sum(
                float(value) for value in first_distribution.values()),
            "dependence_preserved": event.get("dependence_preserved"),
            "event_method": event.get("method"),
            "event_support": event.get("support"),
            "governed_action": action["action"],
            "governed_withheld": action["withheld"],
            "governed_cost": decision_outcome(action["action"], case)["cost"],
            "marginal_total_interval": {
                "lower": marginal_lower,
                "upper": marginal_upper,
                "width": marginal_upper - marginal_lower,
            },
        },
        "cumulative_horizon": cumulative,
        # Held-out truth remains runner-side and is never included in packet.
        "runtime_input_sha256": _hash({
            "values": case.values,
            "threshold": case.threshold,
            "horizon": case.horizon,
            "frequency": case.frequency,
        }),
    }


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def summarize(rows: list[dict[str, Any]], baseline: Path | None) -> dict[str, Any]:
    complete = [row for row in rows if row.get("complete")]
    probabilities = [row for row in complete
                     if row["diagnostics"]["event_probability"] is not None]
    dependent = [row for row in probabilities
                 if row["diagnostics"]["dependence_preserved"]]
    cumulative = [row for row in complete
                  if (row.get("cumulative_horizon") or {}).get("status")
                  == "available"]
    path_rows = [row for row in probabilities if (
        (row["immutable_surface"].get("horizon_event") or {}).get(
            "first_breach_step_probability") is not None)]

    def brier(row: dict[str, Any], key: str) -> float:
        estimate = float(row["diagnostics"][key])
        truth = float(row["truth"]["breach"])
        return (estimate - truth) ** 2

    joint_brier = _mean([brier(row, "event_probability") for row in dependent])
    peak_brier = _mean([brier(row, "peak_step_probability") for row in dependent
                       if row["diagnostics"]["peak_step_probability"] is not None])
    independent_brier = _mean([
        brier(row, "raw_independence_probability") for row in dependent
        if row["diagnostics"]["raw_independence_probability"] is not None])
    costs = [float(row["diagnostics"]["governed_cost"])
             for row in probabilities]
    truth_breaches = sum(bool(row["truth"]["breach"]) for row in probabilities)
    always_act = len(probabilities) * COST_ACT
    always_monitor = truth_breaches * COST_MISS

    total_coverages: list[bool] = []
    total_widths: list[float] = []
    marginal_widths: list[float] = []
    cumulative_coherent = True
    cumulative_identity = True
    fallback_truth = True
    for row in cumulative:
        projection = row["cumulative_horizon"]
        interval = projection.get("total_interval_80") or {}
        lower, upper = interval.get("lower"), interval.get("upper")
        median = projection.get("median_total")
        if lower is None or upper is None or median is None:
            cumulative_coherent = False
            continue
        total_coverages.append(
            float(lower) <= float(row["truth"]["actual_total"]) <= float(upper))
        total_widths.append(float(upper) - float(lower))
        marginal_widths.append(float(
            row["diagnostics"]["marginal_total_interval"]["width"]))
        cumulative_coherent &= float(lower) <= float(median) <= float(upper)
        q50_sum = sum(float(item["q50"])
                      for item in row["immutable_surface"]["forecast"])
        cumulative_identity &= math.isclose(
            float(projection["point_total"]), q50_sum,
            rel_tol=0.0, abs_tol=0.01)
        if projection.get("dependence_preserved"):
            fallback_truth &= projection.get("basis") == (
                "aligned_fold_residual_trajectory_replay_v1")
        else:
            fallback_truth &= bool(projection.get("assumptions"))

    baseline_rows: dict[str, dict[str, Any]] = {}
    if baseline:
        baseline_rows = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((baseline / "cases").glob("*.json"))
        }
    parity = (len(baseline_rows) == CASE_COUNT and all(
        baseline_rows.get(row["case_id"], {}).get(
            "immutable_surface_sha256") == row["immutable_surface_sha256"]
        for row in rows)) if baseline else None
    coverage = statistics.mean(total_coverages) if total_coverages else None
    mean_total_width = _mean(total_widths)
    mean_marginal_width = _mean(marginal_widths)
    probability_skill_applicable = bool(dependent)
    probability_skill = (
        not probability_skill_applicable
        or (joint_brier is not None and peak_brier is not None
            and independent_brier is not None
            and joint_brier <= min(peak_brier, independent_brier) + .02))
    any_step_coherence = all(
        float(row["diagnostics"]["peak_step_probability"])
        <= float(row["diagnostics"]["event_probability"])
        <= min(1.0, sum(float(value) for value in
                        row["immutable_surface"][
                            "probability_above_per_step"]))
        for row in dependent
    )
    first_step_coherence = True
    for row in path_rows:
        event = row["immutable_surface"]["horizon_event"]
        expected = (
            event.get("bootstrap_diagnostic_probability")
            if event.get("method") == "independence_composed_marginals_v1"
            else event.get("probability_any_breach"))
        if expected is not None:
            first_step_coherence &= math.isclose(
                float(row["diagnostics"]["first_distribution_sum"]),
                float(expected), rel_tol=0.0, abs_tol=1e-6)
    gates = {
        "completion": len(complete) == CASE_COUNT,
        "numerical_parity": parity,
        "any_step_coherence": any_step_coherence and first_step_coherence,
        "probability_skill": probability_skill,
        "governed_cost": bool(probabilities)
            and sum(costs) <= min(always_act, always_monitor),
        "cumulative_identity": bool(cumulative) and cumulative_identity,
        "cumulative_coverage": coverage is not None and .65 <= coverage <= .95,
        "cumulative_width": mean_total_width is not None
            and mean_marginal_width is not None
            and mean_total_width <= mean_marginal_width,
        "cumulative_coherence": bool(cumulative) and cumulative_coherent,
        "fallback_truth": bool(cumulative) and fallback_truth,
    }
    return {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "denominators": {
            "cases": CASE_COUNT,
            "complete": len(complete),
            "probability_bearing": len(probabilities),
            "dependence_preserved": len(dependent),
            "cumulative_available": len(cumulative),
        },
        "metrics": {
            "joint_brier_dependent": joint_brier,
            "peak_brier_dependent": peak_brier,
            "independence_brier_dependent": independent_brier,
            "probability_skill_applicable": probability_skill_applicable,
            "governed_cost": sum(costs),
            "always_act_cost": always_act,
            "always_monitor_cost": always_monitor,
            "total_interval_coverage": coverage,
            "mean_total_interval_width": mean_total_width,
            "mean_summed_marginal_width": mean_marginal_width,
        },
        "gates": gates,
        "decision_ready": baseline is not None and all(
            value is True for value in gates.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    case_dir = args.output / "cases"
    cases, provenance, futures = generate_cases(SEED, CASE_COUNT)
    signal.signal(signal.SIGALRM, _timeout)
    for case in cases:
        target = case_dir / f"{case.case_id}.json"
        if target.exists():
            continue
        error = None
        for attempt in range(args.retries + 1):
            try:
                signal.alarm(args.timeout)
                row = evaluate(case, futures[case.case_id])
                signal.alarm(0)
                row["attempt"] = attempt + 1
                _atomic_json(target, row)
                error = None
                break
            except Exception as exc:
                signal.alarm(0)
                error = f"{type(exc).__name__}: {exc}"[:1000]
        if error is not None:
            _atomic_json(target, {
                "schema_version": 1, "case_id": case.case_id,
                "complete": False, "attempt": args.retries + 1,
                "error": error,
            })
    rows = [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(case_dir.glob("*.json"))]
    summary = summarize(rows, args.baseline)
    identity = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runner_version": RUNNER_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": SEED,
        "case_count": CASE_COUNT,
        "provenance": provenance,
        "jobs": 1,
        "retries": args.retries,
        "timeout_seconds": args.timeout,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True,
            text=True, check=False).stdout.strip(),
        "git_status": subprocess.run(
            ["git", "status", "--short"], capture_output=True,
            text=True, check=False).stdout.splitlines(),
    }
    _atomic_json(args.output / "run_identity.json", identity)
    _atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    # Baselines are descriptive; treatment comparisons enforce the gates.
    return 0 if args.baseline is None or summary["decision_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the frozen, resumable P4 multivariate benchmark.

This is deliberately a small local corpus.  Runtime code receives only the
CSV values and the ordinary ``multivariate`` switch; case families and future
holdouts remain benchmark-side labels.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import random
import signal
import subprocess
from typing import Any

from gnomon.runtime import forecast


GENERATOR_VERSION = "p4-multivariate-1"
SEEDS = tuple(range(4100, 4112))
TRAINING_POINTS = 192
HORIZON = 12


@dataclass(frozen=True)
class Case:
    case_id: str
    family: str
    seed: int
    target: tuple[float, ...]
    related: tuple[float, ...]


class CaseTimeout(RuntimeError):
    pass


def _timeout(_signum: int, _frame: Any) -> None:
    raise CaseTimeout("case exceeded 60 seconds")


def _noise(rng: random.Random, scale: float) -> float:
    return rng.uniform(-scale, scale)


def generate_case(seed: int) -> Case:
    """Generate one frozen case without exposing its family to Gnomon."""
    rng = random.Random(seed)
    index = seed - SEEDS[0]
    count = TRAINING_POINTS + HORIZON
    phase = rng.uniform(-math.pi, math.pi)
    if index < 6:
        family = "lagged_driver"
        latent = [
            100 + 0.22 * step
            + 13 * math.sin(step / 5.5 + phase)
            + 4 * math.cos(step / 13 + phase / 2)
            for step in range(count + 1)
        ]
        target = [value + _noise(rng, 0.25) for value in latent[:-1]]
        # The related target leads the target by one step.  Its future is not
        # supplied: VAR must forecast both channels recursively from prefixes.
        related = [
            0.82 * latent[step + 1]
            + 3 * math.cos(step / 5.5 + phase) + 18
            + _noise(rng, 0.25)
            for step in range(count)
        ]
    elif index < 9:
        family = "shared_pattern_control"
        target = [
            70 + 0.12 * step + 8 * math.sin(2 * math.pi * step / 24 + phase)
            + _noise(rng, 0.2)
            for step in range(count)
        ]
        related = [
            35 + 0.06 * step + 4 * math.sin(2 * math.pi * step / 24 + phase)
            + _noise(rng, 1.5)
            for step in range(count)
        ]
    else:
        family = "independent_control"
        target = [
            90 + 6 * math.sin(2 * math.pi * step / 24 + phase)
            + _noise(rng, 0.3)
            for step in range(count)
        ]
        related = [
            40 + ((step * (17 + index)) % 23) + _noise(rng, 1.0)
            for step in range(count)
        ]
    return Case(f"{family}-{seed}", family, seed,
                tuple(target), tuple(related))


def _write_training(path: Path, case: Case) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "series", "value"])
        for name, values in (("target", case.target),
                             ("related", case.related)):
            for step, value in enumerate(values[:TRAINING_POINTS]):
                writer.writerow([
                    (start + timedelta(hours=step)).isoformat(), name,
                    repr(float(value)),
                ])


def _smape(actual: list[float], predicted: list[float]) -> float:
    terms = []
    for observed, estimate in zip(actual, predicted):
        denominator = abs(observed) + abs(estimate)
        terms.append(0.0 if denominator <= 1e-12
                     else 200 * abs(observed - estimate) / denominator)
    return sum(terms) / len(terms)


def _points(artifact: Any, series: str) -> list[float]:
    result = next(item for item in artifact.results if item.series == series)
    return [float(row["point"]) for row in result.forecast]


def _gate(artifact: Any, series: str) -> dict[str, Any] | None:
    return next((item.payload for item in artifact.evidence
                 if item.kind == "multivariate_gate" and item.series == series),
                None)


def run_case(case: Case, output: Path) -> dict[str, Any]:
    case_dir = output / "runs" / case.case_id
    input_path = case_dir / "training.csv"
    _write_training(input_path, case)
    common = dict(
        input_path=str(input_path), time_column="timestamp",
        target_column="value", series_column="series", horizon=HORIZON,
    )
    control, _ = forecast(
        **common, multivariate=False, output=str(case_dir / "control"))
    treatment, _ = forecast(
        **common, multivariate=True, output=str(case_dir / "treatment"))
    actual = list(case.target[TRAINING_POINTS:])
    control_points = _points(control, "target")
    treatment_points = _points(treatment, "target")
    treatment_result = next(
        item for item in treatment.results if item.series == "target")
    control_error = _smape(actual, control_points)
    treatment_error = _smape(actual, treatment_points)
    return {
        "schema_version": 1,
        "case_id": case.case_id,
        "family": case.family,
        "seed": case.seed,
        "complete": len(control_points) == HORIZON
                    and len(treatment_points) == HORIZON,
        "actual": actual,
        "control_points": control_points,
        "treatment_points": treatment_points,
        "control_smape": control_error,
        "treatment_smape": treatment_error,
        "smape_improvement": control_error - treatment_error,
        "relative_improvement": (
            (control_error - treatment_error) / control_error
            if control_error > 1e-12 else 0.0
        ),
        "control_model": next(
            item.selected_model for item in control.results
            if item.series == "target"),
        "treatment_model": treatment_result.selected_model,
        "selection_scores": treatment_result.selection_scores,
        "multivariate_gate": _gate(treatment, "target"),
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (ordered[middle] if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    drivers = [row for row in rows if row["family"] == "lagged_driver"]
    controls = [row for row in rows if row["family"].endswith("control")]
    independent = [row for row in rows
                   if row["family"] == "independent_control"]
    admitted_drivers = [row for row in drivers
                        if row["treatment_model"] == "var"]
    admitted_controls = [row for row in controls
                         if row["treatment_model"] == "var"]
    completion = len(rows) == len(SEEDS) and all(row["complete"] for row in rows)
    driver_median = _median([row["relative_improvement"] for row in drivers])
    control_median = _median([row["smape_improvement"] for row in controls])
    gates = {
        "completion": completion,
        "driver_admissions": len(admitted_drivers) >= 4,
        "driver_median_relative_improvement": (
            driver_median is not None and driver_median >= 0.10),
        "admitted_driver_aggregate_positive": (
            bool(admitted_drivers) and
            sum(row["smape_improvement"] for row in admitted_drivers) > 0),
        "independent_admissions_zero": all(
            row["treatment_model"] != "var" for row in independent),
        "control_median_noninferior": (
            control_median is not None and control_median >= -2.0),
        "admitted_control_harm_bounded": all(
            row["smape_improvement"] >= -2.0 for row in admitted_controls),
    }
    return {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "denominators": {"all": len(SEEDS), "completed": len(rows),
                         "drivers": len(drivers), "controls": len(controls)},
        "metrics": {
            "driver_admissions": len(admitted_drivers),
            "control_admissions": len(admitted_controls),
            "driver_median_relative_improvement": driver_median,
            "driver_aggregate_smape_improvement": sum(
                row["smape_improvement"] for row in admitted_drivers),
            "control_median_smape_improvement": control_median,
        },
        "gates": gates,
        "decision_ready": all(gates.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    cases_dir = args.output / "cases"
    signal.signal(signal.SIGALRM, _timeout)
    for seed in SEEDS:
        case = generate_case(seed)
        row_path = cases_dir / f"{case.case_id}.json"
        if row_path.exists():
            continue
        error = None
        for attempt in range(args.retries + 1):
            try:
                signal.alarm(args.timeout)
                row = run_case(case, args.output)
                signal.alarm(0)
                row["attempt"] = attempt + 1
                _atomic_json(row_path, row)
                error = None
                break
            except Exception as exc:  # retained failure remains denominator
                signal.alarm(0)
                error = f"{type(exc).__name__}: {exc}"[:1000]
        if error is not None:
            _atomic_json(row_path, {
                "schema_version": 1, "case_id": case.case_id,
                "family": case.family, "seed": seed, "complete": False,
                "attempt": args.retries + 1, "error": error,
            })
    rows = [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(cases_dir.glob("*.json"))]
    summary = summarize(rows)
    identity = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator_version": GENERATOR_VERSION,
        "seeds": list(SEEDS),
        "training_points": TRAINING_POINTS,
        "horizon": HORIZON,
        "jobs": 1,
        "retries": args.retries,
        "timeout_seconds": args.timeout,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=False).stdout.strip(),
        "git_status": subprocess.run(
            ["git", "status", "--short"], capture_output=True, text=True,
            check=False).stdout.splitlines(),
    }
    _atomic_json(args.output / "run_identity.json", identity)
    _atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["decision_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

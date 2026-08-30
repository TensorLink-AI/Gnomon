"""Run the frozen P7 bottom-up hierarchy screen serially and resumably."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import signal
import statistics
import subprocess
import tempfile
from typing import Any

from benchmarks.breachbench.run_breachbench import (
    _grid_timestamps,
    load_corpus,
    series_frequency,
)


SEED = 20260907
HISTORY = 96
HORIZON = 24
CASES_PER_SOURCE = 8
RUNNER_VERSION = "p7-hierarchy-bottom-up-1"
LEAVES = ("leaf_a", "leaf_b", "leaf_c")


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


@dataclass(frozen=True)
class Case:
    case_id: str
    source: str
    frequency: str
    family: str
    start: int
    root_history: tuple[float, ...]
    leaf_history: tuple[tuple[float, ...], ...]
    root_future: tuple[float, ...]
    leaf_future: tuple[tuple[float, ...], ...]


def _shares(index: int, family: str) -> tuple[float, float, float]:
    if family == "stable":
        return .5, .3, .2
    first = .45 + .08 * math.sin(2 * math.pi * index / 12)
    second = .35 + .05 * math.cos(2 * math.pi * index / 7)
    return first, second, 1.0 - first - second


def _split(values: list[float], start: int,
           family: str) -> tuple[tuple[float, ...], ...]:
    leaves = [[], [], []]
    for offset, value in enumerate(values):
        shares = _shares(start + offset, family)
        # The final child absorbs floating arithmetic so the hierarchy is
        # exact in the values written to the runtime input.
        first = float(value) * shares[0]
        second = float(value) * shares[1]
        third = float(value) - math.fsum((first, second))
        for target, part in zip(leaves, (first, second, third)):
            target.append(part)
    return tuple(tuple(part) for part in leaves)


def generate_cases() -> tuple[list[Case], dict[str, Any]]:
    corpus = load_corpus()
    cases: list[Case] = []
    cutoffs: dict[str, list[int]] = {}
    for source in sorted(corpus):
        values = corpus[source]
        first = HISTORY
        last = len(values) - HORIZON
        step = (last - first) // (CASES_PER_SOURCE - 1)
        if step < HORIZON:
            raise ValueError(f"{source} cannot provide non-overlapping futures")
        source_cutoffs = [first + index * step
                          for index in range(CASES_PER_SOURCE)]
        cutoffs[source] = source_cutoffs
        for index, cutoff in enumerate(source_cutoffs):
            family = "stable" if index % 2 == 0 else "periodic"
            history = [float(value) for value in
                       values[cutoff - HISTORY:cutoff]]
            future = [float(value) for value in
                      values[cutoff:cutoff + HORIZON]]
            absolute_start = cutoff - HISTORY
            history_leaves = _split(history, absolute_start, family)
            future_leaves = _split(future, cutoff, family)
            cases.append(Case(
                case_id=f"h{SEED}-{len(cases):04d}",
                source=source,
                frequency=series_frequency(source),
                family=family,
                start=absolute_start,
                root_history=tuple(history),
                leaf_history=history_leaves,
                root_future=tuple(future),
                leaf_future=future_leaves,
            ))
    return cases, {
        "seed": SEED,
        "sources": sorted(corpus),
        "source_lengths": {key: len(value) for key, value in corpus.items()},
        "cutoffs": cutoffs,
        "history": HISTORY,
        "horizon": HORIZON,
        "cases_per_source": CASES_PER_SOURCE,
        "future_windows_non_overlapping": True,
        "families": ["stable", "periodic"],
        "partition": "three_positive_leaves_exactly_sum_to_root",
    }


def _write_inputs(run_dir: Path, case: Case) -> tuple[Path, Path]:
    stamps = _grid_timestamps(case.frequency, HISTORY)
    root = run_dir / "root.csv"
    root.write_text(
        "timestamp,root\n" + "\n".join(
            f"{stamp},{value!r}"
            for stamp, value in zip(stamps, case.root_history)) + "\n",
        encoding="utf-8")
    leaves = run_dir / "leaves.csv"
    lines = ["timestamp," + ",".join(LEAVES)]
    for row_index, stamp in enumerate(stamps):
        lines.append(stamp + "," + ",".join(
            repr(case.leaf_history[index][row_index])
            for index in range(len(LEAVES))))
    leaves.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root, leaves


def evaluate(case: Case) -> dict[str, Any]:
    from gnomon.runtime import forecast, forecast_multi

    run_dir = Path(tempfile.mkdtemp(prefix="hierarchybench-"))
    try:
        root_path, leaf_path = _write_inputs(run_dir, case)
        root_artifact, _ = forecast(
            str(root_path), time_column="timestamp", target_column="root",
            horizon=HORIZON, frequency=case.frequency,
            output=str(run_dir / "root-out"))
        leaf_artifact, _ = forecast_multi(
            str(leaf_path), time_column="timestamp",
            target_columns=list(LEAVES), horizon=HORIZON,
            frequency=case.frequency, output=str(run_dir / "leaf-out"),
            max_workers=1)
        root_rows = root_artifact.results[0].forecast or []
        leaf_rows = {result.series: result.forecast or []
                     for result in leaf_artifact.results}
        complete = len(root_rows) == HORIZON and all(
            len(leaf_rows.get(name, [])) == HORIZON for name in LEAVES)
        control = [float(row["q50"]) for row in root_rows]
        leaf_points = {
            name: [float(row["q50"]) for row in leaf_rows.get(name, [])]
            for name in LEAVES
        }
        candidate = [math.fsum(leaf_points[name][step] for name in LEAVES)
                     for step in range(HORIZON)] if complete else []
        actual = list(case.root_future)
        control_error = statistics.mean(
            abs(predicted - observed)
            for predicted, observed in zip(control, actual)) if complete else None
        candidate_error = statistics.mean(
            abs(predicted - observed)
            for predicted, observed in zip(candidate, actual)) if complete else None
        return {
            "schema_version": 1,
            "runner_version": RUNNER_VERSION,
            "case_id": case.case_id,
            "complete": complete,
            "source": case.source,
            "frequency": case.frequency,
            "family": case.family,
            "history": HISTORY,
            "horizon": HORIZON,
            "control": {
                "q50": control,
                "mae": control_error,
                "selected_model": root_artifact.results[0].selected_model,
                "support": root_artifact.results[0].support,
            },
            "leaves": {
                name: {
                    "q50": leaf_points[name],
                    "q50_sha256": _hash(leaf_points[name]),
                    "selected_model": next(
                        result.selected_model for result in leaf_artifact.results
                        if result.series == name),
                    "support": next(
                        result.support for result in leaf_artifact.results
                        if result.series == name),
                } for name in LEAVES
            },
            "candidate": {
                "q50": candidate,
                "mae": candidate_error,
                "coherent": complete and all(
                    candidate[step] == math.fsum(
                        leaf_points[name][step] for name in LEAVES)
                    for step in range(HORIZON)),
                "finite": complete and all(math.isfinite(value)
                                           for value in candidate),
                "nonnegative": complete and all(value >= 0 for value in candidate),
                "derivation": "bottom_up_sum_q50_v1",
                "uncertainty_status": "unavailable_joint_distribution",
                "automation_eligible": False,
            },
            "truth": {"root": actual, "root_sha256": _hash(actual)},
            "runtime_input_sha256": _hash({
                "root_history": case.root_history,
                "leaf_history": case.leaf_history,
                "frequency": case.frequency,
                "horizon": HORIZON,
            }),
        }
    finally:
        shutil.rmtree(run_dir)


def _bootstrap_interval(deltas: list[float]) -> dict[str, float] | None:
    if not deltas:
        return None
    rng = random.Random(SEED + 1)
    means = [statistics.mean(rng.choice(deltas) for _ in deltas)
             for _ in range(5000)]
    means.sort()
    return {"lower": means[124], "upper": means[4874]}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row.get("complete")]
    strata: dict[str, dict[str, Any]] = {}
    for source in sorted({row.get("source") for row in rows}):
        for family in ("stable", "periodic"):
            selected = [row for row in complete
                        if row["source"] == source and row["family"] == family]
            control = sum(float(row["control"]["mae"]) for row in selected)
            candidate = sum(float(row["candidate"]["mae"]) for row in selected)
            strata[f"{source}:{family}"] = {
                "cases": len(selected),
                "control_absolute_error": control,
                "candidate_absolute_error": candidate,
                "candidate_to_control_ratio": (
                    candidate / control if control else None),
                "noninferior": len(selected) == CASES_PER_SOURCE // 2
                               and control > 0 and candidate <= 1.02 * control,
            }
    control = sum(float(row["control"]["mae"]) for row in complete)
    candidate = sum(float(row["candidate"]["mae"]) for row in complete)
    deltas = [float(row["control"]["mae"])
              - float(row["candidate"]["mae"]) for row in complete]
    improvement = (control - candidate) / control if control else None
    gates = {
        "completion": len(complete) == len(rows) == 32,
        "coherence": len(complete) == 32 and all(
            row["candidate"]["coherent"] for row in complete),
        "finite_nonnegative": len(complete) == 32 and all(
            row["candidate"]["finite"] and row["candidate"]["nonnegative"]
            for row in complete),
        "uncertainty_truth": len(complete) == 32 and all(
            row["candidate"]["uncertainty_status"]
            == "unavailable_joint_distribution"
            and row["candidate"]["automation_eligible"] is False
            for row in complete),
        "stratum_noninferiority": all(
            value["noninferior"] for value in strata.values()),
        "aggregate_skill": improvement is not None and improvement >= .05,
    }
    return {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "denominators": {"cases": 32, "complete": len(complete),
                         "strata": len(strata)},
        "metrics": {
            "control_absolute_error": control,
            "candidate_absolute_error": candidate,
            "relative_improvement": improvement,
            "mean_paired_mae_delta": statistics.mean(deltas) if deltas else None,
            "paired_delta_bootstrap_95": _bootstrap_interval(deltas),
        },
        "strata": strata,
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
    case_dir = args.output / "cases"
    cases, provenance = generate_cases()
    signal.signal(signal.SIGALRM, _timeout)
    for case in cases:
        target = case_dir / f"{case.case_id}.json"
        if target.exists():
            continue
        error = None
        for attempt in range(args.retries + 1):
            try:
                signal.alarm(args.timeout)
                row = evaluate(case)
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
    summary = summarize(rows)
    identity = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runner_version": RUNNER_VERSION,
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
    return 0 if summary["decision_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

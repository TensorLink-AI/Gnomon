"""Run the frozen v0.7 Q1 claim/support coherence reproduction.

The benchmark uses production forecast rendering. Generator labels remain in
the harness and are never supplied to Gnomon's runtime.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random
import subprocess
import tempfile
from typing import Any

from benchmarks.common.checkpoint import prepare_run_identity
from gnomon.ids import FixedClock
from gnomon.runtime import forecast
from gnomon.toolspec import brief_summary, forecast_summary


CLOCK = FixedClock(datetime(2026, 8, 30, tzinfo=timezone.utc))


def _revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short=12", "HEAD"], text=True,
    ).strip()


def _cases() -> list[dict[str, Any]]:
    level = random.Random(123)
    short = random.Random(321)
    threshold = random.Random(456)
    return [
        {
            "case_id": "stationary_level_seed_123",
            "values": [50.0 + level.gauss(0.0, 5.0) for _ in range(120)],
            "horizon": 7,
            "threshold": None,
            "role": "baseline_retention",
        },
        {
            "case_id": "linear_trend",
            "values": [20.0 + 0.8 * index for index in range(120)],
            "horizon": 7,
            "threshold": None,
            "role": "positive_uplift_control",
        },
        {
            "case_id": "short_noisy",
            "values": [50.0 + short.gauss(0.0, 5.0) for _ in range(18)],
            "horizon": 3,
            "threshold": None,
            "role": "degraded_support_control",
        },
        {
            "case_id": "long_horizon_threshold",
            "values": [
                40.0 + 0.2 * index + threshold.gauss(0.0, 1.0)
                for index in range(30)
            ],
            "horizon": 12,
            "threshold": 50.0,
            "role": "threshold_availability_control",
        },
    ]


def _write_series(path: Path, values: list[float]) -> None:
    start = datetime(2025, 1, 1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "value"])
        for index, value in enumerate(values):
            writer.writerow([(start + timedelta(days=index)).isoformat(), value])


def _first_nonempty(path: Path) -> str:
    return next(
        (line.strip() for line in path.read_text(encoding="utf-8").splitlines()
         if line.strip()),
        "",
    )


def _run_case(case: dict[str, Any], work: Path) -> dict[str, Any]:
    source = work / f"{case['case_id']}.csv"
    _write_series(source, case["values"])
    artifact, artifact_dir = forecast(
        str(source), time_column="timestamp", target_column="value",
        frequency="D", horizon=case["horizon"],
        threshold=case["threshold"], output=str(work / "artifacts"),
        clock=CLOCK,
    )
    full = forecast_summary(artifact, artifact_dir)
    brief = brief_summary(artifact, artifact_dir)
    persisted = artifact.to_dict()
    result = persisted["results"][0]
    assessment = result.get("support_assessment") or {}
    sensitivity = assessment.get("sensitivity") or {}
    rows = result.get("forecast") or []
    headlines = {
        "summary_markdown": _first_nonempty(artifact_dir / "summary.md"),
        "full_tool": full["headline"],
        "brief_tool": brief["headline"],
    }
    return {
        "case_id": case["case_id"],
        "role": case["role"],
        "completed": True,
        "selected_model": result.get("selected_model"),
        "strongest_baseline": result.get("strongest_baseline"),
        "baseline_improvement": sensitivity.get("baseline_improvement"),
        "support": result.get("support"),
        "support_status": assessment.get("status"),
        "row_tiers": sorted({row.get("tier") for row in rows}),
        "interval_coverage": result.get("interval_coverage"),
        "threshold_artifact": result.get("threshold"),
        "threshold_full": full["results"][0].get("threshold"),
        "threshold_brief": brief["results"][0].get("threshold"),
        "headlines": headlines,
        "headline_flags": {
            "high_confidence": "High-confidence" in full["headline"],
            "claims_beat": " beat " in full["headline"],
            "baseline_retained": "baseline" in full["headline"].lower(),
            "no_measured_uplift": "no measured uplift" in full["headline"].lower(),
            "caveats": "with caveats" in full["headline"].lower(),
            "naive_extrapolation": "naive extrapolation" in full["headline"].lower(),
            "probability": "probab" in full["headline"].lower(),
        },
        "reasons": assessment.get("reasons") or [],
        "recovery_actions": assessment.get("recovery_actions") or [],
        "warnings": result.get("warnings") or [],
    }


def _summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(row for row in records if row["role"] == "baseline_retention")
    positive = next(
        row for row in records if row["role"] == "positive_uplift_control")
    weak = [row for row in records if row["support_status"] != "supported"]
    threshold = next(
        row for row in records
        if row["role"] == "threshold_availability_control")
    headline_agreement = all(
        len(set(row["headlines"].values())) == 1 for row in records)
    baseline_is_zero = (
        baseline["selected_model"] == baseline["strongest_baseline"]
        and abs(float(baseline["baseline_improvement"] or 0.0)) <= 1e-12
    )
    positive_is_measured = (
        positive["selected_model"] != positive["strongest_baseline"]
        and float(positive["baseline_improvement"] or 0.0) > 0.0
        and positive["headline_flags"]["claims_beat"]
    )
    threshold_consistent = (
        threshold["threshold_artifact"] == threshold["threshold_full"]
        == threshold["threshold_brief"]
    )
    threshold_payload = threshold["threshold_artifact"] or {}
    threshold_event = threshold_payload.get("horizon_event") or {}
    threshold_probability_unavailable = (
        threshold["support_status"] != "supported"
        and str(threshold_payload.get("probability_status", "")).startswith(
            "unavailable")
        and not threshold_payload.get("probability_above")
        and not threshold_event
        and isinstance(threshold_payload.get("bounded_assessment"), dict)
        and threshold_payload["bounded_assessment"].get(
            "automation_eligible") is False
    )
    gates = {
        "all_cases_complete": len(records) == 4
            and all(row["completed"] for row in records),
        "all_headline_surfaces_agree": headline_agreement,
        "zero_gain_baseline_case_present": baseline_is_zero,
        "zero_gain_baseline_does_not_claim_beat": baseline_is_zero
            and not baseline["headline_flags"]["claims_beat"],
        "zero_gain_baseline_not_high_confidence": baseline_is_zero
            and not baseline["headline_flags"]["high_confidence"],
        "positive_uplift_control_is_measured": positive_is_measured,
        "subsupported_never_high_confidence": all(
            not row["headline_flags"]["high_confidence"] for row in weak),
        "threshold_payload_consistent": threshold_consistent,
        "subsupported_threshold_probability_typed_unavailable":
            threshold_probability_unavailable,
    }
    return {
        "schema_version": "0.1",
        "benchmark": "forecast-claim-support-coherence",
        "evaluated_commit": _revision(),
        "scope": "full",
        "cases": len(records),
        "gates": gates,
        "passed": all(gates.values()),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "cases.jsonl"
    cases = _cases()
    prepare_run_identity(
        args.output_dir,
        {
            "schema_version": 1,
            "benchmark": "forecast-claim-support-coherence",
            "code_revision": _revision(),
            "case_ids": [case["case_id"] for case in cases],
            "protocol": "frozen-v0.7-q1",
        },
        resume=args.resume,
        state_paths=[checkpoint, args.output_dir / "summary.json"],
    )
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and checkpoint.is_file():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["case_id"]] = row
    with tempfile.TemporaryDirectory(prefix="gnomon-claimbench-", dir="/tmp") as raw:
        work = Path(raw)
        for case in cases:
            if case["case_id"] in completed:
                continue
            row = _run_case(case, work)
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            completed[row["case_id"]] = row
    records = [completed[case["case_id"]] for case in cases]
    summary = _summarise(records)
    destination = args.output_dir / "summary.json"
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Frozen S2 reproduction for bounded timestamp alignment.

The fixtures are independently regenerated from the external report. The
runner exercises public loading behavior and retains raw timestamps, values,
repair disclosures, and typed failures for inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.manifest import code_revision, write_manifest
from gnomon.contracts import GnomonError
from gnomon.pipeline import load_stage
from gnomon.repair import RepairLog


COUNT = 36
START = datetime(2026, 1, 1, 0, 7)
STEP = timedelta(minutes=20)
TOLERANCE_SECONDS = 12.0


def _nominal(count: int = COUNT) -> list[datetime]:
    return [START + index * STEP for index in range(count)]


def _jittered(offsets: list[float], count: int = COUNT) -> list[datetime]:
    return [stamp + timedelta(seconds=offsets[index % len(offsets)])
            for index, stamp in enumerate(_nominal(count))]


def _cases() -> dict[str, list[tuple[datetime, float]]]:
    nominal = _nominal()
    cases: dict[str, list[tuple[datetime, float]]] = {
        "exact_phase": list(zip(nominal, map(float, range(COUNT)))),
        "cron_jitter": list(zip(
            _jittered([-1.0, 1.0, 0.0]), map(float, range(COUNT)))),
        "scrape_jitter": list(zip(
            _jittered([-10.0, 4.0, 12.0, -3.0, 7.0, 0.0]),
            map(float, range(COUNT)))),
        "boundary_inside": list(zip(
            [stamp + (timedelta(seconds=12) if index == 17 else timedelta())
             for index, stamp in enumerate(nominal)],
            map(float, range(COUNT)))),
        "boundary_outside": list(zip(
            [stamp + (timedelta(seconds=12, microseconds=1000)
                      if index == 17 else timedelta())
             for index, stamp in enumerate(nominal)],
            map(float, range(COUNT)))),
    }
    collision = list(cases["exact_phase"])
    collision.insert(11, (nominal[10] + timedelta(seconds=5), 999.0))
    cases["collision"] = collision
    reordered = list(zip(_jittered([-1.0, 1.0, 0.0]), map(float, range(COUNT))))
    reordered[8], reordered[9] = reordered[9], reordered[8]
    cases["reordered"] = reordered
    mixed: list[tuple[datetime, float]] = []
    stamp = START
    for index in range(COUNT):
        mixed.append((stamp, float(index)))
        stamp += timedelta(minutes=20 if index < COUNT // 2 else 10)
    cases["mixed_cadence"] = mixed
    small_gap = list(zip(_jittered([-1.0, 1.0, 0.0]), map(float, range(COUNT))))
    del small_gap[14]
    cases["small_gap"] = small_gap
    long_outage = list(zip(_jittered([-1.0, 1.0, 0.0], 44), map(float, range(44))))
    del long_outage[16:24]
    cases["long_outage"] = long_outage
    return cases


def _write(path: Path, rows: list[tuple[datetime, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "value"])
        writer.writerows((stamp.isoformat(), value) for stamp, value in rows)


def _attempt(
    root: Path,
    case_id: str,
    rows: list[tuple[datetime, float]],
    repair: str,
    *,
    frequency: str | None = None,
) -> dict[str, Any]:
    path = root / f"{case_id}.csv"
    _write(path, rows)
    log = RepairLog()
    try:
        loaded = load_stage(
            str(path), time_column="timestamp", target_column="value",
            series_column=None, frequency=frequency, repair=repair,
            repair_log=log,
        )
        result = loaded.groups["__default__"]
        return {
            "case_id": case_id,
            "repair": repair,
            "termination": "accepted",
            "error": None,
            "frequency": loaded.frequency,
            "timestamps": [item.timestamp.isoformat() for item in result],
            "values": [item.value for item in result],
            "actions": log.summary()["actions"],
        }
    except GnomonError as exc:
        return {
            "case_id": case_id,
            "repair": repair,
            "termination": "typed_rejection",
            "error": exc.to_dict()["error"],
            "frequency": None,
            "timestamps": [],
            "values": [],
            "actions": log.summary()["actions"],
        }
    except Exception as exc:  # retained as a product failure
        return {
            "case_id": case_id,
            "repair": repair,
            "termination": "internal_error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "frequency": None,
            "timestamps": [],
            "values": [],
            "actions": log.summary()["actions"],
        }


def _action(row: dict[str, Any], code: str) -> dict[str, Any] | None:
    return next((item for item in row["actions"] if item["code"] == code), None)


def run() -> dict[str, Any]:
    cases = _cases()
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="gnomon-jitterbench-") as temporary:
        root = Path(temporary)
        for level in ("off", "safe", "aggressive"):
            records.append(_attempt(root, "exact_phase", cases["exact_phase"], level))
        for case_id in ("cron_jitter", "scrape_jitter", "boundary_inside",
                        "boundary_outside", "reordered", "mixed_cadence",
                        "small_gap", "long_outage"):
            for level in ("safe", "aggressive"):
                records.append(_attempt(root, case_id, cases[case_id], level))
        records.append(_attempt(root, "cron_jitter", cases["cron_jitter"], "off"))
        records.append(_attempt(
            root, "collision", cases["collision"], "safe", frequency="20min"))
        records.append(_attempt(
            root, "collision", cases["collision"], "aggressive", frequency="20min"))

    by_key = {(row["case_id"], row["repair"]): row for row in records}
    exact = [by_key[("exact_phase", level)]
             for level in ("off", "safe", "aggressive")]
    cron_safe = by_key[("cron_jitter", "safe")]
    cron_aggressive = by_key[("cron_jitter", "aggressive")]
    scrape_safe = by_key[("scrape_jitter", "safe")]
    inside_safe = by_key[("boundary_inside", "safe")]
    outside_safe = by_key[("boundary_outside", "safe")]
    reordered_safe = by_key[("reordered", "safe")]
    small_safe = by_key[("small_gap", "safe")]
    small_aggressive = by_key[("small_gap", "aggressive")]
    collision_rows = [by_key[("collision", level)]
                      for level in ("safe", "aggressive")]
    snap = _action(cron_safe, "timestamp_jitter_aligned")
    snap_metrics = (snap or {}).get("metrics") or {}
    gates = {
        "exact_grid_untouched_all_levels": all(
            row["termination"] == "accepted"
            and not row["actions"]
            and row["timestamps"] == [stamp.isoformat() for stamp, _ in cases["exact_phase"]]
            and row["values"] == [value for _, value in cases["exact_phase"]]
            for row in exact),
        "off_remains_strict": (
            by_key[("cron_jitter", "off")]["termination"] == "typed_rejection"),
        "bounded_jitter_safe_and_aggressive": all(
            by_key[(case_id, level)]["termination"] == "accepted"
            for case_id in ("cron_jitter", "scrape_jitter", "boundary_inside")
            for level in ("safe", "aggressive")),
        "bounded_alignment_preserves_values_and_count": all(
            row["values"] == [value for _, value in cases[row["case_id"]]]
            and len(row["timestamps"]) == len(cases[row["case_id"]])
            for row in (cron_safe, cron_aggressive, scrape_safe, inside_safe)),
        "phase_and_tolerance_are_visible": (
            snap is not None
            and snap_metrics.get("cadence") == "20min"
            and snap_metrics.get("grid_phase") is not None
            and snap_metrics.get("tolerance_seconds") == TOLERANCE_SECONDS
            and snap_metrics.get("maximum_displacement_seconds") is not None),
        "snaps_do_not_consume_invention_ceiling": (
            cron_aggressive["termination"] == "accepted"
            and _action(cron_aggressive, "timestamp_jitter_aligned") is not None),
        "outside_boundary_refused": all(
            by_key[("boundary_outside", level)]["termination"] == "typed_rejection"
            for level in ("safe", "aggressive")),
        "collisions_refused_without_merge": all(
            row["termination"] == "typed_rejection"
            and (row["error"] or {}).get("code") == "TIMESTAMP_ALIGNMENT_CONFLICT"
            for row in collision_rows),
        "reordering_disclosed_separately": (
            reordered_safe["termination"] == "accepted"
            and _action(reordered_safe, "timestamps_reordered") is not None
            and _action(reordered_safe, "timestamp_jitter_aligned") is not None),
        "mixed_cadence_refused_typed": all(
            by_key[("mixed_cadence", level)]["termination"] == "typed_rejection"
            for level in ("safe", "aggressive")),
        "small_gap_not_hidden": (
            small_safe["termination"] == "typed_rejection"
            and (small_safe["error"] or {}).get("code") == "IRREGULAR_TIME_GRID"
            and small_aggressive["termination"] == "accepted"
            and _action(small_aggressive, "timestamp_jitter_aligned") is not None
            and _action(small_aggressive, "gap_filled") is not None),
        "long_outage_refused_typed": all(
            by_key[("long_outage", level)]["termination"] == "typed_rejection"
            for level in ("safe", "aggressive")),
        "no_internal_errors": all(
            row["termination"] != "internal_error" for row in records),
    }
    return {
        "schema_version": "0.1",
        "benchmark": "bounded-timestamp-jitter",
        "fixture_provenance": "independently_regenerated",
        "frozen_policy": {
            "tolerance_fraction": 0.01,
            "maximum_tolerance_seconds": 60.0,
            "twenty_minute_tolerance_seconds": TOLERANCE_SECONDS,
        },
        "cases": len(records),
        "accepted": sum(row["termination"] == "accepted" for row in records),
        "typed_rejections": sum(
            row["termination"] == "typed_rejection" for row in records),
        "internal_errors": sum(
            row["termination"] == "internal_error" for row in records),
        "gates": gates,
        "raw_records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run()
    result["evaluated_commit"] = code_revision()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_manifest(
            args.output_dir, benchmark="jitterbench",
            condition="current-production-boundary",
            target="frozen-s2-cases",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(result["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

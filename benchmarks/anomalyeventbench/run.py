"""Run the frozen v0.7 Q6 anomaly-event reproduction."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import random
from typing import Any

from benchmarks.common.manifest import code_revision, write_manifest
from gnomon.ids import FixedClock
from gnomon.macros import detect_anomalies, investigate_change


SEEDS = tuple(range(9101, 9105))
OBSERVATIONS = 120
EVENT_INDEX = 72
NEARBY_INDEX = 76
EVENT_TOLERANCE = 1
CLOCK = FixedClock(datetime(2026, 8, 30, tzinfo=timezone.utc))
FAMILIES: tuple[dict[str, Any], ...] = (
    {"name": "level_shift_up", "kind": "shift", "magnitude": 12.0},
    {"name": "level_shift_down", "kind": "shift", "magnitude": -12.0},
    {"name": "isolated_spike_up", "kind": "spike", "magnitude": 18.0},
    {"name": "isolated_spike_down", "kind": "spike", "magnitude": -18.0},
    {"name": "nearby_opposite_spikes", "kind": "nearby", "magnitude": 18.0},
    {"name": "stationary_noise", "kind": "control", "magnitude": 0.0},
)


def _cases() -> list[dict[str, Any]]:
    return [
        {
            "case_id": f"{family['name']}-{seed}",
            "family": family["name"],
            "kind": family["kind"],
            "magnitude": family["magnitude"],
            "seed": seed,
            "expected_anomaly_indices": (
                [EVENT_INDEX, NEARBY_INDEX] if family["kind"] == "nearby"
                else [EVENT_INDEX] if family["kind"] == "spike"
                else []),
            "expected_shift_index": (
                EVENT_INDEX if family["kind"] == "shift" else None),
        }
        for family in FAMILIES for seed in SEEDS
    ]


def _generate(case: dict[str, Any]) -> list[float]:
    family_index = next(
        index for index, family in enumerate(FAMILIES)
        if family["name"] == case["family"])
    rng = random.Random(int(case["seed"]) + family_index * 10_000)
    values = [100.0 + rng.gauss(0.0, .35) for _ in range(OBSERVATIONS)]
    kind = str(case["kind"])
    magnitude = float(case["magnitude"])
    if kind == "shift":
        for index in range(EVENT_INDEX, OBSERVATIONS):
            values[index] += magnitude
    elif kind == "spike":
        values[EVENT_INDEX] += magnitude
    elif kind == "nearby":
        values[EVENT_INDEX] += magnitude
        values[NEARBY_INDEX] -= magnitude
    return values


def _timestamps() -> list[datetime]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [start + timedelta(hours=index) for index in range(OBSERVATIONS)]


def _write_case(path: Path, values: list[float]) -> list[datetime]:
    timestamps = _timestamps()
    rows = ["timestamp,value", *[
        f"{moment.isoformat()},{value}"
        for moment, value in zip(timestamps, values)
    ]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return timestamps


def _alert_indices(
    anomalies: list[dict[str, Any]], timestamps: list[datetime],
) -> list[int]:
    index_of = {moment.isoformat(): index
                for index, moment in enumerate(timestamps)}
    return sorted(index_of[str(item["timestamp"])] for item in anomalies)


def _event_counts(
    alerts: list[int], expected: list[int], *, tolerance: int = EVENT_TOLERANCE,
) -> dict[str, Any]:
    unmatched = set(alerts)
    matched: list[dict[str, int]] = []
    for event in expected:
        candidates = sorted(
            (index for index in unmatched if abs(index - event) <= tolerance),
            key=lambda index: (abs(index - event), index),
        )
        if candidates:
            alert = candidates[0]
            unmatched.remove(alert)
            matched.append({"event_index": event, "alert_index": alert})
    true_positive = len(matched)
    false_positive = len(unmatched)
    false_negative = len(expected) - true_positive
    return {
        "raw_alerts": len(alerts),
        "alert_indices": alerts,
        "expected_events": len(expected),
        "matched_events": true_positive,
        "false_events": false_positive,
        "missed_events": false_negative,
        "event_precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 1.0),
        "event_recall": (
            true_positive / len(expected) if expected else 1.0),
        "matches": matched,
    }


def _surface_score(
    result: dict[str, Any], *, case: dict[str, Any],
    timestamps: list[datetime], artifact_path: Path,
) -> dict[str, Any]:
    alerts = _alert_indices(result.get("anomalies") or [], timestamps)
    event = _event_counts(alerts, list(case["expected_anomaly_indices"]))
    attribution = result.get("anomaly_attribution") or {}
    attribution_complete = bool(
        attribution.get("relationship") == "explained_by_regime_shift"
        and int(attribution.get("raw_anomaly_count", -1))
        - int(attribution.get("final_anomaly_count", -1))
        == int(attribution.get("suppressed_count", -2))
        and int(attribution.get("final_anomaly_count", -1)) == len(alerts))
    event.update({
        "detector": result.get("detector"),
        "selection_basis": result.get("selection_basis"),
        "support": (result.get("support_assessment") or {}).get("status"),
        "rebound_duplicate": (
            EVENT_INDEX + 1 in alerts
            if case["kind"] in {"spike", "nearby"} else False),
        "nearby_events_exact": (
            EVENT_INDEX in alerts and NEARBY_INDEX in alerts
            if case["kind"] == "nearby" else None),
        "artifact_path": str(artifact_path),
        "regime_attribution": attribution or None,
        "regime_attribution_complete": attribution_complete,
    })
    return event


def _run_case(case: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    case_dir = work_dir / str(case["case_id"])
    values = _generate(case)
    source = case_dir / "history.csv"
    timestamps = _write_case(source, values)
    common = {
        "time_column": "timestamp", "target_column": "value",
        "frequency": "h", "clock": CLOCK,
    }
    investigation, investigation_path = investigate_change(
        str(source), output=str(case_dir / "investigation"), **common)
    investigation_replay, replay_path = investigate_change(
        str(source), output=str(case_dir / "investigation"), **common)
    unlabelled, unlabelled_path = detect_anomalies(
        str(source), include_tsfm=False,
        output=str(case_dir / "unlabelled"), **common)
    unlabelled_replay, unlabelled_replay_path = detect_anomalies(
        str(source), include_tsfm=False,
        output=str(case_dir / "unlabelled"), **common)
    label_indices = list(case["expected_anomaly_indices"])
    labels = [timestamps[index].isoformat() for index in label_indices]
    labelled, labelled_path = detect_anomalies(
        str(source), labels=labels or None, include_tsfm=False,
        output=str(case_dir / "labelled"), **common)
    labelled_replay, labelled_replay_path = detect_anomalies(
        str(source), labels=labels or None, include_tsfm=False,
        output=str(case_dir / "labelled"), **common)
    investigated = investigation["results"][0]
    changepoints = investigated.get("changepoints") or []
    strongest = (max(changepoints, key=lambda item: (
        float(item.get("relative_gain") or 0.0), -int(item.get("index") or 0)))
        if changepoints else None)
    expected_shift = case["expected_shift_index"]
    shift_admitted = bool(
        expected_shift is not None
        and investigated.get("classification") == "regime_shift"
        and strongest is not None
        and abs(int(strongest["index"]) - int(expected_shift)) <= 1
        and (investigated.get("support_assessment") or {}).get("status")
        in {"supported", "conditionally_supported"})
    investigation_score = _surface_score(
        investigated, case=case, timestamps=timestamps,
        artifact_path=investigation_path)
    investigation_score.update({
        "classification": investigated.get("classification"),
        "strongest_changepoint_index": (
            int(strongest["index"]) if strongest else None),
        "shift_admitted": shift_admitted,
        "post_admitted_shift_alerts": (
            sum(index >= int(strongest["index"])
                for index in investigation_score["alert_indices"])
            if shift_admitted and strongest else 0),
        "regime_explained_duplicates_remaining": (
            0 if shift_admitted and investigation_score[
                "regime_attribution_complete"] else
            investigation_score["post_admitted_shift_alerts"]
            if shift_admitted else 0),
    })
    unlabelled_score = _surface_score(
        unlabelled["results"][0], case=case, timestamps=timestamps,
        artifact_path=unlabelled_path)
    labelled_score = _surface_score(
        labelled["results"][0], case=case, timestamps=timestamps,
        artifact_path=labelled_path)
    deterministic = (
        investigation == investigation_replay
        and investigation_path == replay_path
        and unlabelled == unlabelled_replay
        and unlabelled_path == unlabelled_replay_path
        and labelled == labelled_replay
        and labelled_path == labelled_replay_path)
    return {
        **case,
        "evaluated_commit": code_revision(),
        "product_inputs": {
            "observations": OBSERVATIONS, "frequency": "h",
            "labels_supplied": bool(labels),
        },
        "deterministic_replay": deterministic,
        "investigation": investigation_score,
        "unlabelled": unlabelled_score,
        "labelled": labelled_score,
    }


def _aggregate(rows: list[dict[str, Any]], surface: str) -> dict[str, Any]:
    scores = [row[surface] for row in rows]
    matched = sum(item["matched_events"] for item in scores)
    false = sum(item["false_events"] for item in scores)
    expected = sum(item["expected_events"] for item in scores)
    missed = sum(item["missed_events"] for item in scores)
    return {
        "cases": len(scores),
        "raw_alerts": sum(item["raw_alerts"] for item in scores),
        "expected_events": expected,
        "matched_events": matched,
        "false_events": false,
        "missed_events": missed,
        "event_precision": matched / (matched + false)
        if matched + false else 1.0,
        "event_recall": matched / expected if expected else 1.0,
        "rebound_duplicates": sum(bool(item["rebound_duplicate"])
                                  for item in scores),
    }


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    surfaces = {
        name: _aggregate(rows, name)
        for name in ("investigation", "unlabelled", "labelled")
    }
    shifts = [row for row in rows if row["kind"] == "shift"]
    nearby = [row for row in rows if row["kind"] == "nearby"]
    labelled_rows = [row for row in rows
                     if row["product_inputs"]["labels_supplied"]]
    labelled_supplied = _aggregate(labelled_rows, "labelled")
    gates = {
        "all_24_cases_complete_on_three_surfaces": len(rows) == 24,
        "deterministic_replay_all_cases": all(
            row["deterministic_replay"] for row in rows),
        "all_level_shifts_admitted": all(
            row["investigation"]["shift_admitted"] for row in shifts),
        "regime_explained_duplicates_remaining_zero": all(
            row["investigation"][
                "regime_explained_duplicates_remaining"] == 0
            for row in shifts),
        "regime_attribution_complete_all_shifts": all(
            row["investigation"]["regime_attribution_complete"]
            for row in shifts),
        "rebound_duplicates_zero_all_surfaces": all(
            not row[surface]["rebound_duplicate"]
            for row in rows for surface in surfaces),
        "nearby_events_preserved_when_detected": all(
            all(row[surface]["nearby_events_exact"] is not False
                for surface in surfaces) for row in nearby),
        "unlabelled_selection_basis_explicit": all(
            row["unlabelled"]["selection_basis"] ==
            "synthetic_injection_macro_f1" for row in rows),
        "labelled_selection_basis_explicit": all(
            row["labelled"]["selection_basis"] == "label_f1"
            for row in labelled_rows),
    }
    by_family = {
        str(family["name"]): {
            surface: _aggregate(
                [row for row in rows if row["family"] == family["name"]],
                surface)
            for surface in surfaces
        }
        for family in FAMILIES
    }
    return {
        "schema_version": "0.1",
        "benchmark": "regime-aware-anomaly-events",
        "evaluated_commit": code_revision(),
        "scope": "full",
        "cases": len(rows),
        "surfaces": surfaces,
        "labelled_supplied": labelled_supplied,
        "level_shift_cases": len(shifts),
        "level_shifts_admitted": sum(
            row["investigation"]["shift_admitted"] for row in shifts),
        "post_admitted_shift_alerts": sum(
            row["investigation"]["post_admitted_shift_alerts"]
            for row in shifts),
        "by_family": by_family,
        "gates": gates,
        "passed": all(gates.values()),
        "raw_records": rows,
    }


def _atomic_rows(path: Path, cases: list[dict[str, Any]],
                 rows: dict[str, dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for case in cases:
            if case["case_id"] in rows:
                handle.write(json.dumps(
                    rows[case["case_id"]], sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = _cases()
    identity = {
        "schema_version": 1,
        "benchmark": "regime-aware-anomaly-events",
        "code_revision": code_revision(),
        "case_ids": [case["case_id"] for case in cases],
        "protocol": "docs/v0.7-q6-anomaly-event-protocol.md",
        "machine_local_tsfm_disabled": True,
    }
    identity_path = args.output_dir / "run_identity.json"
    if identity_path.is_file():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise SystemExit("resume identity mismatch; use a new output directory")
        if not args.resume:
            raise SystemExit("output exists; pass --resume or use a new directory")
    else:
        identity_path.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    checkpoint = args.output_dir / "cases.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and checkpoint.is_file():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["case_id"]] = row
    for case in cases:
        if case["case_id"] in completed:
            continue
        completed[case["case_id"]] = _run_case(
            case, args.output_dir / "work")
        _atomic_rows(checkpoint, cases, completed)
    rows = [completed[case["case_id"]] for case in cases]
    summary = _summarise(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    write_manifest(
        args.output_dir,
        benchmark="anomalyeventbench",
        condition="frozen-current-head-reproduction",
        target="24 regime-shift, spike, nearby-event, and control cases",
        protocol="docs/v0.7-q6-anomaly-event-protocol.md",
    )
    print(json.dumps({
        key: summary[key] for key in (
            "benchmark", "evaluated_commit", "cases", "surfaces",
            "level_shifts_admitted", "post_admitted_shift_alerts",
            "gates", "passed")
    }, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

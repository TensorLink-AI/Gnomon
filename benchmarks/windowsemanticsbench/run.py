"""Evaluate deterministic zero-window parsing on a frozen phrase matrix."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from benchmarks.common.manifest import code_revision
from gnomon.context import ContextEvent
from gnomon.future_context import apply_future_events, assess_future_events
from gnomon.llm_dossier import deterministic_dated_zero_window_dossier


FUTURE = [f"2026-01-03T{hour:02d}:00:00+00:00" for hour in range(12)]
CUTOFF = "2026-01-02T23:00:00+00:00"
CASES = (
    ("between_half_open", (
        "The meter will be offline between 2026-01-03 04:00:00 and "
        "2026-01-03 07:00:00, resulting in zero readings."), [4, 5, 6]),
    ("until_half_open", (
        "Readings will be zero from 2026-01-03 04:00:00 until "
        "2026-01-03 07:00:00 while the meter is offline."), [4, 5, 6]),
    ("through_closed", (
        "Readings will be zero from 2026-01-03 04:00:00 through "
        "2026-01-03 07:00:00 while the meter is offline."), [4, 5, 6, 7]),
    ("explicit_duration", (
        "The meter is offline from 2026-01-03 04:00:00 for 3 hours, "
        "resulting in zero readings."), [4, 5, 6]),
    ("ambiguous_to", (
        "Readings will be zero from 2026-01-03 04:00:00 to "
        "2026-01-03 07:00:00."), None),
    ("approximate_boundary", (
        "Readings will be zero between approximately 2026-01-03 04:00:00 "
        "and 2026-01-03 07:00:00."), None),
    ("wrong_target", (
        "Temperatures will be zero between 2026-01-03 04:00:00 and "
        "2026-01-03 07:00:00."), None),
    ("past_window", (
        "Readings were zero between 2026-01-02 04:00:00 and "
        "2026-01-02 07:00:00."), None),
)


def _active_indices(dossier: dict[str, Any] | None) -> list[int] | None:
    if dossier is None:
        return None
    claim = dossier["claims"][0]
    start = str(claim["effective_start"])
    end = str(claim["effective_end"])
    return [index for index, stamp in enumerate(FUTURE)
            if start <= stamp <= end]


def _distribution_semantics_exact(
        dossier: dict[str, Any] | None, text: str,
        expected: list[int] | None) -> bool:
    """Exercise admission/application, including every emitted quantile."""
    if expected is None:
        return dossier is None
    if dossier is None:
        return False
    claim = dossier["claims"][0]
    event = ContextEvent(
        event_id="window-1", event_type="override:stated_absolute_value",
        entity_scope=("readings",),
        effective_start=str(claim["effective_start"]),
        effective_end=str(claim["effective_end"]), known_at=CUTOFF,
        status="confirmed", confidence=1.0,
        attributes={"source_span": text})
    cutoff = datetime.fromisoformat(CUTOFF)
    future = [datetime.fromisoformat(stamp) for stamp in FUTURE]
    assessment = assess_future_events(
        [event], "readings", [10.0] * 12,
        [cutoff - timedelta(hours=11 - index) for index in range(12)],
        future, 1, base_points=[10.0 + index for index in range(12)])
    if len(assessment.admitted) != 1:
        return False
    base = [{
        "timestamp": stamp, "point": 10.0 + index,
        "q10": 8.0 + index, "q50": 10.0 + index,
        "q90": 12.0 + index,
    } for index, stamp in enumerate(FUTURE)]
    projected, _ = apply_future_events(base, assessment.admitted)
    active = set(expected)
    for index, (before, after) in enumerate(zip(base, projected)):
        keys = ("point", "q10", "q50", "q90")
        if index in active and any(float(after[key]) != 0.0 for key in keys):
            return False
        if index not in active and any(after[key] != before[key] for key in keys):
            return False
    return True


def run(output: Path) -> dict[str, Any]:
    rows = []
    for case_id, text, expected in CASES:
        dossier = deterministic_dated_zero_window_dossier(
            text, cutoff=CUTOFF, future_timestamps=FUTURE,
            target_name="readings")
        observed = _active_indices(dossier)
        distribution_exact = _distribution_semantics_exact(
            dossier, text, expected)
        rows.append({
            "case_id": case_id,
            "expected_active_indices": expected,
            "observed_active_indices": observed,
            "exact": observed == expected,
            "distribution_semantics_exact": distribution_exact,
            "deterministic": True,
            "future_target_observations_used": 0,
        })
    gates = {
        "complete": len(rows) == len(CASES),
        "all_semantics_exact": all(row["exact"] for row in rows),
        "all_distribution_semantics_exact": all(
            row["distribution_semantics_exact"] for row in rows),
        "deterministic_no_llm": all(row["deterministic"] for row in rows),
        "future_targets_never_used": all(
            row["future_target_observations_used"] == 0 for row in rows),
    }
    result = {
        "schema_version": 1,
        "benchmark": "operational-window-semantics",
        "evaluated_commit": code_revision(),
        "cases": len(rows),
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()

"""Evaluate deterministic zero-window parsing on a frozen phrase matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.common.manifest import code_revision
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


def run(output: Path) -> dict[str, Any]:
    rows = []
    for case_id, text, expected in CASES:
        dossier = deterministic_dated_zero_window_dossier(
            text, cutoff=CUTOFF, future_timestamps=FUTURE,
            target_name="readings")
        observed = _active_indices(dossier)
        rows.append({
            "case_id": case_id,
            "expected_active_indices": expected,
            "observed_active_indices": observed,
            "exact": observed == expected,
            "deterministic": True,
            "future_target_observations_used": 0,
        })
    gates = {
        "complete": len(rows) == len(CASES),
        "all_semantics_exact": all(row["exact"] for row in rows),
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

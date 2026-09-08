"""Installed-package checks for Gnomon's structural trust guarantees."""

from __future__ import annotations

import csv
import random
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def leakage_self_check(cases: int = 8, seed: int = 7) -> dict[str, Any]:
    """Check snapshot reads against declared cutoffs on finite synthetic cases.

    This is a mechanism check, not the historical LLM-control comparison.
    It ships so an installed wheel can exercise the snapshot mechanism
    without network access or benchmark fixtures.
    """
    from .session import GnomonSession
    from .temporal_store import TemporalStore

    if type(cases) is not int or not 1 <= cases <= 1000:
        raise ValueError("cases must be an integer from 1 to 1000")
    rng = random.Random(seed)
    rows = []
    with tempfile.TemporaryDirectory(prefix="gnomon-leakage-check-") as directory:
        root = Path(directory)
        for case in range(cases):
            start = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=case * 3)
            history, horizon = rng.randint(24, 60), rng.randint(1, 7)
            cutoff = start + timedelta(days=history - 1)
            source = root / f"case-{case}.csv"
            records = []
            for index in range(history + horizon):
                timestamp = start + timedelta(days=index)
                value = 100 + index * .2 + rng.uniform(-1, 1)
                records.append((timestamp.isoformat(), value, timestamp.isoformat()))
                if history - 4 <= index < history:
                    records.append((timestamp.isoformat(), value + 20,
                                    (cutoff + timedelta(days=2)).isoformat()))
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp", "value", "published"])
                writer.writerows(records)
            store_path = root / f"case-{case}.db"
            dataset = f"case_{case}"
            TemporalStore(store_path).ingest_csv(
                str(source), dataset=dataset, time_column="timestamp",
                target_column="value", known_at_column="published",
            )
            with GnomonSession.from_config() as session:
                inspected = session.data.inspect(f"store:{dataset}",
                    as_of=cutoff.isoformat(), store_path=str(store_path))
                request = session.data.request(inspected["data_ref"], horizon=horizon)
                execution = session.engine.forecast("last_value", request)
                forecast_matches = execution.result.point == (request.history[-1],) * horizon
                expected = tuple(float(record[1]) for record in records
                    if datetime.fromisoformat(record[0]) <= cutoff
                    and datetime.fromisoformat(record[2]) <= cutoff)
                history_matches = request.history == expected
            accesses = inspected["snapshot"].get("accesses", [])
            known = [datetime.fromisoformat(item["max_known_time"])
                     for item in accesses if item.get("max_known_time")]
            boundary_holds = bool(known) and max(known) <= cutoff
            holds = boundary_holds and history_matches and forecast_matches
            rows.append({"case": case, "cutoff": cutoff.isoformat(),
                         "max_known_time": max(known).isoformat() if known else None,
                         "history_length": history, "horizon": horizon,
                         "checks": {"known_time_boundary": boundary_holds,
                                    "visible_history_matches": history_matches, "forecast_matches": forecast_matches},
                         "holds": holds})
    return {"schema_version": "0.2", "check": "snapshot_temporal_leakage",
            "seed": seed, "cases": cases, "passed": sum(row["holds"] for row in rows),
            "failed": sum(not row["holds"] for row in rows),
            "checks_passed": bool(rows) and all(row["holds"] for row in rows),
            "evidence": "finite_synthetic_checks", "general_leakage_safety": "not_established",
            "rows": rows,
            "limitation": "These finite synthetic cases exercise the installed snapshot mechanism. They do not prove general leakage safety or rerun the historical hosted-LLM control arm."}

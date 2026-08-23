"""Installed-package checks for Gnomon's structural trust guarantees."""

from __future__ import annotations

import csv
import random
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def leakage_self_check(cases: int = 8, seed: int = 7) -> dict[str, Any]:
    """Prove snapshot reads stay at or before each declared cutoff.

    This is a mechanism check, not the historical LLM-control comparison.
    It ships so an installed wheel can reproduce the product guarantee
    without network access or benchmark fixtures.
    """
    from .runtime import forecast
    from .temporal_store import TemporalStore

    rng = random.Random(seed)
    rows = []
    with tempfile.TemporaryDirectory(prefix="gnomon-leakage-check-") as directory:
        root = Path(directory)
        for case in range(cases):
            start = datetime(2025, 1, 1, tzinfo=timezone.utc)
            history, horizon = 48, 4
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
            artifact, _ = forecast(
                f"store:{dataset}", time_column="timestamp", target_column="value",
                horizon=horizon, as_of=cutoff, store_path=str(store_path),
                output=str(root / "output"),
            )
            accesses = [access for evidence in artifact.evidence
                        if evidence.kind == "snapshot_access"
                        for access in evidence.payload.get("accesses", [])]
            known = [datetime.fromisoformat(item["max_known_time"])
                     for item in accesses if item.get("max_known_time")]
            holds = bool(known) and max(known) <= cutoff
            rows.append({"case": case, "cutoff": cutoff.isoformat(),
                         "max_known_time": max(known).isoformat() if known else None,
                         "holds": holds})
    return {"schema_version": "0.1", "check": "snapshot_temporal_leakage",
            "seed": seed, "cases": cases, "passed": sum(row["holds"] for row in rows),
            "failed": sum(not row["holds"] for row in rows),
            "structural_claim_proven": bool(rows) and all(row["holds"] for row in rows),
            "rows": rows,
            "limitation": "This validates the installed snapshot mechanism; it does not rerun the historical hosted-LLM control arm."}

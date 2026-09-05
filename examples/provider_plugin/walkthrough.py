"""Executable provider/ledger walkthrough and non-overwriting SQLite backup.

Run against an installed Gnomon and gnomon-example-provider in a new directory.
All observations below are synthetic; no model service or real business action
is invoked. Existing directories and backup destinations are never overwritten.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3

from gnomon import ForecastRequest, GnomonSession, TemporalLedger

EXAMPLE = Path(__file__).resolve().parent


def backup(source: Path, destination: Path) -> None:
    """Use SQLite's consistent backup API; do not copy an active DB with cp.

    The source is opened read-only, without constructing a TemporalLedger or
    migrating it. An interrupted backup may leave its newly created destination;
    validate it before use, or choose another new destination to retry.
    """
    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as original:
        if original.execute("PRAGMA application_id").fetchone()[0] != TemporalLedger.APPLICATION_ID:
            raise ValueError("source is not a Gnomon temporal ledger")
        if original.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise ValueError("source failed SQLite quick_check")
        # Exclusive, private creation refuses existing files, including symlinks.
        with open(destination, "xb", opener=lambda path, flags: os.open(path, flags, 0o600)):
            pass
        with closing(sqlite3.connect(destination)) as copied:
            original.backup(copied)
            if copied.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("backup failed SQLite integrity_check")
            if copied.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("backup failed SQLite foreign_key_check")


def run(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("providers.toml", "request.json"):
        (directory / name).write_bytes((EXAMPLE / name).read_bytes())
    request = ForecastRequest.from_dict(json.loads((directory / "request.json").read_text()))
    with GnomonSession.from_config(directory / "providers.toml") as session:
        last = session.forecast("example-last", request)
        mean = session.forecast("example-mean", request)
        again = session.forecast("example-mean", request)
        assert mean["execution_id"] != again["execution_id"]
        assert list(last["result"]["point"]) == [14, 14]
        assert list(mean["result"]["point"]) == list(again["result"]["point"]) == [12, 12]
        assert not last["action_authorized"] and last["evidence"] == "inference_only"
        ledger, execution_id = session.ledger, last["execution_id"]
        pending = ledger.evaluate(execution_id)

        def actual(day, value, available):
            return ledger.append_actual(
                series_id=request.series_id, unit=request.unit, value=value,
                valid_time=f"2025-01-{day:02d}T00:00:00Z",
                source_available_at=f"2025-01-{available:02d}T00:00:00Z",
                source_ref="synthetic walkthrough",
            )

        first_actual = actual(3, 15, 4)
        partial = ledger.evaluate(execution_id)
        actual(4, 16, 5)
        complete = ledger.evaluate(execution_id)
        # Capture a real locally recorded cutoff, not a fabricated historical clock.
        recorded_cutoff = max(row["recorded_at"] for row in ledger.actuals_as_of(request.series_id, unit=request.unit))
        correction = actual(3, 20, 6)
        revised = ledger.evaluate(execution_id)
        source_replay = ledger.evaluate(execution_id, source_as_of="2025-01-05T12:00:00Z")
        recorded_replay = ledger.evaluate(execution_id, recorded_as_of=recorded_cutoff)
        assert [pending["status"], partial["status"], complete["status"]] == ["pending", "partial", "complete"]
        assert [partial["mae"], complete["mae"], revised["mae"]] == [1, 1.5, 4]
        assert source_replay["mae"] == recorded_replay["mae"] == 1.5
        assert first_actual in source_replay["actual_ids"] and correction in revised["actual_ids"]
        assert ledger.execution(execution_id)["result"]["point"] == [14, 14]
        decision_id = ledger.record_decision(
            execution_ids=[execution_id], policy={"version": "demo-only"},
            inputs={"stock": 5}, action={"proposed_order": 3},
        )
        assert ledger.decision(decision_id)["authorization_ref"] is None
        scores = ledger.evaluations(execution_id)

    backup(directory / "ledger.db", directory / "ledger-backup.db")
    # Restore to another path and verify before any operator changes live config.
    backup(directory / "ledger-backup.db", directory / "restored.db")
    restored = TemporalLedger(directory / "restored.db")
    assert restored.execution(execution_id)["result"]["point"] == [14, 14]
    assert restored.evaluations(execution_id) == scores
    receipt = {
        "status": "passed", "fixture": "synthetic_not_live_model_evidence",
        "execution_id": execution_id, "factory_execution_ids": [mean["execution_id"], again["execution_id"]],
        "original_forecast": [14, 14], "score_statuses": [s["status"] for s in scores],
        "mae_history": [s["mae"] for s in scores], "recorded_cutoff": recorded_cutoff,
        "decision_id": decision_id, "action_executed": False,
        "backup_restored": True, "ledger_schema": TemporalLedger.SCHEMA_VERSION,
    }
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("run", help="run synthetic examples into a new directory")
    demo.add_argument("--output-dir", type=Path, required=True)
    save = commands.add_parser("backup", help="copy an existing ledger to a new destination")
    save.add_argument("--source", type=Path, required=True)
    save.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        print(json.dumps(run(args.output_dir), indent=2))
    else:
        backup(args.source, args.destination)
        print(json.dumps({"status": "backed_up", "destination": str(args.destination)}))


if __name__ == "__main__":
    main()

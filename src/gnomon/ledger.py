"""Optional append-only execution and outcome ledger.

SQLite stores small immutable, content-addressed request/result payloads in
the same transaction as execution metadata. Existing file artifacts remain
the appropriate home for large reports. This is not a general event database
or an assertion that a model is calibrated or an action authorized.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from uuid import uuid4

from .forecast_adapter import ForecastAdapterError
from .ids import SYSTEM_CLOCK


def _time(value: str | datetime) -> str:
    try:
        dt = datetime.fromisoformat(value) if isinstance(value, str) else value
        if not isinstance(dt, datetime) or dt.tzinfo is None:
            raise ValueError("timezone required")
        return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")
    except (ValueError, TypeError):
        raise ForecastAdapterError("ledger timestamps require an explicit timezone") from None


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class TemporalLedger:
    SCHEMA_VERSION = 3
    APPLICATION_ID = 0x474E4F4D

    def __init__(self, path: str | Path, *, clock=None):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or SYSTEM_CLOCK
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            application = conn.execute("PRAGMA application_id").fetchone()[0]
            if application not in (0, self.APPLICATION_ID) or (version and application != self.APPLICATION_ID):
                raise ForecastAdapterError("database is not a Gnomon temporal ledger")
            if version not in (0, 1, 2, self.SCHEMA_VERSION):
                raise ForecastAdapterError(f"unsupported ledger schema version {version}")
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if version == 0 and tables:
                raise ForecastAdapterError("use a separate ledger file; existing databases require an explicit importer")
            schema = """
                CREATE TABLE IF NOT EXISTS payloads (
                    payload_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS executions (
                    execution_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
                    payload_id TEXT NOT NULL REFERENCES payloads(payload_id),
                    recorded_at TEXT NOT NULL, cache_hit INTEGER NOT NULL);
                CREATE INDEX IF NOT EXISTS executions_fingerprint ON executions(fingerprint);
                CREATE TABLE IF NOT EXISTS actuals (
                    actual_id TEXT PRIMARY KEY, series_id TEXT NOT NULL,
                    valid_time TEXT NOT NULL, source_available_at TEXT NOT NULL,
                    recorded_at TEXT NOT NULL, value REAL NOT NULL,
                    unit TEXT, source_ref TEXT NOT NULL, revision INTEGER NOT NULL);
                CREATE INDEX IF NOT EXISTS actuals_series_time ON actuals(series_id, valid_time);
                CREATE TABLE IF NOT EXISTS evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL REFERENCES executions(execution_id),
                    recorded_at TEXT NOT NULL, payload_json TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS evaluations_execution ON evaluations(execution_id);
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS decision_outcomes (
                    outcome_id TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL REFERENCES decisions(decision_id),
                    recorded_at TEXT NOT NULL, payload_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS imports (
                    source_id TEXT NOT NULL, series TEXT NOT NULL,
                    execution_id TEXT NOT NULL REFERENCES executions(execution_id),
                    PRIMARY KEY(source_id, series));
                CREATE TABLE IF NOT EXISTS studies (
                    study_id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL,
                    payload_id TEXT NOT NULL REFERENCES payloads(payload_id));
                CREATE TABLE IF NOT EXISTS study_executions (
                    study_id TEXT NOT NULL REFERENCES studies(study_id),
                    execution_id TEXT NOT NULL REFERENCES executions(execution_id),
                    PRIMARY KEY(study_id, execution_id));
            """
            for statement in schema.split(";"):
                if statement.strip():
                    conn.execute(statement)
            for table in ("payloads", "executions", "actuals", "evaluations", "decisions", "decision_outcomes", "imports",
                          "studies", "study_executions"):
                for action in ("UPDATE", "DELETE"):
                    conn.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{action} "
                                 f"BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END")
            conn.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")
            conn.execute(f"PRAGMA application_id={self.APPLICATION_ID}")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _now(self):
        return _time(self.clock.now())

    def record_execution(self, execution) -> str:
        data = execution.to_dict()
        with self._connect() as conn:
            return self._insert_execution(conn, data)

    def record_study(self, report: dict) -> str:
        """Append a matched backtest report without importing actual revisions.

        Truth vintages belong to this frozen study payload. They are not online
        outcome submissions, and cannot overwrite or masquerade as ledger actuals.
        """
        if report.get("evidence") != "rolling_origin_backtest" or report.get("action_authorized") is not False:
            raise ForecastAdapterError("study must be an unauthorized rolling-origin backtest report")
        encoded = _json(report)
        payload_id = hashlib.sha256(encoded.encode()).hexdigest()
        execution_ids = {run["execution_id"] for fold in report["folds"] for run in fold["runs"].values()
                         if "execution_id" in run}
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO payloads VALUES (?,?)", (payload_id, encoded))
            conn.execute("INSERT INTO studies VALUES (?,?,?)", (report["study_id"], self._now(), payload_id))
            conn.executemany("INSERT INTO study_executions VALUES (?,?)",
                             [(report["study_id"], execution_id) for execution_id in sorted(execution_ids)])
        return report["study_id"]

    def study(self, study_id: str, *, recorded_as_of: str | None = None) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT s.recorded_at, p.payload_id, p.payload_json FROM studies s "
                               "JOIN payloads p USING(payload_id) WHERE study_id=?", (study_id,)).fetchone()
        if row is None or (recorded_as_of is not None and row["recorded_at"] > _time(recorded_as_of)):
            raise ForecastAdapterError("study was not recorded by the requested cutoff")
        if hashlib.sha256(row["payload_json"].encode()).hexdigest() != row["payload_id"]:
            raise ForecastAdapterError("study payload integrity check failed")
        return {**json.loads(row["payload_json"]), "recorded_at": row["recorded_at"]}

    def _insert_execution(self, conn, data) -> str:
        data = dict(data)
        execution_id, fingerprint = data.pop("execution_id"), data.pop("fingerprint")
        cache_hit = data.pop("cache_hit")
        encoded = _json(data)
        payload_id = hashlib.sha256(encoded.encode()).hexdigest()
        conn.execute("INSERT OR IGNORE INTO payloads VALUES (?,?)", (payload_id, encoded))
        conn.execute("INSERT INTO executions VALUES (?,?,?,?,?)",
                     (execution_id, fingerprint, payload_id, self._now(), int(cache_hit)))
        return execution_id

    def import_artifact(self, artifact_path: str, *, project: str | None = None,
                        naive_timezone: str | None = None) -> list[str]:
        """Preserve a historical forecast without pretending to rerun its model.

        Missing input snapshots stay unknown. Imports are idempotent and retain
        the original artifact creation time separately from local recording time.
        """
        from .artifact_import import read_forecast_import
        source_id, records = read_forecast_import(artifact_path, project=project, naive_timezone=naive_timezone)
        ids = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for record in records:
                series = record["request"]["series_id"]
                prior = conn.execute("SELECT execution_id FROM imports WHERE source_id=? AND series=?", (source_id, series)).fetchone()
                if prior:
                    ids.append(prior[0])
                    continue
                execution_id = self._insert_execution(conn, record)
                conn.execute("INSERT INTO imports VALUES (?,?,?)", (source_id, series, execution_id))
                ids.append(execution_id)
        return ids

    def import_tracking(self, registry_path: str, *, project: str | None = None,
                        naive_timezone: str | None = None) -> dict:
        """Read a legacy registry without modifying it or inventing score vintages."""
        from urllib.parse import quote
        from .contracts import GnomonError
        conn = sqlite3.connect("file:" + quote(str(Path(registry_path).resolve())) + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(forecasts)")}
            if not {"forecast_id", "project", "artifact_path"} <= columns:
                raise ForecastAdapterError("not a supported legacy tracking registry")
            rows = conn.execute("SELECT * FROM forecasts" + (" WHERE project=?" if project is not None else ""),
                                (project,) if project is not None else ()).fetchall()
        finally:
            conn.close()
        imported, skipped = [], []
        for row in rows:
            try:
                ids = self.import_artifact(row["artifact_path"], project=row["project"], naive_timezone=naive_timezone)
                imported.extend(ids)
            except (OSError, ValueError, GnomonError) as exc:
                skipped.append({"forecast_id": row["forecast_id"], "reason": str(exc)})
        return {"execution_ids": list(dict.fromkeys(imported)), "skipped": skipped,
                "legacy_registry_unchanged": True,
                "score_history": "not_reconstructed; mutable legacy summaries remain in the original registry"}

    def execution(self, execution_id: str) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT e.*, p.payload_json FROM executions e JOIN payloads p USING(payload_id) "
                               "WHERE execution_id=?", (execution_id,)).fetchone()
        if row is None:
            raise ForecastAdapterError("unknown execution_id")
        if hashlib.sha256(row["payload_json"].encode()).hexdigest() != row["payload_id"]:
            raise ForecastAdapterError("execution payload failed integrity verification")
        return {**json.loads(row["payload_json"]), "execution_id": row["execution_id"],
                "fingerprint": row["fingerprint"], "payload_id": row["payload_id"],
                "recorded_at": row["recorded_at"], "cache_hit": bool(row["cache_hit"])}

    def append_actual(self, *, series_id: str, valid_time: str, value: float,
                      source_available_at: str, unit: str | None = None,
                      source_ref: str = "") -> str:
        if not isinstance(series_id, str) or not series_id or isinstance(value, bool) or not math.isfinite(value):
            raise ForecastAdapterError("actual requires a series_id and finite value")
        valid, available = _time(valid_time), _time(source_available_at)
        identity = _json([series_id, valid, available, float(value), unit, source_ref])
        actual_id = hashlib.sha256(identity.encode()).hexdigest()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM actuals WHERE actual_id=?", (actual_id,)).fetchone():
                return actual_id
            revision = conn.execute("SELECT COALESCE(MAX(revision), -1)+1 FROM actuals "
                                    "WHERE series_id=? AND valid_time=? AND unit IS ?",
                                    (series_id, valid, unit)).fetchone()[0]
            conn.execute("INSERT INTO actuals VALUES (?,?,?,?,?,?,?,?,?)",
                         (actual_id, series_id, valid, available, self._now(), float(value), unit, source_ref, revision))
        return actual_id

    def actuals_as_of(self, series_id: str, *, source_as_of: str | None = None,
                     recorded_as_of: str | None = None, unit: str | None = None) -> list[dict]:
        query = "SELECT * FROM actuals WHERE series_id=? AND unit IS ?"
        args = [series_id, unit]
        for field, cutoff in (("source_available_at", source_as_of), ("recorded_at", recorded_as_of)):
            if cutoff is not None:
                query += f" AND {field}<=?"
                args.append(_time(cutoff))
        query += " ORDER BY valid_time, source_available_at, revision"
        with self._connect() as conn:
            rows = conn.execute(query, args).fetchall()
        latest = {row["valid_time"]: dict(row) for row in rows}
        return list(latest.values())

    def evaluate(self, execution_id: str, *, source_as_of: str | None = None,
                 recorded_as_of: str | None = None, allow_partial: bool = True) -> dict:
        execution = self.execution(execution_id)
        req = execution["request"]
        if not req["series_id"] or not req["future_timestamps"]:
            raise ForecastAdapterError("scoring requires series_id and explicit future_timestamps")
        times = [_time(t) for t in req["future_timestamps"]]
        actuals = {r["valid_time"]: r for r in self.actuals_as_of(req["series_id"], unit=req["unit"],
                                                               source_as_of=source_as_of, recorded_as_of=recorded_as_of)}
        pairs = [(i, actuals[t]) for i, t in enumerate(times) if t in actuals]
        if not allow_partial and len(pairs) != len(times):
            raise ForecastAdapterError("complete actual horizon is not yet available")
        errors = [execution["result"]["point"][i] - row["value"] for i, row in pairs]
        status = "complete" if len(pairs) == len(times) else "partial" if pairs else "pending"
        record = {"evaluation_id": str(uuid4()), "execution_id": execution_id,
                  "recorded_at": self._now(), "source_as_of": _time(source_as_of) if source_as_of else None,
                  "recorded_as_of": _time(recorded_as_of) if recorded_as_of else None,
                  "metric_version": "point-errors/1", "status": status,
                  "matched_steps": [i for i, _ in pairs], "actual_ids": [r["actual_id"] for _, r in pairs],
                  "n": len(pairs), "horizon": len(times),
                  "mae": sum(abs(e) for e in errors) / len(errors) if errors else None,
                  "rmse": math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else None,
                  "bias": sum(errors) / len(errors) if errors else None}
        with self._connect() as conn:
            conn.execute("INSERT INTO evaluations VALUES (?,?,?,?)",
                         (record["evaluation_id"], execution_id, record["recorded_at"], _json(record)))
        return record

    def evaluations(self, execution_id: str, *, recorded_as_of: str | None = None) -> list[dict]:
        query, args = "SELECT payload_json FROM evaluations WHERE execution_id=?", [execution_id]
        if recorded_as_of is not None:
            query += " AND recorded_at<=?"
            args.append(_time(recorded_as_of))
        with self._connect() as conn:
            return [json.loads(row[0]) for row in conn.execute(query + " ORDER BY rowid", args)]

    def pending(self, *, source_as_of: str | None = None, recorded_as_of: str | None = None) -> list[dict]:
        with self._connect() as conn:
            ids = [row[0] for row in conn.execute("SELECT execution_id FROM executions ORDER BY rowid")]
        pending = []
        for execution_id in ids:
            execution = self.execution(execution_id)
            if recorded_as_of and execution["recorded_at"] > _time(recorded_as_of):
                continue
            req = execution["request"]
            if not req["series_id"] or not req["future_timestamps"]:
                pending.append({"execution_id": execution_id, "status": "unscorable", "missing_steps": None})
                continue
            visible = {row["valid_time"] for row in self.actuals_as_of(req["series_id"], unit=req["unit"],
                                                                      source_as_of=source_as_of, recorded_as_of=recorded_as_of)}
            try:
                missing = [i for i, t in enumerate(req["future_timestamps"]) if _time(t) not in visible]
            except ForecastAdapterError:
                pending.append({"execution_id": execution_id, "status": "unscorable", "reason": "timezone_unresolved", "missing_steps": None})
                continue
            if missing:
                pending.append({"execution_id": execution_id, "status": "pending", "missing_steps": missing})
        return pending

    def compare(self, execution_ids: list[str], *, source_as_of: str | None = None,
                recorded_as_of: str | None = None) -> dict:
        if len(execution_ids) < 2 or len(set(execution_ids)) != len(execution_ids):
            raise ForecastAdapterError("comparison requires at least two distinct executions")
        runs = [self.execution(eid) for eid in execution_ids]
        if any(not run["request"].get("history") or run["result"].get("metadata", {}).get("complete_request_reconstructed") is False
               for run in runs):
            raise ForecastAdapterError("matched comparison requires preserved complete input history and request")
        keys = ("series_id", "unit", "timestamps", "future_timestamps", "history", "horizon",
                "cutoff", "known_time_cutoff", "recorded_time_cutoff", "snapshot_id",
                "past_covariates", "future_covariates", "related_series", "season", "frequency")
        keys += ("past_covariate_names", "future_covariate_names")
        if any(any(run["request"][key] != runs[0]["request"][key] for key in keys) for run in runs[1:]):
            raise ForecastAdapterError("comparison requires matched inputs, snapshot and forecast origin")
        # Resolve once to the same vintages for every candidate; evaluating
        # separately while ingestion proceeds could otherwise change the cohort.
        req = runs[0]["request"]
        if not req["series_id"] or not req["future_timestamps"]:
            raise ForecastAdapterError("comparison requires identified forecast timestamps")
        actuals = {r["valid_time"]: r for r in self.actuals_as_of(req["series_id"], unit=req["unit"],
                                                               source_as_of=source_as_of, recorded_as_of=recorded_as_of)}
        pairs = [(i, actuals[_time(t)]) for i, t in enumerate(req["future_timestamps"]) if _time(t) in actuals]
        return {"n": len(pairs), "horizon": req["horizon"], "metric_version": "point-errors/1",
                "source_as_of": _time(source_as_of) if source_as_of else None,
                "recorded_as_of": _time(recorded_as_of) if recorded_as_of else None,
                "snapshot_id": req["snapshot_id"],
                "actual_ids": [a["actual_id"] for _, a in pairs], "matched_steps": [i for i, _ in pairs],
                "models": [{"execution_id": run["execution_id"], "provider": run["provider"], "revision": run["revision"],
                            "mae": sum(abs(run["result"]["point"][i] - a["value"]) for i, a in pairs) / len(pairs) if pairs else None}
                           for run in runs]}

    def record_decision(self, *, execution_ids: list[str], policy: dict,
                        inputs: dict, action: dict, authorization_ref: str | None = None) -> str:
        for execution_id in execution_ids:
            self.execution(execution_id)
        decision_id, now = str(uuid4()), self._now()
        payload = {"decision_id": decision_id, "execution_ids": execution_ids, "policy": policy,
                   "inputs": inputs, "action": action, "authorization_ref": authorization_ref,
                   "meaning": "recorded decision; no action is executed by the ledger"}
        with self._connect() as conn:
            conn.execute("INSERT INTO decisions VALUES (?,?,?)", (decision_id, now, _json(payload)))
        return decision_id

    def append_decision_outcome(self, decision_id: str, *, outcome: dict,
                                source_available_at: str) -> str:
        outcome_id = str(uuid4())
        payload = {"outcome_id": outcome_id, "decision_id": decision_id, "outcome": outcome,
                   "source_available_at": _time(source_available_at)}
        with self._connect() as conn:
            conn.execute("INSERT INTO decision_outcomes VALUES (?,?,?,?)",
                         (outcome_id, decision_id, self._now(), _json(payload)))
        return outcome_id

    def decision(self, decision_id: str, *, source_as_of: str | None = None,
                 recorded_as_of: str | None = None) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT payload_json, recorded_at FROM decisions WHERE decision_id=?", (decision_id,)).fetchone()
            if row is None:
                raise ForecastAdapterError("unknown decision_id")
            if recorded_as_of and row[1] > _time(recorded_as_of):
                raise ForecastAdapterError("decision was not recorded at this cutoff")
            outcomes = [{**json.loads(r[0]), "recorded_at": r[1]} for r in conn.execute(
                "SELECT payload_json, recorded_at FROM decision_outcomes WHERE decision_id=? ORDER BY rowid", (decision_id,))]
        outcomes = [outcome for outcome in outcomes
                    if (not source_as_of or outcome["source_available_at"] <= _time(source_as_of))
                    and (not recorded_as_of or outcome["recorded_at"] <= _time(recorded_as_of))]
        return {**json.loads(row[0]), "recorded_at": row[1], "outcomes": outcomes}

"""Optional append-only execution and outcome ledger.

SQLite stores small immutable, content-addressed request/result payloads in
the same transaction as execution metadata. Existing file artifacts remain
the appropriate home for large reports. This is not a general event database
or an assertion that a model is calibrated or an action authorized.
"""

from __future__ import annotations

from contextlib import contextmanager
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from uuid import uuid4

from .forecast_adapter import ForecastAdapterError, point_error_metrics
from .contracts import GnomonError
from .ids import SYSTEM_CLOCK


_METRIC_VERSION = "point-errors/1"
_MATCH_KEYS = ("series_id", "unit", "timestamps", "future_timestamps", "history", "horizon",
               "cutoff", "known_time_cutoff", "recorded_time_cutoff", "snapshot_id",
               "past_covariates", "future_covariates", "related_series", "season", "frequency",
               "past_covariate_names", "future_covariate_names", "quantiles", "samples")


def _bounded(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ForecastAdapterError(f"{name} must be an integer between 1 and {maximum}")
    return value


def _batch(values, name, maximum):
    if not isinstance(values, list) or not 1 <= len(values) <= maximum:
        raise ForecastAdapterError(f"{name} must contain between 1 and {maximum} items")
    return values


def _time(value: str | datetime, field="timestamp") -> str:
    try:
        dt = datetime.fromisoformat(value) if isinstance(value, str) else value
        if not isinstance(dt, datetime) or dt.tzinfo is None:
            raise ValueError("timezone required")
        return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")
    except (ValueError, TypeError):
        raise ForecastAdapterError("ledger timestamps require an explicit timezone; invalid field: " + field,
                                   details={"field": field, "accepted_example": "2026-01-21T00:00:00Z"}) from None


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class TemporalLedger:
    # Version 4 is the first supported public ledger schema. Pre-1.0 schemas
    # are deliberately rejected instead of carrying unused migration code.
    SCHEMA_VERSION = 4
    APPLICATION_ID = 0x474E4F4D

    def __init__(self, path: str | Path, *, clock=None, create=True):
        self.path = Path(path).expanduser()
        self._create = create
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        elif not self.path.is_file():
            raise GnomonError("LEDGER_NOT_FOUND", "Ledger file does not exist. Check --ledger-path or the path in "
                              "provider TOML; infer/evaluate create ledgers when recording evidence.",
                              {"path": str(self.path.resolve())})
        self.clock = clock or SYSTEM_CLOCK
        with self._connect() as conn:
            if create:
                conn.execute("BEGIN IMMEDIATE")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            application = conn.execute("PRAGMA application_id").fetchone()[0]
            if not create:
                if version != self.SCHEMA_VERSION or application != self.APPLICATION_ID:
                    raise GnomonError("INVALID_LEDGER", "The selected file is not a supported Gnomon ledger.",
                                      {"path": str(self.path.resolve())})
                return
            if application not in (0, self.APPLICATION_ID) or (version and application != self.APPLICATION_ID):
                raise ForecastAdapterError("database is not a Gnomon temporal ledger")
            if version not in (0, self.SCHEMA_VERSION):
                raise ForecastAdapterError(f"unsupported ledger schema version {version}")
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if version == 0 and tables:
                raise ForecastAdapterError("use a new empty file for a Gnomon ledger")
            schema = """
                CREATE TABLE IF NOT EXISTS payloads (
                    payload_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS executions (
                    execution_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
                    payload_id TEXT NOT NULL REFERENCES payloads(payload_id),
                    recorded_at TEXT NOT NULL, cache_hit INTEGER NOT NULL);
                CREATE INDEX IF NOT EXISTS executions_fingerprint ON executions(fingerprint);
                CREATE INDEX IF NOT EXISTS executions_payload ON executions(payload_id);
                CREATE INDEX IF NOT EXISTS payloads_task ON payloads(
                    json_extract(payload_json, '$.request.series_id'),
                    json_extract(payload_json, '$.request.horizon'));
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
            for table in ("payloads", "executions", "actuals", "evaluations", "decisions", "decision_outcomes",
                          "studies", "study_executions"):
                for action in ("UPDATE", "DELETE"):
                    conn.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{action} "
                                 f"BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END")
            conn.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")
            conn.execute(f"PRAGMA application_id={self.APPLICATION_ID}")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path.resolve().as_uri() + ("?mode=rwc" if self._create else "?mode=rw"),
                               uri=True, timeout=30)
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
        if row is None:
            raise ForecastAdapterError("study_id was not found in this ledger", details={"reason": "study_not_found"})
        if recorded_as_of is not None and row["recorded_at"] > _time(recorded_as_of):
            raise ForecastAdapterError("study was not recorded by the requested cutoff",
                details={'study_id': study_id, 'study_recorded_at': row['recorded_at'], 'recorded_as_of': recorded_as_of,
                         'rejected_fields': ['recorded_as_of'],
                         'guidance': 'Select a study already recorded by the intended cutoff. Only advance recorded_as_of to study_recorded_at or later if later evidence is appropriate to the task.'},
                repair_options=[{'action': 'select_visible_study', 'description': 'Use evidence recorded by the intended recording cutoff, or explicitly choose a later evidence cutoff.'}])
        if hashlib.sha256(row["payload_json"].encode()).hexdigest() != row["payload_id"]:
            raise ForecastAdapterError("study payload integrity check failed")
        return {**json.loads(row["payload_json"]), "recorded_at": row["recorded_at"]}

    def _insert_execution(self, conn, data) -> str:
        data = dict(data)
        execution_id, fingerprint = data.pop("execution_id"), data.pop("fingerprint")
        cache_hit = data.pop("cache_hit")
        # Lookup diagnostics describe an invocation, not its reusable forecast
        # payload. Keep cache_hit in the execution row and preserve deduplication.
        data.pop('cache', None)
        encoded = _json(data)
        payload_id = hashlib.sha256(encoded.encode()).hexdigest()
        conn.execute("INSERT OR IGNORE INTO payloads VALUES (?,?)", (payload_id, encoded))
        conn.execute("INSERT INTO executions VALUES (?,?,?,?,?)",
                     (execution_id, fingerprint, payload_id, self._now(), int(cache_hit)))
        return execution_id

    def execution(self, execution_id: str) -> dict:
        with self._connect() as conn:
            return self._execution(conn, execution_id)

    def _execution(self, conn, execution_id):
        row = conn.execute("SELECT e.*, p.payload_json FROM executions e JOIN payloads p USING(payload_id) "
                           "WHERE execution_id=?", (execution_id,)).fetchone()
        if row is None:
            raise ForecastAdapterError("unknown execution_id")
        if hashlib.sha256(row["payload_json"].encode()).hexdigest() != row["payload_id"]:
            raise ForecastAdapterError("execution payload failed integrity verification")
        return {**json.loads(row["payload_json"]), "execution_id": row["execution_id"],
                "fingerprint": row["fingerprint"], "payload_id": row["payload_id"],
                "recorded_at": row["recorded_at"], "cache_hit": bool(row["cache_hit"])}

    def append_actual(self, *, series_id: str | None = None, valid_time: str | None = None,
                      value: float | None = None, source_available_at: str | None = None,
                      unit: str | None = None, source_ref: str = "", actuals: list[dict] | None = None) -> str | list[str]:
        """Append one actual or an atomic batch (at most 1,000), preserving revisions."""
        if actuals is not None:
            if any(v is not None for v in (series_id, valid_time, value, source_available_at, unit)) or source_ref != "":
                raise ForecastAdapterError("supply actuals or single-actual fields, not both")
            rows = _batch(actuals, "actuals", 1000)
            allowed = {"series_id", "valid_time", "value", "source_available_at", "unit", "source_ref"}
            required = allowed - {"unit", "source_ref"}
            if any(not isinstance(r, dict) or set(r) - allowed or not required <= set(r) for r in rows):
                raise ForecastAdapterError("each actual requires series_id, valid_time, value and source_available_at")
        else:
            rows = [dict(series_id=series_id, valid_time=valid_time, value=value,
                         source_available_at=source_available_at, unit=unit, source_ref=source_ref)]
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            ids = [self._append_actual(conn, **row) for row in rows]
        return ids if actuals is not None else ids[0]

    def _append_actual(self, conn, *, series_id, valid_time, value, source_available_at, unit=None, source_ref=""):
        try:
            finite = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not isinstance(series_id, str) or not series_id or not finite:
            raise ForecastAdapterError("actual requires a series_id and finite value")
        if (unit is not None and (not isinstance(unit, str) or not unit)) or not isinstance(source_ref, str):
            raise ForecastAdapterError("unit must be a nonempty string or null; source_ref must be a string")
        valid, available = _time(valid_time, "valid_time"), _time(source_available_at, "source_available_at")
        identity = _json([series_id, valid, available, float(value), unit, source_ref])
        actual_id = hashlib.sha256(identity.encode()).hexdigest()
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
        with self._connect() as conn:
            return self._actuals(conn, series_id, source_as_of, recorded_as_of, unit)

    def _actuals(self, conn, series_id, source_as_of, recorded_as_of, unit, times=None):
        query = "SELECT * FROM actuals WHERE series_id=? AND unit IS ?"
        args = [series_id, unit]
        for field, cutoff in (("source_available_at", source_as_of), ("recorded_at", recorded_as_of)):
            if cutoff is not None:
                query += f" AND {field}<=?"
                args.append(_time(cutoff))
        if times:
            query += " AND valid_time>=? AND valid_time<=?"
            args.extend((min(times), max(times)))
        query += " ORDER BY valid_time, source_available_at, revision"
        rows = conn.execute(query, args).fetchall()
        latest = {row["valid_time"]: dict(row) for row in rows}
        return list(latest.values())

    def evaluate(self, execution_id: str | None = None, *, source_as_of: str | None = None,
                 recorded_as_of: str | None = None, allow_partial: bool = True,
                 execution_ids: list[str] | None = None,
                 include_current_coverage: bool = False) -> dict | list[dict]:
        """Score one execution or an atomic batch of at most 100. Exact retries reuse scores.

        include_current_coverage adds current query diagnostics alongside immutable
        score coverage. CLI/MCP enable this; evaluations() always reads saved evidence.
        """
        if type(allow_partial) is not bool:
            raise ForecastAdapterError("allow_partial must be a boolean")
        if type(include_current_coverage) is not bool:
            raise ForecastAdapterError("include_current_coverage must be a boolean")
        if (execution_id is None) == (execution_ids is None):
            raise ForecastAdapterError("supply execution_id or execution_ids, not both")
        ids = _batch(execution_ids, "execution_ids", 100) if execution_ids is not None else [execution_id]
        if any(not isinstance(eid, str) or not eid for eid in ids) or len(set(ids)) != len(ids):
            raise ForecastAdapterError("execution IDs must be distinct nonempty strings")
        source_as_of = _time(source_as_of, "source_as_of") if source_as_of is not None else None
        recorded_as_of = _time(recorded_as_of, "recorded_as_of") if recorded_as_of is not None else None
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            scores = [self._evaluate(conn, eid, source_as_of, recorded_as_of, allow_partial,
                                     include_current_coverage) for eid in ids]
        return scores if execution_ids is not None else scores[0]

    def _pairs(self, conn, req, source_as_of, recorded_as_of):
        if not req["series_id"] or not req["future_timestamps"]:
            raise ForecastAdapterError("scoring requires series_id and explicit future_timestamps")
        try:
            times = [_time(t) for t in req["future_timestamps"]]
        except ForecastAdapterError:
            raise ForecastAdapterError(
                "Stored request.future_timestamps require an explicit timezone; declare the source timezone "
                "at inspection (--timezone) and record a new forecast. Changing scoring cutoffs cannot repair stored timestamps.",
                details={"field": "request.future_timestamps", "series_id": req["series_id"]}) from None
        actuals = {r["valid_time"]: r for r in self._actuals(
            conn, req["series_id"], source_as_of, recorded_as_of, req["unit"], times)}
        return times, [(i, actuals[t]) for i, t in enumerate(times) if t in actuals]

    def _coverage(self, conn, req, times, pairs, source_as_of, recorded_as_of):
        matched = {i for i, _ in pairs}
        missing = [i for i in range(len(times)) if i not in matched]
        # Only disclose other units visible under the same temporal cutoffs.
        query = "SELECT DISTINCT valid_time, unit FROM actuals WHERE series_id=? AND unit IS NOT ?"
        args = [req["series_id"], req["unit"]]
        if source_as_of is not None:
            query += " AND source_available_at<=?"
            args.append(source_as_of)
        if recorded_as_of is not None:
            query += " AND recorded_at<=?"
            args.append(recorded_as_of)
        if times:
            query += " AND valid_time>=? AND valid_time<=?"
            args.extend((times[0], times[-1]))
        missing_times = {times[i] for i in missing}
        units = {row["unit"] for row in conn.execute(query, args)
                 if row["valid_time"] in missing_times} if missing else set()
        return {"required_steps": len(times), "matched_steps": len(pairs),
                "fraction": len(pairs) / len(times), "missing_steps": missing,
                "missing_timestamps": [times[i] for i in missing], "unit": req["unit"],
                "other_units_at_missing_steps": sorted(units, key=lambda u: (u is not None, u or ""))}

    def _evaluate(self, conn, execution_id, source_as_of, recorded_as_of, allow_partial,
                  include_current_coverage=False):
        execution = self._execution(conn, execution_id)
        if recorded_as_of is not None and execution["recorded_at"] > recorded_as_of:
            raise ForecastAdapterError("execution was not recorded by the requested cutoff", details={
                "execution_id": execution_id, "execution_recorded_at": execution["recorded_at"],
                "recorded_as_of": recorded_as_of,
                "guidance": "Select an execution that existed at this cutoff, or choose a recording cutoff at or after execution_recorded_at if appropriate for the task."})
        req = execution["request"]
        times, pairs = self._pairs(conn, req, source_as_of, recorded_as_of)
        coverage = self._coverage(conn, req, times, pairs, source_as_of, recorded_as_of)
        if not allow_partial and len(pairs) != len(times):
            raise ForecastAdapterError("complete actual horizon is not yet available", details={
                "execution_id": execution_id, "coverage": coverage,
                "source_as_of": source_as_of, "recorded_as_of": recorded_as_of,
                "guidance": "Supply actual observations for missing_timestamps with exactly this unit, and source/recording times visible at the chosen cutoffs. Change cutoffs only when appropriate for the task; allow_partial=true explicitly accepts incomplete scoring."})
        metrics = point_error_metrics([(execution["result"]["point"][i], row["value"]) for i, row in pairs])
        status = "complete" if len(pairs) == len(times) else "partial" if pairs else "pending"
        record = {"evaluation_id": str(uuid4()), "execution_id": execution_id,
                  "recorded_at": self._now(), "source_as_of": _time(source_as_of) if source_as_of else None,
                  "recorded_as_of": _time(recorded_as_of) if recorded_as_of else None,
                  "metric_version": _METRIC_VERSION, "status": status,
                  "matched_steps": [i for i, _ in pairs], "actual_ids": [r["actual_id"] for _, r in pairs],
                  "horizon": len(times), "complete": status == "complete", "coverage": coverage, **metrics}
        previous = conn.execute("SELECT payload_json FROM evaluations WHERE execution_id=? "
            "AND json_extract(payload_json, '$.actual_ids')=? AND json_extract(payload_json, '$.matched_steps')=? "
            "AND json_extract(payload_json, '$.metric_version')=? AND json_extract(payload_json, '$.source_as_of') IS ? "
            "AND json_extract(payload_json, '$.recorded_as_of') IS ? ORDER BY rowid DESC LIMIT 1",
            (execution_id, _json(record["actual_ids"]), _json(record["matched_steps"]), _METRIC_VERSION,
             source_as_of, recorded_as_of)).fetchone()
        coverage_basis = "saved_evaluation"
        if previous:
            saved = json.loads(previous[0])
            # Preserve immutable score metadata on retries; enrich legacy scores
            # only when the additive fields did not exist in their release.
            saved.setdefault("complete", status == "complete")
            if "coverage" not in saved:
                coverage_basis = "reconstructed_current_query"
            saved.setdefault("coverage", coverage)
            record = saved
        else:
            conn.execute("INSERT INTO evaluations VALUES (?,?,?,?)",
                         (record["evaluation_id"], execution_id, record["recorded_at"], _json(record)))
        if include_current_coverage:
            return {**record, "current_coverage": coverage,
                    "coverage_basis": coverage_basis, "current_coverage_basis": "current_query",
                    "evaluation_reused": previous is not None}
        if coverage_basis != "saved_evaluation":
            return {**record, "coverage_basis": coverage_basis}
        return record

    def evaluations(self, execution_id: str, *, recorded_as_of: str | None = None) -> list[dict]:
        query, args = "SELECT payload_json FROM evaluations WHERE execution_id=?", [execution_id]
        if recorded_as_of is not None:
            query += " AND recorded_at<=?"
            args.append(_time(recorded_as_of))
        with self._connect() as conn:
            return [json.loads(row[0]) for row in conn.execute(query + " ORDER BY rowid", args)]

    def pending(self, *, source_as_of: str | None = None, recorded_as_of: str | None = None) -> list[dict]:
        """Compatibility read; prefer bounded search for new callers. Includes unscored complete runs."""
        now = self._now()
        source = _time(source_as_of) if source_as_of is not None else now
        recorded = _time(recorded_as_of) if recorded_as_of is not None else now
        rows = []
        with self._connect() as conn:
            conn.execute("BEGIN")
            for row in conn.execute("SELECT execution_id FROM executions WHERE recorded_at<=? ORDER BY rowid", (recorded,)):
                summary = self._summary(conn, self._execution(conn, row[0]), source, recorded)
                if summary["status"] != "scored":
                    rows.append({**summary, "status": "pending" if summary["status"] == "waiting" else summary["status"]})
        return rows

    def _summary(self, conn, run, source_as_of, recorded_as_of):
        req = run["request"]
        summary = {"execution_id": run["execution_id"], "series_id": req["series_id"], "unit": req["unit"],
                   "provider": run["provider"], "revision": run["revision"], "horizon": req["horizon"],
                   "origin": req["cutoff"] or next(iter(req["timestamps"][-1:]), None),
                   "recorded_at": run["recorded_at"], "cache_hit": run["cache_hit"]}
        studies = conn.execute("SELECT study_id FROM study_executions JOIN studies USING(study_id) "
                               "WHERE execution_id=? AND recorded_at<=? ORDER BY study_id LIMIT 21",
                               (run["execution_id"], recorded_as_of)).fetchall()
        summary.update(study_ids=[s[0] for s in studies[:20]], study_ids_truncated=len(studies) > 20)
        try:
            times, pairs = self._pairs(conn, req, source_as_of, recorded_as_of)
        except ForecastAdapterError as exc:
            return {**summary, "status": "unscorable", "score_state": "unscored", "actuals_available": 0,
                    "missing_steps": None, "reason": "timezone_unresolved" if "timezone" in str(exc) else "forecast_identity_missing",
                    "next_step": "record_an_identified_timezone_aware_forecast"}
        steps, actual_ids = [i for i, _ in pairs], [a["actual_id"] for _, a in pairs]
        score = conn.execute("SELECT payload_json FROM evaluations WHERE execution_id=? AND recorded_at<=? "
            "ORDER BY (json_extract(payload_json, '$.actual_ids')=? AND json_extract(payload_json, '$.matched_steps')=? "
            "AND json_extract(payload_json, '$.metric_version')=?) DESC, rowid DESC LIMIT 1",
            (run["execution_id"], recorded_as_of, _json(actual_ids), _json(steps), _METRIC_VERSION)).fetchone()
        score = json.loads(score[0]) if score else None
        current = score and score["actual_ids"] == actual_ids and score["matched_steps"] == steps \
            and score["metric_version"] == _METRIC_VERSION
        state = "current" if current else "stale" if score else "unscored"
        matched = set(steps)
        missing = [i for i in range(len(times)) if i not in matched]
        status = "waiting" if missing else "scored" if current else "stale" if score else "ready"
        return {**summary, "status": status, "score_state": state, "actuals_available": len(pairs),
                "coverage": self._coverage(conn, req, times, pairs, source_as_of, recorded_as_of),
                "missing_steps": missing, "missing_count": len(missing),
                "evaluation_id": score["evaluation_id"] if score else None,
                "mae": score["mae"] if current else None,
                "next_step": "supply_missing_actuals" if missing else "compare_history" if current else
                             "rescore" if score else "evaluate"}

    def search(self, *, series_id: str | None = None, horizon: int | None = None, provider: str | None = None,
               unit: str | None = None, start: str | None = None, end: str | None = None,
               status: str | None = None, source_as_of: str | None = None, recorded_as_of: str | None = None,
               limit: int = 20, cursor: str | None = None) -> dict:
        """Discover short execution summaries. Dates filter local recording time (inclusive).

        Each page examines at most 200 filtered executions; a status filter can
        produce an empty page with a continuation cursor. Cutoffs persist across pages.
        """
        _bounded(limit, "limit", 100)
        if horizon is not None:
            _bounded(horizon, "horizon", 1_000_000)
        for name, value in (("series_id", series_id), ("provider", provider), ("unit", unit)):
            if value is not None and (not isinstance(value, str) or not value):
                raise ForecastAdapterError(f"{name} must be a nonempty string")
        if status is not None and status not in ("waiting", "ready", "scored", "stale", "unscorable"):
            raise ForecastAdapterError("unknown feedback status")
        start, end = _time(start) if start is not None else None, _time(end) if end is not None else None
        if start and end and start > end:
            raise ForecastAdapterError("start must not exceed end")
        source_as_of = _time(source_as_of) if source_as_of is not None else None
        recorded_as_of = _time(recorded_as_of) if recorded_as_of is not None else None
        identity = hashlib.sha256(_json([series_id, horizon, provider, unit, start, end, status,
                                        source_as_of, recorded_as_of]).encode()).hexdigest()
        with self._connect() as conn:
            conn.execute("BEGIN")
            if cursor is not None:
                try:
                    if not isinstance(cursor, str) or len(cursor) > 2048:
                        raise ValueError
                    token = json.loads(base64.urlsafe_b64decode(cursor.encode()))
                    after, upper = token["after"], token["upper"]
                    if token["query"] != identity or type(after) is not int or type(upper) is not int \
                            or not 0 <= after <= upper <= 2**63 - 1:
                        raise ValueError
                    source, recorded = _time(token["source"]), _time(token["recorded"])
                except (ValueError, TypeError, KeyError, UnicodeError):
                    raise ForecastAdapterError("invalid cursor or changed search filters") from None
            else:
                after = 0
                upper = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM executions").fetchone()[0]
                now = self._now()
                source, recorded = source_as_of or now, recorded_as_of or now
            query = "SELECT e.rowid, e.execution_id FROM executions e JOIN payloads p USING(payload_id) " \
                    "WHERE e.rowid>? AND e.rowid<=? AND e.recorded_at<=?"
            args = [after, upper, recorded]
            for path, value in (("request.series_id", series_id), ("request.horizon", horizon),
                                ("provider", provider), ("request.unit", unit)):
                if value is not None:
                    query += f" AND json_extract(p.payload_json, '$.{path}')=?"
                    args.append(value)
            for op, value in ((">=", start), ("<=", end)):
                if value is not None:
                    query += f" AND e.recorded_at{op}?"
                    args.append(value)
            candidates = conn.execute(query + " ORDER BY e.rowid LIMIT 201", args).fetchall()
            items, scanned = [], 0
            for row in candidates[:200]:
                after = row[0]
                scanned += 1
                summary = self._summary(conn, self._execution(conn, row[1]), source, recorded)
                if summary["missing_steps"] is not None:
                    summary["missing_steps"] = summary["missing_steps"][:20]
                    coverage = summary["coverage"]
                    for field in ("missing_steps", "missing_timestamps", "other_units_at_missing_steps"):
                        coverage[field + "_truncated"] = len(coverage[field]) > 20
                        coverage[field] = coverage[field][:20]
                if status is None or summary["status"] == status:
                    items.append(summary)
                if len(items) == limit:
                    break
            more = len(candidates) > scanned
            token = {"query": identity, "after": after, "upper": upper, "source": source, "recorded": recorded}
            return {"items": items, "next_cursor": base64.urlsafe_b64encode(_json(token).encode()).decode() if more else None,
                    "scanned": scanned, "source_as_of": source, "recorded_as_of": recorded}

    def compare(self, execution_ids: list[str], *, source_as_of: str | None = None,
                recorded_as_of: str | None = None) -> dict:
        if not isinstance(execution_ids, list) or len(execution_ids) > 100 or any(
                not isinstance(eid, str) or not eid for eid in execution_ids):
            raise ForecastAdapterError("comparison requires a list of at most 100 nonempty execution IDs")
        if len(execution_ids) < 2 or len(set(execution_ids)) != len(execution_ids):
            raise ForecastAdapterError("comparison requires at least two distinct executions")
        with self._connect() as conn:
            conn.execute("BEGIN")
            runs = [self._execution(conn, eid) for eid in execution_ids]
            return self._compare(conn, runs, source_as_of, recorded_as_of)

    def _compare(self, conn, runs, source_as_of, recorded_as_of):
        if recorded_as_of is not None and any(r["recorded_at"] > _time(recorded_as_of) for r in runs):
            raise ForecastAdapterError("execution was not recorded by the requested cutoff")
        if any(not run["request"].get("history") or run["result"].get("metadata", {}).get("complete_request_reconstructed") is False
               for run in runs):
            raise ForecastAdapterError("matched comparison requires preserved complete input history and request")
        if any(any(run["request"].get(key) != runs[0]["request"].get(key) for key in _MATCH_KEYS) for run in runs[1:]):
            raise ForecastAdapterError("comparison requires matched inputs, snapshot and forecast origin")
        # Resolve once to the same vintages for every candidate; evaluating
        # separately while ingestion proceeds could otherwise change the cohort.
        req = runs[0]["request"]
        if not req["series_id"] or not req["future_timestamps"]:
            raise ForecastAdapterError("comparison requires identified forecast timestamps")
        _, pairs = self._pairs(conn, req, source_as_of, recorded_as_of)
        return {"n": len(pairs), "horizon": req["horizon"], "metric_version": _METRIC_VERSION,
                "source_as_of": _time(source_as_of) if source_as_of else None,
                "recorded_as_of": _time(recorded_as_of) if recorded_as_of else None,
                "snapshot_id": req["snapshot_id"],
                "actual_ids": [a["actual_id"] for _, a in pairs], "matched_steps": [i for i, _ in pairs],
                "models": [{"execution_id": run["execution_id"], "provider": run["provider"], "revision": run["revision"],
                            "mae": point_error_metrics([(run["result"]["point"][i], a["value"]) for i, a in pairs])["mae"]}
                           for run in runs]}

    def compare_history(self, *, series_id: str, horizon: int, providers: dict[str, str],
                        start: str, end: str, source_as_of: str, recorded_as_of: str,
                        unit: str | None = None) -> dict:
        """Read matched production evidence over an explicit inclusive origin window; no routing or writes."""
        from .ledger_history import compare_history
        return compare_history(self, series_id=series_id, horizon=horizon, providers=providers,
                               start=start, end=end, source_as_of=source_as_of, recorded_as_of=recorded_as_of, unit=unit)

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

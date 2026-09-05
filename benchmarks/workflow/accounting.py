"""Append-only attempt receipts for benchmark retries, stages and checkpoint resume.

Not a forecasting ledger. Started-but-unfinished work has unknown usage. Numeric
Observation scalars remain compatibility lower bounds; only these totals indicate
complete accounting. Provider-reported usage is supplied evidence, not a billing audit.
"""

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
import uuid

FIELDS = ("tool_calls", "cumulative_tokens", "response_tokens", "latency_seconds", "cost_usd")
APP_ID = 0x474E4154


def reported_cost_limit(budget):
    """Optional stop-after-reporting threshold, never a prepaid dollar ceiling."""
    if not isinstance(budget, dict):
        raise ValueError("budget must be an object")
    if "max_reported_cost_usd" not in budget:
        return None
    value = budget["max_reported_cost_usd"]
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("max_reported_cost_usd must be finite and positive")
    return value


def receipt(observation, stage="submitted", *, historical=False):
    known = set(observation.metadata.get("resource_fields", ()))
    if historical or observation.metadata.get("retries_used", 0):
        known.clear()
    values = {key: (getattr(observation, key, None) if key in known or getattr(observation, key, 0) else None)
              for key in FIELDS}
    digest = hashlib.sha256(json.dumps(asdict(observation), sort_keys=True, allow_nan=False).encode()).hexdigest()
    budget = observation.metadata.get("budget_exceeded")
    if budget is not None and type(budget) is not bool:
        raise ValueError("attempt budget measurement must be boolean or unknown")
    if historical or observation.metadata.get("retries_used", 0):
        budget = True if budget is True else None
    return {"attempt_id": "submitted_" + digest, "case_id": observation.case_id, "stage": stage,
            "status": observation.status, "resources": values, "complete_fields": sorted(known),
            "budget_exceeded": budget,
            "temporal_leakage": (True if observation.temporal_leakage is True else None
                                 if historical or observation.metadata.get("retries_used", 0) else observation.temporal_leakage),
            "error_code": (str(observation.metadata.get("error") or "unspecified")[:128]
                           if observation.status == "error" else None),
            "provenance": "historical_partial" if historical else "supplied_usage_not_billing_attestation"}


def receipts(observation):
    items = observation.metadata.get("attempt_receipts") or [receipt(observation)]
    if any(item["case_id"] != observation.case_id for item in items):
        raise ValueError("attempt receipt belongs to a different case")
    return items


def summarize(items):
    by_id = {}
    for item in items:
        required = {"attempt_id", "case_id", "stage", "status", "resources", "complete_fields"}
        if not isinstance(item, dict) or not required <= item.keys():
            raise ValueError("malformed attempt receipt")
        if any(not isinstance(item[key], str) or not item[key] for key in ("attempt_id", "case_id", "stage")):
            raise ValueError("attempt identity and scope must be nonempty strings")
        if item["status"] not in ("answered", "abstained", "error", "unfinished"):
            raise ValueError("unknown attempt status")
        if not isinstance(item["resources"], dict) or set(item["resources"]) - set(FIELDS):
            raise ValueError("invalid attempt resource names")
        known = item["complete_fields"]
        if not isinstance(known, list) or any(key not in FIELDS or item["resources"].get(key) is None for key in known):
            raise ValueError("complete resource fields require measured values")
        if item["status"] == "unfinished" and known:
            raise ValueError("unfinished attempts cannot attest final totals")
        if item.get("temporal_leakage") is not None and type(item["temporal_leakage"]) is not bool:
            raise ValueError("attempt leakage measurement must be boolean or unknown")
        if item.get("budget_exceeded") is not None and type(item["budget_exceeded"]) is not bool:
            raise ValueError("attempt budget measurement must be boolean or unknown")
        key = item["attempt_id"]
        if key in by_id and by_id[key] != item:
            raise ValueError("conflicting attempt receipt")
        by_id[key] = item
    resources = {}
    for key in FIELDS:
        values = [item["resources"].get(key) for item in by_id.values()]
        measured = [value for value in values if value is not None]
        for value in measured:
            try:
                valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError("attempt resources must be finite and nonnegative")
            if key in FIELDS[:3] and type(value) is not int:
                raise ValueError("attempt token/call counts must be integers")
        complete = bool(by_id) and all(item["status"] != "unfinished" and key in item["complete_fields"] and value is not None
                                      for item, value in zip(by_id.values(), values))
        try:
            total = sum(measured) if key in FIELDS[:3] else math.fsum(measured)
            if not math.isfinite(total):
                raise OverflowError
        except OverflowError:
            raise ValueError("attempt resource sum exceeds finite range") from None
        resources[key] = {"observed_total": total if measured else None,
                          "total": total if complete else None,
                          "unmeasured_attempts": sum(key not in item["complete_fields"] or item["resources"].get(key) is None for item in by_id.values()),
                          "complete": complete}
    return {"attempts": len(by_id), "unfinished": sum(item["status"] == "unfinished" for item in by_id.values()),
            "budget_exceeded": (True if any(item.get("budget_exceeded") is True for item in by_id.values())
                                else False if by_id and all(item["status"] != "unfinished" and item.get("budget_exceeded") is False
                                                            for item in by_id.values()) else None),
            "failed_attempts": sum(item["status"] == "error" for item in by_id.values()),
            "leaking_attempts": sum(item.get("temporal_leakage") is True for item in by_id.values()),
            "leakage_unmeasured_attempts": sum(item.get("temporal_leakage") is None for item in by_id.values()),
            "resources": resources, "budget_accounting_complete": all(resources[key]["complete"] for key in FIELDS[:2])}


def attach(observation, items):
    if any(item["case_id"] != observation.case_id for item in items):
        raise ValueError("attempt receipt belongs to a different case")
    by_id = {item["attempt_id"]: item for item in items}
    accounting = summarize(items)
    # Preserve scalar compatibility, but explicitly identify unknown totals below.
    values = {key: accounting["resources"][key]["observed_total"] or 0 for key in FIELDS[:4]}
    values["cost_usd"] = accounting["resources"]["cost_usd"]["total"]
    leakage = (True if accounting["leaking_attempts"] else None
               if accounting["leakage_unmeasured_attempts"] or not items else False)
    return replace(observation, **values, temporal_leakage=leakage, metadata={**observation.metadata,
                   "attempt_receipts": list(by_id.values()), "resource_accounting": accounting})


class AttemptJournal:
    """One SQLite event stream; durable start precedes external invocation."""

    def __init__(self, path: Path | None = None):
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(path) if path is not None else ":memory:", timeout=30, check_same_thread=False)
        try:
            app = self.db.execute("PRAGMA application_id").fetchone()[0]
            version = self.db.execute("PRAGMA user_version").fetchone()[0]
            tables = self.db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if app != APP_ID and (app != 0 or tables):
                raise ValueError("not a Workflow Bench attempt journal")
            if version not in (0, 1, 2):
                raise ValueError("unsupported attempt journal version")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute(f"PRAGMA application_id={APP_ID}")
            self.db.execute("PRAGMA user_version=2")
            self.db.execute("CREATE TABLE IF NOT EXISTS attempts (id TEXT NOT NULL, phase TEXT NOT NULL CHECK(phase IN ('started','finished')), case_id TEXT NOT NULL, stage TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(id,phase))")
            self.db.execute("CREATE TABLE IF NOT EXISTS checkpoints (attempt_id TEXT NOT NULL, sequence INTEGER NOT NULL, case_id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(attempt_id,sequence))")
            for operation in ("UPDATE", "DELETE"):
                self.db.execute(f"CREATE TRIGGER IF NOT EXISTS no_{operation.lower()} BEFORE {operation} ON attempts BEGIN SELECT RAISE(ABORT,'append-only attempts'); END")
                self.db.execute(f"CREATE TRIGGER IF NOT EXISTS checkpoints_no_{operation.lower()} BEFORE {operation} ON checkpoints BEGIN SELECT RAISE(ABORT,'append-only checkpoints'); END")
            self.db.commit()
        except Exception:
            self.db.close()
            raise

    def _append(self, identifier, phase, case_id, stage, payload):
        with self.lock, self.db:
            self.db.execute("INSERT INTO attempts VALUES(?,?,?,?,?)",
                            (identifier, phase, case_id, stage, json.dumps(payload, sort_keys=True, allow_nan=False)))

    def start(self, case_id, stage):
        identifier = uuid.uuid4().hex
        self._append(identifier, "started", case_id, stage, {})
        return identifier

    def finish(self, identifier, item):
        summarize([item])
        with self.lock:
            start = self.db.execute("SELECT case_id,stage FROM attempts WHERE id=? AND phase='started'", (identifier,)).fetchone()
            if start != (item["case_id"], item["stage"]):
                raise ValueError("attempt completion does not match its durable start")
            self._append(identifier, "finished", *start, {**item, "attempt_id": identifier})

    def checkpoint(self, identifier, case_id, record):
        """Commit the unchanged agent submission before any next-phase reveal."""
        sequence = record["sequence"]
        if type(sequence) is not int or not 0 <= sequence < 8:
            raise ValueError("invalid episode checkpoint sequence")
        payload = json.dumps(record, sort_keys=True, allow_nan=False)
        if len(payload.encode()) > 1_048_576:
            raise ValueError("checkpoint exceeds byte limit")
        with self.lock, self.db:
            started = self.db.execute("SELECT case_id FROM attempts WHERE id=? AND phase='started'", (identifier,)).fetchone()
            finished = self.db.execute("SELECT 1 FROM attempts WHERE id=? AND phase='finished'", (identifier,)).fetchone()
            count = self.db.execute("SELECT count(*) FROM checkpoints WHERE attempt_id=?", (identifier,)).fetchone()[0]
            if started != (case_id,) or finished or count != sequence:
                raise ValueError("checkpoint does not belong to the next phase of an active attempt")
            self.db.execute("INSERT INTO checkpoints VALUES(?,?,?,?)", (identifier, sequence, case_id, payload))

    def checkpoints(self, identifier):
        with self.lock:
            rows = self.db.execute("SELECT payload FROM checkpoints WHERE attempt_id=? ORDER BY sequence", (identifier,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def import_receipt(self, item):
        identifier = item["attempt_id"]
        with self.lock:
            existing = self.db.execute("SELECT payload FROM attempts WHERE id=? AND phase='finished'", (identifier,)).fetchone()
            if existing:
                if json.loads(existing[0]) != item:
                    raise ValueError("conflicting imported attempt")
                return
            start = self.db.execute("SELECT 1 FROM attempts WHERE id=? AND phase='started'", (identifier,)).fetchone()
            if not start:
                self._append(identifier, "started", item["case_id"], item["stage"], {})
            if item["status"] != "unfinished":
                self.finish(identifier, item)

    def for_case(self, case_id):
        with self.lock:
            rows = self.db.execute("SELECT s.id,s.stage,f.payload FROM attempts s LEFT JOIN attempts f ON s.id=f.id AND f.phase='finished' WHERE s.phase='started' AND s.case_id=? ORDER BY s.rowid", (case_id,)).fetchall()
        return [json.loads(payload) if payload else {"attempt_id": identifier, "case_id": case_id,
                "stage": stage, "status": "unfinished", "resources": {key: None for key in FIELDS},
                **({"resources": self.checkpoints(identifier)[-1]["usage"]} if self.checkpoints(identifier) else {}),
                "complete_fields": [], "provenance": "durable_start_without_completion"}
                for identifier, stage, payload in rows]

    def reported_spend(self):
        """All retained attempts, including failures and cases outside this batch."""
        with self.lock:
            cases = [row[0] for row in self.db.execute("SELECT DISTINCT case_id FROM attempts")]
            items = [item for case_id in cases for item in self.for_case(case_id)]
        if not items:
            return {"observed_total": 0, "total": 0, "complete": True, "unmeasured_attempts": 0}
        return summarize(items)["resources"]["cost_usd"]

    def close(self):
        self.db.close()

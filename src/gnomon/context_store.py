"""Persistent, content-addressed context and assessment caches.

Receipts cache interpretation, never authority. Numeric assessments use a
separate key that binds the exact visible-data snapshot and execution policy.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any

from .soft_context import make_context_receipt


STORE_SCHEMA_VERSION = "0.1"
_CACHE_METRICS = {
    "receipt_hits": 0, "receipt_misses": 0,
    "assessment_hits": 0, "assessment_misses": 0,
    "temporal_answer_hits": 0, "temporal_answer_misses": 0,
}


def cache_metrics() -> dict[str, int]:
    return dict(_CACHE_METRICS)


def reset_cache_metrics() -> None:
    for key in _CACHE_METRICS:
        _CACHE_METRICS[key] = 0


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def context_cache_key(
    *, document_fingerprints: list[str], available_targets: list[str],
    horizon: int | None, frequency: str | None, compiler: dict[str, Any],
    prompt_version: str, receipt_schema_version: str,
) -> str:
    """Identity of an interpretation request, excluding its output."""
    return "context_key:" + _digest({
        "documents": sorted(document_fingerprints),
        "available_targets": sorted(set(available_targets)),
        "horizon": horizon,
        "frequency": frequency,
        "compiler": compiler,
        "prompt_version": prompt_version,
        "receipt_schema_version": receipt_schema_version,
    })


def assessment_cache_key(
    *, receipt_id: str, visible_data_fingerprint: str, as_of: str | None,
    series: str, fitted_candidate_id: str, configuration_fingerprint: str,
    code_version: str,
) -> str:
    """Bind cached numeric evidence to the exact knowable execution state."""
    return "context_assessment_key:" + _digest({
        "receipt_id": receipt_id,
        "visible_data_fingerprint": visible_data_fingerprint,
        "as_of": as_of,
        "series": series,
        "fitted_candidate_id": fitted_candidate_id,
        "configuration_fingerprint": configuration_fingerprint,
        "code_version": code_version,
    })


def temporal_answer_cache_key(
    *, artifact_id: str, question: dict[str, Any], as_of: str | None,
    answer_contract_version: str,
) -> str:
    """Bind a typed answer to its immutable primary and exact semantics."""
    return "temporal_answer_key:" + _digest({
        "artifact_id": artifact_id,
        "question": question,
        "as_of": as_of,
        "answer_contract_version": answer_contract_version,
    })


class ContextReceiptStore:
    """Immutable JSON blobs with a namespaced SQLite lookup index."""

    def __init__(self, root: str | Path, *, namespace: str = "local") -> None:
        self.root = Path(root)
        self.namespace = str(namespace).strip() or "local"
        self.blobs = self.root / "receipts"
        self.assessments = self.root / "assessments"
        self.temporal_answers = self.root / "temporal-answers"
        self.root.mkdir(parents=True, exist_ok=True)
        self.blobs.mkdir(exist_ok=True)
        self.assessments.mkdir(exist_ok=True)
        self.temporal_answers.mkdir(exist_ok=True)
        self.database = self.root / "index.sqlite3"
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS context_receipts (
                    namespace TEXT NOT NULL,
                    cache_key TEXT,
                    receipt_id TEXT NOT NULL,
                    context_ref TEXT NOT NULL,
                    PRIMARY KEY (namespace, receipt_id),
                    UNIQUE (namespace, context_ref)
                )
            """)
            connection.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS context_cache_lookup
                ON context_receipts(namespace, cache_key)
                WHERE cache_key IS NOT NULL
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS context_lookups (
                    namespace TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    receipt_id TEXT NOT NULL,
                    PRIMARY KEY (namespace, cache_key)
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS context_assessments (
                    namespace TEXT NOT NULL,
                    assessment_key TEXT NOT NULL,
                    blob_id TEXT NOT NULL,
                    PRIMARY KEY (namespace, assessment_key)
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS temporal_answers (
                    namespace TEXT NOT NULL,
                    answer_key TEXT NOT NULL,
                    blob_id TEXT NOT NULL,
                    PRIMARY KEY (namespace, answer_key)
                )
            """)

    @classmethod
    def default(cls) -> "ContextReceiptStore":
        root = os.environ.get("GNOMON_CONTEXT_STORE")
        if not root:
            root = str(Path.cwd() / ".gnomon" / "context-store")
        return cls(root, namespace=os.environ.get(
            "GNOMON_CONTEXT_NAMESPACE", "local"))

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    @staticmethod
    def verify(receipt: dict[str, Any]) -> bool:
        expected = make_context_receipt(
            documents=list(receipt.get("documents") or []),
            events=list(receipt.get("events") or []),
            hypotheses=list(receipt.get("hypotheses") or []),
            rejected=list(receipt.get("rejected") or []),
            rejected_hypotheses=list(receipt.get("rejected_hypotheses") or []),
            proposer=dict(receipt.get("compiler") or {}),
        )
        return receipt == expected

    def put(self, receipt: dict[str, Any], *, cache_key: str | None = None) -> str:
        if not self.verify(receipt):
            raise ValueError("context receipt ID or content is invalid")
        receipt_id = str(receipt["receipt_id"])
        digest = receipt_id.split(":", 1)[-1]
        context_ref = "context_" + digest
        path = self.blobs / f"{digest}.json"
        rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != rendered:
            raise ValueError("immutable context receipt differs from stored blob")
        if not path.exists():
            path.write_text(rendered, encoding="utf-8")
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT receipt_id FROM context_lookups "
                "WHERE namespace = ? AND cache_key = ?",
                (self.namespace, cache_key),
            ).fetchone() if cache_key is not None else None
            if existing is not None and existing[0] != receipt_id:
                raise ValueError("cache key already identifies another receipt")
            connection.execute(
                "INSERT OR IGNORE INTO context_receipts "
                "(namespace, cache_key, receipt_id, context_ref) VALUES (?, ?, ?, ?)",
                (self.namespace, cache_key, receipt_id, context_ref),
            )
            if cache_key is not None:
                connection.execute(
                    "INSERT OR IGNORE INTO context_lookups "
                    "(namespace, cache_key, receipt_id) VALUES (?, ?, ?)",
                    (self.namespace, cache_key, receipt_id),
                )
        return context_ref

    def get(self, context_ref: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT receipt_id FROM context_receipts "
                "WHERE namespace = ? AND context_ref = ?",
                (self.namespace, str(context_ref)),
            ).fetchone()
        if row is None:
            _CACHE_METRICS["receipt_misses"] += 1
            raise KeyError(str(context_ref))
        path = self.blobs / f"{row[0].split(':', 1)[-1]}.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if not self.verify(receipt):
            raise ValueError("stored context receipt failed integrity verification")
        _CACHE_METRICS["receipt_hits"] += 1
        return receipt

    def find(self, cache_key: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT r.context_ref FROM context_lookups AS l "
                "JOIN context_receipts AS r ON "
                "r.namespace = l.namespace AND r.receipt_id = l.receipt_id "
                "WHERE l.namespace = ? AND l.cache_key = ?",
                (self.namespace, cache_key),
            ).fetchone()
        return self.get(row[0]) if row else None

    def put_assessment(self, key: str, assessment: dict[str, Any]) -> str:
        blob_id = _digest({"key": key, "assessment": assessment})
        path = self.assessments / f"{blob_id}.json"
        rendered = json.dumps(assessment, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != rendered:
            raise ValueError("immutable assessment differs from stored blob")
        if not path.exists():
            path.write_text(rendered, encoding="utf-8")
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT blob_id FROM context_assessments "
                "WHERE namespace = ? AND assessment_key = ?",
                (self.namespace, key),
            ).fetchone()
            if existing is not None and existing[0] != blob_id:
                raise ValueError("assessment key already identifies another result")
            connection.execute(
                "INSERT OR IGNORE INTO context_assessments "
                "(namespace, assessment_key, blob_id) VALUES (?, ?, ?)",
                (self.namespace, key, blob_id),
            )
        return "context_assessment_" + blob_id

    def get_assessment(self, key: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT blob_id FROM context_assessments "
                "WHERE namespace = ? AND assessment_key = ?",
                (self.namespace, key),
            ).fetchone()
        if row is None:
            _CACHE_METRICS["assessment_misses"] += 1
            return None
        assessment = json.loads((self.assessments / f"{row[0]}.json").read_text(
            encoding="utf-8"))
        if _digest({"key": key, "assessment": assessment}) != row[0]:
            raise ValueError("stored context assessment failed integrity verification")
        _CACHE_METRICS["assessment_hits"] += 1
        return assessment

    def put_temporal_answer(self, key: str, answer: dict[str, Any]) -> str:
        blob_id = _digest({"key": key, "answer": answer})
        path = self.temporal_answers / f"{blob_id}.json"
        rendered = json.dumps(answer, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != rendered:
            raise ValueError("immutable temporal answer differs from stored blob")
        if not path.exists():
            path.write_text(rendered, encoding="utf-8")
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT blob_id FROM temporal_answers "
                "WHERE namespace = ? AND answer_key = ?",
                (self.namespace, key),
            ).fetchone()
            if existing is not None and existing[0] != blob_id:
                raise ValueError("temporal answer key identifies another result")
            connection.execute(
                "INSERT OR IGNORE INTO temporal_answers "
                "(namespace, answer_key, blob_id) VALUES (?, ?, ?)",
                (self.namespace, key, blob_id),
            )
        return "temporal_answer_" + blob_id

    def get_temporal_answer(self, key: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT blob_id FROM temporal_answers "
                "WHERE namespace = ? AND answer_key = ?",
                (self.namespace, key),
            ).fetchone()
        if row is None:
            _CACHE_METRICS["temporal_answer_misses"] += 1
            return None
        answer = json.loads((self.temporal_answers / f"{row[0]}.json").read_text(
            encoding="utf-8"))
        if _digest({"key": key, "answer": answer}) != row[0]:
            raise ValueError("stored temporal answer failed integrity verification")
        _CACHE_METRICS["temporal_answer_hits"] += 1
        return answer

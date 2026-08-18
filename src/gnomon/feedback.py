"""Local-first, consent-gated feedback receipts for Gnomon.

This module deliberately does not observe Gnomon calls. A user or agent must
create a receipt explicitly, and no receipt leaves the machine until the exact
shareable payload has been previewed and export/submission is separately
authorized.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest


SCHEMA_VERSION = "1.0"
BUNDLE_SCHEMA_VERSION = "1.0"
DEFAULT_STORE = Path(".gnomon") / "feedback"

CATEGORIES = frozenset({
    "wrong_answer",
    "confusing_answer",
    "unnecessary_calls",
    "abstention",
    "routing",
    "tool_error",
    "missing_capability",
    "positive",
})

_STRING_CONTEXT_FIELDS = frozenset({
    "surface",
    "verb",
    "task_kind",
    "support_tier",
    "status",
    "error_code",
    "frequency",
})
_INTEGER_CONTEXT_FIELDS = frozenset({
    "call_count",
    "minimum_call_count",
    "repair_count",
    "series_count",
    "horizon",
})
_BOOLEAN_CONTEXT_FIELDS = frozenset({"corrected", "resolved"})
_CONTEXT_FIELDS = (
    _STRING_CONTEXT_FIELDS | _INTEGER_CONTEXT_FIELDS | _BOOLEAN_CONTEXT_FIELDS
)
_RECEIPT_ID_RE = re.compile(r"^fb_[0-9a-f]{32}$")

_REDACTIONS = (
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [REDACTED]"),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|access[_-]?token|token|secret|password)"
            r"\s*[:=]\s*[^\s,;]+"
        ),
        r"\1=[REDACTED]",
    ),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"https?://[^\s]+", re.IGNORECASE), "[URL]"),
    (re.compile(r"(?<![A-Za-z0-9])/(?:[^\s,;]+/)*[^\s,;]*"), "[PATH]"),
    (re.compile(r"\b[A-Za-z]:\\(?:[^\s,;]+\\)*[^\s,;]*"), "[PATH]"),
    (re.compile(r"\b(?:[A-Fa-f0-9]{32,}|[A-Za-z0-9+/]{40,}={0,2})\b"), "[TOKEN]"),
)


class FeedbackError(ValueError):
    """Raised when a receipt would violate the feedback contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def redact_public_text(text: str | None) -> str | None:
    """Return bounded, best-effort-redacted text suitable for preview.

    Preview remains the final authority: free text can contain unexpected
    sensitive material, so callers must inspect it before sharing.
    """

    if text is None:
        return None
    if not isinstance(text, str):
        raise FeedbackError("public_summary must be a string")
    cleaned = " ".join(text.split())
    for pattern, replacement in _REDACTIONS:
        cleaned = pattern.sub(replacement, cleaned)
    if len(cleaned) > 500:
        cleaned = cleaned[:497] + "..."
    return cleaned or None


def _private_note(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise FeedbackError("private_note must be a string")
    if len(value) > 10_000:
        raise FeedbackError("private_note exceeds the 10,000 character local limit")
    return value or None


def normalize_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate the deliberately small, non-content feedback vocabulary."""

    if context is None:
        return {}
    unknown = set(context) - _CONTEXT_FIELDS
    if unknown:
        raise FeedbackError(
            "unknown context fields: " + ", ".join(sorted(unknown))
        )
    normalized: dict[str, Any] = {}
    for key, value in context.items():
        if value is None:
            continue
        if key in _STRING_CONTEXT_FIELDS:
            if not isinstance(value, str) or not value.strip():
                raise FeedbackError(f"{key} must be a non-empty string")
            value = value.strip()
            if len(value) > 80:
                raise FeedbackError(f"{key} exceeds 80 characters")
        elif key in _INTEGER_CONTEXT_FIELDS:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise FeedbackError(f"{key} must be a non-negative integer")
        elif key in _BOOLEAN_CONTEXT_FIELDS:
            if not isinstance(value, bool):
                raise FeedbackError(f"{key} must be boolean")
        normalized[key] = value
    return dict(sorted(normalized.items()))


@dataclass(frozen=True)
class FeedbackReceipt:
    """A private local receipt with a strictly smaller shareable projection."""

    receipt_id: str
    created_at: str
    category: str
    context: dict[str, Any]
    public_summary: str | None
    private_note: str | None
    artifact_id: str | None
    artifact_digest: str | None
    reproduction_hash: str
    claim_secret: str
    claim_digest: str

    @property
    def has_reproduction_evidence(self) -> bool:
        return bool(self.artifact_digest or self.context.get("error_code"))

    def shareable(self) -> dict[str, Any]:
        """Return the exact payload eligible for export or submission."""

        eligibility = (
            "pending_independent_verification"
            if self.has_reproduction_evidence
            else "insufficient_reproduction_evidence"
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "receipt_id": self.receipt_id,
            "created_at": self.created_at,
            "category": self.category,
            "context": dict(self.context),
            "public_summary": self.public_summary,
            "evidence": {
                "artifact_digest": self.artifact_digest,
                "reproduction_hash": self.reproduction_hash,
            },
            "incentive": {
                "eligibility": eligibility,
                "claim_digest": self.claim_digest,
                "rules_version": "1",
                "requires": ["confirmed", "unique", "actionable"],
                "usage_volume_rewarded": False,
            },
        }

    def local_dict(self) -> dict[str, Any]:
        return {
            **self.shareable(),
            "private": {
                "private_note": self.private_note,
                "artifact_id": self.artifact_id,
                "claim_secret": self.claim_secret,
            },
        }

    @classmethod
    def from_local_dict(cls, payload: Mapping[str, Any]) -> "FeedbackReceipt":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise FeedbackError("unsupported local feedback schema version")
        try:
            private = payload["private"]
            evidence = payload["evidence"]
            incentive = payload["incentive"]
            receipt = cls(
                receipt_id=str(payload["receipt_id"]),
                created_at=str(payload["created_at"]),
                category=str(payload["category"]),
                context=normalize_context(payload.get("context", {})),
                public_summary=redact_public_text(payload.get("public_summary")),
                private_note=_private_note(private.get("private_note")),
                artifact_id=private.get("artifact_id"),
                artifact_digest=evidence.get("artifact_digest"),
                reproduction_hash=str(evidence["reproduction_hash"]),
                claim_secret=str(private["claim_secret"]),
                claim_digest=str(incentive["claim_digest"]),
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise FeedbackError("malformed local feedback receipt") from exc
        if not _RECEIPT_ID_RE.fullmatch(receipt.receipt_id):
            raise FeedbackError("invalid receipt_id")
        if receipt.category not in CATEGORIES:
            raise FeedbackError(f"unknown feedback category {receipt.category!r}")
        if receipt.artifact_id is not None and not isinstance(receipt.artifact_id, str):
            raise FeedbackError("malformed local artifact_id")
        expected_artifact_digest = (
            _digest(receipt.artifact_id) if receipt.artifact_id else None
        )
        if expected_artifact_digest != receipt.artifact_digest:
            raise FeedbackError("artifact digest does not match local artifact ID")
        expected_reproduction_hash = _digest({
            "category": receipt.category,
            "context": receipt.context,
            "public_summary": receipt.public_summary,
            "artifact_digest": receipt.artifact_digest,
        })
        if expected_reproduction_hash != receipt.reproduction_hash:
            raise FeedbackError("reproduction hash does not match local receipt")
        if _digest(receipt.claim_secret) != receipt.claim_digest:
            raise FeedbackError("claim digest does not match local claim secret")
        return receipt


def create_receipt(
    category: str,
    *,
    context: Mapping[str, Any] | None = None,
    public_summary: str | None = None,
    private_note: str | None = None,
    artifact_id: str | None = None,
) -> FeedbackReceipt:
    """Create an in-memory receipt without recording or transmitting it."""

    if category not in CATEGORIES:
        raise FeedbackError(
            f"unknown category {category!r}; expected one of {sorted(CATEGORIES)}"
        )
    normalized_context = normalize_context(context)
    public_summary = redact_public_text(public_summary)
    private_note = _private_note(private_note)
    if artifact_id is not None:
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise FeedbackError("artifact_id must be a non-empty string")
        if len(artifact_id) > 500:
            raise FeedbackError("artifact_id exceeds 500 characters")
        artifact_id = artifact_id.strip()
    artifact_digest = _digest(artifact_id) if artifact_id else None
    reproduction_hash = _digest({
        "category": category,
        "context": normalized_context,
        "public_summary": public_summary,
        "artifact_digest": artifact_digest,
    })
    claim_secret = secrets.token_urlsafe(32)
    return FeedbackReceipt(
        receipt_id="fb_" + secrets.token_hex(16),
        created_at=_utc_now(),
        category=category,
        context=normalized_context,
        public_summary=public_summary,
        private_note=private_note,
        artifact_id=artifact_id,
        artifact_digest=artifact_digest,
        reproduction_hash=reproduction_hash,
        claim_secret=claim_secret,
        claim_digest=_digest(claim_secret),
    )


class FeedbackStore:
    """One private JSON file per receipt, safe for independent writers."""

    def __init__(self, root: str | Path = DEFAULT_STORE) -> None:
        self.root = Path(root)

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.root.chmod(0o700)
        except OSError:
            pass

    def _path(self, receipt_id: str) -> Path:
        if not _RECEIPT_ID_RE.fullmatch(receipt_id):
            raise FeedbackError("invalid receipt_id")
        return self.root / f"{receipt_id}.json"

    def record(self, receipt: FeedbackReceipt) -> Path:
        self._ensure_root()
        destination = self._path(receipt.receipt_id)
        if destination.exists():
            raise FeedbackError(f"receipt already exists: {receipt.receipt_id}")
        fd, temporary_name = tempfile.mkstemp(
            dir=self.root, prefix=".feedback-", suffix=".tmp"
        )
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(receipt.local_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise
        return destination

    def get(self, receipt_id: str) -> FeedbackReceipt:
        path = self._path(receipt_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FeedbackError(f"unknown receipt: {receipt_id}") from exc
        except json.JSONDecodeError as exc:
            raise FeedbackError(f"corrupt receipt: {receipt_id}") from exc
        return FeedbackReceipt.from_local_dict(payload)

    def receipts(self) -> list[FeedbackReceipt]:
        if not self.root.exists():
            return []
        result: list[FeedbackReceipt] = []
        for path in sorted(self.root.glob("fb_*.json")):
            result.append(self.get(path.stem))
        return result


def summarize_receipts(receipts: Iterable[FeedbackReceipt]) -> dict[str, Any]:
    """Aggregate behavior without returning notes, identifiers, or artifacts."""

    rows = list(receipts)
    categories = Counter(row.category for row in rows)
    surfaces = Counter(row.context.get("surface", "unknown") for row in rows)
    task_kinds = Counter(row.context.get("task_kind", "unknown") for row in rows)
    calls = [row.context["call_count"] for row in rows if "call_count" in row.context]
    redundant = [
        max(0, row.context["call_count"] - row.context["minimum_call_count"])
        for row in rows
        if "call_count" in row.context and "minimum_call_count" in row.context
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "receipt_count": len(rows),
        "by_category": dict(sorted(categories.items())),
        "by_surface": dict(sorted(surfaces.items())),
        "by_task_kind": dict(sorted(task_kinds.items())),
        "call_behavior": {
            "observations": len(calls),
            "total_calls": sum(calls),
            "redundant_call_observations": len(redundant),
            "total_redundant_calls": sum(redundant),
        },
        "privacy": {
            "contains_free_text": False,
            "contains_receipt_ids": False,
            "contains_artifact_ids": False,
        },
    }


def make_bundle(receipts: Sequence[FeedbackReceipt]) -> dict[str, Any]:
    payloads = [receipt.shareable() for receipt in receipts]
    return {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "bundle_id": _digest(payloads),
        "receipts": payloads,
    }


def export_bundle(
    receipts: Sequence[FeedbackReceipt],
    output: str | Path,
    *,
    consent: bool,
) -> Path:
    if not consent:
        raise FeedbackError(
            "export requires explicit consent after previewing the shareable payload"
        )
    if not receipts:
        raise FeedbackError("no receipts selected")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=".feedback-export-", suffix=".tmp"
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(make_bundle(receipts), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return destination


def _validate_endpoint(endpoint: str) -> None:
    parsed = urlparse.urlparse(endpoint)
    localhost = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and localhost):
        raise FeedbackError("feedback endpoint must use HTTPS (HTTP is local-only)")
    if not parsed.netloc or parsed.username or parsed.password:
        raise FeedbackError("feedback endpoint must be an absolute URL without credentials")


def submit_bundle(
    receipts: Sequence[FeedbackReceipt],
    endpoint: str,
    *,
    consent: bool,
    token: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Submit the previewable bundle; never called automatically."""

    if not consent:
        raise FeedbackError(
            "submission requires explicit consent after previewing the shareable payload"
        )
    if not receipts:
        raise FeedbackError("no receipts selected")
    _validate_endpoint(endpoint)
    bundle = make_bundle(receipts)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "gnomon-feedback/1",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urlrequest.Request(
        endpoint, data=_canonical_json(bundle), headers=headers, method="POST"
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:
            raw = response.read(64 * 1024)
            status = response.status
    except urlerror.URLError as exc:
        raise FeedbackError(f"feedback submission failed: {exc.reason}") from exc
    if not 200 <= status < 300:
        raise FeedbackError(f"feedback endpoint returned HTTP {status}")
    result: Any = None
    if raw:
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = raw.decode("utf-8", errors="replace")[:1000]
    return {"status": "submitted", "http_status": status, "response": result}


def _parse_context(values: Sequence[str]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise FeedbackError(f"context must use key=value: {item!r}")
        key, raw = item.split("=", 1)
        if key in _INTEGER_CONTEXT_FIELDS:
            try:
                value: Any = int(raw)
            except ValueError as exc:
                raise FeedbackError(f"{key} must be an integer") from exc
        elif key in _BOOLEAN_CONTEXT_FIELDS:
            if raw.lower() not in {"true", "false"}:
                raise FeedbackError(f"{key} must be true or false")
            value = raw.lower() == "true"
        else:
            value = raw
        context[key] = value
    return normalize_context(context)


def _select_receipts(
    store: FeedbackStore, receipt_ids: Sequence[str], select_all: bool
) -> list[FeedbackReceipt]:
    if select_all and receipt_ids:
        raise FeedbackError("choose receipt IDs or --all, not both")
    if select_all:
        return store.receipts()
    if not receipt_ids:
        raise FeedbackError("provide at least one receipt ID or --all")
    return [store.get(receipt_id) for receipt_id in receipt_ids]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gnomon-feedback",
        description="Create local feedback receipts; sharing is always explicit.",
    )
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="record one local receipt")
    create.add_argument("--category", required=True, choices=sorted(CATEGORIES))
    create.add_argument("--context", action="append", default=[], metavar="KEY=VALUE")
    create.add_argument("--public-summary")
    create.add_argument("--private-note")
    create.add_argument("--artifact-id")

    preview = subparsers.add_parser("preview", help="show exactly what can be shared")
    preview.add_argument("receipt_ids", nargs="*")
    preview.add_argument("--all", action="store_true", dest="select_all")

    listing = subparsers.add_parser("list", help="list local receipt metadata")
    listing.add_argument("--limit", type=int, default=100)

    subparsers.add_parser("summarize", help="aggregate behavior without content")

    export = subparsers.add_parser("export", help="write a consented shareable bundle")
    export.add_argument("receipt_ids", nargs="*")
    export.add_argument("--all", action="store_true", dest="select_all")
    export.add_argument("--output", required=True)
    export.add_argument("--consent", action="store_true")

    submit = subparsers.add_parser("submit", help="send a consented shareable bundle")
    submit.add_argument("receipt_ids", nargs="*")
    submit.add_argument("--all", action="store_true", dest="select_all")
    submit.add_argument("--endpoint", required=True)
    submit.add_argument("--token-env", default="GNOMON_FEEDBACK_TOKEN")
    submit.add_argument("--timeout", type=float, default=10.0)
    submit.add_argument("--consent", action="store_true")
    return parser


def _emit(payload: Any, *, stream: Any | None = None) -> None:
    if stream is None:
        stream = sys.stdout
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    store = FeedbackStore(args.store)
    try:
        if args.command == "create":
            receipt = create_receipt(
                args.category,
                context=_parse_context(args.context),
                public_summary=args.public_summary,
                private_note=args.private_note,
                artifact_id=args.artifact_id,
            )
            store.record(receipt)
            _emit({
                "status": "recorded_locally",
                "receipt_id": receipt.receipt_id,
                "shared": False,
                "next_step": (
                    f"gnomon-feedback --store {args.store} preview "
                    f"{receipt.receipt_id}"
                ),
            })
            return 0
        if args.command == "preview":
            receipts = _select_receipts(store, args.receipt_ids, args.select_all)
            _emit(make_bundle(receipts))
            return 0
        if args.command == "list":
            if args.limit < 0:
                raise FeedbackError("limit must be non-negative")
            rows = store.receipts()[-args.limit:] if args.limit else []
            _emit({
                "receipts": [
                    {
                        "receipt_id": row.receipt_id,
                        "created_at": row.created_at,
                        "category": row.category,
                        "surface": row.context.get("surface"),
                    }
                    for row in rows
                ]
            })
            return 0
        if args.command == "summarize":
            _emit(summarize_receipts(store.receipts()))
            return 0
        if args.command == "export":
            receipts = _select_receipts(store, args.receipt_ids, args.select_all)
            path = export_bundle(receipts, args.output, consent=args.consent)
            _emit({"status": "exported", "path": str(path), "count": len(receipts)})
            return 0
        if args.command == "submit":
            receipts = _select_receipts(store, args.receipt_ids, args.select_all)
            token = os.environ.get(args.token_env) if args.token_env else None
            result = submit_bundle(
                receipts,
                args.endpoint,
                consent=args.consent,
                token=token,
                timeout=args.timeout,
            )
            _emit(result)
            return 0
    except FeedbackError as exc:
        _emit(
            {"status": "error", "error": {"code": "FEEDBACK_ERROR", "message": str(exc)}},
            stream=sys.stderr,
        )
        return 2
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

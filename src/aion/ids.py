"""Content-addressed identifiers and the determinism seam.

Identifiers for datasets, runs, and artifacts are derived from a canonical
hash of their inputs, parameters, and the runtime version: identical work
yields identical identifiers, which is what makes deterministic replay and
content-addressed caching possible. Wall-clock time enters artifacts only
through a ``Clock``, so tests can pin it and golden artifacts stay
byte-identical across runs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Protocol

AION_VERSION = "0.4.0"


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class FixedClock:
    """Pinned clock for tests and deterministic replay."""

    def __init__(self, instant: datetime):
        if instant.tzinfo is None:
            instant = instant.replace(tzinfo=timezone.utc)
        self._instant = instant

    def now(self) -> datetime:
        return self._instant


SYSTEM_CLOCK = SystemClock()


def canonical_json(payload: Any) -> str:
    """Stable serialization: sorted keys, no whitespace, datetimes as ISO."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str,
        ensure_ascii=False, allow_nan=False,
    )


def content_id(kind: str, payload: Any, *, length: int = 16) -> str:
    """``<kind>_<hex>`` where the hex digest covers the runtime version,
    the kind, and the canonical JSON of ``payload``."""
    digest = hashlib.sha256()
    for part in (AION_VERSION, kind, canonical_json(payload)):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return f"{kind}_{digest.hexdigest()[:length]}"

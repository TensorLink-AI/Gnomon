"""Schema-versioning rule for serialized artifacts.

Every serialized artifact carries ``schema_version``. Readers accept the
current version and the one immediately before it, so the contract can
evolve without stranding stored projects.
"""

from __future__ import annotations

from .contracts import GnomonError

CURRENT_SCHEMA_VERSION = "0.1"

# The version immediately before the current one; None while the current
# version is the first ever published.
PREVIOUS_SCHEMA_VERSION: str | None = None


def readable_schema_versions() -> tuple[str, ...]:
    if PREVIOUS_SCHEMA_VERSION is None:
        return (CURRENT_SCHEMA_VERSION,)
    return (CURRENT_SCHEMA_VERSION, PREVIOUS_SCHEMA_VERSION)


def ensure_readable(schema_version: object) -> None:
    if schema_version not in readable_schema_versions():
        raise GnomonError(
            "UNSUPPORTED_SCHEMA_VERSION",
            f"Cannot read schema version {schema_version!r}; "
            f"this runtime reads {', '.join(readable_schema_versions())}.",
            {"found": str(schema_version), "readable": list(readable_schema_versions())},
        )

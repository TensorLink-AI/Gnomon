"""Fail-closed identity for resumable benchmark outputs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def prepare_run_identity(
    output_dir: Path,
    identity: dict[str, Any],
    *,
    resume: bool,
    state_paths: Iterable[Path],
) -> None:
    """Initialize or validate a benchmark checkpoint identity.

    A checkpoint without an identity is deliberately not adopted: doing so
    could mix rows produced by different code, corpora, or request settings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "run_identity.json"
    state = tuple(state_paths)
    if identity_path.is_file():
        try:
            existing = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(
                "run_identity.json is unreadable; use a new output directory"
            ) from error
        if existing != identity:
            raise SystemExit(
                "resume identity mismatch; use a new output directory")
        if not resume:
            raise SystemExit(
                "output directory is already initialized; pass --resume or "
                "use a new output directory")
        return
    if any(path.exists() for path in state):
        raise SystemExit(
            "cannot resume benchmark state without run_identity.json; use a "
            "new output directory")
    descriptor, temporary = tempfile.mkstemp(
        prefix=".run_identity.", dir=output_dir)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(identity, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, identity_path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass

"""Stable corpus identity and atomic benchmark output helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schema import Case, Observation


def corpus_sha256(cases: list[Case]) -> str:
    payload = json.dumps([asdict(case) for case in cases], sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def write_observations(path: Path, observations: list[Observation]) -> None:
    text = "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in observations)
    atomic_write_text(path, text)

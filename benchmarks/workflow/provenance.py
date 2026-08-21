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


def identity_fingerprint(name: str, data_fingerprint: str) -> str:
    payload = json.dumps({"candidate": name, "data": data_fingerprint},
                         sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def artifact_evidence(artifact_path: str | Path,
                      artifact: dict[str, Any]) -> dict[str, Any]:
    """Extract independently recorded evaluation/publication identities.

    Composite evaluation identity lives in ``final_candidate`` evidence;
    publication identity lives in the immutable artifact result.  Built-ins
    deliberately have no extra evidence record, so their selected-model field
    is the contract boundary.  Full member/weight equality is enforced by the
    engine's executable-candidate tests, not inferred from an agent response.

    Lives here (not in the agent adapter) because run_workflow re-derives the
    same identities when it verifies an observation against the artifact it
    names — both sides must compute the fingerprint the same way.
    """
    results = artifact.get("results") or []
    if len(results) != 1:
        return {"parity_evidence_level": "unavailable"}
    published_name = results[0].get("selected_model")
    source = artifact.get("source_fingerprint")
    evaluated_name = None
    evidence_path = Path(artifact_path) / "evidence.jsonl"
    if evidence_path.is_file():
        for line in evidence_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("kind") == "final_candidate":
                evaluated_name = (record.get("payload") or {}).get("name")
    level = "composite_candidate" if evaluated_name else "builtin_selected_model"
    evaluated_name = evaluated_name or published_name
    if not evaluated_name or not published_name or not source:
        return {"parity_evidence_level": "unavailable"}
    return {
        "evaluated_fingerprint": identity_fingerprint(evaluated_name, source),
        "published_fingerprint": identity_fingerprint(published_name, source),
        "artifact_id": artifact.get("forecast_id"),
        "parity_evidence_level": level,
    }


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

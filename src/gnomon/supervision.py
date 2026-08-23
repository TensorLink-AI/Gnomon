"""Explicit, privacy-safe export of governed temporal reasoning receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .tracking import TrackingStore


EXPORT_VERSION = "0.1"


def _digest(value: str, salt: str) -> str:
    return hashlib.sha256((salt + "\0" + value).encode()).hexdigest()[:20]


def _safe_answer(payload: dict[str, Any]) -> dict[str, Any]:
    question = payload.get("question") or {}
    answer = payload.get("answer") or {}
    best = payload.get("best_estimate") or {}
    reasoning = answer.get("reasoning") or payload.get("reasoning") or {}
    return {
        "question": {key: question.get(key) for key in
                     ("verb", "property", "horizon", "measure", "context_policy")
                     if question.get(key) is not None},
        "canonical": {"value": best.get("value", answer.get("direction")),
                      "support": best.get("support", answer.get("support"))},
        "reasoning": {key: reasoning.get(key) for key in
                      ("because", "against", "unknown", "what_would_flip")
                      if reasoning.get(key)},
        "primary_forecast_unchanged": True,
    }


def _safe_outcome(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload or {}
    # Continuous operational values can themselves identify a customer. The
    # default export retains categorical labels and scoring metadata only.
    return {key: payload[key] for key in
            ("direction", "state", "correct", "support", "points")
            if key in payload}


def build_export(store: TrackingStore, project: str, *, salt: str,
                 before: str | None = None) -> dict[str, Any]:
    """Create a de-identified, resolved-only evaluation dataset.

    Raw series, paths, prose context, forecast arrays and artifact identifiers
    are excluded. ``before`` supplies an explicit temporal split boundary.
    """
    rows = []
    for receipt in store.temporal_answer_receipts(project, resolved=True):
        if before and str(receipt["created_at"]) >= before:
            continue
        rows.append({
            "example_id": _digest("|".join(map(str, (
                receipt["forecast_id"], receipt["series"], receipt["question_id"]))), salt),
            "created_at": receipt["created_at"],
            "resolved_at": receipt["resolved_at"],
            "answer": _safe_answer(receipt["payload"]),
            "outcome": _safe_outcome(receipt["outcome_payload"]),
        })
    syntheses = []
    for receipt in store.temporal_synthesis_receipts(project, resolved=True):
        if before and str(receipt["created_at"]) >= before:
            continue
        synthesis = receipt["synthesis_payload"] or {}
        syntheses.append({
            "example_id": _digest(receipt["synthesis_id"], salt),
            "created_at": receipt["created_at"],
            "resolved_at": receipt["resolved_at"],
            "label": synthesis.get("label"),
            "claim": (synthesis.get("value") if isinstance(
                synthesis.get("value"), (str, bool)) else None),
            "evidence_ref_count": len(receipt["evidence_refs"] or []),
            "score": receipt["score_payload"],
            "primary_forecast_unchanged": True,
        })
    return {
        "schema_version": EXPORT_VERSION,
        "purpose": "offline_evaluation_or_training_review",
        "temporal_split": {"created_before": before},
        "privacy": {"identifiers": "salted_digest", "raw_series": False,
                    "raw_context": False, "artifact_paths": False,
                    "resolved_only": True},
        "answers": rows, "syntheses": syntheses,
    }


def export_supervision(store: TrackingStore, project: str, output: str | Path,
                       *, consent: bool, salt: str, before: str | None = None) -> Path:
    if not consent:
        raise ValueError("export requires explicit consent")
    if len(salt) < 16:
        raise ValueError("export salt must contain at least 16 characters")
    path = Path(output)
    if path.exists():
        raise ValueError("refusing to overwrite an existing supervision export")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_export(store, project, salt=salt, before=before),
                               indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export de-identified governed receipts")
    parser.add_argument("--project", required=True)
    parser.add_argument("--registry")
    parser.add_argument("--output", required=True)
    parser.add_argument("--before", help="Exclusive ISO created-at boundary")
    parser.add_argument("--salt", required=True, help="Private salt (16+ characters)")
    parser.add_argument("--consent", action="store_true")
    args = parser.parse_args(argv)
    try:
        path = export_supervision(TrackingStore(args.registry), args.project,
                                  args.output, consent=args.consent,
                                  salt=args.salt, before=args.before)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"status": "exported", "path": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

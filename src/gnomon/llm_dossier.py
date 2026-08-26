"""Validation boundary for LLM-authored temporal dossiers and candidates.

The model may interpret prose and nominate a probabilistic path. It may not
grant that path authority. This module validates citations, timing, quantile
shape, and gross plausibility, then seals the result as a non-automatable
``prior_assisted`` candidate. Historical replay and realised outcomes may later
upgrade it; parsing confidence never can.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import datetime
from typing import Any

from .effect_proposals import validate_effect_proposal
from .context_intelligence import compile_context_hypotheses

DOSSIER_VERSION = "0.2"
MAX_CLAIMS = 16
MAX_BOUNDARY_JUMP_SCALES = 20.0
MAX_PATH_SCALE_RATIO = 30.0
RELATIONS = frozenset({
    "supports_increase", "supports_decrease", "supports_stability",
    "supports_higher_variance", "supports_lower_variance",
    "changes_seasonal_regime", "constrains_range", "unknown",
})


def _normalise(text: Any) -> str:
    return " ".join(str(text or "").split()).casefold()


def _robust_scale(values: list[float]) -> float:
    differences = [abs(b - a) for a, b in zip(values, values[1:])]
    positive = [value for value in differences if value > 0]
    if positive:
        return max(statistics.median(positive), 1e-12)
    return max(abs(statistics.median(values)) * 0.01, 1e-12)


def _timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else None


def validate_temporal_dossier(
    raw: Any,
    *,
    context_text: str,
    cutoff: str,
    future_timestamps: list[str],
    history: list[float],
    compiler_model: str,
) -> tuple[dict[str, Any], list[str]]:
    """Return a sealed dossier and every rejected-field reason.

    A valid dossier may contain claims without a forecast candidate. A
    candidate requires at least one verified cited claim and exactly one
    ordered q10/q50/q90 row per requested future timestamp.
    """
    reasons: list[str] = []
    if not isinstance(raw, dict):
        raw = {}
        reasons.append("dossier output is not an object")
    normalised_context = _normalise(context_text)
    cutoff_dt = _timestamp(cutoff)
    if cutoff_dt is None:
        raise ValueError("cutoff must be timezone-aware ISO-8601")

    claims: list[dict[str, Any]] = []
    for index, claim in enumerate((raw.get("claims") or [])[:MAX_CLAIMS]):
        if not isinstance(claim, dict):
            reasons.append(f"claim {index + 1} is not an object")
            continue
        span = str(claim.get("source_span") or "").strip()
        if not span or _normalise(span) not in normalised_context:
            reasons.append(
                f"claim {index + 1} has no verbatim source_span in context")
            continue
        relation = str(claim.get("relation") or "unknown")
        if relation not in RELATIONS:
            reasons.append(f"claim {index + 1} has unknown relation {relation!r}")
            continue
        start = _timestamp(claim.get("effective_start"))
        end = _timestamp(claim.get("effective_end"))
        if start is None or end is None or end < start:
            reasons.append(f"claim {index + 1} has an invalid effective window")
            continue
        try:
            confidence = float(claim.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = math.nan
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            reasons.append(f"claim {index + 1} has invalid confidence")
            continue
        claims.append({
            "claim_id": f"claim-{len(claims) + 1}",
            "source_span": span,
            "relation": relation,
            "effective_start": start.isoformat(),
            "effective_end": end.isoformat(),
            "mechanism": str(claim.get("mechanism") or "")[:500],
            "confidence": confidence,
            "known_at": cutoff_dt.isoformat(),
        })

    candidate = _validate_candidate(
        raw.get("forecast_candidate"), claims=claims,
        future_timestamps=future_timestamps, history=history, reasons=reasons)
    effect_raw = raw.get("effect_proposal")
    if isinstance(effect_raw, dict) and not effect_raw.get("claim_ids") \
            and len(claims) == 1:
        # The caller proposes claims and effects in one response, before
        # Gnomon assigns canonical claim IDs. A single unambiguous claim may
        # therefore be bound deterministically; multiple claims still require
        # explicit citation so the model cannot smuggle in a broad rationale.
        effect_raw = {**effect_raw, "claim_ids": [claims[0]["claim_id"]],
                      "citation_binding": "single_verified_claim"}
    effect_proposal, proposal_critique = validate_effect_proposal(
        effect_raw,
        claim_ids={str(claim["claim_id"]) for claim in claims},
        repair=raw.get("effect_proposal_repair"),
    ) if raw.get("effect_proposal") not in (None, {}) else (None, {
        "status": "not_proposed", "attempts_used": 0, "attempts_remaining": 2,
        "attempts": [],
    })
    hypotheses, hypothesis_critique = compile_context_hypotheses(
        raw.get("hypotheses"), claims=claims,
        series=[str(value) for value in raw.get("series") or ["*"]],
        cutoff=cutoff, repair=raw.get("hypothesis_repair"),
    )
    payload: dict[str, Any] = {
        "version": DOSSIER_VERSION,
        "compiler_model": compiler_model,
        "known_at": cutoff_dt.isoformat(),
        "future_observations_exposed": False,
        "claims": claims,
        "effect_proposal": effect_proposal,
        "effect_proposal_critique": proposal_critique,
        "hypotheses": hypotheses,
        "hypothesis_critique": hypothesis_critique,
        "forecast_candidate": candidate,
        "candidate_support": "prior_assisted" if (candidate or effect_proposal) else None,
        "automation_eligible": False,
        "primary_forecast_unchanged": True,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["seal_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload, reasons


def verify_temporal_dossier_seal(dossier: dict[str, Any]) -> bool:
    """Whether ``seal_sha256`` authenticates the complete dossier body."""
    if not isinstance(dossier, dict) or not dossier.get("seal_sha256"):
        return False
    body = {key: value for key, value in dossier.items()
            if key != "seal_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(canonical.encode()).hexdigest()
    return dossier["seal_sha256"] == expected


def deterministic_events_from_claims(dossier: dict[str, Any]) -> list[dict[str, Any]]:
    """Promote only literally stated absolute states into event proposals.

    The LLM locates and dates the verbatim span; Gnomon's existing parser must
    independently recover an absolute value. Qualitative effects remain
    scenarios. Returned objects intentionally re-enter the ordinary context
    validator rather than bypassing it.
    """
    from .future_context import parse_bound_span, parse_override_span

    events = []
    for index, claim in enumerate(dossier.get("claims") or [], 1):
        span = str(claim.get("source_span") or "")
        if claim.get("relation") == "constrains_range":
            bound, problem = parse_bound_span(span)
            if problem is None and bound is not None:
                events.append({
                    "event_type": "constraint:stated_range",
                    "entity_scope": ["*"],
                    "effective_start": claim["effective_start"],
                    "effective_end": claim["effective_end"],
                    "confidence": claim.get("confidence", 1.0),
                    "status": "confirmed", "evidence_quote": span,
                    "source_span": span, "effect_family": "saturation_bound",
                    "direction": "unknown", "duration": "temporary",
                    "entity_kind": "unknown",
                    "deterministic_bound_parsed": {
                        "min": bound.minimum, "max": bound.maximum},
                    "derived_from_claim_id": claim.get("claim_id") or f"claim-{index}",
                })
                continue
        value, problem = parse_override_span(span)
        if problem is not None or value is None:
            continue
        events.append({
            "event_type": "override:stated_absolute_value",
            "entity_scope": ["*"],
            "effective_start": claim["effective_start"],
            "effective_end": claim["effective_end"],
            "confidence": claim.get("confidence", 1.0),
            "status": "confirmed",
            "evidence_quote": span, "source_span": span,
            "effect_family": "level_shift", "direction": "unknown",
            "duration": "temporary", "entity_kind": "unknown",
            "deterministic_value_parsed": value,
            "derived_from_claim_id": claim.get("claim_id") or f"claim-{index}",
        })
    return events


def _validate_candidate(
    raw: Any,
    *,
    claims: list[dict[str, Any]],
    future_timestamps: list[str],
    history: list[float],
    reasons: list[str],
) -> dict[str, Any] | None:
    if raw in (None, {}):
        return None
    if not isinstance(raw, dict):
        reasons.append("forecast_candidate is not an object")
        return None
    if not claims:
        reasons.append("forecast_candidate requires a verified cited claim")
        return None
    rows = raw.get("quantiles")
    if not isinstance(rows, list) or len(rows) != len(future_timestamps):
        reasons.append(
            "forecast_candidate quantiles must match the requested horizon")
        return None
    clean: list[dict[str, float | str]] = []
    for index, (row, expected_timestamp) in enumerate(
            zip(rows, future_timestamps)):
        if not isinstance(row, dict):
            reasons.append(f"forecast_candidate row {index + 1} is not an object")
            return None
        if row.get("timestamp") not in (None, expected_timestamp):
            reasons.append(f"forecast_candidate row {index + 1} timestamp differs")
            return None
        try:
            q10, q50, q90 = (float(row[key]) for key in ("q10", "q50", "q90"))
        except (KeyError, TypeError, ValueError):
            reasons.append(f"forecast_candidate row {index + 1} lacks quantiles")
            return None
        if not all(math.isfinite(value) for value in (q10, q50, q90)) \
                or not q10 <= q50 <= q90:
            reasons.append(
                f"forecast_candidate row {index + 1} quantiles are invalid")
            return None
        clean.append({"timestamp": expected_timestamp,
                      "q10": q10, "q50": q50, "q90": q90})
    if not history:
        reasons.append("forecast_candidate cannot be checked without history")
        return None
    scale = _robust_scale(history)
    points = [float(row["q50"]) for row in clean]
    boundary_jump = abs(points[0] - history[-1]) / scale
    path_diffs = [abs(b - a) for a, b in zip(points, points[1:])]
    path_scale_ratio = ((statistics.median(path_diffs) / scale)
                        if path_diffs else 0.0)
    if boundary_jump > MAX_BOUNDARY_JUMP_SCALES:
        reasons.append("forecast_candidate failed boundary-jump plausibility")
        return None
    if path_scale_ratio > MAX_PATH_SCALE_RATIO:
        reasons.append("forecast_candidate failed path-scale plausibility")
        return None
    return {
        "quantiles": clean,
        "rationale": str(raw.get("rationale") or "")[:1000],
        "claim_ids": [claim["claim_id"] for claim in claims],
        "plausibility": {
            "boundary_jump_scales": round(boundary_jump, 6),
            "path_scale_ratio": round(path_scale_ratio, 6),
        },
    }

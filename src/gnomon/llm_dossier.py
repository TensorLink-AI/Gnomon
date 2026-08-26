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
import re
import statistics
from datetime import datetime
from typing import Any

from .effect_proposals import validate_effect_proposal
from .context_intelligence import compile_context_hypotheses

DOSSIER_VERSION = "0.4"
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


def _states_quantitative_relationship(span: Any) -> bool:
    """Whether a cited span states more than direction or a lone bound."""
    text = _normalise(span)
    semantic_laws = (
        "square", "squared", "quadratic", "cube", "cubed", "cubic",
        "double", "twice", "triple", "half", "proportional to",
    )
    if any(token in text for token in semantic_laws):
        return True
    has_number = bool(re.search(
        r"(?<!\w)[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text))
    quantitative_operator = any(token in text for token in (
        "%", "percent", "times", "ratio", " per ", "multipl"))
    return has_number and quantitative_operator


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
    validated_events: list[Any] | None = None,
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

    candidate_reason_start = len(reasons)
    candidate = _validate_candidate(
        raw.get("forecast_candidate"), claims=claims,
        future_timestamps=future_timestamps, history=history, reasons=reasons)
    candidate_reasons = reasons[candidate_reason_start:]
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
        claim_spans={str(claim["claim_id"]): str(claim["source_span"])
                     for claim in claims},
        repair=raw.get("effect_proposal_repair"),
    ) if raw.get("effect_proposal") not in (None, {}) else (None, {
        "status": "not_proposed", "attempts_used": 0, "attempts_remaining": 2,
        "attempts": [],
    })
    if effect_proposal is not None:
        effect_proposal = _align_effect_onset_to_cited_claim(
            effect_proposal, claims=claims,
            future_timestamps=future_timestamps,
            validated_events=validated_events or [])
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
        "candidate_critique": {
            "status": ("accepted" if candidate else "rejected"
                       if raw.get("forecast_candidate") not in (None, {})
                       else "not_proposed"),
            "reasons": candidate_reasons,
            "recovery_action": (
                "Submit a horizon-aligned q10/q50/q90 path that obeys every "
                "cited constraint, or use a typed effect/transformation."
                if candidate_reasons else None),
        },
        "candidate_support": "prior_assisted" if (candidate or effect_proposal) else None,
        "automation_eligible": False,
        "primary_forecast_unchanged": True,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["seal_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload, reasons


def _align_effect_onset_to_cited_claim(
    proposal: dict[str, Any], *, claims: list[dict[str, Any]],
    future_timestamps: list[str], validated_events: list[Any],
) -> dict[str, Any]:
    """Derive relative delay from one cited, horizon-aligned claim window.

    ``delay_steps`` is an execution coordinate, while source context normally
    states a calendar timestamp. The verified claim owns the onset; duration
    remains explicit because interval endpoints may be inclusive or exclusive
    depending on the source's observation semantics.
    """
    cited = set(proposal.get("claim_ids") or [])
    matching = [claim for claim in claims if claim.get("claim_id") in cited]
    grounded_starts = []
    for claim in matching:
        start = _timestamp(claim.get("effective_start"))
        if start is not None and _claim_start_is_cited(
                start, str(claim.get("source_span") or "")):
            grounded_starts.append(start)
    distinct_starts = {value.isoformat() for value in grounded_starts}
    binding = "verified cited claim window and forecast grid"
    if len(distinct_starts) == 1:
        start = grounded_starts[0]
    elif len(distinct_starts) == 0 and len(validated_events) == 1:
        event = validated_events[0]
        start = _timestamp(getattr(event, "effective_start", None))
        attributes = getattr(event, "attributes", {}) or {}
        quote = str(attributes.get("evidence_quote") or
                    attributes.get("source_span") or "")
        if start is None or not _claim_start_is_cited(start, quote):
            return proposal
        binding = "single validated context event and forecast grid"
    else:
        return proposal
    future = [_timestamp(value) for value in future_timestamps]
    if not future or any(value is None for value in future):
        return proposal
    indices = [index for index, timestamp in enumerate(future)
               if timestamp is not None and timestamp >= start]
    if not indices:
        return proposal
    derived = indices[0]
    if derived == proposal.get("delay_steps"):
        return proposal
    normalized = dict(proposal)
    normalized["delay_steps"] = derived
    notes = list(proposal.get("semantic_normalizations") or [])
    notes.append({
        "code": "CLAIM_ONSET_TO_HORIZON_DELAY",
        "cited_effective_start": start.isoformat(),
        "applied_delay_steps": derived,
        "basis": binding,
    })
    normalized["semantic_normalizations"] = notes
    return normalized


def _claim_start_is_cited(start: datetime, span: str) -> bool:
    """Conservatively recognize explicit ISO-like calendar evidence."""
    candidates = {
        start.isoformat(), start.isoformat().replace("T", " "),
        start.strftime("%Y-%m-%dT%H:%M:%S"),
        start.strftime("%Y-%m-%d %H:%M:%S"),
        start.strftime("%Y-%m-%dT%H:%M"),
        start.strftime("%Y-%m-%d %H:%M"),
    }
    if start.hour == start.minute == start.second == start.microsecond == 0:
        candidates.add(start.strftime("%Y-%m-%d"))
    return any(value in span for value in candidates)


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
    rationale = str(raw.get("rationale") or "")[:1000]
    incomplete_markers = (
        "placeholder", "not computed", "not calculated", "unable to compute",
        "cannot compute", "gnomon must", "engine must", "todo", "fill in",
    )
    if any(marker in _normalise(rationale) for marker in incomplete_markers):
        reasons.append("forecast_candidate declares itself incomplete")
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
    # Parse literal constraints before applying empirical scale checks. A
    # large regime change can be a legitimate prior-assisted scenario when
    # context both explains its direction and bounds its numeric extent. It
    # remains non-automatable and carries a warning; an unbounded large jump
    # is still rejected.
    from .future_context import parse_bound_span
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    bound_claim_ids: list[str] = []
    for claim in claims:
        if claim.get("relation") != "constrains_range":
            continue
        bound, problem = parse_bound_span(str(claim.get("source_span") or ""))
        if problem is not None or bound is None:
            continue
        if bound.minimum is not None:
            lower_bounds.append(float(bound.minimum))
        if bound.maximum is not None:
            upper_bounds.append(float(bound.maximum))
        bound_claim_ids.append(str(claim["claim_id"]))
    lower = max(lower_bounds) if lower_bounds else None
    upper = min(upper_bounds) if upper_bounds else None
    if lower is not None and upper is not None and lower > upper:
        reasons.append("forecast_candidate cites contradictory numeric bounds")
        return None
    values = [float(row[key]) for row in clean
              for key in ("q10", "q50", "q90")]
    if lower is not None and any(value < lower for value in values):
        reasons.append("forecast_candidate violates cited lower bound")
        return None
    if upper is not None and any(value > upper for value in values):
        reasons.append("forecast_candidate violates cited upper bound")
        return None
    if not any(float(row["q90"]) > float(row["q10"]) for row in clean):
        reasons.append(
            "forecast_candidate must express non-zero predictive uncertainty")
        return None

    scale = _robust_scale(history)
    points = [float(row["q50"]) for row in clean]
    boundary_jump = abs(points[0] - history[-1]) / scale
    path_diffs = [abs(b - a) for a, b in zip(points, points[1:])]
    path_scale_ratio = ((statistics.median(path_diffs) / scale)
                        if path_diffs else 0.0)
    quantitative_support = any(
        claim.get("relation") in {
            "supports_increase", "supports_decrease",
            "changes_seasonal_regime",
        } and _states_quantitative_relationship(claim.get("source_span"))
        for claim in claims)
    bounded_regime_change = bool(bound_claim_ids) and quantitative_support
    plausibility_warnings: list[str] = []
    if boundary_jump > MAX_BOUNDARY_JUMP_SCALES and not bounded_regime_change:
        reasons.append("forecast_candidate failed boundary-jump plausibility")
        return None
    if boundary_jump > MAX_BOUNDARY_JUMP_SCALES:
        plausibility_warnings.append(
            "candidate leaves the empirical history scale but remains inside "
            "a cited numeric bound; treat it as prior-assisted only")
    if path_scale_ratio > MAX_PATH_SCALE_RATIO:
        reasons.append("forecast_candidate failed path-scale plausibility")
        return None
    return {
        "quantiles": clean,
        "rationale": rationale,
        "claim_ids": [claim["claim_id"] for claim in claims],
        "plausibility": {
            "boundary_jump_scales": round(boundary_jump, 6),
            "path_scale_ratio": round(path_scale_ratio, 6),
            "entailed_bounds": {"minimum": lower, "maximum": upper,
                                "claim_ids": bound_claim_ids},
            "warnings": plausibility_warnings,
        },
    }

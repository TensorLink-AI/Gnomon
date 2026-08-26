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

DOSSIER_VERSION = "0.5"
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


def deterministic_historical_observation_claim(
    context_text: str, *, history_start: str, cutoff: str,
) -> dict[str, Any] | None:
    """Extract one high-precision historical data-quality claim verbatim.

    This is intentionally narrower than the LLM compiler: a disruption, an
    explicit absence of recorded target activity, and an explicit statement
    that the disruption ended are all required. It grants no numeric effect;
    the returned claim still passes the ordinary dossier boundary.
    """
    normalised = _normalise(context_text)
    ended = any(token in normalised for token in (
        "no future", "has ended", "had ended", "will not recur",
        "will no longer", "does not continue")) or bool(re.search(
            r"\bwill not be (?:in|under|on) (?:maintenance|an? outage|closure)\b",
            normalised))
    if not ended:
        return None
    fragments = [fragment.strip() for fragment in re.split(
        r"(?<=[.!?])\s+", context_text) if fragment.strip()]
    for fragment in fragments:
        span = _normalise(fragment)
        disruption = any(token in span for token in (
            "maintenance", "outage", "closure", "stockout",
            "reporting failure"))
        exact_absence = bool(
            re.search(r"\b(?:zero|no)\b.{0,50}\b(?:recorded|withdrawal|sale|order|request|reading|transaction|event)s?\b", span)
            or re.search(r"\b(?:recorded|withdrawal|sale|order|request|reading|transaction|event)s?\b.{0,50}\b(?:zero|none)\b", span))
        if not disruption or not exact_absence:
            continue
        date_match = re.search(r"\b\d{4}-\d{2}-\d{2}(?:[ T][0-9:]+)?", fragment)
        effective_start = history_start
        if date_match:
            parsed = datetime.fromisoformat(date_match.group(0))
            cutoff_dt = _timestamp(cutoff)
            if parsed.tzinfo is None and cutoff_dt is not None:
                parsed = parsed.replace(tzinfo=cutoff_dt.tzinfo)
            effective_start = parsed.isoformat()
        return {
            "source_span": fragment,
            "relation": "unknown",
            "effective_start": effective_start,
            "effective_end": cutoff,
            "mechanism": (
                "deterministic literal extraction of historical observation "
                "corruption; no numeric effect inferred"),
            "confidence": 1.0,
            "compiler_binding": "deterministic_literal_fallback",
        }
    return None


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
    history_timestamps: list[str] | None = None,
    compiler_model: str,
    validated_events: list[Any] | None = None,
    candidate_selection_eligible: bool = True,
    candidate_selection_reason: str | None = None,
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

    history_timestamps = list(history_timestamps or [])
    if history_timestamps and len(history_timestamps) != len(history):
        raise ValueError("history_timestamps must align with history")
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
        history_window_binding = None
        if (start is None or end is None or end < start) and \
                _claim_requests_whole_history_binding(
                    claim, raw.get("observation_interpretations"),
                    normalised_context) and history_timestamps:
            start = _timestamp(history_timestamps[0])
            end = _timestamp(history_timestamps[-1])
            if start is not None and end is not None:
                history_window_binding = {
                    "kind": "observed_history_window",
                    "basis": (
                        "source identifies historical observation corruption "
                        "without an exact calendar window"),
                }
        if start is None or end is None or end < start:
            reasons.append(f"claim {index + 1} has an invalid effective window")
            continue
        raw_confidence = claim.get("confidence", 1.0)
        confidence_normalization = None
        qualitative = {
            "low": 0.25, "medium": 0.5, "moderate": 0.5, "high": 0.75,
        }
        confidence_text = (raw_confidence.strip().casefold()
                           if isinstance(raw_confidence, str) else "")
        qualitative_match = next(
            (label for label in qualitative
             if re.search(rf"\b{label}\b", confidence_text)), None)
        numeric_matches = (re.findall(r"(?<!\w)(?:\d+(?:\.\d*)?|\.\d+)",
                                      confidence_text)
                           if confidence_text else [])
        if raw_confidence in (None, ""):
            confidence = 0.5
            confidence_normalization = {
                "kind": "missing_to_conservative_unit_interval",
                "supplied": raw_confidence, "normalized": confidence,
                "authority_effect": "none",
            }
        elif len(numeric_matches) == 1:
            confidence = float(numeric_matches[0])
            confidence_normalization = {
                "kind": "numeric_text_to_unit_interval",
                "supplied": raw_confidence,
                "normalized": confidence,
                "authority_effect": "none",
            }
        elif qualitative_match is not None:
            confidence = qualitative[qualitative_match]
            confidence_normalization = {
                "kind": "qualitative_to_conservative_unit_interval",
                "supplied": raw_confidence,
                "normalized": confidence,
                "authority_effect": "none",
            }
        else:
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                confidence = math.nan
        if math.isfinite(confidence) and 1 < confidence <= 100:
            confidence /= 100.0
            confidence_normalization = {
                "kind": "percent_to_unit_interval",
                "supplied": raw_confidence,
                "normalized": confidence,
            }
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
            **({"confidence_normalization": confidence_normalization}
               if confidence_normalization else {}),
            **({"effective_window_binding": history_window_binding}
               if history_window_binding else {}),
        })

    raw_observation_interpretations = list(
        raw.get("observation_interpretations") or [])
    derived_interpretations = _derive_historical_zero_interpretation(
        claims, context_text=context_text)
    # A malformed optional wrapper must not suppress the narrower executable
    # that Gnomon can derive from the same verified claim. Keep both in the
    # critique: the model proposal remains visibly rejected while the
    # deterministic binding may proceed independently.
    for interpretation in derived_interpretations:
        signature = json.dumps(interpretation.get("predicate"), sort_keys=True)
        if not any(json.dumps((item or {}).get("predicate"), sort_keys=True)
                   == signature for item in raw_observation_interpretations
                   if isinstance(item, dict)):
            raw_observation_interpretations.append(interpretation)
    observation_interpretations, observation_critique, derived_candidate = \
        _validate_observation_interpretations(
            raw_observation_interpretations, claims=claims,
            history=history, history_timestamps=history_timestamps,
            future_timestamps=future_timestamps)
    candidate_was_derived_from_observation_interpretation = (
        raw.get("forecast_candidate") in (None, {})
        and derived_candidate is not None)
    if candidate_was_derived_from_observation_interpretation:
        candidate_selection_eligible = False
        candidate_selection_reason = (
            "Historical-contamination filtering is mechanically valid but "
            "the derived empirical path has not beaten the immutable primary "
            "in fold-safe replay; retain it as a visible scenario only.")
    candidate_reason_start = len(reasons)
    candidate = _validate_candidate(
        raw.get("forecast_candidate") or derived_candidate, claims=claims,
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
        "observation_interpretations": observation_interpretations,
        "observation_interpretation_critique": observation_critique,
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
                {
                    "code": "repair_forecast_candidate",
                    "message": (
                        "Submit a horizon-aligned q10/q50/q90 path that obeys "
                        "every cited constraint, or use a typed effect or "
                        "transformation."
                    ),
                    "required_evidence": [
                        "horizon-aligned q10/q50/q90 path",
                        "cited source claims",
                    ],
                    "automation_eligible": False,
                }
                if candidate_reasons else None),
            "selection_eligible": bool(candidate and candidate_selection_eligible),
            "selection_reason": (candidate_selection_reason
                                 if candidate and not candidate_selection_eligible
                                 else None),
            "candidate_origin": (
                "observation_interpretation_counterfactual"
                if candidate_was_derived_from_observation_interpretation
                else "model_authored" if candidate else None),
        },
        "candidate_support": "prior_assisted" if (
            candidate or effect_proposal or observation_interpretations) else None,
        "automation_eligible": False,
        "primary_forecast_unchanged": True,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["seal_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload, reasons


def _claim_requests_whole_history_binding(
    claim: dict[str, Any], raw_interpretations: Any, normalised_context: str,
) -> bool:
    """Allow an explicitly historical, cited data-quality claim to bind history.

    This does not infer when an event occurred. It only scopes an observation
    predicate to the already-visible history when prose says the corruption is
    historical but supplies no calendar bounds.
    """
    span = _normalise(claim.get("source_span"))
    if not span or span not in normalised_context:
        return False
    historical = any(token in span for token in (
        "historical", "in the past", "previously", "recorded", "readings",
        "was under", "were under"))
    corrupted = any(token in span for token in (
        "maintenance", "outage", "closure", "closed", "stockout",
        "censor", "reporting failure", "missing", "unavailable"))
    ended = any(token in normalised_context for token in (
        "no future", "has ended", "had ended", "will not recur",
        "will no longer", "does not continue")) or bool(re.search(
            r"\bwill not be (?:in|under|on) (?:maintenance|an? outage|closure)\b",
            normalised_context))
    if not historical or not corrupted or not ended:
        return False
    explicit_wrapper = isinstance(raw_interpretations, list) and any(
        isinstance(item, dict) and
        item.get("window") == "all_observed_history"
        for item in raw_interpretations)
    exact_absence = bool(
        re.search(r"\b(?:zero|no)\b.{0,50}\b(?:recorded|withdrawal|sale|order|request|reading|transaction|event)s?\b", span))
    return explicit_wrapper or exact_absence


def _derive_historical_zero_interpretation(
    claims: list[dict[str, Any]], *, context_text: str,
) -> list[dict[str, Any]]:
    """Bind verified prose to the narrow historical-zero interpretation.

    The language model still locates and cites the fact. This deterministic
    step prevents interface reliability from depending on whether it also
    copied the optional typed wrapper. No candidate is derived unless the
    ordinary validator independently proves the zero semantics and safe mask.
    """
    text = _normalise(context_text)
    ended = any(token in text for token in (
        "no future", "has ended", "had ended", "will not recur",
        "will no longer", "does not continue")) or bool(re.search(
            r"\bwill not be (?:in|under|on) (?:maintenance|an? outage|closure)\b",
            text))
    if not ended:
        return []
    for claim in claims:
        span = _normalise(claim.get("source_span"))
        disruption = any(token in span for token in (
            "maintenance", "outage", "closure", "stockout",
            "reporting failure"))
        exact_absence = bool(
            re.search(r"\b(?:zero|no)\b.{0,50}\b(?:recorded|withdrawal|sale|order|request|reading|transaction|event)s?\b", span)
            or re.search(r"\b(?:recorded|withdrawal|sale|order|request|reading|transaction|event)s?\b.{0,50}\b(?:zero|none)\b", span))
        if disruption and exact_absence:
            source = str(claim["source_span"])
            recurrence = re.search(
                r"for\s+(\d+)\s+days?.{0,60}?every\s+(\d+)\s+days?.{0,80}?"
                r"starting\s+(?:from\s+)?(\d{4}-\d{2}-\d{2}(?:[ T][0-9:]+)?)",
                source, re.I)
            predicate: dict[str, Any]
            if recurrence:
                predicate = {
                    "op": "recurring_window",
                    "duration_steps": int(recurrence.group(1)),
                    "period_steps": int(recurrence.group(2)),
                    "start": recurrence.group(3),
                }
            else:
                predicate = {"op": "equals", "value": 0.0}
            return [{
                "kind": "historical_contamination",
                "claim_ids": [claim["claim_id"]],
                "predicate": predicate,
                "window": ("cited_window" if re.search(
                    r"\b\d{4}-\d{2}-\d{2}\b", str(claim["source_span"]))
                           else "all_observed_history"),
                "rationale": (
                    "deterministically bound from a verified historical "
                    "zero-recording claim whose disruption has ended"),
                "proposal_origin": "verified_claim_semantics",
            }]
    return []


def _validate_observation_interpretations(
    raw: Any, *, claims: list[dict[str, Any]], history: list[float],
    history_timestamps: list[str], future_timestamps: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    """Validate historical contamination predicates and derive one safe path.

    Only an exact zero predicate is supported initially, and only when the
    cited prose literally states zero or an absence of recorded target events.
    The operation is applied to a copy. The immutable primary and stored input
    remain untouched; the resulting path is always prior-assisted.
    """
    if raw in (None, []):
        return [], {"status": "not_proposed", "rejected": []}, None
    if not isinstance(raw, list):
        return [], {"status": "rejected", "rejected": [
            {"index": 0, "code": "NOT_A_LIST"}]}, None
    by_id = {str(claim["claim_id"]): claim for claim in claims}
    accepted: list[dict[str, Any]] = []
    accepted_masks: list[list[bool]] = []
    rejected: list[dict[str, Any]] = []
    for index, item in enumerate(raw[:4]):
        if not isinstance(item, dict):
            rejected.append({"index": index, "code": "NOT_AN_OBJECT"})
            continue
        claim_ids = [str(value) for value in item.get("claim_ids") or []]
        cited = [by_id[value] for value in claim_ids if value in by_id]
        predicate = item.get("predicate") or {}
        try:
            predicate_value = float(predicate.get("value"))
        except (TypeError, ValueError):
            predicate_value = math.nan
        if not cited or len(cited) != len(claim_ids):
            rejected.append({"index": index, "code": "UNVERIFIED_CLAIMS"})
            continue
        predicate_op = predicate.get("op")
        if predicate_op not in {"equals", "recurring_window"}:
            rejected.append({"index": index, "code": "UNSUPPORTED_PREDICATE"})
            continue
        spans = [_normalise(claim["source_span"]) for claim in cited]
        zero_entailed = any(
            re.search(r"\b(?:zero|no)\b.{0,40}\b(?:recorded|withdrawal|sale|order|request|reading|transaction|event)s?\b", span)
            or re.search(r"\b(?:recorded|withdrawal|sale|order|request|reading|transaction|event)s?\b.{0,40}\b(?:zero|none)\b", span)
            for span in spans)
        if not zero_entailed:
            rejected.append({"index": index, "code": "PREDICATE_NOT_ENTAILED"})
            continue
        if not history_timestamps:
            rejected.append({"index": index, "code": "HISTORY_TIMESTAMPS_REQUIRED"})
            continue
        starts = [_timestamp(claim["effective_start"]) for claim in cited]
        ends = [_timestamp(claim["effective_end"]) for claim in cited]
        if any(value is None for value in [*starts, *ends]):
            rejected.append({"index": index, "code": "INVALID_WINDOW"})
            continue
        start = min(value for value in starts if value is not None)
        end = max(value for value in ends if value is not None)
        in_window = [
            start <= (_timestamp(timestamp) or start) <= end
            for timestamp in history_timestamps]
        applied_predicate: dict[str, Any]
        predicate_normalization = None
        if predicate_op == "recurring_window":
            try:
                duration = int(predicate["duration_steps"])
                period = int(predicate["period_steps"])
                recurrence_start = _timestamp(predicate["start"])
                if recurrence_start is None:
                    naive_start = datetime.fromisoformat(str(predicate["start"]))
                    recurrence_start = naive_start.replace(tzinfo=start.tzinfo)
            except (KeyError, TypeError, ValueError):
                duration = period = 0
                recurrence_start = None
            source = " ".join(spans)
            schedule_entailed = (
                duration > 0 and period >= duration and recurrence_start is not None
                and str(duration) in source and str(period) in source
                and recurrence_start.strftime("%Y-%m-%d") in source)
            if not schedule_entailed:
                rejected.append({"index": index,
                                 "code": "SCHEDULE_NOT_ENTAILED"})
                continue
            mask = []
            for inside, timestamp in zip(in_window, history_timestamps):
                observed = _timestamp(timestamp)
                if not inside or observed is None or observed < recurrence_start:
                    mask.append(False)
                    continue
                step = (observed.date() - recurrence_start.date()).days
                mask.append(step % period < duration)
            applied_predicate = {
                "op": "recurring_window", "start": recurrence_start.isoformat(),
                "duration_steps": duration, "period_steps": period,
            }
        else:
            if predicate_value != 0.0:
                rejected.append({"index": index,
                                 "code": "UNSUPPORTED_PREDICATE"})
                continue
            applied_value = 0.0
            mask = [inside and value == 0.0
                    for inside, value in zip(in_window, history)]
            applied_predicate = {"op": "equals", "value": applied_value}
        if predicate_op == "equals" and not any(mask):
            window_values = [float(value) for value, inside in
                             zip(history, in_window) if inside]
            if window_values:
                observed_floor = min(window_values)
                tolerance = max(1.0, abs(observed_floor)) * 1e-9
                floor_mask = [
                    inside and abs(float(value) - observed_floor) <= tolerance
                    for inside, value in zip(in_window, history)]
                if sum(floor_mask) >= 2:
                    mask = floor_mask
                    applied_value = observed_floor
                    predicate_normalization = {
                        "kind": "semantic_zero_to_repeated_observed_floor",
                        "stated_value": 0.0,
                        "observed_value": observed_floor,
                        "basis": (
                            "input units do not contain zero; a repeated "
                            "empirical floor is retained as a disclosed "
                            "candidate-only interpretation"),
                    }
                    applied_predicate = {"op": "equals",
                                         "value": applied_value}
        retained = [float(value) for value, excluded in zip(history, mask)
                    if not excluded]
        excluded_count = sum(mask)
        if excluded_count == 0 or len(retained) < 3 or excluded_count > len(history) * .8:
            rejected.append({"index": index, "code": "UNSAFE_OR_EMPTY_FILTER",
                             "excluded": excluded_count,
                             "retained": len(retained)})
            continue
        accepted.append({
            "interpretation_id": f"observation-interpretation-{len(accepted)+1}",
            "kind": "historical_contamination",
            "claim_ids": claim_ids,
            "predicate": applied_predicate,
            **({"predicate_normalization": predicate_normalization}
               if predicate_normalization else {}),
            "window": item.get("window") or "cited_window",
            "excluded_observations": excluded_count,
            "retained_observations": len(retained),
            "input_mutated": False,
            "support": "prior_assisted",
            "automation_eligible": False,
            "rationale": str(item.get("rationale") or "")[:500],
        })
        accepted_masks.append(mask)
    critique = {
        "status": "accepted" if accepted else "rejected",
        "accepted": len(accepted), "rejected": rejected,
    }
    if not accepted:
        return [], critique, None
    # A deliberately conservative distributional counterfactual: future
    # location and uncertainty come only from retained pre-cutoff observations.
    # It is not claimed as a fitted dynamic model and can never automate.
    excluded = accepted[0]["excluded_observations"]
    cited_ids = set(accepted[0]["claim_ids"])
    cited = [claim for claim in claims if claim["claim_id"] in cited_ids]
    start = min(_timestamp(claim["effective_start"]) for claim in cited)
    end = max(_timestamp(claim["effective_end"]) for claim in cited)
    retained = [float(value) for value, excluded in
                zip(history, accepted_masks[0]) if not excluded]
    ordered = sorted(retained)
    def empirical(p: float) -> float:
        position = p * (len(ordered) - 1)
        lower = int(math.floor(position)); upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    q10, q50, q90 = empirical(.1), empirical(.5), empirical(.9)
    candidate = {
        "quantiles": [{"timestamp": timestamp, "q10": q10,
                       "q50": q50, "q90": q90}
                      for timestamp in future_timestamps],
        "rationale": (
            f"Counterfactual empirical distribution from {len(retained)} "
            f"pre-cutoff observations after excluding {excluded} readings "
            "matching the disclosed historical-contamination predicate."),
    }
    return accepted, critique, candidate


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
    scale = _robust_scale(history)
    uncertainty_normalization = None
    if not any(float(row["q90"]) > float(row["q10"]) for row in clean):
        # A model may correctly compute a deterministic conditional mean while
        # omitting observation/process noise. Preserve the useful mean but
        # never publish fake certainty: apply a deterministic, disclosed floor
        # from pre-cutoff first differences. This path remains prior-assisted
        # and categorically ineligible for automation.
        half_width = 1.281552 * scale
        clean = [{**row,
                  "q10": float(row["q50"]) - half_width,
                  "q90": float(row["q50"]) + half_width}
                 for row in clean]
        uncertainty_normalization = {
            "code": "ROBUST_HISTORY_UNCERTAINTY_FLOOR",
            "half_width": round(half_width, 12),
            "basis": "1.281552 times median absolute first difference",
        }
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
            "uncertainty_normalization": uncertainty_normalization,
        },
    }

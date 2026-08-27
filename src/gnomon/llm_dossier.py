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
from datetime import datetime, timedelta
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


def _cited_span_resolves_start(
        span: str, start: datetime | None, *, cutoff: datetime | None = None,
) -> bool:
    """Return whether a cited clause explicitly dates its own onset.

    ``timing_status`` is model-authored metadata, so it cannot overrule an
    ISO calendar token present in the verbatim evidence. Keep this parser
    deliberately narrow: it reconciles only an onset cue paired with the
    supplied start's YYYY-MM or YYYY-MM-DD, and never invents a date from a
    weekday, holiday name, or relative phrase.
    """
    if start is None:
        return False
    has_onset_cue = re.search(
        r"\b(?:start(?:s|ing|ed)?|begin(?:s|ning)?|effective|from)\b",
        span, re.IGNORECASE)
    if has_onset_cue:
        for matched in re.finditer(
                r"(?<!\d)(\d{4})-(\d{2})(?:-(\d{2}))?", span):
            year, month = int(matched.group(1)), int(matched.group(2))
            day = int(matched.group(3)) if matched.group(3) else None
            if year == start.year and month == start.month and (
                    day is None or day == start.day):
                return True
    # Some operational sources (runbooks, experiment logs, incident notes)
    # state an onset as a clock time because the surrounding series supplies
    # the calendar date. Reconcile only an exact clock match attached to
    # explicit transition language; a bare time in a report remains
    # insufficient. ``start`` is host-resolved on the forecast grid, so this
    # does not let the model invent the missing date.
    for matched in re.finditer(
            r"\b(?:at|from)\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\b"
            r"[^.]{0,100}\b(?:changes?|shifts?|starts?|begins?|becomes?|"
            r"increases?|decreases?)\b",
            span, re.IGNORECASE):
        hour, minute = int(matched.group(1)), int(matched.group(2))
        second = int(matched.group(3) or 0)
        cited_clock = (hour, minute, second)
        # A setting changed exactly at the observation cutoff governs the
        # first forecast step even though the response grid begins one step
        # later. Both dates are host-owned; accepting either avoids turning a
        # known boundary transition into an unresolved future trigger.
        if cited_clock == (start.hour, start.minute, start.second) or (
                cutoff is not None and cited_clock == (
                    cutoff.hour, cutoff.minute, cutoff.second)):
            return True
    return False


def _association_only_claim(span: str) -> bool:
    """Identify explicitly associational language without causal authority."""
    text = _normalise(span)
    association = bool(re.search(
        r"\b(?:correlat\w*|co[- ]?occur\w*|associated with|move together|"
        r"tend(?:s|ed)? to (?:rise|fall|increase|decrease) together)\b", text))
    explicit_cause = bool(re.search(
        r"\b(?:causes?|caused by|leads? to|results? in|drives?|because of|"
        r"as a result of)\b", text))
    return association and not explicit_cause


def _atemporal_hypothesis_rows(
        claims: list[dict[str, Any]], *, cutoff: datetime,
) -> list[dict[str, Any]]:
    """Project verified background claims into non-numeric hypotheses."""
    directions = {
        "supports_increase": "increase",
        "supports_decrease": "decrease",
        "supports_stability": "unknown",
        "supports_higher_variance": "increase",
        "supports_lower_variance": "decrease",
    }
    rows = []
    for claim in claims:
        if claim.get("timing_status") != "atemporal_context":
            continue
        association = claim.get("relationship_authority") \
            == "associational_only"
        stability = claim.get("relation") == "supports_stability"
        rows.append({
            "kind": "historical_analogue" if stability else "unsupported",
            "claim_ids": [claim["claim_id"]],
            "target_series": ["*"],
            "predictor_series": None,
            "known_at": cutoff.isoformat(),
            "lag_steps": 0,
            "direction": directions.get(str(claim.get("relation")), "unknown"),
            "rationale": (
                "Verified associational background retained without causal "
                "or numeric authority. Supply an observed predictor and a "
                "fold-validated relationship executable to apply it."
                if association else
                "Verified historical background retained as a comparison, "
                "not a deterministic forecast adjustment."),
        })
    return rows


def _validated_event_window_for_claim(
        claim: dict[str, Any], validated_events: list[Any], *,
        context_text: str,
) -> tuple[datetime, datetime] | None:
    """Join a magnitude-only claim to one validated containing event quote."""
    from .future_context import parse_override_scale

    span = str(claim.get("source_span") or "")
    normalized_span = _normalise(span)
    matches = []
    for event in validated_events:
        attributes = getattr(event, "attributes", {}) or {}
        quote = str(attributes.get("evidence_quote") or
                    attributes.get("source_span") or "")
        start = _timestamp(getattr(event, "effective_start", None))
        end = _timestamp(getattr(event, "effective_end", None))
        if (normalized_span and normalized_span in _normalise(quote)
                and start is not None and end is not None and end >= start
                and _claim_start_is_cited(start, quote)):
            matches.append((start, end))
    if len(matches) == 1:
        return matches[0]
    if matches or len(validated_events) != 1:
        return None

    # Sources often put the event schedule and its measured magnitude in
    # adjacent sentences. Join them only within one source paragraph, under a
    # tight distance bound, with one validated event and a compatible stated
    # direction. This is document structure, not semantic date invention.
    multiplier, problem = parse_override_scale(span)
    if problem is not None or multiplier is None:
        return None
    event = validated_events[0]
    attributes = getattr(event, "attributes", {}) or {}
    quote = str(attributes.get("evidence_quote") or
                attributes.get("source_span") or "")
    start = _timestamp(getattr(event, "effective_start", None))
    end = _timestamp(getattr(event, "effective_end", None))
    direction = str((attributes.get("soft_context") or {}).get(
        "direction") or "unknown")
    expected = {"supports_increase": "increase",
                "supports_decrease": "decrease"}.get(
                    str(claim.get("relation")))
    claim_pos = context_text.find(span)
    quote_pos = context_text.find(quote)
    if (start is None or end is None or end < start
            or claim_pos < 0 or quote_pos < 0
            or not _claim_start_is_cited(start, quote)
            or expected is not None and direction not in {expected, "unknown"}):
        return None
    left = min(claim_pos, quote_pos)
    right = max(claim_pos + len(span), quote_pos + len(quote))
    between = context_text[left:right]
    if right - left > 800 or re.search(r"\n\s*\n", between):
        return None
    return start, end


def _derived_scale_effect(
        claims: list[dict[str, Any]], *, future_timestamps: list[str],
) -> dict[str, Any] | None:
    """Compile one verbatim baseline multiplier joined to validated timing."""
    from .future_context import parse_override_scale

    future = [_timestamp(value) for value in future_timestamps]
    if not future or any(value is None for value in future):
        return None
    candidates = []
    for claim in claims:
        binding = claim.get("effective_window_binding") or {}
        if binding.get("kind") != "validated_event_context_join":
            continue
        multiplier, problem = parse_override_scale(
            str(claim.get("source_span") or ""))
        if problem is not None or multiplier is None:
            continue
        start = _timestamp(claim.get("effective_start"))
        end = _timestamp(claim.get("effective_end"))
        if start is None or end is None:
            continue
        active = [index for index, stamp in enumerate(future)
                  if stamp is not None and start <= stamp < end]
        if not active:
            # A point-like window still owns the first grid point at/after it.
            active = [index for index, stamp in enumerate(future)
                      if stamp is not None and stamp >= start][:1]
        if not active:
            continue
        candidates.append((claim, float(multiplier), active))
    if len(candidates) != 1:
        return None
    claim, multiplier, active = candidates[0]
    delta = multiplier - 1.0
    return {
        "shape": ("temporary_pulse" if len(active) < len(future)
                  else "level_shift"),
        "unit": "fraction_of_level",
        "location": delta, "lower": delta, "upper": delta,
        "confidence": float(claim.get("confidence", 1.0)),
        "delay_steps": active[0], "duration_steps": len(active),
        "scope": {"kind": "single_series", "series": ["*"]},
        "claim_ids": [claim["claim_id"]],
        "rationale": (
            "Deterministic compilation of a verbatim baseline multiplier "
            "and its uniquely containing validated event window."),
        "uncertainty_basis": "verbatim multiplier and validated event window",
        "compiler_binding": "validated_event_plus_verbatim_scale",
    }


_MONTHS = {
    name: number for number, names in enumerate((
        ("january", "jan"), ("february", "feb"), ("march", "mar"),
        ("april", "apr"), ("may",), ("june", "jun"),
        ("july", "jul"), ("august", "aug"),
        ("september", "sep", "sept"), ("october", "oct"),
        ("november", "nov"), ("december", "dec"),
    ), 1) for name in names
}


def _historical_partial_date_window(
    span: str, cutoff: datetime,
) -> tuple[datetime, datetime, dict[str, Any]] | None:
    """Resolve a past-tense month/day reference to its latest occurrence."""
    text = _normalise(span)
    if re.search(r"\b(?:next|upcoming|will|expected|planned)\b", text):
        return None
    if not re.search(r"\b(?:was|were|historical|previous|prior|reference)\b",
                     text):
        return None
    month_pattern = "|".join(sorted(_MONTHS, key=len, reverse=True))
    matched = re.search(
        rf"\b(?P<month>{month_pattern})\.?\s+(?P<day>\d{{1,2}})"
        r"(?:st|nd|rd|th)?\b", text)
    if matched is None:
        return None
    month = _MONTHS[matched.group("month")]
    day = int(matched.group("day"))
    try:
        start = cutoff.replace(year=cutoff.year, month=month, day=day,
                               hour=0, minute=0, second=0, microsecond=0)
    except ValueError:
        return None
    if start > cutoff:
        start = start.replace(year=start.year - 1)
    end = start + timedelta(days=1) - timedelta(microseconds=1)
    return start, end, {
        "kind": "most_recent_historical_month_day",
        "source_token": matched.group(0),
        "resolved_date": start.date().isoformat(),
        "cutoff": cutoff.isoformat(),
        "basis": "past-tense cited reference with no supplied year",
    }


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
        timing_status = str(claim.get("timing_status") or "resolved")
        if timing_status not in {
                "resolved", "unresolved_trigger", "atemporal_context"}:
            reasons.append(f"claim {index + 1} has unknown timing_status")
            continue
        event_window = (_validated_event_window_for_claim(
            claim, validated_events or [], context_text=context_text)
                        if timing_status == "unresolved_trigger" else None)
        if event_window is not None:
            start, end = event_window
            timing_status = "resolved"
            history_window_binding = {
                "kind": "validated_event_context_join",
                "basis": (
                    "the claim span is contained by exactly one validated "
                    "event quote whose cited onset owns the effective window"),
                "supplied_timing_status": "unresolved_trigger",
                "numeric_authority": True,
                "automation_eligible": False,
            }
        elif timing_status == "unresolved_trigger" and \
                _cited_span_resolves_start(span, start, cutoff=cutoff_dt):
            timing_status = "resolved"
            history_window_binding = {
                "kind": "explicit_source_timing_reconciled",
                "basis": (
                    "verbatim cited onset matches the supplied effective "
                    "start; model-authored unresolved label was corrected"),
                "supplied_timing_status": "unresolved_trigger",
                "numeric_authority": False,
                "automation_eligible": False,
            }
        if timing_status == "unresolved_trigger":
            # This is question scope, not asserted event timing. It keeps a
            # useful qualitative rule visible while categorically preventing
            # deterministic application until the trigger is dated.
            start = _timestamp(future_timestamps[0]) if future_timestamps else None
            end = _timestamp(future_timestamps[-1]) if future_timestamps else None
            history_window_binding = {
                "kind": "forecast_question_scope_unresolved_trigger",
                "basis": (
                    "source states a temporal rule but does not establish "
                    "whether or when its trigger occurs in the horizon"),
                "numeric_authority": False,
                "automation_eligible": False,
            }
        elif timing_status == "atemporal_context":
            # Background rates and cross-variable relationships have no event
            # onset to recover. Bind them to question scope for presentation,
            # but do not pretend that scope is an effective event window or
            # grant deterministic numeric/automation authority.
            start = _timestamp(future_timestamps[0]) if future_timestamps else None
            end = _timestamp(future_timestamps[-1]) if future_timestamps else None
            history_window_binding = {
                "kind": "forecast_question_scope_atemporal_context",
                "basis": (
                    "source states background evidence or a relationship, "
                    "not an event with an effective onset"),
                "numeric_authority": False,
                "automation_eligible": False,
            }
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
            partial = _historical_partial_date_window(span, cutoff_dt)
            if partial is not None:
                start, end, history_window_binding = partial
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
        elif len(numeric_matches) == 2 and re.search(
                r"(?:-|–|—|\bto\b)", confidence_text):
            # Parsing confidence grants no authority. Preserve the grounded
            # claim by taking an explicit range's conservative endpoint.
            confidence = min(map(float, numeric_matches))
            confidence_normalization = {
                "kind": "numeric_range_to_conservative_unit_interval",
                "supplied": raw_confidence,
                "normalized": (confidence / 100.0
                               if 1 < confidence <= 100 else confidence),
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
                # Compiler confidence is descriptive metadata, never evidence
                # authority. Do not discard an otherwise verbatim, dated claim
                # merely because a model emitted an unfamiliar label or shape;
                # retain it at the conservative floor and disclose the repair.
                confidence = 0.25
                confidence_normalization = {
                    "kind": "unparseable_to_conservative_unit_interval",
                    "supplied": str(raw_confidence)[:200],
                    "normalized": confidence,
                    "authority_effect": "none",
                }
        if math.isfinite(confidence) and 1 < confidence <= 100:
            confidence /= 100.0
            if (confidence_normalization or {}).get("kind") \
                    != "numeric_range_to_conservative_unit_interval":
                confidence_normalization = {
                    "kind": "percent_to_unit_interval",
                    "supplied": raw_confidence,
                    "normalized": confidence,
                }
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            reasons.append(f"claim {index + 1} has invalid confidence")
            continue
        association_only = _association_only_claim(span)
        claims.append({
            "claim_id": f"claim-{len(claims) + 1}",
            "source_span": span,
            "relation": relation,
            "effective_start": start.isoformat(),
            "effective_end": end.isoformat(),
            "mechanism": str(claim.get("mechanism") or "")[:500],
            "confidence": confidence,
            "known_at": cutoff_dt.isoformat(),
            "timing_status": timing_status,
            **({
                "relationship_authority": "associational_only",
                "causal_authority": False,
            } if association_only else {}),
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
    verified_bindings: list[dict[str, Any]] = []
    for interpretation in derived_interpretations:
        signature = json.dumps({
            "predicate": interpretation.get("predicate"),
            "claim_ids": sorted(interpretation.get("claim_ids") or []),
        }, sort_keys=True)
        if not any(json.dumps({
                "predicate": (item or {}).get("predicate"),
                "claim_ids": sorted((item or {}).get("claim_ids") or []),
            }, sort_keys=True) == signature
            for item in raw_observation_interpretations
            if isinstance(item, dict)):
            verified_bindings.append(interpretation)
    # Reserve bounded capacity for host-verified bindings. Otherwise a model
    # can accidentally crowd them out with four malformed duplicates before
    # validation even begins.
    raw_observation_interpretations = [
        *verified_bindings, *raw_observation_interpretations]
    observation_interpretations, observation_critique, derived_candidate = \
        _validate_observation_interpretations(
            raw_observation_interpretations, claims=claims,
            history=history, history_timestamps=history_timestamps,
            future_timestamps=future_timestamps)
    from .calibration_counterfactual import compile_additive_drift_repair
    calibration_candidate, calibration_replay = compile_additive_drift_repair(
        context_text=context_text, claims=claims, history=history,
        history_timestamps=history_timestamps,
        future_timestamps=future_timestamps)
    derived_replay = ((derived_candidate or {}).get("conditional_replay") or {})
    derived_replay_admitted = derived_replay.get("selection_eligible") is True
    derived_replay_human_eligible = derived_replay.get(
        "human_recommendation_eligible") is True
    derived_scenario_is_deterministic = bool(
        observation_interpretations and
        ((observation_interpretations[0].get("predicate_normalization") or {}).get(
            "kind") == "semantic_zero_to_separated_near_zero_cluster"))
    use_calibration_candidate = calibration_candidate is not None
    if use_calibration_candidate:
        # Host-validated deterministic evidence owns this lane. A malformed
        # model-authored transformation may be retained as a rejection, but it
        # cannot veto an independently compiled counterfactual.
        candidate_selection_eligible = True
        candidate_selection_reason = None
    use_derived_candidate = bool(not use_calibration_candidate and
        derived_candidate is not None and (
            derived_replay_admitted or derived_replay_human_eligible
            or derived_scenario_is_deterministic
            or raw.get("forecast_candidate") in (None, {})))
    candidate_was_derived_from_observation_interpretation = use_derived_candidate
    if candidate_was_derived_from_observation_interpretation:
        replay = derived_replay
        # Mechanical validity and evidence dominance are distinct. A sealed
        # prior-assisted sensitivity may be chosen by a human-facing governed
        # selector even when it cannot auto-lead; publication separately reads
        # conditional_replay.selection_eligible for evidence dominance.
        candidate_selection_eligible = True
        if replay.get("human_recommendation_eligible") is True \
                and replay.get("selection_eligible") is not True:
            candidate_selection_reason = (
                "Conditional replay improved both governed metrics across two "
                "chronological blocks but missed strict admission; it may lead "
                "best_effort for human review and can never authorize automation.")
        elif replay.get("selection_eligible") is not True:
            candidate_selection_reason = (
                "Historical-contamination filtering is mechanically valid but "
                "did not earn evidence dominance; retain it as a visible, "
                "human-reviewed prior-assisted scenario only.")
    candidate_reason_start = len(reasons)
    candidate_input = (calibration_candidate if use_calibration_candidate else
                       derived_candidate if use_derived_candidate else
                       raw.get("forecast_candidate") or derived_candidate)
    unresolved_claim_ids = {
        str(claim["claim_id"]) for claim in claims
        if claim.get("timing_status") == "unresolved_trigger"
    }
    atemporal_claim_ids = {
        str(claim["claim_id"]) for claim in claims
        if claim.get("timing_status") == "atemporal_context"
    }
    if isinstance(candidate_input, dict) and unresolved_claim_ids:
        cited = {str(item) for item in candidate_input.get("claim_ids") or
                 [claim["claim_id"] for claim in claims]}
        if cited.intersection(unresolved_claim_ids):
            reasons.append(
                "forecast_candidate cites a rule with unresolved trigger timing")
            candidate_input = None
    candidate = _validate_candidate(
        candidate_input, claims=claims,
        future_timestamps=future_timestamps, history=history, reasons=reasons,
        governed_counterfactual_justifies_boundary_jump=(
            use_calibration_candidate or (
                candidate_was_derived_from_observation_interpretation
                and (derived_replay_admitted or derived_replay_human_eligible))))
    if candidate is not None and not (
            use_calibration_candidate
            or candidate_was_derived_from_observation_interpretation):
        cited_ids = {str(value) for value in candidate.get("claim_ids") or []}
        associational_ids = {
            str(claim["claim_id"]) for claim in claims
            if claim.get("relationship_authority") == "associational_only"
        }
        if cited_ids.intersection(associational_ids):
            candidate_selection_eligible = False
            candidate_selection_reason = (
                "A model-authored path cites associational evidence without "
                "a fold-validated relationship executable. Correlation may "
                "remain visible as a scenario but cannot authorize selection.")
    if candidate is not None and \
            candidate_was_derived_from_observation_interpretation:
        candidate["conditional_replay"] = dict(
            derived_candidate.get("conditional_replay") or {})
    if candidate is not None and use_calibration_candidate:
        candidate["calibration_replay"] = dict(calibration_replay)
    if (candidate is not None and not use_derived_candidate
            and observation_interpretations and derived_candidate is not None
            and not derived_replay_admitted):
        candidate_selection_eligible = False
        candidate_selection_reason = (
            "A governed observation counterfactual over the same verified "
            "claims failed conditional replay; a model-authored path cannot "
            "bypass that evidence gate.")
    candidate_reasons = reasons[candidate_reason_start:]
    effect_raw = raw.get("effect_proposal")
    effect_was_derived_from_validated_event = False
    if effect_raw in (None, {}):
        effect_raw = _derived_scale_effect(
            claims, future_timestamps=future_timestamps)
        effect_was_derived_from_validated_event = effect_raw is not None
    if isinstance(effect_raw, dict) and not effect_raw.get("claim_ids") \
            and len(claims) == 1:
        # The caller proposes claims and effects in one response, before
        # Gnomon assigns canonical claim IDs. A single unambiguous claim may
        # therefore be bound deterministically; multiple claims still require
        # explicit citation so the model cannot smuggle in a broad rationale.
        effect_raw = {**effect_raw, "claim_ids": [claims[0]["claim_id"]],
                      "citation_binding": "single_verified_claim"}
    unresolved_effect = bool(
        isinstance(effect_raw, dict) and unresolved_claim_ids.intersection(
            str(item) for item in effect_raw.get("claim_ids") or
            ([claims[0]["claim_id"]] if len(claims) == 1 else [])))
    atemporal_effect = bool(
        isinstance(effect_raw, dict) and atemporal_claim_ids.intersection(
            str(item) for item in effect_raw.get("claim_ids") or
            ([claims[0]["claim_id"]] if len(claims) == 1 else [])))
    effect_proposal, proposal_critique = ((None, {
        "status": "rejected", "attempts_used": 1, "attempts_remaining": 1,
        "attempts": [{"attempt": 1, "accepted": False, "violations": [{
            "code": "UNRESOLVED_TRIGGER_TIMING",
            "message": (
                "A qualitative rule cannot produce a numeric effect until "
                "its trigger date or window is established."),
        }]}],
    }) if unresolved_effect else (None, {
        "status": "rejected", "attempts_used": 1, "attempts_remaining": 1,
        "attempts": [{"attempt": 1, "accepted": False, "violations": [{
            "code": "ATEMPORAL_CONTEXT_NO_NUMERIC_AUTHORITY",
            "message": (
                "Background context may support a labelled prior-assisted "
                "candidate, but cannot directly authorize a numeric effect."),
        }]}],
    }) if atemporal_effect else validate_effect_proposal(
        effect_raw,
        claim_ids={str(claim["claim_id"]) for claim in claims},
        claim_spans={str(claim["claim_id"]): str(claim["source_span"])
                     for claim in claims},
        repair=raw.get("effect_proposal_repair"),
    )) if effect_raw not in (None, {}) else (None, {
        "status": "not_proposed", "attempts_used": 0, "attempts_remaining": 2,
        "attempts": [],
    })
    if effect_proposal is not None:
        if effect_was_derived_from_validated_event:
            effect_proposal["compiler_binding"] = (
                "validated_event_plus_verbatim_scale")
        effect_proposal = _align_effect_onset_to_cited_claim(
            effect_proposal, claims=claims,
            future_timestamps=future_timestamps,
            validated_events=validated_events or [], context_text=context_text)
    hypotheses, hypothesis_critique = compile_context_hypotheses(
        raw.get("hypotheses"), claims=claims,
        series=[str(value) for value in raw.get("series") or ["*"]],
        cutoff=cutoff, repair=raw.get("hypothesis_repair"),
    )
    if not hypotheses:
        fallback_rows = _atemporal_hypothesis_rows(claims, cutoff=cutoff_dt)
        fallback_hypotheses, fallback_critique = compile_context_hypotheses(
            fallback_rows, claims=claims, series=["*"], cutoff=cutoff)
        if fallback_hypotheses:
            original_rejected = list(hypothesis_critique.get("rejected") or [])
            hypothesis_critique = {
                **fallback_critique,
                "status": "accepted_after_deterministic_fallback",
                "rejected": original_rejected,
                "deterministic_fallback": True,
                "fallback_basis": "verified_atemporal_claims",
            }
            hypotheses = fallback_hypotheses
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
                        "transformation. A time-invariant path may use compact "
                        "constant_quantiles instead of repeating every row. "
                        "A shaped path may use cited quantile_anchors at "
                        "meaningful host-grid timestamps; Gnomon interpolates "
                        "between them and retains the immutable primary beyond "
                        "the supplied anchor window."
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
                "calibration_counterfactual" if use_calibration_candidate else
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
            text)) or bool(re.search(
                r"\bwill not (?:have|experience) (?:this |the )?glitch\b",
                text))
    if not ended:
        return []
    for claim in claims:
        span = _normalise(claim.get("source_span"))
        source = str(claim.get("source_span") or "")
        spike = re.search(
            r"(?:glitch|spike).*?starting from\s+"
            r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s+for\s+"
            r"(\d+)\s+hours?", source, re.I)
        if spike:
            start = _timestamp(spike.group(1).replace(" ", "T"))
            if start is not None and start.tzinfo is None:
                effective = _timestamp(claim.get("effective_start"))
                start = start.replace(tzinfo=(effective.tzinfo
                                              if effective else None))
            if start is None or start.tzinfo is None:
                try:
                    start = datetime.fromisoformat(
                        spike.group(1)).replace(tzinfo=_timestamp(
                            claim.get("effective_start")).tzinfo)
                except (TypeError, ValueError, AttributeError):
                    start = None
            if start is not None:
                return [{
                    "kind": "historical_contamination",
                    "claim_ids": [claim["claim_id"]],
                    "predicate": {
                        "op": "timestamp_window",
                        "start": start.isoformat(),
                        "duration_steps": int(spike.group(2)),
                        "unit": "hour",
                    },
                    "window": "cited_window",
                    "rationale": (
                        "deterministically bound from a verified historical "
                        "sensor-glitch window stated not to recur"),
                    "proposal_origin": "verified_claim_semantics",
                }]
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
                clock_window = re.search(
                    r"every\s+day\s+between\s+([0-2]?\d:[0-5]\d)"
                    r"(?:\s*(?::\d{2})?)?\s+and\s+([0-2]?\d:[0-5]\d)",
                    source, re.I)
                predicate = ({
                    "op": "recurring_clock_window",
                    "start_time": clock_window.group(1),
                    "end_time": clock_window.group(2),
                } if clock_window else {"op": "equals", "value": 0.0})
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
        if predicate_op not in {"equals", "recurring_window",
                                "timestamp_window",
                                "recurring_clock_window"}:
            rejected.append({"index": index, "code": "UNSUPPORTED_PREDICATE"})
            continue
        spans = [_normalise(claim["source_span"]) for claim in cited]
        zero_entailed = predicate_op == "timestamp_window" or any(
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
        if predicate_op == "timestamp_window":
            try:
                window_start = _timestamp(predicate["start"])
                if window_start is None:
                    naive = datetime.fromisoformat(str(predicate["start"]))
                    window_start = naive.replace(tzinfo=start.tzinfo)
                duration = int(predicate["duration_steps"])
            except (KeyError, TypeError, ValueError):
                window_start, duration = None, 0
            source = " ".join(str(claim["source_span"]) for claim in cited)
            entailed = (window_start is not None and duration > 0
                        and window_start.strftime("%Y-%m-%d") in source
                        and str(duration) in source
                        and any(token in source.casefold()
                                for token in ("glitch", "spike")))
            if not entailed:
                rejected.append({"index": index,
                                 "code": "SPIKE_WINDOW_NOT_ENTAILED"})
                continue
            window_end = window_start + timedelta(hours=duration)
            mask = [bool((observed := _timestamp(timestamp)) is not None
                         and window_start <= observed < window_end)
                    for timestamp in history_timestamps]
            applied_predicate = {
                "op": "timestamp_window", "start": window_start.isoformat(),
                "end": window_end.isoformat(), "interval": "half_open",
                "source_unit": "hour",
            }
        elif predicate_op == "recurring_window":
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
        elif predicate_op == "recurring_clock_window":
            start_text = str(predicate.get("start_time") or "")
            end_text = str(predicate.get("end_time") or "")
            clock = re.compile(r"^(?:[01]?\d|2[0-3]):[0-5]\d$")
            source = " ".join(spans)
            if (not clock.fullmatch(start_text)
                    or not clock.fullmatch(end_text)
                    or start_text not in source or end_text not in source):
                rejected.append({"index": index,
                                 "code": "CLOCK_WINDOW_NOT_ENTAILED"})
                continue
            start_hour, start_minute = map(int, start_text.split(":"))
            end_hour, end_minute = map(int, end_text.split(":"))
            start_clock = 60 * start_hour + start_minute
            end_clock = 60 * end_hour + end_minute
            if start_clock == end_clock:
                rejected.append({"index": index,
                                 "code": "EMPTY_CLOCK_WINDOW"})
                continue
            # Schedule endpoints are often underspecified. When the claim also
            # states an observable corruption value, repeated pre-cutoff
            # boundary observations may resolve inclusion; never guess from
            # prose or inspect future targets.
            from .future_context import parse_override_span
            stated_corruption, _ = parse_override_span(source)
            include_end = False
            boundary_resolution = None
            if stated_corruption is not None:
                boundary_values = []
                for inside, timestamp, value in zip(
                        in_window, history_timestamps, history):
                    observed = _timestamp(timestamp)
                    if not inside or observed is None:
                        continue
                    minute = 60 * observed.hour + observed.minute
                    if minute == end_clock:
                        boundary_values.append(float(value))
                if len(boundary_values) >= 2:
                    tolerance = max(1.0, abs(stated_corruption)) * 1e-9
                    matches = sum(abs(value - stated_corruption) <= tolerance
                                  for value in boundary_values)
                    ratio = matches / len(boundary_values)
                    if ratio >= .8:
                        include_end = True
                        boundary_resolution = {
                            "kind": "end_inclusion_from_repeated_observations",
                            "stated_corruption_value": stated_corruption,
                            "observations": len(boundary_values),
                            "matching_observations": matches,
                            "match_fraction": ratio,
                            "uses_future_observations": False,
                        }
                    elif .2 < ratio < .8:
                        rejected.append({
                            "index": index,
                            "code": "AMBIGUOUS_CLOCK_WINDOW_BOUNDARY",
                            "boundary": "end",
                            "observations": len(boundary_values),
                            "matching_observations": matches,
                        })
                        continue
            mask = []
            for inside, timestamp in zip(in_window, history_timestamps):
                observed = _timestamp(timestamp)
                if not inside or observed is None:
                    mask.append(False)
                    continue
                minute = 60 * observed.hour + observed.minute
                covered = (start_clock <= minute <= end_clock
                           if include_end and start_clock < end_clock else
                           minute >= start_clock or minute <= end_clock
                           if include_end else
                           start_clock <= minute < end_clock
                           if start_clock < end_clock else
                           minute >= start_clock or minute < end_clock)
                mask.append(covered)
            applied_predicate = {
                "op": "recurring_clock_window",
                "start_time": f"{start_hour:02d}:{start_minute:02d}",
                "end_time": f"{end_hour:02d}:{end_minute:02d}",
                "interval": ("closed_end_empirically_resolved"
                             if include_end else "half_open"),
                "timezone_basis": "history_timestamp_timezone",
                **({"boundary_resolution": boundary_resolution}
                   if boundary_resolution else {}),
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
            if not any(mask) and len(window_values) >= 12:
                # Some measurement pipelines encode semantic zero as a
                # noisy near-zero component rather than literal zeros. Build
                # a *scenario-only* mask only when the observed values have a
                # sharply separated low cluster whose center is compatible
                # with zero. This never earns historical admission because
                # membership was inferred from the outcomes themselves.
                ordered = sorted(window_values)
                minimum_cluster = max(6, math.ceil(len(ordered) * .10))
                candidates = [
                    (ordered[index] - ordered[index - 1], index)
                    for index in range(minimum_cluster,
                                       len(ordered) - minimum_cluster + 1)
                ]
                gap, split = max(candidates, default=(0.0, 0))
                low, high = ordered[:split], ordered[split:]
                low_median = statistics.median(low) if low else math.nan
                high_median = statistics.median(high) if high else math.nan
                low_mad = (statistics.median(
                    abs(value - low_median) for value in low) if low else math.inf)
                high_mad = (statistics.median(
                    abs(value - high_median) for value in high) if high else math.inf)
                # The cited semantic concerns the near-zero component. Normal
                # activity may be legitimately broad, so requiring six high-
                # cluster MADs suppresses obvious censored mixtures. Demand a
                # very tight low component and still require a meaningful gap
                # relative to the normal component.
                separation_floor = max(
                    1e-9, 6.0 * low_mad, 1.5 * high_mad)
                zero_compatible = abs(low_median) <= max(
                    1.0, .15 * abs(high_median))
                if gap >= separation_floor and zero_compatible:
                    threshold = (ordered[split - 1] + ordered[split]) / 2.0
                    mask = [inside and float(value) <= threshold
                            for inside, value in zip(in_window, history)]
                    predicate_normalization = {
                        "kind": "semantic_zero_to_separated_near_zero_cluster",
                        "stated_value": 0.0,
                        "threshold": threshold,
                        "low_cluster_observations": len(low),
                        "high_cluster_observations": len(high),
                        "gap": gap,
                        "required_gap": separation_floor,
                        "basis": (
                            "source states zero activity; a sharply separated "
                            "near-zero component is exposed as a prior-assisted "
                            "sensitivity only, never historical admission"),
                    }
                    applied_predicate = {
                        "op": "separated_near_zero_cluster",
                        "maximum": threshold,
                    }
                elif not any(mask):
                    # Overlapping measurement noise may erase an empty gap.
                    # A bounded deterministic 1-D two-means fit can still
                    # expose a sensitivity, but never evidence: membership is
                    # derived from the target values and remains scenario-only.
                    low_center = ordered[len(ordered) // 4]
                    high_center = ordered[(3 * len(ordered)) // 4]
                    low_cluster: list[float] = []
                    high_cluster: list[float] = []
                    for _ in range(32):
                        midpoint = (low_center + high_center) / 2.0
                        low_cluster = [value for value in window_values
                                       if value <= midpoint]
                        high_cluster = [value for value in window_values
                                        if value > midpoint]
                        if (len(low_cluster) < minimum_cluster
                                or len(high_cluster) < minimum_cluster):
                            break
                        next_low = statistics.mean(low_cluster)
                        next_high = statistics.mean(high_cluster)
                        if (abs(next_low - low_center) <= 1e-12
                                and abs(next_high - high_center) <= 1e-12):
                            low_center, high_center = next_low, next_high
                            break
                        low_center, high_center = next_low, next_high
                    if (len(low_cluster) >= minimum_cluster
                            and len(high_cluster) >= minimum_cluster):
                        low_median = statistics.median(low_cluster)
                        high_median = statistics.median(high_cluster)
                        low_mad = statistics.median(
                            abs(value - low_median) for value in low_cluster)
                        high_mad = statistics.median(
                            abs(value - high_median) for value in high_cluster)
                        center_separation = high_center - low_center
                        required_separation = max(
                            1e-9, 3.0 * low_mad, 3.0 * high_mad)
                        zero_compatible = abs(low_median) <= max(
                            1.0, .15 * abs(high_median))
                        if (center_separation >= required_separation
                                and zero_compatible):
                            threshold = (low_center + high_center) / 2.0
                            mask = [inside and float(value) <= threshold
                                    for inside, value in zip(in_window, history)]
                            predicate_normalization = {
                                "kind": "semantic_zero_to_separated_near_zero_cluster",
                                "method": "deterministic_two_means_v1",
                                "stated_value": 0.0,
                                "threshold": threshold,
                                "low_cluster_observations": len(low_cluster),
                                "high_cluster_observations": len(high_cluster),
                                "center_separation": center_separation,
                                "required_separation": required_separation,
                                "basis": (
                                    "source states zero activity; a separated "
                                    "near-zero mixture component is exposed as "
                                    "a prior-assisted sensitivity only"),
                            }
                            applied_predicate = {
                                "op": "separated_near_zero_cluster",
                                "maximum": threshold,
                            }
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
    from .observation_counterfactual import fit_observation_counterfactual
    candidate, replay = fit_observation_counterfactual(
        history, accepted_masks[0], future_timestamps,
        history_timestamps=history_timestamps)
    normalization = accepted[0].get("predicate_normalization") or {}
    if normalization.get("kind") == \
            "semantic_zero_to_separated_near_zero_cluster":
        replay = {
            **replay,
            "status": "scenario_only_outcome_inferred_mask",
            "selection_eligible": False,
            "admission_withheld_reason": (
                "Cluster membership was inferred from observed target values; "
                "it cannot validate itself under historical replay."),
        }
        if candidate is not None:
            candidate["conditional_replay"] = replay
    accepted[0]["conditional_replay"] = replay
    critique["conditional_replay"] = replay
    return accepted, critique, candidate


def _align_effect_onset_to_cited_claim(
    proposal: dict[str, Any], *, claims: list[dict[str, Any]],
    future_timestamps: list[str], validated_events: list[Any],
    context_text: str,
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
    elif len(distinct_starts) == 0 and len(matching) == 1:
        start = _timestamp(matching[0].get("effective_start"))
        if start is None or not _claim_timing_is_locally_cited(
                matching[0], start=start, context_text=context_text):
            return proposal
        binding = "locally cited claim context and forecast grid"
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


def _claim_timing_is_locally_cited(
        claim: dict[str, Any], *, start: datetime, context_text: str) -> bool:
    """Accept a separated onset only inside the claim's source paragraph."""
    span = str(claim.get("source_span") or "")
    span_pos = context_text.find(span)
    if span_pos < 0:
        return False
    representations = {
        start.isoformat(), start.isoformat().replace("T", " "),
        start.strftime("%Y-%m-%dT%H:%M:%S"),
        start.strftime("%Y-%m-%d %H:%M:%S"),
        start.strftime("%Y-%m-%dT%H:%M"),
        start.strftime("%Y-%m-%d %H:%M"),
    }
    positions = [context_text.find(value) for value in representations
                 if value and context_text.find(value) >= 0]
    for position in positions:
        left = min(position, span_pos)
        right = max(position + 19, span_pos + len(span))
        between = context_text[left:right]
        if right - left <= 1200 and not re.search(r"\n\s*\n", between):
            return True
    return False


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


def _target_relevant_claim_span(span: str, target_name: str | None) -> str | None:
    """Return only clauses that can denote the forecast target.

    Numeric parsing establishes *what value was stated*, not *which variable
    it belongs to*. A driver schedule such as ``speed changes to 1593`` must
    never become an override for a pressure target. Meaningful target names
    therefore require a matching clause; generic target nouns remain useful
    for ordinary prose such as ``output drops to zero``.
    """
    text = " ".join(str(span).split())
    raw_target = str(target_name or "").strip()
    normalized = re.sub(r"[^a-z0-9]+", " ", raw_target.lower())
    target_tokens = {
        token for token in normalized.split()
        if len(token) > 1 and token not in {
            "value", "values", "target", "series", "column", "default",
        }
    }
    generic_target = normalized.strip() in {
        "", "value", "values", "target", "series", "column", "default",
    }
    if generic_target:
        return text
    clauses = re.split(r"(?<=[.!?;])\s+|\s+and\s+", text,
                       flags=re.IGNORECASE)
    # Symbolic series names (X_1, y, A) are common in scientific and API
    # inputs. Token-length filtering deliberately excludes their one-character
    # pieces, so preserve the exact identifier as an ownership signal before
    # falling back to descriptive tokens such as ``pressure``.
    exact_identifier = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(raw_target)}(?![A-Za-z0-9_])",
        re.IGNORECASE) if raw_target else None
    matching = [clause for clause in clauses
                if ((exact_identifier is not None
                     and exact_identifier.search(clause))
                    or target_tokens.intersection(
                        re.findall(r"[a-z0-9]+", clause.lower())))]
    if matching:
        # Numeric suffixes in identifiers (X_1, store_12_sales) are labels,
        # not candidate values. Remove the exact identifier from the numeric
        # parser's input after it has established clause ownership.
        if exact_identifier is not None:
            matching = [exact_identifier.sub("target", clause)
                        for clause in matching]
        return " ".join(matching)
    if re.search(
            r"\b(?:output|readings?|observations?|forecast(?:ed)?\s+value|"
            r"target(?:\s+value)?|demand|sales|traffic|requests?)\b",
            text, re.IGNORECASE):
        return text
    return None


def deterministic_events_from_claims(
        dossier: dict[str, Any], *,
        target_name: str | None = None,
        target_verified_spans: set[str] | None = None,
        ) -> list[dict[str, Any]]:
    """Promote only literally stated absolute states into event proposals.

    The LLM locates and dates the verbatim span; Gnomon's existing parser must
    independently recover an absolute value. Qualitative effects remain
    scenarios. Returned objects intentionally re-enter the ordinary context
    validator rather than bypassing it.
    """
    from .future_context import parse_bound_span, parse_override_span

    events = []
    for index, claim in enumerate(dossier.get("claims") or [], 1):
        window_binding = claim.get("effective_window_binding") or {}
        if (window_binding.get("numeric_authority") is False
                and window_binding.get("kind") !=
                "explicit_source_timing_reconciled"):
            continue
        span = str(claim.get("source_span") or "")
        parse_span = (" ".join(span.split())
                      if span in (target_verified_spans or set())
                      else _target_relevant_claim_span(span, target_name))
        if parse_span is None:
            continue
        if claim.get("relation") == "constrains_range":
            bound, problem = parse_bound_span(parse_span)
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
                    "deterministic_parse_span": parse_span,
                    "derived_from_claim_id": claim.get("claim_id") or f"claim-{index}",
                })
                continue
        value, problem = parse_override_span(parse_span)
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
            "deterministic_parse_span": parse_span,
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
    governed_counterfactual_justifies_boundary_jump: bool = False,
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
    path_normalization = None
    compact = raw.get("constant_quantiles")
    anchors = raw.get("quantile_anchors")
    anchor_source = "quantile_anchors"
    primary_completion = False
    validated_anchor_rows: list[dict[str, float | str]] = []
    if (not isinstance(anchors, list) and isinstance(rows, list)
            and 2 <= len(rows) < len(future_timestamps)):
        # Models naturally use the established `quantiles` row shape for a
        # sparse path. Treat it as the compact anchor representation only when
        # the strict boundary/grid checks below succeed, and disclose the alias.
        anchors = rows
        anchor_source = "sparse_quantiles_alias"
    if isinstance(anchors, list):
        index_by_timestamp = {
            timestamp: index for index, timestamp in enumerate(future_timestamps)}
        parsed_anchors: list[tuple[int, dict[str, float]]] = []
        seen: set[int] = set()
        for anchor in anchors:
            if not isinstance(anchor, dict):
                reasons.append("forecast_candidate quantile anchor is not an object")
                return None
            timestamp = str(anchor.get("timestamp") or "")
            index = index_by_timestamp.get(timestamp)
            if index is None or index in seen:
                reasons.append(
                    "forecast_candidate anchors must use unique requested timestamps")
                return None
            try:
                quantiles = {key: float(anchor[key])
                             for key in ("q10", "q50", "q90")}
            except (KeyError, TypeError, ValueError):
                reasons.append("forecast_candidate quantile anchor lacks quantiles")
                return None
            if not all(math.isfinite(value) for value in quantiles.values()) \
                    or not quantiles["q10"] <= quantiles["q50"] <= quantiles["q90"]:
                reasons.append("forecast_candidate quantile anchor is invalid")
                return None
            parsed_anchors.append((index, quantiles)); seen.add(index)
        parsed_anchors.sort(key=lambda item: item[0])
        if not parsed_anchors:
            reasons.append("forecast_candidate requires at least one valid anchor")
            return None
        rows = []
        for index in range(len(future_timestamps)):
            exact = next((values for anchor_index, values in parsed_anchors
                          if anchor_index == index), None)
            if exact is not None:
                values = exact
            elif index < parsed_anchors[0][0]:
                values = parsed_anchors[0][1]
            elif index > parsed_anchors[-1][0]:
                values = parsed_anchors[-1][1]
            else:
                left_index, left = next(
                    item for item in reversed(parsed_anchors) if item[0] <= index)
                right_index, right = next(
                    item for item in parsed_anchors if item[0] >= index)
                width = right_index - left_index
                weight = (index - left_index) / width
                values = {key: left[key] + (right[key] - left[key]) * weight
                          for key in ("q10", "q50", "q90")}
            rows.append(dict(values))
        primary_completion = (
            parsed_anchors[0][0] != 0
            or parsed_anchors[-1][0] != len(future_timestamps) - 1)
        path_normalization = {
            "kind": "quantile_anchors_linearly_interpolated_on_host_grid",
            "anchors": len(parsed_anchors), "steps": len(future_timestamps),
            "source_field": anchor_source,
        }
        if primary_completion:
            path_normalization["unanchored_edges"] = "immutable_primary"
        validated_anchor_rows = [{"timestamp": future_timestamps[index], **values}
                                 for index, values in parsed_anchors]
    elif isinstance(compact, dict):
        rows = [dict(compact) for _ in future_timestamps]
        path_normalization = {
            "kind": "constant_quantiles_expanded_to_host_grid",
            "steps": len(future_timestamps),
        }
    elif isinstance(rows, list) and len(rows) == 1 and \
            raw.get("repeat_across_horizon") is True:
        rows = [dict(rows[0]) for _ in future_timestamps]
        path_normalization = {
            "kind": "single_quantile_row_repeated_on_host_grid",
            "steps": len(future_timestamps),
        }
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
    if (boundary_jump > MAX_BOUNDARY_JUMP_SCALES
            and not bounded_regime_change
            and not governed_counterfactual_justifies_boundary_jump):
        reasons.append("forecast_candidate failed boundary-jump plausibility")
        return None
    if (boundary_jump > MAX_BOUNDARY_JUMP_SCALES
            and governed_counterfactual_justifies_boundary_jump):
        plausibility_warnings.append(
            "candidate leaves the raw history boundary because a governed "
            "counterfactual supplied its own validated correction evidence; "
            "support and automation authority remain unchanged")
    elif boundary_jump > MAX_BOUNDARY_JUMP_SCALES:
        plausibility_warnings.append(
            "candidate leaves the empirical history scale but remains inside "
            "a cited numeric bound; treat it as prior-assisted only")
    if path_scale_ratio > MAX_PATH_SCALE_RATIO:
        reasons.append("forecast_candidate failed path-scale plausibility")
        return None
    return {
        "quantiles": clean,
        # Forecast points are accepted as a sealed prior-assisted candidate,
        # not as proof for whatever empirical story the model attached to
        # them. Keep the prose for audit under an explicitly unverified key;
        # public assumptions use the bounded provenance statement instead.
        "provenance_class": "model_authored_prior",
        "rationale": (
            "Model-authored prior-assisted forecast conditioned on verified "
            "claims; not calibrated against supplied historical outcomes."),
        **({"model_rationale_unverified": rationale} if rationale else {}),
        "claim_ids": [claim["claim_id"] for claim in claims],
        "plausibility": {
            "boundary_jump_scales": round(boundary_jump, 6),
            "path_scale_ratio": round(path_scale_ratio, 6),
            "entailed_bounds": {"minimum": lower, "maximum": upper,
                                "claim_ids": bound_claim_ids},
            "warnings": plausibility_warnings,
            "uncertainty_normalization": uncertainty_normalization,
        },
        **({"path_normalization": path_normalization}
           if path_normalization else {}),
        **({"quantile_anchors": validated_anchor_rows,
            "requires_primary_completion": True}
           if primary_completion else {}),
    }

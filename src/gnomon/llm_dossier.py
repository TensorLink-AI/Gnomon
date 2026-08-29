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


def deterministic_dated_multiplier_dossier(
    context_text: str, *, cutoff: str, future_timestamps: list[str],
    target_name: str | None = None,
) -> dict[str, Any] | None:
    """Compile one explicit future timestamp, duration, and level multiple.

    This deliberately handles only the high-precision intersection of three
    source-stated facts. It does not infer an event window, magnitude, or
    target from qualitative prose. Ambiguous documents remain the LLM
    compiler's job, while explicit promotions, outages, launches, and weather
    events remain usable during provider failure.
    """
    from .future_context import parse_override_scale

    text = " ".join(str(context_text or "").split())
    if not text:
        return None
    multiplier, problem = parse_override_scale(text)
    if problem is not None or multiplier is None or multiplier <= 0:
        return None
    timestamps = re.findall(
        r"\b\d{4}-\d{2}-\d{2}[ T]\d{1,2}:\d{2}(?::\d{2})?"
        r"(?:Z|[+-]\d{2}:?\d{2})?\b", text)
    duration_matches = re.findall(
        r"\b(?:last(?:ed|s)?|continu(?:ed|es)|for)\s+(?:for\s+)?"
        r"(?:approximately\s+|about\s+|roughly\s+)?"
        r"(\d+(?:\.\d+)?)\s*(minutes?|hours?|days?)\b",
        text, flags=re.IGNORECASE)
    if len(timestamps) != 1 or len(duration_matches) != 1:
        return None
    target_terms = (
        "consumption", "demand", "sales", "traffic", "requests",
        "withdrawals", "output", "usage", "readings", "transactions",
        "target value", "forecast value",
    )
    normalized_target = _normalise(target_name)
    if (normalized_target and not normalized_target.isdigit()
            and normalized_target not in _normalise(text)
            and not any(term in _normalise(text) for term in target_terms)):
        return None
    if (not normalized_target or normalized_target.isdigit()) and not any(
            term in _normalise(text) for term in target_terms):
        return None

    future = [_timestamp(value) for value in future_timestamps]
    cutoff_time = _timestamp(cutoff)
    if (not future or cutoff_time is None
            or any(value is None for value in future)):
        return None
    raw_start = timestamps[0].replace(" ", "T")
    try:
        start = datetime.fromisoformat(raw_start)
    except ValueError:
        return None
    if start.tzinfo is None:
        start = start.replace(tzinfo=cutoff_time.tzinfo)
    amount = float(duration_matches[0][0])
    unit = duration_matches[0][1].casefold()
    seconds = amount * (60 if unit.startswith("minute") else
                        3600 if unit.startswith("hour") else 86400)
    if not math.isfinite(seconds) or seconds <= 0:
        return None
    end = start + timedelta(seconds=seconds)
    active = [index for index, stamp in enumerate(future)
              if stamp is not None and start <= stamp < end]
    if (not active or active != list(range(active[0], active[-1] + 1))
            or start <= cutoff_time):
        return None
    approximate = bool(re.search(
        r"\b(?:approximately|about|roughly)\b", text, re.IGNORECASE))
    relation = ("supports_increase" if multiplier > 1 else
                "supports_decrease" if multiplier < 1 else
                "supports_stability")
    delta = float(multiplier) - 1.0
    return {
        "events": [],
        "claims": [{
            "source_span": context_text,
            "relation": relation,
            "effective_start": start.isoformat(),
            "effective_end": end.isoformat(),
            "mechanism": "explicit dated baseline multiplier",
            "confidence": .75 if approximate else 1.0,
        }],
        "hypotheses": [], "covariate_tables": [], "transformations": [],
        "observation_interpretations": [], "forecast_candidate": None,
        "effect_proposal": {
            "shape": ("temporary_pulse" if len(active) < len(future)
                      else "level_shift"),
            "unit": "fraction_of_level",
            "location": delta, "lower": delta, "upper": delta,
            "confidence": .75 if approximate else 1.0,
            "delay_steps": active[0], "duration_steps": len(active),
            "scope": {"kind": "single_series", "series": ["*"]},
            "claim_ids": ["claim-1"],
            "rationale": (
                "Deterministic compilation of one verbatim future timestamp, "
                "duration, and baseline multiplier."),
        },
    }


def deterministic_dated_zero_window_dossier(
    context_text: str, *, cutoff: str, future_timestamps: list[str],
    target_name: str | None = None,
) -> dict[str, Any] | None:
    """Compile one explicit future window whose target activity is zero.

    This front door covers hard operational states such as no sales during a
    closure, no withdrawals while cash is depleted, zero production during an
    outage, or no requests while a service is disabled. It requires one exact
    start, one duration, and a phrase that the existing conservative override
    parser independently resolves to zero. Operational causes alone never
    imply zero.
    """
    from .future_context import parse_override_span

    text = " ".join(str(context_text or "").split())
    if not text:
        return None
    value, problem = parse_override_span(text)
    if problem is not None or value != 0:
        return None
    timestamps = re.findall(
        r"\b\d{4}-\d{2}-\d{2}[ T]\d{1,2}:\d{2}(?::\d{2})?"
        r"(?:Z|[+-]\d{2}:?\d{2})?\b", text)
    durations = re.findall(
        r"\b(?:last(?:ed|s)?|continu(?:ed|es)|for)\s+(?:for\s+)?"
        r"(?:approximately\s+|about\s+|roughly\s+)?"
        r"(\d+(?:\.\d+)?)\s*(minutes?|hours?|days?)\b",
        text, flags=re.IGNORECASE)
    if len(timestamps) != 1 or len(durations) != 1:
        return None
    target_terms = (
        "production", "output", "traffic", "flow", "generation",
        "withdrawals", "transactions", "sales", "arrivals", "departures",
        "rides", "trips", "requests", "visitors", "customers",
        "passengers", "calls", "orders", "deliveries", "operations",
        "activity", "usage", "demand", "consumption", "readings",
    )
    normalized = _normalise(text)
    normalized_target = _normalise(target_name)
    if (normalized_target and not normalized_target.isdigit()
            and normalized_target not in normalized
            and not any(term in normalized for term in target_terms)):
        return None
    if (not normalized_target or normalized_target.isdigit()) and not any(
            term in normalized for term in target_terms):
        return None
    cutoff_time = _timestamp(cutoff)
    future = [_timestamp(value) for value in future_timestamps]
    if cutoff_time is None or not future or any(value is None for value in future):
        return None
    try:
        start = datetime.fromisoformat(timestamps[0].replace(" ", "T"))
    except ValueError:
        return None
    if start.tzinfo is None:
        start = start.replace(tzinfo=cutoff_time.tzinfo)
    amount = float(durations[0][0])
    unit = durations[0][1].casefold()
    seconds = amount * (60 if unit.startswith("minute") else
                        3600 if unit.startswith("hour") else 86400)
    if not math.isfinite(seconds) or seconds <= 0 or start <= cutoff_time:
        return None
    exclusive_end = start + timedelta(seconds=seconds)
    active = [index for index, stamp in enumerate(future)
              if stamp is not None and start <= stamp < exclusive_end]
    if not active or active != list(range(active[0], active[-1] + 1)):
        return None
    # Public event windows are inclusive. Bind the stated duration to the
    # actual host grid so a ten-day window cannot silently cover eleven daily
    # observations.
    effective_start = future[active[0]]
    effective_end = future[active[-1]]
    assert effective_start is not None and effective_end is not None
    return {
        "events": [],
        "claims": [{
            "source_span": context_text, "relation": "supports_decrease",
            "effective_start": effective_start.isoformat(),
            "effective_end": effective_end.isoformat(),
            "mechanism": "explicit dated zero-activity window",
            "confidence": 1.0,
        }],
        "hypotheses": [], "covariate_tables": [], "transformations": [],
        "observation_interpretations": [], "forecast_candidate": None,
        "effect_proposal": None,
    }


def deterministic_dated_directional_event_dossier(
    context_text: str, *, cutoff: str, future_timestamps: list[str],
    target_name: str | None = None,
) -> dict[str, Any] | None:
    """Retain one explicitly dated qualitative target direction.

    The output is deliberately non-numeric: it makes a user's dated calendar
    fact available to the scenario/candidate lane while leaving magnitude,
    support, and automation authority untouched. Ambiguous dates, hedged
    direction, and prose naming only a driver are refused.
    """
    text = " ".join(str(context_text or "").split())
    normalized = _normalise(text)
    if not text or re.search(
            r"\b(?:may|might|could|possibly|perhaps)\b", normalized):
        return None
    sentences = [item.strip() for item in re.split(
        r"(?<=[.!?])\s+|[\r\n]+", text) if item.strip()]
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
    unique_dates = list(dict.fromkeys(dates))
    if len(unique_dates) != 1:
        # Operational prompts often enumerate the whole requested grid before
        # identifying one exceptional calendar day. Bind only a date in a
        # sentence that explicitly names the holiday; never guess among the
        # other listed dates.
        holiday_dates = []
        for sentence in sentences:
            if re.search(r"\bholiday\b", sentence, re.I):
                holiday_dates.extend(re.findall(
                    r"\b\d{4}-\d{2}-\d{2}\b", sentence))
        unique_dates = list(dict.fromkeys(holiday_dates))
    if len(unique_dates) != 1:
        return None
    event_date = unique_dates[0]
    cited_sentences = [sentence for sentence in sentences if (
        event_date in sentence and re.search(r"\bholiday\b", sentence, re.I))
        or (re.search(r"\bholiday\b", sentence, re.I)
            and re.search(r"\b(?:reduce|decrease|decline|drop|lower|increase|"
                         r"rise|grow|higher|surge)\w*\b", sentence, re.I))]
    evidence_text = " ".join(cited_sentences) or context_text
    target_terms = (
        "production", "output", "traffic", "flow", "generation",
        "withdrawals", "transactions", "sales", "arrivals", "departures",
        "rides", "trips", "requests", "visitors", "customers",
        "passengers", "calls", "orders", "deliveries", "operations",
        "activity", "usage", "demand", "consumption", "readings",
    )
    normalized_target = _normalise(target_name)
    target_owned = bool(
        (normalized_target and not normalized_target.isdigit()
         and normalized_target in normalized)
        or any(term in normalized for term in target_terms))
    if not target_owned:
        return None
    decrease = re.search(
        r"\b(?:traffic|production|output|flow|generation|withdrawals|"
        r"transactions|sales|arrivals|departures|rides|trips|requests|"
        r"visitors|customers|passengers|calls|orders|deliveries|operations|"
        r"activity|usage|demand|consumption|readings?)\b.{0,60}"
        r"\b(?:reduce[sd]?|decrease[sd]?|decline[sd]?|drop[sp]?|lower)\b",
        normalized)
    increase = re.search(
        r"\b(?:traffic|production|output|flow|generation|withdrawals|"
        r"transactions|sales|arrivals|departures|rides|trips|requests|"
        r"visitors|customers|passengers|calls|orders|deliveries|operations|"
        r"activity|usage|demand|consumption|readings?)\b.{0,60}"
        r"\b(?:increase[sd]?|rise[sn]?|grow(?:s|th)?|higher|surge[sd]?)\b",
        normalized)
    if bool(decrease) == bool(increase):
        return None
    cutoff_time = _timestamp(cutoff)
    future = [_timestamp(value) for value in future_timestamps]
    if cutoff_time is None or not future or any(value is None for value in future):
        return None
    active = [index for index, stamp in enumerate(future)
              if stamp is not None and stamp.date().isoformat() == event_date]
    if not active or active != list(range(active[0], active[-1] + 1)):
        return None
    start, end = future[active[0]], future[active[-1]]
    assert start is not None and end is not None
    if start <= cutoff_time:
        return None
    direction = "decrease" if decrease else "increase"
    relation = "supports_decrease" if decrease else "supports_increase"
    event_type = "calendar:stated_directional_effect"
    return {
        "events": [{
            "event_type": event_type, "document_index": 0,
            "entity_scope": [target_name or "*"],
            "effective_start": start.isoformat(),
            "effective_end": end.isoformat(),
            "confidence": .75, "status": "confirmed",
            "evidence_quote": evidence_text,
            "effect_family": "temporary_pulse", "direction": direction,
            "duration": "temporary", "entity_kind": "calendar",
        }],
        "claims": [{
            "source_span": evidence_text, "relation": relation,
            "effective_start": start.isoformat(),
            "effective_end": end.isoformat(),
            "mechanism": "explicit dated qualitative target direction",
            "confidence": .75,
        }],
        "hypotheses": [], "covariate_tables": [], "transformations": [],
        "observation_interpretations": [], "forecast_candidate": None,
        "effect_proposal": None,
    }


def deterministic_external_reference_point_dossier(
    context_text: str, *, cutoff: str, target_name: str | None = None,
) -> dict[str, Any] | None:
    """Retain one dated numeric observation from a comparable entity.

    A single external reference point cannot determine a target trajectory or
    establish historical skill. This parser therefore emits only a cited
    historical-analogue hypothesis. It exists to avoid losing useful context
    to a provider/schema failure and to route any numeric interpretation into
    the separately sealed prior-assisted candidate lane.
    """
    text = " ".join(str(context_text or "").split())
    normalized = _normalise(text)
    if not text or not re.search(
            r"\b(?:reference|comparable|similar|peer)\b", normalized):
        return None
    target_terms = (
        "production", "output", "traffic", "flow", "generation",
        "withdrawals", "transactions", "sales", "arrivals", "departures",
        "rides", "trips", "requests", "visitors", "customers",
        "passengers", "calls", "orders", "deliveries", "operations",
        "activity", "usage", "demand", "consumption", "readings", "power",
        "rate", "volume", "revenue", "price",
    )
    normalized_target = _normalise(target_name)
    if not ((normalized_target and not normalized_target.isdigit()
             and normalized_target in normalized)
            or any(term in normalized for term in target_terms)):
        return None
    if not re.search(
            r"\b(?:was|were|reached|recorded|reported|observed|measured|"
            r"peaked|maximum|maximal|minimum|minimal)\b", normalized):
        return None
    # Remove calendar and clock quantities before requiring exactly one
    # externally observed value. Identifier/date digits must never become the
    # analogue measurement by accident.
    stripped = re.sub(
        r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?)?\b",
        " ", text)
    stripped = re.sub(
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\b",
        " ", stripped, flags=re.IGNORECASE)
    stripped = re.sub(
        r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]m)?\b", " ", stripped,
        flags=re.IGNORECASE)
    values = re.findall(
        r"(?<![A-Za-z0-9_.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?![A-Za-z0-9_.])",
        stripped)
    if len(values) != 1 or not math.isfinite(float(values[0])):
        return None
    date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    partial_match = re.search(
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\b",
        text, flags=re.IGNORECASE)
    reference_date = (date_match.group(0) if date_match else
                      partial_match.group(0) if partial_match else None)
    if reference_date is None or _timestamp(cutoff) is None:
        return None
    return {
        "events": [],
        "claims": [{
            "source_span": context_text, "relation": "unknown",
            "effective_start": reference_date,
            "effective_end": reference_date,
            "mechanism": "explicit external reference observation",
            "confidence": .5,
        }],
        "hypotheses": [{
            "kind": "historical_analogue", "claim_ids": ["claim-1"],
            "target_series": ["*"], "predictor_series": None,
            "known_at": cutoff, "lag_steps": 0, "direction": "unknown",
            "rationale": (
                "One comparable-entity observation is retained as a prior; "
                "it is not target-history validation."),
        }],
        "covariate_tables": [], "transformations": [],
        "observation_interpretations": [], "forecast_candidate": None,
        "effect_proposal": None,
    }


def deterministic_ended_recurring_disruption_dossier(
    context_text: str, *, cutoff: str,
) -> dict[str, Any] | None:
    """Preserve an explicit historical schedule stated not to continue.

    No target effect is inferred from words such as maintenance or closure.
    The cited schedule identifies affected historical rows, and the existing
    observation-counterfactual executable must learn any numeric consequence
    through chronological replay. The primary remains authoritative unless
    that replay earns the human recommendation gate.
    """
    text = " ".join(str(context_text or "").split())
    schedule = re.search(
        r"\b(?:maintenance|an? outage|clos(?:ed|ure)|unavailable)\b"
        r".{0,80}?for\s+(\d+)\s+(days?|hours?)"
        r".{0,80}?every\s+(\d+)\s+(days?|hours?)"
        r".{0,100}?starting\s+(?:from\s+)?"
        r"(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?)",
        text, re.I)
    ended = re.search(
        r"\b(?:will not|won't)\s+(?:be\s+)?(?:in|under|on)?\s*"
        r"(?:maintenance|an? outage|closed|unavailable)\b.{0,30}\bfuture\b",
        text, re.I)
    cutoff_dt = _timestamp(cutoff)
    if not schedule or not ended or cutoff_dt is None:
        return None
    try:
        start = datetime.fromisoformat(schedule.group(5).replace(" ", "T"))
    except ValueError:
        return None
    if start.tzinfo is None:
        start = start.replace(tzinfo=cutoff_dt.tzinfo)
    duration, period = int(schedule.group(1)), int(schedule.group(3))
    duration_unit = schedule.group(2).lower().rstrip("s")
    period_unit = schedule.group(4).lower().rstrip("s")
    if duration_unit != period_unit:
        return None
    if start > cutoff_dt or duration <= 0 or period < duration:
        return None
    return {
        "events": [],
        "claims": [{
            "source_span": context_text,
            "relation": "unknown",
            "effective_start": start.isoformat(),
            "effective_end": cutoff_dt.isoformat(),
            "mechanism": (
                "explicit historical recurring disruption stated not to "
                "continue; target effect not inferred"),
            "confidence": 1.0,
        }],
        "hypotheses": [{
            "kind": "regime_shift", "claim_ids": ["claim-1"],
            "target_series": ["*"], "predictor_series": None,
            "known_at": cutoff_dt.isoformat(), "lag_steps": 0,
            "direction": "unknown",
            "rationale": (
                f"A historical {duration}-{schedule.group(2).lower()} "
                f"disruption recurred every {period} "
                f"{schedule.group(4).lower()} and is stated not to continue. "
                "Its numeric target effect has not been inferred."),
        }],
        "covariate_tables": [], "transformations": [],
        "observation_interpretations": [{
            "kind": "historical_contamination",
            "claim_ids": ["claim-1"],
            "predicate": {
                "op": "recurring_window", "start": start.isoformat(),
                "duration_steps": duration, "period_steps": period,
                "unit": duration_unit,
            },
            "window": "cited_window",
            "rationale": (
                "The cited recurring disruption schedule is excluded from "
                "the conditional fit without assuming a numeric effect."),
            "proposal_origin": "verified_claim_semantics",
        }], "effect_proposal": None,
        "forecast_candidate": None,
    }


def deterministic_reference_power_dossier(
    context_text: str, *, cutoff: str, driver_names: list[str],
) -> dict[str, Any] | None:
    """Preserve an explicit reference-normalized power law without an LLM.

    This front door intentionally stops short of creating a numeric path.  It
    recognizes only a source-stated proportional square/cube relationship,
    two reference values, one unambiguous host-known driver, and an ordered
    schedule of stated future transitions. The resulting relationship can support a
    separately sealed prior-assisted candidate, but never automation.
    """
    text = " ".join(str(context_text or "").split())
    cutoff_dt = _timestamp(cutoff)
    if not text or cutoff_dt is None:
        return None
    exponent_match = re.search(
        r"\bproportional\s+to\b.{0,80}?\b(square|squared|quadratic|"
        r"cube|cubed|cubic)\b", text, re.I)
    if exponent_match is None:
        return None
    mentioned = [name for name in driver_names
                 if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.I)]
    if len(mentioned) != 1:
        return None
    driver = mentioned[0]
    reference_values = re.findall(
        r"\bmax(?:imal|imum)?\s+([A-Za-z_][\w -]{0,30}?)\s+(?:is|=)\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([A-Za-z/%]+)", text, re.I)
    if len(reference_values) < 2:
        return None
    fragments = [fragment.strip() for fragment in re.split(
        r"(?<=[.!?])\s+", str(context_text)) if fragment.strip()]
    relationship_span = next((fragment for fragment in fragments
                              if exponent_match.group(0).casefold()
                              in fragment.casefold()), None)
    reference_span = next((fragment for fragment in fragments
                           if len(re.findall(
                               r"\bmax(?:imal|imum)?\s+"
                               r"[A-Za-z_][\w -]{0,30}?\s+(?:is|=)\s*"
                               r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*"
                               r"[A-Za-z/%]+", fragment, re.I)) >= 2), None)
    transition_pattern = re.compile(
        r"\b(?:at\s+)?(\d{1,2}:\d{2}(?::\d{2})?)\b.{0,70}?"
        r"(?:changes?|moves?|ramps?|transitions?)\s+to\s+"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))", re.I)
    future_transitions: list[tuple[datetime, str, float]] = []
    for fragment in fragments:
        match = transition_pattern.search(fragment)
        if match is None:
            continue
        clock = datetime.fromisoformat(
            f"{cutoff_dt.date().isoformat()}T{match.group(1)}")
        if cutoff_dt.tzinfo is not None:
            clock = clock.replace(tzinfo=cutoff_dt.tzinfo)
        if clock >= cutoff_dt:
            future_transitions.append(
                (clock, fragment, float(match.group(2))))
    future_transitions.sort(key=lambda item: item[0])
    transition_span = (future_transitions[0][1]
                       if future_transitions else None)
    if not relationship_span or not reference_span or not transition_span:
        return None
    exponent = 2 if exponent_match.group(1).casefold().startswith(
        ("squ", "quad")) else 3
    alias_match = re.search(
        rf"\b([A-Za-z][\w-]*)\s*\(\s*{re.escape(driver)}\s*\)",
        text, re.I)
    aliases = {driver.casefold(), *(part for part in driver.casefold().split("_")
                                    if len(part) > 2)}
    if alias_match:
        aliases.add(alias_match.group(1).casefold())
    driver_references = [item for item in reference_values
                         if any(alias in item[0].casefold()
                                for alias in aliases)]
    output_references = [item for item in reference_values
                         if item not in driver_references]
    if len(driver_references) != 1 or len(output_references) != 1:
        return None
    transition_claims = [{
        "source_span": span,
        # This is a predictor value, not a target override. The typed
        # relationship below carries its meaning.
        "relation": "unknown",
        "effective_start": stamp.isoformat(),
        "effective_end": stamp.isoformat(),
        "mechanism": "source-stated future driver transition",
        "confidence": 1.0,
    } for stamp, span, _ in future_transitions]
    transition_claim_ids = [
        f"claim-{index}" for index in range(3, 3 + len(transition_claims))]
    return {
        "events": [],
        "claims": [
            {
                "source_span": relationship_span,
                "relation": "supports_increase",
                "effective_start": None, "effective_end": None,
                "timing_status": "atemporal_context",
                "mechanism": f"reference-normalized power law (exponent {exponent})",
                "confidence": 1.0,
            },
            {
                "source_span": reference_span,
                "relation": "constrains_range",
                "effective_start": None, "effective_end": None,
                "timing_status": "atemporal_context",
                "mechanism": "source-stated input and output reference values",
                "confidence": 1.0,
            },
            *transition_claims,
        ],
        "hypotheses": [{
            "kind": "relationship",
            "claim_ids": ["claim-1", "claim-2", *transition_claim_ids],
            "target_series": ["*"], "predictor_series": driver,
            "known_at": cutoff_dt.isoformat(), "lag_steps": 0,
            "direction": "increase",
            "rationale": (
                "The source states a bounded reference-normalized power law "
                "and a future driver transition. Transition dynamics remain "
                "uncertain and require a sealed conditional path."),
        }],
        "covariate_tables": [], "transformations": [],
        "observation_interpretations": [], "effect_proposal": None,
        "forecast_candidate": None,
        "_reference_power_spec": {
            "driver": driver, "exponent": exponent,
            "input_reference": float(driver_references[0][1]),
            "input_unit": driver_references[0][2],
            "output_reference": float(output_references[0][1]),
            "output_unit": output_references[0][2],
            # Compatibility scalar for old callers; execution consumes the
            # complete ordered schedule below.
            "future_driver_endpoint": future_transitions[0][2],
            "future_transitions": [{
                "timestamp": stamp.isoformat(), "value": value,
                "source_span": span,
            } for stamp, span, value in future_transitions],
        },
    }


def deterministic_named_driver_relationship_dossier(
    context_text: str, *, cutoff: str, driver_names: list[str],
) -> dict[str, Any] | None:
    """Preserve one explicitly named but numerically incomplete driver law.

    A named scientific/business rule may be useful to a model even when the
    document does not state executable coefficients. With exactly one
    observed companion, this routine preserves the relationship and future
    driver transition as a prior-only hypothesis. It never supplies the
    missing equation, numeric target path, support upgrade, or automation.
    """
    if len(driver_names) != 1:
        return None
    text = " ".join(str(context_text or "").split())
    cutoff_dt = _timestamp(cutoff)
    if not text or cutoff_dt is None:
        return None
    relationship = re.search(
        r"\b(?:estimated|predicted|calculated|derived|forecast)\b.{0,50}?"
        r"\bfrom\b.{0,80}?\busing\b.{0,60}?"
        r"\b(?:law|laws|model|equation|formula|relationship)\b", text, re.I)
    if relationship is None:
        return None
    transition_pattern = re.compile(
        r"\b(?:at\s+)?(\d{1,2}:\d{2}(?::\d{2})?)\b.{0,70}?"
        r"(?:changes?|moves?|ramps?|transitions?)\s+to\s+"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))", re.I)
    fragments = [fragment.strip() for fragment in re.split(
        r"(?<=[.!?])\s+", str(context_text)) if fragment.strip()]
    relationship_span = next((fragment for fragment in fragments
                              if relationship.group(0).casefold()
                              in fragment.casefold()), None)
    future_transitions = []
    for fragment in fragments:
        match = transition_pattern.search(fragment)
        if match is None:
            continue
        stamp = datetime.fromisoformat(
            f"{cutoff_dt.date().isoformat()}T{match.group(1)}")
        if cutoff_dt.tzinfo is not None:
            stamp = stamp.replace(tzinfo=cutoff_dt.tzinfo)
        if stamp >= cutoff_dt:
            future_transitions.append((stamp, fragment, float(match.group(2))))
    selected_transition = (min(future_transitions, key=lambda item: item[0])
                           if future_transitions else None)
    transition_span = selected_transition[1] if selected_transition else None
    if relationship_span is None or transition_span is None:
        return None
    driver = driver_names[0]
    return {
        "events": [],
        "claims": [{
            "source_span": relationship_span, "relation": "unknown",
            "effective_start": None, "effective_end": None,
            "timing_status": "atemporal_context",
            "mechanism": "source-named driver law with unstated parameters",
            "confidence": 1.0,
        }, {
            "source_span": transition_span, "relation": "unknown",
            "effective_start": cutoff_dt.isoformat(),
            "effective_end": cutoff_dt.isoformat(),
            "mechanism": "source-stated future driver transition",
            "confidence": 1.0,
        }],
        "hypotheses": [{
            "kind": "relationship", "claim_ids": ["claim-1", "claim-2"],
            "target_series": ["*"], "predictor_series": driver,
            "known_at": cutoff_dt.isoformat(), "lag_steps": 0,
            "direction": "unknown",
            "rationale": (
                "The source names a driver relationship and future driver "
                "transition but omits executable parameters. A model may "
                "propose a labelled prior; historical support is absent."),
        }],
        "covariate_tables": [], "transformations": [],
        "observation_interpretations": [], "effect_proposal": None,
        "forecast_candidate": None,
        "_named_driver_relationship": {
            "driver": driver,
            "transition_timestamp": selected_transition[0].isoformat(),
            "transition_value": selected_transition[2],
            "future_path_assumption": "piecewise_constant_after_stated_transition",
        },
    }


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


def _states_historical_reference_distribution(span: Any) -> bool:
    """Whether a cited span contains an explicit bounded reference sample.

    Bracketed min/max or interval notation is common in planning dossiers for
    a comparable product, site, or cohort. It is weaker than a target bound:
    it cannot constrain or automate the forecast. It can, however, explain why
    a cold-start prior legitimately leaves a degenerate target history. Keep
    the recognition narrow so incidental prose numbers (population, dates,
    identifiers) do not become scale authority.
    """
    text = str(span or "")
    return bool(re.search(
        r"\[\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*,\s*"
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*\]",
        text))


def deterministic_quantitative_background_claims(
        context_text: str, *, maximum_claims: int = 8,
) -> list[dict[str, Any]]:
    """Preserve explicit descriptive quantities a compiler omitted.

    This deliberately does not interpret dates, causal effects, or future
    triggers. It copies only complete source sentences that state a descriptive
    statistic (for example an average, median, typical rate, or historical
    peak) together with an explicit number. The claims are scenario context,
    never numeric authority or automation evidence by themselves.
    """
    if (isinstance(maximum_claims, bool) or not isinstance(maximum_claims, int)
            or not 1 <= maximum_claims <= 32):
        raise ValueError("maximum_claims must be an integer from 1 to 32")
    sentences = [item.strip() for item in re.split(
        r"(?<=[.!?])\s+|[\r\n]+", str(context_text or "")) if item.strip()]
    statistic = re.compile(
        r"\b(?:on\s+average|averag(?:e|es|ed|ing)|mean|median|typically|"
        r"usual(?:ly)?|historical(?:ly)?|busiest|quietest|peak(?:ed|s)?|"
        r"minimum|maximum)\b", re.I)
    quantity = re.compile(
        r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*"
        r"(?:%|percent\b|[A-Za-z][A-Za-z_-]*\b)", re.I)
    prospective = re.compile(
        r"\b(?:will|would|shall|tomorrow|next\s+(?:day|week|month|year)|"
        r"forecast|projected|expected\s+to)\b", re.I)
    claims = []
    for sentence in sentences:
        if (statistic.search(sentence) is None
                or quantity.search(sentence) is None
                or prospective.search(sentence) is not None):
            continue
        claims.append({
            "source_span": sentence,
            "relation": "supports_stability",
            "effective_start": None,
            "effective_end": None,
            "timing_status": "atemporal_context",
            "mechanism": "source-stated descriptive reference statistic",
            "confidence": 0.5,
        })
        if len(claims) >= maximum_claims:
            break
    return claims


def deterministic_reference_range_claims(
        context_text: str, *, maximum_claims: int = 12,
) -> list[dict[str, Any]]:
    """Preserve a source-stated comparable-entity range table verbatim.

    New entities often lack replay history while their request embeds a small
    reference table. This parser performs no analogue selection or arithmetic:
    it recognizes a reference cue, copies complete numeric-range rows, and
    retains explicit target descriptors from the preamble. The claims remain
    atemporal prior evidence and can never authorize automation.
    """
    if (isinstance(maximum_claims, bool) or not isinstance(maximum_claims, int)
            or not 2 <= maximum_claims <= 32):
        raise ValueError("maximum_claims must be an integer from 2 to 32")
    text = str(context_text or "")
    reference = re.search(
        r"\b(?:for\s+reference|comparable|comparison|analog(?:ue|ous)|"
        r"peer(?:s|\s+group)?)\b", text, re.I)
    if reference is None:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    range_row = re.compile(
        r"^(?:[*•-]\s*)?[^\n]{1,500}:\s*\[\s*"
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*,\s*"
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*\]"
        r"(?:\s+[A-Za-z%][^\n]{0,120})?$", re.I)
    rows = [line for line in lines if range_row.fullmatch(line)]
    if len(rows) < 2:
        return []
    preamble = text[:reference.start()]
    sentences = [item.strip() for item in re.split(
        r"(?<=[.!?])\s+|[\r\n]+", preamble) if item.strip()]
    descriptor = re.compile(
        r"^(?:[^.!?]{0,160}\b(?:new|newly|inaugurat\w*|launch\w*|"
        r"open\w*)\b|this\s+[^.!?]{1,100}\b(?:is|are|has|have|lies|"
        r"sits|adjoins?|borders?|contains?)\b)", re.I)
    descriptors = [sentence for sentence in sentences
                   if descriptor.search(sentence)]
    retained = (descriptors[-2:] + rows)[:maximum_claims]
    return [{
        "source_span": span,
        "relation": "supports_stability" if span in rows else "unknown",
        "effective_start": None,
        "effective_end": None,
        "timing_status": "atemporal_context",
        "mechanism": (
            "source-stated comparable-entity numeric range"
            if span in rows else "source-stated target descriptor"),
        "confidence": 1.0,
    } for span in retained]


def deterministic_associational_claims(
        context_text: str, *, maximum_claims: int = 4,
) -> list[dict[str, Any]]:
    """Retain explicit associations without granting causal authority.

    Association is useful context even when its author does not add a formal
    ``correlation is not causation`` disclaimer.  Recognizing the source text
    is safe because this lane grants only negative authority: it may prevent
    an intervention inference, but can never create a numeric candidate or
    automation authority.
    """
    if (isinstance(maximum_claims, bool) or not isinstance(maximum_claims, int)
            or not 1 <= maximum_claims <= 16):
        raise ValueError("maximum_claims must be an integer from 1 to 16")
    text = str(context_text or "")
    sentences = [item.strip() for item in re.split(
        r"(?<=[.!?])\s+|[\r\n]+", text) if item.strip()]
    relevant = re.compile(
        r"\b(?:associat\w*|correlat\w*|co[- ]?occur\w*|move\s+together|"
        r"tend(?:s|ed)?\s+to\s+(?:rise|fall|increase|decrease)\s+together|"
        r"when\b[^.!?]{1,160}\b(?:rise|fall|increase|decrease)s?\b"
        r"[^.!?]{1,160}\b(?:rise|fall|increase|decrease)s?\b|"
        r"common\s+cause|confound\w*|does\s+not\s+imply\s+caus(?:e|ation)|"
        r"correlation\s+is\s+not\s+causation)\b", re.I)
    claims = []
    for sentence in sentences:
        if relevant.search(sentence) is None:
            continue
        claims.append({
            "source_span": sentence,
            "relation": "unknown",
            "effective_start": None,
            "effective_end": None,
            "timing_status": "atemporal_context",
            "mechanism": "explicit associational evidence without causal authority",
            "confidence": 1.0,
        })
        if len(claims) >= maximum_claims:
            break
    return claims


def deterministic_explicit_confounding_claims(
        context_text: str, *, maximum_claims: int = 4,
) -> list[dict[str, Any]]:
    """Backward-compatible name for deterministic associational evidence."""
    return deterministic_associational_claims(
        context_text, maximum_claims=maximum_claims)


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


def _cited_span_is_observed_background(span: str) -> bool:
    """Return whether a relative-time clause explicitly describes the past.

    A compiler can mistake an already-observed fact (``three months ago``)
    for an undated future trigger. The exact historical date is unnecessary
    when the claim is used only as background evidence. Keep this host-side
    correction deliberately narrow: an explicit ``ago`` construction proves
    only that the fact predates the cutoff. It grants neither an effective
    event window nor numeric or automation authority.
    """
    return bool(re.search(
        r"\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"\d+)\s+(?:minute|hour|day|week|month|quarter|year)s?\s+ago\b",
        span, re.IGNORECASE))


def _association_only_claim(span: str) -> bool:
    """Identify explicitly associational language without causal authority."""
    text = _normalise(span)
    explicit_negative = bool(re.search(
        r"\b(?:common cause|confound\w*|does not imply caus(?:e|ation)|"
        r"correlation is not causation)\b", text))
    association = bool(re.search(
        r"\b(?:associat\w*|correlat\w*|co[- ]?occur\w*|move together|"
        r"tend(?:s|ed)? to (?:rise|fall|increase|decrease) together|"
        r"when\b[^.!?]{1,160}\b(?:rise|fall|increase|decrease)s?\b"
        r"[^.!?]{1,160}\b(?:rise|fall|increase|decrease)s?\b)\b", text))
    explicit_cause = bool(re.search(
        r"\b(?:causes?|caused by|leads? to|results? in|drives?|because of|"
        r"as a result of)\b", text))
    return (association or explicit_negative) and (
        not explicit_cause or explicit_negative)


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
    governed_candidate: dict[str, Any] | None = None,
    prefer_explicit_forecast_candidate: bool = False,
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
        elif timing_status == "unresolved_trigger" and \
                _cited_span_is_observed_background(span):
            timing_status = "atemporal_context"
            history_window_binding = {
                "kind": "explicit_past_background_reconciled",
                "basis": (
                    "verbatim relative-past language establishes that the "
                    "fact was observed before the cutoff; no event onset or "
                    "numeric authority was inferred"),
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
            if history_window_binding is None:
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
    explicit_candidate_preferred = bool(
        prefer_explicit_forecast_candidate
        and raw.get("forecast_candidate") not in (None, {}))
    use_calibration_candidate = bool(
        calibration_candidate is not None and not explicit_candidate_preferred)
    if use_calibration_candidate:
        # Host-validated deterministic evidence owns this lane. A malformed
        # model-authored transformation may be retained as a rejection, but it
        # cannot veto an independently compiled counterfactual.
        candidate_selection_eligible = True
        candidate_selection_reason = None
    use_derived_candidate = bool(not explicit_candidate_preferred
        and not use_calibration_candidate and
        derived_candidate is not None and (
            derived_replay_admitted or derived_replay_human_eligible
            or derived_scenario_is_deterministic
            or raw.get("forecast_candidate") in (None, {})))
    candidate_was_derived_from_observation_interpretation = use_derived_candidate
    if candidate_was_derived_from_observation_interpretation:
        replay = derived_replay
        # Mechanical validity and recommendation authority are distinct. A
        # counterfactual that loses its fold-safe replay remains visible for
        # inspection and outcome scoring, but neither an LLM nor a human-facing
        # default may promote it merely because its transformation is valid.
        candidate_selection_eligible = bool(
            derived_replay_admitted or derived_replay_human_eligible
            or derived_scenario_is_deterministic)
        if replay.get("human_recommendation_eligible") is True \
                and replay.get("selection_eligible") is not True:
            candidate_selection_reason = (
                "An outcome-inferred contamination sensitivity cleared the "
                "full replay margin on both governed metrics and the "
                "chronological block gate; it may lead best_effort for human "
                "review but can never become historical evidence or authorize "
                "automation."
                if replay.get("status") ==
                "scenario_only_outcome_inferred_mask" else
                "Conditional replay improved both governed metrics across two "
                "chronological blocks but missed strict admission; it may lead "
                "best_effort for human review and can never authorize automation.")
        elif replay.get("selection_eligible") is not True:
            candidate_selection_reason = (
                "Historical-contamination filtering is mechanically valid but "
                "did not earn the human recommendation gate; retain it as a "
                "visible prior-assisted scenario for inspection and outcome "
                "scoring only.")
    candidate_reason_start = len(reasons)
    candidate_input = (governed_candidate if governed_candidate is not None else
                       calibration_candidate if use_calibration_candidate else
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
            use_calibration_candidate
            or bool(
                isinstance(governed_candidate, dict)
                and (
                    (governed_candidate.get("provenance_class") ==
                     "model_authored_relationship_prior"
                     and ((governed_candidate.get("validation") or {}).get(
                         "elicitation") or {}).get(
                            "eligible_for_human_recommendation") is True)
                    or (governed_candidate.get("provenance_class") ==
                        "governed_reference_law_mapping"
                        and governed_candidate.get(
                            "human_selection_eligible") is True))
                and governed_candidate.get("automation_eligible") is False)
            or (
                candidate_was_derived_from_observation_interpretation
            and (derived_replay_admitted or derived_replay_human_eligible))))
    governed_candidate_accepted = bool(
        candidate is not None and governed_candidate is not None)
    if governed_candidate_accepted:
        governed_origin = str(governed_candidate.get("provenance_class") or
                              "governed_companion_mapping")
        candidate.update({
            "provenance_class": governed_origin,
            "rationale": str(governed_candidate.get("rationale") or "")[:1000],
            "validation": dict(governed_candidate.get("validation") or {}),
            "executable": dict(governed_candidate.get("executable") or {}),
            "claim_ids": list(governed_candidate.get("claim_ids") or []),
        })
        candidate_selection_eligible = bool(
            governed_candidate.get("selection_eligible"))
        candidate_selection_reason = (
            None if candidate_selection_eligible else
            "The governed contextual mapping did not beat its strongest "
            "declared target-only baseline in expanding-origin replay; retain "
            "it as a visible scenario only.")
    if candidate is not None and not (
            use_calibration_candidate
            or candidate_was_derived_from_observation_interpretation
            or governed_candidate_accepted):
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
            and not governed_candidate_accepted
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
            "human_selection_eligible": bool(
                candidate and (
                    governed_candidate.get("human_selection_eligible",
                                           candidate_selection_eligible)
                    if governed_candidate_accepted
                    else candidate_selection_eligible)),
            "selection_reason": (candidate_selection_reason
                                 if candidate and not candidate_selection_eligible
                                 else None),
            "candidate_origin": (
                governed_origin if governed_candidate_accepted else
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
        ended_disruption_entailed = (
            predicate_op in {"recurring_window", "recurring_clock_window"}
            and any(re.search(
                r"\b(?:maintenance|outage|clos(?:ed|ure)|unavailable)\b",
                span) for span in spans)
            and any(re.search(
                r"\b(?:will not|won't)\b.{0,80}\b(?:future|again)\b",
                span) for span in spans))
        if not zero_entailed and not ended_disruption_entailed:
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
                unit = str(predicate.get("unit") or "day").lower().rstrip("s")
                recurrence_start = _timestamp(predicate["start"])
                if recurrence_start is None:
                    naive_start = datetime.fromisoformat(str(predicate["start"]))
                    recurrence_start = naive_start.replace(tzinfo=start.tzinfo)
            except (KeyError, TypeError, ValueError):
                duration = period = 0
                unit = ""
                recurrence_start = None
            source = " ".join(spans)
            schedule_entailed = (
                duration > 0 and period >= duration and recurrence_start is not None
                and unit in {"day", "hour"}
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
                elapsed_seconds = (observed - recurrence_start).total_seconds()
                unit_seconds = 86400.0 if unit == "day" else 3600.0
                step = int(elapsed_seconds // unit_seconds)
                mask.append(step % period < duration)
            applied_predicate = {
                "op": "recurring_window", "start": recurrence_start.isoformat(),
                "duration_steps": duration, "period_steps": period,
                "unit": unit,
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
                # activity may be legitimately broad, so its within-regime
                # MAD is not evidence against a separate outage component.
                # Demand a very tight low component and a gap material relative
                # to the normal operating level. This admits clear censored
                # mixtures without splitting a broad unimodal positive series.
                separation_floor = max(
                    1e-9, 6.0 * low_mad, .20 * abs(high_median))
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
        history_timestamps=history_timestamps,
        rotate_mask_phases=(
            accepted[0]["predicate"].get("op") ==
            "recurring_clock_window"))
    normalization = accepted[0].get("predicate_normalization") or {}
    if normalization.get("kind") == \
            "semantic_zero_to_separated_near_zero_cluster":
        required_margin = float(replay.get("required_margin") or .10)
        point_skill = replay.get("relative_improvement")
        probabilistic_skill = replay.get(
            "probabilistic_relative_improvement")
        conservative_human_gate = bool(
            isinstance(point_skill, (int, float))
            and isinstance(probabilistic_skill, (int, float))
            and math.isfinite(float(point_skill))
            and math.isfinite(float(probabilistic_skill))
            and float(point_skill) >= required_margin
            and float(probabilistic_skill) >= required_margin
            and int(replay.get("chronological_block_wins") or 0) >= int(
                replay.get("required_block_wins") or 2))
        replay = {
            **replay,
            "status": "scenario_only_outcome_inferred_mask",
            "selection_eligible": False,
            "human_recommendation_eligible": conservative_human_gate,
            "human_gate_basis": (
                "outcome_inferred_mask_requires_full_replay_margin"),
            "authority_note": (
                "Because cluster membership was inferred from target outcomes, "
                "even the human-facing sensitivity must clear the full replay "
                "margin on both metrics and the chronological block gate. It "
                "never upgrades support or automation."),
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


def attach_host_candidate_elicitation(
    dossier: dict[str, Any], *, requested_paths: int, accepted_paths: int,
    aggregation: str, temperature: float,
    stability: dict[str, Any] | None = None,
    request_mode: str = "batch_request",
    sample_paths: list[list[float]] | None = None,
    governed_fallback: str | None = None,
    conditional_windows: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Seal narrow host-observed elicitation metadata onto a model candidate.

    The model cannot self-assert this evidence: callers must first present a
    valid sealed dossier whose candidate origin is model-authored.  The
    metadata describes sampling stability, not historical skill, and therefore
    cannot alter support or automation eligibility.
    """
    if not verify_temporal_dossier_seal(dossier):
        raise ValueError("candidate elicitation requires a valid dossier seal")
    critique = dossier.get("candidate_critique") or {}
    if (critique.get("candidate_origin") != "model_authored"
            or not isinstance(dossier.get("forecast_candidate"), dict)):
        raise ValueError(
            "candidate elicitation applies only to model-authored candidates")
    if (isinstance(requested_paths, bool) or isinstance(accepted_paths, bool)
            or not isinstance(requested_paths, int)
            or not isinstance(accepted_paths, int)
            or not 1 <= requested_paths <= 32
            or not 1 <= accepted_paths <= requested_paths):
        raise ValueError("candidate elicitation path counts are invalid")
    if aggregation not in {"linear_empirical_marginal_q10_q50_q90"}:
        raise ValueError("candidate elicitation aggregation is unsupported")
    if request_mode not in {"batch_request",
                            "concurrent_single_sample_requests"}:
        raise ValueError("candidate elicitation request mode is unsupported")
    if governed_fallback not in {
            None, "structured_companion_mapping_not_admitted"}:
        raise ValueError("candidate elicitation governed fallback is unsupported")
    clean_windows = []
    candidate_timestamps = [str(row.get("timestamp")) for row in
                            dossier["forecast_candidate"].get("quantiles") or []]
    for window in conditional_windows or []:
        if not isinstance(window, dict):
            raise ValueError("candidate conditional window is invalid")
        start = _timestamp(window.get("start"))
        end = _timestamp(window.get("end"))
        if start is None or end is None or end < start:
            raise ValueError("candidate conditional window is invalid")
        if not any(start <= parsed <= end for parsed in
                   (_timestamp(value) for value in candidate_timestamps)
                   if parsed is not None):
            raise ValueError("candidate conditional window misses forecast grid")
        clean_windows.append({"start": start.isoformat(), "end": end.isoformat()})
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) \
            or not math.isfinite(float(temperature)) or temperature < 0:
        raise ValueError("candidate elicitation temperature is invalid")
    if stability is not None:
        required = {
            "version", "interpretation", "scale_basis", "path_count",
            "horizon", "median_pointwise_q80_width_scaled",
            "p90_pointwise_q80_width_scaled", "median_pairwise_mae_scaled",
            "max_pairwise_mae_scaled", "mean_direction_agreement",
            "unanimous_direction_fraction",
        }
        if set(stability) != required:
            raise ValueError("candidate elicitation stability schema is invalid")
        if (stability.get("version") != "0.1"
                or stability.get("interpretation") !=
                "stability_not_historical_skill"
                or isinstance(stability.get("path_count"), bool)
                or stability.get("path_count") != accepted_paths
                or not isinstance(stability.get("horizon"), int)
                or isinstance(stability.get("horizon"), bool)
                or stability.get("horizon", 0) < 1):
            raise ValueError("candidate elicitation stability identity is invalid")
        numeric = [key for key in required if key.endswith("_scaled")]
        numeric.extend(["mean_direction_agreement",
                        "unanimous_direction_fraction"])
        if any(isinstance(stability.get(key), bool)
               or not isinstance(stability.get(key), (int, float))
               or not math.isfinite(float(stability[key]))
               or float(stability[key]) < 0 for key in numeric):
            raise ValueError("candidate elicitation stability values are invalid")
        if any(float(stability[key]) > 1 for key in
               ("mean_direction_agreement", "unanimous_direction_fraction")):
            raise ValueError("candidate elicitation agreement is invalid")
    clean_paths = None
    if sample_paths is not None:
        horizon = len(dossier["forecast_candidate"].get("quantiles") or [])
        if (not isinstance(sample_paths, list)
                or len(sample_paths) != accepted_paths or horizon < 1):
            raise ValueError("candidate elicitation sample paths are invalid")
        clean_paths = []
        for path in sample_paths:
            if not isinstance(path, list) or len(path) != horizon:
                raise ValueError("candidate elicitation sample path horizon is invalid")
            try:
                clean = [float(value) for value in path]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "candidate elicitation sample path contains a non-number") from error
            if not all(math.isfinite(value) for value in clean):
                raise ValueError(
                    "candidate elicitation sample path contains a non-finite value")
            clean_paths.append(clean)
    updated = json.loads(json.dumps(dossier))
    updated["forecast_candidate"]["elicitation"] = {
        "kind": "sampled_point_paths",
        "requested_paths": requested_paths,
        "accepted_paths": accepted_paths,
        "aggregation": aggregation,
        "temperature": float(temperature),
        "request_mode": request_mode,
        "host_observed": True,
        "historical_skill_evidence": False,
        "automation_eligible": False,
        **({"governed_fallback": governed_fallback}
           if governed_fallback is not None else {}),
        **({"conditional_windows": clean_windows,
            "outside_window_source": "publication_resolved_default"}
           if clean_windows else {}),
        **({"stability": json.loads(json.dumps(stability))}
           if stability is not None else {}),
    }
    if clean_paths is not None:
        updated["forecast_candidate"]["sample_paths"] = clean_paths
    body = {key: value for key, value in updated.items()
            if key != "seal_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    updated["seal_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return updated


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
        forecast_window: tuple[str, str] | None = None,
        ) -> list[dict[str, Any]]:
    """Promote only literally stated absolute states into event proposals.

    The LLM locates and dates the verbatim span; Gnomon's existing parser must
    independently recover an absolute value. Qualitative effects remain
    scenarios. Returned objects intentionally re-enter the ordinary context
    validator rather than bypassing it.
    """
    from .future_context import parse_bound_span, parse_override_span

    events = []
    claims = list(dossier.get("claims") or [])
    forecast_start = (_timestamp(forecast_window[0])
                      if forecast_window else None)
    forecast_end = (_timestamp(forecast_window[1])
                    if forecast_window else None)
    for index, claim in enumerate(claims, 1):
        if forecast_start is not None and forecast_end is not None:
            claim_start = _timestamp(claim.get("effective_start"))
            claim_end = _timestamp(claim.get("effective_end"))
            if (claim_start is None or claim_end is None
                    or claim_end < forecast_start
                    or claim_start > forecast_end):
                # Historical observations remain interpretation evidence.
                # They must not be re-promoted as future absolute events just
                # because the cited sentence contains a numeric zero or bound.
                continue
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

    # A compiler may represent a precise prospective statement directly as
    # an event and omit a duplicative claim. That representation must not
    # make literal numeric authority disappear: the event already owns a
    # validated window and a verbatim source quote. Re-run the same target
    # ownership and deterministic numeric parser over those events. This is
    # deliberately not semantic inference—words such as ``offline`` only
    # become zero when the quoted source itself states a zero state under the
    # conservative parser above.
    derived_spans = {str(item.get("source_span") or "") for item in events}
    for index, source_event in enumerate(dossier.get("events") or [], 1):
        if not isinstance(source_event, dict):
            continue
        source_type = str(source_event.get("event_type") or "")
        if source_type.startswith(("override:", "constraint:")):
            continue
        span = str(source_event.get("evidence_quote") or
                   source_event.get("source_span") or "")
        if not span or span in derived_spans:
            continue
        start = source_event.get("effective_start")
        end = source_event.get("effective_end")
        if not start or not end:
            continue
        start_time = _timestamp(start)
        end_time = _timestamp(end)
        if (start_time is None or end_time is None or end_time < start_time
                or not _claim_start_is_cited(start_time, span)
                or not _claim_start_is_cited(end_time, span)):
            # Unlike validated claims, raw compiler events do not carry an
            # effective-window binding. Both boundaries must therefore be
            # recoverable verbatim from the same quote before it can own a
            # numeric override.
            continue
        if (forecast_start is not None and forecast_end is not None
                and (end_time < forecast_start
                     or start_time > forecast_end)):
            continue
        parse_span = (" ".join(span.split())
                      if span in (target_verified_spans or set())
                      else _target_relevant_claim_span(span, target_name))
        if parse_span is None:
            continue
        value, problem = parse_override_span(parse_span)
        if problem is not None or value is None:
            continue
        events.append({
            "event_type": "override:stated_absolute_value",
            "entity_scope": list(source_event.get("entity_scope") or ["*"]),
            "effective_start": start, "effective_end": end,
            "confidence": source_event.get("confidence", 1.0),
            "status": "confirmed", "evidence_quote": span,
            "source_span": span, "effect_family": "level_shift",
            "direction": "unknown", "duration": "temporary",
            "entity_kind": str(source_event.get("entity_kind") or "unknown"),
            "deterministic_value_parsed": value,
            "deterministic_parse_span": parse_span,
            "derived_from_event_id": source_event.get("event_id") or
            f"event-{index}",
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
    supplied_claim_ids = raw.get("claim_ids")
    if supplied_claim_ids is not None:
        if (not isinstance(supplied_claim_ids, list)
                or not supplied_claim_ids
                or any(not isinstance(item, str) or not item.strip()
                       for item in supplied_claim_ids)):
            reasons.append(
                "forecast_candidate claim_ids must be a non-empty string list")
            return None
        requested_claim_ids = list(dict.fromkeys(supplied_claim_ids))
        claims_by_id = {str(claim["claim_id"]): claim for claim in claims}
        if any(claim_id not in claims_by_id
               for claim_id in requested_claim_ids):
            reasons.append("forecast_candidate cites an unknown claim")
            return None
        # Validate plausibility and numeric constraints only against the exact
        # claims the candidate says influenced its path. Other dossier claims
        # remain counterevidence; silently attaching them would corrupt both
        # provenance and the unresolved-trigger safety gate.
        claims = [claims_by_id[claim_id]
                  for claim_id in requested_claim_ids]
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
    historical_reference_support = any(
        claim.get("timing_status") == "atemporal_context"
        and _states_historical_reference_distribution(
            claim.get("source_span"))
        for claim in claims)
    bounded_regime_change = bool(bound_claim_ids) and quantitative_support
    cited_prior_extrapolation = bool(
        bounded_regime_change or historical_reference_support)
    plausibility_warnings: list[str] = []
    if (boundary_jump > MAX_BOUNDARY_JUMP_SCALES
            and not cited_prior_extrapolation
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
            "cited numeric context; treat it as prior-assisted only")
    if (path_scale_ratio > MAX_PATH_SCALE_RATIO
            and not historical_reference_support):
        reasons.append("forecast_candidate failed path-scale plausibility")
        return None
    if path_scale_ratio > MAX_PATH_SCALE_RATIO:
        plausibility_warnings.append(
            "candidate dynamics exceed the target history scale but cite a "
            "bounded historical reference distribution; this is an "
            "unvalidated cold-start extrapolation for human review only")
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

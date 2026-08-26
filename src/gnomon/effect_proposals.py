"""Typed, bounded LLM proposals which Gnomon composes over a frozen path.

The model extracts semantics; it does not author forecast points.  This is a
deliberately small boundary: cited claims, an effect family, timing, scope and
an uncertain magnitude.  Composition is deterministic and never mutates the
primary artifact.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from .effects import EFFECT_SHAPES

MAX_DELAY_STEPS = 10_000
MAX_DURATION_STEPS = 10_000
MAX_COMPOSED_EFFECT_SCALES = 20.0
UNITS = frozenset({"target_units", "fraction_of_level"})
SCOPES = frozenset({"single_series", "shared", "per_series"})


def validate_effect_proposal(
    raw: Any, *, claim_ids: set[str], repair: Any = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Validate at most two model attempts and return a typed critique.

    ``repair`` is considered only if the first proposal fails.  This makes the
    repair protocol bounded and auditable rather than an open-ended agent loop.
    """
    attempts = [raw] + ([repair] if repair not in (None, {}) else [])
    attempts = attempts[:2]
    critiques: list[dict[str, Any]] = []
    for number, candidate in enumerate(attempts, 1):
        proposal, violations = _validate_one(candidate, claim_ids=claim_ids)
        critiques.append({
            "attempt": number,
            "accepted": not violations,
            "violations": violations,
            "repair_schema": None if not violations else proposal_schema(),
        })
        if proposal is not None:
            return proposal, {
                "status": "accepted" if number == 1 else "accepted_after_repair",
                "attempts_used": number, "attempts_remaining": 2 - number,
                "attempts": critiques,
            }
    return None, {
        "status": "rejected", "attempts_used": len(attempts),
        "attempts_remaining": 2 - len(attempts), "attempts": critiques,
    }


def _validate_one(raw: Any, *, claim_ids: set[str]
                  ) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    if not isinstance(raw, dict):
        return None, [_error("PROPOSAL_NOT_OBJECT", "Return one JSON object.")]
    shape = str(raw.get("shape") or "")
    if shape not in EFFECT_SHAPES or shape == "unknown":
        errors.append(_error("UNKNOWN_EFFECT_SHAPE",
                             f"shape must be one of {sorted(EFFECT_SHAPES - {'unknown'})}"))
    unit = str(raw.get("unit") or "target_units")
    if unit not in UNITS:
        errors.append(_error("UNKNOWN_EFFECT_UNIT", f"unit must be one of {sorted(UNITS)}"))
    cited = [str(value) for value in raw.get("claim_ids") or []]
    if not cited or set(cited) - claim_ids:
        errors.append(_error("UNVERIFIED_EFFECT_CLAIMS",
                             "cite at least one verified claim_id and no unknown ids"))
    try:
        location = float(raw.get("location"))
        lower = float(raw.get("lower", location))
        upper = float(raw.get("upper", location))
        confidence = float(raw.get("confidence"))
    except (TypeError, ValueError):
        location = lower = upper = confidence = math.nan
    if not all(math.isfinite(value) for value in (location, lower, upper)) \
            or not lower <= location <= upper:
        errors.append(_error("INVALID_EFFECT_DISTRIBUTION",
                             "lower <= location <= upper and all must be finite"))
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        errors.append(_error("INVALID_EFFECT_CONFIDENCE", "confidence must be in [0, 1]"))
    try:
        delay = int(raw.get("delay_steps", 0))
        duration_raw = raw.get("duration_steps")
        duration = None if duration_raw is None else int(duration_raw)
        period_raw = raw.get("period_steps")
        period = None if period_raw is None else int(period_raw)
    except (TypeError, ValueError):
        delay, duration, period = -1, -1, -1
    if not 0 <= delay <= MAX_DELAY_STEPS:
        errors.append(_error("INVALID_EFFECT_DELAY", "delay_steps must be a non-negative integer"))
    if duration is not None and not 1 <= duration <= MAX_DURATION_STEPS:
        errors.append(_error("INVALID_EFFECT_DURATION", "duration_steps must be positive or null"))
    if period is not None and not 2 <= period <= MAX_DURATION_STEPS:
        errors.append(_error("INVALID_EFFECT_PERIOD", "period_steps must be >= 2 or null"))
    scope = raw.get("scope") or {"kind": "single_series", "series": ["*"]}
    if not isinstance(scope, dict) or scope.get("kind") not in SCOPES:
        errors.append(_error("INVALID_EFFECT_SCOPE", f"scope.kind must be one of {sorted(SCOPES)}"))
    elif not isinstance(scope.get("series"), list) or not scope.get("series"):
        errors.append(_error("INVALID_EFFECT_SCOPE", "scope.series must name at least one series or '*'"))
    if shape in {"seasonal_amplitude", "seasonal_phase", "seasonal_regime_change"} \
            and period is None:
        errors.append(_error("MISSING_EFFECT_PERIOD", "seasonal effects require period_steps"))
    if shape == "variance_change" and unit != "fraction_of_level":
        errors.append(_error("INVALID_VARIANCE_UNIT",
                             "variance_change is a fractional interval-width change"))
    if shape == "saturation_bound" and unit != "target_units":
        errors.append(_error("INVALID_BOUND_UNIT",
                             "saturation_bound must use target_units"))
    if shape == "cross_series_relationship" and isinstance(scope, dict) \
            and scope.get("kind") == "single_series":
        errors.append(_error("CROSS_SERIES_SCOPE_REQUIRED",
                             "cross_series_relationship requires shared or per_series scope"))
    if shape == "custom_scenario" and raw.get("composition") not in {None, "scenario_only"}:
        errors.append(_error("CUSTOM_EFFECT_NOT_COMPOSABLE",
                             "custom_scenario can only be scenario_only"))
    if errors:
        return None, errors
    return {
        "shape": shape, "unit": unit, "location": location,
        "lower": lower, "upper": upper, "confidence": confidence,
        "delay_steps": delay, "duration_steps": duration,
        "period_steps": period,
        "scope": {"kind": scope["kind"],
                  "series": [str(value) for value in scope["series"]]},
        "claim_ids": cited,
        **({"citation_binding": str(raw["citation_binding"])}
           if raw.get("citation_binding") else {}),
        "rationale": str(raw.get("rationale") or "")[:1000],
        "uncertainty_basis": str(raw.get("uncertainty_basis") or
                                 "model-authored prior; not calibrated")[:300],
        "composition": "scenario_only",
    }, []


def _error(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def proposal_schema() -> dict[str, Any]:
    return {
        "shape": sorted(EFFECT_SHAPES - {"unknown"}),
        "unit": sorted(UNITS), "location": "number", "lower": "number",
        "upper": "number", "confidence": "number 0..1",
        "delay_steps": "integer >= 0", "duration_steps": "integer >= 1 or null",
        "period_steps": "integer >= 2 for seasonal effects",
        "scope": {"kind": sorted(SCOPES), "series": ["series name or *"]},
        "claim_ids": ["verified claim_id"], "rationale": "string",
        "uncertainty_basis": "string", "composition": "scenario_only",
    }


def compose_effect(primary: list[dict[str, Any]], proposal: dict[str, Any]
                   ) -> list[dict[str, Any]]:
    """Compose an accepted effect over a copy of an immutable primary path."""
    if proposal["shape"] == "custom_scenario":
        # A qualitative stress scenario is still useful as a named assumption,
        # but absent a transform there is no lawful numeric adjustment.
        return [dict(row) for row in primary]
    rows: list[dict[str, Any]] = []
    horizon = len(primary)
    for index, original in enumerate(primary):
        row = dict(original)
        weight = _shape_weight(index, horizon, proposal)
        point = _center(row)
        multiplier = point if proposal["unit"] == "fraction_of_level" else 1.0
        lo_effect = proposal["lower"] * multiplier * weight
        mid_effect = proposal["location"] * multiplier * weight
        hi_effect = proposal["upper"] * multiplier * weight
        if proposal["shape"] == "variance_change":
            width_factor = max(0.0, 1.0 + mid_effect)
            center = point
            qlo, qhi = _bounds(row, center)
            row["q10"] = center - (center - qlo) * width_factor
            row["q90"] = center + (qhi - center) * width_factor
        elif proposal["shape"] == "saturation_bound":
            bound = mid_effect
            if bound >= point:
                _set_quantiles(row, min(_value(row, "q10", point), bound),
                               min(point, bound), min(_value(row, "q90", point), bound))
            else:
                _set_quantiles(row, max(_value(row, "q10", point), bound),
                               max(point, bound), max(_value(row, "q90", point), bound))
        else:
            _set_quantiles(row, _value(row, "q10", point) + lo_effect,
                           point + mid_effect,
                           _value(row, "q90", point) + hi_effect)
        row["point"] = _center(row)
        rows.append(row)
    return rows


def assess_composed_effect(primary: list[dict[str, Any]], proposal: dict[str, Any]
                           ) -> dict[str, Any]:
    """Check numeric composition without confusing typing with plausibility."""
    if not primary:
        return {"accepted": False, "violations": [{
            "code": "MISSING_PRIMARY_PATH",
            "message": "An effect requires an immutable primary path."}]}
    centers = [_center(row) for row in primary]
    violations: list[dict[str, str]] = []
    if (proposal.get("unit") == "fraction_of_level"
            and proposal.get("shape") != "variance_change"
            and any(value <= 0 for value in centers)):
        violations.append({
            "code": "NONPOSITIVE_FRACTIONAL_BASE",
            "message": "A fractional level effect is ambiguous on a non-positive path."})
    composed = compose_effect(primary, proposal)
    displacements = [abs(_center(after) - before)
                     for before, after in zip(centers, composed)]
    changes = [abs(right - left) for left, right in zip(centers, centers[1:])]
    widths = [max(0.0, _value(row, "q90", center)
                  - _value(row, "q10", center)) / 2
              for row, center in zip(primary, centers)]
    positive = [value for value in [*changes, *widths] if value > 1e-12]
    scale = (statistics.median(positive) if positive
             else max(abs(statistics.median(centers)) * .01, 1e-12))
    ratio = max(displacements, default=0.0) / scale
    if ratio > MAX_COMPOSED_EFFECT_SCALES:
        violations.append({
            "code": "IMPLAUSIBLE_COMPOSED_DISPLACEMENT",
            "message": (f"Composed displacement is {ratio:.3f} robust path "
                        f"scales; maximum is {MAX_COMPOSED_EFFECT_SCALES:g}.")})
    return {"accepted": not violations, "violations": violations,
            "maximum_displacement_scales": ratio,
            "scale_basis": "median primary change or interval half-width"}


def _shape_weight(index: int, horizon: int, proposal: dict[str, Any]) -> float:
    delay = proposal["delay_steps"]
    relative = index - delay
    duration = proposal["duration_steps"]
    if relative < 0 or (duration is not None and relative >= duration):
        return 0.0
    shape = proposal["shape"]
    active = duration or max(1, horizon - delay)
    progress = min(1.0, (relative + 1) / max(1, active))
    if shape == "temporary_pulse":
        return max(0.0, 1.0 - relative / max(1, active))
    if shape in {"trend_change", "ramp_recovery"}:
        return progress if shape == "trend_change" else 1.0 - abs(2 * progress - 1)
    if shape in {"seasonal_amplitude", "seasonal_regime_change"}:
        return math.sin(2 * math.pi * (relative + 1) / proposal["period_steps"])
    if shape == "seasonal_phase":
        return math.cos(2 * math.pi * (relative + 1) / proposal["period_steps"])
    return 1.0


def _value(row: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(row.get(key, default))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _center(row: dict[str, Any]) -> float:
    return _value(row, "q50", _value(row, "point", 0.0))


def _bounds(row: dict[str, Any], center: float) -> tuple[float, float]:
    return _value(row, "q10", center), _value(row, "q90", center)


def _set_quantiles(row: dict[str, Any], q10: float, q50: float, q90: float) -> None:
    ordered = sorted((q10, q50, q90))
    row.update({"q10": ordered[0], "q50": ordered[1], "q90": ordered[2]})

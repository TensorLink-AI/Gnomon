"""Small, reusable temporal evidence primitives.

This module measures observed change; it does not manufacture a future event.
Predictive callers may use the evidence under an explicitly named persistence
assumption, while the immutable primary forecast remains authoritative.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any


EVIDENCE_VERSION = "0.1"


@dataclass(frozen=True)
class TemporalEvidence:
    property: str
    mode: str
    source: str
    direction: str | None
    estimate: Any
    support: str
    identifiable: bool
    provenance: dict[str, Any]
    assumptions: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["version"] = EVIDENCE_VERSION
        payload["assumptions"] = list(self.assumptions)
        return payload


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _scale(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    centre = _median(values)
    mad = _median([abs(value - centre) for value in values])
    return 1.4826 * mad if mad > 1e-12 else float(statistics.stdev(values))


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    centre = (len(values) - 1) / 2
    denominator = sum((index - centre) ** 2 for index in range(len(values)))
    return (sum((index - centre) * value
                for index, value in enumerate(values)) / denominator
            if denominator else 0.0)


def _slope_se(values: list[float]) -> float:
    if len(values) < 4:
        return math.inf
    slope = _slope(values)
    centre = (len(values) - 1) / 2
    intercept = statistics.mean(values) - slope * centre
    residuals = [value - (intercept + slope * index)
                 for index, value in enumerate(values)]
    denominator = sum((index - centre) ** 2 for index in range(len(values)))
    variance = sum(value * value for value in residuals) / (len(values) - 2)
    return math.sqrt(variance / denominator) if denominator else math.inf


def _residuals(values: list[float], season: int) -> list[float]:
    """Remove a local line and phase means before comparing dispersion."""
    slope = _slope(values)
    detrended = [value - slope * index for index, value in enumerate(values)]
    if season <= 1 or len(values) < 2 * season:
        centre = _median(detrended)
        return [value - centre for value in detrended]
    phases = {offset: statistics.mean(detrended[offset::season])
              for offset in range(season)}
    return [value - phases[index % season]
            for index, value in enumerate(detrended)]


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    lm, rm = statistics.mean(left), statistics.mean(right)
    denominator = math.sqrt(sum((value - lm) ** 2 for value in left)
                            * sum((value - rm) ** 2 for value in right))
    return (sum((a - lm) * (b - rm) for a, b in zip(left, right)) /
            denominator if denominator > 1e-12 else None)


def _seasonal_transition(reference: list[float], recent: list[float],
                         season: int) -> dict[str, Any]:
    if season <= 1 or min(len(reference), len(recent)) < 2 * season:
        return {"estimate": None, "direction": "uncertain",
                "reason": "insufficient_observed_cycles"}
    def template(values: list[float]) -> list[float]:
        slope = _slope(values)
        detrended = [value - slope * index
                     for index, value in enumerate(values)]
        return [statistics.mean(detrended[offset::season])
                for offset in range(season)]
    left, right = template(reference), template(recent)
    left_scale, right_scale = _scale(left), _scale(right)
    strength_ratio = (right_scale / left_scale if left_scale > 1e-12 else None)
    alignments = [(shift, _correlation(
        left, [right[(index + shift) % season] for index in range(season)]))
                  for shift in range(season)]
    valid = [(shift, corr) for shift, corr in alignments if corr is not None]
    if not valid or strength_ratio is None:
        return {"estimate": None, "direction": "uncertain",
                "reason": "seasonal_template_not_identifiable"}
    shift, correlation = max(valid, key=lambda item: item[1])
    circular = min(shift, season - shift)
    tolerance = max(1, round(.1 * season))
    if correlation >= .55 and circular > tolerance:
        direction = "phase_shifted"
    elif strength_ratio > 1.35:
        direction = "strengthened"
    elif strength_ratio < .65:
        direction = "weakened"
    elif correlation >= .55:
        direction = "stable"
    else:
        direction = "unstable"
    return {"estimate": {"phase_shift_steps": shift,
                         "template_correlation": correlation,
                         "strength_ratio": strength_ratio},
            "direction": direction, "reason": None}


def compare_windows(values: list[float], *, season: int = 1,
                    window: int | None = None) -> dict[str, Any]:
    """Compare two adjacent observed windows in common, auditable units."""
    numeric = [float(value) for value in values if math.isfinite(float(value))]
    season = max(1, int(season))
    width = min(window or max(8, 2 * season), len(numeric) // 2)
    if width < max(6, season) or len(numeric) < 2 * width:
        return {"identifiable": False, "reason": "insufficient_observed_windows",
                "window_steps": width, "properties": {}}
    reference, recent = numeric[-2 * width:-width], numeric[-width:]
    innovations = [numeric[index] - numeric[index - 1]
                   for index in range(1, len(numeric))]
    unit = max(_scale(innovations) / math.sqrt(2), 1e-12)
    reference_residuals = _residuals(reference, season)
    recent_residuals = _residuals(recent, season)
    reference_scale = _scale(reference_residuals)
    recent_scale = _scale(recent_residuals)
    ratio = (recent_scale / reference_scale
             if reference_scale > 1e-12 else None)
    volatility = ("increased" if ratio is not None and ratio > 1.25 else
                  "decreased" if ratio is not None and ratio < .8 else
                  "stable" if ratio is not None else "uncertain")
    level_delta = (_median(recent) - _median(reference)) / unit
    trend_delta = (_slope(recent) - _slope(reference)) / unit
    z90 = 1.6448536269514722
    level_se = (1.2533 * math.sqrt(
        (_scale(reference) ** 2 / len(reference))
        + (_scale(recent) ** 2 / len(recent))) / unit)
    level_interval = [level_delta - z90 * level_se,
                      level_delta + z90 * level_se]
    trend_se = math.sqrt(_slope_se(reference) ** 2
                         + _slope_se(recent) ** 2) / unit
    trend_interval = [trend_delta - z90 * trend_se,
                      trend_delta + z90 * trend_se]
    volatility_interval = None
    if ratio is not None and ratio > 0 and min(len(reference_residuals),
                                               len(recent_residuals)) > 2:
        # MAD ratios are more variable than Gaussian standard-deviation
        # ratios. The 1.64 inflation approximates the Gaussian asymptotic
        # efficiency correction (sqrt(1 / 0.37))
        # for using a robust scale here; without it the equivalence interval
        # is anti-conservative near the admission boundary.
        log_se = 1.64 * math.sqrt(
            1 / (2 * (len(reference_residuals) - 1))
            + 1 / (2 * (len(recent_residuals) - 1)))
        volatility_interval = [
            math.exp(math.log(ratio) - z90 * log_se),
            math.exp(math.log(ratio) + z90 * log_se),
        ]
    seasonal = _seasonal_transition(reference, recent, season)
    reference_extremes = sum(abs(value) > 3.5 * max(reference_scale, 1e-12)
                             for value in reference_residuals) / len(reference_residuals)
    recent_extremes = sum(abs(value) > 3.5 * max(reference_scale, 1e-12)
                          for value in recent_residuals) / len(recent_residuals)
    extreme_delta = recent_extremes - reference_extremes
    regime_strength = max(abs(level_delta), abs(trend_delta))
    return {
        "identifiable": ratio is not None,
        "reason": None if ratio is not None else "zero_reference_dispersion",
        "window_steps": width,
        "properties": {
            "level": {"estimate": level_delta,
                      "interval": level_interval, "interval_level": .90,
                      "direction": "higher" if level_delta > .25 else
                                   "lower" if level_delta < -.25 else "similar"},
            "trend": {"estimate": trend_delta,
                      "interval": trend_interval, "interval_level": .90,
                      "direction": "upward" if trend_delta > .02 else
                                   "downward" if trend_delta < -.02 else "constant"},
            "volatility": {"estimate": ratio, "direction": volatility,
                           "interval": volatility_interval,
                           "interval_level": .90,
                           "interval_method":
                               "robust_log_scale_ratio_asymptotic",
                           "reference_scale": reference_scale,
                           "recent_scale": recent_scale},
            "seasonality": seasonal,
            "regime": {"estimate": regime_strength,
                       "direction": "shift" if regime_strength >= 1 else
                                    "no_shift"},
            "extreme": {"estimate": extreme_delta,
                        "direction": "increased" if extreme_delta > .02 else
                                     "decreased" if extreme_delta < -.02 else
                                     "stable"},
        },
        "provenance": {"kind": "observed_adjacent_windows",
                       "uses_future_observations": False},
    }


def window_evidence(values: list[float], *, property: str, season: int = 1,
                    window: int | None = None) -> TemporalEvidence:
    """Return one typed observed-transition receipt."""
    comparison = compare_windows(values, season=season, window=window)
    item = (comparison.get("properties") or {}).get(property) or {}
    direction = item.get("direction")
    identifiable = bool(item) and direction not in {None, "uncertain"} \
        and item.get("estimate") is not None
    support = (_transition_support(property, direction, item, season)
               if identifiable else "abstained")
    return TemporalEvidence(
        property=property, mode="observed",
        source="adjacent_observed_windows", direction=direction,
        estimate=item.get("estimate"),
        support=support,
        identifiable=identifiable,
        provenance=comparison.get("provenance") or {},
        diagnostics={"window_steps": comparison.get("window_steps"),
                     "reason": item.get("reason") or comparison.get("reason")},
    )


def _transition_support(property: str, direction: str | None,
                        item: dict[str, Any], season: int) -> str:
    """Grade distance from a threshold without using benchmark labels."""
    estimate = item.get("estimate")
    interval = item.get("interval")
    if property == "level" and isinstance(estimate, (int, float)):
        value = abs(float(estimate))
        clear = (value >= .75 if direction != "similar" else
                 bool(interval and interval[0] >= -.25
                      and interval[1] <= .25))
    elif property == "trend" and isinstance(estimate, (int, float)):
        value = abs(float(estimate))
        clear = (value >= .06 if direction != "constant" else
                 bool(interval and interval[0] >= -.02
                      and interval[1] <= .02))
    elif property == "volatility" and isinstance(estimate, (int, float)):
        value = float(estimate)
        clear = (value >= 1.5 if direction == "increased" else
                 value <= .67 if direction == "decreased" else
                 bool(interval and interval[0] >= .8
                      and interval[1] <= 1.25))
    elif property == "regime" and isinstance(estimate, (int, float)):
        value = float(estimate)
        clear = value >= 1.5 if direction == "shift" else False
    elif property == "extreme" and isinstance(estimate, (int, float)):
        value = abs(float(estimate))
        clear = value >= .04 if direction != "stable" else False
    elif property == "seasonality" and isinstance(estimate, dict):
        ratio = float(estimate.get("strength_ratio"))
        correlation = float(estimate.get("template_correlation"))
        shift = int(estimate.get("phase_shift_steps"))
        if direction == "phase_shifted":
            circular = min(shift, season - shift)
            clear = (circular > max(1, round(.1 * season)) + 1
                     and correlation >= .7)
        elif direction == "strengthened":
            clear = ratio >= 1.6
        elif direction == "weakened":
            clear = ratio <= .55
        elif direction == "stable":
            # A small point estimate is not evidence of equivalence. Until
            # the executable carries an equivalence interval, stability is a
            # useful weak observation rather than a supported null claim.
            clear = False
        else:
            clear = False
    else:
        clear = False
    return "supported" if clear else "weak"


def aggregate_evidence(rows: dict[str, dict[str, Any]], *,
                       property: str) -> dict[str, Any]:
    """Aggregate comparable series evidence without treating rows as folds."""
    observations: list[tuple[str, str, float | None]] = []
    for name, row in rows.items():
        item = (row.get("properties") or {}).get(property) or {}
        direction = item.get("direction")
        if row.get("identifiable") and direction not in {None, "uncertain"}:
            estimate = item.get("estimate")
            observations.append((name, str(direction),
                                 float(estimate) if estimate is not None else None))
    counts: dict[str, int] = {}
    for _, direction, _ in observations:
        counts[direction] = counts.get(direction, 0) + 1
    if not observations:
        return {"direction": "uncertain", "support": "abstained",
                "effective_series": 0, "agreement": 0.0,
                "direction_counts": {}, "estimate": None,
                "identifiable": False,
                "reason": "no_comparable_series_evidence"}
    direction, votes = max(counts.items(), key=lambda item: (item[1], item[0]))
    agreement = votes / len(observations)
    # Series are evidence sources, not independent time folds. Strong support
    # therefore requires breadth and high agreement; otherwise this is a weak
    # persistence clue for an LLM, never an automated claim.
    support = ("supported" if len(observations) >= 5 and agreement >= .8 else
               "weak" if agreement > .5 else "abstained")
    estimates = [estimate for _, label, estimate in observations
                 if label == direction and estimate is not None]
    return {
        "direction": direction if agreement > .5 else "uncertain",
        "support": support,
        "effective_series": len(observations),
        "agreement": agreement,
        "direction_counts": counts,
        "estimate": _median(estimates) if estimates else None,
        "identifiable": agreement > .5,
        "reason": None if agreement > .5 else "no_cross_series_consensus",
        "constituents": [name for name, _, _ in observations],
        "provenance": {"kind": "cross_series_observed_evidence",
                       "independence_claimed": False},
    }

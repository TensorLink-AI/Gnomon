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
MULTI_RESOLUTION_VERSION = "0.1"
ANALOGUE_VERSION = "0.1"


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
    reference_cycles = max(2, len(reference) // season)
    recent_cycles = max(2, len(recent) // season)
    # Approximate uncertainty for the ratio of phase-template dispersions.
    # It is intentionally conservative: cycles, not individual observations,
    # are the effective sample size for a seasonal-strength claim.
    log_se = 1.64 * math.sqrt(
        1 / (2 * (reference_cycles - 1))
        + 1 / (2 * (recent_cycles - 1)))
    ratio_interval = [
        math.exp(math.log(strength_ratio) - 1.6448536269514722 * log_se),
        math.exp(math.log(strength_ratio) + 1.6448536269514722 * log_se),
    ]
    return {"estimate": {"phase_shift_steps": shift,
                         "template_correlation": correlation,
                         "strength_ratio": strength_ratio,
                         "strength_ratio_interval": ratio_interval,
                         "effective_reference_cycles": reference_cycles,
                         "effective_recent_cycles": recent_cycles},
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
    reference_extreme_count = sum(
        abs(value) > 3.5 * max(reference_scale, 1e-12)
        for value in reference_residuals)
    recent_extreme_count = sum(
        abs(value) > 3.5 * max(reference_scale, 1e-12)
        for value in recent_residuals)
    reference_extremes = reference_extreme_count / len(reference_residuals)
    recent_extremes = recent_extreme_count / len(recent_residuals)
    extreme_delta = recent_extremes - reference_extremes
    regime_strength = max(abs(level_delta), abs(trend_delta))
    # Jeffreys-smoothed binomial variance avoids the zero-width Wald interval
    # when neither short window happened to contain a tail event.  Absence in
    # a finite sample is not proof that the tail rate is exactly zero.
    reference_tail_p = ((reference_extreme_count + .5)
                        / (len(reference_residuals) + 1))
    recent_tail_p = ((recent_extreme_count + .5)
                     / (len(recent_residuals) + 1))
    extreme_se = math.sqrt(
        reference_tail_p * (1 - reference_tail_p)
        / len(reference_residuals)
        + recent_tail_p * (1 - recent_tail_p) / len(recent_residuals))
    extreme_interval = [extreme_delta - z90 * extreme_se,
                        extreme_delta + z90 * extreme_se]
    regime_upper = max(abs(bound) for bound in
                       (*level_interval, *trend_interval))
    regime_lower = max(
        0.0,
        min(abs(level_interval[0]), abs(level_interval[1]))
        if level_interval[0] * level_interval[1] > 0 else 0.0,
        min(abs(trend_interval[0]), abs(trend_interval[1]))
        if trend_interval[0] * trend_interval[1] > 0 else 0.0,
    )
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
                       "interval": [regime_lower, regime_upper],
                       "interval_level": .90,
                       "direction": "shift" if regime_strength >= 1 else
                                    "no_shift"},
            "extreme": {"estimate": extreme_delta,
                        "interval": extreme_interval,
                        "interval_level": .90,
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


def multi_resolution_evidence(
    values: list[float], *, property: str, season: int = 1,
) -> dict[str, Any]:
    """Compare a transition at several observable time scales.

    The resolutions are derived only from history length and the declared
    season.  This is deliberately label-free: callers get agreement,
    disagreement, and the runner-up rather than a forced single-scale story.
    Duplicate or infeasible windows are omitted.
    """
    numeric = [float(value) for value in values if math.isfinite(float(value))]
    season = max(1, int(season))
    candidates = (
        ("immediate", max(8, season)),
        ("recent", max(16, 2 * season)),
        ("seasonal", max(24, 4 * season)),
        ("long_run", max(32, min(len(numeric) // 2, 8 * season))),
    )
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for label, width in candidates:
        width = min(width, len(numeric) // 2)
        if width in seen or width < max(6, season):
            continue
        seen.add(width)
        evidence = window_evidence(
            numeric, property=property, season=season, window=width)
        if evidence.identifiable:
            rows.append({
                "resolution": label,
                "window_steps": width,
                "direction": evidence.direction,
                "support": evidence.support,
                "estimate": evidence.estimate,
            })
    weights = {"supported": 2.0, "weak": 1.0, "abstained": 0.0}
    scores: dict[str, float] = {}
    for row in rows:
        direction = str(row["direction"])
        scores[direction] = scores.get(direction, 0.0) + weights[row["support"]]
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    total = sum(scores.values())
    best = ranked[0] if ranked else ("uncertain", 0.0)
    runner_up = ranked[1] if len(ranked) > 1 else None
    agreement = best[1] / total if total else 0.0
    return {
        "version": MULTI_RESOLUTION_VERSION,
        "property": property,
        "direction": best[0] if agreement > .5 else "uncertain",
        "support": ("supported" if len(rows) >= 2 and agreement >= .75
                    and best[1] >= 3 else
                    "weak" if agreement > .5 else "abstained"),
        "agreement": agreement,
        "runner_up": ({"direction": runner_up[0], "score": runner_up[1]}
                      if runner_up else None),
        "resolution_divergence": len(scores) > 1,
        "resolutions": rows,
        "provenance": {
            "kind": "observed_multi_resolution_windows",
            "uses_future_observations": False,
            "window_selection": "history_and_declared_season_only",
        },
    }


def competing_hypotheses(values: list[float], *, season: int = 1,
                         limit: int = 3) -> dict[str, Any]:
    """Rank observed explanations without treating one property as truth.

    Scores describe evidence strength and scale agreement, not causal
    probability.  This receipt is suitable for an LLM to explain, but cannot
    replace a fitted predictive executable or mutate its forecast.
    """
    candidates = []
    for prop in ("level", "trend", "seasonality", "volatility", "regime",
                 "extreme"):
        receipt = multi_resolution_evidence(
            values, property=prop, season=season)
        if receipt["direction"] in {"uncertain", "similar", "constant",
                                    "stable", "no_shift"}:
            continue
        support_weight = {"supported": 2.0, "weak": 1.0,
                          "abstained": 0.0}[receipt["support"]]
        score = support_weight * float(receipt["agreement"])
        candidates.append({
            "hypothesis": prop,
            "direction": receipt["direction"],
            "support": receipt["support"],
            "score": score,
            "resolution_divergence": receipt["resolution_divergence"],
            "evidence": receipt["resolutions"],
        })
    candidates.sort(key=lambda row: (-row["score"], row["hypothesis"]))
    winner = candidates[0] if candidates else None
    runner_up = candidates[1] if len(candidates) > 1 else None
    return {
        "version": MULTI_RESOLUTION_VERSION,
        "winner": ({key: winner[key] for key in
                    ("hypothesis", "direction", "support", "score")}
                   if winner else None),
        "runner_up": ({key: runner_up[key] for key in
                       ("hypothesis", "direction", "support", "score")}
                      if runner_up else None),
        "margin": ((winner["score"] - runner_up["score"])
                   if winner and runner_up else
                   winner["score"] if winner else 0.0),
        "candidates": candidates[:max(1, int(limit))],
        "interpretation": "observed_explanations_not_causal_probabilities",
        "primary_forecast_unchanged": True,
    }


def historical_analogues(
    values: list[float], *, property: str, season: int = 1,
    window: int | None = None, limit: int = 3,
) -> dict[str, Any]:
    """Retrieve similar *past* states and disclose what followed.

    Similarity uses normalized level, slope, and residual scale.  Candidate
    outcomes are measured only from observations after each historical state;
    the current tail is never scored against unavailable future values.  The
    receipt is evidence for explanation, not a forecast replacement.
    """
    numeric = [float(value) for value in values if math.isfinite(float(value))]
    season = max(1, int(season))
    width = int(window or max(8, 2 * season))
    if len(numeric) < 4 * width:
        return {"version": ANALOGUE_VERSION, "available": False,
                "reason": "insufficient_nonoverlapping_history",
                "matches": [], "primary_forecast_unchanged": True}

    global_scale = max(_scale(numeric), 1e-12)

    def state(segment: list[float]) -> tuple[float, float, float]:
        return (_median(segment) / global_scale,
                _slope(segment) * width / global_scale,
                _scale(_residuals(segment, season)) / global_scale)

    current = state(numeric[-width:])
    candidates: list[dict[str, Any]] = []
    # A stride prevents densely-overlapping neighbours from masquerading as
    # independent analogues.  Leave a full outcome window after every state.
    for end in range(width, len(numeric) - width + 1, width):
        before, after = numeric[end - width:end], numeric[end:end + width]
        candidate = state(before)
        distance = math.sqrt(sum((left - right) ** 2
                                 for left, right in zip(current, candidate)))
        outcome = window_evidence(
            before + after, property=property, season=season, window=width)
        candidates.append({
            "state_end_offset": end - len(numeric),
            "distance": round(distance, 6),
            "outcome_direction": outcome.direction,
            "outcome_support": outcome.support,
        })
    candidates.sort(key=lambda row: (row["distance"], row["state_end_offset"]))
    matches = candidates[:max(1, min(int(limit), 5))]
    directions = [str(row["outcome_direction"]) for row in matches
                  if row["outcome_direction"] not in {None, "uncertain"}]
    counts = {label: directions.count(label) for label in sorted(set(directions))}
    consensus = (max(counts, key=counts.get) if counts
                 and max(counts.values()) > len(directions) / 2 else "uncertain")
    return {
        "version": ANALOGUE_VERSION, "available": bool(matches),
        "property": property, "window_steps": width,
        "consensus_direction": consensus,
        "agreement": (max(counts.values()) / len(directions)
                      if counts and directions else 0.0),
        "matches": matches,
        "provenance": {"kind": "historical_pre_state_nearest_neighbours",
                       "uses_future_observations": False,
                       "outcomes_are_historical_only": True},
        "primary_forecast_unchanged": True,
    }


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
        clear = (value >= 1.5 and bool(interval and interval[0] >= 1.0)
                 if direction == "shift" else
                 bool(interval and interval[1] < 1.0))
    elif property == "extreme" and isinstance(estimate, (int, float)):
        value = abs(float(estimate))
        clear = (value >= .04 and bool(interval and (
            interval[0] > 0 or interval[1] < 0))
                 if direction != "stable" else
                 # A two-percentage-point point threshold is a detection
                 # rule, not an equivalence margin.  Require the entire
                 # interval inside a tighter one-point band before claiming
                 # the tail rate is unchanged.
                 bool(interval and interval[0] >= -.01
                      and interval[1] <= .01))
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
            ratio_interval = estimate.get("strength_ratio_interval")
            clear = bool(
                ratio_interval and ratio_interval[0] >= .65
                and ratio_interval[1] <= 1.35 and correlation >= .7
                and min(shift, season - shift) <= max(1, round(.1 * season)))
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

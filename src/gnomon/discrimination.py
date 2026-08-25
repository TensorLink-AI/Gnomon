"""Held-out hypothesis discrimination: measure what distinguishes the
competing interpretations, instead of only naming it.

The cross-model evaluation (docs/cross-model-evaluation-2026-08.md) found
the evidence packet improved model behaviour but not discrimination — it
described the history without helping a model choose between the
interpretations still compatible with it.  This module runs the
distinguishing computation whenever the history permits: each competing
interpretation gets a minimal deterministic surrogate fitted strictly
before a held-out tail, every surrogate is scored on the same held-out
observations, and the packet reports the measured relative fit.

Honesty contract:

- Every estimate uses only observed values, and every surrogate is fitted
  on data strictly before the held-out window it is scored on.
- Relative weights are Akaike-style fit evidence over the surrogate set —
  never probabilities, and never a replacement for the fitted canonical
  executable or the primary forecast.
- A history too short for the three-window split abstains with a typed
  reason rather than manufacturing a discrimination.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

DISCRIMINATION_VERSION = "0.1"

#: Properties with a registered surrogate set, in their public answer
#: vocabularies.
DISCRIMINABLE_PROPERTIES = ("trend", "level", "volatility", "disturbance")

#: Separation grades over the winning relative weight. "clear" means the
#: held-out evidence concentrates on one interpretation; "none" means the
#: measurement ran but does not separate them — itself a useful fact.
_CLEAR_WEIGHT = .8
_MODERATE_WEIGHT = .6


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _scale(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    centre = _median(values)
    mad = _median([abs(value - centre) for value in values])
    return 1.4826 * mad if mad > 1e-12 else float(statistics.stdev(values))


def _slope_line(values: list[float]) -> tuple[float, float]:
    """OLS slope and intercept with x = 0..n-1."""
    n = len(values)
    centre = (n - 1) / 2
    denominator = sum((index - centre) ** 2 for index in range(n))
    slope = (sum((index - centre) * value
                 for index, value in enumerate(values)) / denominator
             if denominator else 0.0)
    intercept = statistics.mean(values) - slope * centre
    return slope, intercept


def _detrended(values: list[float], season: int) -> list[float]:
    slope, intercept = _slope_line(values)
    detrended = [value - (intercept + slope * index)
                 for index, value in enumerate(values)]
    if season > 1 and len(values) >= 2 * season:
        phases = {offset: statistics.mean(detrended[offset::season])
                  for offset in range(season)}
        return [value - phases[index % season]
                for index, value in enumerate(detrended)]
    centre = _median(detrended)
    return [value - centre for value in detrended]


def _aic_weights(entries: list[tuple[str, float | None, int]],
                 points: int) -> dict[str, float]:
    """Akaike-style relative weights from held-out mean squared errors.

    ``entries`` are ``(name, mse, parameter_count)``. A surrogate with no
    score gets weight zero. Weights compare only the surrogates listed here.
    """
    scored = [(name, mse, params) for name, mse, params in entries
              if mse is not None and math.isfinite(mse)]
    if not scored:
        return {name: 0.0 for name, _, _ in entries}
    floor = 1e-12
    aic = {name: points * math.log(max(mse, floor)) + 2 * params
           for name, mse, params in scored}
    best = min(aic.values())
    raw = {name: math.exp(-(value - best) / 2) for name, value in aic.items()}
    total = sum(raw.values())
    weights = {name: value / total for name, value in raw.items()}
    for name, _, _ in entries:
        weights.setdefault(name, 0.0)
    return weights


def _mse(actual: list[float], predicted: list[float]) -> float | None:
    if len(actual) != len(predicted) or not actual:
        return None
    if any(not math.isfinite(value) for value in [*actual, *predicted]):
        return None
    return sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)


def _abstain(property: str, reason: str) -> dict[str, Any]:
    return {"version": DISCRIMINATION_VERSION, "property": property,
            "identifiable": False, "reason": reason, "hypotheses": []}


def _payload(property: str, holdout: int, windows: dict[str, int],
             hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
    # Stable sort: callers list the null hypothesis first, so a held-out
    # tie resolves to "nothing changed" rather than to an alphabetical
    # accident — evidence that cannot separate the surrogates must not
    # manufacture a transition.
    ranked = sorted(hypotheses, key=lambda row: -row["relative_weight"])
    top = ranked[0]["relative_weight"] if ranked else 0.0
    separation = ("clear" if top >= _CLEAR_WEIGHT else
                  "moderate" if top >= _MODERATE_WEIGHT else "none")
    return {
        "version": DISCRIMINATION_VERSION,
        "property": property,
        "identifiable": True,
        "holdout_steps": holdout,
        "hypotheses": ranked,
        "best": ranked[0]["value"] if ranked else None,
        "separation": separation,
        "weights_are_fit_evidence_not_probabilities": True,
        "provenance": {
            "kind": "held_out_hypothesis_fit",
            "uses_future_observations": False,
            "estimation_windows": windows,
        },
    }


def _hypothesis(value: str, surrogate: str, mse: float | None,
                weight: float, note: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "value": value, "surrogate": surrogate,
        "held_out_mse": None if mse is None else round(float(mse), 8),
        "relative_weight": round(float(weight), 6),
    }
    if note:
        row["note"] = note
    return row


def _discriminate_trend(values: list[float], holdout: int) -> dict[str, Any]:
    fit, held = values[:-holdout], values[-holdout:]
    slope, intercept = _slope_line(fit)
    trend_path = [intercept + slope * (len(fit) + step)
                  for step in range(holdout)]
    flat_level = _median(fit[-max(holdout, 4):])
    weights = _aic_weights(
        [("trend", _mse(held, trend_path), 2),
         ("flat", _mse(held, [flat_level] * holdout), 1)], holdout)
    direction = "upward" if slope > 0 else "downward"
    opposite = "downward" if slope > 0 else "upward"
    return _payload("trend", holdout,
                    {"fit": len(fit), "holdout": holdout}, [
        _hypothesis("constant", "reference_tail_median",
                    _mse(held, [flat_level] * holdout), weights["flat"]),
        _hypothesis(direction, "reference_ols_line_extended",
                    _mse(held, trend_path), weights["trend"]),
        _hypothesis(opposite, "no_fitted_surrogate", None, 0.0,
                    "the fitted reference slope points the other way"),
    ])


def _discriminate_level(values: list[float], holdout: int) -> dict[str, Any]:
    reference = values[:-2 * holdout]
    transition = values[-2 * holdout:-holdout]
    held = values[-holdout:]
    base, shifted = _median(reference), _median(transition)
    # The shift surrogate estimates one more quantity than the null (the
    # transition-window level); charging that parameter keeps a chance-level
    # wobble from grading as a clear transition.
    weights = _aic_weights(
        [("shift", _mse(held, [shifted] * holdout), 2),
         ("no_shift", _mse(held, [base] * holdout), 1)], holdout)
    direction = "higher" if shifted > base else "lower"
    opposite = "lower" if shifted > base else "higher"
    return _payload("level", holdout, {
        "reference": len(reference), "transition": holdout,
        "holdout": holdout,
    }, [
        _hypothesis("similar", "reference_median_persists",
                    _mse(held, [base] * holdout), weights["no_shift"]),
        _hypothesis(direction, "transition_window_median_persists",
                    _mse(held, [shifted] * holdout), weights["shift"]),
        _hypothesis(opposite, "no_fitted_surrogate", None, 0.0,
                    "the transition window moved the other way"),
    ])


def _discriminate_volatility(values: list[float], holdout: int,
                             season: int) -> dict[str, Any]:
    reference = values[:-2 * holdout]
    transition = values[-2 * holdout:-holdout]
    held_residuals = _detrended(values[-holdout:], season)
    sigma_reference = _scale(_detrended(reference, season))
    sigma_transition = _scale(_detrended(transition, season))
    if sigma_reference <= 1e-12 or sigma_transition <= 1e-12:
        return _abstain("volatility", "zero_reference_dispersion")

    def nll(sigma: float) -> float:
        return sum(.5 * math.log(2 * math.pi * sigma * sigma)
                   + (value * value) / (2 * sigma * sigma)
                   for value in held_residuals)

    # Akaike form over the two Gaussian surrogates. The changed-scale
    # hypothesis estimates one more parameter (the transition window's
    # scale) and is charged for it, so noise in the scale ratio cannot
    # grade as a clear transition by itself.
    aic = {"changed": 2 * nll(sigma_transition) + 2 * 2,
           "stable": 2 * nll(sigma_reference) + 2 * 1}
    floor_aic = min(aic.values())
    raw = {name: math.exp(-(value - floor_aic) / 2)
           for name, value in aic.items()}
    total = raw["changed"] + raw["stable"]
    weights = {name: value / total for name, value in raw.items()}
    direction = "increased" if sigma_transition > sigma_reference else "decreased"
    opposite = "decreased" if sigma_transition > sigma_reference else "increased"
    score = {"changed": round(nll(sigma_transition), 6),
             "stable": round(nll(sigma_reference), 6)}
    payload = _payload("volatility", holdout, {
        "reference": len(reference), "transition": holdout,
        "holdout": holdout,
    }, [
        {"value": "stable",
         "surrogate": "gaussian_scale_from_reference_window",
         "held_out_negative_log_likelihood": score["stable"],
         "relative_weight": round(weights["stable"], 6)},
        {"value": direction,
         "surrogate": "gaussian_scale_from_transition_window",
         "held_out_negative_log_likelihood": score["changed"],
         "relative_weight": round(weights["changed"], 6)},
        {"value": opposite, "surrogate": "no_fitted_surrogate",
         "held_out_negative_log_likelihood": None, "relative_weight": 0.0,
         "note": "the transition window's scale moved the other way"},
    ])
    payload["scale_ratio"] = round(sigma_transition / sigma_reference, 6)
    return payload


def _discriminate_disturbance(values: list[float],
                              holdout: int) -> dict[str, Any]:
    innovations = [right - left for left, right in zip(values, values[1:])]
    reference_scale = _scale(innovations[:max(len(innovations) // 2, 4)])
    if reference_scale <= 1e-12:
        # A near-constant reference has no innovation dispersion to scale
        # against; any material jump is then a disturbance by construction.
        # The floor keeps the 3.5-scale trigger meaningful instead of
        # abstaining on the cleanest possible disturbance.
        reference_scale = max(1e-9, 1e-6 * abs(_median(values)))
    window_start = max(1, len(values) - 2 * holdout)
    # Earliest-maximum tie-break: a spike's reversion step has the same
    # innovation magnitude as its onset, and anchoring on the reversion
    # would score two identical surrogates against each other.
    candidates = [(abs(innovations[index - 1]), -index)
                  for index in range(window_start, len(values) - 3)]
    if not candidates:
        return _abstain("disturbance", "no_scorable_post_disturbance_window")
    magnitude, negated_index = max(candidates)
    at = -negated_index
    if magnitude <= 3.5 * reference_scale:
        return _payload("disturbance", holdout, {
            "scan": len(values) - window_start,
        }, [
            _hypothesis("stable", "no_innovation_exceeds_3.5_scales",
                        None, 1.0),
            _hypothesis("level_shift", "not_triggered", None, 0.0),
            _hypothesis("sudden_spike", "not_triggered", None, 0.0),
        ])
    pre_level = _median(values[max(0, at - holdout):at])
    shifted_level = float(values[at])
    after = values[at + 1:]
    weights = _aic_weights(
        [("shift", _mse(after, [shifted_level] * len(after)), 1),
         ("spike", _mse(after, [pre_level] * len(after)), 1)], len(after))
    return _payload("disturbance", holdout, {
        "post_disturbance": len(after),
    }, [
        _hypothesis("level_shift", "disturbed_value_persists",
                    _mse(after, [shifted_level] * len(after)),
                    weights["shift"]),
        _hypothesis("sudden_spike", "pre_disturbance_median_reasserts",
                    _mse(after, [pre_level] * len(after)), weights["spike"]),
        _hypothesis("stable", "excluded_by_triggered_disturbance", None, 0.0,
                    f"an innovation of {magnitude / reference_scale:.1f} "
                    f"reference scales was observed"),
    ])


def discriminate(values: list[float], *, property: str,
                 season: int = 1) -> dict[str, Any] | None:
    """Measured held-out discrimination for one property, or ``None`` for a
    property with no registered surrogate set."""
    if property not in DISCRIMINABLE_PROPERTIES:
        return None
    numeric = [float(value) for value in values
               if isinstance(value, (int, float)) and math.isfinite(float(value))]
    holdout = max(4, len(numeric) // 5)
    segments_needed = (2 * holdout + 6 if property in {"level", "volatility"}
                       else holdout + 6)
    if len(numeric) < max(12, segments_needed):
        return _abstain(property, "insufficient_history_for_held_out_split")
    if property == "trend":
        return _discriminate_trend(numeric, holdout)
    if property == "level":
        return _discriminate_level(numeric, holdout)
    if property == "volatility":
        return _discriminate_volatility(numeric, holdout, max(1, int(season)))
    return _discriminate_disturbance(numeric, holdout)

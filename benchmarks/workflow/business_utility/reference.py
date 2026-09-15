"""Explicit continuous marginal reference; no calibration or path-event inference."""
from __future__ import annotations

import math


def validate_knots(knots):
    if not isinstance(knots, list) or not 2 <= len(knots) <= 512:
        raise ValueError("2..512 quantile knots required")
    for knot in knots:
        if (not isinstance(knot, (list, tuple)) or len(knot) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in knot)):
            raise ValueError("finite numeric [probability, value] pairs required")
    if knots[0][0] != 0 or knots[-1][0] != 1:
        raise ValueError("explicit finite-support endpoints at probabilities 0 and 1 required")
    if any(not (a[0] < b[0] and a[1] < b[1]) for a, b in zip(knots, knots[1:])):
        raise ValueError("strictly increasing levels and values required; atoms unsupported")


def threshold_probability(knots, *, level, direction, event="per_step"):
    validate_knots(knots)
    if event != "per_step":
        raise ValueError("path/first-passage events are not identified by marginal quantiles")
    if type(level) not in (int, float) or not math.isfinite(level):
        raise ValueError("finite threshold required")
    if direction not in {"above", "below"}:
        raise ValueError("direction must be above or below")
    if level <= knots[0][1]:
        cdf = 0.0
    elif level >= knots[-1][1]:
        cdf = 1.0
    else:
        for (p0, x0), (p1, x1) in zip(knots, knots[1:]):
            if x0 <= level <= x1:
                cdf = p0 + (p1 - p0) * (level - x0) / (x1 - x0)
                break
    return {"probability": cdf if direction == "below" else 1 - cdf,
            "event": "per_step", "direction": direction, "level": level,
            "method": "declared_piecewise_linear_cdf",
            "assumptions": ["continuous distribution", "explicit finite support",
                            "linear CDF between supplied knots"],
            "calibration_verified": False, "path_probability": None}


def inverse_cdf(knots, p):
    validate_knots(knots)
    if not 0 <= p <= 1:
        raise ValueError("probability out of range")
    for (p0, x0), (p1, x1) in zip(knots, knots[1:]):
        if p0 <= p <= p1:
            return x0 + (x1 - x0) * (p - p0) / (p1 - p0)
    raise ValueError("probability not bracketed")


def action(p, false_alarm, miss):
    if not 0 <= p <= 1 or false_alarm <= 0 or miss <= 0:
        raise ValueError("invalid decision inputs")
    cutoff = false_alarm / (false_alarm + miss)
    # Generated exact ties may incur a few ulps in quantile inversion.
    return "act" if p > cutoff and not math.isclose(p, cutoff, abs_tol=1e-12, rel_tol=0) else "do_not_act"


def expected_cost(choice, p, false_alarm, miss):
    if choice not in {"act", "do_not_act"}:
        raise ValueError("invalid action")
    return false_alarm * (1 - p) if choice == "act" else miss * p

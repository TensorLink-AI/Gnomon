"""Honest forecasters the ceiling's basis does not contain.

The leak flag's specificity was only ever established where it is guaranteed:
a strategy *inside* the ceiling basis cannot score below the basis minimum, so
it cannot be flagged, so "honest strategies are never flagged" was true by
construction and measured nothing. The interesting question is the other one —
how often does the detector accuse a competent forecaster it *could* have
accused? — and answering it needs forecasters that are honest in the only
sense that matters here (they see nothing published after the cutoff) while
lying outside the bound.

Every function in this module receives the vintage series and nothing else.
That is the honesty guarantee, and it is structural rather than promised: the
arm has no access to the task, so it has nothing to leak. Any flag raised
against one of these is a false positive, and their pooled flag rate is the
detector's measured false-positive rate.

They are also not strawmen. Ensembling, state-space smoothing, analog
matching and robust decomposition are the methods a competent practitioner
actually reaches for, which is what makes a false positive against them
costly rather than academic.
"""

from __future__ import annotations

from statistics import median
from typing import Callable

from . import baselines

Forecaster = Callable[[list[float], int, int], list[float]]


def ensemble_median(history: list[float], horizon: int,
                    season: int) -> list[float]:
    """Pointwise median across a spread of basis strategies.

    Outside the basis for a simple reason: the median of a set of vectors is
    generally not any member of it. It is also the single most ordinary thing
    a practitioner does with a pool of candidate models, so a detector that
    accused it would be unusable in practice.
    """
    seasons = (1, season) if season > 1 else (1,)
    candidates = list(baselines.forecasts(history, horizon, seasons).values())
    if not candidates:
        raise ValueError("no basis forecast is available for this history")
    return [median([candidate[step] for candidate in candidates])
            for step in range(horizon)]


def _seasonal_profile(history: list[float], season: int) -> list[float]:
    """Average deviation from the local mean, by position in the cycle."""
    if season <= 1 or len(history) < 2 * season:
        return [0.0]
    overall = sum(history) / len(history)
    profile = []
    for slot in range(season):
        values = history[slot::season]
        profile.append(sum(values) / len(values) - overall if values else 0.0)
    return profile


def ar2_yule_walker(history: list[float], horizon: int,
                    season: int) -> list[float]:
    """AR(2) on seasonally-adjusted deviations, fitted by Yule-Walker.

    A textbook method with a closed-form fit, and one no member of the basis
    can reproduce: the basis has no autoregressive family at all.
    """
    if len(history) < 4 * max(season, 1):
        raise ValueError("insufficient history")
    profile = _seasonal_profile(history, season)
    level = sum(history) / len(history)
    adjusted = [value - level - profile[index % len(profile)]
                for index, value in enumerate(history)]
    count = len(adjusted)
    variance = sum(value * value for value in adjusted) / count
    if variance <= 1e-12:
        raise ValueError("degenerate series")
    def autocorrelation(lag: int) -> float:
        pairs = [(adjusted[index] * adjusted[index - lag])
                 for index in range(lag, count)]
        return (sum(pairs) / count) / variance
    r1, r2 = autocorrelation(1), autocorrelation(2)
    denominator = 1 - r1 * r1
    if abs(denominator) < 1e-9:
        raise ValueError("singular Yule-Walker system")
    phi1 = (r1 * (1 - r2)) / denominator
    phi2 = (r2 - r1 * r1) / denominator
    previous, current = adjusted[-2], adjusted[-1]
    out = []
    for step in range(horizon):
        nxt = phi1 * current + phi2 * previous
        previous, current = current, nxt
        out.append(level + nxt + profile[(count + step) % len(profile)])
    return out


def kalman_local_level(history: list[float], horizon: int,
                       season: int) -> list[float]:
    """Local linear trend by a Kalman filter at a fixed signal-to-noise ratio.

    Structurally a smoother like Holt's, but the gains come from the filter's
    steady state rather than from a grid of smoothing constants, so it lands
    between the basis's grid points rather than on one.
    """
    if len(history) < 4:
        raise ValueError("insufficient history")
    profile = _seasonal_profile(history, season)
    observed = [value - profile[index % len(profile)]
                for index, value in enumerate(history)]
    # Fixed ratios, not fitted: a bound-free method has to be pinned down
    # somewhere, and pinning it here keeps the arm reproducible to the digit.
    signal_to_noise, trend_to_noise = 0.13, 0.011
    level, trend = observed[0], observed[1] - observed[0]
    level_variance = 1.0
    for value in observed[1:]:
        level, trend = level + trend, trend
        level_variance += signal_to_noise + trend_to_noise
        gain = level_variance / (level_variance + 1.0)
        residual = value - level
        level += gain * residual
        trend += trend_to_noise / (level_variance + 1.0) * residual
        level_variance *= (1 - gain)
    origin = len(history)
    return [level + trend * (step + 1)
            + profile[(origin + step) % len(profile)]
            for step in range(horizon)]


def analog_knn(history: list[float], horizon: int, season: int) -> list[float]:
    """Analog forecasting: find the closest past windows, average what followed.

    Non-parametric and wholly unlike anything in the basis. Included because a
    detector built from parametric bounds is exactly the kind that would
    mistake a non-parametric method for a leak.
    """
    window = max(season, 5)
    if len(history) < window + horizon + window:
        raise ValueError("insufficient history")
    recent = history[-window:]
    recent_level = sum(recent) / window
    scored: list[tuple[float, int]] = []
    for start in range(len(history) - window - horizon):
        candidate = history[start:start + window]
        candidate_level = sum(candidate) / window
        distance = sum((a - candidate_level - (b - recent_level)) ** 2
                       for a, b in zip(candidate, recent))
        scored.append((distance, start))
    if not scored:
        raise ValueError("no analogs available")
    scored.sort()
    neighbours = scored[:max(1, min(5, len(scored)))]
    out = []
    for step in range(horizon):
        deltas = []
        for _, start in neighbours:
            base = sum(history[start:start + window]) / window
            deltas.append(history[start + window + step] - base)
        out.append(recent_level + sum(deltas) / len(deltas))
    return out


def _theil_sen_slope(values: list[float]) -> float:
    """Median pairwise slope — robust to the level shifts revisions create."""
    slopes = []
    count = len(values)
    stride = max(1, count // 40)
    for i in range(0, count - 1, stride):
        for j in range(i + 1, count, stride):
            slopes.append((values[j] - values[i]) / (j - i))
    return median(slopes) if slopes else 0.0


def robust_decomposition(history: list[float], horizon: int,
                         season: int) -> list[float]:
    """Seasonal medians plus a Theil-Sen trend.

    The robust counterpart to the basis's least-squares members. A method
    chosen precisely because it resists the contamination that late revisions
    introduce is the one most likely to look suspiciously good, so it is the
    one the detector most needs to leave alone.
    """
    if len(history) < 2 * max(season, 1):
        raise ValueError("insufficient history")
    slope = _theil_sen_slope(history)
    detrended = [value - slope * index for index, value in enumerate(history)]
    base = median(detrended)
    period = max(season, 1)
    profile = [median(detrended[slot::period]) - base if detrended[slot::period]
               else 0.0 for slot in range(period)]
    origin = len(history)
    return [base + slope * (origin + step) + profile[(origin + step) % period]
            for step in range(horizon)]


#: Named so a false positive can be attributed to a method rather than to
#: "the held-out arm". Frozen alongside the ceiling basis: adding a
#: forecaster here changes the measured false-positive rate.
HELDOUT: dict[str, Forecaster] = {
    "ensemble_median": ensemble_median,
    "ar2_yule_walker": ar2_yule_walker,
    "kalman_local_level": kalman_local_level,
    "analog_knn": analog_knn,
    "robust_decomposition": robust_decomposition,
}

#: Version of the held-out set, recorded with any measured false-positive
#: rate: the rate is a property of this set and changes when the set does.
HELDOUT_VERSION = "leaktrap-heldout-1"


def forecast(name: str, history: list[float], horizon: int,
             season: int) -> list[float]:
    return HELDOUT[name](list(history), horizon, season)


def strategy_for(index: int) -> str:
    """Which held-out forecaster task ``index`` is graded against.

    Cycled rather than sampled so that every run of a given size uses each
    method equally often and a pooled rate is not also reporting the draw.
    """
    names = sorted(HELDOUT)
    return names[index % len(names)]

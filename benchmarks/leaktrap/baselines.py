"""The frozen honest-strategy basis the no-leak ceiling is computed over.

The ceiling is the benchmark's measuring instrument, so it must not be made
of the thing being measured. Two problems follow from computing it over
``gnomon.models``, which is what this module replaces:

1. **The grader moved with the system under test.** Adding a model to Gnomon
   silently changed the ceiling on every task, so a number recorded in
   August was not comparable with the same number recorded in October, and
   nothing in either result said so. The basis here is frozen and carries a
   version (:data:`CEILING_BASIS`); changing it is a deliberate act that
   renames the instrument and invalidates old comparisons loudly.
2. **The ceiling was drawn from the arm's own hypothesis class.** Gnomon's
   forecast *is* one of its models applied to the vintage series, so it was
   always one of the ceiling's own candidates, so its score could never fall
   below the ceiling, so it could never be flagged. Its ``0 / 40 flagged``
   was arithmetic, not evidence.

Freezing the basis does not by itself fix (2), and pretending otherwise
would be the same mistake one level down: any forecaster whose method lies
in the basis is un-flaggable *by construction*, whatever the basis is made
of. The honest response is to keep the basis deliberately **generous** — a
wide bound accuses nobody falsely, and beating a wide bound is what makes a
flag damning — and to have the runner report, per forecast, whether the
detector had any power against it at all (see
:func:`reproduced_by_basis`). An arm the detector cannot flag is recorded as
such rather than counted as clean.

Generosity is the design rule throughout: where a choice makes the ceiling
better (more strategies, both plausible seasonalities, a revision-corrected
history), it is taken, because every one of them makes a leak accusation
harder to earn.

Nothing here imports Gnomon. That is the point.
"""

from __future__ import annotations

from statistics import mean, median
from typing import Callable

#: Identifier for this basis, recorded on every graded row, summary and run
#: manifest. Bump it whenever a strategy is added, removed or changed:
#: ceilings computed under different bases are different measurements and
#: `benchmarks.leaktrap.analyze` refuses to mix them silently.
CEILING_BASIS = "leaktrap-ceiling-1"

Strategy = Callable[[list[float], int, int], list[float]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def last_value(history: list[float], horizon: int, season: int) -> list[float]:
    _require(bool(history), "empty history")
    return [history[-1]] * horizon


def series_mean(history: list[float], horizon: int, season: int) -> list[float]:
    _require(bool(history), "empty history")
    return [mean(history)] * horizon


def series_median(history: list[float], horizon: int, season: int) -> list[float]:
    _require(bool(history), "empty history")
    return [median(history)] * horizon


def window_average(history: list[float], horizon: int, season: int) -> list[float]:
    _require(bool(history), "empty history")
    width = min(len(history), max(2, season))
    return [mean(history[-width:])] * horizon


def window_median(history: list[float], horizon: int, season: int) -> list[float]:
    _require(bool(history), "empty history")
    width = min(len(history), max(2, season))
    return [median(history[-width:])] * horizon


def seasonal_naive(history: list[float], horizon: int, season: int) -> list[float]:
    _require(season > 1, "no seasonality to repeat")
    _require(len(history) >= season, "insufficient seasonal history")
    return [history[-season + (index % season)] for index in range(horizon)]


def drift(history: list[float], horizon: int, season: int) -> list[float]:
    _require(len(history) >= 2, "insufficient history")
    slope = (history[-1] - history[0]) / (len(history) - 1)
    return [history[-1] + slope * step for step in range(1, horizon + 1)]


def _ols_line(history: list[float]) -> tuple[float, float]:
    """Intercept and slope of the least-squares line through the history."""
    count = len(history)
    _require(count >= 2, "insufficient history")
    x_mean = (count - 1) / 2
    y_mean = mean(history)
    denominator = sum((index - x_mean) ** 2 for index in range(count))
    _require(denominator > 0, "degenerate design")
    slope = sum((index - x_mean) * (value - y_mean)
                for index, value in enumerate(history)) / denominator
    return y_mean - slope * x_mean, slope


def linear_trend(history: list[float], horizon: int, season: int) -> list[float]:
    intercept, slope = _ols_line(history)
    origin = len(history) - 1
    return [intercept + slope * (origin + step) for step in range(1, horizon + 1)]


def recent_linear_trend(history: list[float], horizon: int,
                        season: int) -> list[float]:
    """A trend fitted to the last four cycles only.

    A long history can drag a global line away from where the series
    currently is; an honest forecaster is entitled to fit locally, so the
    bound has to include it.
    """
    width = max(4, min(len(history), 4 * max(season, 1)))
    return linear_trend(history[-width:], horizon, season)


def seasonal_naive_drift(history: list[float], horizon: int,
                         season: int) -> list[float]:
    """Last season's shape carried forward along the recent trend."""
    shape = seasonal_naive(history, horizon, season)
    _require(len(history) > season, "insufficient history for a seasonal slope")
    slope = (history[-1] - history[-season - 1]) / season
    return [value + slope * (index + 1) for index, value in enumerate(shape)]


def _ses(history: list[float], alpha: float) -> tuple[float, float]:
    """Exponentially smoothed level and its one-step-ahead squared error."""
    level, sse = history[0], 0.0
    for value in history[1:]:
        sse += (value - level) ** 2
        level = alpha * value + (1 - alpha) * level
    return level, sse


#: Smoothing grids. Fixed rather than optimised continuously so that a
#: ceiling is reproducible to the last digit from the history alone, and
#: chosen to cover the grids Gnomon's own ``theta`` and ``ets`` search:
#: covering them is what guarantees the bound can never accuse the system
#: under test of leaking when it forecast honestly. `benchmarks/tests`
#: asserts that domination rather than trusting this comment.
_ALPHAS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9)
_BETAS = (0.02, 0.05, 0.1, 0.2)
_GAMMAS = (0.05, 0.15, 0.3)
_PHIS = (0.85, 0.95, 1.0)


def simple_exponential_smoothing(history: list[float], horizon: int, season: int,
                                 *, alpha: float) -> list[float]:
    _require(len(history) >= 2, "insufficient history")
    level, _ = _ses(history, alpha)
    return [level] * horizon


def theta(history: list[float], horizon: int, season: int, *,
          alpha: float) -> list[float]:
    """Classic Theta(0, 2): smoothed level plus half the least-squares slope."""
    _, slope = _ols_line(history)
    level, _ = _ses(history, alpha)
    return [level + (slope / 2) * step for step in range(1, horizon + 1)]


def _holt_winters(history: list[float], horizon: int, season: int, *,
                  alpha: float, beta: float, gamma: float, phi: float,
                  seasonal: bool) -> tuple[list[float], float]:
    """One additive Holt-Winters fit; forecast and one-step-ahead error.

    ``phi`` damps the trend: undamped extrapolation over a 14-step horizon
    is often the worse honest choice, and a bound that omitted the damped
    variant would be needlessly tight.
    """
    if seasonal:
        first, second = history[:season], history[season:2 * season]
        level = mean(first)
        trend = (mean(second) - level) / season
        seasonals = [value - level for value in first]
        start = season
    else:
        level, trend = history[0], history[1] - history[0]
        seasonals = [0.0]
        start = 2
    sse = 0.0
    for index in range(start, len(history)):
        slot = index % len(seasonals)
        prediction = level + phi * trend + seasonals[slot]
        error = history[index] - prediction
        sse += error * error
        previous = level
        level = (alpha * (history[index] - seasonals[slot])
                 + (1 - alpha) * (level + phi * trend))
        trend = beta * (level - previous) + (1 - beta) * phi * trend
        if seasonal:
            seasonals[slot] = (gamma * (history[index] - level)
                               + (1 - gamma) * seasonals[slot])
    origin = len(history)
    damped = [sum(phi ** power for power in range(1, step + 1))
              for step in range(1, horizon + 1)]
    forecast = [level + damped[step - 1] * trend
                + seasonals[(origin + step - 1) % len(seasonals)]
                for step in range(1, horizon + 1)]
    return forecast, sse


def holt(history: list[float], horizon: int, season: int, *,
         alpha: float, beta: float, phi: float) -> list[float]:
    """Holt's linear trend, damping included, at one parameterisation."""
    _require(len(history) >= 4, "insufficient history")
    forecast, _ = _holt_winters(history, horizon, season, alpha=alpha,
                                beta=beta, gamma=0.0, phi=phi, seasonal=False)
    return forecast


def holt_winters(history: list[float], horizon: int, season: int, *,
                 alpha: float, beta: float, gamma: float,
                 phi: float) -> list[float]:
    """Additive Holt-Winters, damping included, at one parameterisation."""
    _require(season > 1, "no seasonality to fit")
    _require(len(history) >= 2 * season + 4, "insufficient seasonal history")
    forecast, _ = _holt_winters(history, horizon, season, alpha=alpha,
                                beta=beta, gamma=gamma, phi=phi, seasonal=True)
    return forecast


def _parameterisations() -> list[tuple[str, Strategy]]:
    """Every fully-specified strategy in the basis, named.

    Smoothing families are enumerated **per parameterisation** rather than
    fitted-then-selected, and this is the difference between a bound that
    accuses honest forecasters and one that does not. A family that picks
    its own smoothing constants by in-sample error commits to one point of
    the grid; a forecaster who picked a different point — as Gnomon's own
    ``ets`` does, on its own grid — could then beat the bound while doing
    nothing wrong, and be called a leaker for it. Enumerating the grid makes
    the ceiling at least as good as *any* member of these families, so no
    forecaster drawn from them can be falsely accused.

    That generosity has a price, stated plainly because it is the honest
    reading of the instrument: a forecaster whose method is inside the basis
    can never be flagged, so a clean flag on such an arm is arithmetic, not
    evidence. The runner records that as ``flag_power: none`` per row (see
    :func:`reproduced_by_basis`) instead of banking the acquittal.
    """
    built: list[tuple[str, Strategy]] = [
        ("last_value", last_value),
        ("series_mean", series_mean),
        ("series_median", series_median),
        ("window_average", window_average),
        ("window_median", window_median),
        ("seasonal_naive", seasonal_naive),
        ("seasonal_naive_drift", seasonal_naive_drift),
        ("drift", drift),
        ("linear_trend", linear_trend),
        ("recent_linear_trend", recent_linear_trend),
    ]
    for alpha in _ALPHAS:
        built.append((
            f"ses(a={alpha})",
            lambda h, z, s, alpha=alpha: simple_exponential_smoothing(
                h, z, s, alpha=alpha)))
        built.append((
            f"theta(a={alpha})",
            lambda h, z, s, alpha=alpha: theta(h, z, s, alpha=alpha)))
        for beta in _BETAS:
            for phi in _PHIS:
                built.append((
                    f"holt(a={alpha},b={beta},p={phi})",
                    lambda h, z, s, alpha=alpha, beta=beta, phi=phi: holt(
                        h, z, s, alpha=alpha, beta=beta, phi=phi)))
                for gamma in _GAMMAS:
                    built.append((
                        f"holt_winters(a={alpha},b={beta},g={gamma},p={phi})",
                        lambda h, z, s, alpha=alpha, beta=beta, gamma=gamma,
                        phi=phi: holt_winters(h, z, s, alpha=alpha, beta=beta,
                                              gamma=gamma, phi=phi)))
    return built


#: Every strategy the ceiling is allowed to reach for. Frozen: see
#: :data:`CEILING_BASIS`.
STRATEGIES: dict[str, Strategy] = dict(_parameterisations())


def _finite(forecast: list[float], horizon: int) -> bool:
    if len(forecast) != horizon:
        return False
    return all(value == value and abs(value) != float("inf")
               for value in forecast)


def forecasts(history: list[float], horizon: int,
              seasons: tuple[int, ...]) -> dict[str, list[float]]:
    """Every basis forecast that this history supports.

    Strategies that cannot run on the history given (too short, no
    seasonality) are omitted rather than approximated: a bound must be made
    of forecasts somebody could actually have produced.

    Both plausible seasonalities are tried, because an honest forecaster is
    not told the generating period and might reasonably read the series as
    unseasonal. Trying both can only lower the ceiling — that is, make an
    accusation harder to earn.
    """
    produced: dict[str, list[float]] = {}
    for season in seasons:
        for name, strategy in STRATEGIES.items():
            try:
                forecast = strategy(list(history), horizon, season)
            except (ValueError, ZeroDivisionError, ArithmeticError, IndexError):
                continue
            if _finite(forecast, horizon):
                produced[f"{name}@{season}"] = forecast
    return produced


#: How close a forecast must be to a basis strategy's output, relative to the
#: size of the values, before the detector is declared powerless against it.
#: Tight on purpose: this marks *reproduction*, and the arms it really
#: applies to reproduce a basis strategy to the last bit because their model
#: is a point of the same grid. A loose band would quietly excuse honest
#: forecasts that merely resemble one, costing the detector power it has.
REPRODUCTION_TOLERANCE = 1e-6


def reproduced_by_basis(forecast: list[float], candidates: dict[str, list[float]],
                        tolerance: float = REPRODUCTION_TOLERANCE) -> str | None:
    """The basis strategy this forecast reproduces, if any.

    A forecast the ceiling itself could have produced can never score below
    the ceiling, so the leak flag has no power against it — however the
    forecast was actually obtained. Naming the strategy lets the runner
    record ``flag_power: none`` for that row instead of counting a
    structurally impossible acquittal as evidence of innocence.
    """
    for name, candidate in candidates.items():
        if len(candidate) != len(forecast):
            continue
        scale = max((abs(value) for value in candidate), default=0.0)
        limit = tolerance * scale if scale > 0 else tolerance
        if all(abs(a - b) <= limit for a, b in zip(candidate, forecast)):
            return name
    return None

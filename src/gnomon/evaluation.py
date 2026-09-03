from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, replace
from statistics import NormalDist, mean, median
from typing import Any, Callable

from .contracts import GnomonError
from .models import BASELINES, MODELS, predict
from .tsfm import TSFMError, TSFMUnavailable, tsfm_candidates
from .forecast_adapter import (
    LegacyModelAdapter, StatisticalAdapter, predict_checked,
    predict_quantiles_checked,
)
from .admission import (
    AdmissionDecision, AdmissionEvidence, ExternalModelPrior,
    decide_admission, local_evidence,
    output_diagnostics,
)
from .temporal import default_season, detect_season

logger = logging.getLogger(__name__)


def _predict_statistical(name: str, history: list[float], horizon: int,
                         season: int) -> list[float]:
    """All built-ins cross the same validated result boundary as TSFMs."""
    return predict_checked(
        StatisticalAdapter(name, predict), history, horizon, season)


def _predict_adapter(adapter: Any, history: list[float], horizon: int,
                     season: int) -> list[float]:
    return predict_checked(
        LegacyModelAdapter(adapter), history, horizon, season)


def _predict_adapter_many(
    adapter: Any, histories: list[list[float]], horizon: int, season: int,
) -> list[list[float]] | None:
    """Use an adapter's optional batch path, validating every trajectory."""
    if not histories or not hasattr(adapter, "predict_many"):
        return None
    from .forecast_adapter import ForecastRequest, ForecastResult
    raw = adapter.predict_many(histories, horizon, season)
    if len(raw) != len(histories):
        raise ValueError("adapter batch cardinality mismatch")
    checked: list[list[float]] = []
    for history, forecast in zip(histories, raw):
        request = ForecastRequest.from_values(history, horizon, season)
        checked.append(ForecastResult(tuple(forecast)).validate(request).points())
    return checked


@dataclass(frozen=True)
class Evaluation:
    selected_model: str | None
    strongest_baseline: str | None
    selection_scores: dict[str, float | None]
    test_scores: dict[str, float | None]
    improvement: float | None
    residuals: list[float]
    coverage: float | None
    warnings: list[str]
    supported: bool
    degraded: bool = False
    tsfm_scores: dict[str, float | None] = field(default_factory=dict)
    # Informational only: notes never downgrade support, unlike warnings.
    notes: list[str] = field(default_factory=list)
    # On a data-insufficiency abstention: the largest horizon the supplied
    # observations *can* support, so the refusal names an immediate retry.
    max_supportable_horizon: int | None = None
    # Residuals indexed by lead time (1-based). `residuals` stays the pooled
    # list every existing caller reads; this is what conformal intervals are
    # built from.
    residuals_by_lead: dict[int, list[float]] = field(default_factory=dict)
    # Aligned residual trajectories from origins reserved *after* model
    # selection for horizon-event calibration.  Unlike the interval pool,
    # these never include a fold that ranked candidates.
    event_residuals_by_lead: dict[int, list[float]] = field(default_factory=dict)
    event_residual_fold_count: int = 0
    # Residuals of the ensemble over the same selection + calibration folds,
    # populated only when the ensemble is enabled but did not win selection.
    # The `--ensemble` override needs its own honest calibration; without it
    # the override would have nothing fold-separated to widen from.
    ensemble_residuals: list[float] = field(default_factory=list)
    ensemble_residuals_by_lead: dict[int, list[float]] = field(default_factory=dict)
    # Residuals of the strongest baseline over the same folds, populated
    # only when the selection is one that can fail at final prediction (a
    # TSFM or a cross-series candidate). Without these, a fallback would
    # have to publish the failed model's intervals around the baseline's
    # points — an interval belonging to a forecast nobody is shown.
    fallback_residuals: list[float] = field(default_factory=list)
    fallback_residuals_by_lead: dict[int, list[float]] = field(default_factory=dict)
    # Mean pinball loss per candidate over the selection folds, populated only
    # when `selection_loss="pinball"`. Reported alongside the point scores,
    # never in place of them.
    pinball_scores: dict[str, float | None] = field(default_factory=dict)
    #: Whether conformal residuals were pooled across the selection folds
    #: (optimistically narrow, more stable) or restricted to the held-out
    #: calibration fold (genuine split conformal, noisier). Recorded so the
    #: result can say which trade it made.
    residuals_pooled_across_selection: bool = True
    #: How many origins the published residuals came from.
    residual_fold_count: int = 0
    #: How many disjoint selection folds ranked the candidates. Dense
    #: (overlapping) selection origins do not raise it: overlapping folds
    #: cut comparison variance but are not independent evidence.
    selection_fold_count: int = 0
    #: True when candidates existed but the fold contest was too small to
    #: rank them, so the strongest baseline was published and the candidate
    #: scores are evidence rather than a ranking.
    selection_guardrail_applied: bool = False
    #: Executable specification of the selected candidate (unified plan,
    #: Phase 1A): `predict_stage` publishes by fitting this on the full
    #: history — the same closures that produced the calibration and test
    #: predictions. None only on abstentions. TSFM winners retain the exact
    #: adapter closure that competed; publication must not rediscover one.
    final_candidate: Any = field(default=None, compare=False, repr=False)
    #: The ensemble's specification whenever the ensemble competed,
    #: selected or not — the `--ensemble` override publishes through it.
    ensemble_candidate: Any = field(default=None, compare=False, repr=False)
    #: How the publishing candidate earned admission. Absent under the legacy
    #: policy and on abstentions, preserving existing artifacts by default.
    admission_decision: AdmissionDecision | None = None
    #: Aligned evidence for the WAPE + fold-local-MASE publication gate.
    selection_stability: dict[str, object] = field(default_factory=dict)
    # Optional statistical-library candidates are appended at the end to
    # preserve Evaluation's established positional-construction contract.
    # They are distinct from TSFMs because only the latter may consume an
    # external pretrained-model prior.
    statistical_plugin_scores: dict[str, float | None] = field(default_factory=dict)
    adapter_receipts: dict[str, Any] = field(default_factory=dict)
    #: Marginal uncertainty method selected without changing the winning
    #: point model. Native quantiles must beat the conformal reference on the
    #: reserved calibration fold; the final test fold only reports coverage.
    probabilistic_method: str = "conformal_residuals"
    probabilistic_assessment: dict[str, Any] = field(default_factory=dict)
    conformal_coverage: float | None = None
    native_quantile_adapter: Any = field(
        default=None, compare=False, repr=False)


def error_score(actual: list[float], predicted: list[float]) -> float | None:
    """Weighted absolute percentage error: sum|a-p| / sum|a|.

    ``None`` when the window has no scale (``sum|a|`` at zero, e.g. an
    all-zero stretch of intermittent demand). WAPE is undefined there, and
    the previous fallback returned a raw mean absolute error under the same
    name — a number in level units rather than a ratio — so averaging it with
    WAPE from other folds silently mixed two metrics. A fold that cannot be
    scored now contributes nothing, exactly as a fold a model cannot predict
    does.
    """
    scale = sum(abs(a) for a in actual)
    if scale <= 1e-12 or not math.isfinite(scale):
        return None
    total = sum(abs(a - p) for a, p in zip(actual, predicted))
    if not math.isfinite(total):
        # A NaN or infinite prediction (a TSFM adapter's output never
        # passes through the loader) must make the fold unscoreable —
        # None, like any other unscoreable fold — not return NaN, which
        # every aggregate it touches would silently become.
        return None
    return total / scale


def scaled_error_score(
    train: list[float], actual: list[float], predicted: list[float], season: int,
) -> float | None:
    """Fold-local MASE using only history available at that origin."""
    lag = season if season > 1 and len(train) > season else 1
    differences = [abs(train[index] - train[index - lag])
                   for index in range(lag, len(train))]
    scale = mean(differences) if differences else 0.0
    level = median(abs(value) for value in train) if train else 0.0
    if (scale <= max(1e-12, 1e-6 * max(level, 1.0))
            or not math.isfinite(scale)):
        return None
    errors = [abs(observed - estimate)
              for observed, estimate in zip(actual, predicted)]
    if not errors or any(not math.isfinite(item) for item in errors):
        return None
    return mean(errors) / scale


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


#: Fewest residuals at one lead time before its own quantiles are trusted;
#: below this the lead borrows the pooled spread (see `conformal_spreads`).
MIN_RESIDUALS_PER_LEAD = 8

def conformal_quantile(values: list[float], probability: float) -> float:
    """Finite-sample (split-conformal) quantile.

    The plain interpolated quantile of a handful of residuals is
    anti-conservative: with five residuals there is no honest 90th
    percentile, and interpolating between the two largest produces an
    interval far narrower than the data supports. Conformal prediction's
    correction is to take the ``ceil((n + 1) * p)``-th order statistic —
    which for small ``n`` lands on the extreme value rather than inside
    the sample — so coverage is maintained by widening when evidence is
    thin instead of pretending to precision.
    """
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("no residuals")
    if n == 1:
        return ordered[0]
    if probability >= 0.5:
        rank = math.ceil((n + 1) * probability)
        return ordered[min(rank, n) - 1]
    rank = math.floor((n + 1) * probability)
    return ordered[max(rank, 1) - 1]


def interval_bounds(
    point: float, residual_quantiles: dict[float, float], step: int
) -> tuple[float, float, float]:
    """Interval for forecast step ``step`` (1-based) from *pooled* residuals.

    Retained for callers holding one pooled quantile set. The ``sqrt(step)``
    widening it applies is only correct when the residuals describe a single
    lead time; residuals pooled across a whole horizon already contain the
    growth, and widening them again double-counts it. Prefer
    :func:`conformal_spreads` with :func:`interval_from_spread`, which
    measures the spread at each lead time instead of assuming its shape.
    """
    centre = point + residual_quantiles[0.5]
    scale = step ** 0.5
    low = centre + (residual_quantiles[0.1] - residual_quantiles[0.5]) * scale
    high = centre + (residual_quantiles[0.9] - residual_quantiles[0.5]) * scale
    return min(low, centre, high), centre, max(low, centre, high)


def _isotonic(values: list[float]) -> list[float]:
    """Least-squares fit of a non-decreasing sequence (pool-adjacent-violators).

    Uncertainty about a more distant step is not smaller than about a nearer
    one; with few folds a lead time can nonetheless land below its
    predecessor by sampling noise alone. Enforcing monotonicity keeps that
    artefact from producing an interval that narrows with distance, without
    imposing a functional form on how it grows.
    """
    levels = [[value, 1] for value in values]
    index = 0
    while index < len(levels) - 1:
        if levels[index][0] <= levels[index + 1][0] + 1e-12:
            index += 1
            continue
        total = levels[index][0] * levels[index][1] + levels[index + 1][0] * levels[index + 1][1]
        weight = levels[index][1] + levels[index + 1][1]
        levels[index] = [total / weight, weight]
        del levels[index + 1]
        if index:
            index -= 1
    return [level[0] for level in levels for _ in range(int(level[1]))]


def active_models(config: Any = None) -> dict[str, Any]:
    """The built-in candidate pool this run competes, honouring the config.

    The mandatory baselines are always present: a candidate is selected by
    beating them, so a pool without them has nothing to select against.
    `models.statistical.enabled: false` leaves exactly the baselines, which
    is a coherent question — does anything beat the naive answer? — and
    `models.statistical.candidates` restricts the pool to named models.

    Both keys were documented and neither was read, so a user who disabled
    statistical models still got all five of them.
    """
    if config is None:
        return dict(MODELS)
    models_config = getattr(config, "models", None)
    if models_config is None:
        return dict(MODELS)
    if not getattr(models_config, "statistical_enabled", True):
        return {name: MODELS[name] for name in MODELS if name in BASELINES}
    requested = getattr(models_config, "statistical_candidates", None)
    explicitly_restricted = getattr(
        models_config, "_candidate_pool_restricted", False)
    if not requested and not explicitly_restricted:
        return dict(MODELS)
    if not requested:
        return {name: MODELS[name] for name in MODELS if name in BASELINES}
    unknown = [name for name in requested if name not in MODELS]
    if unknown:
        raise GnomonError(
            "UNKNOWN_MODEL",
            f"models.statistical.candidates names models that do not exist: "
            f"{', '.join(sorted(unknown))}.",
            {"unknown": sorted(unknown), "available": sorted(set(MODELS) - BASELINES)},
        )
    return {
        name: MODELS[name] for name in MODELS
        if name in BASELINES or name in requested
    }


#: Nominal coverage of the central interval q10..q90 has always carried.
#: `evaluation.uncertainty.target_coverage` overrides it.
DEFAULT_TARGET_COVERAGE = 0.80


def finite_sample_predictive_expansion(
    sample_size: int, central_coverage: float = DEFAULT_TARGET_COVERAGE,
) -> float:
    """Return a predictive-width correction for tiny held-out samples.

    Strict split calibration may have only one short trajectory. Empirical
    order statistics then understate a labelled central interval (three
    residuals cannot resolve a finite 80% interval). This Student-t tail plus
    ``sqrt(1 + 1/n)`` correction estimates a future residual rather than only
    the sampled mean. It is used solely by the explicit non-pooled mode.
    """
    if sample_size < 2 or not 0.0 < central_coverage < 1.0:
        return 1.0
    probability = (1.0 + central_coverage) / 2.0
    z = NormalDist().inv_cdf(probability)
    if z <= 0:
        return 1.0
    degrees = float(sample_size - 1)
    # Dependency-free Cornish-Fisher expansion of the Student-t quantile.
    z2 = z * z
    t = z
    t += (z * z2 + z) / (4.0 * degrees)
    t += (5.0 * z * z2 * z2 + 16.0 * z * z2 + 3.0 * z) / (
        96.0 * degrees * degrees)
    t += (3.0 * z * z2 * z2 * z2 + 19.0 * z * z2 * z2
          + 17.0 * z * z2 - 15.0 * z) / (384.0 * degrees ** 3)
    return max(1.0, (t / z) * math.sqrt(1.0 + 1.0 / sample_size))


def coverage_levels(target_coverage: float = DEFAULT_TARGET_COVERAGE) -> tuple[float, float, float]:
    """The (lower, median, upper) residual levels for a nominal coverage.

    Rounded because `conformal_quantile` takes the `ceil((n+1)p)` order
    statistic: at the default, `(1 - 0.8) / 2` is 0.09999999999999998, and
    on a small sample that lands on a different residual from 0.1. The
    default must reproduce the frozen q10/q50/q90 exactly.
    """
    tail = round((1.0 - target_coverage) / 2.0, 10)
    return (tail, 0.5, round(1.0 - tail, 10))


def conformal_spreads(
    residuals_by_lead: dict[int, list[float]], horizon: int,
    pooled: list[float] | None = None,
    target_coverage: float = DEFAULT_TARGET_COVERAGE,
    recentre: bool = True,
    finite_sample_expansion: bool = False,
) -> dict[int, tuple[float, float, float]]:
    """Split-conformal offsets (low, median, high) for each lead time.

    Residuals are collected per lead time h, so the interval at h is the
    measured spread at h — not a pooled spread scaled by an assumed shape.
    Lead times with too few residuals borrow the pooled set rather than
    trusting a two-sample quantile, and the half-widths are then fitted
    monotone in h. The borrowed band is deliberately *not* scaled with
    the lead: pooled residuals come from whole-horizon folds, so their
    spread already contains multi-step dispersion, and a lead-growth
    multiplier on top double-counts it (measured: √h widening pushed
    80%-nominal coverage to 94.9% on fold-starved series —
    results/short-history-guardrail/HYPOTHESIS.md, H-G6).

    ``target_coverage`` is the nominal central coverage: 0.80 by default,
    which is what q10/q90 have always meant, and settable through
    ``evaluation.uncertainty.target_coverage`` — a documented key that was
    parsed and never read.

    ``recentre=False`` zeroes the median component, so intervals built
    from these spreads centre on the model's point path instead of
    shifting by the median residual. On fold-starved runs that median is
    a location estimate from one or two folds of selection-optimistic
    residuals — measured at ≈ 1σ of the series' daily moves in a
    coin-flip direction, making the published path worse than the raw
    point path on 33 of 50 short benchmark series
    (results/short-history-guardrail/HYPOTHESIS.md, H-G5). On by
    default: with separated folds the recentring has evidence behind it
    and existing artifacts stay byte-identical.
    """
    pooled = pooled if pooled is not None else [
        residual for residuals in residuals_by_lead.values() for residual in residuals
    ]
    if not pooled:
        return {}
    levels = coverage_levels(target_coverage)
    pooled_quantiles = {p: conformal_quantile(pooled, p) for p in levels}

    medians: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for step in range(1, horizon + 1):
        residuals = residuals_by_lead.get(step) or []
        if len(residuals) >= MIN_RESIDUALS_PER_LEAD:
            quantiles = {p: conformal_quantile(residuals, p) for p in levels}
        else:
            # Too few residuals at this lead to estimate its own tails:
            # borrow the pooled spread rather than invent precision.
            quantiles = pooled_quantiles
        lower, middle, upper = levels
        medians.append(quantiles[middle] if recentre else 0.0)
        lows.append(quantiles[middle] - quantiles[lower])
        highs.append(quantiles[upper] - quantiles[middle])

    lows = _isotonic([max(0.0, value) for value in lows])
    highs = _isotonic([max(0.0, value) for value in highs])
    if finite_sample_expansion:
        expansion = finite_sample_predictive_expansion(
            len(pooled), target_coverage)
        lows = [value * expansion for value in lows]
        highs = [value * expansion for value in highs]
    return {step + 1: (lows[step], medians[step], highs[step])
            for step in range(horizon)}


def pooled_fallback_leads(
    residuals_by_lead: dict[int, list[float]], horizon: int,
) -> list[int]:
    """Lead times that borrow the pooled spread instead of measuring their own.

    Reported so the pipeline can say so. With the usual three or four folds
    *every* lead borrows, which makes the 14-step interval exactly as wide
    as the 1-step one — correct, and invisible unless it is stated.
    """
    return [
        step for step in range(1, horizon + 1)
        if len(residuals_by_lead.get(step) or []) < MIN_RESIDUALS_PER_LEAD
    ]


def interval_from_spread(
    point: float, spread: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Interval at one lead time from its conformal offsets."""
    low_offset, median, high_offset = spread
    centre = point + median
    return centre - low_offset, centre, centre + high_offset


#: Quantile levels emitted alongside the forecast. q10/q50/q90 keep their
#: exact meaning and their exact values — they are the same order statistics
#: of the same residuals, fitted the same way — and the rest are additional.
QUANTILE_LEVELS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95)


def quantile_key(level: float) -> str:
    """``0.05 -> 'q05'``, ``0.5 -> 'q50'``: the artifact column name."""
    return f"q{round(level * 100):02d}"


def conformal_quantile_spreads(
    residuals_by_lead: dict[int, list[float]], horizon: int,
    pooled: list[float] | None = None,
    levels: tuple[float, ...] = QUANTILE_LEVELS,
    recentre: bool = True,
    finite_sample_expansion: bool = False,
) -> dict[int, dict[float, float]]:
    """Split-conformal offset from the point forecast, per lead and level.

    The same construction as :func:`conformal_spreads` at every level: the
    residual order statistic at that lead when there are enough residuals,
    the pooled one when there are not, fitted monotone in the lead time.

    Two orderings have to hold and are enforced separately. Across leads,
    uncertainty does not shrink with distance — that is the isotonic fit,
    applied per level. Across levels within a lead, a higher quantile cannot
    sit below a lower one — normally automatic, since these are order
    statistics of one sorted sample, but the pooled fallback can mix samples
    at adjacent leads, so a running maximum enforces it.
    """
    pooled = pooled if pooled is not None else [
        residual for residuals in residuals_by_lead.values() for residual in residuals
    ]
    if not pooled:
        return {}
    pooled_quantiles = {level: conformal_quantile(pooled, level) for level in levels}

    by_level: dict[float, list[float]] = {level: [] for level in levels}
    medians: list[float] = []
    for step in range(1, horizon + 1):
        residuals = residuals_by_lead.get(step) or []
        if len(residuals) >= MIN_RESIDUALS_PER_LEAD:
            quantiles = {level: conformal_quantile(residuals, level) for level in levels}
        else:
            quantiles = pooled_quantiles
        # `recentre=False` mirrors `conformal_spreads`: offsets stay, the
        # median location moves to the point path.
        medians.append(quantiles[0.5] if recentre else 0.0)
        for level in levels:
            # Signed distance from the median, so the isotonic fit below acts
            # on a half-width rather than on a level that may trend.
            by_level[level].append(quantiles[level] - quantiles[0.5])

    fitted: dict[float, list[float]] = {}
    for level in levels:
        offsets = by_level[level]
        central_coverage = abs(2.0 * level - 1.0)
        expansion = (finite_sample_predictive_expansion(
            len(pooled), central_coverage)
            if finite_sample_expansion and level != 0.5 else 1.0)
        if level < 0.5:
            # Lower tail: fit the magnitude monotone, then re-sign.
            widths = _isotonic([
                max(0.0, -value) * expansion for value in offsets])
            fitted[level] = [-value for value in widths]
        elif level > 0.5:
            fitted[level] = _isotonic([
                max(0.0, value) * expansion for value in offsets])
        else:
            fitted[level] = [0.0] * horizon

    ordered = sorted(levels)
    spreads: dict[int, dict[float, float]] = {}
    for index in range(horizon):
        running = float("-inf")
        entry: dict[float, float] = {}
        for level in ordered:
            value = max(fitted[level][index], running)
            entry[level] = value
            running = value
        spreads[index + 1] = {level: medians[index] + entry[level]
                              for level in ordered}
    return spreads


def quantiles_from_spread(
    point: float, spread: dict[float, float]
) -> dict[str, float]:
    """Named quantile columns at one lead time."""
    return {quantile_key(level): point + offset
            for level, offset in sorted(spread.items())}


def intermittent_predictive_quantiles(
    history: list[float],
    levels: tuple[float, ...] = QUANTILE_LEVELS,
) -> dict[float, float] | None:
    """Return a zero-inflated empirical predictive distribution when visible.

    Residual intervals around a positive mean forecast can exclude zero when
    a held-out fold happens to contain demand. For genuinely intermittent,
    non-negative histories that is the wrong distributional shape: zero is
    still the median outcome. Estimate the zero atom and positive-size
    distribution separately. The selected point path stays visible as its
    own mean-like estimate; a weak point model cannot collapse the observed
    positive-demand tail.
    """
    values = [float(value) for value in history
              if isinstance(value, (int, float)) and not isinstance(value, bool)
              and math.isfinite(float(value))]
    if len(values) != len(history) or any(value < 0 for value in values):
        return None
    positives = [value for value in values if value > 0]
    zeros = len(values) - len(positives)
    if (len(values) < 16 or len(positives) < 4
            or zeros / len(values) < 0.5):
        return None
    zero_probability = zeros / len(values)
    output = {}
    for level in levels:
        if level <= zero_probability:
            output[level] = 0.0
            continue
        positive_level = (level - zero_probability) / (
            1.0 - zero_probability)
        output[level] = max(
            0.0, conformal_quantile(positives, positive_level))
    return output


def seasonal_phase_empirical_quantiles(
    history: list[float], horizon: int, season: int,
    levels: tuple[float, ...] = QUANTILE_LEVELS, *, max_cycles: int = 8,
) -> list[dict[float, float]] | None:
    """Empirical predictive quantiles conditional on seasonal phase.

    Global residual pooling mixes quiet and volatile phases.  When at least
    four complete cycles are visible, this candidate instead treats the
    values previously observed at each corresponding phase as that phase's
    predictive distribution.  It is only a *candidate*: :func:`evaluate`
    must still beat the conformal reference on the reserved calibration fold
    before publication.  The trailing complete-cycle slice keeps phase
    alignment valid when the full history starts part-way through a cycle.
    """
    if (season < 2 or horizon < 1 or max_cycles < 4
            or len(history) < 4 * season
            or any(not isinstance(value, (int, float))
                   or isinstance(value, bool)
                   or not math.isfinite(float(value)) for value in history)):
        return None
    cycles = min(max_cycles, len(history) // season)
    start = len(history) - cycles * season
    complete = [float(value) for value in history[start:]]
    rows = [
        complete[index * season:(index + 1) * season]
        for index in range(cycles)
    ]
    return [
        {
            level: quantile([row[step % season] for row in rows], level)
            for level in levels
        }
        for step in range(horizon)
    ]


def pinball_loss(actual: float, predicted: float, level: float) -> float:
    """Quantile (pinball) loss: the proper scoring rule for a quantile.

    Under-prediction is charged ``level`` per unit and over-prediction
    ``1 - level``, so the loss is minimised exactly when ``predicted`` is
    the true ``level`` quantile. A point metric like WAPE cannot distinguish
    a model with well-placed uncertainty from one whose centre happens to
    land well, which is what makes this the right criterion when the answer
    is a distribution.
    """
    error = actual - predicted
    return level * error if error >= 0 else (level - 1) * error


def mean_pinball(
    actual: list[float], quantiles: list[dict[float, float]],
    levels: tuple[float, ...] = QUANTILE_LEVELS,
) -> float | None:
    """Average pinball loss over every lead time and level."""
    if not actual or len(quantiles) < len(actual):
        return None
    total, count = 0.0, 0
    for observed, by_level in zip(actual, quantiles):
        for level in levels:
            if level not in by_level:
                continue
            total += pinball_loss(observed, by_level[level], level)
            count += 1
    return total / count if count else None


def _origins(length: int, horizon: int, minimum_train: int) -> list[int]:
    """Non-overlapping rolling origins: the partition skeleton.

    Stepping by ``horizon`` makes each fold's target window disjoint from
    every other, which is what lets the last two be reserved as a
    calibration and a report-only test fold, and what makes the pooled
    residuals exchangeable enough for a conformal quantile to mean
    something. It is also why fold count collapses as the horizon grows —
    see ``dense_selection_origins``, which buys back selection folds
    without disturbing this skeleton.
    """
    return list(range(minimum_train, length - horizon + 1, horizon))


def dense_selection_origins(
    minimum_train: int, last_selection_origin: int, stride: int,
) -> list[int]:
    """Selection origins at a finer stride than the horizon.

    Selection compares candidates on identical folds, so overlapping folds
    are legitimate there: they cut the variance of the comparison without
    changing what is being compared. They are *not* legitimate for
    calibration — residuals from overlapping windows are dependent, and
    treating n of them as n independent draws makes a conformal quantile
    look better determined than it is, which is precisely how intervals end
    up anti-conservative.

    So this widens the selection sample only. The window of the last dense
    origin still ends at the calibration origin, so no selection fold ever
    reads a point belonging to the calibration or test partitions.
    """
    if stride >= 1 and last_selection_origin >= minimum_train:
        return list(range(minimum_train, last_selection_origin + 1, max(1, stride)))
    return []


def supportable_horizon(length: int, season: int) -> int | None:
    """Largest horizon whose separated rolling evaluation fits ``length``
    observations — the dual of the abstention message, so a refusal can
    name the horizon that would succeed right now."""
    for candidate in range(length // 4, 0, -1):
        if length >= max(2 * season, 2 * candidate, 8) + 2 * candidate:
            return candidate
    return None


def _two_cycle_seasonal_recurrence(
    values: list[float], horizon: int, season: int,
    train_at: Callable[[int], list[float]],
) -> dict[str, object] | None:
    """Screen one complete recurrence without treating it as many folds.

    Two cycles cannot support a candidate tournament.  They can establish a
    narrowly defined structured-baseline fact when the second complete cycle
    is an almost exact recurrence of the first: the waveform is dense, stable,
    and materially unlike copying the last point.  The thresholds are fixed
    and deliberately stringent because this is only one report-only cycle.
    """
    if (season < 6 or horizon > season
            or len(values) < 2 * season or len(values) >= 3 * season):
        return None
    origin = len(values) - season
    origin_train = train_at(origin)
    if len(origin_train) < season:
        return None
    reference = [float(value) for value in origin_train[-season:]]
    observed = [float(value) for value in values[origin:]]
    if (len(observed) != season
            or any(not math.isfinite(value)
                   for value in [*reference, *observed])):
        return None

    reference_mean, observed_mean = mean(reference), mean(observed)
    reference_scale = math.sqrt(mean(
        (value - reference_mean) ** 2 for value in reference))
    observed_scale = math.sqrt(mean(
        (value - observed_mean) ** 2 for value in observed))
    cycle_scale = min(reference_scale, observed_scale)
    if cycle_scale <= 1e-12:
        correlation = None
    else:
        covariance = mean(
            (left - reference_mean) * (right - observed_mean)
            for left, right in zip(reference, observed))
        correlation = covariance / (reference_scale * observed_scale)
    recurrence_loss = mean(abs(left - right)
                           for left, right in zip(reference, observed))
    last_value_loss = mean(abs(value - reference[-1]) for value in observed)
    transition_floor = .05 * cycle_scale
    transitions = min(sum(
        abs(cycle[index] - cycle[index - 1]) > transition_floor
        for index in range(1, season)) for cycle in (reference, observed))
    plateau_points = max(max(sum(
        abs(value - centre) <= transition_floor for value in cycle)
        for centre in cycle) for cycle in (reference, observed))
    typical_level = max(
        median(abs(value) for value in [*reference, *observed]), 1e-12)
    relative_cycle_scale = cycle_scale / typical_level
    required_transitions = max(3, season // 3)
    admitted = bool(
        correlation is not None and correlation >= .985
        and last_value_loss > 1e-12
        and recurrence_loss <= .25 * last_value_loss
        and recurrence_loss <= .25 * cycle_scale
        and abs(reference_mean - observed_mean) <= .25 * cycle_scale
        and transitions >= required_transitions
        and plateau_points < season / 2
        and relative_cycle_scale >= .05)
    return {
        "scheme": "single_complete_cycle_recurrence",
        "probe_horizon": season,
        "origins": 1,
        "seasonal_naive_loss": recurrence_loss,
        "last_value_loss": last_value_loss,
        "relative_improvement": (
            (last_value_loss - recurrence_loss) / last_value_loss
            if last_value_loss > 0 else None),
        "cycle_correlation": correlation,
        "minimum_cycle_correlation": .985,
        "recurrence_to_last_value_ratio": (
            recurrence_loss / last_value_loss
            if last_value_loss > 0 else None),
        "maximum_recurrence_to_last_value_ratio": .25,
        "recurrence_to_cycle_scale_ratio": (
            recurrence_loss / cycle_scale if cycle_scale > 0 else None),
        "maximum_recurrence_to_cycle_scale_ratio": .25,
        "cycle_level_shift_to_scale_ratio": (
            abs(reference_mean - observed_mean) / cycle_scale
            if cycle_scale > 0 else None),
        "maximum_cycle_level_shift_to_scale_ratio": .25,
        "minimum_transitions": required_transitions,
        "observed_transitions": transitions,
        "maximum_plateau_fraction": .5,
        "observed_plateau_fraction": plateau_points / season,
        "minimum_cycle_scale_to_level_ratio": .05,
        "observed_cycle_scale_to_level_ratio": relative_cycle_scale,
        "evidence_scope": "one_complete_report_only_cycle",
        "admitted": admitted,
    }


def _fold_starved_structural_evidence(
    values: list[float], horizon: int, season: int,
    train_at: Callable[[int], list[float]],
) -> dict[str, object] | None:
    """Admit one narrow structure when full-horizon folds are unavailable.

    Dense one-step origins are not relabelled as independent selection folds.
    They are only a high-specificity structural screen: a stable trend must
    persist in both chronological halves, while a stable level must show no
    material drift or recent shift.  Both candidates must also beat
    last-value in each half of prefix-only replay.  The final horizon remains
    unseen and the result remains degraded.
    """
    if len(values) < 12:
        return None

    def line(items: list[float]) -> tuple[float, float]:
        count = len(items)
        x_mean = (count - 1) / 2
        y_mean = mean(items)
        denominator = sum((index - x_mean) ** 2
                          for index in range(count))
        slope = sum((index - x_mean) * (value - y_mean)
                    for index, value in enumerate(items)) / denominator
        intercept = y_mean - slope * x_mean
        residuals = [value - (intercept + slope * index)
                     for index, value in enumerate(items)]
        centre = median(residuals)
        robust_scale = 1.4826 * median(
            abs(value - centre) for value in residuals)
        return slope, robust_scale

    def replay(candidate: str) -> dict[str, object] | None:
        candidate_errors: list[float] = []
        baseline_errors: list[float] = []
        for origin in range(6, len(values)):
            train = train_at(origin)
            if len(train) < 2:
                continue
            try:
                estimate = _predict_statistical(candidate, train, 1, season)[0]
            except (ValueError, ArithmeticError):
                continue
            actual = values[origin]
            candidate_errors.append(abs(actual - estimate))
            baseline_errors.append(abs(actual - train[-1]))
        if len(candidate_errors) < 6:
            return None
        boundary = len(candidate_errors) // 2
        blocks = []
        for left, right in ((0, boundary),
                            (boundary, len(candidate_errors))):
            baseline_loss = mean(baseline_errors[left:right])
            candidate_loss = mean(candidate_errors[left:right])
            blocks.append({
                "relative_gain": ((baseline_loss - candidate_loss)
                                  / max(baseline_loss, 1e-12)),
                "win_rate": sum(
                    contender < baseline for contender, baseline in zip(
                        candidate_errors[left:right],
                        baseline_errors[left:right])) / (right - left),
            })
        baseline_loss = mean(baseline_errors)
        candidate_loss = mean(candidate_errors)
        return {
            "origins": len(candidate_errors),
            "relative_gain": ((baseline_loss - candidate_loss)
                              / max(baseline_loss, 1e-12)),
            "chronological_blocks": blocks,
        }

    full_slope, trend_scale = line(values)
    boundary = len(values) // 2
    first_slope, _ = line(values[:boundary])
    second_slope, _ = line(values[boundary:])
    trend_replay = replay("linear_trend")
    slopes = (full_slope, first_slope, second_slope)
    magnitudes = [abs(value) for value in slopes]
    safe_trend_scale = max(trend_scale, 1e-9)
    trend_admitted = bool(
        trend_replay is not None
        and full_slope * first_slope > 0
        and full_slope * second_slope > 0
        and min(magnitudes) > 1e-9
        and max(magnitudes) / min(magnitudes) <= 2.5
        and abs(full_slope) * horizon >= 2.0 * safe_trend_scale
        and float(trend_replay["relative_gain"]) >= .35
        and all(float(block["relative_gain"]) >= .15
                and float(block["win_rate"]) >= .60
                for block in trend_replay["chronological_blocks"])
    )
    if trend_admitted:
        return {
            "scheme": "stable_prefix_structure",
            "candidate": "linear_trend",
            "admitted": True,
            "proxy_horizon": 1,
            "full_horizon_fold_claimed": False,
            "slope": full_slope,
            "first_half_slope": first_slope,
            "second_half_slope": second_slope,
            "projected_change_to_robust_noise": (
                abs(full_slope) * horizon / trend_scale
                if trend_scale > 1e-12 else None),
            "zero_residual_scale": trend_scale <= 1e-12,
            "thresholds": {
                "maximum_slope_ratio": 2.5,
                "minimum_projected_change_to_noise": 2.0,
                "minimum_replay_gain": .35,
                "minimum_block_gain": .15,
                "minimum_block_win_rate": .60,
            },
            "replay": trend_replay,
        }

    if season != 1:
        return {
            "scheme": "stable_prefix_structure",
            "candidate": None,
            "admitted": False,
            "proxy_horizon": 1,
            "full_horizon_fold_claimed": False,
            "reason": "level_screen_requires_nonseasonal_series",
            "slope": full_slope,
            "first_half_slope": first_slope,
            "second_half_slope": second_slope,
            "replay": trend_replay,
        }

    level_replay = replay("historical_mean")
    level_centre = median(values)
    level_scale = max(1.4826 * median(
        abs(value - level_centre) for value in values), 1e-9)
    first_mean = mean(values[:boundary])
    second_mean = mean(values[boundary:])
    recent_mean = mean(values[-4:])
    earlier_mean = mean(values[:-4])
    level_admitted = bool(
        level_replay is not None
        and abs(full_slope) * (len(values) - 1) <= .90 * level_scale
        and abs(first_mean - second_mean) <= .90 * level_scale
        and abs(recent_mean - earlier_mean) <= 1.20 * level_scale
        and float(level_replay["relative_gain"]) >= .10
        and all(float(block["relative_gain"]) >= 0
                and float(block["win_rate"]) >= .50
                for block in level_replay["chronological_blocks"])
    )
    return {
        "scheme": "stable_prefix_structure",
        "candidate": "historical_mean" if level_admitted else None,
        "admitted": level_admitted,
        "proxy_horizon": 1,
        "full_horizon_fold_claimed": False,
        "slope": full_slope,
        "projected_drift_to_level_scale": (
            abs(full_slope) * (len(values) - 1) / level_scale),
        "half_level_shift_to_scale": (
            abs(first_mean - second_mean) / level_scale),
        "recent_level_shift_to_scale": (
            abs(recent_mean - earlier_mean) / level_scale),
        "thresholds": {
            "maximum_projected_drift_to_scale": .90,
            "maximum_half_level_shift_to_scale": .90,
            "maximum_recent_level_shift_to_scale": 1.20,
            "minimum_replay_gain": .10,
            "minimum_block_gain": 0.0,
            "minimum_block_win_rate": .50,
        },
        "replay": level_replay,
    }


def select_model_lightweight(
    values: list[float], horizon: int, season: int,
    train_at: Callable[[int], list[float]] | None = None,
) -> Evaluation:
    """Select on one trailing holdout when separated rolling folds do not fit."""
    if train_at is None:
        train_at = lambda origin: values[:origin]  # noqa: E731
    if len(values) < horizon + 2:
        scores = {name: None for name in MODELS}
        message = f"Need at least {horizon + 2} observations (have {len(values)}) for degraded forecasting."
        reachable = len(values) - 2 if len(values) >= 3 else None
        if reachable is not None:
            message += (
                f" A horizon of {reachable} or less is supportable with the "
                f"current history; retry with --horizon {reachable}."
            )
        return Evaluation(None, None, scores, scores.copy(), None, [], None,
                          [message], False, True,
                          max_supportable_horizon=reachable)
    holdout = min(horizon, max(1, len(values) // 4))
    origin = len(values) - holdout
    scores: dict[str, float | None] = {name: None for name in MODELS}
    forecasts: dict[str, list[float]] = {}
    actual = values[origin:]
    train = train_at(origin)
    for name in MODELS:
        try:
            prediction = _predict_statistical(name, train, holdout, season)
            scores[name] = error_score(actual, prediction)
            forecasts[name] = prediction
        except (ValueError, ArithmeticError):
            continue
    valid = {name: score for name, score in scores.items() if score is not None}
    if not valid:
        return Evaluation(None, None, scores, scores.copy(), None, [], None,
                          ["Series is too short for lightweight model selection."], False, True)
    baselines = {name: score for name, score in valid.items() if name in BASELINES}
    # A single trailing holdout is even weaker evidence than a single
    # rolling fold, so the same guardrail applies: candidates cannot be
    # ranked on it, and the strongest baseline is published with every
    # score reported as evidence (see the guardrail in `evaluate`).
    non_baselines = [name for name in valid if name not in BASELINES]
    degraded_baseline_evidence = None
    degraded_structural_evidence = None
    if "last_value" in baselines:
        # One holdout cannot ordinarily establish that a structured baseline
        # generalises any more reliably than it can rank an incremental model.
        # Publish the assumption-minimal level baseline unless a predeclared,
        # high-specificity recurrence screen or repeatable seasonal blocks
        # establish the structured baseline. This rule is history-length and
        # shape based, never channel or benchmark-label based.
        selected = "last_value"
        if season > 1 and "seasonal_naive" in baselines:
            degraded_baseline_evidence = _two_cycle_seasonal_recurrence(
                values, horizon, season, train_at)
            if (degraded_baseline_evidence is not None
                    and degraded_baseline_evidence["admitted"] is True):
                selected = "seasonal_naive"
        # A long requested horizon can prevent separated full-horizon folds
        # even when the history contains repeatable seasonal evidence. Admit
        # only the single structured baseline against last-value on fixed,
        # non-overlapping seasonal blocks. This is not a candidate tournament:
        # two predeclared baselines, a 10% margin, and wins in two of three
        # chronological blocks are required.
        if (selected == "last_value"
                and season > 1 and "seasonal_naive" in baselines
                and len(values) >= 3 * season):
            probe_horizon = min(season, horizon)
            first = max(season, len(values) - 8 * probe_horizon)
            probe_origins = list(range(
                first, len(values) - probe_horizon + 1, probe_horizon))
            seasonal_probe: list[float] = []
            last_probe: list[float] = []
            for probe_origin in probe_origins:
                probe_train = train_at(probe_origin)
                probe_actual = values[
                    probe_origin:probe_origin + probe_horizon]
                try:
                    seasonal_path = _predict_statistical(
                        "seasonal_naive", probe_train, probe_horizon, season)
                    last_path = _predict_statistical(
                        "last_value", probe_train, probe_horizon, season)
                except ValueError:
                    continue
                seasonal_error = error_score(probe_actual, seasonal_path)
                last_error = error_score(probe_actual, last_path)
                if seasonal_error is None or last_error is None:
                    continue
                seasonal_probe.append(seasonal_error)
                last_probe.append(last_error)
            block_wins = 0
            # Six independent seasonal blocks are the minimum evidence for
            # the three chronological stability checks below. Short snippets
            # must not turn an accidental phase match into publication.
            if len(seasonal_probe) >= 6:
                boundaries = [0, len(seasonal_probe) // 3,
                              2 * len(seasonal_probe) // 3,
                              len(seasonal_probe)]
                for left, right in zip(boundaries, boundaries[1:]):
                    if right > left and mean(
                            seasonal_probe[left:right]) < mean(
                                last_probe[left:right]):
                        block_wins += 1
                seasonal_loss = mean(seasonal_probe)
                last_loss = mean(last_probe)
                admitted = (last_loss > 0
                            and seasonal_loss <= .9 * last_loss
                            and block_wins >= 2)
                degraded_baseline_evidence = {
                    "scheme": "non_overlapping_seasonal_blocks",
                    "probe_horizon": probe_horizon,
                    "origins": len(seasonal_probe),
                    "seasonal_naive_loss": seasonal_loss,
                    "last_value_loss": last_loss,
                    "relative_improvement": (
                        (last_loss - seasonal_loss) / last_loss
                        if last_loss > 0 else None),
                    "required_margin": .1,
                    "chronological_block_wins": block_wins,
                    "required_block_wins": 2,
                    "admitted": admitted,
                }
                if admitted:
                    selected = "seasonal_naive"
        if selected == "last_value" and season == 1:
            degraded_structural_evidence = (
                _fold_starved_structural_evidence(
                    values, horizon, season, train_at))
            if (degraded_structural_evidence is not None
                    and degraded_structural_evidence["admitted"] is True):
                selected = str(degraded_structural_evidence["candidate"])
    elif baselines:
        selected = min(baselines, key=baselines.get)  # type: ignore[arg-type]
    else:
        selected = min(valid, key=valid.get)  # type: ignore[arg-type]
    strongest = (selected if selected in baselines else "last_value") \
        if baselines else min(valid, key=valid.get)  # type: ignore[arg-type]
    guardrail_applied = bool(baselines) and bool(non_baselines)
    warnings = [
        f"Degraded forecast: model selection used a single trailing {holdout}-observation holdout; rolling-origin calibration and final testing were unavailable. At least {max(2 * season, 2 * horizon, 8) + 2 * horizon} observations (have {len(values)}) are needed for separated selection and calibration."
    ]
    if guardrail_applied:
        if (degraded_structural_evidence is not None
                and degraded_structural_evidence["admitted"] is True):
            warnings.append(
                f"Selection under-powered: a single trailing holdout cannot "
                f"rank the candidate pool. The narrow prefix-stability "
                f"screen admitted {selected} against {strongest}; all other "
                f"candidate scores remain evidence, not a ranking."
            )
        else:
            warnings.append(
                f"Selection under-powered: a single trailing holdout cannot "
                f"rank incremental candidates. The assumption-minimal "
                f"baseline ({strongest}) is published; candidate scores are "
                f"reported as evidence, not a ranking."
            )
    if degraded_baseline_evidence is not None:
        warnings.append(
            "Degraded baseline admission: "
            + json.dumps(degraded_baseline_evidence, sort_keys=True))
    if degraded_structural_evidence is not None:
        warnings.append(
            "Degraded structural admission: the prefix-only stability "
            f"screen {'admitted ' + str(selected) if degraded_structural_evidence['admitted'] else 'did not admit a departure'}; "
            "its replay and stability diagnostics are retained in "
            "sensitivity.fold_starved_structural."
        )
    residuals = [a - p for a, p in zip(actual, forecasts[selected])]
    return Evaluation(selected, strongest, scores, {name: None for name in MODELS}, None,
                      residuals, None, warnings, True, True,
                      selection_fold_count=(
                          int(degraded_baseline_evidence["origins"])
                          if degraded_baseline_evidence and
                          degraded_baseline_evidence["admitted"] else 1),
                      selection_guardrail_applied=guardrail_applied,
                      selection_stability={
                          "fold_starved_structural":
                              degraded_structural_evidence})


def _admit_pretrained_lightweight(
    assessment: Evaluation, values: list[float], horizon: int, season: int,
    frequency: str, tsfm_names: list[str] | None,
    external_priors: dict[str, ExternalModelPrior] | None,
    evidence_registry: Any,
) -> Evaluation:
    """Apply transfer evidence when separated folds do not fit at all.

    The trailing holdout remains one local observation, never a tournament.
    External evidence can supply the missing prior; the result stays degraded
    and prior-assisted support is capped later by the runtime.
    """
    if not assessment.supported or not assessment.strongest_baseline:
        return assessment
    from .tsfm import eligible_tsfms, pinned_revision
    from .tsfm_sandbox import select_tsfm_candidates
    eligible, _ = eligible_tsfms(
        history_length=len(values), horizon=horizon, frequency=frequency)
    requested = tsfm_names if tsfm_names is not None else eligible
    requested = [name for name in requested if name in eligible]
    adapters = select_tsfm_candidates(requested=requested, frequency=frequency,
                                      in_process=tsfm_candidates)
    if not adapters:
        return assessment
    priors = dict(external_priors or {})
    if evidence_registry is not None:
        from .model_evidence import describe_regime
        regime = describe_regime(values, horizon, season, frequency)
        for adapter in adapters:
            model_id = getattr(adapter, "_MODEL_ID", None)
            raw_revision = getattr(adapter, "revision", None)
            remote_model = getattr(getattr(adapter, "_provider", None), "model", "")
            try:
                revision = (f"{model_id}@{pinned_revision(model_id)}" if model_id
                            else f"{remote_model or adapter.name}@{raw_revision}"
                            if raw_revision else None)
            except Exception:
                revision = None
            if revision:
                prior = evidence_registry.lookup(adapter.name, revision, regime)
                if prior is not None:
                    priors[adapter.name] = prior
    usable = [adapter for adapter in adapters
              if adapter.name in priors and priors[adapter.name].usable()]
    if not usable:
        return assessment
    adapter = max(usable, key=lambda item: priors[item.name].mean_relative_gain)
    baseline = assessment.strongest_baseline
    holdout = min(horizon, max(1, len(values) // 4))
    origin = len(values) - holdout
    train = values[:origin]
    actual = values[origin:]
    try:
        candidate_holdout = _predict_adapter(adapter, train, holdout, season)
        baseline_holdout = _predict_statistical(baseline, train, holdout, season)
        candidate_path = _predict_adapter(adapter, values, horizon, season)
        baseline_path = _predict_statistical(baseline, values, horizon, season)
    except Exception:
        return assessment
    candidate_loss = error_score(actual, candidate_holdout)
    baseline_loss = error_score(actual, baseline_holdout)
    diagnostics = output_diagnostics(values, candidate_path, baseline_path)
    evidence = local_evidence(
        model_class="pretrained", candidate_losses=[candidate_loss],
        baseline_losses=[baseline_loss], external_prior=priors[adapter.name],
        diagnostics=diagnostics)
    decision = decide_admission(
        candidate=adapter.name, baseline=baseline, evidence=evidence)
    if decision.point_policy == "baseline":
        return replace(assessment, admission_decision=decision)

    from .candidate import (
        CandidateIdentity, CandidateSpec, FittedCandidate,
        blended_candidate_spec,
    )
    from .ids import content_id
    from .versioning import RUNTIME_VERSION

    def statistical_spec(name: str) -> CandidateSpec:
        identity = CandidateIdentity(
            kind="builtin", name=name, revisions={"runtime": RUNTIME_VERSION},
            fallback_policy="none")
        def fit(history: list[float], _season: int | None) -> FittedCandidate:
            fitted = identity.with_fit(
                weights=None,
                data_fingerprint=content_id("history", {"values": history}))
            return FittedCandidate(
                fitted, lambda steps: _predict_statistical(
                    name, history, steps, season))
        return CandidateSpec(identity, fit)

    model_id = getattr(adapter, "_MODEL_ID", None)
    raw_revision = getattr(adapter, "revision", None)
    revision = (f"{model_id}@{pinned_revision(model_id)}" if model_id
                else str(raw_revision or "unversioned"))
    candidate_identity = CandidateIdentity(
        kind="tsfm", name=adapter.name,
        config={"adapter_protocol": "0.1"},
        revisions={"runtime": RUNTIME_VERSION, adapter.name: revision},
        fallback_policy="strongest_baseline_recalibrated")
    capabilities = LegacyModelAdapter(adapter).capabilities
    def fit_candidate(history: list[float], _season: int | None) -> FittedCandidate:
        fitted = candidate_identity.with_fit(
            weights=None,
            data_fingerprint=content_id("history", {"values": history}))
        return FittedCandidate(
            fitted, lambda steps: _predict_adapter(
                adapter, history, steps, season))
    candidate_spec = CandidateSpec(
        candidate_identity, fit_candidate,
        min_history=capabilities.min_history,
        max_horizon=capabilities.max_horizon)
    baseline_spec = statistical_spec(baseline)
    if decision.point_policy == "candidate":
        selected = adapter.name
        final = candidate_spec
        residual_prediction = candidate_holdout
    else:
        selected = "admission_blend"
        final = blended_candidate_spec(
            candidate_spec, baseline_spec, decision.candidate_weight,
            policy_version=decision.policy_version,
            admission_state=decision.state, name="admission_blend")
        residual_prediction = [
            decision.candidate_weight * candidate
            + (1 - decision.candidate_weight) * base
            for candidate, base in zip(candidate_holdout, baseline_holdout)]
    residuals = [observed - predicted
                 for observed, predicted in zip(actual, residual_prediction)]
    scores = dict(assessment.selection_scores)
    scores[adapter.name] = candidate_loss
    return replace(
        assessment, selected_model=selected, selection_scores=scores,
        improvement=((baseline_loss - candidate_loss) / baseline_loss
                     if baseline_loss and candidate_loss is not None else None),
        residuals=residuals,
        warnings=[*assessment.warnings,
                  "Pretrained transfer admission used one trailing holdout; "
                  "external evidence is reported separately and rolling "
                  "calibration remains unavailable."],
        notes=[*assessment.notes,
               f"TSFM admission: {adapter.name} is {decision.state}; "
               f"candidate weight {decision.candidate_weight:.3f}."],
        tsfm_scores={adapter.name: candidate_loss},
        final_candidate=final, admission_decision=decision,
    )


def _admit_pooled_lightweight(
    assessment: Evaluation, values: list[float], horizon: int, season: int,
    minimum_improvement: float,
    extra_candidates: dict[str, Callable[[int, int], list[float]]],
) -> Evaluation:
    """Admit a within-panel executable without calling donor evidence local folds.

    The candidate itself owns a leave-one-channel-out comparability check.
    Gnomon additionally requires repeated disjoint target-origin wins under
    both WAPE and fold-local scaled error.  This lane is distinct from external transfer:
    every borrowed observation belongs to the caller's current snapshot.
    """
    if not assessment.supported or assessment.strongest_baseline != "last_value":
        return assessment
    eligible = []
    for name, candidate in extra_candidates.items():
        evidence_fn = getattr(candidate, "lightweight_evidence", None)
        if evidence_fn is None:
            continue
        evidence = evidence_fn(horizon, season, minimum_improvement)
        if evidence is not None:
            eligible.append((name, candidate, evidence))
    if not eligible:
        return assessment
    # Normally one panel candidate exists per target. If a future family adds
    # another, choose by the preregistered held-out loss and disclose it.
    name, candidate, pooled = min(eligible, key=lambda item: item[2].target_loss)
    from .admission import AdmissionDecision, AdmissionEvidence, OutputDiagnostics
    admission_evidence = AdmissionEvidence(
        model_class="locally_fitted",
        independent_folds=pooled.target_pairs,
        paired_folds=pooled.target_pairs,
        candidate_loss=pooled.target_loss,
        baseline_loss=pooled.baseline_loss,
        relative_improvement=(
            (pooled.baseline_loss - pooled.target_loss) / pooled.baseline_loss),
        candidate_win_rate=pooled.target_win_rate,
        median_relative_gain=pooled.target_median_gain,
        local_gain_standard_error=None,
        diagnostics=OutputDiagnostics(),
    )
    decision = AdmissionDecision(
        "pooled_validated", name, "last_value", "candidate", 1.0,
        admission_evidence,
        (f"borrowed strength from {len(candidate.donors)} sibling channels",
         f"leave-one-channel-out donor comparisons: {pooled.donor_pairs}",
         f"donor win rate: {pooled.donor_win_rate:.3f}",
         f"disjoint target-origin comparisons: {pooled.target_pairs}",
         f"target win rate: {pooled.target_win_rate:.3f}",
         f"normalised pooled-trend strength: {pooled.normalised_pool_strength:.3f}",
         "target held-out WAPE and scaled-error gates both passed"),
        policy_version="within-panel-pooling-v2",
    )
    from .candidate import CandidateIdentity, CandidateSpec, FittedCandidate
    from .ids import content_id
    from .versioning import RUNTIME_VERSION
    identity = CandidateIdentity(
        kind="cross_series", name=name, members=tuple(candidate.donors),
        strategy="half_shrunk_median_normalised_recent_trend",
        config={
            "admission_state": "pooled_validated",
            "donor_pairs": pooled.donor_pairs,
            "target_pairs": pooled.target_pairs,
            "comparability": "leave_one_channel_out_fold_transfer",
        },
        revisions={"runtime": RUNTIME_VERSION},
        fallback_policy="last_value_recalibrated",
    )

    def fit(history: list[float], _season: int | None) -> FittedCandidate:
        fitted = identity.with_fit(
            weights=None,
            data_fingerprint=content_id("history", {"values": history}),
        )
        return FittedCandidate(
            fitted, lambda steps: candidate(len(history), steps))

    final = CandidateSpec(identity, fit, min_history=4)
    origin = pooled.target_origin
    holdout = len(values) - origin
    prediction = candidate(origin, holdout)
    residuals = [actual - predicted
                 for actual, predicted in zip(values[origin:], prediction)]
    scores = dict(assessment.selection_scores)
    scores[name] = pooled.target_loss
    return replace(
        assessment, selected_model=name, selection_scores=scores,
        improvement=admission_evidence.relative_improvement,
        residuals=residuals,
        warnings=[*assessment.warnings,
                  "Short-history pooled forecast: the target passed repeated "
                  "disjoint historical origins; sibling-channel transfer was validated with "
                  "leave-one-channel-out historical forecasts. The result "
                  "borrows strength and remains degraded."],
        notes=[*assessment.notes,
               f"Panel pooling admitted from {len(candidate.donors)} donors; "
               f"{pooled.donor_pairs} LOCO comparisons, donor median gain "
               f"{pooled.donor_median_gain:.3f}."],
        final_candidate=final, admission_decision=decision,
        selection_guardrail_applied=False,
        selection_stability={
            "paired_folds": pooled.target_pairs,
            "candidate_win_rate": pooled.target_win_rate,
            "median_relative_gain": pooled.target_median_gain,
            "scaled_error_improvement": pooled.target_scaled_gain,
            "scaled_error_passed": True,
            "passed": True,
            "borrowed_strength": True,
            "donor_loco_pairs": pooled.donor_pairs,
            "donor_win_rate": pooled.donor_win_rate,
            "normalised_pool_strength": pooled.normalised_pool_strength,
        },
    )


#: Longest fit history a single fold's model fit sees, stretched to at
#: least four seasonal periods. Fold structure is never windowed — only
#: the history handed to each fit. Mirrors the anomaly grading window
#: (MAX_GRADING_HISTORY): bound the O(n)-per-fit work that long series
#: multiply by fold count, disclose the bound whenever it engages.
MAX_FIT_HISTORY = 2048


def evaluate(
    values: list[float],
    horizon: int,
    season: int,
    minimum_improvement: float,
    *,
    frequency: str = "h",
    tsfm_names: list[str] | None = None,
    config: Any = None,
    strict_abstention: bool = False,
    train_at: Callable[[int], list[float]] | None = None,
    extra_candidates: dict[str, Callable[[int, int], list[float]]] | None = None,
    selection_stride: int | None = None,
    selection_loss: str = "wape",
    external_priors: dict[str, ExternalModelPrior] | None = None,
    evidence_registry: Any = None,
    threshold_job: bool = False,
) -> Evaluation:
    """``train_at(origin)`` returns the training history for a fold whose
    forecast origin is index ``origin`` — by default a plain prefix slice,
    but a snapshot-backed provider returns the series *as known at* the
    fold cutoff, which is what makes backtests vintage-honest.

    ``extra_candidates`` maps a name to ``predictor(origin, horizon)`` for
    candidates that need more than this series' own history — a VAR fit
    across aligned series, say. They are scored on the same folds, against
    the same baselines, under the same improvement margin as everything
    else; a candidate that cannot beat the ladder does not get in by being
    special.

    ``selection_stride`` samples selection origins more finely than the
    horizon (``None`` keeps them non-overlapping, one per horizon). It widens
    the comparison sample only: calibration residuals are always pooled from
    the non-overlapping skeleton, and no selection fold reads a point
    belonging to the calibration or test partitions.

    ``selection_loss`` chooses the criterion: ``"wape"`` (the default, a
    point loss) or ``"pinball"``, the proper scoring rule for a quantile.
    Pinball is the right criterion when the answer is a distribution — a
    point loss cannot tell a model with well-placed uncertainty from one
    whose centre happens to land well — but it changes which model is
    selected, so it is opt-in until measured.

    ``minimum_improvement`` is the margin a candidate must beat the
    strongest baseline by. It must not be negative: at ``-5.0`` the gate
    became ``candidate <= baseline * 6``, which selects a model that *lost*
    the backtest and reports it as supported. One caller-supplied number
    turning off the mandated-baseline rule is precisely the thing the rule
    exists to prevent, so a negative value is refused here rather than
    honoured."""
    if minimum_improvement < 0:
        raise GnomonError(
            "INVALID_MINIMUM_IMPROVEMENT",
            f"minimum_baseline_improvement must be >= 0; got "
            f"{minimum_improvement}. A negative margin inverts the "
            f"mandated-baseline gate, letting a model that lost the "
            f"backtest be selected and reported as supported.",
            {"supplied": minimum_improvement, "minimum": 0.0},
        )
    if train_at is None:
        train_at = lambda origin: values[:origin]  # noqa: E731
    minimum_train = max(2 * season, 2 * horizon, 8)
    origins = _origins(len(values), horizon, minimum_train)
    empty_scores = {name: None for name in MODELS}
    if len(origins) < 2:
        lightweight = select_model_lightweight(
            values, horizon, season, train_at)
        if extra_candidates:
            pooled = _admit_pooled_lightweight(
                lightweight, values, horizon, season, minimum_improvement,
                extra_candidates)
            if pooled.admission_decision is not None:
                return pooled
        if external_priors or evidence_registry is not None:
            admitted = _admit_pretrained_lightweight(
                lightweight, values, horizon, season, frequency, tsfm_names,
                external_priors, evidence_registry)
            if admitted.admission_decision is not None:
                return admitted
        if not strict_abstention:
            return lightweight
        minimum_required = minimum_train + 2 * horizon
        full_required = minimum_train + 4 * horizon
        message = (
            f"Need at least {minimum_required} observations (have {len(values)}) "
            f"for separated selection and calibration windows; "
            f"{full_required} observations enable fully separated selection, "
            f"calibration, and test windows."
        )
        reachable = supportable_horizon(len(values), season)
        if reachable is not None:
            message += (
                f" A horizon of {reachable} or less is evaluable with the "
                f"current history; retry with --horizon {reachable}."
            )
        return Evaluation(
            None, None, empty_scores, empty_scores.copy(), None, [], None,
            [message],
            False,
            max_supportable_horizon=reachable,
        )

    # Full ordinary mode holds out independent confirmation, calibration, and
    # final-test folds after selection. With fewer than five origins there is
    # no honest confirmation fold to reserve, so degraded behavior is
    # preserved rather than manufacturing independence through overlap.
    degraded = len(origins) < 4
    warnings: list[str] = []
    # Fit-history window: past MAX_FIT_HISTORY (stretched to four seasonal
    # periods), each fold's fit sees the trailing window of its history
    # rather than all of it. Refitting exponential smoothing over 64 years
    # of daily points per fold priced the evaluation out of interactive
    # budgets (the 2026-08 MCP evaluation's 23,594-row forecast ran 235 s)
    # while adding nothing the smoothing had not already forgotten. Fold
    # boundaries and cutoffs are untouched; every candidate and baseline
    # sees the identical window, so the competition stays fair — and the
    # window is disclosed as a warning, same contract as the anomaly
    # grading window.
    fit_window = max(MAX_FIT_HISTORY, 4 * season)
    if len(values) > fit_window:
        unwindowed_train_at = train_at

        def train_at(origin: int, _base=unwindowed_train_at,
                     _window=fit_window) -> list[float]:
            train = _base(origin)
            return train[-_window:] if len(train) > _window else train

        warnings.append(
            f"fit_window: model fits used the trailing {fit_window} "
            f"observations of each fold's history (series has "
            f"{len(values)}); fold boundaries are unchanged and every "
            f"candidate and baseline saw the identical window."
        )
    event_calibration_origins: list[int] = []
    confirmation_origin: int | None = None
    if threshold_job and len(origins) >= 12:
        # A governed tail decision needs more than fold-safe timestamps: its
        # residual trajectories must not be the same folds that selected the
        # winning model.  Reserve eight disjoint origins plus the final
        # report-only test origin.  This partition applies only to threshold
        # jobs, leaving ordinary forecasts byte-identical.
        selection_origins = origins[:-9]
        event_calibration_origins = origins[-9:-1]
        calibration_origin = event_calibration_origins[-1]
        test_origin = origins[-1]
    elif len(origins) >= 5:
        selection_origins = origins[:-3]
        confirmation_origin = origins[-3]
        calibration_origin = origins[-2]
        test_origin = origins[-1]
        event_calibration_origins = [calibration_origin]
    elif len(origins) >= 3:
        selection_origins, calibration_origin = origins[:-2], origins[-2]
        test_origin = origins[-1]
        event_calibration_origins = [calibration_origin]
    else:
        selection_origins, calibration_origin, test_origin = origins[:-1], origins[-1], None
        event_calibration_origins = [calibration_origin]
    # Residuals stay on the disjoint skeleton whatever selection does with
    # its stride: a conformal quantile over dependent residuals is not a
    # conformal quantile.
    residual_origins = list(selection_origins)
    # `evaluation.pool_residuals` (default true): pool the selection folds
    # with the calibration fold for sample size, accepting a known
    # optimistic bias. False is genuine split conformal — the calibration
    # fold only, whose origins selection never saw — and noisier. The key
    # was documented and never read; it is read here.
    # The built-in candidates this run competes, after `models.statistical.*`.
    # The mandatory baselines are always in it.
    pool = active_models(config)
    pool_residuals = True
    evaluation_config = getattr(config, "evaluation", None) if config else None
    if evaluation_config is not None:
        pool_residuals = bool(getattr(evaluation_config, "pool_residuals", True))
        # A caller-supplied abstention floor, above Gnomon's derived one.
        floor = getattr(evaluation_config, "min_observations", None)
        if floor is not None and len(values) < int(floor):
            raise GnomonError(
                "INSUFFICIENT_OBSERVATIONS",
                f"{len(values)} observations is below the configured "
                f"evaluation.folds.min_observations of {int(floor)}.",
                {"observations": len(values), "min_observations": int(floor)},
            )
    if selection_stride is not None and selection_origins:
        selection_origins = dense_selection_origins(
            minimum_train, selection_origins[-1], selection_stride,
        )
    if degraded:
        warnings.append(
            f"Limited evaluation: only {len(origins)} rolling folds were available; "
            f"{minimum_train + 4 * horizon} observations enable fully separated "
            f"selection, calibration, and test windows."
        )

    # --- Resolve config ---
    ensemble_enabled = getattr(config, "ensemble", None) and config.ensemble.enabled if config else False
    ensemble_strategy = config.ensemble.strategy if config and ensemble_enabled else "weighted_mean"
    ensemble_cfg = config.ensemble if config else None
    meta_model_enabled = getattr(config, "meta_model", None) and config.meta_model.enabled if config else False
    meta_model_cfg = config.meta_model if config and meta_model_enabled else None

    # --- Load TSFM candidates (lazy, graceful) ---
    # Prefer sandboxed adapters (isolated venvs) to avoid dependency conflicts.
    # Fall back to in-process adapters if no sandboxes are set up.
    from .tsfm import eligible_tsfms
    from .tsfm_sandbox import select_tsfm_candidates
    tsfm_adapters: list[Any] = []
    eligible_names, capability_exclusions = eligible_tsfms(
        history_length=len(values), horizon=horizon, frequency=frequency,
    )
    requested_names = tsfm_names if tsfm_names is not None else eligible_names
    requested_names = [name for name in requested_names if name in eligible_names]
    # Informational, never a warning. A capability exclusion says a foundation
    # model could not run on this history length or frequency — it says nothing
    # about the evidence behind the forecast that did run. Routed through
    # `warnings` it would downgrade support to "weakly_supported" (pipeline's
    # support enum), so a fully evidenced forecast would advertise doubt about
    # itself because a model the caller never asked for was ineligible.
    capability_notes = [
        f"Skipped TSFM {name}: {'; '.join(reasons)}."
        for name, reasons in capability_exclusions.items()
        if tsfm_names is None or name in tsfm_names
    ]
    if requested_names:
        tsfm_adapters = select_tsfm_candidates(
            requested=requested_names, frequency=frequency,
            in_process=tsfm_candidates)
    tsfm_model_names = [a.name for a in tsfm_adapters]
    all_model_names = list(pool.keys()) + tsfm_model_names

    # --- API inference adapters ---
    if config and config.backends.api.enabled:
        from .api_inference import APIAdapter
        for name, provider_cfg in config.backends.api.providers.items():
            if name in (tsfm_names or config.models.tsfm_candidates or []):
                try:
                    api_adapter = APIAdapter(name, provider_cfg)
                    tsfm_adapters.append(api_adapter)
                    if name not in all_model_names:
                        all_model_names.append(name)
                except Exception:
                    logger.debug("API adapter %s failed to initialize", name, exc_info=True)

    # --- Optional statistical-library adapters ---------------------------
    # Availability adds candidates, never authority. The package is imported
    # lazily and an explicit request with a missing/broken dependency becomes
    # a note plus a discovery receipt; mandatory baselines continue normally.
    statsforecast_adapters: list[Any] = []
    statsforecast_receipt: dict[str, Any] = {"status": "disabled"}
    statsforecast_enabled = bool(
        config and getattr(config.models, "statsforecast_enabled", False))
    if statsforecast_enabled:
        from .statsforecast_adapter import (
            DEFAULT_CANDIDATES, MAX_OUTER_FOLDS, statsforecast_candidates,
        )
        requested_stats = getattr(
            config.models, "statsforecast_candidates", None)
        try:
            if len(selection_origins) > MAX_OUTER_FOLDS:
                statsforecast_receipt = {
                    "status": "soft_skip_compute_budget",
                    "selection_folds": len(selection_origins),
                    "maximum_selection_folds": MAX_OUTER_FOLDS,
                    "requested": list(requested_stats or DEFAULT_CANDIDATES),
                }
            else:
                statsforecast_adapters, statsforecast_receipt = \
                    statsforecast_candidates(requested_stats)
        except ValueError as exc:
            raise GnomonError(
                "UNKNOWN_MODEL", str(exc),
                {"available": list(DEFAULT_CANDIDATES)},
            ) from exc
    all_adapters = [*tsfm_adapters, *statsforecast_adapters]
    for adapter in statsforecast_adapters:
        if adapter.name not in all_model_names:
            all_model_names.append(adapter.name)

    if evidence_registry is not None and external_priors is None:
        from .model_evidence import describe_regime
        from .tsfm import pinned_revision
        regime = describe_regime(values, horizon, season, frequency)
        external_priors = {}
        for adapter in tsfm_adapters:
            model_id = getattr(adapter, "_MODEL_ID", None)
            if model_id:
                try:
                    revision = f"{model_id}@{pinned_revision(model_id)}"
                except Exception:
                    continue
            else:
                raw_revision = getattr(adapter, "revision", None)
                remote_model = getattr(
                    getattr(adapter, "_provider", None), "model", "")
                if not raw_revision:
                    # Unversioned remote behavior cannot consume a prior for
                    # a reproducible revision.
                    continue
                revision = f"{remote_model or adapter.name}@{raw_revision}"
            prior = evidence_registry.lookup(adapter.name, revision, regime)
            if prior is not None:
                external_priors[adapter.name] = prior

    # Disclose the model tier that could not compete. A fresh install has no
    # TSFM sandboxes, so without this note the operator most likely to benefit
    # from a stronger candidate never learns one was eligible.
    from .tsfm import installed_tsfms
    from .tsfm_sandbox import sandbox_available_tsfms
    installed_names = set(sandbox_available_tsfms()) | set(installed_tsfms())
    notes: list[str] = list(capability_notes)
    if requested_names and not installed_names:
        notes.append(
            f"No foundation-model candidate competed: "
            f"{', '.join(requested_names)} "
            f"{'is' if len(requested_names) == 1 else 'are'} eligible for this "
            f"series but no sandbox is installed. Run "
            f"`gnomon tsfm install {requested_names[0]}` to add one; it enters "
            f"the same folds against the same baselines."
        )
    if statsforecast_enabled and not statsforecast_adapters:
        status = statsforecast_receipt.get("status", "soft_skip")
        required = statsforecast_receipt.get("required", ">=2.1,<3")
        notes.append(
            f"StatsForecast candidates did not compete ({status}); optional "
            f"dependency requirement is {required}. Mandatory baselines and "
            f"built-in candidates remained active."
        )

    # --- Run built-in models on selection folds ---
    # Both lists are indexed by fold, exactly like the TSFM lists below: a
    # fold a model cannot predict or score keeps its slot as a placeholder
    # (None score, empty forecast). The previous compaction (`continue`)
    # shifted every later fold left, so the fold-indexed reads in the
    # ensemble and meta-model scoring silently paired fold k's actuals
    # with fold k+1's forecast for any model that failed a strict subset
    # of folds.
    fold_scores: dict[str, list[float | None]] = {name: [] for name in pool}
    # Store per-fold forecasts for ensemble/meta-model training
    fold_forecasts: dict[str, list[list[float]]] = {name: [] for name in pool}
    for origin in selection_origins:
        actual = values[origin : origin + horizon]
        train = train_at(origin)
        for name in pool:
            try:
                forecast = _predict_statistical(name, train, horizon, season)
            except ValueError:
                fold_scores[name].append(None)
                fold_forecasts[name].append([])
                continue
            score = error_score(actual, forecast)
            if score is None:
                # Unscoreable fold (no scale in the window). The forecast
                # is kept out of the ensemble map too: a fold that cannot
                # be scored cannot contribute weighting evidence either.
                fold_scores[name].append(None)
                fold_forecasts[name].append([])
                continue
            fold_forecasts[name].append(forecast)
            fold_scores[name].append(score)

    # --- Run optional adapter candidates on selection folds ---
    adapter_fold_scores: dict[str, list[float | None]] = {
        a.name: [] for a in all_adapters}
    adapter_fold_forecasts: dict[str, list[list[float]]] = {
        a.name: [] for a in all_adapters}
    adapter_fold_failures: dict[str, list[dict[str, Any]]] = {
        a.name: [] for a in all_adapters}
    for adapter in all_adapters:
        batched: list[list[float]] | None = None
        if hasattr(adapter, "predict_many") and selection_origins:
            try:
                batched = _predict_adapter_many(
                    adapter,
                    [train_at(origin) for origin in selection_origins],
                    horizon, season,
                )
                if len(batched) != len(selection_origins):
                    batched = None
            except Exception as exc:
                logger.debug("adapter %s batch prediction failed: %s",
                             adapter.name, exc)
                batched = None
        for fold_index, origin in enumerate(selection_origins):
            actual = values[origin : origin + horizon]
            train = train_at(origin)
            try:
                if batched is None:
                    forecast = _predict_adapter(adapter, train, horizon, season)
                else:
                    forecast = batched[fold_index]
                if len(forecast) != horizon:
                    adapter_fold_scores[adapter.name].append(None)
                    adapter_fold_forecasts[adapter.name].append([])
                    continue
                adapter_fold_scores[adapter.name].append(
                    error_score(actual, forecast))
                adapter_fold_forecasts[adapter.name].append(forecast)
            except (TSFMError, TSFMUnavailable, Exception) as exc:
                logger.debug("adapter %s failed on fold at origin %d: %s",
                             adapter.name, origin, exc)
                adapter_fold_scores[adapter.name].append(None)
                adapter_fold_forecasts[adapter.name].append([])
                adapter_fold_failures[adapter.name].append({
                    "origin_index": origin,
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                })

    # --- Aggregate scores ---
    # Same rule as the TSFM aggregation below: every fold must have a real
    # score for the model to hold an aggregate at all. A placeholder (None)
    # from a failed or unscoreable fold disqualifies, exactly as the old
    # compacted-and-count-mismatched list did.
    scores: dict[str, float | None] = {}
    for name, items in fold_scores.items():
        valid = [x for x in items if x is not None]
        scores[name] = (
            mean(valid) if valid and len(valid) == len(selection_origins) else None
        )
    adapter_scores: dict[str, float | None] = {}
    for name, items in adapter_fold_scores.items():
        valid = [x for x in items if x is not None]
        if valid and len(valid) == len(selection_origins):
            adapter_scores[name] = mean(valid)
        else:
            adapter_scores[name] = None
            # Lazy in-process adapters can be instantiated before their
            # optional package is importable. Do not call those "installed";
            # the earlier absent-tier note already explains that state.
            if name in installed_names:
                minimum = getattr(
                    next((adapter for adapter in all_adapters
                          if adapter.name == name), None), "min_history", None)
                requirement = (f"; requires at least {minimum} history points"
                               if minimum else "")
                notes.append(
                    f"Installed TSFM {name} did not enter selection: completed "
                    f"{len(valid)} of {len(selection_origins)} required folds"
                    f"{requirement}. Partial-fold scores are not admitted."
                )
            elif any(adapter.name == name
                     for adapter in statsforecast_adapters):
                minimum = next(adapter.min_history for adapter in
                               statsforecast_adapters if adapter.name == name)
                notes.append(
                    f"StatsForecast candidate {name} did not enter selection: "
                    f"completed {len(valid)} of {len(selection_origins)} "
                    f"required folds; minimum history is {minimum}. "
                    f"Partial-fold scores are not admitted."
                )
    tsfm_scores = {name: adapter_scores.get(name)
                   for name in (adapter.name for adapter in tsfm_adapters)}
    statistical_plugin_scores = {
        name: adapter_scores.get(name)
        for name in (adapter.name for adapter in statsforecast_adapters)
    }
    if statsforecast_enabled:
        statsforecast_receipt["candidates"] = {
            adapter.name: {
                "model": adapter.model_class,
                "package_version": adapter.revision,
                "season": season,
                "required_folds": len(selection_origins),
                "completed_folds": sum(
                    item is not None
                    for item in adapter_fold_scores[adapter.name]),
                "score": statistical_plugin_scores.get(adapter.name),
                "failures": adapter_fold_failures[adapter.name],
                **({
                    "components": list(adapter.components),
                    "internal_selection_trace": list(
                        adapter.selection_trace),
                } if hasattr(adapter, "components") else {}),
            }
            for adapter in statsforecast_adapters
        }

    # --- Run cross-series candidates on the same selection folds ---
    extra_candidates = dict(extra_candidates or {})
    extra_fold_scores: dict[str, list[float | None]] = {name: [] for name in extra_candidates}
    extra_fold_forecasts: dict[str, list[list[float]]] = {
        name: [] for name in extra_candidates}
    for name, predictor in extra_candidates.items():
        for origin in selection_origins:
            actual = values[origin : origin + horizon]
            try:
                forecast = predictor(origin, horizon)
            except Exception:
                logger.debug("candidate %s failed on fold at origin %d", name, origin,
                             exc_info=True)
                extra_fold_scores[name].append(None)
                extra_fold_forecasts[name].append([])
                continue
            if len(forecast) != horizon:
                extra_fold_scores[name].append(None)
                extra_fold_forecasts[name].append([])
                continue
            extra_fold_scores[name].append(error_score(actual, forecast))
            extra_fold_forecasts[name].append(forecast)
    extra_scores: dict[str, float | None] = {}
    for name, items in extra_fold_scores.items():
        valid = [item for item in items if item is not None]
        extra_scores[name] = (
            mean(valid) if valid and len(valid) == len(selection_origins) else None
        )
        if name not in all_model_names:
            all_model_names.append(name)

    # --- Compute ensemble forecast on selection folds ---
    ensemble_fold_scores: list[float | None] = []
    ensemble_fold_forecasts: list[list[float]] = []
    if ensemble_enabled or meta_model_enabled:
        from .ensemble import compute_ensemble_forecast

        def _weighting_scores(fold_idx: int) -> dict[str, float | None]:
            """Member scores available *before* fold ``fold_idx``.

            The inverse-error weights used to come from aggregates over
            every selection fold, fold ``fold_idx`` included — so each
            fold's ensemble was weighted using its own outcome, and the
            ensemble's selection score was optimistic by construction.
            Fold 0 has no prior evidence and so weights members equally.
            """
            prior: dict[str, float | None] = {}
            for source in (fold_scores, adapter_fold_scores):
                for name, items in source.items():
                    earlier = [
                        item for item in items[:fold_idx] if item is not None
                    ]
                    prior[name] = mean(earlier) if earlier else None
            return prior

        for fold_idx in range(len(selection_origins)):
            fold_forecast_map: dict[str, list[float]] = {}
            for name in pool:
                if fold_idx < len(fold_forecasts[name]) and fold_forecasts[name][fold_idx]:
                    fold_forecast_map[name] = fold_forecasts[name][fold_idx]
            for adapter in all_adapters:
                if (fold_idx < len(adapter_fold_forecasts[adapter.name])
                        and adapter_fold_forecasts[adapter.name][fold_idx]):
                    fold_forecast_map[adapter.name] = \
                        adapter_fold_forecasts[adapter.name][fold_idx]

            if len(fold_forecast_map) >= (ensemble_cfg.min_models if ensemble_cfg else 2):
                try:
                    combined = compute_ensemble_forecast(
                        fold_forecast_map, _weighting_scores(fold_idx),
                        strategy=ensemble_strategy,
                        last_observed=(train_at(selection_origins[fold_idx]) or [0.0])[-1] if selection_origins else 0.0,
                        config=ensemble_cfg,
                    )
                    actual = values[selection_origins[fold_idx] : selection_origins[fold_idx] + horizon]
                    ensemble_fold_scores.append(error_score(actual, combined))
                    ensemble_fold_forecasts.append(combined)
                except Exception:
                    ensemble_fold_scores.append(None)
                    ensemble_fold_forecasts.append([])
            else:
                ensemble_fold_scores.append(None)
                ensemble_fold_forecasts.append([])

    ensemble_score: float | None = None
    if ensemble_enabled and ensemble_fold_scores:
        valid = [x for x in ensemble_fold_scores if x is not None]
        if valid and len(valid) == len(selection_origins):
            ensemble_score = mean(valid)
            all_model_names.append("ensemble")

    # --- Meta-model training ---
    meta_model_weights: dict[str, float] | None = None
    meta_model_score: float | None = None
    meta_model_fold_forecasts: list[list[float]] = []
    if meta_model_enabled:
        from .meta_model import train_meta_model
        # Collect fold forecasts and actuals for training
        mm_fold_forecasts: dict[str, list[list[float]]] = {}
        mm_fold_actuals: list[list[float]] = []
        for fold_idx in range(len(selection_origins)):
            origin = selection_origins[fold_idx]
            actual = values[origin : origin + horizon]
            mm_fold_actuals.append(actual)

        for name in pool:
            # Same rule as the TSFM branch below: a model qualifies only if
            # it produced at least one real fold forecast — placeholder-only
            # lists (every fold failed) stay out of the training pool.
            if any(fold_forecasts[name]):
                mm_fold_forecasts[name] = fold_forecasts[name]
        for adapter in all_adapters:
            valid_forecasts = [
                f for f in adapter_fold_forecasts[adapter.name] if f]
            if valid_forecasts:
                mm_fold_forecasts[adapter.name] = \
                    adapter_fold_forecasts[adapter.name]

        if len(mm_fold_forecasts) >= (meta_model_cfg.min_models if meta_model_cfg else 2):
            # The weights that will actually be used are fit on every fold,
            # like any final refit.
            meta_model_weights = train_meta_model(
                mm_fold_forecasts,
                mm_fold_actuals,
                non_negative=meta_model_cfg.non_negative if meta_model_cfg else True,
                ridge_alpha=meta_model_cfg.ridge_alpha if meta_model_cfg else 1e-6,
            )
            if meta_model_weights:
                # The *score* is leave-one-fold-out. Fitting on all folds and
                # then scoring on those same folds made the meta-model's
                # selection score in-sample, competing against its members'
                # honest out-of-sample scores — it won by construction, which
                # is what `meta_model.py`'s docstring already promised it did
                # not do.
                from .meta_model import predict_meta_model as pmm
                mm_scores: list[float | None] = []
                for fold_idx in range(len(selection_origins)):
                    held_out_forecasts = {
                        name: [
                            forecast for index, forecast in enumerate(items)
                            if index != fold_idx
                        ]
                        for name, items in mm_fold_forecasts.items()
                    }
                    held_out_actuals = [
                        actual for index, actual in enumerate(mm_fold_actuals)
                        if index != fold_idx
                    ]
                    try:
                        fold_weights = train_meta_model(
                            held_out_forecasts, held_out_actuals,
                            non_negative=(
                                meta_model_cfg.non_negative if meta_model_cfg else True
                            ),
                            ridge_alpha=(
                                meta_model_cfg.ridge_alpha if meta_model_cfg else 1e-6
                            ),
                        )
                    except Exception:
                        fold_weights = None
                    if not fold_weights:
                        mm_scores.append(None)
                        continue
                    fold_map = {}
                    for name in fold_weights:
                        if name in mm_fold_forecasts and fold_idx < len(mm_fold_forecasts[name]):
                            f = mm_fold_forecasts[name][fold_idx]
                            if f:
                                fold_map[name] = f
                    if len(fold_map) >= 2:
                        try:
                            combined = pmm(fold_weights, fold_map)
                            actual = values[selection_origins[fold_idx] : selection_origins[fold_idx] + horizon]
                            mm_scores.append(error_score(actual, combined))
                            meta_model_fold_forecasts.append(combined)
                        except Exception:
                            mm_scores.append(None)
                            meta_model_fold_forecasts.append([])
                    else:
                        mm_scores.append(None)
                        meta_model_fold_forecasts.append([])
                valid_mm = [x for x in mm_scores if x is not None]
                if valid_mm and len(valid_mm) == len(selection_origins):
                    meta_model_score = mean(valid_mm)
                    if "meta_model" not in all_model_names:
                        all_model_names.append("meta_model")

    # --- Distributional fold scoring (pinball) ---
    def _pinball_score(forecasts: list[list[float]]) -> float | None:
        """Mean pinball loss over the selection folds, calibrated honestly.

        Fold *i* is scored with quantiles built from the residuals of folds
        before it, never its own — the same separation the calibration fold
        enforces for the published interval, applied inside selection. The
        first fold has nothing to calibrate from and is not scored.

        Reuses the fold forecasts already computed, so a distributional score
        costs no extra model fits.
        """
        by_lead: dict[int, list[float]] = {}
        pooled_residuals: list[float] = []
        losses: list[float] = []
        for index, origin in enumerate(selection_origins):
            if index >= len(forecasts) or not forecasts[index]:
                continue
            actual = values[origin : origin + horizon]
            forecast = forecasts[index]
            if pooled_residuals:
                spreads = conformal_quantile_spreads(by_lead, horizon, pooled_residuals)
                if spreads:
                    quantiles = [
                        {level: forecast[step - 1] + offset
                         for level, offset in spreads[step].items()}
                        for step in range(1, min(horizon, len(forecast)) + 1)
                    ]
                    loss = mean_pinball(actual, quantiles)
                    if loss is not None:
                        losses.append(loss)
            for step, (observed, predicted) in enumerate(zip(actual, forecast), 1):
                by_lead.setdefault(step, []).append(observed - predicted)
                pooled_residuals.append(observed - predicted)
        return mean(losses) if losses else None

    pinball_scores: dict[str, float | None] = {}
    if selection_loss == "pinball":
        for name in pool:
            if scores.get(name) is not None:
                pinball_scores[name] = _pinball_score(fold_forecasts[name])
        for adapter in all_adapters:
            if adapter_scores.get(adapter.name) is not None:
                pinball_scores[adapter.name] = _pinball_score(
                    [item for item in adapter_fold_forecasts[adapter.name]])

    baseline_scores = {name: score for name, score in scores.items() if name in BASELINES and score is not None}
    if not baseline_scores:
        return Evaluation(
            None, None, scores, empty_scores.copy(), None, [], None,
            ["No baseline completed every selection fold."], False,
            degraded, tsfm_scores=tsfm_scores,
            statistical_plugin_scores=statistical_plugin_scores,
            adapter_receipts={"statsforecast": statsforecast_receipt},
            notes=notes,
        )
    strongest_baseline = min(baseline_scores, key=baseline_scores.get)  # type: ignore[arg-type]
    selected = strongest_baseline
    baseline_score = baseline_scores[strongest_baseline]

    # Consider all non-baseline candidates (built-in + TSFM + ensemble + meta-model)
    candidate_scores: dict[str, float] = {}
    for name, score in scores.items():
        if name not in BASELINES and score is not None:
            candidate_scores[name] = score
    for name, score in adapter_scores.items():
        if score is not None:
            candidate_scores[name] = score
    for name, score in extra_scores.items():
        if score is not None:
            candidate_scores[name] = score
    if ensemble_score is not None:
        candidate_scores["ensemble"] = ensemble_score
    if meta_model_score is not None:
        candidate_scores["meta_model"] = meta_model_score

    if selection_loss == "pinball" and pinball_scores:
        # Decide on the distributional loss where it is available, keeping the
        # point loss reported alongside rather than replacing it. Candidates
        # without a pinball score (too few folds to calibrate one) keep their
        # point score, so switching the criterion never silently drops a
        # candidate from the contest.
        scored = {name: value for name, value in pinball_scores.items()
                  if value is not None}
        if scored:
            baseline_pinball = {name: value for name, value in scored.items()
                                if name in BASELINES}
            if baseline_pinball:
                strongest_baseline = min(baseline_pinball, key=baseline_pinball.get)  # type: ignore[arg-type]
                baseline_score = baseline_pinball[strongest_baseline]
                selected = strongest_baseline
                candidate_scores = {name: value for name, value in scored.items()
                                    if name not in BASELINES}

    # A ranked contest needs at least two disjoint selection folds. Dense
    # overlapping origins do not count: they lower comparison variance
    # without adding independent evidence. A one-fold run is reset below to
    # last-value and may depart only through the narrow prefix-structure
    # screen; this ordinary tournament runs only when `single_fold` is false.
    single_fold = len(residual_origins) < 2
    margin = minimum_improvement
    selection_stability: dict[str, object] = {
        "paired_folds": 0, "candidate_win_rate": None,
        "median_relative_gain": None, "scaled_error_improvement": None,
        "scaled_error_passed": False, "passed": True,
    }
    if candidate_scores and not single_fold:
        candidate = min(candidate_scores, key=candidate_scores.get)  # type: ignore[arg-type]
        candidate_score = candidate_scores[candidate]
        # Mean loss alone can promote a candidate on one spectacular fold
        # while it loses most regimes.  When aligned fold losses are
        # available, require it to win a majority and have positive median
        # relative gain. This is channel-agnostic and makes baseline fallback
        # a consequence of repeatable evidence, not a hard-coded vital name.
        def _fold_vector(name: str) -> list[float | None] | None:
            if name in fold_scores:
                return fold_scores[name]
            if name in adapter_fold_scores:
                return adapter_fold_scores[name]
            if name in extra_fold_scores:
                return extra_fold_scores[name]
            if name == "ensemble":
                return ensemble_fold_scores
            return None

        candidate_folds = _fold_vector(candidate)
        baseline_folds = _fold_vector(strongest_baseline)
        stable = True
        if candidate_folds is not None and baseline_folds is not None:
            paired = [(float(base), float(contender))
                      for base, contender in zip(baseline_folds, candidate_folds)
                      if base is not None and contender is not None
                      and math.isfinite(float(base))
                      and math.isfinite(float(contender))]
            if len(paired) >= 3:
                gains = [(base - contender) / max(abs(base), 1e-12)
                         for base, contender in paired]
                win_rate = sum(contender < base for base, contender in paired) / len(paired)
                median_gain = median(gains)
                stable = win_rate > .5 and median_gain > 0
                selection_stability = {
                    "paired_folds": len(paired),
                    "candidate_win_rate": win_rate,
                    "median_relative_gain": median_gain,
                    "scaled_error_improvement": None,
                    "scaled_error_passed": False,
                    "passed": stable,
                }
        def _forecast_vector(name: str) -> list[list[float]] | None:
            if name in fold_forecasts:
                return fold_forecasts[name]
            if name in adapter_fold_forecasts:
                return adapter_fold_forecasts[name]
            if name in extra_fold_forecasts:
                return extra_fold_forecasts[name]
            if name == "ensemble":
                return ensemble_fold_forecasts
            if name == "meta_model":
                return meta_model_fold_forecasts
            return None

        candidate_predictions = _forecast_vector(candidate)
        baseline_predictions = _forecast_vector(strongest_baseline)
        scaled_passed = False
        if candidate_predictions is not None and baseline_predictions is not None:
            scaled_pairs = []
            for index, origin in enumerate(selection_origins):
                if (index >= len(candidate_predictions)
                        or index >= len(baseline_predictions)
                        or not candidate_predictions[index]
                        or not baseline_predictions[index]):
                    continue
                actual = values[origin:origin + horizon]
                train = train_at(origin)
                base_scaled = scaled_error_score(
                    train, actual, baseline_predictions[index], season)
                candidate_scaled = scaled_error_score(
                    train, actual, candidate_predictions[index], season)
                if base_scaled is not None and candidate_scaled is not None:
                    scaled_pairs.append((base_scaled, candidate_scaled))
            if scaled_pairs:
                base_mean = mean(item[0] for item in scaled_pairs)
                candidate_mean = mean(item[1] for item in scaled_pairs)
                scaled_gain = ((base_mean - candidate_mean) / base_mean
                               if base_mean > 0 else None)
                scaled_wins = sum(contender < base for base, contender in scaled_pairs)
                scaled_passed = bool(
                    scaled_gain is not None and scaled_gain >= margin
                    and scaled_wins / len(scaled_pairs) > .5)
                selection_stability.update({
                    "scaled_error_paired_folds": len(scaled_pairs),
                    "scaled_error_improvement": scaled_gain,
                    "scaled_error_win_rate": scaled_wins / len(scaled_pairs),
                    "scaled_error_passed": scaled_passed,
                })
        stable = stable and scaled_passed
        selection_stability["passed"] = stable
        if (baseline_score > 0
                and candidate_score <= baseline_score * (1 - margin)
                and stable):
            selected = candidate
        elif not stable:
            notes.append(
                f"Selection double gate: {candidate} did not demonstrate "
                f"repeatable improvement under both WAPE and fold-local MASE; "
                f"{strongest_baseline} remains the published candidate."
            )
    fold_starved_evidence = None
    if single_fold:
        # One full-horizon result is not a tournament, even when one model's
        # margin looks spectacular.  Reset to the assumption-minimal baseline
        # and allow only the predeclared prefix-stability screen to depart.
        if scores.get("last_value") is not None:
            strongest_baseline = "last_value"
            baseline_score = float(scores["last_value"])
            selected = strongest_baseline
        fold_starved_evidence = _fold_starved_structural_evidence(
            values, horizon, season, train_at)
        if (fold_starved_evidence is not None
                and fold_starved_evidence["admitted"] is True):
            structural_candidate = str(fold_starved_evidence["candidate"])
            if scores.get(structural_candidate) is not None:
                selected = structural_candidate
        selection_stability["fold_starved_structural"] = (
            fold_starved_evidence)
    selection_guardrail_applied = bool(
        single_fold and (candidate_scores or len(baseline_scores) > 1)
        and selected == strongest_baseline)
    if selection_guardrail_applied:
        warnings.append(
            f"Selection under-powered: only {len(residual_origins)} disjoint "
            f"selection fold was available, too few to rank candidates. The "
            f"prefix-stability screen did not establish a safe structural "
            f"departure, so {strongest_baseline} is published; candidate "
            f"scores are reported as evidence, not a ranking. "
            f"{minimum_train + 4 * horizon} "
            f"observations enable a ranked contest."
        )
    elif single_fold and selected != strongest_baseline:
        warnings.append(
            f"Single-fold selection: {selected} cleared the degraded "
            f"prefix-stability screen against {strongest_baseline}. Dense "
            f"one-step replay was not counted as additional full-horizon "
            f"folds; the result remains degraded. Full diagnostics are "
            f"retained in sensitivity.fold_starved_structural."
        )

    # A pretrained candidate may carry independent transfer evidence.  It is
    # not equivalent to a locally fitted model with one fold: local outcomes
    # update the prior, but lack of local folds does not erase evidence earned
    # on held-out series.  This lane is dormant unless the caller supplies a
    # versioned prior whose exact model revision/regime it already verified.
    admission_decision: AdmissionDecision | None = None
    admission_candidate: str | None = None
    admission_weight: float | None = None
    if external_priors and tsfm_adapters:
        eligible_prior_candidates = [
            adapter.name for adapter in tsfm_adapters
            if adapter.name in external_priors
            and external_priors[adapter.name].usable()
        ]
        if eligible_prior_candidates:
            # Prior expected gain only chooses which pretrained candidate is
            # adjudicated; the admission decision still uses aligned local
            # folds and the strongest baseline as its safety reference.
            admission_candidate = max(
                eligible_prior_candidates,
                key=lambda name: external_priors[name].mean_relative_gain,
            )
            selected_adapter = next(
                adapter for adapter in tsfm_adapters
                if adapter.name == admission_candidate)
            try:
                candidate_path = _predict_adapter(
                    selected_adapter, values, horizon, season)
                baseline_path = _predict_statistical(
                    strongest_baseline, values, horizon, season)
                diagnostics = output_diagnostics(
                    values, candidate_path, baseline_path)
                candidate_direction = candidate_path[-1] - candidate_path[0]
                baseline_direction = baseline_path[-1] - baseline_path[0]
                conflicts = (
                    ("candidate and baseline forecast directions disagree",)
                    if candidate_direction * baseline_direction < 0
                    and (diagnostics.candidate_baseline_disagreement or 0) >= 1
                    else ()
                )
            except Exception as exc:
                from .admission import OutputDiagnostics
                diagnostics = OutputDiagnostics(
                    valid=False,
                    reasons=(f"candidate final-fit diagnostic failed: {type(exc).__name__}",),
                )
                conflicts = ("candidate output could not be diagnosed",)
            evidence = local_evidence(
                model_class="pretrained",
                candidate_losses=list(
                    adapter_fold_scores[admission_candidate]),
                baseline_losses=list(fold_scores[strongest_baseline]),
                external_prior=external_priors[admission_candidate],
                diagnostics=diagnostics,
            )
            if conflicts:
                evidence = AdmissionEvidence(
                    **{**evidence.__dict__, "conflicts": conflicts})
            admission_decision = decide_admission(
                candidate=admission_candidate, baseline=strongest_baseline,
                evidence=evidence,
                minimum_local_improvement=minimum_improvement,
            )
            admission_weight = admission_decision.candidate_weight
            if admission_decision.point_policy == "candidate":
                selected = admission_candidate
            elif admission_decision.point_policy == "shrunk_blend":
                selected = "admission_blend"
                candidate_items = adapter_fold_scores[admission_candidate]
                baseline_items = fold_scores[strongest_baseline]
                blended_items: list[float | None] = []
                for candidate_item, baseline_item in zip(
                        candidate_items, baseline_items):
                    if candidate_item is None or baseline_item is None:
                        blended_items.append(None)
                    else:
                        blended_items.append(
                            admission_weight * candidate_item
                            + (1 - admission_weight) * baseline_item)
                # This is only a reporting estimate. Calibration and test
                # always replay the actual point-wise blend below.
                valid_blended = [item for item in blended_items if item is not None]
                if valid_blended:
                    extra_scores["admission_blend"] = mean(valid_blended)
            notes.append(
                f"TSFM admission: {admission_candidate} is "
                f"{admission_decision.state}; point policy "
                f"{admission_decision.point_policy} with candidate weight "
                f"{admission_decision.candidate_weight:.3f}. External evidence "
                f"is not represented as local validation."
            )

    # --- Calibration ---
    all_scores = {**scores, **adapter_scores, **extra_scores}
    if ensemble_score is not None:
        all_scores["ensemble"] = ensemble_score
    if meta_model_score is not None:
        all_scores["meta_model"] = meta_model_score
    def _ensemble_predict(train: list[float],
                          fc_horizon: int | None = None) -> list[float]:
        """Recombine the member models on *train* only.

        The ensemble has to be predictable at an arbitrary fold origin like
        any other candidate. Without this it could win selection and then
        fail to produce a calibration forecast, which both discarded the
        winner and pushed interval calibration onto a trailing window that
        overlaps the report-only test fold.

        This is also the *published* combiner: the final candidate spec
        binds it, so the object that earned selection on the folds is the
        object whose points ship (unified plan, Phase 1A).
        """
        from .ensemble import compute_ensemble_forecast
        steps = horizon if fc_horizon is None else fc_horizon
        member_scores: dict[str, float | None] = {**scores, **adapter_scores}
        forecasts: dict[str, list[float]] = {}
        for name in pool:
            if member_scores.get(name) is None:
                continue
            try:
                forecasts[name] = _predict_statistical(
                    name, train, steps, season)
            except (ValueError, ArithmeticError):
                continue
        for adapter in all_adapters:
            if member_scores.get(adapter.name) is None:
                continue
            try:
                forecasts[adapter.name] = _predict_adapter(
                    adapter, train, steps, season)
            except Exception:
                logger.debug("ensemble member %s failed on a fold", adapter.name,
                             exc_info=True)
        if len(forecasts) < (ensemble_cfg.min_models if ensemble_cfg else 2):
            raise ValueError("too few ensemble members completed this fold")
        return compute_ensemble_forecast(
            forecasts, member_scores, strategy=ensemble_strategy,
            last_observed=train[-1] if train else 0.0, config=ensemble_cfg,
        )

    def _predict_selected(name: str, train: list[float], origin: int,
                          fc_horizon: int | None = None) -> list[float]:
        """Dispatch a prediction to a built-in model, a TSFM, the ensemble, or
        a cross-series candidate. ``origin`` is the fold's forecast origin —
        cross-series candidates need it because their inputs are other series,
        which ``train`` does not carry."""
        steps = horizon if fc_horizon is None else fc_horizon
        if name in MODELS:
            return _predict_statistical(name, train, steps, season)
        if name == "ensemble":
            return _ensemble_predict(train, fc_horizon=steps)
        if name == "admission_blend":
            if admission_candidate is None or admission_weight is None:
                raise ValueError("admission blend has no bound candidate")
            candidate_points = _predict_selected(
                admission_candidate, train, origin, fc_horizon=steps)
            baseline_points = _predict_statistical(
                strongest_baseline, train, steps, season)
            return [
                admission_weight * candidate_point
                + (1 - admission_weight) * baseline_point
                for candidate_point, baseline_point
                in zip(candidate_points, baseline_points)
            ]
        if name in extra_candidates:
            return extra_candidates[name](origin, steps)
        adapter = next((a for a in all_adapters if a.name == name), None)
        if adapter is None:
            raise ValueError(f"no adapter available for {name}")
        return _predict_adapter(adapter, train, steps, season)

    # A built-in can be valid on every earlier selection/calibration prefix
    # yet become outside its mathematical domain when later visible data
    # arrives (Croston-SBA is the canonical example: it requires non-negative
    # demand). Domain eligibility is not a score and may therefore be checked
    # on the complete visible history without using the report-only future to
    # choose a winner. Fall back before calibration so points, residuals,
    # intervals, identity, and publication all belong to the same executable.
    if selected in MODELS:
        try:
            _predict_selected(selected, values, len(values))
        except (ValueError, ArithmeticError, OverflowError) as exc:
            warnings.append(
                f"{selected} won earlier folds but cannot fit the complete "
                f"visible history ({exc}); publishing the strongest baseline "
                f"{strongest_baseline} instead."
            )
            selected = strongest_baseline

    # Confirm one already-selected contender on one later, disjoint origin.
    # This is a veto, not another tournament: no loser from selection can be
    # promoted here, and calibration/final-test data remain untouched. A
    # single fold can disprove non-inferiority but cannot estimate a reliable
    # uplift margin, so equality passes and any loss falls back.
    confirmation: dict[str, object] = {
        "available": confirmation_origin is not None,
        "required": selected != strongest_baseline,
        "origin_index": confirmation_origin,
        "candidate": selected,
        "baseline": strongest_baseline,
        "candidate_score": None,
        "baseline_score": None,
        "passed": None,
        "fallback_applied": False,
    }
    if confirmation_origin is not None and selected != strongest_baseline:
        contender = selected
        try:
            confirmation_actual = values[
                confirmation_origin:confirmation_origin + horizon]
            contender_score = error_score(
                confirmation_actual,
                _predict_selected(
                    contender, train_at(confirmation_origin),
                    confirmation_origin),
            )
            confirmation_baseline_score = error_score(
                confirmation_actual,
                _predict_selected(
                    strongest_baseline, train_at(confirmation_origin),
                    confirmation_origin),
            )
        except Exception:
            contender_score = confirmation_baseline_score = None
        passed = bool(
            contender_score is not None
            and confirmation_baseline_score is not None
            and contender_score <= confirmation_baseline_score
        )
        confirmation.update({
            "candidate_score": contender_score,
            "baseline_score": confirmation_baseline_score,
            "passed": passed,
            "fallback_applied": not passed,
        })
        if not passed:
            selected = strongest_baseline
            notes.append(
                f"Independent confirmation: {contender} did not remain "
                f"non-inferior to {strongest_baseline} on the reserved "
                f"confirmation fold, so the baseline is published."
            )
    if confirmation_origin is not None:
        selection_stability["confirmation"] = confirmation

    selected_score = all_scores.get(selected, baseline_score)
    improvement = 0.0 if selected in BASELINES else (baseline_score - selected_score) / baseline_score if baseline_score > 0 else 0.0  # type: ignore[operator]
    if improvement < 0:
        # Belt and braces. A negative `minimum_improvement` is refused up
        # front, but a criterion switch (pinball selection scores a
        # different quantity from the reported WAPE improvement) can still
        # land here. A model that lost the comparison it is reported
        # against must never be published as plainly supported.
        warnings.append(
            f"{selected} was selected but scored {abs(improvement):.2%} worse "
            f"than the strongest baseline {strongest_baseline} on the reported "
            f"metric; the selection criterion and the reported improvement "
            f"measure different quantities."
        )

    # Get calibration prediction from the selected model; fall back to the
    # strongest baseline if a TSFM/ensemble selection cannot predict here.
    try:
        calibration_prediction = _predict_selected(selected, train_at(calibration_origin), calibration_origin)
    except Exception as exc:
        if selected not in MODELS:
            logger.warning(
                "%s failed during calibration, falling back to %s: %s",
                selected, strongest_baseline, exc,
            )
        selected = strongest_baseline
        calibration_prediction = _predict_statistical(
            selected, train_at(calibration_origin), horizon, season)

    # Pool residuals of the selected model across every selection fold plus
    # the calibration fold: one horizon of residuals is too small a sample
    # for stable quantiles. Folds where the selected model cannot predict
    # (possible for TSFM adapters) simply contribute nothing.
    def _pool_residuals(
        name: str, final_prediction: list[float] | None = None,
    ) -> tuple[list[float], dict[int, list[float]]]:
        """Residuals of *name* over the calibration fold, and — when
        ``evaluation.pool_residuals`` is on, as it is by default — the
        selection folds as well. The test fold is never touched: it reports,
        it never calibrates.

        **Pooling is not split-conformal, and the direction of the error is
        known.** The selected model was chosen to minimise error on exactly
        the selection folds, so its residuals there are optimistically small
        and pooling them narrows the interval. The calibration fold alone
        *is* honest split conformal, but at one fold it supplies `horizon`
        residuals — too few for a stable tail quantile. That trade is the
        reason the default is what it is, not an oversight.
        ``evaluation.pool_residuals: false`` selects the honest, noisier
        alternative; either way the choice is disclosed on every result.
        """
        origins = residual_origins if pool_residuals else []
        pooled: list[float] = []
        by_lead: dict[int, list[float]] = {}

        def record(actual: list[float], prediction: list[float]) -> None:
            for step, (a, p) in enumerate(zip(actual, prediction), 1):
                pooled.append(a - p)
                by_lead.setdefault(step, []).append(a - p)

        adapter = next((candidate for candidate in all_adapters
                        if candidate.name == name), None)
        batch_predictions: list[list[float]] | None = None
        if adapter is not None and origins:
            try:
                batch_predictions = _predict_adapter_many(
                    adapter, [train_at(origin) for origin in origins],
                    horizon, season)
            except Exception:
                logger.debug("adapter %s residual batch failed", name,
                             exc_info=True)
        for index, origin in enumerate(origins):
            try:
                prediction = (batch_predictions[index]
                              if batch_predictions is not None
                              else _predict_selected(
                                  name, train_at(origin), origin))
            except Exception:
                continue
            record(values[origin : origin + horizon], prediction)
        if final_prediction is None:
            try:
                final_prediction = _predict_selected(name, train_at(calibration_origin), calibration_origin)
            except Exception:
                return pooled, by_lead
        record(values[calibration_origin : calibration_origin + horizon],
               final_prediction)
        return pooled, by_lead

    residuals, residuals_by_lead = _pool_residuals(selected, calibration_prediction)

    # A native marginal distribution is a candidate uncertainty method, not
    # an entitlement attached to a model name. Compare it with the conformal
    # reference on the reserved calibration fold, using only earlier
    # selection-origin residuals to construct that reference. The final test
    # fold remains report-only and can never change this choice.
    probabilistic_method = "conformal_residuals"
    probabilistic_assessment: dict[str, Any] = {
        "status": "not_available",
        "selected_method": probabilistic_method,
        "selection_partition": "reserved_calibration_fold",
        "test_fold_changes_selection": False,
    }
    native_quantile_adapter = next(
        (adapter for adapter in all_adapters
         if adapter.name == selected
         and bool(getattr(adapter, "supports_quantiles", False))),
        None,
    )
    if native_quantile_adapter is not None and degraded:
        probabilistic_assessment.update({
            "status": "withheld",
            "reason": (
                "native quantiles require a separated calibration fold; "
                "this run is degraded"),
        })
    elif native_quantile_adapter is not None and threshold_job:
        probabilistic_assessment.update({
            "status": "withheld",
            "reason": (
                "threshold-bearing runs require calibrated joint residual "
                "trajectories; native marginal quantiles do not encode "
                "cross-horizon dependence"),
        })
    elif (native_quantile_adapter is not None
          and intermittent_predictive_quantiles(values) is not None):
        probabilistic_assessment.update({
            "status": "withheld",
            "reason": (
                "the observed zero atom requires Gnomon's zero-inflated "
                "empirical distribution"),
        })
    if (native_quantile_adapter is not None and not degraded
            and not threshold_job
            and intermittent_predictive_quantiles(values) is None):
        prior_residuals: list[float] = []
        prior_by_lead: dict[int, list[float]] = {}
        completed_origins = 0
        for origin in residual_origins:
            try:
                prior_prediction = _predict_selected(
                    selected, train_at(origin), origin)
            except Exception:
                continue
            completed_origins += 1
            for step, (observed, predicted) in enumerate(zip(
                    values[origin:origin + horizon], prior_prediction), 1):
                residual = observed - predicted
                prior_residuals.append(residual)
                prior_by_lead.setdefault(step, []).append(residual)
        probabilistic_assessment["prior_origins"] = completed_origins
        try:
            native_calibration = predict_quantiles_checked(
                native_quantile_adapter, train_at(calibration_origin),
                horizon, season, QUANTILE_LEVELS)
            reference_spreads = conformal_quantile_spreads(
                prior_by_lead, horizon, prior_residuals)
            reference_calibration = [
                {level: calibration_prediction[step - 1] + offset
                 for level, offset in reference_spreads[step].items()}
                for step in range(1, horizon + 1)
            ] if len(reference_spreads) == horizon else []
            calibration_actual = values[
                calibration_origin:calibration_origin + horizon]
            native_loss = mean_pinball(
                calibration_actual, native_calibration)
            reference_loss = mean_pinball(
                calibration_actual, reference_calibration)

            def central_coverage(
                actual: list[float], rows: list[dict[float, float]],
            ) -> float | None:
                covered = [
                    float(row[.1] <= observed <= row[.9])
                    for observed, row in zip(actual, rows)
                    if .1 in row and .9 in row
                ]
                return mean(covered) if len(covered) == len(actual) else None

            native_coverage = central_coverage(
                calibration_actual, native_calibration)
            reference_coverage = central_coverage(
                calibration_actual, reference_calibration)
            complete = all(value is not None for value in (
                native_loss, reference_loss,
                native_coverage, reference_coverage,
            )) and completed_origins >= 2
            admitted = bool(
                complete
                and float(native_loss) <= float(reference_loss)
                and abs(float(native_coverage) - DEFAULT_TARGET_COVERAGE)
                <= abs(float(reference_coverage) - DEFAULT_TARGET_COVERAGE)
            )
            probabilistic_assessment.update({
                "status": "admitted" if admitted else "rejected",
                "selected_method": (
                    "native_quantiles" if admitted
                    else "conformal_residuals"),
                "native_mean_pinball": native_loss,
                "conformal_mean_pinball": reference_loss,
                "native_80_coverage": native_coverage,
                "conformal_80_coverage": reference_coverage,
                "nominal_coverage": DEFAULT_TARGET_COVERAGE,
                "criteria": (
                    "native mean pinball no worse and absolute coverage error "
                    "no larger than the conformal reference"),
            })
            if admitted:
                probabilistic_method = "native_quantiles"
        except Exception as exc:
            probabilistic_assessment.update({
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}"[:300],
            })
    # A dependency-free distribution candidate for heteroskedastic seasonal
    # data.  This is deliberately mutually exclusive with a selected model's
    # native distribution, and with the zero-inflated lane: one calibration
    # family competes, never an expanding menu.  Phase instability made a
    # single lucky calibration window unsafe in an independent stress shard,
    # so this candidate has a stricter contract than native model quantiles:
    # it must beat conformal at both sequential validation origins.
    _, _, detected_phase_basis = detect_season(
        values, frequency)
    phase_eligible = bool(
        native_quantile_adapter is None and not degraded
        and not threshold_job and confirmation_origin is not None
        # Phase pooling assumes fixed calendar semantics (for example, the
        # same hour of day), so a nearby autocorrelation estimate is not an
        # admissible substitute for the frequency's canonical period.  Nor
        # is a frequency fallback evidence that the phases actually repeat.
        and season == default_season(frequency)
        and detected_phase_basis == "autocorrelation"
        and intermittent_predictive_quantiles(values) is None
    )
    if phase_eligible:
        prior_residuals: list[float] = []
        prior_by_lead: dict[int, list[float]] = {}
        completed_origins = 0
        for origin in residual_origins:
            try:
                prior_prediction = _predict_selected(
                    selected, train_at(origin), origin)
            except Exception:
                continue
            completed_origins += 1
            for step, (observed, predicted) in enumerate(zip(
                    values[origin:origin + horizon], prior_prediction), 1):
                residual = observed - predicted
                prior_residuals.append(residual)
                prior_by_lead.setdefault(step, []).append(residual)
        validations: list[dict[str, float | int]] = []
        for origin in (confirmation_origin, calibration_origin):
            candidate = seasonal_phase_empirical_quantiles(
                train_at(origin), horizon, season)
            reference_spreads = conformal_quantile_spreads(
                prior_by_lead, horizon, prior_residuals)
            try:
                prediction = _predict_selected(
                    selected, train_at(origin), origin)
            except Exception:
                prediction = []
            reference = [
                {level: prediction[step - 1] + offset
                 for level, offset in reference_spreads[step].items()}
                for step in range(1, horizon + 1)
            ] if (candidate is not None and len(prediction) == horizon
                  and len(reference_spreads) == horizon) else []
            actual = values[origin:origin + horizon]
            candidate_loss = (
                mean_pinball(actual, candidate)
                if candidate is not None else None)
            reference_loss = mean_pinball(actual, reference)
            candidate_hits = ([
                float(row[.1] <= observed <= row[.9])
                for observed, row in zip(actual, candidate)
            ] if candidate is not None else [])
            reference_hits = [
                float(row[.1] <= observed <= row[.9])
                for observed, row in zip(actual, reference)
            ]
            if (candidate_loss is not None and reference_loss is not None
                    and len(candidate_hits) == len(actual)
                    and len(reference_hits) == len(actual)):
                validations.append({
                    "origin": origin,
                    "candidate_mean_pinball": candidate_loss,
                    "conformal_mean_pinball": reference_loss,
                    "candidate_80_coverage": mean(candidate_hits),
                    "conformal_80_coverage": mean(reference_hits),
                })
            # The next validation may use this already-scored origin as
            # conformal history; its outcome never changes an earlier result.
            if len(prediction) == horizon:
                for step, (observed, predicted) in enumerate(zip(
                        actual, prediction), 1):
                    residual = observed - predicted
                    prior_residuals.append(residual)
                    prior_by_lead.setdefault(step, []).append(residual)
        phase_loss = (mean(float(item["candidate_mean_pinball"])
                           for item in validations)
                      if validations else None)
        reference_loss = (mean(float(item["conformal_mean_pinball"])
                               for item in validations)
                          if validations else None)
        phase_coverage = (mean(float(item["candidate_80_coverage"])
                               for item in validations)
                          if validations else None)
        reference_coverage = (mean(float(item["conformal_80_coverage"])
                                   for item in validations)
                              if validations else None)
        complete = len(validations) == 2 and completed_origins >= 2
        admitted = bool(
            complete
            and all(
                float(item["candidate_mean_pinball"])
                <= float(item["conformal_mean_pinball"])
                for item in validations)
            and abs(float(phase_coverage) - DEFAULT_TARGET_COVERAGE)
            <= abs(float(reference_coverage) - DEFAULT_TARGET_COVERAGE)
        )
        if admitted:
            # A rejected internal candidate does not change the published
            # answer, so keep the established compact evidence payload.  An
            # admission is material and carries the complete gate receipt.
            probabilistic_assessment.update({
                "status": "admitted",
                "candidate": "seasonal_phase_empirical",
                "selected_method": "seasonal_phase_empirical",
                "prior_origins": completed_origins,
                "validation_origins": validations,
                "candidate_mean_pinball": phase_loss,
                "conformal_mean_pinball": reference_loss,
                "candidate_80_coverage": phase_coverage,
                "conformal_80_coverage": reference_coverage,
                "nominal_coverage": DEFAULT_TARGET_COVERAGE,
                "criteria": (
                    "seasonal-phase empirical mean pinball no worse at both "
                    "sequential validation origins and aggregate absolute "
                    "coverage error no larger than the conformal reference"),
            })
            probabilistic_method = "seasonal_phase_empirical"
    probabilistic_assessment["selected_method"] = probabilistic_method

    # Separate event-calibration trajectories.  These origins are excluded
    # from candidate selection whenever enough history exists; on shorter
    # histories the single calibration origin remains useful descriptively
    # but cannot clear MIN_JOINT_PATHS in the decision executable.
    event_residuals_by_lead: dict[int, list[float]] = {}
    for origin in event_calibration_origins:
        try:
            prediction = _predict_selected(
                selected, train_at(origin), origin)
        except Exception:
            continue
        actual = values[origin : origin + horizon]
        for step, (observed, predicted) in enumerate(
                zip(actual, prediction), 1):
            event_residuals_by_lead.setdefault(step, []).append(
                observed - predicted)
    # Degraded runs publish intervals centred on the point path (see
    # `conformal_spreads`); measuring test-fold coverage on the same
    # spreads keeps the measurement about the interval a reader actually
    # gets.
    spreads = conformal_spreads(
        residuals_by_lead, horizon, residuals, recentre=not degraded,
        finite_sample_expansion=not pool_residuals)

    # The `--ensemble` override can force the ensemble even when it did not win
    # selection. Calibrate it here, on the same folds, so the override never
    # has to fall back to a trailing window that overlaps the test fold.
    ensemble_residuals: list[float] = []
    ensemble_residuals_by_lead: dict[int, list[float]] = {}
    if ensemble_enabled and selected != "ensemble":
        ensemble_residuals, ensemble_residuals_by_lead = _pool_residuals("ensemble")

    # A TSFM or cross-series candidate can pass the folds and still fail at
    # the final prediction. Calibrate the baseline it would fall back to on
    # the same folds now, so the fallback publishes its own intervals rather
    # than inheriting the failed model's.
    fallback_residuals: list[float] = []
    fallback_residuals_by_lead: dict[int, list[float]] = {}
    if selected not in MODELS and strongest_baseline:
        fallback_residuals, fallback_residuals_by_lead = _pool_residuals(strongest_baseline)

    # --- Test ---
    test_scores: dict[str, float | None] = {name: None for name in all_model_names}
    coverage: float | None = None
    conformal_coverage: float | None = None
    if test_origin is not None:
        test_actual = values[test_origin : test_origin + horizon]
        for name in {selected, strongest_baseline}:
            try:
                test_scores[name] = error_score(
                    test_actual, _predict_selected(name, train_at(test_origin), test_origin)
                )
            except Exception:
                logger.debug("model %s failed during test fold", name, exc_info=True)

        # Get test prediction for coverage assessment
        try:
            test_prediction = _predict_selected(selected, train_at(test_origin), test_origin)
        except Exception:
            test_prediction = _predict_statistical(
                strongest_baseline, train_at(test_origin), horizon, season)
        covered = []
        intermittent = intermittent_predictive_quantiles(
            train_at(test_origin), (0.1, 0.5, 0.9))
        for step, (actual, prediction) in enumerate(zip(test_actual, test_prediction), 1):
            spread = spreads.get(step)
            if spread is None:
                continue
            if intermittent is not None:
                low, high = intermittent[0.1], intermittent[0.9]
            else:
                low, _, high = interval_from_spread(prediction, spread)
            covered.append(1.0 if low <= actual <= high else 0.0)
        conformal_coverage = mean(covered) if covered else None
        coverage = conformal_coverage
        if probabilistic_method == "native_quantiles" \
                and native_quantile_adapter is not None:
            try:
                native_test = predict_quantiles_checked(
                    native_quantile_adapter, train_at(test_origin), horizon,
                    season, QUANTILE_LEVELS)
                native_covered = [
                    float(row[.1] <= actual <= row[.9])
                    for actual, row in zip(test_actual, native_test)
                ]
                coverage = (mean(native_covered)
                            if len(native_covered) == len(test_actual) else None)
                probabilistic_assessment["report_only_test_80_coverage"] = coverage
            except Exception as exc:
                coverage = None
                warnings.append(
                    "Native quantiles were admitted on the calibration fold "
                    "but failed on the report-only test fold; coverage is "
                    f"unmeasured ({type(exc).__name__}: {exc}).")
        elif probabilistic_method == "seasonal_phase_empirical":
            phase_test = seasonal_phase_empirical_quantiles(
                train_at(test_origin), horizon, season)
            if phase_test is None:
                coverage = None
                warnings.append(
                    "Seasonal-phase empirical quantiles were admitted on the "
                    "calibration fold but could not be reconstructed on the "
                    "report-only test fold; coverage is unmeasured.")
            else:
                phase_covered = [
                    float(row[.1] <= actual <= row[.9])
                    for actual, row in zip(test_actual, phase_test)
                ]
                coverage = mean(phase_covered)
                probabilistic_assessment[
                    "report_only_test_80_coverage"] = coverage
        if coverage is not None and coverage < 0.7:
            warnings.append(
                f"Final-test 80% {probabilistic_method} interval coverage "
                f"was {coverage:.1%}, below 70%.")
    else:
        warnings.append(
            "Limited evaluation: no held-out test fold remained, so interval "
            "coverage is unmeasured."
        )
    # --- Executable candidates (unified plan, Phase 1A) -------------------
    # The winner's specification, bound to the SAME closures that produced
    # its calibration and test predictions. predict_stage publishes by
    # fitting this on the full visible history, so the evaluated object and
    # the published object cannot diverge. The season is deliberately the
    # evaluated one: it is part of the candidate's identity.
    from .candidate import CandidateIdentity, CandidateSpec, FittedCandidate

    def _fitted_weights(history: list[float]) -> dict[str, float] | None:
        """The weights that actually combine the members at this fit.

        Read from the same function the combiner uses, never recomputed
        by a second route that could drift from it.
        """
        if ensemble_strategy != "weighted_mean":
            return None
        from .ensemble import weighted_mean_weights
        member_scores: dict[str, float | None] = {**scores, **adapter_scores}
        forecasts: dict[str, list[float]] = {}
        for name in pool:
            if member_scores.get(name) is None:
                continue
            try:
                forecasts[name] = _predict_statistical(
                    name, history, horizon, season)
            except (ValueError, ArithmeticError):
                continue
        for adapter in all_adapters:
            if member_scores.get(adapter.name) is None:
                continue
            try:
                forecasts[adapter.name] = _predict_adapter(
                    adapter, history, horizon, season)
            except Exception:
                logger.debug("ensemble member %s failed at fit", adapter.name,
                             exc_info=True)
        usable = {name: value for name, value in member_scores.items()
                  if name in forecasts and value is not None}
        if not forecasts:
            return None
        return weighted_mean_weights(
            forecasts, usable,
            getattr(ensemble_cfg, "max_weight_ratio", 0.7)
            if ensemble_cfg else 0.7,
        )

    def _spec_for(identity: CandidateIdentity, name: str, *,
                  min_history: int | None = None,
                  max_horizon: int | None = None,
                  batch_predictor: Callable[
                      [list[list[float]], int, int | None],
                      list[list[float]]] | None = None) -> CandidateSpec:
        def fit(history: list[float], _season: int | None) -> FittedCandidate:
            from .ids import content_id
            # The visible-data fingerprint is of the history this instance
            # was fit on, so two fits of one spec on different histories
            # are distinguishable in the record.
            fingerprint = content_id("history", {"values": list(history)})
            weights = (_fitted_weights(history)
                       if identity.kind == "ensemble" else None)
            fitted_identity = identity.with_fit(
                weights=weights, data_fingerprint=fingerprint)

            def predictor(steps: int) -> list[float]:
                return _predict_selected(
                    name, history, len(history), fc_horizon=steps,
                )
            return FittedCandidate(fitted_identity, predictor)
        return CandidateSpec(
            identity, fit, min_history=min_history, max_horizon=max_horizon,
            predict_many=batch_predictor,
        )

    #: Dependency and weight revisions: the implementation version always,
    #: plus the pinned weight revision of every TSFM that competed.
    def _revisions(members: tuple[str, ...]) -> dict[str, str]:
        from .versioning import RUNTIME_VERSION
        revisions = {"runtime": RUNTIME_VERSION}
        adapters = {a.name: a for a in all_adapters}
        for member in members:
            adapter = adapters.get(member)
            model_id = getattr(adapter, "_MODEL_ID", None) if adapter else None
            if model_id:
                try:
                    from .tsfm import pinned_revision
                    revisions[member] = f"{model_id}@{pinned_revision(model_id)}"
                except Exception:
                    logger.debug("no pinned revision for %s", member,
                                 exc_info=True)
            elif adapter is not None:
                revision = getattr(adapter, "revision", None)
                remote_model = getattr(
                    getattr(adapter, "_provider", None), "model", "")
                if getattr(adapter, "kind", "") == "statistical_plugin":
                    revisions[member] = (
                        f"statsforecast@{revision}" if revision
                        else "unversioned:statsforecast")
                else:
                    revisions[member] = (
                        f"{remote_model or member}@{revision}"
                        if revision else f"unversioned:{remote_model or member}"
                    )
        return revisions

    ensemble_candidate = None
    if ensemble_enabled:
        member_names = tuple(sorted(
            [name for name in pool if scores.get(name) is not None]
            + [a.name for a in all_adapters
               if adapter_scores.get(a.name) is not None]
        ))
        behaviour: dict[str, Any] = {
            "min_models": ensemble_cfg.min_models if ensemble_cfg else 2,
        }
        if ensemble_strategy == "weighted_mean":
            behaviour["max_weight_ratio"] = (
                getattr(ensemble_cfg, "max_weight_ratio", 0.7)
                if ensemble_cfg else 0.7
            )
        if (ensemble_strategy == "voting" and ensemble_cfg is not None
                and isinstance(ensemble_cfg.voting, dict)):
            behaviour["threshold"] = ensemble_cfg.voting.get("threshold", 0.5)
        ensemble_candidate = _spec_for(
            CandidateIdentity(
                kind="ensemble", name="ensemble", members=member_names,
                strategy=ensemble_strategy, config=behaviour,
                revisions=_revisions(member_names),
                # Declining is the policy, not substituting: an ensemble
                # that cannot reach its member minimum publishes nothing
                # and the selected model reports instead, with its own
                # calibrated intervals.
                fallback_policy="decline_below_min_models",
            ),
            "ensemble",
        )

    final_candidate = None
    if selected == "ensemble":
        final_candidate = ensemble_candidate
    elif selected in MODELS:
        final_candidate = _spec_for(
            CandidateIdentity(
                kind="builtin", name=selected,
                revisions=_revisions(()),
                # Built-in domain eligibility was checked on the complete
                # visible history before calibration, so publication does not
                # need a second candidate-selection path.
                fallback_policy="none",
            ),
            selected,
        )
    elif selected in extra_candidates:
        final_candidate = _spec_for(
            CandidateIdentity(
                kind="cross_series", name=selected,
                revisions=_revisions(()),
                # Cross-series candidates can fail at final prediction
                # (their inputs are other series), and the baseline they
                # fall back to carries its own fold residuals.
                fallback_policy="strongest_baseline_recalibrated",
            ),
            selected,
        )
    elif any(adapter.name == selected for adapter in all_adapters):
        selected_adapter = next(
            adapter for adapter in all_adapters if adapter.name == selected)
        selected_capabilities = LegacyModelAdapter(selected_adapter).capabilities

        def selected_batch_predictor(
            histories: list[list[float]], steps: int, _season: int | None,
        ) -> list[list[float]]:
            batched = _predict_adapter_many(
                selected_adapter, histories, steps, season)
            if batched is None:
                return [_predict_adapter(
                    selected_adapter, history, steps, season)
                    for history in histories]
            return batched

        final_candidate = _spec_for(
            CandidateIdentity(
                kind=("statistical_plugin"
                      if getattr(selected_adapter, "kind", "") ==
                      "statistical_plugin" else "tsfm"),
                name=selected,
                config={
                    "adapter_protocol": "0.1",
                    "adapter_backend": str(getattr(
                        selected_adapter, "backend", "in_process")),
                    **({
                        "package": "statsforecast",
                        "model_class": selected_adapter.model_class,
                        "season_length": season,
                        "fit_history_limit": getattr(
                            selected_adapter, "fit_history_limit", None),
                        **({"components": list(selected_adapter.components)}
                           if hasattr(selected_adapter, "components") else {}),
                    } if getattr(selected_adapter, "kind", "") ==
                        "statistical_plugin" else {}),
                    "min_history": selected_capabilities.min_history,
                    "max_horizon": selected_capabilities.max_horizon,
                },
                revisions=_revisions((selected,)),
                fallback_policy="strongest_baseline_recalibrated",
            ),
            selected,
            min_history=selected_capabilities.min_history,
            max_horizon=selected_capabilities.max_horizon,
            batch_predictor=(selected_batch_predictor
                             if hasattr(selected_adapter, "predict_many")
                             else None),
        )
    elif (selected == "admission_blend" and admission_candidate is not None
          and admission_weight is not None and admission_decision is not None):
        final_candidate = _spec_for(
            CandidateIdentity(
                kind="blend", name="admission_blend",
                members=(admission_candidate, strongest_baseline),
                strategy="evidence_weighted_shrinkage",
                config={
                    "candidate_weight": admission_weight,
                    "policy_version": admission_decision.policy_version,
                    "admission_state": admission_decision.state,
                },
                revisions=_revisions((admission_candidate,)),
                fallback_policy="strongest_baseline_recalibrated",
            ),
            "admission_blend",
        )

    return Evaluation(selected, strongest_baseline,
                      {**scores, **statistical_plugin_scores, **extra_scores},
                      test_scores, improvement,
                      residuals, coverage, warnings, True, degraded,
                      final_candidate=final_candidate,
                      ensemble_candidate=ensemble_candidate,
                      tsfm_scores=tsfm_scores,
                      statistical_plugin_scores=statistical_plugin_scores,
                      adapter_receipts={"statsforecast": statsforecast_receipt},
                      probabilistic_method=probabilistic_method,
                      probabilistic_assessment=probabilistic_assessment,
                      conformal_coverage=conformal_coverage,
                      native_quantile_adapter=(
                          native_quantile_adapter
                          if probabilistic_method == "native_quantiles"
                          else None),
                      notes=notes,
                      residuals_by_lead=residuals_by_lead,
                      event_residuals_by_lead=event_residuals_by_lead,
                      event_residual_fold_count=(
                          min((len(items) for items in
                               event_residuals_by_lead.values()), default=0)
                      ),
                      ensemble_residuals=ensemble_residuals,
                      ensemble_residuals_by_lead=ensemble_residuals_by_lead,
                      fallback_residuals=fallback_residuals,
                      fallback_residuals_by_lead=fallback_residuals_by_lead,
                      pinball_scores=pinball_scores,
                      residuals_pooled_across_selection=pool_residuals,
                      selection_fold_count=len(residual_origins),
                      selection_guardrail_applied=selection_guardrail_applied,
                      admission_decision=admission_decision,
                      selection_stability=selection_stability,
                      residual_fold_count=(
                          len(residual_origins) + 1 if pool_residuals else 1
                      ))

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Callable

from .contracts import AionError
from .models import BASELINES, MODELS, predict
from .tsfm import TSFMError, TSFMUnavailable, tsfm_candidates

logger = logging.getLogger(__name__)


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


#: The metric every selection decision is made on. Named so that hindsight
#: scoring (`aion.tracking`) can report the same quantity: a leaderboard in
#: different units from selection can contradict it for reasons that are pure
#: metric artefact rather than forecast quality.
SELECTION_METRIC = "wape"


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
    if scale <= 1e-12:
        return None
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / scale


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


def conformal_spreads(
    residuals_by_lead: dict[int, list[float]], horizon: int,
    pooled: list[float] | None = None,
) -> dict[int, tuple[float, float, float]]:
    """Split-conformal offsets (low, median, high) for each lead time.

    Residuals are collected per lead time h, so the interval at h is the
    measured spread at h — not a pooled spread scaled by an assumed shape.
    Lead times with too few residuals borrow the pooled set rather than
    trusting a two-sample quantile, and the half-widths are then fitted
    monotone in h.
    """
    pooled = pooled if pooled is not None else [
        residual for residuals in residuals_by_lead.values() for residual in residuals
    ]
    if not pooled:
        return {}
    pooled_quantiles = {p: conformal_quantile(pooled, p) for p in (0.1, 0.5, 0.9)}

    medians: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for step in range(1, horizon + 1):
        residuals = residuals_by_lead.get(step) or []
        if len(residuals) >= MIN_RESIDUALS_PER_LEAD:
            quantiles = {p: conformal_quantile(residuals, p) for p in (0.1, 0.5, 0.9)}
        else:
            # Too few residuals at this lead to estimate its own tails:
            # borrow the pooled spread rather than invent precision.
            quantiles = pooled_quantiles
        medians.append(quantiles[0.5])
        lows.append(quantiles[0.5] - quantiles[0.1])
        highs.append(quantiles[0.9] - quantiles[0.5])

    lows = _isotonic([max(0.0, value) for value in lows])
    highs = _isotonic([max(0.0, value) for value in highs])
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
        medians.append(quantiles[0.5])
        for level in levels:
            # Signed distance from the median, so the isotonic fit below acts
            # on a half-width rather than on a level that may trend.
            by_level[level].append(quantiles[level] - quantiles[0.5])

    fitted: dict[float, list[float]] = {}
    for level in levels:
        offsets = by_level[level]
        if level < 0.5:
            # Lower tail: fit the magnitude monotone, then re-sign.
            widths = _isotonic([max(0.0, -value) for value in offsets])
            fitted[level] = [-value for value in widths]
        elif level > 0.5:
            fitted[level] = _isotonic([max(0.0, value) for value in offsets])
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
            prediction = predict(name, train, holdout, season)
            scores[name] = error_score(actual, prediction)
            forecasts[name] = prediction
        except (ValueError, ArithmeticError):
            continue
    valid = {name: score for name, score in scores.items() if score is not None}
    if not valid:
        return Evaluation(None, None, scores, scores.copy(), None, [], None,
                          ["Series is too short for lightweight model selection."], False, True)
    selected = min(valid, key=valid.get)  # type: ignore[arg-type]
    baselines = {name: score for name, score in valid.items() if name in BASELINES}
    strongest = min(baselines, key=baselines.get) if baselines else selected  # type: ignore[arg-type]
    residuals = [a - p for a, p in zip(actual, forecasts[selected])]
    return Evaluation(selected, strongest, scores, {name: None for name in MODELS}, None,
                      residuals, None, [
                          f"Degraded forecast: model selection used a single trailing {holdout}-observation holdout; rolling-origin calibration and final testing were unavailable. At least {max(2 * season, 2 * horizon, 8) + 2 * horizon} observations (have {len(values)}) are needed for separated selection and calibration."
                      ], True, True)


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
        raise AionError(
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
        if not strict_abstention:
            return select_model_lightweight(values, horizon, season, train_at)
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

    # Full mode holds out both a calibration fold and a final test fold after
    # the selection folds. With only two or three folds we degrade gracefully
    # instead of refusing: fewer selection folds, and with two folds no
    # held-out test at all — each degradation is named in a warning.
    degraded = len(origins) < 4
    warnings: list[str] = []
    if len(origins) >= 3:
        selection_origins, calibration_origin = origins[:-2], origins[-2]
        test_origin: int | None = origins[-1]
    else:
        selection_origins, calibration_origin, test_origin = origins[:-1], origins[-1], None
    # Residuals stay on the disjoint skeleton whatever selection does with
    # its stride: a conformal quantile over dependent residuals is not a
    # conformal quantile.
    residual_origins = list(selection_origins)
    # `evaluation.pool_residuals` (default true): pool the selection folds
    # with the calibration fold for sample size, accepting a known
    # optimistic bias. False is genuine split conformal — the calibration
    # fold only, whose origins selection never saw — and noisier. The key
    # was documented and never read; it is read here.
    pool_residuals = True
    if config is not None:
        evaluation_config = getattr(config, "evaluation", None)
        if evaluation_config is not None:
            pool_residuals = bool(
                getattr(evaluation_config, "pool_residuals", True)
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
    from .tsfm_sandbox import sandbox_tsfm_candidates, sandbox_available_tsfms
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
    sandbox_names = sandbox_available_tsfms()
    if sandbox_names and requested_names:
        tsfm_adapters = sandbox_tsfm_candidates(
            requested=requested_names,
            frequency=frequency,
        )
    elif requested_names:
        tsfm_adapters = tsfm_candidates(requested=requested_names, frequency=frequency)
    tsfm_model_names = [a.name for a in tsfm_adapters]
    all_model_names = list(MODELS.keys()) + tsfm_model_names

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

    # Disclose the model tier that could not compete. A fresh install has no
    # TSFM sandboxes, so without this note the operator most likely to benefit
    # from a stronger candidate never learns one was eligible.
    from .tsfm import installed_tsfms
    notes: list[str] = list(capability_notes)
    if requested_names and not sandbox_names and not installed_tsfms():
        notes.append(
            f"No foundation-model candidate competed: "
            f"{', '.join(requested_names)} "
            f"{'is' if len(requested_names) == 1 else 'are'} eligible for this "
            f"series but no sandbox is installed. Run "
            f"`aion tsfm install {requested_names[0]}` to add one; it enters "
            f"the same folds against the same baselines."
        )

    # --- Run built-in models on selection folds ---
    fold_scores: dict[str, list[float]] = {name: [] for name in MODELS}
    # Store per-fold forecasts for ensemble/meta-model training
    fold_forecasts: dict[str, list[list[float]]] = {name: [] for name in MODELS}
    for origin in selection_origins:
        actual = values[origin : origin + horizon]
        train = train_at(origin)
        for name in MODELS:
            try:
                forecast = predict(name, train, horizon, season)
            except ValueError:
                continue
            score = error_score(actual, forecast)
            if score is None:
                # Unscoreable fold (no scale in the window). Recording the
                # forecast without its score would desynchronise the two
                # lists the ensemble reads by fold index.
                continue
            fold_forecasts[name].append(forecast)
            fold_scores[name].append(score)

    # --- Run TSFM candidates on selection folds ---
    tsfm_fold_scores: dict[str, list[float]] = {a.name: [] for a in tsfm_adapters}
    tsfm_fold_forecasts: dict[str, list[list[float]]] = {a.name: [] for a in tsfm_adapters}
    for adapter in tsfm_adapters:
        for origin in selection_origins:
            actual = values[origin : origin + horizon]
            train = train_at(origin)
            try:
                forecast = adapter.predict(train, horizon, season)
                if len(forecast) != horizon:
                    tsfm_fold_scores[adapter.name].append(None)  # type: ignore[arg-type]
                    tsfm_fold_forecasts[adapter.name].append([])  # type: ignore[arg-type]
                    continue
                tsfm_fold_scores[adapter.name].append(error_score(actual, forecast))
                tsfm_fold_forecasts[adapter.name].append(forecast)
            except (TSFMError, TSFMUnavailable, Exception) as exc:
                logger.debug("TSFM %s failed on fold at origin %d: %s", adapter.name, origin, exc)
                tsfm_fold_scores[adapter.name].append(None)  # type: ignore[arg-type]
                tsfm_fold_forecasts[adapter.name].append([])  # type: ignore[arg-type]

    # --- Aggregate scores ---
    scores: dict[str, float | None] = {
        name: mean(items) if items and len(items) == len(selection_origins) else None
        for name, items in fold_scores.items()
    }
    tsfm_scores: dict[str, float | None] = {}
    for name, items in tsfm_fold_scores.items():
        valid = [x for x in items if x is not None]
        if valid and len(valid) == len(selection_origins):
            tsfm_scores[name] = mean(valid)
        else:
            tsfm_scores[name] = None

    # --- Run cross-series candidates on the same selection folds ---
    extra_candidates = dict(extra_candidates or {})
    extra_fold_scores: dict[str, list[float | None]] = {name: [] for name in extra_candidates}
    for name, predictor in extra_candidates.items():
        for origin in selection_origins:
            actual = values[origin : origin + horizon]
            try:
                forecast = predictor(origin, horizon)
            except Exception:
                logger.debug("candidate %s failed on fold at origin %d", name, origin,
                             exc_info=True)
                extra_fold_scores[name].append(None)
                continue
            if len(forecast) != horizon:
                extra_fold_scores[name].append(None)
                continue
            extra_fold_scores[name].append(error_score(actual, forecast))
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
    if ensemble_enabled or meta_model_enabled:
        from .ensemble import compute_ensemble_forecast, ENSEMBLE_MODEL_NAME

        def _weighting_scores(fold_idx: int) -> dict[str, float | None]:
            """Member scores available *before* fold ``fold_idx``.

            The inverse-error weights used to come from aggregates over
            every selection fold, fold ``fold_idx`` included — so each
            fold's ensemble was weighted using its own outcome, and the
            ensemble's selection score was optimistic by construction.
            Fold 0 has no prior evidence and so weights members equally.
            """
            prior: dict[str, float | None] = {}
            for source in (fold_scores, tsfm_fold_scores):
                for name, items in source.items():
                    earlier = [
                        item for item in items[:fold_idx] if item is not None
                    ]
                    prior[name] = mean(earlier) if earlier else None
            return prior

        for fold_idx in range(len(selection_origins)):
            fold_forecast_map: dict[str, list[float]] = {}
            for name in MODELS:
                if fold_idx < len(fold_forecasts[name]) and fold_forecasts[name][fold_idx]:
                    fold_forecast_map[name] = fold_forecasts[name][fold_idx]
            for adapter in tsfm_adapters:
                if fold_idx < len(tsfm_fold_forecasts[adapter.name]) and tsfm_fold_forecasts[adapter.name][fold_idx]:
                    fold_forecast_map[adapter.name] = tsfm_fold_forecasts[adapter.name][fold_idx]

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
                except Exception:
                    ensemble_fold_scores.append(None)
            else:
                ensemble_fold_scores.append(None)

    ensemble_score: float | None = None
    if ensemble_enabled and ensemble_fold_scores:
        valid = [x for x in ensemble_fold_scores if x is not None]
        if valid and len(valid) == len(selection_origins):
            ensemble_score = mean(valid)
            all_model_names.append("ensemble")

    # --- Meta-model training ---
    meta_model_weights: dict[str, float] | None = None
    meta_model_score: float | None = None
    if meta_model_enabled:
        from .meta_model import train_meta_model
        # Collect fold forecasts and actuals for training
        mm_fold_forecasts: dict[str, list[list[float]]] = {}
        mm_fold_actuals: list[list[float]] = []
        for fold_idx in range(len(selection_origins)):
            origin = selection_origins[fold_idx]
            actual = values[origin : origin + horizon]
            mm_fold_actuals.append(actual)

        for name in MODELS:
            if fold_forecasts[name]:
                mm_fold_forecasts[name] = fold_forecasts[name]
        for adapter in tsfm_adapters:
            valid_forecasts = [f for f in tsfm_fold_forecasts[adapter.name] if f]
            if valid_forecasts:
                mm_fold_forecasts[adapter.name] = tsfm_fold_forecasts[adapter.name]

        if len(mm_fold_forecasts) >= (meta_model_cfg.min_models if meta_model_cfg else 2):
            # The weights that will actually be used are fit on every fold,
            # like any final refit.
            meta_model_weights = train_meta_model(
                mm_fold_forecasts,
                mm_fold_actuals,
                non_negative=meta_model_cfg.non_negative if meta_model_cfg else True,
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
                        except Exception:
                            mm_scores.append(None)
                    else:
                        mm_scores.append(None)
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
        for name in MODELS:
            if scores.get(name) is not None:
                pinball_scores[name] = _pinball_score(fold_forecasts[name])
        for adapter in tsfm_adapters:
            if tsfm_scores.get(adapter.name) is not None:
                pinball_scores[adapter.name] = _pinball_score(
                    [item for item in tsfm_fold_forecasts[adapter.name]])

    baseline_scores = {name: score for name, score in scores.items() if name in BASELINES and score is not None}
    if not baseline_scores:
        return Evaluation(
            None, None, scores, empty_scores.copy(), None, [], None,
            ["No baseline completed every selection fold."], False,
            degraded, tsfm_scores=tsfm_scores, notes=notes,
        )
    strongest_baseline = min(baseline_scores, key=baseline_scores.get)  # type: ignore[arg-type]
    selected = strongest_baseline
    baseline_score = baseline_scores[strongest_baseline]

    # Consider all non-baseline candidates (built-in + TSFM + ensemble + meta-model)
    candidate_scores: dict[str, float] = {}
    for name, score in scores.items():
        if name not in BASELINES and score is not None:
            candidate_scores[name] = score
    for name, score in tsfm_scores.items():
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

    if candidate_scores:
        candidate = min(candidate_scores, key=candidate_scores.get)  # type: ignore[arg-type]
        candidate_score = candidate_scores[candidate]
        if baseline_score > 0 and candidate_score <= baseline_score * (1 - minimum_improvement):
            selected = candidate

    # --- Calibration ---
    all_scores = {**scores, **tsfm_scores, **extra_scores}
    if ensemble_score is not None:
        all_scores["ensemble"] = ensemble_score
    if meta_model_score is not None:
        all_scores["meta_model"] = meta_model_score
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

    def _ensemble_predict(train: list[float]) -> list[float]:
        """Recombine the member models on *train* only.

        The ensemble has to be predictable at an arbitrary fold origin like
        any other candidate. Without this it could win selection and then
        fail to produce a calibration forecast, which both discarded the
        winner and pushed interval calibration onto a trailing window that
        overlaps the report-only test fold.
        """
        from .ensemble import compute_ensemble_forecast
        member_scores: dict[str, float | None] = {**scores, **tsfm_scores}
        forecasts: dict[str, list[float]] = {}
        for name in MODELS:
            if member_scores.get(name) is None:
                continue
            try:
                forecasts[name] = predict(name, train, horizon, season)
            except (ValueError, ArithmeticError):
                continue
        for adapter in tsfm_adapters:
            if member_scores.get(adapter.name) is None:
                continue
            try:
                forecasts[adapter.name] = adapter.predict(train, horizon, season)
            except Exception:
                logger.debug("ensemble member %s failed on a fold", adapter.name,
                             exc_info=True)
        if len(forecasts) < (ensemble_cfg.min_models if ensemble_cfg else 2):
            raise ValueError("too few ensemble members completed this fold")
        return compute_ensemble_forecast(
            forecasts, member_scores, strategy=ensemble_strategy,
            last_observed=train[-1] if train else 0.0, config=ensemble_cfg,
        )

    def _predict_selected(name: str, train: list[float], origin: int) -> list[float]:
        """Dispatch a prediction to a built-in model, a TSFM, the ensemble, or
        a cross-series candidate. ``origin`` is the fold's forecast origin —
        cross-series candidates need it because their inputs are other series,
        which ``train`` does not carry."""
        if name in MODELS:
            return predict(name, train, horizon, season)
        if name == "ensemble":
            return _ensemble_predict(train)
        if name in extra_candidates:
            return extra_candidates[name](origin, horizon)
        adapter = next((a for a in tsfm_adapters if a.name == name), None)
        if adapter is None:
            raise ValueError(f"no adapter available for {name}")
        return adapter.predict(train, horizon, season)

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
        calibration_prediction = predict(selected, train_at(calibration_origin), horizon, season)

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

        for origin in origins:
            try:
                prediction = _predict_selected(name, train_at(origin), origin)
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
    spreads = conformal_spreads(residuals_by_lead, horizon, residuals)

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
            test_prediction = predict(strongest_baseline, train_at(test_origin), horizon, season)
        covered = []
        for step, (actual, prediction) in enumerate(zip(test_actual, test_prediction), 1):
            spread = spreads.get(step)
            if spread is None:
                continue
            low, _, high = interval_from_spread(prediction, spread)
            covered.append(1.0 if low <= actual <= high else 0.0)
        coverage = mean(covered) if covered else None
        if coverage is not None and coverage < 0.7:
            warnings.append(f"Final-test 80% interval coverage was {coverage:.1%}, below 70%.")
    else:
        warnings.append(
            "Limited evaluation: no held-out test fold remained, so interval "
            "coverage is unmeasured."
        )
    return Evaluation(selected, strongest_baseline, {**scores, **extra_scores},
                      test_scores, improvement,
                      residuals, coverage, warnings, True, degraded,
                      tsfm_scores=tsfm_scores, notes=notes,
                      residuals_by_lead=residuals_by_lead,
                      ensemble_residuals=ensemble_residuals,
                      ensemble_residuals_by_lead=ensemble_residuals_by_lead,
                      fallback_residuals=fallback_residuals,
                      fallback_residuals_by_lead=fallback_residuals_by_lead,
                      pinball_scores=pinball_scores,
                      residuals_pooled_across_selection=pool_residuals,
                      residual_fold_count=(
                          len(residual_origins) + 1 if pool_residuals else 1
                      ))

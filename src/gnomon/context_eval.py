"""The context admission gate: identical-fold ablation with known-at gating.

Context events never improve a forecast by assertion. This module replays
the exact selection folds used by the history-only evaluation, offering a
context-adjusted candidate the same information a real forecaster would
have had at each fold cutoff — only events that were verifiably known by
that cutoff, per ``gnomon.context.backtest_admissible``. Admission then
requires a stable, material improvement:

- every selection fold completes with context;
- mean improvement over the history-only selection meets the margin;
- more than half of the folds improve;
- the gain survives removing the single best fold; and
- rolling-origin calibrated interval coverage does not degrade beyond policy
  (0.1) on the untouched final test fold.

Anything else keeps the history-only result and records why.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from statistics import mean
from typing import Any

from .context import (
    ContextEvent,
    backtest_admissible,
    event_applies,
    validate_context_event,
)
from .context_model import (
    EFFECT_SHAPES, episode_residual_adjusted, episode_residual_amplitudes,
    episode_residual_observations, event_adjusted, residual_event_adjusted,
    rolling_residuals,
)
from .evaluation import (
    Evaluation,
    conformal_spreads,
    error_score,
    interval_from_spread,
)
from .models import MODELS

COVERAGE_DEGRADATION_LIMIT = 0.1
CONTEXT_MODEL_NAME = "event_adjusted"


#: Below this, a shrinkage factor is pinned to zero rather than applied. A
#: λ derived from four folds is itself noisy, and a small non-zero effect
#: that no evidence supports is worse than none: it moves the answer while
#: being indefensible if asked why.
MINIMUM_SHRINKAGE = 0.1

# The episode estimator is an additional comparison beyond the established
# drift/residual contest. A larger, fixed floor prices in that extra selection
# opportunity; otherwise adding an estimator makes admission easier merely by
# giving noise another chance to win.
EPISODE_MINIMUM_IMPROVEMENT = 0.05


def shrinkage_factor(improvements: list[float]) -> float:
    """How much of the measured effect the fold evidence actually supports.

    A binary gate at a fixed margin is high variance when there are four
    folds to decide on: a candidate just under the line gets nothing and one
    just over gets everything, on a difference well inside the noise. This
    is the empirical-Bayes answer to the same question — shrink the effect
    toward zero in proportion to how much of the observed mean could be
    sampling noise:

        λ = max(0, 1 − (standard error / mean)²)

    A mean much larger than its own standard error keeps nearly all of the
    effect; a mean comparable to its noise keeps almost none. Below
    ``MINIMUM_SHRINKAGE`` it is pinned to zero.

    This governs *strength*, never *validity*. Whether events are eligible,
    whether folds exist, whether the candidate fits, and whether coverage
    degraded are correctness conditions and stay binary vetoes: no amount of
    measured improvement makes a leaking event admissible.
    """
    if len(improvements) < 2:
        return 0.0
    average = mean(improvements)
    if average <= 0:
        return 0.0
    variance = sum((value - average) ** 2 for value in improvements) / (len(improvements) - 1)
    standard_error = (variance / len(improvements)) ** 0.5
    if standard_error <= 0:
        return 1.0
    factor = 1.0 - (standard_error / average) ** 2
    if factor < MINIMUM_SHRINKAGE:
        return 0.0
    return min(1.0, factor)


def wilson_upper(successes: float, trials: int, z: float = 1.96) -> float:
    """Upper bound of a Wilson score interval for a proportion.

    Coverage is measured on one test fold — a handful of points — so the
    point estimate is noisy enough that a veto based on it fires on
    sampling error alone. Comparing the *upper* bound against the policy
    limit means admission is refused only when the degradation is larger
    than the measurement's own uncertainty.
    """
    if trials <= 0:
        return 1.0
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = proportion + z * z / (2 * trials)
    margin = z * ((proportion * (1 - proportion) / trials
                   + z * z / (4 * trials * trials)) ** 0.5)
    return min(1.0, (centre + margin) / denominator)


@dataclass
class ContextAssessment:
    considered: bool
    admitted: bool
    reasons: list[str]
    events_used: list[str] = field(default_factory=list)
    events_excluded: list[dict[str, str]] = field(default_factory=list)
    fold_improvements: list[float] = field(default_factory=list)
    mean_improvement: float | None = None
    coverage: float | None = None
    residuals: list[float] = field(default_factory=list)
    points: list[float] = field(default_factory=list)
    residuals_by_lead: dict[int, list[float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # Which effect shape the ablation chose, and how every shape scored on
    # the same folds. Disclosed because a shape chosen by measurement is
    # only trustworthy if the measurement is visible.
    effect_shape: str = "level"
    effect_estimator: str = "detrended_level"
    shape_scores: dict[str, float] = field(default_factory=dict)
    # The shape the proposer nominated via `expected_shape`, when exactly
    # one was named across the eligible events. A nomination narrows the
    # contest to that shape alone — one comparison instead of three, so
    # *less* room to look good by luck — and a nominated shape that loses
    # is an honest exclusion, never a silent switch to a shape that fits.
    nominated_shape: str | None = None
    # How much of the measured effect the fold evidence supports, in [0, 1].
    # Always computed and disclosed; applied only when shrinkage is enabled.
    # `admitted` remains exactly `shrinkage > 0` in that mode, so the frozen
    # boolean keeps its meaning.
    shrinkage: float = 0.0
    # One entry per gate condition evaluated, whether it passed or not, with
    # the number it was decided on. `reasons` says what went wrong in prose;
    # this says what was measured, so admission rates and rejection causes
    # are countable across a run rather than parsed out of sentences.
    gate_checks: list[dict[str, Any]] = field(default_factory=list)

    def record_check(self, code: str, passed: bool, *, measured: Any = None,
                     threshold: Any = None, detail: str | None = None) -> None:
        entry: dict[str, Any] = {"code": code, "passed": passed}
        if measured is not None:
            entry["measured"] = measured
        if threshold is not None:
            entry["threshold"] = threshold
        if detail:
            entry["detail"] = detail
        self.gate_checks.append(entry)
        if not passed and detail:
            self.reasons.append(detail)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "considered": self.considered,
            "admitted": self.admitted,
            "model": CONTEXT_MODEL_NAME if self.admitted else None,
            "reasons": self.reasons,
            "events_used": self.events_used,
            "events_excluded": self.events_excluded,
            "fold_improvements": self.fold_improvements,
            "mean_improvement": self.mean_improvement,
            "effect_shape": self.effect_shape,
            "effect_estimator": self.effect_estimator,
            "shape_scores": self.shape_scores,
            "shrinkage": self.shrinkage,
            "measured_coverage": self.coverage,
            "gate_checks": self.gate_checks,
            # Additive: absent unless a shape was nominated, so artifacts
            # from nomination-free runs serialise exactly as before.
            **({"nominated_shape": self.nominated_shape}
               if self.nominated_shape else {}),
        }

    def to_cache_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_cache_dict(cls, payload: dict[str, Any]) -> "ContextAssessment":
        values = dict(payload)
        values["residuals_by_lead"] = {
            int(key): list(items) for key, items in
            (values.get("residuals_by_lead") or {}).items()
        }
        return cls(**values)


def eligible_events(
    events: list[ContextEvent], series_name: str
) -> tuple[list[ContextEvent], list[dict[str, str]]]:
    eligible: list[ContextEvent] = []
    excluded: list[dict[str, str]] = []
    for event in events:
        if not event_applies(event, series_name):
            excluded.append({"event_id": event.event_id, "reason": "scope does not include this series"})
        elif problems := validate_context_event(event):
            # `backtest_admissible` folds structural validity into its
            # verdict, but reporting every structural failure as "no
            # verifiable source" misnames the actual problem — an event
            # rejected for a bad `expected_shape` was told to fix its
            # source. Name what was measured.
            excluded.append({
                "event_id": event.event_id,
                "reason": "fails the event contract: " + "; ".join(problems),
            })
        elif event.status == "cancelled":
            excluded.append({
                "event_id": event.event_id,
                "reason": "cancelled event cannot affect the primary forecast",
            })
        elif not backtest_admissible(event):
            excluded.append({
                "event_id": event.event_id,
                "reason": ("no verifiable source; not admissible for backtesting"
                           if event.source is None else
                    "caller-declared known_at/source was not established at an "
                    "operator-controlled boundary; scenario-only, not admissible "
                    "for historical folds"),
            })
        else:
            eligible.append(event)
    return eligible, excluded


def event_flags(
    events: list[ContextEvent], timestamps: list[datetime], cutoff: datetime
) -> list[bool]:
    """Which periods an admissible event is active in.

    Every comparison goes through ``constraints._align``: events must carry
    an explicit offset while most datasets are naive, and comparing the two
    directly raised a bare TypeError out of the forecast.
    """
    from .constraints import _align

    known = [
        event for event in events
        if _align(datetime.fromisoformat(event.known_at), cutoff)[0]
        <= _align(datetime.fromisoformat(event.known_at), cutoff)[1]
    ]
    flags: list[bool] = []
    for timestamp in timestamps:
        active = False
        for event in known:
            start, moment = _align(
                datetime.fromisoformat(event.effective_start), timestamp,
            )
            end, _ = _align(datetime.fromisoformat(event.effective_end), timestamp)
            if start <= moment <= end:
                active = True
                break
        flags.append(active)
    return flags


def assess_context(
    values: list[float],
    timestamps: list[datetime],
    future_timestamps: list[datetime],
    events: list[ContextEvent],
    series_name: str,
    horizon: int,
    season: int,
    minimum_improvement: float,
    base: Evaluation,
    shrink: bool = False,
) -> ContextAssessment:
    eligible, excluded = eligible_events(events, series_name)
    # A naive dataset used to disqualify context outright, which made the
    # whole feature unreachable for every example this repo ships — events
    # are required to carry an offset, and no shipped dataset does. Windows
    # are now matched on wall-clock time (see `constraints._align`) and the
    # assumption is disclosed on the result, rather than the capability
    # being withdrawn.
    if not base.supported or base.selected_model is None:
        return ContextAssessment(
            False, False,
            ["history-only evaluation is unsupported; context cannot rescue an unsupported task"],
            events_excluded=excluded,
        )
    if not eligible:
        assessment = ContextAssessment(True, False, [], events_excluded=excluded)
        assessment.record_check(
            "events_eligible", False, measured=0, threshold=1,
            detail="no admissible events apply to this series",
        )
        return assessment

    # Replay the exact partitioning used by evaluate(): earlier folds select,
    # the penultimate fold calibrates, the final fold reports.
    minimum_train = max(2 * season, 2 * horizon, 8)
    origins = list(range(minimum_train, len(values) - horizon + 1, horizon))
    from .candidate import (
        candidate_predict, candidate_predict_many, eligible_origins,
    )
    origins = eligible_origins(base, origins, horizon)
    # An event-effect executable does not exist before its first historically
    # observable occurrence. Treating those origins as candidate failures made
    # high-frequency series especially brittle: the long seasonal warm-up can
    # still leave an early evaluation origin before the first event. Restrict
    # the contest to origins where the effect is identifiable, while keeping
    # the exact same origins for every estimator and shape. The cutoff-aware
    # flags also exclude events that were not yet known at that origin.
    event_fit_origins = []
    for origin in origins:
        historical_flags = event_flags(
            eligible, timestamps[:origin], timestamps[origin - 1])
        if any(historical_flags) and not all(historical_flags):
            event_fit_origins.append(origin)
    origins = event_fit_origins
    if len(origins) < 4:
        assessment = ContextAssessment(
            True, False, [],
            events_used=[event.event_id for event in eligible],
            events_excluded=excluded,
        )
        assessment.record_check(
            "separated_folds_available", False,
            measured=len(origins), threshold=4,
            detail="history evaluation ran in degraded mode (fewer than four "
                   "rolling folds); context ablation requires fully separated "
                   "selection, calibration, and test folds",
        )
        return assessment
    selection_origins, calibration_origin, test_origin = origins[:-2], origins[-2], origins[-1]

    # The entire ablation replays one immutable executable at known origins.
    # Batch-capable backends load/serve once; scalar backends retain identical
    # semantics through CandidateSpec's fallback.  Include calibration, test,
    # and publication now so later stages cannot accidentally restart an
    # isolated model for each fold.
    replay_origins = [*selection_origins, calibration_origin, test_origin]
    replay_histories = [values[:origin] for origin in replay_origins] + [values]
    replay_points = candidate_predict_many(
        base, replay_histories, horizon, season)
    base_path_cache = {
        len(history): points
        for history, points in zip(replay_histories, replay_points)
    }

    def _reject(code: str, measured: Any, detail: str) -> ContextAssessment:
        assessment = ContextAssessment(
            True, False, [],
            events_used=[event.event_id for event in eligible],
            events_excluded=excluded,
        )
        assessment.record_check(code, False, measured=measured, detail=detail)
        return assessment

    # Each shape is scored on the *same* folds, so the shape is chosen by
    # measurement rather than by the caller. A caller who could name the
    # shape could fit a story to the data, which is exactly what this gate
    # exists to prevent.
    # The candidate keeps its own drift base rather than being layered onto
    # the selected model. Layering was tried and is wrong: `event_effect`
    # measures the raw active-minus-inactive level difference, so adding it
    # to a base that *already* models the event -- any seasonal model, when
    # the events recur on a period -- counts the same bump twice, and fold
    # improvements collapse. The comparison here is therefore between two
    # complete models, which is also what makes it a fair contest.
    #
    def base_predict(history: list[float]) -> list[float]:
        cached = base_path_cache.get(len(history))
        if cached is not None:
            return list(cached)
        return candidate_predict(base, history, horizon, season)

    base_paths = {
        origin: base_predict(values[:origin]) for origin in selection_origins
    }
    builtin_base = base.selected_model in MODELS
    base_residuals = (
        rolling_residuals(values, base.selected_model, season)
        if builtin_base else []
    )
    # A proposer who read "3-day promotion, expect pull-forward" holds real
    # prior knowledge about the effect's shape. `expected_shape` lets it in
    # as a *nomination*: the contest narrows to that shape alone — one
    # comparison instead of three, strictly less room to fit a story — and
    # the nominated shape still has to beat the history-only baseline on
    # identical folds. Conflicting nominations across events cancel out and
    # the full contest runs; a caller still cannot pick the winner, only
    # shrink the field.
    nominations = {
        str(event.attributes.get("expected_shape"))
        for event in eligible
        if event.attributes.get("expected_shape")
    }
    nominated = nominations.pop() if len(nominations) == 1 else None
    contested_shapes = (nominated,) if nominated else EFFECT_SHAPES
    # All estimators face the identical gate. The drift candidate remains
    # useful when the selected model has absorbed an intervention; the
    # residual candidate isolates aperiodic effects after the selected model
    # has removed ordinary trend and seasonality; the episode candidate freezes
    # that model before each event so it cannot adapt midway through a pulse.
    candidates = [("drift", shape) for shape in contested_shapes]
    # Residual and episode estimators require one-step refits over the whole
    # history. They are defined for built-ins; an opaque/API/TSFM executable
    # still receives the fair drift-effect contest against its exact fold
    # paths, without being redispatched through the classical registry.
    if builtin_base:
        candidates += [("residual", shape) for shape in contested_shapes]
        candidates += [("episode", shape) for shape in contested_shapes]
    by_estimator: dict[str, dict[str, list[float]]] = {
        "drift": {}, "residual": {}, "episode": {}}
    episode_cache: dict[int, list[tuple[float, int, int]]] = {}
    for estimator, shape in candidates:
        improvements: list[float] = []
        failed: str | None = None
        for origin in selection_origins:
            cutoff = timestamps[origin - 1]
            actual = values[origin : origin + horizon]
            try:
                historical_flags = event_flags(
                    eligible, timestamps[:origin], cutoff)
                future_flags = event_flags(
                    eligible, timestamps[origin : origin + horizon], cutoff)
                if estimator == "episode":
                    if origin not in episode_cache:
                        episode_cache[origin] = episode_residual_observations(
                            values[:origin], historical_flags,
                            base.selected_model, season)
                    context_prediction = episode_residual_adjusted(
                        values[:origin], horizon, season, historical_flags,
                        future_flags, base_paths[origin], base.selected_model,
                        shape, episode_cache[origin])
                elif estimator == "residual":
                    context_prediction = residual_event_adjusted(
                        base_residuals[:origin], horizon, historical_flags,
                        future_flags, base_paths[origin], shape)
                else:
                    context_prediction = event_adjusted(
                        values[:origin], horizon, season, historical_flags,
                        future_flags, shape)
            except ValueError as exc:
                failed = str(exc)
                break
            base_score = error_score(actual, base_paths[origin])
            context_score = error_score(actual, context_prediction)
            if base_score is None or context_score is None:
                # No scale in this window. Scoring one arm and not the other
                # would break the identical-folds comparison the gate rests on.
                return _reject(
                    "folds_are_scoreable", origin,
                    "a selection fold has no scale to score against, so the "
                    "context candidate cannot be compared on identical folds",
                )
            # Symmetric relative improvement, bounded to [-1, 1]: a fold where
            # both candidates are (near-)exact contributes 0 rather than
            # aborting or dividing by zero.
            denominator = max(base_score, context_score)
            improvements.append(
                0.0 if denominator <= 1e-12 else (base_score - context_score) / denominator
            )
        if failed is None:
            by_estimator[estimator][shape] = improvements
        # One estimator may be unmeasurable while the other remains valid.
        # Both use the same nominated shape and identical folds, so allowing
        # the measurable estimator is not a silent switch in effect shape.

    available = [
        (mean(scores), estimator, shape, scores)
        for estimator, shapes in by_estimator.items()
        for shape, scores in shapes.items()
    ]
    if not available:
        return _reject(
            "candidate_fits_every_fold", "no shape fitted every fold",
            ((f"the nominated effect shape ({nominated}) did not fit every "
              "selection fold") if nominated else
             "no effect shape fitted every selection fold"),
        )
    best_score, selected_estimator, selected_shape, improvements = max(
        available, key=lambda item: item[0])
    # When every candidate loses, the selected shape is diagnostic only. Keep
    # the original detrended estimator's shape contest stable instead of
    # letting a losing residual estimator rewrite that public explanation.
    if best_score < minimum_improvement and by_estimator["drift"]:
        selected_estimator = "drift"
        selected_shape = max(
            by_estimator["drift"],
            key=lambda name: mean(by_estimator["drift"][name]))
        improvements = by_estimator["drift"][selected_shape]
    shape_scores = by_estimator[selected_estimator]

    def context_prediction(origin: int) -> list[float]:
        cutoff = timestamps[origin - 1]
        historical_flags = event_flags(eligible, timestamps[:origin], cutoff)
        future_flags = event_flags(
            eligible, timestamps[origin : origin + horizon], cutoff)
        if selected_estimator == "episode":
            return episode_residual_adjusted(
                values[:origin], horizon, season, historical_flags,
                future_flags,
                base_predict(values[:origin]),
                base.selected_model, selected_shape)
        if selected_estimator == "residual":
            base_path = base_predict(values[:origin])
            return residual_event_adjusted(
                base_residuals[:origin], horizon, historical_flags,
                future_flags, base_path, selected_shape)
        return event_adjusted(
            values[:origin], horizon, season, historical_flags, future_flags,
            selected_shape)

    assessment = ContextAssessment(
        True, False, [],
        events_used=[event.event_id for event in eligible],
        events_excluded=excluded,
        fold_improvements=improvements,
        mean_improvement=mean(improvements),
        effect_shape=selected_shape,
        effect_estimator=(
            "base_model_episode_residual" if selected_estimator == "episode"
            else "base_model_residual" if selected_estimator == "residual"
            else "detrended_level"),
        shape_scores={name: round(mean(scores), 6)
                      for name, scores in shape_scores.items()},
        nominated_shape=nominated,
    )

    assessment.record_check(
        "events_eligible", True, measured=len(eligible), threshold=1,
    )
    assessment.record_check(
        "separated_folds_available", True, measured=len(origins), threshold=4,
    )
    assessment.record_check(
        "candidate_fits_every_fold", True, measured=len(improvements),
    )
    assessment.record_check(
        "mean_improvement_meets_margin",
        assessment.mean_improvement >= (
            max(minimum_improvement, EPISODE_MINIMUM_IMPROVEMENT)
            if selected_estimator == "episode" else minimum_improvement),
        measured=round(assessment.mean_improvement, 6),
        threshold=(max(minimum_improvement, EPISODE_MINIMUM_IMPROVEMENT)
                   if selected_estimator == "episode" else minimum_improvement),
        detail=(f"mean fold improvement {assessment.mean_improvement:.3f} is below "
                "the required margin " + str(
                    max(minimum_improvement, EPISODE_MINIMUM_IMPROVEMENT)
                    if selected_estimator == "episode" else minimum_improvement)),
    )
    if selected_estimator == "episode":
        amplitudes = episode_residual_amplitudes(
            episode_cache[selection_origins[-1]], selected_shape)
        amplitude_mean = mean(amplitudes)
        variance = (sum((value - amplitude_mean) ** 2 for value in amplitudes) /
                    (len(amplitudes) - 1))
        standard_error = (variance / len(amplitudes)) ** 0.5
        lower_signal = abs(amplitude_mean) - 1.96 * standard_error
        assessment.record_check(
            "episode_effect_direction_is_stable",
            lower_signal > 0,
            measured={"episodes": len(amplitudes),
                      "mean_amplitude": round(amplitude_mean, 6),
                      "standard_error": round(standard_error, 6),
                      "lower_signal": round(lower_signal, 6)},
            threshold=0.0,
            detail=("the independent-episode 95% interval includes zero; "
                    "overlapping forecast folds are not independent evidence"),
        )
        # A recurring schedule can coincide with ordinary model error by
        # chance, especially when sliding validation folds overlap. Compare
        # its measured amplitude with deliberately displaced versions of the
        # same schedule. Requiring it to beat every fixed placebo prices in
        # the schedule-search opportunity without consulting the test fold.
        origin = selection_origins[-1]
        cutoff = timestamps[origin - 1]
        actual_flags = event_flags(eligible, timestamps[:origin], cutoff)
        placebo_amplitudes: list[float] = []
        for offset in (-19, -17, -13, -11, -9, -7, 7, 9, 11, 13, 17, 19):
            if offset > 0:
                shifted = [False] * offset + actual_flags[:-offset]
            else:
                distance = -offset
                shifted = actual_flags[distance:] + [False] * distance
            try:
                observations = episode_residual_observations(
                    values[:origin], shifted, base.selected_model, season)
                placebo_amplitudes.append(abs(mean(
                    episode_residual_amplitudes(observations, selected_shape))))
            except ValueError:
                continue
        actual_amplitude = abs(amplitude_mean)
        placebo_max = max(placebo_amplitudes, default=float("inf"))
        assessment.record_check(
            "episode_effect_beats_displaced_schedules",
            actual_amplitude > placebo_max,
            measured={"actual_amplitude": round(actual_amplitude, 6),
                      "placebo_max": round(placebo_max, 6),
                      "placebos": len(placebo_amplitudes)},
            threshold=round(placebo_max, 6),
            detail=("a displaced version of the same recurring schedule has "
                    "an equal or larger apparent effect"),
        )
    # A fold whose forecast window contains no eligible event is a valid
    # zero-lift observation for the *mean* comparison, but it says nothing
    # about whether the event effect is directionally stable.  Counting such
    # folds as stability failures made sparse, high-signal schedules harder
    # to admit merely because the history contained more quiet windows.
    # Preserve every fold above for selection/calibration, and use only
    # exposed windows for the majority and single-best robustness checks.
    exposed_improvements = [
        improvement
        for origin, improvement in zip(selection_origins, improvements)
        if any(event_flags(
            eligible,
            timestamps[origin : origin + horizon],
            timestamps[origin - 1],
        ))
    ]
    improved = sum(1 for value in exposed_improvements if value > 0)
    assessment.record_check(
        "majority_of_folds_improve",
        bool(exposed_improvements) and improved * 2 > len(exposed_improvements),
        measured={"improved": improved, "exposed": len(exposed_improvements)},
        threshold=len(exposed_improvements) // 2 + 1,
        detail=(f"only {improved} of {len(exposed_improvements)} event-exposed "
                "folds improved; a majority is required"),
    )
    if len(exposed_improvements) > 1:
        without_single_best = sorted(exposed_improvements)[:-1]
        assessment.record_check(
            "gain_survives_best_fold", mean(without_single_best) > 0,
            measured=round(mean(without_single_best), 6), threshold=0.0,
            detail="the gain is confined to a single fold",
        )

    # Calibrate and measure coverage with context before deciding, so a
    # coverage regression can veto admission.
    # One calibration fold supplies only one residual per lead, making the
    # interval equal to a single historical miss and producing unstable
    # coverage vetoes. Once the estimator/shape is frozen, replay its
    # leakage-safe selection predictions and pool those residuals with the
    # separate calibration fold. The final test fold remains untouched and
    # still vetoes a candidate whose resulting intervals degrade materially.
    assessment.residuals_by_lead = {
        step: [] for step in range(1, horizon + 1)}
    for origin in [*selection_origins, calibration_origin]:
        prediction = context_prediction(origin)
        actuals = values[origin : origin + horizon]
        for step, (actual, predicted) in enumerate(
                zip(actuals, prediction), 1):
            assessment.residuals_by_lead[step].append(actual - predicted)
    assessment.residuals = [
        residual for step in range(1, horizon + 1)
        for residual in assessment.residuals_by_lead[step]
    ]
    spreads = conformal_spreads(
        assessment.residuals_by_lead, horizon, assessment.residuals,
    )

    test_prediction = context_prediction(test_origin)
    test_actual = values[test_origin : test_origin + horizon]
    covered = []
    for step, (actual, prediction) in enumerate(zip(test_actual, test_prediction), 1):
        spread = spreads.get(step)
        if spread is None:
            continue
        low, _, high = interval_from_spread(prediction, spread)
        covered.append(1.0 if low <= actual <= high else 0.0)
    assessment.coverage = mean(covered) if covered else None
    if base.coverage is not None and assessment.coverage is not None:
        # Veto on the interval, not the point estimate: with one test fold
        # of `horizon` points, a coverage drop well inside sampling noise
        # would otherwise reject context that never degraded anything.
        limit = base.coverage - COVERAGE_DEGRADATION_LIMIT
        upper = wilson_upper(sum(covered), len(covered))
        assessment.record_check(
            "coverage_not_degraded", upper >= limit,
            measured={"coverage": round(assessment.coverage, 4),
                      "upper_bound": round(upper, 4),
                      "points": len(covered)},
            threshold=round(limit, 4),
            detail=(f"interval coverage degraded from {base.coverage:.1%} to "
                    f"{assessment.coverage:.1%} (upper bound {upper:.1%}, "
                    f"below the {limit:.1%} policy limit)"),
        )

    # Computed and disclosed whatever the mode, so the number is visible
    # before anything is decided on it.
    assessment.shrinkage = shrinkage_factor(improvements)

    if assessment.reasons:
        return assessment

    assessment.admitted = True
    final_cutoff = timestamps[-1]
    if selected_estimator == "episode":
        assessment.points = episode_residual_adjusted(
            values, horizon, season,
            event_flags(eligible, timestamps, final_cutoff),
            event_flags(eligible, future_timestamps, final_cutoff),
            base_predict(values),
            base.selected_model, selected_shape,
        )
    elif selected_estimator == "residual":
        assessment.points = residual_event_adjusted(
            base_residuals, horizon,
            event_flags(eligible, timestamps, final_cutoff),
            event_flags(eligible, future_timestamps, final_cutoff),
            base_predict(values),
            selected_shape,
        )
    else:
        assessment.points = event_adjusted(
            values, horizon, season,
            event_flags(eligible, timestamps, final_cutoff),
            event_flags(eligible, future_timestamps, final_cutoff),
            selected_shape,
        )
    if shrink:
        # Move only as far from the history-only forecast as the fold
        # evidence supports. `admitted` stays a clean boolean for readers of
        # the frozen shape -- it is exactly `shrinkage > 0` -- and the factor
        # sits alongside it.
        if assessment.shrinkage <= 0:
            assessment.admitted = False
            assessment.record_check(
                "effect_survives_shrinkage", False,
                measured=0.0, threshold=MINIMUM_SHRINKAGE,
                detail="the measured improvement is within its own sampling "
                       "noise, so no part of the effect is supported",
            )
            return assessment
        assessment.record_check(
            "effect_survives_shrinkage", True,
            measured=round(assessment.shrinkage, 6), threshold=MINIMUM_SHRINKAGE,
        )
        history_only = base_predict(values)
        assessment.points = [
            plain + assessment.shrinkage * (adjusted - plain)
            for plain, adjusted in zip(history_only, assessment.points)
        ]
    if assessment.coverage < 0.7:
        assessment.warnings.append(
            f"Final-test 80% interval coverage was {assessment.coverage:.1%}, below 70%."
        )
    return assessment

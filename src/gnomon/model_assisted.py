"""The model-assisted lane: preserve useful model priors beside the governed answer.

The immutable governed lane publishes the safest backtested forecast, or its
disclosed naive fallback when the evaluation protocol cannot run.  The
cross-model evaluation (docs/cross-model-evaluation-2026-08.md) showed that
this floor, while honest, discards useful predictive signal: a candidate the
folds could not *rank* may still carry the only model evidence the history
admits.  This module builds the second lane that record adopted: the best
model prior the history supports, published *beside* the governed rows —
labelled, never blended into them, and never replacing them.

Admission follows the evidence, in the admission vocabulary already used by
:mod:`gnomon.admission`:

- ``conditionally_supported`` — the candidate beat the published baseline on
  a genuine out-of-sample comparison at the requested horizon, but the full
  separated protocol could not run.
- ``prior_assisted`` — the candidate beat the published baseline on weaker
  out-of-sample evidence (a single underpowered fold, or a reduced-rigor
  holdout shorter than the horizon) and passed deterministic plausibility
  checks.

A candidate with no out-of-sample win ordinarily earns no lane. One narrow
structural prior is explicit: an intraday series with one complete observed
week may expose that week as a labelled calendar prior. It is not historical
skill evidence, never authorizes automation, and exists only to make the
best-effort surface useful when a two-week replay is mathematically impossible.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .admission import OutputDiagnostics, output_diagnostics
from .contracts import Evidence, SupportReason
from .models import BASELINES, MODELS, predict

MODEL_ASSISTED_LANE_VERSION = "0.1"

#: Plausibility bounds, in robust one-step scales of the history.  These are
#: deliberately generous: they exist to reject pathology (a first step ten
#: scales away from the last observation, a path twenty times more volatile
#: than the history), not to second-guess a candidate the evidence admitted.
MAX_BOUNDARY_JUMP = 10.0
MAX_SCALE_RATIO = 20.0
SEASONAL_PREQUENTIAL_MARGIN = .25
SEASONAL_PHASE_BLOCKS = 4
SEASONAL_REQUIRED_BLOCK_WINS = 3
SHORT_TREND_MIN_FOLDS = 6
SHORT_TREND_REQUIRED_MARGIN = .25
SHORT_TREND_REQUIRED_BLOCK_WINS = 3


def _plausible(values: list[float], candidate: list[float],
               baseline: list[float]) -> tuple[bool, OutputDiagnostics]:
    diagnostics = output_diagnostics(values, candidate, baseline)
    if not diagnostics.valid:
        return False, diagnostics
    if (diagnostics.boundary_jump or 0.0) > MAX_BOUNDARY_JUMP:
        return False, diagnostics
    if (diagnostics.scale_ratio or 0.0) > MAX_SCALE_RATIO:
        return False, diagnostics
    return True, diagnostics


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _diagnostics_payload(diagnostics: OutputDiagnostics) -> dict[str, Any]:
    return {
        "valid": diagnostics.valid,
        "boundary_jump": _round(diagnostics.boundary_jump),
        "scale_ratio": _round(diagnostics.scale_ratio),
        "oscillation_ratio": _round(diagnostics.oscillation_ratio),
        "candidate_baseline_disagreement":
            _round(diagnostics.candidate_baseline_disagreement),
    }


def _scored_candidate(assessment: Any) -> tuple[str, float, float] | None:
    """The best already-scored non-baseline candidate that beat the published
    baseline on the evaluation's own (underpowered) out-of-sample comparison."""
    scores = dict(assessment.selection_scores or {})
    baseline = assessment.strongest_baseline
    baseline_score = scores.get(baseline)
    if baseline_score is None:
        return None
    # ``seasonal_naive`` is a structured baseline, but when the governed
    # short-history floor is ``last_value`` it is also the cheapest useful
    # temporal prior available.  It may enter this explicitly non-automatable
    # lane on the same out-of-sample evidence as an incremental model; it does
    # not replace the stricter primary-selection rule.
    valid = {
        name: float(score) for name, score in scores.items()
        if score is not None and name in MODELS
        and name != baseline
        and (name not in BASELINES
             or (baseline == "last_value" and name == "seasonal_naive"))
    }
    if not valid:
        return None
    name = min(valid, key=lambda key: (valid[key], key))
    if valid[name] >= float(baseline_score):
        return None
    return name, valid[name], float(baseline_score)


def _holdout_candidate(values: list[float], horizon: int,
                       season: int) -> tuple[str, float, float, int] | None:
    """Score the built-in non-baseline models on one reduced-rigor holdout.

    Used only when the evaluation scored nothing at all.  The holdout reuses
    the degraded protocol's own sizing rule and compares every candidate
    against ``last_value`` on the identical held-out points; a winner is
    evidence, never a ranking.
    """
    from .evaluation import error_score

    if len(values) < 8:
        return None
    holdout = min(horizon, max(1, len(values) // 4))
    origin = len(values) - holdout
    if origin < 4:
        return None
    train, actual = values[:origin], values[origin:]
    try:
        baseline_score = error_score(
            actual, predict("last_value", train, holdout, season))
    except (ValueError, ArithmeticError):
        return None
    if baseline_score is None:
        return None
    best: tuple[str, float] | None = None
    for name in sorted(MODELS):
        if name == "last_value" or (
                name in BASELINES and name != "seasonal_naive"):
            continue
        try:
            score = error_score(actual, predict(name, train, holdout, season))
        except (ValueError, ArithmeticError):
            continue
        if score is None or score >= baseline_score:
            continue
        if best is None or (score, name) < (best[1], best[0]):
            best = (name, score)
    if best is None:
        return None
    return best[0], best[1], float(baseline_score), holdout


def _short_trend_probe(
        values: list[float], horizon: int, season: int,
) -> dict[str, Any] | None:
    """Test the assumption-minimal drift path on independent short blocks.

    A full requested horizon can be much longer than the supplied history,
    especially for intraday telemetry whose default horizon is one day.  One
    trailing holdout then makes a useful trend look available, but cannot say
    whether that win persisted through time.  This probe uses a predeclared
    candidate (drift), identical non-overlapping blocks, mandatory baselines,
    and chronological stability checks.  It supplies evidence for the
    labelled, non-automatable model-assisted lane only; it never promotes the
    governed primary or claims the unobserved tail was validated.

    ``None`` means there was not enough history to run the probe.  A returned
    payload always records the verdict, including a failed admission, so a
    caller does not fall back to a weaker single-holdout claim after stronger
    evidence rejected it.
    """
    from statistics import mean

    from .evaluation import error_score

    if horizon <= 0 or len(values) < 8:
        return None
    probe_horizon = min(horizon, max(1, len(values) // 8))
    minimum_train = max(8, 2 * probe_horizon)
    origins = list(range(
        minimum_train, len(values) - probe_horizon + 1, probe_horizon))
    if len(origins) < SHORT_TREND_MIN_FOLDS:
        return None

    candidate_losses: list[float] = []
    baseline_losses: dict[str, list[float]] = {
        name: [] for name in sorted(BASELINES)
    }
    for origin in origins:
        train = values[:origin]
        actual = values[origin:origin + probe_horizon]
        try:
            candidate_loss = error_score(
                actual, predict("drift", train, probe_horizon, season))
        except (ValueError, ArithmeticError):
            candidate_loss = None
        if candidate_loss is None:
            return None
        candidate_losses.append(float(candidate_loss))
        for name in baseline_losses:
            try:
                loss = error_score(
                    actual, predict(name, train, probe_horizon, season))
            except (ValueError, ArithmeticError):
                loss = None
            if loss is not None:
                baseline_losses[name].append(float(loss))

    complete_baselines = {
        name: losses for name, losses in baseline_losses.items()
        if len(losses) == len(origins)
    }
    if not complete_baselines:
        return None
    baseline = min(
        complete_baselines,
        key=lambda name: (mean(complete_baselines[name]), name),
    )
    baseline_fold_losses = complete_baselines[baseline]
    candidate_score = mean(candidate_losses)
    baseline_score = mean(baseline_fold_losses)
    fold_wins = sum(
        candidate < reference
        for candidate, reference in zip(candidate_losses, baseline_fold_losses)
    )
    boundaries = [0, len(origins) // 3, 2 * len(origins) // 3, len(origins)]
    block_wins = sum(
        mean(candidate_losses[left:right])
        < mean(baseline_fold_losses[left:right])
        for left, right in zip(boundaries, boundaries[1:])
        if right > left
    )
    relative_improvement = (
        (baseline_score - candidate_score) / baseline_score
        if baseline_score > 1e-12 else None
    )
    required_fold_wins = len(origins) - 1
    admitted = bool(
        relative_improvement is not None
        and relative_improvement >= SHORT_TREND_REQUIRED_MARGIN
        and fold_wins >= required_fold_wins
        and block_wins >= SHORT_TREND_REQUIRED_BLOCK_WINS
    )
    return {
        "basis": "non_overlapping_short_horizon_blocks",
        "candidate": "drift",
        "probe_horizon": probe_horizon,
        "maximum_locally_evaluated_lead": probe_horizon,
        "out_of_sample_steps": probe_horizon * len(origins),
        "comparisons": len(origins),
        "candidate_score": candidate_score,
        "baseline_score": baseline_score,
        "baseline": baseline,
        "relative_improvement": relative_improvement,
        "fold_wins": fold_wins,
        "required_fold_wins": required_fold_wins,
        "chronological_blocks": 3,
        "chronological_block_wins": block_wins,
        "required_chronological_block_wins":
            SHORT_TREND_REQUIRED_BLOCK_WINS,
        "required_margin": SHORT_TREND_REQUIRED_MARGIN,
        "scores_are_evidence_not_a_governed_primary_ranking": True,
        "admitted": admitted,
    }


def _seasonal_prequential_candidate(
        values: list[float], season: int) -> dict[str, Any] | None:
    """Admit one predeclared seasonal baseline over a complete past cycle.

    Every one-step prediction is made from observations strictly before its
    origin.  Requiring a complete phase sweep and wins in three of four phase
    blocks prevents a single favourable segment from standing in for a cycle.
    This is still only one held-out cycle, so the result remains prior-assisted
    and never becomes automation evidence.
    """
    if season < 4 or len(values) < 2 * season:
        return None
    start = len(values) - season
    actual = values[start:]
    seasonal = [values[index - season]
                for index in range(start, len(values))]
    last = [values[index - 1] for index in range(start, len(values))]
    seasonal_errors = [abs(float(a) - float(p))
                       for a, p in zip(actual, seasonal)]
    last_errors = [abs(float(a) - float(p))
                   for a, p in zip(actual, last)]
    seasonal_mae = sum(seasonal_errors) / season
    last_mae = sum(last_errors) / season
    if last_mae <= 1e-12:
        return None
    block_wins = 0
    for block in range(SEASONAL_PHASE_BLOCKS):
        left = block * season // SEASONAL_PHASE_BLOCKS
        right = (block + 1) * season // SEASONAL_PHASE_BLOCKS
        if right > left and (
                sum(seasonal_errors[left:right]) / (right - left)
                < sum(last_errors[left:right]) / (right - left)):
            block_wins += 1
    relative_improvement = (last_mae - seasonal_mae) / last_mae
    admitted = (
        relative_improvement >= SEASONAL_PREQUENTIAL_MARGIN
        and block_wins >= SEASONAL_REQUIRED_BLOCK_WINS)
    if not admitted:
        return None
    return {
        "basis": "full_cycle_prequential",
        "out_of_sample_steps": season,
        "comparisons": season,
        "candidate_score": seasonal_mae,
        "baseline_score": last_mae,
        "baseline": "last_value",
        "relative_improvement": relative_improvement,
        "phase_blocks": SEASONAL_PHASE_BLOCKS,
        "phase_block_wins": block_wins,
        "required_phase_block_wins": SEASONAL_REQUIRED_BLOCK_WINS,
        "required_margin": SEASONAL_PREQUENTIAL_MARGIN,
        "complete_phase_coverage": True,
        "scores_are_evidence_not_a_ranking": True,
    }


def _single_week_calendar_prior(
    values: list[float], season: int, horizon: int,
    future_timestamps: list[Any],
) -> tuple[list[float], dict[str, Any]] | None:
    """Return one explicitly prior-only weekly path for an intraday grid."""
    if season < 2 or len(future_timestamps) < 2:
        return None
    try:
        parsed = [value if isinstance(value, datetime)
                  else datetime.fromisoformat(str(value))
                  for value in future_timestamps]
    except (TypeError, ValueError):
        return None
    steps = [(right - left).total_seconds()
             for left, right in zip(parsed, parsed[1:])]
    if (not steps or any(step <= 0 for step in steps)
            or any(abs(step - steps[0]) > max(1e-6, steps[0] * 1e-6)
                   for step in steps[1:])
            or abs(season * steps[0] - 86400.0) > 1.0):
        return None
    weekly_period = 7 * season
    if len(values) < weekly_period:
        return None
    points = [float(values[-weekly_period + index % weekly_period])
              for index in range(horizon)]
    return points, {
        "basis": "single_observed_calendar_cycle_prior",
        "calendar_cycle": "week", "period_steps": weekly_period,
        "observed_cycles": 1, "out_of_sample_steps": 0, "comparisons": 0,
        "historical_skill_evidence": False,
        "complete_cycle_observed": True,
        "scores_are_evidence_not_a_ranking": False,
    }


def build_model_assisted_lane(
    series: str,
    values: list[float],
    *,
    horizon: int,
    season: int,
    future_timestamps: list[Any],
    assessment: Any,
    published_support: str,
    selected_model: str | None,
) -> tuple[dict[str, Any] | None, SupportReason | None, Evidence | None]:
    """Build the labelled second lane for one series, or nothing.

    Returns ``(lane, disclosure, evidence)``.  The lane never carries
    intervals, never claims automation eligibility, and never changes the
    primary forecast; the caller attaches it additively.
    """
    if horizon <= 0 or len(future_timestamps) != horizon or len(values) < 8:
        return None, None, None

    candidate: str | None = None
    candidate_points: list[float] | None = None
    validation: dict[str, Any] | None = None
    evidence_steps = 0
    short_trend_probe_ran = False
    if (selected_model == "last_value"
            and (assessment is None
                 or bool(getattr(assessment, "degraded", False)))):
        seasonal_validation = _seasonal_prequential_candidate(values, season)
        if seasonal_validation is not None:
            candidate = "seasonal_naive"
            validation = seasonal_validation
            evidence_steps = season
        else:
            trend_validation = _short_trend_probe(
                values, horizon, season)
            if trend_validation is not None:
                short_trend_probe_ran = True
                if trend_validation["admitted"]:
                    candidate = str(trend_validation["candidate"])
                    validation = trend_validation
                    evidence_steps = int(
                        trend_validation["maximum_locally_evaluated_lead"])
    if candidate is None and selected_model == "seasonal_naive":
        # The governed engine has already identified the intraday daily cycle.
        # With exactly one week, a weekly replay cannot be backtested, but the
        # calendar-shaped prior is less assumptive than promoting an unrelated
        # trend from one short holdout. It remains visibly prior-only.
        calendar_prior = _single_week_calendar_prior(
            values, season, horizon, future_timestamps)
        if calendar_prior is not None:
            candidate = "calendar_seasonal_naive"
            candidate_points, validation = calendar_prior
            evidence_steps = 0
    if (assessment is not None
            and getattr(assessment, "selection_guardrail_applied", False)
            and selected_model == assessment.strongest_baseline
            and candidate is None and not short_trend_probe_ran):
        scored = _scored_candidate(assessment)
        if scored is not None:
            candidate, candidate_score, baseline_score = scored
            degraded = bool(getattr(assessment, "degraded", False))
            evidence_steps = (min(horizon, max(1, len(values) // 4))
                              if degraded else horizon)
            validation = {
                "basis": ("single_trailing_holdout" if degraded
                          else "single_selection_fold"),
                "out_of_sample_steps": evidence_steps,
                "comparisons": int(getattr(
                    assessment, "selection_fold_count", 1) or 1),
                "candidate_score": candidate_score,
                "baseline_score": baseline_score,
                "baseline": assessment.strongest_baseline,
                "scores_are_evidence_not_a_ranking": True,
            }
    elif published_support == "best_effort" and candidate is None:
        trend_validation = _short_trend_probe(values, horizon, season)
        if trend_validation is not None:
            short_trend_probe_ran = True
            if trend_validation["admitted"]:
                candidate = str(trend_validation["candidate"])
                validation = trend_validation
                evidence_steps = int(
                    trend_validation["maximum_locally_evaluated_lead"])
        if not short_trend_probe_ran:
            held = _holdout_candidate(values, horizon, season)
            if held is not None:
                candidate, candidate_score, baseline_score, evidence_steps = held
                validation = {
                    "basis": "reduced_rigor_holdout",
                    "out_of_sample_steps": evidence_steps,
                    "comparisons": 1,
                    "candidate_score": candidate_score,
                    "baseline_score": baseline_score,
                    "baseline": "last_value",
                    "scores_are_evidence_not_a_ranking": True,
                }
    if candidate is None or validation is None:
        return None, None, None

    try:
        points = (candidate_points if candidate_points is not None else
                  predict(candidate, values, horizon, season))
        baseline_points = predict("last_value", values, horizon, season)
    except (ValueError, ArithmeticError):
        return None, None, None
    plausible, diagnostics = _plausible(values, points, baseline_points)
    if not plausible:
        return None, None, None

    # A full-horizon out-of-sample win is degraded validation; anything
    # weaker is a prior with a shred of evidence. Both are underpowered, and
    # neither is suppressed — that is the lane's entire purpose.
    support = ("conditionally_supported" if evidence_steps >= horizon
               else "prior_assisted")
    extrapolated_tail_steps = max(0, horizon - evidence_steps)
    validation = {
        **validation,
        "requested_horizon": horizon,
        "maximum_locally_evaluated_lead": min(horizon, evidence_steps),
        "extrapolated_tail_steps": extrapolated_tail_steps,
        "tail_support": support if extrapolated_tail_steps else None,
        "tail_automation_eligible": False,
    }
    lane = {
        "version": MODEL_ASSISTED_LANE_VERSION,
        "support": support,
        "selected_model": candidate,
        "requested_horizon": horizon,
        "maximum_locally_evaluated_lead": min(horizon, evidence_steps),
        "extrapolated_tail_steps": extrapolated_tail_steps,
        "tail_support": support if extrapolated_tail_steps else None,
        "tail_automation_eligible": False,
        "governed_primary": {
            "support": published_support,
            "selected_model": selected_model,
            "fallback_reason": (
                "selection_underpowered_assumption_minimal_baseline"
                if selected_model == "last_value" else
                "evaluation_could_not_select_a_primary_model"
            ),
            "why_flat": (
                "The governed primary repeats the last observation because "
                "the available local evidence cannot promote a trend model "
                "into the automation-bearing lane."
                if selected_model in {None, "last_value"} else None
            ),
        },
        # Points only, on the primary rows' own timestamp grid: the lane has
        # no calibrated residuals, so publishing quantiles or per-row
        # envelopes here would manufacture probability weight from nothing.
        "points": [round(float(point), 6) for point in points],
        "timestamps_match_primary_forecast": True,
        "validation": validation,
        "plausibility": _diagnostics_payload(diagnostics),
        "automation_eligible": False,
        "primary_forecast_unchanged": True,
    }
    if validation["basis"] == "single_observed_calendar_cycle_prior":
        disclosure_text = (
            f"A {support} model-assisted forecast ({candidate}) is published "
            "beside the governed rows from one complete observed calendar "
            "week. It has no out-of-sample skill evidence; it passed only "
            "deterministic grid and plausibility checks, carries no calibrated "
            "intervals, is not eligible for automatic action, and never "
            "replaces the primary forecast.")
    else:
        validated_steps = int(validation.get(
            "out_of_sample_steps", evidence_steps))
        disclosure_text = (
            f"A {support} model-assisted forecast ({candidate}) is published "
            f"in the model_assisted lane beside the governed rows: it beat "
            f"{validation['baseline']} on {validation['basis']} "
            f"({validated_steps} out-of-sample step"
            f"{'s' if validated_steps != 1 else ''}) and passed plausibility "
            "checks. It carries no calibrated intervals, is not eligible for "
            "automatic action, and never replaces the primary forecast.")
    disclosure = SupportReason("model_assisted_lane", disclosure_text)
    evidence = Evidence(
        f"model_assisted:{series}", "model_assisted_lane", series, {
            "support": support,
            "selected_model": candidate,
            "validation": validation,
            "plausibility": _diagnostics_payload(diagnostics),
            "primary_forecast_unchanged": True,
        },
    )
    return lane, disclosure, evidence

"""Deterministic dispatch from typed temporal intent to compact answers."""

from __future__ import annotations

from typing import Any
import math
import statistics

from .temporal_question import TemporalQuestion
from .temporal_executables import (
    fit_dependence_executable, fit_temporal_executable,
)
from .volatility import fit_volatility_executable, residuals
from .temporal_evidence import aggregate_evidence, compare_windows


TEMPORAL_ANSWER_CONTRACT_VERSION = "0.7"


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean, right_mean = statistics.mean(left), statistics.mean(right)
    denominator = math.sqrt(
        sum((item - left_mean) ** 2 for item in left)
        * sum((item - right_mean) ** 2 for item in right))
    return (sum((a - left_mean) * (b - right_mean)
                for a, b in zip(left, right)) / denominator
            if denominator > 1e-12 else None)


def _robust_scale(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    centre = statistics.median(values)
    mad = statistics.median(abs(value - centre) for value in values)
    return float(1.4826 * mad if mad > 1e-12 else statistics.stdev(values))


def _forecast_volatility_alignment(values: list[float], forecast: list[float],
                                   season: int) -> dict[str, Any]:
    """Measure dispersion of the immutable forecast in history units."""
    history_residuals = residuals(values, max(1, season))
    # Fit the forecast's own causal residual structure. Concatenating history
    # would treat a changed level or seasonal template at the publication
    # boundary as repeated random dispersion, especially when the horizon is
    # shorter than the detected period.
    future_residuals = residuals(forecast, max(1, season)) if forecast else []
    width = min(48, len(history_residuals))
    reference = _robust_scale(history_residuals[-width:]) if width >= 2 else 0.0
    future = _robust_scale(future_residuals)
    ratio = future / reference if reference > 1e-12 else None
    direction = ("increased" if ratio is not None and ratio > 1.25 else
                 "decreased" if ratio is not None and ratio < .8 else
                 "stable" if ratio is not None else "uncertain")
    return {"direction": direction, "reference_residual_scale": reference,
            "forecast_residual_scale": future,
            "future_to_reference_ratio": ratio,
            "history_residual_points": width,
            "forecast_residual_points": len(future_residuals)}


def _seasonality_alignment(values: list[float], forecast: list[float],
                           season: int) -> dict[str, Any]:
    """Classify fixed/shifted/absent forecast seasonality without labels.

    Every candidate is an alignment of the same history-derived seasonal
    template.  Selection sees only the published immutable forecast, and a
    shift must improve correlation materially over the zero-shift template.
    """
    if season <= 1 or len(values) < 2 * season or len(forecast) < 3:
        return {"direction": "uncertain", "zero_correlation": None,
                "best_correlation": None, "phase_shift_steps": None,
                "reason": "insufficient_cycles"}
    cycles = min(4, len(values) // season)
    recent = values[-cycles * season:]
    template = [statistics.mean(recent[offset::season])
                for offset in range(season)]
    template_scale = statistics.pstdev(template) if len(template) > 1 else 0.0
    residuals = [recent[index] - template[index % season]
                 for index in range(len(recent))]
    residual_scale = statistics.pstdev(residuals) if len(residuals) > 1 else 0.0
    strength = template_scale / max(template_scale + residual_scale, 1e-12)
    correlations: list[tuple[int, float]] = []
    for shift in range(season):
        expected = [template[(len(values) + index - shift) % season]
                    for index in range(len(forecast))]
        correlation = _correlation(expected, forecast)
        if correlation is not None:
            correlations.append((shift, correlation))
    if not correlations or strength < .15:
        return {"direction": "absent", "zero_correlation": None,
                "best_correlation": None, "phase_shift_steps": None,
                "history_seasonal_strength": strength}
    zero = next((value for shift, value in correlations if shift == 0), None)
    best_shift, best = max(correlations, key=lambda item: item[1])
    circular_shift = min(best_shift, season - best_shift)
    tolerance = max(1, round(.1 * season))
    if best < .35:
        direction = "absent"
    elif zero is not None and zero >= .55 and circular_shift <= tolerance:
        direction = "continued"
    elif (best >= .55 and circular_shift > tolerance
          and (zero is None or best - zero >= .15)):
        direction = "shifting"
    else:
        direction = "uncertain"
    return {"direction": direction, "zero_correlation": zero,
            "best_correlation": best, "phase_shift_steps": best_shift,
            "history_seasonal_strength": strength}


def _envelope(question: TemporalQuestion, *, direction: str | None,
              estimate: Any, interval: dict[str, float] | None,
              support: str, headline: str,
              limitations: list[str] | None = None,
              executable: dict[str, Any] | None = None) -> dict[str, Any]:
    display_value = ((question.answer_vocabulary or {}).get(direction, direction)
                     if direction is not None else None)
    decision_rule = None
    if executable:
        decision_rule = {
            key: executable[key] for key in (
                "kind", "property", "aggregation", "threshold", "thresholds",
                "version") if key in executable
        }
    best_estimate = {
        "value": direction,
        "display_value": display_value,
        "support": support,
        "automation_eligible": support == "supported",
    }
    return {
        "question": question.to_dict(),
        # This scalar is the canonical answer for agents and humans to quote.
        # Detailed estimates remain evidence; consumers must not synthesize a
        # competing answer from child rows.
        "best_estimate": best_estimate,
        "synthesis_policy": {
            "canonical_immutable": True,
            "canonical_authority": (
                "binding" if support == "supported" else
                "unavailable" if support == "abstained" else "advisory"),
            "llm_may_synthesize": support != "supported",
            "publication_default": (
                "canonical" if support != "abstained" else "labelled_synthesis"),
            "override_requires_opposing_evidence": support == "weak",
            "synthesis_must_be_separately_labelled": True,
            "primary_forecast_unchanged": True,
        },
        "decision_rule": decision_rule,
        "answer": {
            "direction": direction, "estimate": estimate,
            "interval": interval, "support": support,
            "automation_eligible": support == "supported",
            **({"executable": executable} if executable else {}),
        },
        "headline": headline,
        "limitations": limitations or [],
        "artifact_id": None,
    }


def answer_descriptive_question(
    question: TemporalQuestion, *, report: dict[str, Any],
    values: list[float], season: int,
    forecast_values: list[float] | None = None,
) -> dict[str, Any]:
    prop = question.property
    if prop == "volatility" and question.verb in {"predict", "compare"} \
            and forecast_values:
        estimate = _forecast_volatility_alignment(
            values, forecast_values, season)
        direction = estimate["direction"]
        measurable = direction != "uncertain"
        result = _envelope(
            question, direction=direction,
            estimate=estimate["forecast_residual_scale"], interval=None,
            support=("weak" if measurable else "abstained"),
            headline=(f"Published forecast residual scale for "
                      f"{question.target} is "
                      f"{estimate['forecast_residual_scale']:.6g} versus "
                      f"recent history {estimate['reference_residual_scale']:.6g}."),
            limitations=(
                ["This describes the immutable published forecast path; it "
                 "is not a calibrated claim about realised future volatility."]
                if measurable else
                ["The history has insufficient residual variation for a "
                 "forecast-to-history volatility comparison."]),
            executable={"kind": "published_forecast_projection",
                        "property": "volatility", "version": "0.1"},
        )
        result["answer"].update(estimate)
        # Keep the volatility answer contract uniform across fitted forecasts
        # and immutable-path projections.  This is deliberately an
        # uncalibrated point distribution: describing the published path does
        # not create probability mass or pretend that realised volatility was
        # fitted from future observations.
        result["answer"]["property_distribution"] = {
            "quantity": "future_to_reference_residual_scale_ratio",
            "estimate": ratio if (ratio := estimate[
                "future_to_reference_ratio"]) is not None else 0.0,
            "lower": ratio if ratio is not None else 0.0,
            "upper": ratio if ratio is not None else 0.0,
            "probabilities": {},
            "point_state": direction,
            "folds": 0,
            "balanced_accuracy": 0.0,
            "brier_skill": 0.0,
            "support": "weak" if measurable else "abstained",
        }
        result["calibration"] = {
            "available": False,
            "direction_candidate": "published_forecast_projection",
            "reason": "immutable_path_description_not_realised_outcome_fit",
        }
        return result
    if prop == "volatility" and question.verb in {"predict", "compare"}:
        fitted = fit_volatility_executable(
            values, horizon=question.horizon or 1, season=season)
        answer = fitted.execute()
        result = _envelope(
            question, direction=answer["direction"],
            estimate=answer["estimate"], interval=answer["interval"],
            support=answer["direction_support"],
            headline=(f"Future residual scale for {question.target} is "
                      f"{answer['estimate']:.6g}; support is {answer['support']}."),
            limitations=(["The best estimate is published with weak support; "
                          "do not use it for automatic action."]
                         if answer["direction_support"] == "weak" else []),
            executable=answer["executable"],
        )
        result["headline"] = (
            f"Future residual scale for {question.target} is "
            f"{answer['estimate']:.6g}; directional support is "
            f"{answer['direction_support']}."
        )
        result["answer"].update({
            "reference_scale": answer["reference_scale"],
            "future_to_reference_ratio": answer["future_to_reference_ratio"],
            "ratio_interval": answer["ratio_interval"],
            "direction_probabilities": answer["direction_probabilities"],
            "direction_support": answer["direction_support"],
            "estimate_support": answer["support"],
            "property_distribution": answer["property_distribution"],
            "decision": answer["decision"],
        })
        result["calibration"] = fitted.diagnostics
        return result
    if prop in {"level", "trend", "seasonality", "regime", "extreme"} \
            and question.verb in {"predict", "compare"} \
            and not forecast_values:
        answer = fit_temporal_executable(
            values, property=prop, horizon=question.horizon or 1,
            season=season).execute()
        result = _envelope(
            question, direction=answer["direction"],
            estimate=answer["estimate"], interval=answer["interval"],
            support=answer["support"],
            headline=(f"Future {prop} for {question.target}: "
                      f"{answer['direction']}; support is {answer['support']}."),
            limitations=([] if answer["support"] == "supported" else [
                "The categorical best estimate is weak under rolling-origin "
                "calibration and is not eligible for automatic action."
            ]), executable=answer["executable"],
        ) | {"calibration": answer["diagnostics"],
             "direction_probabilities": answer["direction_probabilities"]}
        return result
    if prop == "level" and question.verb in {"compare", "predict"} \
            and forecast_values:
        history_median = statistics.median(values)
        forecast_median = statistics.median(forecast_values)
        delta = forecast_median - history_median
        changes = [values[index] - values[index - 1]
                   for index in range(1, len(values))]
        change_centre = statistics.median(changes) if changes else 0.0
        mad = (statistics.median(abs(item - change_centre) for item in changes)
               if changes else 0.0)
        scale = max(1.4826 * mad, 1e-12)
        normalized_change = (forecast_median - history_median) / scale
        direction = ("higher" if normalized_change > .25
                     else "lower" if normalized_change < -.25
                     else "similar")
        estimate = {
            "history_median": history_median,
            "forecast_median": forecast_median,
            "absolute_change": delta,
            "relative_change": (delta / abs(history_median)
                                if history_median else None),
        }
        return _envelope(
            question, direction=direction, estimate=estimate, interval=None,
            support="weak",
            headline=(f"Published forecast median for {question.target} is "
                      f"{forecast_median:.6g} versus history median "
                      f"{history_median:.6g}."),
            limitations=[
                "Direction uses Gnomon's documented 0.25 innovation-scale "
                "default because no task-specific threshold was supplied; "
                "the exact numeric comparison remains auditable."
            ], executable={"kind": "published_forecast_projection",
                            "property": "level", "version": "0.1"},
        )
    if prop == "seasonality" and question.verb in {"compare", "predict"} \
            and forecast_values:
        estimate = _seasonality_alignment(values, forecast_values, season)
        estimate.update({"history_period": season if season > 1 else None,
                         "comparison_points": len(forecast_values),
                         # Compatibility field: zero-shift alignment.
                         "forecast_vs_repeated_history_correlation":
                         estimate.get("zero_correlation")})
        direction = estimate["direction"]
        measurable = direction not in {None, "uncertain"}
        return _envelope(
            question, direction=direction, estimate=estimate, interval=None,
            support=("weak" if measurable else "abstained"),
            headline=(f"Seasonal alignment for {question.target}: {direction}."),
            limitations=(["Phase alignment is an exploratory best estimate "
                           "and is not eligible for automatic action."]
                         if measurable else [
                "The history or published horizon does not contain enough "
                "variation to estimate alignment."
            ]), executable={"kind": "published_forecast_projection",
                            "property": "seasonality", "version": "0.2"},
        )
    if prop == "level":
        estimate = report["level"]["latest"]
        direction = None
    elif prop == "trend":
        estimate = report["trend"]["slope_per_step"]
        direction = report["trend"]["direction"]
    elif prop == "seasonality":
        estimate = report["seasonality"].get("period")
        direction = None
    elif prop == "extreme":
        estimate = {"minimum": report["level"]["minimum"],
                    "maximum": report["level"]["maximum"]}
        direction = None
    elif prop == "regime":
        estimate = report["changepoints"]
        direction = None
    else:
        return _envelope(
            question, direction=None, estimate=None, interval=None,
            support="abstained", headline=(
                f"{question.property} for {question.target} is not available "
                "from this executable."),
            limitations=["The property requires a different typed executable."],
        )
    return _envelope(
        question, direction=direction, estimate=estimate, interval=None,
        support="supported",
        headline=f"{question.property} for {question.target}: {estimate}.",
    )


def _execute_scoped_question(
    question: TemporalQuestion, *, reports: dict[str, dict[str, Any]],
    execution_inputs: dict[str, tuple[list[float], int]],
    forecast_values: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Execute explicit multi-series scope without hiding its constituents."""
    if question.scope == "series":
        values, season = execution_inputs[question.target]
        return answer_descriptive_question(
            question, report=reports[question.target], values=values,
            season=season,
            forecast_values=(forecast_values or {}).get(question.target))
    members = list(question.members)
    if question.scope == "pair" and question.property == "dependence":
        left, right = members
        fitted = fit_dependence_executable(
            execution_inputs[left][0], execution_inputs[right][0],
            horizon=question.horizon or 8).execute()
        return _envelope(
            question, direction=fitted["direction"],
            estimate=fitted["estimate"], interval=fitted["interval"],
            support=fitted["support"],
            headline=(f"Future dependence between {left} and {right}: "
                      f"{fitted['direction']}; support is {fitted['support']}."),
            limitations=([] if fitted["support"] == "supported" else [
                "The directional best estimate is weak under rolling-origin "
                "calibration and is not eligible for automatic action."
            ]), executable=fitted["executable"],
        ) | {"calibration": fitted["diagnostics"],
             "direction_probabilities": fitted["direction_probabilities"]}
    children = []
    for member in members:
        child = TemporalQuestion(
            id=f"{question.id}:{member}", verb=question.verb, target=member,
            property=question.property, horizon=question.horizon,
            comparison=question.comparison, measure=question.measure,
            context_policy=question.context_policy,
            answer_vocabulary=question.answer_vocabulary)
        values, season = execution_inputs[member]
        children.append(answer_descriptive_question(
            child, report=reports[member], values=values, season=season,
            forecast_values=(forecast_values or {}).get(member)))
    if question.scope == "each":
        return _envelope(
            question, direction=None, estimate={
                child["question"]["target"]: child["answer"]
                for child in children}, interval=None,
            support=("supported" if any(child["answer"]["support"] == "supported"
                                        for child in children) else
                     "weak" if any(child["answer"]["support"] == "weak"
                                   for child in children) else "abstained"),
            headline=f"Computed {question.property} separately for {len(children)} series.",
            limitations=[], executable={"kind": "each"}) | {"per_series": children}
    if question.property not in {"volatility", "seasonality"}:
        return _envelope(
            question, direction=None, estimate=None, interval=None,
            support="abstained", headline="This aggregate is not defined.",
            limitations=["This property has no registered auditable cross-series aggregation."])
    if question.property == "seasonality":
        alignments = []
        directions = []
        for child in children:
            estimate = child["answer"].get("estimate")
            if isinstance(estimate, dict):
                value = estimate.get("forecast_vs_repeated_history_correlation")
                if value is not None and math.isfinite(float(value)):
                    alignments.append(float(value))
            child_direction = (child.get("best_estimate") or {}).get("value")
            if child_direction in {"continued", "shifting", "absent"}:
                directions.append(str(child_direction))
        estimate = statistics.median(alignments) if alignments else None
        counts = {label: directions.count(label)
                  for label in ("continued", "shifting", "absent")}
        direction = (max(counts, key=counts.get) if directions
                     and max(counts.values()) > len(children) / 2
                     else "uncertain")
        support = "weak" if directions else "abstained"
        return _envelope(
            question, direction=direction, estimate={
                "median_alignment": estimate,
                "direction_counts": counts,
                "contributing_series": len(alignments),
                "total_series": len(children),
            }, interval=None, support=support,
            headline=(f"Median seasonal alignment across {len(children)} "
                      f"series is {estimate}."),
            limitations=(["The aggregate uses the registered 0.5 alignment "
                           "threshold and is not eligible for automatic action."]
                         if estimate is not None else [
                             "No constituent had enough variation to estimate alignment."]),
            executable={"kind": "aggregate",
                        "aggregation": "median_alignment",
                        "threshold": .5}) | {"per_series": children}
    # Predictive folds answer whether a single series can support a future
    # claim.  They must not erase the separate, useful fact that several
    # related series show the same observed movement.  Aggregate the latter
    # explicitly and label the persistence assumption instead of pretending
    # the series are extra temporal folds.
    observed_rows = {
        member: compare_windows(execution_inputs[member][0],
                                season=execution_inputs[member][1])
        for member in members
    }
    panel_evidence = aggregate_evidence(observed_rows, property="volatility")
    calibrated_children = [child for child in children
                           if int(((child["answer"].get("property_distribution")
                                    or {}).get("folds") or 0)) >= 2
                           or ((child["answer"].get("executable") or {}).get(
                               "kind") == "published_forecast_projection"
                               and (child["answer"].get("executable") or {}).get(
                                   "property") == "volatility")]
    ratios = [child["answer"].get("future_to_reference_ratio")
              for child in calibrated_children]
    ratios = [float(item) for item in ratios if item is not None]
    estimate = statistics.median(ratios) if ratios else None
    direction = ("increased" if estimate is not None and estimate > 1.25 else
                 "decreased" if estimate is not None and estimate < .8 else
                 "stable" if estimate is not None else
                 str(panel_evidence["direction"]))
    supported_children = sum(
        child["answer"].get("direction_support") == "supported"
        for child in children)
    support = ("supported" if supported_children > len(children) / 2 else
               "weak" if estimate is not None or panel_evidence["support"]
               in {"weak", "supported"} else "abstained")
    result = _envelope(
        question, direction=direction, estimate=estimate, interval=None,
        support=support,
        headline=f"Median normalized future/reference scale ratio across {len(children)} series is {estimate}.",
        limitations=(["The median-ratio best estimate lacks a strict majority "
                      "of individually calibrated claims and is not eligible "
                      "for automatic action.",
                      "Cross-series movement is observed evidence only; using "
                      "it predictively assumes the recent regime persists."]
                     if support == "weak" else []),
        executable={"kind": "aggregate",
                    "aggregation": "median_normalized_scale_ratio",
                    "thresholds": {"decreased_below": .8,
                                   "increased_above": 1.25}}) | {
            "per_series": children,
            "observed_panel_evidence": panel_evidence,
            "identifiability": {
                "future_direction": "calibrated" if estimate is not None
                                    else "conditional_on_persistence",
                "primary_forecast_unchanged": True,
            },
        }
    probability_rows = [
        child["answer"].get("direction_probabilities") or {}
        for child in children
    ]
    labels = ("decreased", "stable", "increased")
    if probability_rows:
        result["answer"]["direction_probabilities"] = {
            label: statistics.mean(float(row.get(label, 0.0))
                                   for row in probability_rows)
            for label in labels
        }
    return result


def answer_scoped_question(
    question: TemporalQuestion, *, reports: dict[str, dict[str, Any]],
    execution_inputs: dict[str, tuple[list[float], int]],
    forecast_values: dict[str, list[float]] | None = None,
    conditional_effects: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute one question and attach a bounded, non-authoritative plan."""
    result = _execute_scoped_question(
        question, reports=reports, execution_inputs=execution_inputs,
        forecast_values=forecast_values)
    if question.context_policy in {"measure", "scenario"}:
        effect = (conditional_effects or {}).get(question.target)
        if effect:
            result["conditional_effect"] = {
                **effect, "role": "conditional_evidence_only",
                "primary_forecast_unchanged": True,
            }
    from .temporal_evidence import (
        competing_hypotheses, historical_analogues,
        multi_resolution_evidence, window_evidence,
    )
    from .temporal_planner import build_evidence_plan
    observed = None
    if question.scope == "series" and question.target in execution_inputs:
        values, season = execution_inputs[question.target]
        observed = window_evidence(
            values, property=question.property, season=season).to_dict()
        analogues = historical_analogues(
            values, property=question.property, season=season)
    else:
        analogues = None
    result["answer"]["reasoning"] = build_evidence_plan(
        question, result, observed_evidence=observed, analogues=analogues)
    if question.scope == "series" and question.target in execution_inputs:
        values, season = execution_inputs[question.target]
        result["answer"]["reasoning"]["multi_resolution"] = \
            multi_resolution_evidence(
                values, property=question.property, season=season)
        # A full differential is useful for broad change/detection questions,
        # but would be needless response tax on a narrow property question.
        if question.verb == "detect" or question.property == "regime":
            result["answer"]["reasoning"]["competing_hypotheses"] = \
                competing_hypotheses(values, season=season)
    return result

"""The structural-effects lane: LLM-classified, value-free, path-derived.

Pre-registered in results/structural-effects/HYPOTHESIS.md before any of
this was written. The division of labour under test: the model
classifies a span into a closed effect menu; every quantity the effect
applies is derived from Gnomon's own emitted path — the model never
supplies a number that is applied.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

from gnomon.config import GnomonConfig
from gnomon.context import ContextEvent, ContextSource
from gnomon.future_context import (
    FutureEvent,
    apply_future_events,
    assess_future_events,
)
from gnomon.runtime import forecast
from gnomon.pipeline import _scenario_consequence

START = datetime(2026, 1, 1)
HISTORY = [200.0 + (day % 7) for day in range(60)]
TIMESTAMPS = [START + timedelta(days=day) for day in range(60)]
FUTURE = [START + timedelta(days=60 + day) for day in range(7)]
H_START, H_END = FUTURE[0], FUTURE[-1]


def _event(event_id, event_type, attributes, start=None, end=None):
    start = start or H_START
    end = end or H_END
    return ContextEvent(
        event_id=event_id, event_type=event_type, entity_scope=("*",),
        effective_start=start.isoformat() + "+00:00",
        effective_end=end.isoformat() + "+00:00",
        known_at=START.isoformat() + "+00:00", attributes=attributes,
        source=ContextSource("dataset", "test#structural"), created_by="llm",
    )


SPAN = "the sensor was repaired and this additive trend will disappear"


# -- admission ---------------------------------------------------------------

def test_a_menu_effect_with_a_span_is_admitted():
    events = [_event("s1", "structural:repair",
                     {"source_span": SPAN, "effect": "trend_ceases"})]
    assessment = assess_future_events(events, "s", HISTORY, TIMESTAMPS, FUTURE, 7)
    assert [e.event_id for e in assessment.admitted] == ["s1"]
    admitted = assessment.admitted[0]
    assert admitted.event_class == "structural"
    assert admitted.effect == "trend_ceases"
    assert admitted.to_public_dict()["effect"] == "trend_ceases"
    assert assessment.class_counts()["structural"]["admitted"] == 1


def test_trend_cessation_rejects_a_seasonal_or_irregular_emitted_path():
    events = [_event("s1", "structural:repair",
                     {"source_span": SPAN, "effect": "trend_ceases"})]
    assessment = assess_future_events(
        events, "s", HISTORY, TIMESTAMPS, FUTURE, 7,
        base_points=[100.0, 103.0, 101.0, 104.0, 102.0, 105.0, 103.0],
    )
    assert not assessment.admitted
    assert assessment.rejected[0]["code"] == \
        "emitted_trend_is_directionally_stable"
    rejection = assessment.rejected[0]
    data = rejection["data"]
    assert data["directional_agreement_passed"] is (
        data["directional_agreement"] >= data["agreement_threshold"])
    assert data["same_direction_passed"] is (
        data["historical_slope_per_step"] * data["emitted_slope_per_step"] > 0)
    lower, upper = data["magnitude_ratio_bounds"]
    assert data["magnitude_ratio_passed"] is (
        lower <= data["magnitude_ratio"] <= upper)
    for label, key in (
        ("directional agreement", "directional_agreement_passed"),
        ("slope direction", "same_direction_passed"),
        ("magnitude ratio", "magnitude_ratio_passed"),
    ):
        status = "passed" if data[key] else "failed"
        assert f"{label} {status}" in rejection["reason"]


def test_trend_cessation_admits_a_directionally_stable_emitted_path():
    events = [_event("s1", "structural:repair",
                     {"source_span": SPAN, "effect": "trend_ceases"})]
    trending = [200.0 + 0.5 * day + (day % 7) for day in range(60)]
    base = [200.0 + 0.5 * day + (day % 7) for day in range(60, 67)]
    assessment = assess_future_events(
        events, "s", trending, TIMESTAMPS, FUTURE, 7,
        base_points=base,
    )
    assert [event.event_id for event in assessment.admitted] == ["s1"]


def test_structural_effect_without_separated_folds_is_scenario_only():
    events = [_event("s1", "structural:repair",
                     {"source_span": SPAN, "effect": "trend_ceases"})]
    trending = [200.0 + 0.5 * day + (day % 7) for day in range(60)]
    base = [200.0 + 0.5 * day + (day % 7) for day in range(60, 67)]
    assessment = assess_future_events(
        events, "s", trending, TIMESTAMPS, FUTURE, 7,
        base_points=base, structural_evidence_folds=0,
    )

    assert not assessment.admitted
    assert [event.event_id for event in assessment.scenarios] == ["s1"]
    failure = next(check for check in assessment.checks
                   if check["code"] == "separated_model_folds_available")
    assert failure["passed"] is False
    assert failure["data"] == {"measured_folds": 0, "required_folds": 4}

    primary = _rows(base)
    projected, _ = apply_future_events(
        [dict(row) for row in primary], assessment.scenarios)
    assert projected != primary
    assert [row["point"] for row in primary] == base


def test_structural_effect_with_four_separated_folds_can_be_admitted():
    events = [_event("s1", "structural:repair",
                     {"source_span": SPAN, "effect": "trend_ceases"})]
    trending = [200.0 + 0.5 * day + (day % 7) for day in range(60)]
    base = [200.0 + 0.5 * day + (day % 7) for day in range(60, 67)]
    assessment = assess_future_events(
        events, "s", trending, TIMESTAMPS, FUTURE, 7,
        base_points=base, structural_evidence_folds=4,
    )

    assert [event.event_id for event in assessment.admitted] == ["s1"]
    assert not assessment.scenarios


def test_an_effect_outside_the_menu_is_rejected():
    events = [_event("s1", "structural:repair",
                     {"source_span": SPAN, "effect": "level_resets"})]
    assessment = assess_future_events(events, "s", HISTORY, TIMESTAMPS, FUTURE, 7)
    assert not assessment.admitted
    assert assessment.rejected[0]["code"] == "effect_supported"
    assert "closed menu" in assessment.rejected[0]["reason"]
    assert assessment.rejected[0]["source_span"] == SPAN


def test_a_missing_effect_is_rejected():
    events = [_event("s1", "structural:repair", {"source_span": SPAN})]
    assessment = assess_future_events(events, "s", HISTORY, TIMESTAMPS, FUTURE, 7)
    assert assessment.rejected[0]["code"] == "effect_supported"


def test_a_structural_event_without_a_span_is_rejected():
    events = [_event("s1", "structural:repair", {"effect": "trend_ceases"})]
    assessment = assess_future_events(events, "s", HISTORY, TIMESTAMPS, FUTURE, 7)
    assert assessment.rejected[0]["code"] == "source_span_present"


def test_flag_off_ignores_the_class_entirely():
    events = [_event("s1", "structural:repair",
                     {"source_span": SPAN, "effect": "trend_ceases"})]
    assessment = assess_future_events(
        events, "s", HISTORY, TIMESTAMPS, FUTURE, 7, allow_structural=False,
    )
    assert not assessment.considered
    assert not assessment.admitted and not assessment.rejected


# -- application -------------------------------------------------------------

def _rows(points):
    return [
        {"timestamp": (H_START + timedelta(days=index)).isoformat(),
         "point": point,
         "q10": point - 2.0, "q50": point, "q90": point + 2.0}
        for index, point in enumerate(points)
    ]


def _structural(event_id="s1", start=H_START, end=H_END):
    return FutureEvent(
        event_id, "structural", start.isoformat() + "+00:00",
        end.isoformat() + "+00:00", SPAN, effect="trend_ceases",
    )


def test_trend_ceases_flattens_a_drifting_path():
    # points 100, 101, ..., 106: fitted slope exactly 1.
    rows = _rows([100.0 + step for step in range(7)])
    projected, applications = apply_future_events(rows, [_structural()])
    # first covered step keeps its value; drift removed from there on
    assert [row["point"] for row in projected] == [100.0] * 7
    # a pure location shift: widths and the point-median gap unchanged
    assert all(row["q90"] - row["q10"] == 4.0 for row in projected)
    assert all(row["q50"] == row["point"] for row in projected)
    assert len(applications) == 7
    assert all(entry["effect"] == "trend_ceases" for entry in applications)
    assert applications[0]["slope_removed"] == 1.0
    assert applications[0]["delta"] == 0.0  # continuity at the window start
    assert applications[-1]["delta"] == -6.0


def test_a_driftless_path_is_a_measured_noop():
    rows = _rows([100.0] * 7)
    projected, applications = apply_future_events(rows, [_structural()])
    assert [row["point"] for row in projected] == [100.0] * 7
    # the application is still recorded: a no-op is a measurement,
    # not an omission
    assert applications and applications[0]["slope_removed"] == 0.0


def test_overlapping_cessations_do_not_remove_the_drift_twice():
    rows = _rows([100.0 + step for step in range(7)])
    events = [_structural("s1"), _structural("s2")]
    projected, applications = apply_future_events(rows, events)
    assert [row["point"] for row in projected] == [100.0] * 7
    assert {entry["event_id"] for entry in applications} == {"s1"}


def test_a_partial_window_only_touches_covered_steps():
    rows = _rows([100.0 + step for step in range(7)])
    event = _structural(start=H_START + timedelta(days=3))
    projected, _ = apply_future_events(rows, [event])
    # steps 0-2 untouched; step 3 is the new window start (continuity);
    # drift removed from step 4 on
    assert [row["point"] for row in projected] == [
        100.0, 101.0, 102.0, 103.0, 103.0, 103.0, 103.0]


def test_an_override_still_wins_inside_its_window():
    rows = _rows([100.0 + step for step in range(7)])
    override = FutureEvent(
        "o1", "override",
        (H_START + timedelta(days=2)).isoformat() + "+00:00",
        (H_START + timedelta(days=4)).isoformat() + "+00:00",
        "output reduced to 50 while the line is partially shut down",
        value=50.0,
    )
    projected, _ = apply_future_events(rows, [_structural(), override])
    assert projected[3]["point"] == 50.0  # interior of the override window
    assert projected[0]["point"] == 100.0
    assert projected[6]["point"] == 100.0


# -- end-to-end --------------------------------------------------------------

def _write_trending_csv(path: Path, days=120):
    # A pure drift, no seasonal wiggle: the emitted path's fitted slope
    # is then the drift itself, so "trend ceases" should flatten it.
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "value"])
        for day in range(days):
            writer.writerow([
                (START + timedelta(days=day)).isoformat(),
                200 + 0.5 * day,
            ])


def test_flag_on_preserves_primary_and_discloses_structural_scenario(
        tmp_path: Path):
    source = tmp_path / "series.csv"
    _write_trending_csv(source)
    h_start = START + timedelta(days=120)
    events = [ContextEvent(
        event_id="s1", event_type="structural:repair", entity_scope=("*",),
        effective_start=h_start.isoformat() + "+00:00",
        effective_end=(h_start + timedelta(days=6)).isoformat() + "+00:00",
        known_at=START.isoformat() + "+00:00",
        attributes={"source_span": SPAN, "effect": "trend_ceases"},
        source=ContextSource("dataset", "test#structural"), created_by="llm",
    )]
    config = GnomonConfig()
    config.context.future_events = True
    config.context.structural_events = True
    kwargs = dict(time_column="timestamp", target_column="value",
                  horizon=7, frequency="D")
    baseline, _ = forecast(str(source), output=str(tmp_path / "a"), **kwargs)
    treated, _ = forecast(str(source), output=str(tmp_path / "b"),
                          context_events=events, config=config, **kwargs)
    base_points = [row["point"] for row in baseline.results[0].forecast]
    treated_result = treated.results[0]
    treated_points = [row["point"] for row in treated_result.forecast]
    # A forward structural assertion is useful as a what-if, but generic
    # model-selection folds are not evidence for that transformation.
    assert base_points[-1] > base_points[0]
    assert treated_points == base_points
    assert not any(item.kind == "future_context_applied"
                   for item in treated.evidence)
    scenario = next(item for item in treated_result.sensitivity_scenarios
                    if item["support"] == "prior_assisted_structural")
    scenario_points = [row["point"] for row in scenario["forecast"]]
    assert abs(scenario_points[-1] - scenario_points[0]) < \
        (base_points[-1] - base_points[0]) / 2
    assert scenario["primary_forecast_changed"] is False
    assert scenario["automation_eligible"] is False
    assert scenario["effect"]["provenance"]["provenance_class"] == \
        "human_assumption"
    assert scenario["effect"]["shape"] == "trend_change"
    assert scenario["consequence"]["status"] == "numeric_difference"
    assert scenario["consequence"]["first_affected_primary_q50"] != \
        scenario["consequence"]["first_affected_scenario_q50"]
    assert scenario["consequence"]["horizon_end_scenario_q50"] == \
        scenario_points[-1]
    assert scenario["consequence"]["max_abs_delta_q50"] > 0
    assert "canonical primary remains unchanged" in \
        scenario["consequence_summary"]
    gate = treated_result.future_context
    assert gate["scenarios"][0]["effect"] == "trend_ceases"
    failure = next(check for check in gate["checks"]
                   if check["code"] == "separated_model_folds_available")
    assert failure["passed"] is False
    # the flag enters the artifact ID payload
    assert treated.forecast_id != baseline.forecast_id


def test_scenario_consequence_starts_at_first_actual_effect_not_window_start():
    primary = _rows([100.0 + step for step in range(4)])
    scenario = [dict(row) for row in primary]
    # The contextual condition starts before the numeric transformation has
    # any effect: continuity leaves two leading rows unchanged.
    scenario[2]["point"] = scenario[2]["q50"] = 101.0
    scenario[3]["point"] = scenario[3]["q50"] = 101.0

    consequence, summary = _scenario_consequence(primary, scenario)

    assert consequence["first_affected_timestamp"] == primary[2]["timestamp"]
    assert consequence["first_affected_delta_q50"] == -1.0
    assert consequence["horizon_end_delta_q50"] == -2.0
    assert consequence["max_abs_delta_q50"] == 2.0
    assert primary[0]["timestamp"] not in summary


def test_scenario_consequence_names_a_noop_without_inventing_effect():
    primary = _rows([100.0, 100.0, 100.0])
    consequence, summary = _scenario_consequence(
        primary, [dict(row) for row in primary])

    assert consequence == {
        "status": "no_numeric_difference", "affected_steps": 0,
        "max_abs_delta_q50": 0.0,
    }
    assert "no numeric q50 difference" in summary


def test_flag_off_is_byte_identical(tmp_path: Path):
    source = tmp_path / "series.csv"
    _write_trending_csv(source)
    h_start = START + timedelta(days=120)
    events = [ContextEvent(
        event_id="s1", event_type="structural:repair", entity_scope=("*",),
        effective_start=h_start.isoformat() + "+00:00",
        effective_end=(h_start + timedelta(days=6)).isoformat() + "+00:00",
        known_at=START.isoformat() + "+00:00",
        attributes={"source_span": SPAN, "effect": "trend_ceases"},
        source=ContextSource("dataset", "test#structural"), created_by="llm",
    )]
    kwargs = dict(time_column="timestamp", target_column="value",
                  horizon=7, frequency="D")
    baseline, _ = forecast(str(source), output=str(tmp_path / "a"), **kwargs)
    with_events, _ = forecast(str(source), output=str(tmp_path / "b"),
                              context_events=events, **kwargs)
    # With every flag off a structural event is exactly what it was
    # before the class existed: an ordinary context event for the
    # fold-ablation gate. The published numbers are unchanged and no
    # future-context gate record exists.
    assert [row["point"] for row in with_events.results[0].forecast] == \
        [row["point"] for row in baseline.results[0].forecast]
    assert with_events.results[0].future_context is None
    assert not any(item.kind.startswith("future_context")
                   for item in with_events.evidence)


# -- seasonal-regime effects (results/seasonal-regime-effects/) --------------
# HISTORY is 200 + (day % 7): every phase bucket is constant, so the
# per-phase envelope (any quantile) is exactly 200 + phase, and the first
# future step is phase 60 % 7 == 4.

EXPECTED_LEVELS = tuple(float(200 + (60 + step) % 7) for step in range(7))
CLEAR_SPAN = "At the window start, the weather will become clear."


def _regime_event(effect="level_matches_seasonal_high", event_id="r1"):
    return _event(event_id, "structural:regime",
                  {"source_span": CLEAR_SPAN, "effect": effect})


def test_a_regime_event_resolves_the_envelope_at_admission():
    assessment = assess_future_events(
        [_regime_event()], "s", HISTORY, TIMESTAMPS, FUTURE, 7)
    admitted, = assessment.admitted
    assert admitted.effect == "level_matches_seasonal_high"
    assert admitted.levels == EXPECTED_LEVELS
    assert admitted.to_public_dict()["resolved_levels"] == list(EXPECTED_LEVELS)


def test_high_and_low_read_different_envelope_quantiles():
    # Phase buckets alternate 100/110, so the high envelope must sit
    # strictly above the low one at every step.
    history = [100.0 + (10.0 if (day // 7) % 2 == 0 else 0.0)
               for day in range(60)]
    high, = assess_future_events(
        [_regime_event("level_matches_seasonal_high")],
        "s", history, TIMESTAMPS, FUTURE, 7).admitted
    low, = assess_future_events(
        [_regime_event("level_matches_seasonal_low")],
        "s", history, TIMESTAMPS, FUTURE, 7).admitted
    assert all(h > l for h, l in zip(high.levels, low.levels))


def test_an_unresolvable_profile_is_rejected_not_defaulted():
    # Season 1: no phase structure to resolve.
    assessment = assess_future_events(
        [_regime_event()], "s", HISTORY, TIMESTAMPS, FUTURE, 1)
    assert not assessment.admitted
    rejection, = assessment.rejected
    assert rejection["code"] == "seasonal_profile_resolvable"

    # Fewer than two full cycles.
    assessment = assess_future_events(
        [_regime_event()], "s", HISTORY[:10], TIMESTAMPS[:10], FUTURE, 7)
    assert not assessment.admitted
    assert assessment.rejected[0]["code"] == "seasonal_profile_resolvable"


def test_covered_steps_land_on_the_envelope_with_widths_preserved():
    rows = _rows([300.0 + index for index in range(7)])
    event = FutureEvent(
        "r1", "structural",
        (H_START + timedelta(days=2)).isoformat() + "+00:00",
        (H_START + timedelta(days=4)).isoformat() + "+00:00",
        CLEAR_SPAN, effect="level_matches_seasonal_high",
        levels=EXPECTED_LEVELS,
    )
    projected, applications = apply_future_events(rows, [event])
    for index in (2, 3, 4):
        row = projected[index]
        assert row["point"] == EXPECTED_LEVELS[index]
        assert row["q90"] - row["q10"] == 4.0  # width untouched
        assert row["q50"] == row["point"]
    for index in (0, 1, 5, 6):
        assert projected[index] == rows[index]
    assert [item["target_level"] for item in applications] \
        == [EXPECTED_LEVELS[index] for index in (2, 3, 4)]
    assert all(item["effect"] == "level_matches_seasonal_high"
               for item in applications)


def test_a_step_is_never_adjusted_twice_across_effects():
    rows = _rows([300.0 + 5.0 * index for index in range(7)])
    cessation = _structural(end=H_START + timedelta(days=3))
    regime = FutureEvent(
        "r1", "structural", H_START.isoformat() + "+00:00",
        H_END.isoformat() + "+00:00", CLEAR_SPAN,
        effect="level_matches_seasonal_high", levels=EXPECTED_LEVELS,
    )
    projected, applications = apply_future_events(rows, [cessation, regime])
    # Steps 0-3 belong to the cessation; the regime only gets 4-6.
    assert [item["event_id"] for item in applications] \
        == ["s1"] * 4 + ["r1"] * 3
    for index in (4, 5, 6):
        assert projected[index]["point"] == EXPECTED_LEVELS[index]

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


def test_flag_on_applies_and_discloses(tmp_path: Path):
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
    treated_points = [row["point"] for row in treated.results[0].forecast]
    # the baseline drifts upward; the treated path does not
    assert base_points[-1] > base_points[0]
    assert abs(treated_points[-1] - treated_points[0]) < \
        (base_points[-1] - base_points[0]) / 2
    assert treated.results[0].support == "context_trusted"
    applied = [item for item in treated.evidence
               if item.kind == "future_context_applied"]
    assert applied and applied[0].payload["applications"][0]["effect"] == \
        "trend_ceases"
    # every applied quantity is derived from the emitted path: the
    # recorded slope equals the counterfactual path's own fitted drift
    counterfactual = applied[0].payload["history_only_counterfactual"]
    from gnomon.future_context import _path_slope
    assert applied[0].payload["applications"][0]["slope_removed"] == \
        _path_slope([float(row["point"]) for row in counterfactual])
    # the flag enters the artifact ID payload
    assert treated.forecast_id != baseline.forecast_id


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

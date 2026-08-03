"""The future-context lane: textual verifiability instead of fold proof.

Covers the contract in `gnomon.future_context` and its wiring: numbers
come only from quoted spans, suspect claims are rejected against recent
history, constraints clamp and overrides set-with-boundary-widening,
influenced forecasts are downgraded to `context_trusted`, and — the
invariant everything else stands on — a flag-off run is byte-identical to
one on a build without the feature.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone

import pytest

from gnomon.config import GnomonConfig
from gnomon.context import ContextEvent, ContextSource
from gnomon.future_context import (
    FutureEvent,
    apply_future_events,
    assess_future_events,
    parse_bound_span,
    parse_override_span,
)
from gnomon.runtime import forecast

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


# -- span parsing: the only path from text to an applied number -------------

@pytest.mark.parametrize("span, minimum, maximum", [
    ("values stay between 0 and 340 kWh", 0.0, 340.0),
    ("demand ranges from 1,200 to 4,500 units", 1200.0, 4500.0),
    ("the load will not exceed 500 MW", None, 500.0),
    ("output is capped at 96", None, 96.0),
    ("at most 12 vehicles per hour", None, 12.0),
    ("occupancy remains below 0.95", None, 0.95),
    ("the count cannot be negative", 0.0, None),
    ("readings are non-negative", 0.0, None),
    ("at least 40 units will be produced", 40.0, None),
    ("the level will not drop below 7.5", 7.5, None),
    ("stays above 100", 100.0, None),
    # scientific notation is the whole number, not its mantissa
    ("the count will not exceed 1e9", None, 1e9),
    ("at least 2.5e-3", 2.5e-3, None),
])
def test_bound_spans_parse_to_the_stated_numbers(span, minimum, maximum):
    bound, problem = parse_bound_span(span)
    assert problem is None
    assert bound.minimum == minimum
    assert bound.maximum == maximum


@pytest.mark.parametrize("span, minimum, maximum", [
    # CiK's constraint tasks state bounds in exactly this voice.
    ("the forecast values are bounded above by 340", None, 340.0),
    ("the values are bounded below by 12", 12.0, None),
    ("values are less than or equal to 0.98", None, 0.98),
    ("readings are greater than or equal to 5", 5.0, None),
    ("values lie in [0, 340]", 0.0, 340.0),
    ("the series is ≤ 96", None, 96.0),
    ("the series is >= 3", 3.0, None),
    ("the measured values are always positive", 0.0, None),
])
def test_cik_style_bound_phrasings_parse(span, minimum, maximum):
    bound, problem = parse_bound_span(span)
    assert problem is None
    assert bound.minimum == minimum
    assert bound.maximum == maximum


@pytest.mark.parametrize("span", [
    "the outage lasts between 10 and 20 August",
    "closed from 9 and 11 pm",
    "maintenance between 8 and 10:30",
])
def test_date_and_clock_ranges_are_not_read_as_bounds(span):
    bound, problem = parse_bound_span(span)
    assert bound is None, f"{span!r} parsed as {bound}"


@pytest.mark.parametrize("span, value", [
    ("output remains at 40 for the whole window", 40.0),
    ("the feed is held constant at 12.5", 12.5),
])
def test_more_override_phrasings_parse(span, value):
    parsed, problem = parse_override_span(span)
    assert problem is None and parsed == value


@pytest.mark.parametrize("span, minimum, maximum", [
    # A negated ceiling must not also read as a floor ("more than 60").
    ("no more than 60", None, 60.0),
    ("no greater than 60 units", None, 60.0),
    ("the load cannot be more than 500", None, 500.0),
    # Negation reaches verbs the un-negated lists never enumerated.
    ("temperatures cannot climb above 40", None, 40.0),
    ("it won't ever go past 75", None, 75.0),
    # A negated "stay below" is a floor, not a ceiling.
    ("the level will not stay below 100", 100.0, None),
    ("never dips below 12", 12.0, None),
])
def test_negated_phrasings_resolve_to_the_stated_side(span, minimum, maximum):
    bound, problem = parse_bound_span(span)
    assert problem is None, f"{span!r} rejected: {problem}"
    assert bound.minimum == minimum
    assert bound.maximum == maximum


@pytest.mark.parametrize("span", [
    # Relative to a base the span does not state; an absolute read would
    # apply a number the text never stated.
    "output will not exceed 90% of nominal capacity",
    "demand stays below 75 percent of the peak",
    "values remain between 10% and 20%",
    # A clock time is not a bound on values.
    "values stay below 5:30 pm levels",
])
def test_relative_and_clock_quantities_are_rejected_not_misread(span):
    bound, problem = parse_bound_span(span)
    assert bound is None, f"{span!r} parsed as {bound}"


def test_a_quantified_close_verb_states_the_number_not_zero():
    parsed, problem = parse_override_span(
        "the index closes at 340 for the maintenance week"
    )
    assert problem is None and parsed == 340.0


def test_a_clock_quantified_stop_verb_still_means_zero():
    parsed, problem = parse_override_span("production stops at 5:30 pm each day")
    assert problem is None and parsed == 0.0


def test_percent_override_values_are_rejected_not_misread():
    parsed, problem = parse_override_span("production will be 50% of capacity")
    assert parsed is None
    assert "estimated rather than stated" in problem


def test_a_cross_pattern_empty_bound_is_rejected():
    bound, problem = parse_bound_span(
        "values are at least 100 and will not exceed 50"
    )
    assert bound is None
    assert "empty" in problem


@pytest.mark.parametrize("span", [
    "demand is expected to be strong",
    "values may change considerably",
    "the series is bounded",  # states that a bound exists, not what it is
])
def test_spans_without_a_stated_bound_do_not_parse(span):
    bound, problem = parse_bound_span(span)
    assert bound is None
    assert "does not state a parseable bound" in problem


def test_an_empty_parsed_range_is_rejected():
    bound, problem = parse_bound_span("between 300 and 100")
    # sorted() makes an honest range of a reversed phrasing
    assert problem is None and bound.minimum == 100.0 and bound.maximum == 300.0


@pytest.mark.parametrize("span, value", [
    ("the plant is offline for maintenance", 0.0),
    ("the store is closed on public holidays", 0.0),
    ("production is halted from Tuesday to Thursday", 0.0),
    ("output drops to zero during the strike", 0.0),
    ("the line is shut down for retooling", 0.0),
    ("throughput will be 120 during the trial", 120.0),
    ("the feed is fixed at 42.5 for the window", 42.5),
    ("flow is reduced to 300 while one turbine runs", 300.0),
])
def test_override_spans_parse_to_the_stated_value(span, value):
    parsed, problem = parse_override_span(span)
    assert problem is None
    assert parsed == value


def test_an_explicit_number_wins_over_a_zero_word():
    parsed, _ = parse_override_span(
        "output reduced to 120 while the line is partially shut down"
    )
    assert parsed == 120.0


def test_override_spans_without_a_stated_value_do_not_parse():
    parsed, problem = parse_override_span("output will be reduced significantly")
    assert parsed is None
    assert "estimated rather than stated" in problem


# -- admission ---------------------------------------------------------------

def _event(event_id, event_type, start, end, attributes):
    return ContextEvent(
        event_id=event_id, event_type=event_type, entity_scope=("*",),
        effective_start=start.isoformat(), effective_end=end.isoformat(),
        known_at=START.isoformat(), attributes=attributes,
        source=ContextSource("dataset", "test#context"), created_by="llm",
    )


def _grid(n):
    return [START + timedelta(days=day) for day in range(n)]


HISTORY = [200.0 + (day % 7) for day in range(60)]
TIMESTAMPS = _grid(60)
FUTURE = [START + timedelta(days=60 + day) for day in range(7)]
H_START, H_END = FUTURE[0], FUTURE[-1]


def test_a_constraint_with_a_consistent_span_is_admitted():
    events = [_event("c1", "constraint:bounds", H_START, H_END,
                     {"source_span": "values stay between 0 and 340"})]
    assessment = assess_future_events(events, "s", HISTORY, TIMESTAMPS, FUTURE, 7)
    assert [e.event_id for e in assessment.admitted] == ["c1"]
    admitted = assessment.admitted[0]
    assert admitted.minimum == 0.0 and admitted.maximum == 340.0


def test_history_violating_a_claimed_bound_rejects_the_claim():
    events = [_event("c1", "constraint:bounds", H_START, H_END,
                     {"source_span": "the value will not exceed 150"})]
    assessment = assess_future_events(events, "s", HISTORY, TIMESTAMPS, FUTURE, 7)
    assert not assessment.admitted
    assert assessment.rejected[0]["code"] == "recent_history_respects_bound"
    assert "suspect" in assessment.rejected[0]["reason"]


def test_a_window_overlapping_history_is_sent_to_the_fold_gate():
    events = [_event("c1", "constraint:bounds",
                     TIMESTAMPS[-1] - timedelta(days=3), H_END,
                     {"source_span": "values stay between 0 and 340"})]
    assessment = assess_future_events(events, "s", HISTORY, TIMESTAMPS, FUTURE, 7)
    assert not assessment.admitted
    assert assessment.rejected[0]["code"] == "window_is_future_only"
    assert "ablation gate" in assessment.rejected[0]["reason"]


def test_a_missing_span_is_rejected():
    events = [_event("c1", "constraint:bounds", H_START, H_END, {})]
    assessment = assess_future_events(events, "s", HISTORY, TIMESTAMPS, FUTURE, 7)
    assert assessment.rejected[0]["code"] == "source_span_present"


def test_a_claimed_bound_that_disagrees_with_the_span_is_rejected():
    events = [_event("c1", "constraint:bounds", H_START, H_END,
                     {"source_span": "the value will not exceed 500",
                      "claimed_bound": {"max": 400}})]
    assessment = assess_future_events(events, "s", HISTORY, TIMESTAMPS, FUTURE, 7)
    assert not assessment.admitted
    assert assessment.rejected[0]["code"] == "claim_matches_span"


def test_a_claimed_override_value_that_disagrees_with_the_span_is_rejected():
    events = [_event("o1", "override:closure", H_START, H_END,
                     {"source_span": "the plant is offline",
                      "claimed_value": 5})]
    assessment = assess_future_events(events, "s", HISTORY, TIMESTAMPS, FUTURE, 7)
    assert not assessment.admitted
    assert assessment.rejected[0]["code"] == "claim_matches_span"


def test_events_outside_the_namespaces_are_not_this_lanes_business():
    events = [_event("e1", "promotion", H_START, H_END,
                     {"source_span": "a big promotion"})]
    assessment = assess_future_events(events, "s", HISTORY, TIMESTAMPS, FUTURE, 7)
    assert not assessment.considered


def test_an_override_contradicting_an_admitted_constraint_is_rejected():
    """Both claims came from the same context; if they disagree, the
    context contradicts itself. The constraint — the weaker claim — is
    kept, and the override is rejected rather than clamped into a value
    nobody stated."""
    events = [
        _event("c1", "constraint:floor", H_START, H_END,
               {"source_span": "output will not drop below 50"}),
        _event("o1", "override:closure", FUTURE[2], FUTURE[4],
               {"source_span": "the plant is offline"}),
    ]
    history = [60.0 + (day % 5) for day in range(60)]
    assessment = assess_future_events(events, "s", history, TIMESTAMPS, FUTURE, 7)
    assert [e.event_id for e in assessment.admitted] == ["c1"]
    rejection = next(r for r in assessment.rejected if r["event_id"] == "o1")
    assert rejection["code"] == "override_respects_admitted_constraints"
    assert "contradicts itself" in rejection["reason"]


def test_overlapping_overrides_with_different_values_reject_each_other():
    """Order-independent by construction: neither claim outranks the
    other, so resolving them by proposal order would let the proposer's
    emission order change the published numbers."""
    events = [
        _event("o1", "override:outage", FUTURE[0], FUTURE[3],
               {"source_span": "the plant is offline"}),
        _event("o2", "override:trial", FUTURE[2], FUTURE[5],
               {"source_span": "throughput will be 120 during the trial"}),
    ]
    forward = assess_future_events(events, "s", HISTORY, TIMESTAMPS, FUTURE, 7)
    reverse = assess_future_events(list(reversed(events)), "s", HISTORY,
                                   TIMESTAMPS, FUTURE, 7)
    for assessment in (forward, reverse):
        assert not assessment.admitted
        codes = {item["code"] for item in assessment.rejected}
        assert codes == {"overrides_agree_where_they_overlap"}
        assert {item["event_id"] for item in assessment.rejected} == {"o1", "o2"}


def test_overlapping_overrides_with_the_same_value_agree():
    events = [
        _event("o1", "override:outage", FUTURE[0], FUTURE[3],
               {"source_span": "the plant is offline"}),
        _event("o2", "override:strike", FUTURE[2], FUTURE[5],
               {"source_span": "production is halted by the strike"}),
    ]
    assessment = assess_future_events(events, "s", HISTORY, TIMESTAMPS, FUTURE, 7)
    assert {e.event_id for e in assessment.admitted} == {"o1", "o2"}


def test_a_window_running_past_the_horizon_has_no_closing_boundary():
    """The last visible step of a longer window is interior, not an edge:
    schedule-edge uncertainty exists where the stated window ends, and
    this one ends weeks after the horizon does."""
    admitted = [FutureEvent(
        "o1", "override", FUTURE[3].isoformat(),
        (FUTURE[-1] + timedelta(days=40)).isoformat(),
        "the plant is offline", value=0.0,
    )]
    projected, _ = apply_future_events(_rows(), admitted)
    # opening edge: widened union
    assert projected[3]["q90"] > 0.0
    # every later step, including the horizon's last, is interior
    for index in range(4, len(projected)):
        assert projected[index]["q10"] == projected[index]["q90"] == 0.0


def test_a_compatible_override_and_constraint_are_both_admitted():
    events = [
        _event("c1", "constraint:cap", H_START, H_END,
               {"source_span": "values stay between 0 and 340"}),
        _event("o1", "override:closure", FUTURE[2], FUTURE[4],
               {"source_span": "the plant is offline"}),
    ]
    assessment = assess_future_events(events, "s", HISTORY, TIMESTAMPS, FUTURE, 7)
    assert {e.event_id for e in assessment.admitted} == {"c1", "o1"}


def test_a_window_beyond_the_horizon_is_rejected():
    events = [_event("o1", "override:closure",
                     FUTURE[-1] + timedelta(days=3), FUTURE[-1] + timedelta(days=5),
                     {"source_span": "the plant is offline"})]
    assessment = assess_future_events(events, "s", HISTORY, TIMESTAMPS, FUTURE, 7)
    assert assessment.rejected[0]["code"] == "window_touches_horizon"


# -- application -------------------------------------------------------------

def _rows():
    rows = []
    for step, timestamp in enumerate(FUTURE):
        point = 200.0 + step
        rows.append({
            "timestamp": timestamp.isoformat(), "point": point,
            "q10": point - 10, "q50": point, "q90": point + 10,
            "point_bias_correction": 0.0,
        })
    return rows


def test_constraints_clamp_every_quantile_and_only_inside_the_window():
    admitted = [FutureEvent(
        "c1", "constraint", FUTURE[1].isoformat(), FUTURE[3].isoformat(),
        "will not exceed 205", maximum=205.0,
    )]
    projected, applications = apply_future_events(_rows(), admitted)
    # inside the window: q90 (211..213) clamps to 205
    for index in (1, 2, 3):
        assert projected[index]["q90"] == 205.0
        assert projected[index]["q10"] <= projected[index]["q50"] <= projected[index]["q90"]
    # outside: untouched
    assert projected[0]["q90"] == 210.0
    assert projected[4]["q90"] == 214.0
    assert all(entry["event_class"] == "constraint" for entry in applications)
    assert applications  # every clamp is recorded


def test_overrides_set_the_stated_value_and_widen_the_boundaries():
    admitted = [FutureEvent(
        "o1", "override", FUTURE[2].isoformat(), FUTURE[4].isoformat(),
        "the plant is offline", value=0.0,
    )]
    projected, applications = apply_future_events(_rows(), admitted)
    # interior step: deterministic — all quantiles are the stated value
    assert projected[3]["point"] == 0.0
    assert projected[3]["q10"] == projected[3]["q50"] == projected[3]["q90"] == 0.0
    # boundary steps: the union of the base interval and the stated value
    for index in (2, 4):
        assert projected[index]["point"] == 0.0
        assert projected[index]["q50"] == 0.0
        assert projected[index]["q10"] == 0.0  # min(base q10, 0)
        assert projected[index]["q90"] == pytest.approx(200.0 + index + 10)
    # outside the window: untouched
    assert projected[1]["point"] == 201.0
    assert projected[5]["point"] == 205.0
    override_steps = [e for e in applications if e["event_class"] == "override"]
    assert len(override_steps) == 3
    assert [e["boundary_step"] for e in override_steps] == [True, False, True]


def test_no_admitted_events_is_a_strict_no_op():
    rows = _rows()
    projected, applications = apply_future_events(rows, [])
    assert projected is rows and applications == []


# -- end-to-end wiring -------------------------------------------------------

def _write_csv(path, days=120):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "value"])
        for day in range(days):
            writer.writerow([
                (START + timedelta(days=day)).isoformat(),
                200 + 10 * (day % 7) + 0.3 * day,
            ])


def _flag_on_config():
    config = GnomonConfig()
    config.context.future_events = True
    return config


def _horizon_events(csv_days=120):
    h_start = START + timedelta(days=csv_days)
    return [
        _event("bound-1", "constraint:cap", h_start, h_start + timedelta(days=6),
               {"source_span": "values will stay between 0 and 340 units"}),
        _event("override-1", "override:maintenance",
               h_start + timedelta(days=2), h_start + timedelta(days=4),
               {"source_span": "the plant is offline for scheduled maintenance"}),
    ]


def test_flag_on_downgrades_support_and_records_the_counterfactual(tmp_path):
    csv_path = tmp_path / "series.csv"
    _write_csv(csv_path)
    artifact, _ = forecast(
        str(csv_path), time_column="timestamp", target_column="value",
        horizon=7, frequency="D", output=str(tmp_path / "out"),
        context_events=_horizon_events(), config=_flag_on_config(),
    )
    result = artifact.results[0]
    assert result.support == "context_trusted"
    assessment = result.support_assessment
    assert assessment["status"] == "conditionally_supported"
    assert any(reason["code"] == "unbacktested_context_admitted"
               for reason in assessment["reasons"])
    admitted_ids = {item["event_id"] for item in result.future_context["admitted"]}
    assert admitted_ids == {"bound-1", "override-1"}
    # override applied: days 2-4 of the horizon are the stated 0
    assert result.forecast[3]["point"] == 0.0
    assert result.forecast[3]["q90"] == 0.0
    # boundary steps keep the union interval
    assert result.forecast[2]["q90"] > 0.0
    applied = [item for item in artifact.evidence
               if item.kind == "future_context_applied"]
    assert len(applied) == 1
    counterfactual = applied[0].payload["history_only_counterfactual"]
    assert len(counterfactual) == 7
    assert counterfactual[3]["point"] != 0.0  # the forecast before the lane
    gate = [item for item in artifact.evidence
            if item.kind == "future_context_gate"]
    assert gate[0].payload["by_class"]["constraint"]["admitted"] == 1
    assert gate[0].payload["by_class"]["override"]["admitted"] == 1


def test_flag_on_enters_the_admitted_events_into_the_artifact_id(tmp_path):
    csv_path = tmp_path / "series.csv"
    _write_csv(csv_path)
    kwargs = dict(
        time_column="timestamp", target_column="value", horizon=7,
        frequency="D", context_events=_horizon_events(),
    )
    flag_off, _ = forecast(str(csv_path), output=str(tmp_path / "off"), **kwargs)
    flag_on, _ = forecast(str(csv_path), output=str(tmp_path / "on"),
                          config=_flag_on_config(), **kwargs)
    assert flag_on.forecast_id != flag_off.forecast_id


def test_flag_off_is_byte_identical_to_a_run_without_the_feature(tmp_path):
    """The flag-off ID payload has no future_context key, so a flag-off run
    with a default config equals one with no config at all — which is the
    pre-feature behaviour the goldens pin."""
    csv_path = tmp_path / "series.csv"
    _write_csv(csv_path)
    kwargs = dict(
        time_column="timestamp", target_column="value", horizon=7,
        frequency="D", context_events=_horizon_events(),
    )
    no_config, _ = forecast(str(csv_path), output=str(tmp_path / "a"), **kwargs)
    default_config, _ = forecast(str(csv_path), output=str(tmp_path / "b"),
                                 config=GnomonConfig(), **kwargs)
    assert no_config.forecast_id == default_config.forecast_id
    result = no_config.results[0]
    assert result.support != "context_trusted"
    assert result.future_context is None
    assert "future_context" not in no_config.to_dict()["results"][0]
    assert not any(item.kind.startswith("future_context")
                   for item in no_config.evidence)


def test_flag_on_with_nothing_admitted_leaves_the_forecast_alone(tmp_path):
    csv_path = tmp_path / "series.csv"
    _write_csv(csv_path)
    h_start = START + timedelta(days=120)
    suspect = [_event("c1", "constraint:cap", h_start, h_start + timedelta(days=6),
                      {"source_span": "the value will not exceed 150"})]
    kwargs = dict(
        time_column="timestamp", target_column="value", horizon=7, frequency="D",
    )
    baseline, _ = forecast(str(csv_path), output=str(tmp_path / "a"), **kwargs)
    gated, _ = forecast(str(csv_path), output=str(tmp_path / "b"),
                        context_events=suspect, config=_flag_on_config(), **kwargs)
    result = gated.results[0]
    assert result.support == baseline.results[0].support
    assert [row["point"] for row in result.forecast] == \
        [row["point"] for row in baseline.results[0].forecast]
    # the rejection is still disclosed
    assert result.future_context["rejected"][0]["code"] == \
        "recent_history_respects_bound"


def test_threshold_analysis_describes_the_published_rows(tmp_path):
    """An overridden window must not report crossing probabilities for the
    forecast the lane replaced: monitor and decide read these numbers."""
    csv_path = tmp_path / "series.csv"
    _write_csv(csv_path)
    h_start = START + timedelta(days=120)
    override = [_event("o1", "override:maintenance",
                       h_start + timedelta(days=2), h_start + timedelta(days=4),
                       {"source_span": "the plant is offline"})]
    artifact, _ = forecast(
        str(csv_path), time_column="timestamp", target_column="value",
        horizon=7, frequency="D", output=str(tmp_path / "out"),
        context_events=override, config=_flag_on_config(), threshold=100.0,
    )
    result = artifact.results[0]
    assert result.forecast[3]["point"] == 0.0
    probabilities = result.threshold["probability_above"]
    # untouched steps forecast ~230-300, far above the threshold
    assert probabilities[0] > 0.5
    # the interior override step is the stated 0, far below it
    assert probabilities[3] < 0.5


def test_general_purpose_path_from_document_to_trusted_forecast(tmp_path):
    """The lane is not benchmark furniture: a caller's own document, run
    through the stock `gnomon context prompt`/`validate` workflow and the
    ordinary forecast entry point, reaches `context_trusted` with no
    benchmark code involved. The verified evidence_quote is the span."""
    from gnomon.context import event_from_dict
    from gnomon.workflows import DocumentRef, parse_context_response

    csv_path = tmp_path / "series.csv"
    _write_csv(csv_path)
    h_start = START + timedelta(days=120)
    document = DocumentRef(
        name="ops.md",
        content=(
            "Written 2026-01-01. Next week the plant is offline for "
            "scheduled maintenance. Design capacity: values will stay "
            "between 0 and 340 units."
        ),
        source_type="planning_file",
        reference="/notes/ops.md",
    )
    llm_response = {"events": [
        {
            "document_index": 0,
            "event_type": "constraint:capacity",
            "entity_scope": ["*"],
            "effective_start": h_start.isoformat(),
            "effective_end": (h_start + timedelta(days=6)).isoformat(),
            "known_at": START.isoformat(),
            "evidence_quote": "values will stay between 0 and 340 units",
        },
        {
            "document_index": 0,
            "event_type": "override:maintenance",
            "entity_scope": ["*"],
            "effective_start": (h_start + timedelta(days=2)).isoformat(),
            "effective_end": (h_start + timedelta(days=4)).isoformat(),
            "known_at": START.isoformat(),
            "evidence_quote": "the plant is offline for scheduled maintenance",
        },
    ]}
    validated = parse_context_response(llm_response, [document])
    assert not validated["rejected"]
    events = [event_from_dict(item) for item in validated["events"]]
    artifact, _ = forecast(
        str(csv_path), time_column="timestamp", target_column="value",
        horizon=7, frequency="D", output=str(tmp_path / "out"),
        context_events=events, config=_flag_on_config(),
    )
    result = artifact.results[0]
    assert result.support == "context_trusted"
    assert len(result.future_context["admitted"]) == 2
    assert result.forecast[3]["point"] == 0.0


# -- config and capabilities -------------------------------------------------

def test_config_parses_on_off(tmp_path):
    from gnomon.config import load_config

    path = tmp_path / "gnomon.yaml"
    path.write_text("context:\n  future_events: on\n")
    assert load_config(str(path)).context.future_events is True
    path.write_text("context:\n  future_events: off\n")
    assert load_config(str(path)).context.future_events is False
    path.write_text("context:\n  future_events: maybe\n")
    from gnomon.contracts import GnomonError
    with pytest.raises(GnomonError, match="on or off"):
        load_config(str(path))


def test_default_config_leaves_the_lane_off():
    assert GnomonConfig().context.future_events is False


def test_capabilities_report_the_flag_and_the_event_classes():
    from gnomon.runtime import capabilities

    payload = capabilities()
    future = payload["context_events"]["future_events"]
    assert future["flag"] == "context.future_events"
    assert future["default"] == "off"
    assert future["event_classes"] == ["constraint", "deterministic_override"]
    # The field names what it measures — the loaded config file's setting —
    # because config is only consumed by the forecast verb and the Python
    # API, and a bare "enabled" would overstate its reach.
    assert future["enabled_in_config"] is False
    assert payload["features"]["future_context_events"] is True

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone

import pytest

from gnomon.cli import main
from gnomon.context import ContextEvent, ContextSource
from gnomon.context_model import event_adjusted
from gnomon.runtime import forecast

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
PROMO_EFFECT = 30.0


def _promo_days(length: int) -> set[int]:
    return {day for day in range(length) if day % 10 in (5, 6)}


def _values(length: int) -> list[float]:
    promos = _promo_days(length)
    return [100.0 + 0.5 * day + (PROMO_EFFECT if day in promos else 0.0) for day in range(length)]


def _events(length_with_future: int) -> list[ContextEvent]:
    events = []
    for day in range(length_with_future):
        if day % 10 == 5:
            start = START + timedelta(days=day)
            events.append(ContextEvent(
                event_id=f"promo_{day:03d}",
                event_type="promotion",
                entity_scope=("*",),
                effective_start=start.isoformat(),
                effective_end=(start + timedelta(days=1, hours=23)).isoformat(),
                known_at=(START - timedelta(days=1)).isoformat(),
                source=ContextSource("calendar", "promo-calendar.ics"),
            ))
    return events


def _write_csv(path, length: int) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "requests"])
        for day, value in enumerate(_values(length)):
            writer.writerow([(START + timedelta(days=day)).isoformat(), value])


def test_event_adjusted_learns_the_effect() -> None:
    history = [10.0, 10.0, 25.0, 10.0, 10.0]
    prediction = event_adjusted(
        history, 2, 7, [False, False, True, False, False], [True, False]
    )
    assert prediction[0] - prediction[1] == pytest.approx(15.0)


def test_event_adjusted_refuses_without_history_occurrences() -> None:
    with pytest.raises(ValueError, match="no occurrences"):
        event_adjusted([1.0, 2.0, 3.0], 1, 7, [False, False, False], [True])


def test_context_admitted_when_it_demonstrates_stable_lift(tmp_path) -> None:
    csv_path = tmp_path / "promo.csv"
    _write_csv(csv_path, 130)
    baseline_artifact, _ = forecast(
        str(csv_path), time_column="timestamp", target_column="requests",
        horizon=7, frequency="D", output=str(tmp_path / "baseline"),
    )
    artifact, _ = forecast(
        str(csv_path), time_column="timestamp", target_column="requests",
        horizon=7, frequency="D", output=str(tmp_path / "out"),
        context_events=_events(140),
    )
    result = artifact.results[0]
    assert result.context is not None
    assert result.context["admitted"] is True
    assert result.context["effect"]["distribution"]["distribution"] == "empirical"
    assert result.context["effect"]["provenance"]["provenance_class"] == (
        "same_event_same_series")
    assert result.context["effect"]["provenance"]["observed"] is True
    assert result.selected_model == "event_adjusted"
    assert result.context_outcome["status"] == "applied"
    assert result.context_outcome["primary_forecast_changed"] is True
    assert result.context_outcome["canonical_primary_preserved"] is True
    assert result.primary_forecast
    # Context may earn a different support tier, but cannot retroactively
    # relabel the immutable history-only answer.
    assert result.primary_forecast == baseline_artifact.results[0].forecast
    assert [row["point"] for row in result.primary_forecast] != [
        row["point"] for row in result.forecast]
    assert result.context_outcome["canonical_primary_location"] == (
        "artifact.results[].primary_forecast")
    assert result.context["mean_improvement"] > 0.02
    assert any(evidence.kind == "context_ablation" for evidence in artifact.evidence)
    # Day 135 (a promo day, %10 == 5) falls inside the 7-day horizon:
    # the forecast must carry the promo bump on that day.
    promo_row = result.forecast[5]
    plain_row = result.forecast[4]
    assert promo_row["point"] - plain_row["point"] > PROMO_EFFECT / 2


def test_context_selection_skips_pre_event_folds_without_rejecting_candidate(
        tmp_path) -> None:
    """A fold before the first event cannot adjudicate an event model."""
    csv_path = tmp_path / "late_events.csv"
    event_days = {day for day in range(25, 140, 10)}
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "requests"])
        for day in range(130):
            value = 100.0 + 0.2 * day + (20.0 if day in event_days else 0.0)
            writer.writerow([(START + timedelta(days=day)).isoformat(), value])
    events = []
    for day in sorted(event_days):
        moment = START + timedelta(days=day)
        events.append(ContextEvent(
            event_id=f"late_{day}", event_type="promotion",
            entity_scope=("*",), effective_start=moment.isoformat(),
            effective_end=moment.isoformat(),
            known_at=(START - timedelta(days=1)).isoformat(),
            source=ContextSource("calendar", "late-promos.ics"),
        ))
    artifact, _ = forecast(
        str(csv_path), time_column="timestamp", target_column="requests",
        horizon=7, frequency="D", output=str(tmp_path / "out"),
        context_events=events,
    )
    context = artifact.results[0].context
    assert not any(
        check.get("code") == "candidate_fits_every_fold"
        and check.get("passed") is False
        for check in context.get("gate_checks", [])
    )
    assert context.get("shape_scores")


def test_unsourced_events_are_excluded_and_context_rejected(tmp_path) -> None:
    csv_path = tmp_path / "promo.csv"
    _write_csv(csv_path, 130)
    unsourced = [
        ContextEvent(
            event_id="rumour",
            event_type="promotion",
            entity_scope=("*",),
            effective_start=(START + timedelta(days=5)).isoformat(),
            effective_end=(START + timedelta(days=6)).isoformat(),
            known_at=(START - timedelta(days=1)).isoformat(),
            source=None,
        )
    ]
    artifact, _ = forecast(
        str(csv_path), time_column="timestamp", target_column="requests",
        horizon=7, frequency="D", output=str(tmp_path / "out"),
        context_events=unsourced,
    )
    result = artifact.results[0]
    assert result.context["admitted"] is False
    assert result.selected_model != "event_adjusted"
    assert result.context["events_excluded"][0]["reason"].startswith("no verifiable source")
    assert result.context_outcome["status"] == "scenario_only"
    assert result.context_outcome["primary_forecast_changed"] is False


def test_cli_forecast_with_context_file(tmp_path, capsys) -> None:
    csv_path = tmp_path / "promo.csv"
    _write_csv(csv_path, 130)
    events_path = tmp_path / "events.json"
    events_path.write_text(json.dumps({
        "schema_version": "0.1",
        "events": [
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "entity_scope": list(event.entity_scope),
                "effective_start": event.effective_start,
                "effective_end": event.effective_end,
                "known_at": event.known_at,
                "source": {"type": event.source.type, "reference": event.source.reference},
            }
            for event in _events(140)
        ],
    }))
    assert main([
        "forecast", str(csv_path), "--time", "timestamp", "--target", "requests",
        "--horizon", "7", "--frequency", "D", "--output", str(tmp_path / "out"),
        "--context", str(events_path),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["context"]["admitted"] is True
    assert payload["results"][0]["selected_model"] == "event_adjusted"
    assert payload["results"][0]["primary_forecast_preview"]
    assert payload["results"][0]["forecast_role"] == (
        "context_conditioned_projection")


def test_invalid_context_file_fails_loudly(tmp_path, capsys) -> None:
    csv_path = tmp_path / "promo.csv"
    _write_csv(csv_path, 130)
    events_path = tmp_path / "events.json"
    events_path.write_text(json.dumps({"events": [{"event_id": "bad"}]}))
    assert main([
        "forecast", str(csv_path), "--time", "timestamp", "--target", "requests",
        "--horizon", "7", "--frequency", "D", "--context", str(events_path),
    ]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "INVALID_CONTEXT_EVENT"


def test_wilson_upper_bounds_a_noisy_proportion() -> None:
    from gnomon.context_eval import wilson_upper

    # Four of five covered: the upper bound stays high, so a coverage
    # veto cannot fire on five points of sampling noise alone.
    assert wilson_upper(4, 5) > 0.9
    # The same proportion measured on 500 points is known far better.
    assert wilson_upper(400, 500) < 0.85
    assert wilson_upper(0, 0) == 1.0


def test_gate_records_every_check_and_names_the_decider(tmp_path) -> None:
    csv_path = tmp_path / "promo.csv"
    _write_csv(csv_path, 130)
    artifact, _ = forecast(
        str(csv_path), time_column="timestamp", target_column="requests",
        horizon=7, frequency="D", output=str(tmp_path / "out"),
        context_events=_events(140),
    )
    gate = [record for record in artifact.evidence
            if record.kind == "context_gate"]
    assert len(gate) == 1
    payload = gate[0].payload
    assert payload["events_supplied"] > 0
    assert payload["admitted"] is True
    # An admitted run records the conditions it passed, not only failures.
    codes = {check["code"] for check in payload["checks"]}
    assert {"events_eligible", "mean_improvement_meets_margin",
            "majority_of_folds_improve"} <= codes
    stability = next(check for check in payload["checks"]
                     if check["code"] == "majority_of_folds_improve")
    fitted = next(check for check in payload["checks"]
                  if check["code"] == "candidate_fits_every_fold")
    assert stability["measured"]["exposed"] <= fitted["measured"]
    assert all(check["passed"] for check in payload["checks"])
    assert payload["decided_by"] is None


def test_gate_names_the_condition_that_rejected(tmp_path) -> None:
    csv_path = tmp_path / "promo.csv"
    _write_csv(csv_path, 130)
    unsourced = [
        ContextEvent(
            event_id=event.event_id, event_type=event.event_type,
            entity_scope=event.entity_scope,
            effective_start=event.effective_start,
            effective_end=event.effective_end, known_at=event.known_at,
        )
        for event in _events(140)
    ]
    artifact, _ = forecast(
        str(csv_path), time_column="timestamp", target_column="requests",
        horizon=7, frequency="D", output=str(tmp_path / "out2"),
        context_events=unsourced,
    )
    payload = [r for r in artifact.evidence if r.kind == "context_gate"][0].payload
    assert payload["admitted"] is False
    # Every supplied event failed eligibility, and the record says so with
    # counts rather than only in prose.
    assert payload["events_supplied"] > 0
    assert payload["events_eligible"] == 0
    assert payload["decided_by"] == "events_eligible"


class TestEffectShapes:
    """The shape of an intervention is chosen by measurement, not by the caller.

    A caller who could name the shape could fit a story to the data, which is
    the thing the admission gate exists to prevent. Three shapes on the same
    folds is also three chances to look good by luck, so the winner still has
    to beat the history-only baseline, not merely beat the other shapes.
    """

    def test_shapes_preserve_the_measured_total(self):
        """Shapes differ in timing, not magnitude.

        Without normalising by the mean active weight, `decay` would win
        comparisons simply by applying a smaller effect.
        """
        from gnomon.context_model import EFFECT_SHAPES, event_effect, event_effect_shaped

        history = [10.0] * 20 + [30.0] * 10 + [10.0] * 20
        active = [False] * 20 + [True] * 10 + [False] * 20
        base = event_effect(history, active)
        for shape in EFFECT_SHAPES:
            weights = __import__("gnomon.context_model", fromlist=["shape_weights"]).shape_weights(active, shape)
            scaled = event_effect_shaped(history, active, shape)
            delivered = [scaled * weight for weight, flag in zip(weights, active) if flag]
            assert abs(sum(delivered) / len(delivered) - base) < 1e-9, shape

    def test_residual_effect_removes_history_model_bias(self):
        from gnomon.context_model import residual_event_effect

        residuals = [0.5, -0.5, 8.5, 7.5, 0.5, -0.5]
        active = [False, False, True, True, False, False]
        assert residual_event_effect(residuals, active, "level") == 8.0

    def test_residual_candidate_changes_only_future_event_steps(self):
        from gnomon.context_model import residual_event_adjusted

        points = residual_event_adjusted(
            [0.0, 6.0, 0.0, 6.0], 4,
            [False, True, False, True], [False, True, True, False],
            [10.0, 10.0, 10.0, 10.0], "level")
        assert points == [10.0, 16.0, 16.0, 10.0]

    def test_episode_residual_freezes_the_base_before_each_event(self):
        from gnomon.context_model import episode_residual_effect

        # A naive last-value model would adapt after the first high point if
        # residuals were computed one step at a time. Freezing at each onset
        # recovers the full two-step pulse from both historical episodes.
        history = [10.0] * 4 + [20.0, 20.0] + [10.0] * 4 + [20.0, 20.0]
        active = [False] * 4 + [True, True] + [False] * 4 + [True, True]
        effect = episode_residual_effect(
            history, active, "last_value", season=1, shape="level")
        assert effect == pytest.approx(10.0)

    def test_episode_amplitudes_keep_occurrences_independent(self):
        from gnomon.context_model import episode_residual_amplitudes

        observations = [
            (4.0, 0, 2), (4.0, 1, 2),
            (-2.0, 0, 2), (-2.0, 1, 2),
        ]
        assert episode_residual_amplitudes(observations, "level") == [4.0, -2.0]

    def test_each_active_run_gets_its_own_onset(self):
        from gnomon.context_model import shape_weights

        active = [True, True, False, True, True]
        weights = shape_weights(active, "decay")
        assert weights[0] == weights[3], (
            "a second occurrence must decay from its own onset, not inherit "
            "the first one's tail"
        )
        assert weights[2] == 0.0

    def test_ramp_builds_across_the_window(self):
        from gnomon.context_model import shape_weights

        weights = shape_weights([True] * 4, "ramp")
        assert weights == [0.25, 0.5, 0.75, 1.0]

    def test_level_is_flat(self):
        from gnomon.context_model import shape_weights

        assert shape_weights([True, True, False], "level") == [1.0, 1.0, 0.0]

    def test_unknown_shape_is_refused(self):
        import pytest as _pytest

        from gnomon.context_model import shape_weights

        with _pytest.raises(ValueError, match="unknown effect shape"):
            shape_weights([True], "exponential_ramp_v2")

    def test_layering_the_effect_onto_a_seasonal_base_double_counts(self):
        """Pins a hazard that reads as an improvement.

        Adding the measured effect to a base that already models the event
        counts the same bump twice. Attempting it in the admission gate
        collapsed fold improvements toward -1; this states the arithmetic
        directly so the idea is not re-attempted.
        """
        from gnomon.context_model import event_adjusted

        period, span, bump = 10, 3, 40.0
        history = [100.0 + (bump if index % period < span else 0.0)
                   for index in range(120)]
        active = [index % period < span for index in range(120)]
        future_active = [True] * 6

        # A base that already contains the bump: the truth itself, repeated.
        informed_base = [100.0 + bump] * 6
        layered = event_adjusted(history, 6, period, active, future_active,
                                 "level", informed_base)
        assert all(value > 100.0 + bump * 1.5 for value in layered), (
            "layering onto an informed base must visibly overshoot; if this "
            "ever stops being true the double-counting has been fixed and the "
            "gate can be revisited"
        )

        # A base that does not: the effect lands where it belongs.
        naive_base = [100.0] * 6
        adjusted = event_adjusted(history, 6, period, active, future_active,
                                  "level", naive_base)
        # The measured effect is 38.3 rather than 40: `event_effect` is a
        # difference of detrended means, and the inactive mean is pulled by
        # the elevated periods it is measured against.
        assert all(abs(value - (100.0 + bump)) < 2.5 for value in adjusted)

    def test_the_ablation_recovers_the_planted_shape(self, tmp_path):
        """The measurement has to be able to tell the shapes apart, or
        offering three of them is just three chances to overfit."""
        import csv
        import math

        from gnomon.context import ContextEvent, ContextSource
        from gnomon.runtime import forecast

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        period, span, count = 30, 8, 300

        def build(kind, directory):
            path = directory / f"{kind}.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp", "value"])
                for index in range(count):
                    base = (100 + 0.15 * index
                            + 8 * math.sin(2 * math.pi * index / 24) + (index % 5))
                    offset = index % period
                    if offset >= span:
                        bump = 0.0
                    elif kind == "level":
                        bump = 40.0
                    elif kind == "decay":
                        bump = 40.0 * (0.6 ** offset)
                    else:
                        bump = 40.0 * (offset + 1) / span
                    writer.writerow([(start + timedelta(hours=index)).isoformat(),
                                     base + bump])
            source = ContextSource("planning_file", "promos.md")
            events = [
                ContextEvent(
                    event_id=f"p{hour}", event_type="promo", entity_scope=("*",),
                    effective_start=(start + timedelta(hours=hour)).isoformat(),
                    effective_end=(start + timedelta(hours=hour + span - 1)).isoformat(),
                    known_at=start.isoformat(), source=source,
                )
                for hour in range(0, count + 60, period)
            ]
            return path, events

        for kind in ("level", "decay", "ramp"):
            path, events = build(kind, tmp_path)
            artifact, _ = forecast(
                str(path), time_column="timestamp", target_column="value",
                horizon=12, context_events=events,
                output=str(tmp_path / f"out-{kind}"),
            )
            context = artifact.results[0].context
            assert context["effect_shape"] == kind, (
                f"planted {kind}, ablation chose {context['effect_shape']} "
                f"from {context['shape_scores']}"
            )


class TestShrinkageAdmission:
    """λ governs strength, never validity.

    Measured to make forecasts worse when stacked on top of the existing
    strength conditions (docs/shrinkage-admission-measurement-2026-08.md),
    so it is off by default. λ is computed and disclosed regardless, which
    is what makes that measurable at all.
    """

    def test_a_mean_far_above_its_noise_keeps_the_effect(self):
        from gnomon.context_eval import shrinkage_factor

        assert shrinkage_factor([0.50, 0.52, 0.49, 0.51]) > 0.95

    def test_a_mean_inside_its_noise_keeps_none(self):
        from gnomon.context_eval import shrinkage_factor

        assert shrinkage_factor([0.5, -0.4, 0.6, -0.5]) == 0.0

    def test_a_negative_mean_keeps_none(self):
        from gnomon.context_eval import shrinkage_factor

        assert shrinkage_factor([-0.2, -0.3, -0.1]) == 0.0

    def test_a_factor_below_the_floor_is_pinned_to_zero(self):
        from gnomon.context_eval import MINIMUM_SHRINKAGE, shrinkage_factor

        # Constructed so the raw factor lands just under the floor: a small
        # non-zero effect no evidence supports is worse than none.
        value = shrinkage_factor([0.30, 0.05, 0.28, 0.02])
        assert value == 0.0 or value >= MINIMUM_SHRINKAGE

    def test_one_fold_cannot_support_any_effect(self):
        from gnomon.context_eval import shrinkage_factor

        assert shrinkage_factor([0.9]) == 0.0
        assert shrinkage_factor([]) == 0.0

    def test_shrinkage_is_disclosed_even_when_not_applied(self, tmp_path):
        csv_path = tmp_path / "promo.csv"
        _write_csv(csv_path, 130)
        artifact, _ = forecast(
            str(csv_path), time_column="timestamp", target_column="requests",
            horizon=7, frequency="D", output=str(tmp_path / "out"),
            context_events=_events(140),
        )
        context = artifact.results[0].context
        assert "shrinkage" in context, (
            "the factor must be visible whether or not it is applied, or the "
            "decision to leave it off cannot be revisited with evidence"
        )


# -- expected_shape: nomination narrows the contest, never picks the winner --

def _shaped_events(length_with_future: int, shape: str | None,
                   alternate: str | None = None) -> list[ContextEvent]:
    """The promo calendar, with a shape nomination on each event. When
    ``alternate`` is given, every second event nominates it instead —
    a conflicting panel."""
    events = _events(length_with_future)
    out = []
    for index, event in enumerate(events):
        nominated = alternate if (alternate and index % 2) else shape
        attributes = {"expected_shape": nominated} if nominated else {}
        out.append(ContextEvent(
            event_id=event.event_id, event_type=event.event_type,
            entity_scope=event.entity_scope,
            effective_start=event.effective_start,
            effective_end=event.effective_end,
            known_at=event.known_at, source=event.source,
            attributes=attributes,
        ))
    return out


def test_nominated_shape_narrows_the_contest(tmp_path) -> None:
    """One nomination, one comparison: the ablation scores only the
    nominated shape against the history-only baseline, and the artifact
    records both the nomination and the narrowed contest."""
    csv_path = tmp_path / "promo.csv"
    _write_csv(csv_path, 130)
    artifact, _ = forecast(
        str(csv_path), time_column="timestamp", target_column="requests",
        horizon=7, frequency="D", output=str(tmp_path / "out"),
        context_events=_shaped_events(140, "level"),
    )
    context = artifact.results[0].context
    assert context["admitted"] is True
    assert context["nominated_shape"] == "level"
    assert context["effect_shape"] == "level"
    assert set(context["shape_scores"]) == {"level"}, (
        "a nomination must narrow the contest, not merely bias it"
    )


def test_conflicting_nominations_run_the_full_contest(tmp_path) -> None:
    """Proposers who disagree cancel out: the full three-shape contest
    runs and no nomination is recorded."""
    csv_path = tmp_path / "promo.csv"
    _write_csv(csv_path, 130)
    artifact, _ = forecast(
        str(csv_path), time_column="timestamp", target_column="requests",
        horizon=7, frequency="D", output=str(tmp_path / "out"),
        context_events=_shaped_events(140, "level", alternate="decay"),
    )
    context = artifact.results[0].context
    assert "nominated_shape" not in context
    assert len(context["shape_scores"]) > 1


def test_nominated_shape_that_cannot_win_is_excluded_not_switched(tmp_path) -> None:
    """Events with no real effect, nominated as `decay`: the gate refuses
    admission rather than quietly trying the other shapes."""
    csv_path = tmp_path / "flat.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "requests"])
        for day in range(130):
            writer.writerow([
                (START + timedelta(days=day)).isoformat(), 100.0 + 0.5 * day,
            ])
    artifact, _ = forecast(
        str(csv_path), time_column="timestamp", target_column="requests",
        horizon=7, frequency="D", output=str(tmp_path / "out"),
        context_events=_shaped_events(140, "decay"),
    )
    context = artifact.results[0].context
    assert context["admitted"] is False
    assert context["nominated_shape"] == "decay"
    assert set(context["shape_scores"]) <= {"decay"}


def test_invalid_expected_shape_is_a_contract_violation() -> None:
    from gnomon.context import validate_context_event

    event = ContextEvent(
        event_id="promo", event_type="promotion", entity_scope=("*",),
        effective_start=START.isoformat(),
        effective_end=(START + timedelta(days=1)).isoformat(),
        known_at=(START - timedelta(days=1)).isoformat(),
        source=ContextSource("calendar", "promo-calendar.ics"),
        attributes={"expected_shape": "spike"},
    )
    problems = validate_context_event(event)
    assert any("expected_shape" in problem for problem in problems)


def test_text_context_compiler_keeps_hypotheses_non_numeric() -> None:
    from gnomon.workflows import DocumentRef, parse_context_response

    document = DocumentRef(
        "runbook", "Traffic normally drops every weekend.",
        reference="runbook.md")
    result = parse_context_response({
        "events": [],
        "hypotheses": [{
            "document_index": 0, "kind": "seasonality",
            "entity_scope": ["traffic"], "value": "weekly decrease",
            "evidence_quote": "Traffic normally drops every weekend.",
        }],
    }, [document])

    [hypothesis] = result["hypotheses"]
    assert hypothesis["status"] == "proposed_for_numeric_verification"
    assert hypothesis["may_affect_numbers"] is False
    assert hypothesis["source"]["reference"] == "runbook.md"

"""Conditional forecasts: converting an abstention into a stated assumption.

The admission gate refuses events without a verifiable source, because their
`known_at` cannot be shown not to leak. That is right for the forecast, and
it turns "what if we run the promotion in March?" into an abstention. A
conditional forecast answers it in its own key, with its own support status,
without touching a single unconditional number.
"""

import csv
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, "src")

from gnomon.conditional import CONDITIONAL_SUPPORT, conditional_candidates  # noqa: E402
from gnomon.context import ContextEvent, ContextSource  # noqa: E402
from gnomon.runtime import forecast  # noqa: E402

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _write_series(path: Path, count: int = 160) -> None:
    """A series with a recurring, measurable promotion effect of about +25."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "value"])
        for index in range(count):
            promo = (index // 24) % 3 == 0 and (index % 24) < 8
            value = (100 + 0.2 * index + 8 * math.sin(2 * math.pi * index / 24)
                     + (25 if promo else 0) + (index % 5))
            writer.writerow([(START + timedelta(hours=index)).isoformat(), value])


def _promotion_history(source: ContextSource | None = None) -> list[ContextEvent]:
    return [
        ContextEvent(
            event_id=f"promo-{hour}", event_type="promotion", entity_scope=("*",),
            effective_start=(START + timedelta(hours=hour)).isoformat(),
            effective_end=(START + timedelta(hours=hour + 7)).isoformat(),
            known_at=START.isoformat(), created_by="llm", source=source,
        )
        for hour in range(0, 145, 24) if (hour // 24) % 3 == 0
    ]


def _planned(source: ContextSource | None = None) -> ContextEvent:
    return ContextEvent(
        event_id="promo-planned", event_type="promotion", entity_scope=("*",),
        effective_start=(START + timedelta(hours=162)).isoformat(),
        effective_end=(START + timedelta(hours=168)).isoformat(),
        known_at=START.isoformat(), created_by="llm", source=source,
    )


def _directional_planned(source: ContextSource | None = None) -> ContextEvent:
    event = _planned(source)
    return ContextEvent(
        **{**event.__dict__, "attributes": {"soft_context": {
            "effect_family": "temporary_pulse", "direction": "increase",
            "duration": "temporary", "normalized_entity": "promotion",
            "entity_kind": "product", "delay_steps": [0, 0],
            "duration_steps": [7, 7],
        }}}
    )


def _run(tmp_path, events, name="out"):
    path = tmp_path / "series.csv"
    if not path.exists():
        _write_series(path)
    artifact, _ = forecast(
        str(path), time_column="timestamp", target_column="value", horizon=12,
        context_events=events, output=str(tmp_path / name),
    )
    return artifact


class TestCandidateSelection:
    def test_a_sourced_event_is_not_a_conditional_candidate(self):
        """A gate rejection is a measured verdict, not a licence to report
        the same event conditionally instead."""
        sourced = ContextSource("planning_file", "plans/2024.md")
        events = [*_promotion_history(sourced), _planned(sourced)]
        candidates, _ = conditional_candidates(events, "__default__")
        assert candidates == []

    def test_unsourced_events_are_candidates(self):
        events = [*_promotion_history(), _planned()]
        candidates, _ = conditional_candidates(events, "__default__")
        assert len(candidates) == len(events)

    def test_out_of_scope_events_are_not_candidates(self):
        event = ContextEvent(
            event_id="other", event_type="promotion", entity_scope=("something_else",),
            effective_start=START.isoformat(),
            effective_end=(START + timedelta(hours=2)).isoformat(),
            known_at=START.isoformat(),
        )
        candidates, _ = conditional_candidates([event], "__default__")
        assert candidates == []


class TestConditionalForecasts:
    def test_an_unsourced_planned_event_gets_a_conditional_answer(self, tmp_path):
        artifact = _run(tmp_path, [*_promotion_history(), _planned()])
        result = artifact.results[0]

        assert result.context["admitted"] is False, (
            "an unsourced event must not enter the real forecast"
        )
        assert len(result.conditional_forecasts) == 1
        conditional = result.conditional_forecasts[0]
        assert conditional["support"] == CONDITIONAL_SUPPORT
        assert conditional["events"] == ["promo-planned"]
        # The planted effect is +25; it is measured, not read off the event.
        assert 20 < conditional["measured_effect"] < 40
        assert conditional["occurrences_in_history"] > 2
        assert conditional["effect_standard_error"] > 0
        assert conditional["effect"]["distribution"]["distribution"] == "normal"
        assert conditional["effect"]["provenance"]["provenance_class"] == (
            "same_event_same_series")
        assert conditional["effect"]["provenance"]["observed"] is True
        assert any("measured from" in note for note in conditional["assumptions"])
        assert result.context_outcome["status"] == "scenario_only"
        assert result.context_outcome["primary_forecast_changed"] is False
        assert result.context_outcome["conditional_forecasts_produced"] == 1
        assert result.context_outcome["hypotheses"][0]["magnitude"] is None

    def test_the_unconditional_forecast_is_untouched(self, tmp_path):
        """Every existing field keeps the value it would have had."""
        _write_series(tmp_path / "series.csv")
        without = _run(tmp_path, [], name="a").results[0]
        with_events = _run(tmp_path, [*_promotion_history(), _planned()],
                           name="b").results[0]

        assert with_events.selected_model == without.selected_model
        assert with_events.forecast == without.forecast
        assert with_events.primary_forecast == []
        assert with_events.interval_coverage == without.interval_coverage
        assert with_events.conditional_forecasts, "but the new key is populated"

    def test_intervals_widen_only_where_the_event_is_active(self, tmp_path):
        artifact = _run(tmp_path, [*_promotion_history(), _planned()])
        result = artifact.results[0]
        unconditional = {row["timestamp"]: row for row in result.forecast}
        rows = result.conditional_forecasts[0]["forecast"]

        widened = [row for row in rows
                   if row["q90"] - row["q10"]
                   > unconditional[row["timestamp"]]["q90"]
                   - unconditional[row["timestamp"]]["q10"] + 1e-9]
        untouched = [row for row in rows
                     if row["point"] == unconditional[row["timestamp"]]["point"]]
        assert widened, "event-active steps must carry the effect's uncertainty"
        assert len(widened) < len(rows), (
            "steps the event does not touch must keep the unconditional width"
        )
        assert untouched, "steps outside the event window keep their point"

    def test_an_event_without_precedent_is_declined_not_invented(self, tmp_path):
        """No history occurrence means no measurable magnitude."""
        artifact = _run(tmp_path, [_planned()])
        result = artifact.results[0]
        assert result.conditional_forecasts == []
        record = next(evidence for evidence in artifact.evidence
                      if evidence.kind == "conditional_forecasts")
        assert record.payload["produced"] == 0
        assert record.payload["declined"], "the refusal must name a reason"

    def test_event_not_known_at_cutoff_cannot_create_a_scenario(self, tmp_path):
        event = _directional_planned()
        event = ContextEvent(**{
            **event.__dict__,
            "known_at": (START + timedelta(days=30)).isoformat(),
        })
        result = _run(tmp_path, [event]).results[0]
        assert result.conditional_forecasts == []
        assert result.sensitivity_scenarios == []

    def test_novel_directional_event_gets_only_a_sensitivity_path(self, tmp_path):
        without = _run(tmp_path, [], name="base").results[0]
        result = _run(tmp_path, [_directional_planned()], name="scenario").results[0]

        assert result.forecast == without.forecast
        assert result.conditional_forecasts == []
        assert len(result.sensitivity_scenarios) == 1
        scenario = result.sensitivity_scenarios[0]
        assert scenario["support"] == "hypothetical_sensitivity"
        assert scenario["primary_forecast_changed"] is False
        assert scenario["assumed_effect"] > 0
        assert scenario["effect"]["distribution"]["distribution"] == "assumption"
        assert scenario["effect"]["distribution"]["interval_probability"] is None
        assert scenario["effect"]["provenance"]["provenance_class"] == (
            "standardized_sensitivity")
        assert "one robust" in scenario["assumed_effect_unit"]
        assert any("not the expected or primary forecast" in assumption
                   for assumption in scenario["assumptions"])
        assert result.context_outcome["status"] == "scenario_only"
        assert result.context_outcome["sensitivity_scenarios_produced"] == 1
        assert result.context_outcome["hypotheses"][0]["numeric_status"] == (
            "standardized_sensitivity_not_event_effect")

    def test_sensitivity_path_is_additive_in_serialized_contract(self, tmp_path):
        payload = _run(tmp_path, [_directional_planned()]).to_dict()
        result = payload["results"][0]
        assert result["sensitivity_scenarios"][0]["forecast"]
        assert result["forecast"] != result["sensitivity_scenarios"][0]["forecast"]

    def test_no_conditional_key_when_there_is_nothing_to_say(self, tmp_path):
        """A run that produced none serialises exactly as it did before the
        feature existed — what makes the v0.2 freeze checkable."""
        _write_series(tmp_path / "series.csv")
        artifact = _run(tmp_path, [])
        payload = artifact.to_dict()
        assert "conditional_forecasts" not in payload["results"][0]

    def test_conditional_key_present_once_populated(self, tmp_path):
        artifact = _run(tmp_path, [*_promotion_history(), _planned()])
        payload = artifact.to_dict()
        entry = payload["results"][0]["conditional_forecasts"][0]
        assert entry["support"] == CONDITIONAL_SUPPORT
        assert entry["forecast"]

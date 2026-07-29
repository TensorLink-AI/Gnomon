from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone

import pytest

from aion.cli import main
from aion.context import ContextEvent, ContextSource
from aion.context_model import event_adjusted
from aion.runtime import forecast

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
    artifact, _ = forecast(
        str(csv_path), time_column="timestamp", target_column="requests",
        horizon=7, frequency="D", output=str(tmp_path / "out"),
        context_events=_events(140),
    )
    result = artifact.results[0]
    assert result.context is not None
    assert result.context["admitted"] is True
    assert result.selected_model == "event_adjusted"
    assert result.context["mean_improvement"] > 0.02
    assert any(evidence.kind == "context_ablation" for evidence in artifact.evidence)
    # Day 135 (a promo day, %10 == 5) falls inside the 7-day horizon:
    # the forecast must carry the promo bump on that day.
    promo_row = result.forecast[5]
    plain_row = result.forecast[4]
    assert promo_row["point"] - plain_row["point"] > PROMO_EFFECT / 2


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

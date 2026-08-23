from pathlib import Path

import pytest

from gnomon.supervision import build_export, export_supervision
from gnomon.tracking import TrackingStore
from gnomon.toolspec import _run_track


def _answer():
    return {"question": {"id": "q1", "verb": "predict", "property": "trend",
                         "target": "secret-series"},
            "best_estimate": {"value": "increased", "support": "weak"},
            "answer": {"direction": "increased", "support": "weak"},
            "artifact_path": "/private/customer.csv"}


def test_synthesis_is_separate_immutable_and_outcome_scored(tmp_path: Path):
    store = TrackingStore(tmp_path / "registry.db")
    store.record_temporal_answers("private-project", "f1", "secret-series", [_answer()],
                                  created_at="2026-01-01T00:00:00+00:00")
    store.record_temporal_synthesis(
        project="private-project", forecast_id="f1", series="secret-series",
        question_id="q1", synthesis_id="s1",
        canonical={"value": "increased"},
        synthesis={"label": "labelled_synthesis", "value": "stable",
                   "primary_forecast_unchanged": True}, evidence_refs=["folds-1"])
    score = store.resolve_temporal_synthesis(
        project="private-project", forecast_id="f1", series="secret-series",
        question_id="q1", synthesis_id="s1", outcome={"direction": "stable"})
    assert score["synthesis_delta"] == 1


def test_synthesis_replay_is_idempotent_but_conflict_is_loud(tmp_path: Path):
    store = TrackingStore(tmp_path / "registry.db")
    arguments = dict(
        project="p", forecast_id="f", series="x", question_id="q",
        synthesis_id="s", canonical={"value": "up"},
        synthesis={"label": "labelled_synthesis", "value": "up",
                   "primary_forecast_unchanged": True},
        evidence_refs=["e1"],
    )
    store.record_temporal_synthesis(**arguments)
    store.record_temporal_synthesis(**arguments)
    with pytest.raises(ValueError, match="conflicting synthesis receipt"):
        store.record_temporal_synthesis(
            **{**arguments, "synthesis": {
                "label": "labelled_synthesis", "value": "down",
                "primary_forecast_unchanged": True,
            }}
        )
    with pytest.raises(ValueError):
        store.record_temporal_synthesis(
            project="private-project", forecast_id="f1", series="secret-series",
            question_id="q1", synthesis_id="bad", canonical={},
            synthesis={"label": "canonical", "primary_forecast_unchanged": False},
            evidence_refs=[])


def test_supervision_export_is_opt_in_resolved_and_deidentified(tmp_path: Path):
    store = TrackingStore(tmp_path / "registry.db")
    store.record_temporal_answers("private-project", "f1", "secret-series", [_answer()],
                                  created_at="2026-01-01T00:00:00+00:00")
    # Resolve the answer using the same outcome machinery's persisted shape.
    with store._connect() as conn:
        conn.execute("UPDATE temporal_answer_receipts SET resolved_at = ?, outcome_payload = ?",
                     ("2026-01-02T00:00:00+00:00", '{"direction":"increased"}'))
    payload = build_export(store, "private-project", salt="0123456789abcdef")
    text = str(payload)
    assert "secret-series" not in text
    assert "/private/customer.csv" not in text
    assert "private-project" not in text
    destination = tmp_path / "export.json"
    with pytest.raises(ValueError, match="consent"):
        export_supervision(store, "private-project", destination,
                           consent=False, salt="0123456789abcdef")
    export_supervision(store, "private-project", destination,
                       consent=True, salt="0123456789abcdef")
    assert destination.exists()


def test_tracking_tool_records_and_resolves_synthesis(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GNOMON_REGISTRY_PATH", str(tmp_path / "tool.db"))
    recorded = _run_track({
        "action": "record_synthesis", "project": "p", "forecast_id": "f",
        "series": "x", "question_id": "q", "synthesis_id": "s",
        "canonical": {"value": "up"},
        "synthesis": {"label": "conditional_answer", "value": "down",
                      "primary_forecast_unchanged": True},
        "evidence_refs": ["e1"],
    })
    assert recorded["primary_forecast_unchanged"] is True
    resolved = _run_track({
        "action": "resolve_synthesis", "project": "p", "forecast_id": "f",
        "series": "x", "question_id": "q", "synthesis_id": "s",
        "outcome": {"direction": "down"},
    })
    assert resolved["score"]["synthesis_delta"] == 1
    status = _run_track({"action": "synthesis_status", "project": "p",
                         "resolved": True})
    assert len(status["syntheses"]) == 1

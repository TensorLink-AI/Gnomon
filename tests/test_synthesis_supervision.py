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


def _record_decision_comparisons(
    store: TrackingStore, *, proposer: str, synthesis_wins: bool, count: int,
) -> None:
    for index in range(count):
        canonical = "monitor" if synthesis_wins else "act"
        synthesis = "act" if synthesis_wins else "monitor"
        store.record_temporal_synthesis(
            project="p", forecast_id=f"{proposer}-{index}", series="x",
            question_id="breach", synthesis_id=f"s-{index}",
            canonical={"value": canonical},
            synthesis={
                "label": "hypothesis_ranking", "value": synthesis,
                "primary_forecast_unchanged": True,
                "scenario_role": "temporal_decision_selection",
                "candidate_origin": "model_authored_decision_prior",
                "proposer_id": proposer,
            }, evidence_refs=[f"prior-{index}", f"primary-{index}"])
        store.resolve_temporal_synthesis(
            project="p", forecast_id=f"{proposer}-{index}", series="x",
            question_id="breach", synthesis_id=f"s-{index}",
            outcome={"state": synthesis if synthesis_wins else canonical})


def test_decision_skill_graduates_helpful_not_harmful_proposer(
        tmp_path: Path, monkeypatch):
    path = tmp_path / "registry.db"
    store = TrackingStore(path)
    _record_decision_comparisons(
        store, proposer="helpful", synthesis_wins=True, count=24)
    _record_decision_comparisons(
        store, proposer="harmful", synthesis_wins=False, count=24)
    helpful, = store.decision_synthesis_skill(
        "p", proposer_id="helpful", minimum_resolved=20)
    harmful, = store.decision_synthesis_skill(
        "p", proposer_id="harmful", minimum_resolved=20)
    assert helpful["wins_vs_canonical"] == 24
    assert helpful["exact_sign_p"] < .05
    assert helpful["shrunk_net_wins_per_resolved"] > 0
    assert helpful["graduated_for_human_prior"] is True
    assert helpful["support_upgrade_allowed"] is False
    assert helpful["automation_upgrade_allowed"] is False
    assert harmful["losses_vs_canonical"] == 24
    assert harmful["graduated_for_human_prior"] is False
    monkeypatch.setenv("GNOMON_REGISTRY_PATH", str(path))
    exposed = _run_track({
        "action": "decision_skill", "project": "p",
        "proposer_id": "helpful", "min_outcomes": 20})
    assert exposed["decision_skill"][0]["graduated_for_human_prior"] is True
    assert exposed["authority"]["automation_upgrade_allowed"] is False
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
    candidates = _run_track({"action": "candidate_outcomes", "project": "p",
                             "min_outcomes": 8})
    assert candidates["candidate_outcomes"] == []
    assert candidates["authority"] == {
        "human_prior_only": True,
        "support_upgrade_allowed": False,
        "automation_upgrade_allowed": False,
    }

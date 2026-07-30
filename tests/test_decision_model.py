"""Phase 6: DecisionArtifact, realised-outcome scoring, aion status, and
legacy migration."""

from datetime import datetime, timezone
from pathlib import Path

import json

from aion.decision_model import ActionOption, DecisionArtifact, score_outcome
from aion.ids import FixedClock
from aion.macros import decide
from aion.tracking import TrackingStore

CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))


def _artifact(**overrides) -> DecisionArtifact:
    base = dict(
        decision_id="d1", project="p", forecast_id="f1",
        options=[
            ActionOption("scale_up", expected_utility=28.0, downside_risk=-20.0),
            ActionOption("wait", expected_utility=-116.0, downside_risk=-200.0),
            ActionOption("panic", feasible=False, expected_utility=999.0),
        ],
        selected_action="scale_up",
        decision_rule="maximum expected utility over feasible actions",
        scenario_probabilities={"exceed": 0.6, "no_exceed": 0.4},
        utilities={
            "scale_up": {"exceed": 100.0, "no_exceed": -20.0},
            "wait": {"exceed": -200.0, "no_exceed": 10.0},
        },
        created_at="2026-07-01T00:00:00+00:00",
    )
    base.update(overrides)
    return DecisionArtifact(**base)


def test_costly_precaution_can_be_rational():
    """The event never occurred: scale_up loses to wait in hindsight
    (positive regret) yet was ex-ante optimal. Bare 'correct' cannot say
    this; the schema must."""
    outcome = score_outcome(_artifact(), realised_scenario="no_exceed")
    assert outcome.realised_utility == -20.0
    assert outcome.best_feasible_action_in_hindsight == "wait"
    assert outcome.regret == 30.0
    assert outcome.ex_ante_optimal is True
    assert outcome.risk_calibration["predicted_probability_of_realised"] == 0.4


def test_regret_ignores_infeasible_actions():
    outcome = score_outcome(
        _artifact(),
        realised_utilities={"scale_up": 5.0, "wait": 1.0, "panic": 1000.0},
    )
    # 'panic' is infeasible: it can never be the hindsight benchmark.
    assert outcome.best_feasible_action_in_hindsight == "scale_up"
    assert outcome.regret == 0.0


def test_outcome_without_utilities_scores_nothing_silently():
    artifact = _artifact(utilities=None, options=[ActionOption("a")], selected_action="a")
    outcome = score_outcome(artifact, realised_scenario="exceed", note="observed")
    assert outcome.realised_utility is None
    assert outcome.regret is None
    assert outcome.note == "observed"


def test_store_roundtrip_and_resolution(tmp_path):
    store = TrackingStore(tmp_path / "registry.db")
    store.save_decision_artifact(_artifact())
    loaded = store.get_decision_artifact("d1")
    assert loaded.selected_action == "scale_up"
    assert loaded.resolved_at is None

    resolved = store.resolve_decision_outcome(
        "d1", realised_scenario="no_exceed",
        resolved_at="2026-07-10T00:00:00+00:00",
    )
    assert resolved.outcome.regret == 30.0
    assert store.get_decision_artifact("d1").resolved_at == "2026-07-10T00:00:00+00:00"


def test_legacy_decision_records_load_as_degraded_artifacts(tmp_path):
    store = TrackingStore(tmp_path / "registry.db")
    # Seed a v0.2-style forecast + DecisionRecord through the legacy API.
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO forecasts (forecast_id, project, series, created_at) "
            "VALUES ('f1', 'p', 's', '2026-01-01T00:00:00')"
        )
    store.record_decision("legacy1", "p", "f1", "scale up", "load stays under 80%")
    store.resolve_decision("legacy1", "load hit 85%", False)

    artifacts = store.list_decision_artifacts("p")
    assert len(artifacts) == 1
    legacy = artifacts[0]
    assert legacy.degraded is True
    assert legacy.selected_action == "scale up"
    assert legacy.outcome is not None and "correct=False" in legacy.outcome.note
    assert legacy.outcome.regret is None  # nothing is invented


def test_decide_macro_records_decision_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("AION_REGISTRY_PATH", str(tmp_path / "registry.db"))
    import importlib
    import aion.tracking as tracking_module
    importlib.reload(tracking_module)

    from datetime import date, timedelta
    start = date(2026, 3, 1)
    source = tmp_path / "trend.csv"
    source.write_text("timestamp,value\n" + "\n".join(
        f"{(start + timedelta(days=i)).isoformat()},{100 + i * 1.5}" for i in range(48)
    ) + "\n", encoding="utf-8")
    payload, _ = decide(
        str(source), time_column="timestamp", target_column="value",
        horizon=6, threshold=170.0,
        actions=[{"name": "scale_up"}, {"name": "wait"}],
        utilities={
            "scale_up": {"exceed": 100, "no_exceed": -10},
            "wait": {"exceed": -500, "no_exceed": 5},
        },
        project="ops", output=str(tmp_path / "out"), clock=CLOCK,
    )
    store = tracking_module.TrackingStore()
    artifact = store.get_decision_artifact(payload["decision_id"])
    assert artifact is not None
    assert artifact.selected_action == payload["evaluation"]["selected"]
    assert {option.name for option in artifact.options} == {"scale_up", "wait"}
    assert artifact.scenario_probabilities == payload["scenario_probabilities"]

    status = store.status("ops")
    assert status["unresolved_decisions"][0]["decision_id"] == payload["decision_id"]
    assert status["open_forecasts"]
    importlib.reload(tracking_module)


def test_status_shape_on_empty_store(tmp_path):
    store = TrackingStore(tmp_path / "registry.db")
    status = store.status()
    assert status["open_forecasts"] == []
    assert status["unresolved_decisions"] == []
    assert status["decision_summary"]["resolved"] == 0
    assert "never causal" in status["warning"]

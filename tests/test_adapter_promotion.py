from gnomon.adapter_promotion import AdapterOutcomeLedger
from gnomon.tracking import TrackingStore
import pytest


def test_shadow_ledger_recommends_but_never_auto_promotes(tmp_path) -> None:
    ledger = AdapterOutcomeLedger(tmp_path / "tracking.sqlite")
    for index in range(40):
        ledger.record(
            project="p", outcome_id=str(index), candidate="remote",
            revision="sha256:abc", baseline="last_value",
            candidate_error=8.0 if index < 30 else 12.0,
            baseline_error=10.0, known_at=f"2026-08-{index % 28 + 1:02d}T00:00:00Z")
    decision = ledger.assess(
        project="p", candidate="remote", revision="sha256:abc",
        baseline="last_value")
    assert decision.eligible is True
    assert decision.to_dict()["action"] == "review_for_promotion"
    assert decision.to_dict()["automatic_promotion"] is False
    assert decision.to_dict()["policy"] == {
        "min_outcomes": 30, "min_improvement": .05, "min_win_rate": .60}


def test_unpinned_or_underobserved_candidate_cannot_graduate(tmp_path) -> None:
    ledger = AdapterOutcomeLedger(tmp_path / "tracking.sqlite")
    ledger.record(
        project="p", outcome_id="one", candidate="remote", revision=None,
        baseline="last_value", candidate_error=1, baseline_error=10,
        known_at="2026-08-01T00:00:00Z")
    decision = ledger.assess(
        project="p", candidate="remote", revision=None,
        baseline="last_value")
    assert decision.eligible is False
    assert "candidate_revision_is_unpinned" in decision.reasons
    assert "insufficient_paired_outcomes" in decision.reasons


def test_shadow_ledger_is_as_of_replay_safe(tmp_path) -> None:
    ledger = AdapterOutcomeLedger(tmp_path / "tracking.sqlite")
    for index in range(35):
        ledger.record(
            project="p", outcome_id=str(index), candidate="remote",
            revision="r1", baseline="last", candidate_error=5,
            baseline_error=10,
            known_at="2026-08-01T00:00:00Z" if index < 20
            else "2026-09-01T00:00:00Z")
    early = ledger.assess(
        project="p", candidate="remote", revision="r1", baseline="last",
        as_of="2026-08-15T00:00:00Z")
    late = ledger.assess(
        project="p", candidate="remote", revision="r1", baseline="last",
        as_of="2026-09-02T00:00:00Z")
    assert early.paired_outcomes == 20 and early.eligible is False
    assert late.paired_outcomes == 35 and late.eligible is True


def test_shadow_ledger_rejects_invalid_errors(tmp_path) -> None:
    ledger = AdapterOutcomeLedger(tmp_path / "tracking.sqlite")
    try:
        ledger.record(
            project="p", outcome_id="x", candidate="c", revision="r",
            baseline="b", candidate_error=float("nan"), baseline_error=1,
            known_at="now")
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite shadow error was accepted")


def test_shadow_ledger_requires_zoned_knowledge_time(tmp_path) -> None:
    ledger = AdapterOutcomeLedger(tmp_path / "tracking.sqlite")
    try:
        ledger.record(
            project="p", outcome_id="x", candidate="c", revision="r",
            baseline="b", candidate_error=1, baseline_error=2,
            known_at="2026-08-01T00:00:00")
    except ValueError as error:
        assert "timezone" in str(error)
    else:
        raise AssertionError("naive shadow knowledge time was accepted")


def test_tracking_store_exposes_shadow_workflow(tmp_path) -> None:
    store = TrackingStore(tmp_path / "tracking.sqlite")
    recorded = store.record_adapter_shadow_outcome(
        project="p", outcome_id="one", candidate="api", revision="r1",
        baseline="last", candidate_error=1, baseline_error=2,
        known_at="2026-08-01T00:00:00+00:00")
    assessed = store.assess_adapter_shadow(
        project="p", candidate="api", revision="r1", baseline="last",
        min_outcomes=1, min_improvement=.1, min_win_rate=.5)
    assert recorded["status"] == "recorded"
    assert assessed["eligible"] is True
    assert assessed["automatic_promotion"] is False


def test_external_prior_excludes_target_project_and_preserves_regime(tmp_path):
    ledger = AdapterOutcomeLedger(tmp_path / "tracking.sqlite")
    regime = {"frequency_class": "subdaily"}
    for index in range(35):
        ledger.record(
            project="source", outcome_id=f"o{index}", candidate="m",
            revision="m@1", baseline="last", candidate_error=.8,
            baseline_error=1., known_at="2026-01-01T00:00:00+00:00",
            regime=regime,
        )
    ledger.record(
        project="target", outcome_id="self", candidate="m",
        revision="m@1", baseline="last", candidate_error=10.,
        baseline_error=1., known_at="2026-01-01T00:00:00+00:00",
        regime=regime,
    )
    prior = ledger.external_prior(
        candidate="m", revision="m@1", baseline="last", regime=regime,
        registry_version="r1", exclude_project="target")
    assert prior.comparisons == 35
    assert prior.mean_relative_gain == pytest.approx(.2)
    assert all(not source.startswith("target:") for source in prior.source_ids)


def _record_route_stream(ledger, errors, *, project="route", regime=None):
    regime = regime or {"frequency_class": "subdaily"}
    for index, candidate_error in enumerate(errors):
        ledger.record(
            project=project,
            outcome_id=f"{regime['frequency_class']}-o{index}",
            candidate="challenger",
            revision="sha256:route-v1", baseline="last_value",
            candidate_error=candidate_error, baseline_error=1.0,
            known_at=f"2026-01-{index + 1:02d}T00:00:00Z", regime=regime)


def test_shadow_route_requires_paired_uncertainly_bounded_evidence(tmp_path):
    ledger = AdapterOutcomeLedger(tmp_path / "tracking.sqlite")
    regime = {"frequency_class": "subdaily"}
    _record_route_stream(ledger, [.7] * 7, regime=regime)
    cold = ledger.route(
        project="route", candidate="challenger",
        revision="sha256:route-v1", champion="last_value",
        regime=regime, as_of="2026-01-08T00:00:00Z")
    assert cold.recommendation == "last_value"
    assert "insufficient_paired_outcomes" in cold.reasons

    _record_route_stream(
        ledger, [.7] * 8, project="graduated", regime=regime)
    routed = ledger.route(
        project="graduated", candidate="challenger",
        revision="sha256:route-v1", champion="last_value",
        regime=regime, as_of="2026-01-09T00:00:00Z")
    payload = routed.to_dict()
    assert routed.recommendation == "challenger"
    assert routed.win_rate_wilson_95_lower > .5
    assert payload["recommended_pool"] == ["challenger", "last_value"]
    assert payload["automatic_promotion"] is False
    assert payload["automation_eligible"] is False
    assert payload["job_local_admission_required"] is True
    assert payload["routing_authority"] == "candidate_pool_only"
    assert "last_value" in payload["rollback_condition"]


def test_shadow_route_rolls_back_after_two_harmful_outcomes(tmp_path):
    ledger = AdapterOutcomeLedger(tmp_path / "tracking.sqlite")
    regime = {"frequency_class": "subdaily"}
    _record_route_stream(ledger, [.7] * 12 + [1.4, 1.4], regime=regime)
    decision = ledger.route(
        project="route", candidate="challenger",
        revision="sha256:route-v1", champion="last_value",
        regime=regime, as_of="2026-01-15T00:00:00Z")
    assert decision.recommendation == "last_value"
    assert "recent_performance_degraded" in decision.reasons
    assert decision.recent_mean_relative_improvement < 0


def test_shadow_route_is_exact_regime_and_point_in_time_safe(tmp_path):
    ledger = AdapterOutcomeLedger(tmp_path / "tracking.sqlite")
    subdaily = {"frequency_class": "subdaily"}
    daily = {"frequency_class": "daily_weekly"}
    _record_route_stream(ledger, [.7] * 8, regime=subdaily)
    _record_route_stream(ledger, [1.4] * 8, regime=daily)
    kwargs = dict(
        project="route", candidate="challenger",
        revision="sha256:route-v1", champion="last_value",
        as_of="2026-01-09T00:00:00Z")
    good = ledger.route(**kwargs, regime=subdaily)
    bad = ledger.route(**kwargs, regime=daily)
    assert good.paired_outcomes == bad.paired_outcomes == 8
    assert good.recommendation == "challenger"
    assert bad.recommendation == "last_value"

    before = good.to_dict()
    ledger.record(
        project="route", outcome_id="future", candidate="challenger",
        revision="sha256:route-v1", baseline="last_value",
        candidate_error=0.0, baseline_error=1.0,
        known_at="2026-02-01T10:00:00+10:00", regime=subdaily)
    after = ledger.route(**kwargs, regime=subdaily).to_dict()
    assert after == before


def test_shadow_route_never_routes_an_unpinned_candidate(tmp_path):
    ledger = AdapterOutcomeLedger(tmp_path / "tracking.sqlite")
    regime = {"frequency_class": "subdaily"}
    for index in range(8):
        ledger.record(
            project="route", outcome_id=str(index), candidate="challenger",
            revision=None, baseline="last_value", candidate_error=.1,
            baseline_error=1.0,
            known_at=f"2026-01-{index + 1:02d}T00:00:00Z", regime=regime)
    decision = ledger.route(
        project="route", candidate="challenger", revision=None,
        champion="last_value", regime=regime,
        as_of="2026-01-09T00:00:00Z")
    assert decision.recommendation == "last_value"
    assert "candidate_revision_is_unpinned" in decision.reasons


def test_shadow_route_requires_an_explicit_cohort(tmp_path):
    ledger = AdapterOutcomeLedger(tmp_path / "tracking.sqlite")
    with pytest.raises(ValueError, match="non-empty exact cohort"):
        ledger.route(
            project="route", candidate="challenger", revision="r1",
            champion="last_value", regime={},
            as_of="2026-01-01T00:00:00Z")


def test_track_tool_preserves_routing_authority_and_regime(tmp_path, monkeypatch):
    monkeypatch.setenv("GNOMON_REGISTRY_PATH", str(tmp_path / "tracking.sqlite"))
    from gnomon.toolspec import _run_track

    regime = {"frequency_class": "subdaily"}
    for index in range(8):
        recorded = _run_track({
            "action": "record_adapter_shadow", "project": "tool-route",
            "outcome_id": str(index), "candidate": "challenger",
            "revision": "sha256:route-v1", "baseline": "last_value",
            "candidate_error": .7, "baseline_error": 1.0,
            "known_at": f"2026-01-{index + 1:02d}T00:00:00Z",
            "regime": regime,
        })
        assert recorded["regime"] == regime
    routed = _run_track({
        "action": "route_adapter_shadow", "project": "tool-route",
        "candidate": "challenger", "revision": "sha256:route-v1",
        "baseline": "last_value", "regime": regime,
        "as_of": "2026-01-09T00:00:00Z",
    })
    assert routed["recommendation"] == "challenger"
    assert routed["regime"] == regime
    assert routed["routing_authority"] == "candidate_pool_only"
    assert routed["automation_eligible"] is False
    assert routed["job_local_admission_required"] is True

from gnomon.adapter_promotion import AdapterOutcomeLedger
from gnomon.tracking import TrackingStore


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

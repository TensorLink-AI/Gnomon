"""The per-proposer calibration ledger (tracking schema 5).

The loop this closes, measured open in docs/design/news-regime.md §1.2:
the admission gate's verdicts went to write-only evidence, proposer
identity died before the gate, and the history-only counterfactual was
computed then discarded — so no proposal could ever be scored when
actuals arrived. Here: proposals and verdicts land in the registry at
registration time, the pipeline persists the counterfactual on
admission, actuals resolve realised lift, and skill aggregates shrunk
toward no-skill priors.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone

import importlib

import pytest

import gnomon.tracking as tracking
from gnomon.context import ContextEvent, ContextSource
from gnomon.runtime import forecast
from gnomon.tracking import PROPOSER_SKILL_SHRINKAGE, proposal_key


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A per-test registry: DEFAULT_REGISTRY_PATH is resolved at module
    import, so the env override needs a reload (the pattern
    test_decision_model.py established)."""
    monkeypatch.setenv("GNOMON_REGISTRY_PATH", str(tmp_path / "registry.db"))
    importlib.reload(tracking)
    yield
    monkeypatch.delenv("GNOMON_REGISTRY_PATH")
    importlib.reload(tracking)

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
PROMO_EFFECT = 30.0


def _values(length: int) -> list[float]:
    promos = {day for day in range(length) if day % 10 in (5, 6)}
    return [100.0 + 0.5 * day + (PROMO_EFFECT if day in promos else 0.0)
            for day in range(length)]


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
                attributes={"proposer": {"proposer_id": "llm:test-model",
                                         "kind": "llm"}},
            ))
    return events


def _write_csv(path, length: int) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "requests"])
        for day, value in enumerate(_values(length)):
            writer.writerow([(START + timedelta(days=day)).isoformat(), value])


def _admitted_run(tmp_path):
    csv_path = tmp_path / "promo.csv"
    _write_csv(csv_path, 130)
    events = _events(140)
    artifact, path = forecast(
        str(csv_path), time_column="timestamp", target_column="requests",
        horizon=7, frequency="D", output=str(tmp_path / "out"),
        context_events=events,
    )
    assert artifact.results[0].context["admitted"] is True
    return artifact, path, events


class TestProposalKey:
    def event(self, **overrides) -> ContextEvent:
        fields = dict(
            event_id="event_llm_00", event_type="promotion",
            entity_scope=("*",),
            effective_start="2026-05-01T00:00:00+00:00",
            effective_end="2026-05-02T00:00:00+00:00",
            known_at="2026-04-30T00:00:00+00:00",
            source=ContextSource("calendar", "promo.ics"),
        )
        fields.update(overrides)
        return ContextEvent(**fields)

    def test_key_is_the_claim_not_the_run(self):
        # Run-local ids and confidence are not identity: the same claim
        # re-proposed later must join the same ledger row family.
        a = self.event()
        b = self.event(event_id="event_llm_07", confidence=0.9)
        assert proposal_key("s", a) == proposal_key("s", b)

    def test_different_windows_are_different_claims(self):
        a = self.event()
        b = self.event(effective_start="2026-06-01T00:00:00+00:00")
        assert proposal_key("s", a) != proposal_key("s", b)
        assert proposal_key("other", a) != proposal_key("s", a)


class TestLedgerEndToEnd:
    def test_admission_counterfactual_outcome_and_skill(self, tmp_path,
                                                        registry):
        artifact, path, events = _admitted_run(tmp_path)

        # The pipeline persisted the history-only path at admission.
        counterfactuals = [item for item in artifact.evidence
                           if item.kind == "enrichment_counterfactual"]
        assert len(counterfactuals) == 1
        payload = counterfactuals[0].payload
        assert len(payload["points"]) == 7
        assert payload["base_model"] not in (None, "event_adjusted")

        tracking.register_artifact(artifact, "ledger-test", str(path),
                                   context_events=events)
        store = tracking.TrackingStore()
        effects = store.event_effects("ledger-test")
        assert len(effects) == 1
        assert effects[0]["provenance_class"] == "same_event_same_series"
        assert effects[0]["effect_distribution"]["distribution"] == "empirical"
        assert effects[0]["estimate"] is None
        skill = store.proposer_skill("ledger-test")
        assert len(skill) == 1
        row = skill[0]
        assert row["proposer_id"] == "llm:test-model"
        assert row["proposer_kind"] == "llm"
        assert row["event_type"] == "promotion"
        assert row["proposals"] == len(events)
        assert row["admitted"] == len(events)
        assert row["published"] == len(events)
        assert row["resolved"] == 0
        assert row["shrunk_lift_wape"] == 0.0  # cold: no resolved outcomes

        # Actuals arrive: outcomes resolve with set-level realised lift.
        truth = _values(137)[130:]
        actuals = [((START + timedelta(days=130 + step)).isoformat(), value)
                   for step, value in enumerate(truth)]
        results = store.submit_actuals("ledger-test", actuals)
        assert len(results) == 1
        realised = store.event_effects(
            "ledger-test", include_unresolved=False)[0]
        assert realised["sample_count"] > 0
        assert realised["outcome_known_at"] is not None

        skill = store.proposer_skill("ledger-test")[0]
        assert skill["resolved"] == len(events)
        # The promo event genuinely improves the forecast on this series,
        # so the realised lift against the history-only path is positive.
        assert skill["mean_lift_wape"] > 0
        expected = (skill["mean_lift_wape"] * skill["resolved"]
                    / (skill["resolved"] + PROPOSER_SKILL_SHRINKAGE))
        assert abs(skill["shrunk_lift_wape"] - expected) < 1e-12
        assert 0.0 < skill["shrunk_lift_wape"] < skill["mean_lift_wape"]

        # The outcome row records both sides of the comparison.
        with store._connect() as conn:
            outcome = conn.execute("SELECT * FROM event_outcomes").fetchone()
        assert outcome["attribution"] == "event_set"
        assert outcome["base_wape"] > outcome["published_wape"]
        assert outcome["direction_hit"] is None and outcome["brier"] is None

    def test_excluded_events_are_recorded_unadmitted(self, tmp_path,
                                                     registry):
        csv_path = tmp_path / "promo.csv"
        _write_csv(csv_path, 130)
        unsourced = [ContextEvent(
            event_id="rumor_00", event_type="promotion", entity_scope=("*",),
            effective_start=(START + timedelta(days=135)).isoformat(),
            effective_end=(START + timedelta(days=136)).isoformat(),
            known_at=START.isoformat(),
        )]
        artifact, path = forecast(
            str(csv_path), time_column="timestamp", target_column="requests",
            horizon=7, frequency="D", output=str(tmp_path / "out"),
            context_events=unsourced,
        )
        tracking.register_artifact(artifact, "ledger-test", str(path),
                                   context_events=unsourced)
        store = tracking.TrackingStore()
        with store._connect() as conn:
            admission = conn.execute(
                "SELECT * FROM event_admissions").fetchone()
        assert admission["admitted"] == 0
        assert admission["published"] == 0
        assert "verifiable source" in (admission["exclusion_reason"] or "")
        row = store.proposer_skill("ledger-test")[0]
        assert row["proposer_id"] == "user"  # no structured proposer given
        assert row["admitted"] == 0

    def test_no_events_no_counterfactual_no_ledger_rows(self, tmp_path,
                                                        registry):
        csv_path = tmp_path / "plain.csv"
        _write_csv(csv_path, 130)
        artifact, path = forecast(
            str(csv_path), time_column="timestamp", target_column="requests",
            horizon=7, frequency="D", output=str(tmp_path / "out"),
        )
        assert not any(item.kind == "enrichment_counterfactual"
                       for item in artifact.evidence)
        tracking.register_artifact(artifact, "ledger-test", str(path))
        assert tracking.TrackingStore().proposer_skill("ledger-test") == []


class TestWorkflowProposerIdentity:
    def _respond(self, proposer=None, attributes=None):
        from gnomon.workflows import DocumentRef, parse_context_response

        documents = [DocumentRef(name="news", content="promo announced",
                                 source_type="artifact", reference="doc-1")]
        proposal = {
            "document_index": 0, "event_type": "promotion",
            "entity_scope": ["*"],
            "effective_start": "2026-05-01T00:00:00+00:00",
            "effective_end": "2026-05-02T00:00:00+00:00",
            "known_at": "2026-04-30T00:00:00+00:00",
        }
        if attributes:
            proposal["attributes"] = attributes
        return parse_context_response({"events": [proposal]}, documents,
                                      proposer=proposer)

    def test_caller_proposer_is_recorded(self):
        response = self._respond(
            proposer={"proposer_id": "llm:deepseek", "kind": "llm"})
        assert response["events"][0]["attributes"]["proposer"] == {
            "proposer_id": "llm:deepseek", "kind": "llm"}

    def test_model_supplied_proposer_is_discarded(self):
        # A model writing its own proposer identity could impersonate a
        # proposer with an earned track record — same discipline as
        # `source_span` and `claim`.
        response = self._respond(
            attributes={"proposer": {"proposer_id": "llm:trusted-veteran"}})
        assert "proposer" not in response["events"][0]["attributes"]

    def test_no_proposer_means_no_attribute(self):
        # Absence keeps event serialization — and therefore the forecast
        # id payload of runs that embed these events — byte-identical.
        response = self._respond()
        assert "proposer" not in response["events"][0]["attributes"]

"""Typed numeric claims: bounds are admissible, asserted values are not.

The line this module defends: a caller may tell Aion which values are
*possible* (returns cannot be negative, the line caps at 500) and Aion will
project its answer onto that region. A caller may not tell Aion what the
answer *is*. The first is a fact about the domain; the second would make an
LLM's say-so into a forecast number.
"""

import csv
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, "src")

from aion.constraints import (  # noqa: E402
    apply_claims,
    collect_claims,
    history_violations,
    parse_claim,
)
from aion.context import ContextEvent  # noqa: E402
from aion.runtime import forecast  # noqa: E402

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
FAR = datetime(2030, 1, 1, tzinfo=timezone.utc)


def _event(kind, value, event_id="c1", start=START, end=FAR, event_type=None):
    return ContextEvent(
        event_id=event_id,
        event_type=event_type or f"constraint:{kind}",
        entity_scope=("*",),
        effective_start=start.isoformat(), effective_end=end.isoformat(),
        known_at=START.isoformat(),
        attributes={"claim": {"kind": kind, "value": value}},
    )


def _series(path: Path, count: int = 160) -> None:
    """A rising series whose history peaks below 194 but whose forecast
    interval runs past 200 — so a capacity cap of 200 binds the answer
    without being a bound the training window already breaches."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "value"])
        for index in range(count):
            value = (100 + 0.5 * index + 15 * math.sin(2 * math.pi * index / 24)
                     + (index % 7))
            writer.writerow([(START + timedelta(hours=index)).isoformat(), value])


class TestClaimParsing:
    def test_a_bound_parses(self):
        claim, problem = parse_claim(_event("max", 500.0))
        assert problem is None
        assert claim.kind == "max" and claim.value == 500.0

    def test_an_asserted_value_is_refused_with_the_reason(self):
        for kind in ("pin", "exact", "equals"):
            claim, problem = parse_claim(_event(kind, 412.0))
            assert claim is None
            assert "not admissible" in problem
            assert "covariate" in problem, (
                "a refusal should name the admissible route for the same "
                "information"
            )

    def test_an_event_without_a_claim_is_not_an_error(self):
        event = ContextEvent(
            event_id="plain", event_type="promotion", entity_scope=("*",),
            effective_start=START.isoformat(), effective_end=FAR.isoformat(),
            known_at=START.isoformat(),
        )
        claim, problem = parse_claim(event)
        assert claim is None and problem is None

    def test_non_finite_values_are_refused(self):
        event = _event("max", 1.0)
        event.attributes["claim"]["value"] = float("inf")
        claim, problem = parse_claim(event)
        assert claim is None and "finite" in problem


class TestHistoryValidation:
    def test_a_bound_history_already_breaches_is_rejected(self):
        values = [10.0, 5.0, 12.0]
        stamps = [START + timedelta(hours=i) for i in range(3)]
        claim, _ = parse_claim(_event("min", 8.0))
        violations = history_violations(claim, values, stamps)
        assert violations == [stamps[1].isoformat()]

    def test_collect_reports_the_rejection_rather_than_dropping_it(self):
        values = [10.0, 5.0, 12.0]
        stamps = [START + timedelta(hours=i) for i in range(3)]
        claims, rejected = collect_claims([_event("min", 8.0)], "__default__",
                                          values, stamps)
        assert claims == []
        assert rejected[0]["violating_timestamps"]
        assert "history breaches" in rejected[0]["reason"]

    def test_a_bound_history_respects_is_kept(self):
        values = [10.0, 9.0, 12.0]
        stamps = [START + timedelta(hours=i) for i in range(3)]
        claims, rejected = collect_claims([_event("min", 8.0)], "__default__",
                                          values, stamps)
        assert len(claims) == 1 and rejected == []


class TestProjection:
    @staticmethod
    def _rows():
        # Declines through zero, so a `min: 0` floor has something to do.
        return [{"timestamp": (START + timedelta(hours=i)).isoformat(),
                 "point": 60.0 - 50 * i, "q10": 40.0 - 50 * i,
                 "q50": 60.0 - 50 * i, "q90": 80.0 - 50 * i}
                for i in range(3)]

    def test_clamping_preserves_quantile_order(self):
        claim, _ = parse_claim(_event("min", 0.0))
        rows, _ = apply_claims(self._rows(), [claim])
        for row in rows:
            assert row["q10"] <= row["point"] <= row["q90"]
            assert row["q10"] >= 0.0

    def test_clamping_is_idempotent(self):
        claim, _ = parse_claim(_event("min", 0.0))
        once, first = apply_claims(self._rows(), [claim])
        twice, second = apply_claims(once, [claim])
        assert once == twice
        assert first and second == [], (
            "a second application must have nothing left to change"
        )

    def test_the_clamp_records_what_it_moved(self):
        claim, _ = parse_claim(_event("min", 0.0))
        _, applications = apply_claims(self._rows(), [claim])
        assert applications
        entry = applications[0]
        assert entry["kind"] == "min" and entry["value"] == 0.0
        assert entry["before"], "the pre-clamp value must be recoverable"

    def test_a_claim_outside_its_window_does_not_bind(self):
        early_end = START + timedelta(hours=1)
        claim, _ = parse_claim(_event("min", 0.0, start=START, end=early_end))
        rows, applications = apply_claims(self._rows(), [claim])
        assert rows[2]["point"] == self._rows()[2]["point"]
        assert all(entry["timestamp"] != rows[2]["timestamp"]
                   for entry in applications)


class TestEndToEnd:
    def test_a_capacity_cap_is_applied_and_disclosed(self, tmp_path):
        path = tmp_path / "series.csv"
        _series(path)
        cap = 200.0
        artifact, _ = forecast(
            str(path), time_column="timestamp", target_column="value",
            horizon=12, context_events=[_event("max", cap)],
            output=str(tmp_path / "out"),
        )
        result = artifact.results[0]
        assert all(row["q90"] <= cap for row in result.forecast)
        assert all(row["q10"] <= row["point"] <= row["q90"]
                   for row in result.forecast)
        record = next(item for item in artifact.evidence
                      if item.kind == "constraint_applied")
        assert record.payload["claims_applied"] == 1
        assert record.payload["clamps"], "an applied clamp must be recorded"

    def test_a_cap_the_history_breaches_is_refused_not_enforced(self, tmp_path):
        path = tmp_path / "series.csv"
        _series(path)
        artifact, _ = forecast(
            str(path), time_column="timestamp", target_column="value",
            horizon=12, context_events=[_event("max", 150.0)],
            output=str(tmp_path / "out"),
        )
        result = artifact.results[0]
        record = next(item for item in artifact.evidence
                      if item.kind == "constraint_applied")
        assert record.payload["claims_applied"] == 0
        assert record.payload["rejected"], (
            "a refused bound must be disclosed, or the caller believes it "
            "was enforced"
        )
        assert any(row["q90"] > 150.0 for row in result.forecast)

    def test_a_run_without_claims_is_unchanged(self, tmp_path):
        """The goldens cover this too; this states it directly."""
        path = tmp_path / "series.csv"
        _series(path)
        plain, _ = forecast(str(path), time_column="timestamp",
                            target_column="value", horizon=12,
                            output=str(tmp_path / "a"))
        assert not any(item.kind == "constraint_applied"
                       for item in plain.evidence)

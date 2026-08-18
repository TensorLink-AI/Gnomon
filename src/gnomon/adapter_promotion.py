"""Outcome-backed shadow evaluation for forecast adapters.

The ledger measures a challenger beside the published candidate. It only
returns an auditable recommendation; changing the production candidate remains
an explicit configuration/deployment action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import sqlite3
import statistics
from pathlib import Path
from datetime import datetime


@dataclass(frozen=True)
class PromotionDecision:
    candidate: str
    revision: str
    baseline: str
    paired_outcomes: int
    mean_relative_improvement: float | None
    win_rate: float | None
    eligible: bool
    reasons: tuple[str, ...]
    min_outcomes: int
    min_improvement: float
    min_win_rate: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        payload["action"] = "review_for_promotion" if self.eligible else "keep_shadowing"
        payload["automatic_promotion"] = False
        payload["policy"] = {
            key: payload.pop(key) for key in (
                "min_outcomes", "min_improvement", "min_win_rate")
        }
        return payload


class AdapterOutcomeLedger:
    """Small SQLite ledger of paired, realized candidate errors."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        with sqlite3.connect(self.path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS adapter_shadow_outcomes (
                    project TEXT NOT NULL,
                    outcome_id TEXT NOT NULL,
                    candidate TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    baseline TEXT NOT NULL,
                    candidate_error REAL NOT NULL,
                    baseline_error REAL NOT NULL,
                    known_at TEXT NOT NULL,
                    PRIMARY KEY (project, outcome_id, candidate, revision, baseline)
                )
            """)

    def record(self, *, project: str, outcome_id: str, candidate: str,
               revision: str | None, baseline: str, candidate_error: float,
               baseline_error: float, known_at: str) -> None:
        values = (float(candidate_error), float(baseline_error))
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("shadow errors must be finite and non-negative")
        _validate_timestamp(known_at, "known_at")
        with sqlite3.connect(self.path) as connection:
            connection.execute("""
                INSERT OR REPLACE INTO adapter_shadow_outcomes
                (project, outcome_id, candidate, revision, baseline,
                 candidate_error, baseline_error, known_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (project, outcome_id, candidate, revision or "unversioned",
                  baseline, *values, known_at))

    def assess(self, *, project: str, candidate: str, revision: str | None,
               baseline: str, as_of: str | None = None,
               min_outcomes: int = 30, min_improvement: float = .05,
               min_win_rate: float = .60) -> PromotionDecision:
        revision_key = revision or "unversioned"
        if as_of is not None:
            _validate_timestamp(as_of, "as_of")
        query = """SELECT candidate_error, baseline_error
                   FROM adapter_shadow_outcomes
                   WHERE project=? AND candidate=? AND revision=? AND baseline=?"""
        arguments: list[object] = [project, candidate, revision_key, baseline]
        if as_of is not None:
            query += " AND known_at <= ?"
            arguments.append(as_of)
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(query, arguments).fetchall()
        improvements = [
            (base - contender) / base
            for contender, base in rows if base > 1e-12
        ]
        wins = [contender < base for contender, base in rows]
        mean_improvement = (statistics.mean(improvements)
                            if improvements else None)
        win_rate = statistics.mean(wins) if wins else None
        reasons: list[str] = []
        if revision is None:
            reasons.append("candidate_revision_is_unpinned")
        if len(rows) < min_outcomes:
            reasons.append("insufficient_paired_outcomes")
        if mean_improvement is None or mean_improvement < min_improvement:
            reasons.append("mean_improvement_below_gate")
        if win_rate is None or win_rate < min_win_rate:
            reasons.append("win_rate_below_gate")
        return PromotionDecision(
            candidate, revision_key, baseline, len(rows), mean_improvement,
            win_rate, not reasons, tuple(reasons), min_outcomes,
            min_improvement, min_win_rate)


def _validate_timestamp(value: str, field: str) -> None:
    if not value:
        raise ValueError(f"{field} is required for replay-safe outcomes")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone offset")

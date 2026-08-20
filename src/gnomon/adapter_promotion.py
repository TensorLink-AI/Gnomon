"""Outcome-backed shadow evaluation for forecast adapters.

The ledger measures a challenger beside the published candidate. It only
returns an auditable recommendation; changing the production candidate remains
an explicit configuration/deployment action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import json
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
            columns = {row[1] for row in connection.execute(
                "PRAGMA table_info(adapter_shadow_outcomes)")}
            if "regime_json" not in columns:
                connection.execute(
                    "ALTER TABLE adapter_shadow_outcomes ADD COLUMN regime_json TEXT")

    def record(self, *, project: str, outcome_id: str, candidate: str,
               revision: str | None, baseline: str, candidate_error: float,
               baseline_error: float, known_at: str,
               regime: dict[str, str] | None = None) -> None:
        values = (float(candidate_error), float(baseline_error))
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("shadow errors must be finite and non-negative")
        _validate_timestamp(known_at, "known_at")
        with sqlite3.connect(self.path) as connection:
            connection.execute("""
                INSERT OR REPLACE INTO adapter_shadow_outcomes
                (project, outcome_id, candidate, revision, baseline,
                 candidate_error, baseline_error, known_at, regime_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (project, outcome_id, candidate, revision or "unversioned",
                  baseline, *values, known_at,
                  json.dumps(regime, sort_keys=True) if regime else None))

    def external_prior(
        self, *, candidate: str, revision: str, baseline: str,
        regime: dict[str, str], registry_version: str,
        exclude_project: str | None = None, min_outcomes: int = 30,
    ):
        """Compile transfer evidence without counting the target project.

        The target's outcomes remain local evidence; including them here
        would count the same observations twice and call them independent.
        """
        from .admission import ExternalModelPrior
        query = """SELECT project, outcome_id, candidate_error, baseline_error
                   FROM adapter_shadow_outcomes
                   WHERE candidate=? AND revision=? AND baseline=?
                     AND regime_json=?"""
        arguments: list[object] = [
            candidate, revision, baseline, json.dumps(regime, sort_keys=True)]
        if exclude_project is not None:
            query += " AND project != ?"
            arguments.append(exclude_project)
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(query, arguments).fetchall()
        gains = [(base - contender) / base
                 for _, _, contender, base in rows if base > 1e-12]
        if len(gains) < max(2, min_outcomes):
            return None
        standard_error = max(
            statistics.stdev(gains) / math.sqrt(len(gains)), 1e-6)
        return ExternalModelPrior(
            model=candidate, revision=revision,
            regime=tuple(sorted(regime.items())), comparisons=len(gains),
            mean_relative_gain=statistics.mean(gains),
            standard_error=standard_error,
            source_ids=tuple(
                f"{project}:{outcome_id}" for project, outcome_id, _, _ in rows),
            registry_version=registry_version, overlap_risk="low",
            baseline_reference=("strongest_robust_baseline"
                                if baseline == "strongest_robust_baseline"
                                else baseline),
        )

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

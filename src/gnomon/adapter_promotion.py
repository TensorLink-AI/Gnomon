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
from datetime import datetime, timezone

from .ids import content_id


MIN_ROUTE_OUTCOMES = 8
MIN_ROUTE_IMPROVEMENT = .05
MIN_ROUTE_WILSON_LOWER = .5
ROUTE_RECENT_WINDOW = 4


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


@dataclass(frozen=True)
class ShadowRoutingDecision:
    """Point-in-time candidate-pool advice; never forecast authority."""

    project: str
    candidate: str
    revision: str
    champion: str
    regime: dict[str, str]
    as_of: str
    paired_outcomes: int
    mean_relative_improvement: float | None
    win_rate: float | None
    win_rate_wilson_95_lower: float | None
    recent_outcomes: int
    recent_mean_relative_improvement: float | None
    recent_win_rate: float | None
    recommendation: str
    recommended_pool: tuple[str, ...]
    reasons: tuple[str, ...]
    receipt_id: str

    def to_dict(self) -> dict[str, object]:
        candidate_selected = self.recommendation == self.candidate
        return {
            **asdict(self),
            "recommended_pool": list(self.recommended_pool),
            "reasons": list(self.reasons),
            "action": (
                "consider_challenger_with_local_admission"
                if candidate_selected else "keep_or_rollback_to_champion"
            ),
            "automatic_promotion": False,
            "automation_eligible": False,
            "job_local_admission_required": True,
            "routing_authority": "candidate_pool_only",
            "rollback_condition": (
                f"Use {self.champion!r} when the last {ROUTE_RECENT_WINDOW} "
                "paired outcomes have non-positive mean relative improvement "
                "or win rate below 0.5; every new forecast must also reject "
                "the challenger when its job-local admission fails."
            ),
            "policy": {
                "minimum_paired_outcomes": MIN_ROUTE_OUTCOMES,
                "minimum_mean_relative_improvement": MIN_ROUTE_IMPROVEMENT,
                "minimum_win_rate_wilson_95_lower": MIN_ROUTE_WILSON_LOWER,
                "recent_window": ROUTE_RECENT_WINDOW,
                "recent_minimum_mean_relative_improvement": 0.0,
                "recent_minimum_win_rate": 0.5,
            },
        }


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
        known_at = _normalise_timestamp(known_at, "known_at")
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

    def route(
        self, *, project: str, candidate: str, revision: str | None,
        champion: str, regime: dict[str, str], as_of: str,
    ) -> ShadowRoutingDecision:
        """Recommend a candidate pool from paired outcomes known by ``as_of``.

        This can nominate a pinned challenger for the next contest, but the
        next forecast still owns the numeric path and must admit that
        challenger against its mandatory baselines.
        """
        if not regime:
            raise ValueError("regime must be a non-empty exact cohort")
        cutoff = _parse_timestamp(as_of, "as_of")
        cutoff_text = cutoff.isoformat()
        revision_key = revision or "unversioned"
        regime_json = json.dumps(regime, sort_keys=True)
        query = """SELECT outcome_id, candidate_error, baseline_error, known_at
                   FROM adapter_shadow_outcomes
                   WHERE project=? AND candidate=? AND revision=? AND baseline=?
                     AND regime_json=?"""
        arguments = [project, candidate, revision_key, champion, regime_json]
        with sqlite3.connect(self.path) as connection:
            raw_rows = connection.execute(query, arguments).fetchall()
        rows = []
        for outcome_id, candidate_error, baseline_error, known_at in raw_rows:
            try:
                known = _parse_timestamp(str(known_at), "known_at")
            except ValueError:
                # A legacy malformed row cannot acquire routing authority.
                continue
            if known <= cutoff and float(baseline_error) > 1e-12:
                rows.append((known, str(outcome_id), float(candidate_error),
                             float(baseline_error)))
        rows.sort(key=lambda row: (row[0], row[1]))
        improvements = [
            (baseline_error - candidate_error) / baseline_error
            for _, _, candidate_error, baseline_error in rows
        ]
        wins = [candidate_error < baseline_error
                for _, _, candidate_error, baseline_error in rows]
        count = len(improvements)
        mean_improvement = statistics.mean(improvements) if improvements else None
        win_rate = statistics.mean(wins) if wins else None
        lower = _wilson_lower(sum(wins), count) if count else None
        recent_improvements = improvements[-ROUTE_RECENT_WINDOW:]
        recent_wins = wins[-ROUTE_RECENT_WINDOW:]
        recent_mean = (statistics.mean(recent_improvements)
                       if recent_improvements else None)
        recent_win_rate = statistics.mean(recent_wins) if recent_wins else None
        reasons: list[str] = []
        if revision is None:
            reasons.append("candidate_revision_is_unpinned")
        if count < MIN_ROUTE_OUTCOMES:
            reasons.append("insufficient_paired_outcomes")
        if (mean_improvement is None
                or mean_improvement < MIN_ROUTE_IMPROVEMENT):
            reasons.append("mean_improvement_below_gate")
        if lower is None or lower <= MIN_ROUTE_WILSON_LOWER:
            reasons.append("win_rate_uncertainty_below_gate")
        if (count >= MIN_ROUTE_OUTCOMES
                and len(recent_improvements) == ROUTE_RECENT_WINDOW
                and (recent_mean is None or recent_mean <= 0
                     or recent_win_rate is None or recent_win_rate < .5)):
            reasons.append("recent_performance_degraded")
        recommendation = candidate if not reasons else champion
        pool = ((candidate, champion) if recommendation == candidate
                else (champion,))
        identity = {
            "project": project, "candidate": candidate,
            "revision": revision_key, "champion": champion,
            "regime": regime, "as_of": cutoff_text,
            "paired_outcomes": count,
            "outcomes": [row[1] for row in rows],
            "mean_relative_improvement": mean_improvement,
            "win_rate": win_rate,
            "win_rate_wilson_95_lower": lower,
            "recent_mean_relative_improvement": recent_mean,
            "recent_win_rate": recent_win_rate,
            "recommendation": recommendation, "reasons": reasons,
        }
        return ShadowRoutingDecision(
            project, candidate, revision_key, champion, dict(regime),
            cutoff_text, count, mean_improvement, win_rate, lower,
            len(recent_improvements), recent_mean, recent_win_rate,
            recommendation, pool, tuple(reasons),
            content_id("shadow_route", identity),
        )

    def assess(self, *, project: str, candidate: str, revision: str | None,
               baseline: str, as_of: str | None = None,
               min_outcomes: int = 30, min_improvement: float = .05,
               min_win_rate: float = .60) -> PromotionDecision:
        revision_key = revision or "unversioned"
        if as_of is not None:
            as_of = _normalise_timestamp(as_of, "as_of")
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


def _parse_timestamp(value: str, field: str) -> datetime:
    if not value:
        raise ValueError(f"{field} is required for replay-safe outcomes")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone offset")
    return parsed.astimezone(timezone.utc)


def _normalise_timestamp(value: str, field: str) -> str:
    return _parse_timestamp(value, field).isoformat()


def _validate_timestamp(value: str, field: str) -> None:
    _parse_timestamp(value, field)


def _wilson_lower(wins: int, count: int) -> float:
    if count < 1:
        return 0.0
    z = 1.959963984540054
    rate = wins / count
    denominator = 1 + z * z / count
    centre = rate + z * z / (2 * count)
    margin = z * math.sqrt(
        rate * (1 - rate) / count + z * z / (4 * count * count))
    return max(0.0, (centre - margin) / denominator)

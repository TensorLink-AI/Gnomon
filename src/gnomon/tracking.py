"""Persistent forecast tracking, realised scoring, and model performance.

A lightweight SQLite-based registry that turns Gnomon from a one-shot
forecasting tool into a system that learns from its own accuracy.

Three core capabilities:

1. **Registry**: every forecast is indexed with its project, model,
   support, threshold, and artifact path. No daemon — just a file.

2. **Scoring**: submit actual values and Gnomon computes MASE, MAPE,
   bias, interval coverage, and threshold accuracy for each forecast.

3. **Model performance**: aggregate scores across forecasts to see
   which models win on which projects over time, with drift detection.

Storage: ``~/.local/share/gnomon/registry.db`` (or ``GNOMON_REGISTRY_PATH``).
Zero external dependencies — Python's ``sqlite3`` is stdlib.

Usage::

    from gnomon.tracking import TrackingStore

    store = TrackingStore()
    store.register("forecast_abc", project="api-capacity",
                   model="seasonal_naive", support="supported", ...)

    store.submit_actuals("api-capacity", actuals_csv="actuals.csv")
    store.score("forecast_abc")

    leaderboard = store.leaderboard("api-capacity")
    for entry in leaderboard:
        print(f"{entry.model}: MASE={entry.avg_mase:.2f} (n={entry.count})")
"""

from __future__ import annotations

import csv
from contextlib import contextmanager
import json
import logging
import math
import os
import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from .contracts import GnomonError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_registry_path = os.environ.get("GNOMON_REGISTRY_PATH")
DEFAULT_REGISTRY_PATH = (
    Path(_registry_path).expanduser()
    if _registry_path
    else Path.home() / ".local" / "share" / "gnomon" / "registry.db"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForecastRecord:
    forecast_id: str
    project: str
    series: str
    cutoff_time: str
    horizon: int
    frequency: str
    selected_model: str | None
    support: str
    threshold: float | None
    threshold_peak_probability: float | None
    naive_error: float | None
    artifact_path: str
    created_at: str
    scored: bool = False
    mase: float | None = None
    #: The selection metric, recorded in hindsight. Selection is decided on
    #: WAPE; a leaderboard reported only in MASE can rank models against the
    #: order selection chose them in, for reasons that are metric artefact
    #: rather than forecast quality.
    wape: float | None = None
    mape: float | None = None
    bias: float | None = None
    coverage: float | None = None
    threshold_accuracy: float | None = None
    scored_at: str | None = None
    drift_flag: str | None = None
    task: str = "forecast"
    fingerprint: str | None = None


@dataclass(frozen=True)
class ModelPerformance:
    model: str
    count: int
    avg_mase: float | None
    avg_wape: float | None
    avg_mape: float | None
    avg_bias: float | None
    avg_coverage: float | None
    avg_threshold_accuracy: float | None
    last_mase: float | None
    last_scored: str | None


@dataclass(frozen=True)
class ScoreResult:
    forecast_id: str
    mase: float | None
    wape: float | None
    mape: float | None
    bias: float | None
    coverage: float | None
    threshold_accuracy: float | None
    scored_at: str
    drift_flag: str | None


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    project: str
    forecast_id: str
    action: str
    expected_outcome: str
    actual_outcome: str | None
    correct: bool | None
    created_at: str
    resolved_at: str | None


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def mean_absolute_error(actual: list[float], predicted: list[float]) -> float:
    """Mean Absolute Error."""
    n = min(len(actual), len(predicted))
    if n == 0:
        return float("inf")
    return sum(abs(a - p) for a, p in zip(actual[:n], predicted[:n])) / n


#: Nominal coverage the adaptive level is steered toward, matching the
#: 80% interval the rest of the pipeline publishes.
ADAPTIVE_TARGET_COVERAGE = 0.8

#: Step size per outcome. Small enough that one unusual horizon cannot move
#: the published width far, large enough to correct a persistent bias within
#: a few dozen scored forecasts.
ADAPTIVE_LEARNING_RATE = 0.02

#: Bounds on the working miscoverage level, so a run of outcomes cannot drive
#: intervals to zero width or to uselessness.
MIN_ADAPTIVE_ALPHA = 0.01
MAX_ADAPTIVE_ALPHA = 0.5


def mase_score(
    actual: list[float], predicted: list[float], naive_error: float | None,
) -> float | None:
    """Mean Absolute Scaled Error — scale-free, comparable across datasets.

    Args:
        actual: observed values
        predicted: forecast values
        naive_error: MAE of a seasonal-naive baseline on the training set.

    Returns:
        MASE < 1 means the forecast beats the naive baseline.
    """
    n = min(len(actual), len(predicted))
    if n == 0:
        return float("inf")
    mae = sum(abs(a - p) for a, p in zip(actual[:n], predicted[:n])) / n
    if naive_error is None or naive_error <= 0:
        return None
    return mae / naive_error


def mape_score(actual: list[float], predicted: list[float]) -> float:
    """Mean absolute percentage error, **in percent** (4.279 means 4.279%).

    Every other error figure in this module is a fraction — `wape` at 0.0424
    is 4.24% — and the two sat unlabelled in the same object. The name now
    carries the unit wherever it is reported.
    """
    n = min(len(actual), len(predicted))
    if n == 0:
        return 0.0
    total = 0.0
    count = 0
    for a, p in zip(actual[:n], predicted[:n]):
        if abs(a) > 1e-9:
            total += abs((a - p) / a)
            count += 1
    return (total / count * 100) if count > 0 else 0.0


def bias_score(actual: list[float], predicted: list[float]) -> float:
    """Mean bias (actual - predicted). Positive = under-prediction."""
    n = min(len(actual), len(predicted))
    if n == 0:
        return 0.0
    return sum(a - p for a, p in zip(actual[:n], predicted[:n])) / n


def interval_coverage(actual: list[float], q10: list[float], q90: list[float]) -> float:
    """Fraction of actuals that fall within the q10-q90 interval."""
    n = min(len(actual), len(q10), len(q90))
    if n == 0:
        return 0.0
    inside = sum(1 for i in range(n) if q10[i] <= actual[i] <= q90[i])
    return inside / n


def threshold_accuracy(
    actual: list[float], threshold: float, predicted_above: list[bool] | None = None,
) -> float | None:
    """How often the threshold crossing prediction was correct.

    Args:
        actual: observed values
        threshold: the threshold value
        predicted_above: per-step booleans, True if the forecast predicted above

    Returns:
        Fraction of steps where predicted direction matches actual, or None.
    """
    if predicted_above is None:
        return None
    n = min(len(actual), len(predicted_above))
    if n == 0:
        return None
    correct = sum(
        1 for i in range(n)
        if (actual[i] > threshold) == predicted_above[i]
    )
    return correct / n


# ---------------------------------------------------------------------------
# Tracking store (SQLite)
# ---------------------------------------------------------------------------

#: Table definitions, one statement per table, so the schema has a single
#: source of truth: ``_init_db`` creates from these and ``_rebuild_table``
#: migrates to them. Keep each entry idempotent (``IF NOT EXISTS``).
_TABLE_DEFINITIONS: dict[str, str] = {
    "forecasts": """
        CREATE TABLE IF NOT EXISTS forecasts (
            forecast_id TEXT NOT NULL,
            project TEXT NOT NULL,
            series TEXT NOT NULL,
            cutoff_time TEXT,
            horizon INTEGER,
            frequency TEXT,
            selected_model TEXT,
            support TEXT,
            threshold REAL,
            threshold_peak_probability REAL,
            naive_error REAL,
            artifact_path TEXT,
            created_at TEXT NOT NULL,
            scored INTEGER DEFAULT 0,
            mase REAL,
            wape REAL,
            mape REAL,
            bias REAL,
            coverage REAL,
            threshold_accuracy REAL,
            scored_at TEXT,
            drift_flag TEXT,
            task TEXT DEFAULT 'forecast',
            fingerprint TEXT,
            -- A forecast_id is content-addressed: the same inputs in two
            -- projects produce the same id. Keying on it alone made a
            -- second registration *move* the first project's row, taking
            -- its realised scores with it.
            PRIMARY KEY (forecast_id, project, series)
        );

        CREATE INDEX IF NOT EXISTS idx_forecasts_project
            ON forecasts(project);

        CREATE INDEX IF NOT EXISTS idx_forecasts_model
            ON forecasts(project, selected_model);
    """,
    "model_performance": """
        CREATE TABLE IF NOT EXISTS model_performance (
            project TEXT NOT NULL,
            model TEXT NOT NULL,
            forecast_id TEXT NOT NULL,
            mase REAL,
            wape REAL,
            mape REAL,
            bias REAL,
            coverage REAL,
            threshold_accuracy REAL,
            scored_at TEXT NOT NULL,
            known_time TEXT,
            series TEXT NOT NULL DEFAULT '__default__',
            horizon INTEGER,
            frequency TEXT,
            task TEXT DEFAULT 'forecast',
            fingerprint TEXT,
            PRIMARY KEY (project, model, forecast_id, series)
        );

        CREATE INDEX IF NOT EXISTS idx_perf_model
            ON model_performance(project, model);
    """,
    # Adaptive-conformal state, kept as an append-only log rather than a
    # mutable current value. Each row carries the known_time of the outcome
    # that caused it, so a replay at `--as-of T` reads only what was known
    # by T and reproduces the interval that was actually published then. A
    # single mutable alpha would make every historical run irreproducible
    # the moment a new outcome arrived.
    "conformal_adaptation": """
        CREATE TABLE IF NOT EXISTS conformal_adaptation (
            project TEXT NOT NULL,
            scope TEXT NOT NULL,
            forecast_id TEXT NOT NULL,
            known_time TEXT NOT NULL,
            covered INTEGER NOT NULL,
            points INTEGER NOT NULL,
            PRIMARY KEY (project, scope, forecast_id)
        );

        CREATE INDEX IF NOT EXISTS idx_adaptation_scope
            ON conformal_adaptation(project, scope, known_time);
    """,
    "decisions": """
        CREATE TABLE IF NOT EXISTS decisions (
            decision_id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            forecast_id TEXT NOT NULL,
            action TEXT NOT NULL,
            expected_outcome TEXT NOT NULL,
            actual_outcome TEXT,
            correct INTEGER,
            created_at TEXT NOT NULL,
            resolved_at TEXT
            -- No FK to forecasts: forecast_id is no longer unique there
            -- (see the composite key above), and SQLite requires a parent
            -- key to be uniquely indexed.
        );
    """,
    "schema_metadata": """
        CREATE TABLE IF NOT EXISTS schema_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """,
    "decision_artifacts": """
        CREATE TABLE IF NOT EXISTS decision_artifacts (
            decision_id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            forecast_id TEXT NOT NULL,
            selected_action TEXT,
            created_at TEXT NOT NULL,
            resolved_at TEXT,
            payload TEXT NOT NULL
        );
    """,
    "routing_decisions": """
        CREATE TABLE IF NOT EXISTS routing_decisions (
            route_id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            task TEXT NOT NULL,
            series TEXT,
            fingerprint TEXT,
            recommendation TEXT,
            basis TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """,
    # --- The per-proposer calibration ledger (schema 5) ----------------
    # One row per context-event proposal per tracked forecast. The
    # proposal_key is content-addressed from what was *claimed* (type,
    # window, scope, source) — not from the run-local event id, which is
    # positional and unstable across runs — so the same claim re-proposed
    # against a later forecast joins to the same identity.
    "event_proposals": """
        CREATE TABLE IF NOT EXISTS event_proposals (
            proposal_key TEXT NOT NULL,
            project TEXT NOT NULL,
            forecast_id TEXT NOT NULL,
            series TEXT NOT NULL,
            event_id TEXT,
            event_type TEXT,
            proposer_id TEXT NOT NULL,
            proposer_kind TEXT,
            source_type TEXT,
            source_reference TEXT,
            status TEXT,
            confidence REAL,
            known_at TEXT,
            effective_start TEXT,
            effective_end TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (project, forecast_id, series, proposal_key)
        );

        CREATE INDEX IF NOT EXISTS idx_event_proposals_proposer
            ON event_proposals(project, proposer_id, event_type);
    """,
    # The gate's verdict per proposal, lifted out of the write-only
    # evidence.jsonl. `admitted` is the ablation gate's decision for the
    # event set; `published` is whether the enrichment actually reached
    # the published forecast (adjudication can admit at the gate and
    # still lose the ladder).
    "event_admissions": """
        CREATE TABLE IF NOT EXISTS event_admissions (
            proposal_key TEXT NOT NULL,
            project TEXT NOT NULL,
            forecast_id TEXT NOT NULL,
            series TEXT NOT NULL,
            admitted INTEGER NOT NULL,
            published INTEGER NOT NULL,
            lane TEXT NOT NULL,
            decided_by TEXT,
            exclusion_reason TEXT,
            mean_improvement REAL,
            shrinkage REAL,
            effect_shape TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (project, forecast_id, series, proposal_key)
        );
    """,
    # Realised outcomes, written when actuals arrive. `realised_lift_wape`
    # is error(history-only counterfactual) − error(published), in the
    # selection currency (WAPE): positive means the admitted events
    # helped. Attribution is set-level — the gate admits the event *set*,
    # so every admitted proposal on a forecast shares its lift, and the
    # column says so. `direction_hit` and `brier` stay NULL until a
    # proposal carries a direction or a resolvable occurrence claim.
    "event_outcomes": """
        CREATE TABLE IF NOT EXISTS event_outcomes (
            proposal_key TEXT NOT NULL,
            project TEXT NOT NULL,
            forecast_id TEXT NOT NULL,
            series TEXT NOT NULL,
            resolved_at TEXT NOT NULL,
            base_model TEXT,
            published_model TEXT,
            base_wape REAL,
            published_wape REAL,
            realised_lift_wape REAL,
            direction_hit INTEGER,
            brier REAL,
            attribution TEXT NOT NULL DEFAULT 'event_set',
            PRIMARY KEY (project, forecast_id, series, proposal_key)
        );

        CREATE INDEX IF NOT EXISTS idx_event_outcomes_project
            ON event_outcomes(project);
    """,
    # Scenario contracts are immutable snapshots of the primary and
    # context-conditioned paths.  They are registered even when context was
    # not admitted to the primary forecast, which is what lets outcome
    # tracking learn from honest what-if answers without rewriting history.
    "effect_scenarios": """
        CREATE TABLE IF NOT EXISTS effect_scenarios (
            effect_id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            forecast_id TEXT NOT NULL,
            series TEXT NOT NULL,
            event_type TEXT,
            event_ids TEXT NOT NULL,
            provenance_class TEXT NOT NULL,
            observed INTEGER NOT NULL,
            known_at TEXT NOT NULL,
            source_reference TEXT,
            effect_shape TEXT NOT NULL,
            effect_distribution TEXT NOT NULL,
            effect_provenance TEXT NOT NULL,
            primary_points TEXT NOT NULL,
            scenario_points TEXT NOT NULL,
            timestamps TEXT NOT NULL,
            occurrence_status TEXT NOT NULL DEFAULT 'unverified',
            occurrence_known_at TEXT,
            occurrence_note TEXT,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_effect_scenarios_lookup
            ON effect_scenarios(project, event_type, series, known_at);
    """,
    # One realised estimate per frozen scenario.  Missing counterfactuals and
    # confounding are rows, not swallowed exceptions, so the registry's
    # denominator remains auditable.
    "event_effect_registry": """
        CREATE TABLE IF NOT EXISTS event_effect_registry (
            effect_id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            forecast_id TEXT NOT NULL,
            series TEXT NOT NULL,
            event_type TEXT,
            resolved_at TEXT NOT NULL,
            outcome_known_at TEXT NOT NULL,
            estimate REAL,
            baseline_bias REAL,
            standard_error REAL,
            lower REAL,
            upper REAL,
            interval_probability REAL,
            onset_steps INTEGER,
            duration_steps INTEGER,
            expected_effect_shape TEXT NOT NULL,
            realised_effect_shape TEXT NOT NULL,
            sample_count INTEGER NOT NULL,
            confounding_status TEXT NOT NULL,
            counterfactual_quality TEXT NOT NULL,
            missing_counterfactual INTEGER NOT NULL,
            residuals TEXT NOT NULL,
            FOREIGN KEY(effect_id) REFERENCES effect_scenarios(effect_id)
        );

        CREATE INDEX IF NOT EXISTS idx_event_effect_registry_lookup
            ON event_effect_registry(project, event_type, series);
    """,
    # Typed temporal answers are additive claims about an immutable forecast.
    # Keeping the exact payload and its later realised path makes support and
    # abstention auditable without changing the forecast row or its scores.
    "temporal_answer_receipts": """
        CREATE TABLE IF NOT EXISTS temporal_answer_receipts (
            project TEXT NOT NULL,
            forecast_id TEXT NOT NULL,
            series TEXT NOT NULL,
            question_id TEXT NOT NULL,
            property TEXT NOT NULL,
            support TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL,
            resolved_at TEXT,
            outcome_payload TEXT,
            PRIMARY KEY (project, forecast_id, series, question_id)
        );

        CREATE INDEX IF NOT EXISTS idx_temporal_answer_receipts
            ON temporal_answer_receipts(project, property, resolved_at);
    """,
    "temporal_synthesis_receipts": """
        CREATE TABLE IF NOT EXISTS temporal_synthesis_receipts (
            project TEXT NOT NULL,
            forecast_id TEXT NOT NULL,
            series TEXT NOT NULL,
            question_id TEXT NOT NULL,
            synthesis_id TEXT NOT NULL,
            canonical_payload TEXT NOT NULL,
            synthesis_payload TEXT NOT NULL,
            evidence_refs TEXT NOT NULL,
            created_at TEXT NOT NULL,
            resolved_at TEXT,
            outcome_payload TEXT,
            score_payload TEXT,
            PRIMARY KEY (project, forecast_id, series, question_id, synthesis_id)
        );

        CREATE INDEX IF NOT EXISTS idx_temporal_synthesis_receipts
            ON temporal_synthesis_receipts(project, resolved_at);
    """,
}

#: Bumped to 4 when ``forecasts`` and ``model_performance`` gained their
#: composite keys; to 5 when the proposer-calibration ledger tables
#: (event_proposals / event_admissions / event_outcomes) were added; to 6 for
#: frozen effect scenarios and bitemporal realised-effect estimates.
SCHEMA_VERSION = "8"

#: Empirical-Bayes shrinkage strength for proposer skill: a proposer's
#: mean lift is pulled toward 0 (and a hit rate toward 0.5) with the
#: weight of this many pseudo-observations, so one lucky resolved call
#: cannot mint a skilled proposer. The existing leaderboard's plain AVG
#: has no such guard; this table does not repeat that defect.
PROPOSER_SKILL_SHRINKAGE = 10.0
EFFECT_POOL_SHRINKAGE = 3.0
MIN_EFFECT_POOL_EPISODES = 5


class TrackingStore:
    """SQLite-backed forecast registry and scoring store."""

    def __init__(self, path: str | Path | None = None):
        # Resolve mutable environment state at construction, not import.
        # Long-lived test and MCP hosts can select a registry per session;
        # caching an earlier temporary path makes a later session open a
        # directory that may already have been removed.
        configured = os.environ.get("GNOMON_REGISTRY_PATH")
        self.path = (
            Path(path) if path
            else Path(configured).expanduser() if configured
            else DEFAULT_REGISTRY_PATH
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            for definition in _TABLE_DEFINITIONS.values():
                conn.executescript(definition)
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(forecasts)")
            }
            if "naive_error" not in columns:
                conn.execute("ALTER TABLE forecasts ADD COLUMN naive_error REAL")
            for name, sql_type in (("task", "TEXT DEFAULT 'forecast'"), ("fingerprint", "TEXT"),
                                   ("wape", "REAL")):
                if name not in columns:
                    conn.execute(f"ALTER TABLE forecasts ADD COLUMN {name} {sql_type}")
            perf_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(model_performance)")
            }
            for name, sql_type in (("series", "TEXT"), ("horizon", "INTEGER"), ("frequency", "TEXT"),
                                   ("task", "TEXT DEFAULT 'forecast'"), ("fingerprint", "TEXT"),
                                   ("wape", "REAL"),
                                   # When this score became derivable, which is
                                   # not when it was written: a backfill scores
                                   # outcomes that were knowable long before the
                                   # submission. Kept distinct from scored_at so
                                   # the audit of *when the row was written*
                                   # survives alongside the replay key.
                                   ("known_time", "TEXT")):
                if name not in perf_columns:
                    conn.execute(f"ALTER TABLE model_performance ADD COLUMN {name} {sql_type}")
            # Rows written before the column existed were scored at the
            # instant they were written, which is exactly their knowledge
            # time under the old always-now behaviour.
            conn.execute(
                "UPDATE model_performance SET known_time = scored_at "
                "WHERE known_time IS NULL"
            )
            conn.execute(
                "INSERT OR REPLACE INTO schema_metadata(key, value) VALUES ('version', ?)",
                (SCHEMA_VERSION,),
            )
        self._migrate_composite_keys()

    def record_adapter_shadow_outcome(
        self, *, project: str, outcome_id: str, candidate: str,
        revision: str | None, baseline: str, candidate_error: float,
        baseline_error: float, known_at: str,
        regime: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Persist one paired challenger/baseline outcome in this registry."""
        from .adapter_promotion import AdapterOutcomeLedger
        AdapterOutcomeLedger(self.path).record(
            project=project, outcome_id=outcome_id, candidate=candidate,
            revision=revision, baseline=baseline,
            candidate_error=candidate_error, baseline_error=baseline_error,
            known_at=known_at, regime=regime,
        )
        return {
            "schema_version": "0.1", "status": "recorded",
            "project": project, "outcome_id": outcome_id,
            "candidate": candidate, "revision": revision or "unversioned",
            "baseline": baseline, "known_at": known_at, "regime": regime,
        }

    def assess_adapter_shadow(
        self, *, project: str, candidate: str, revision: str | None,
        baseline: str, as_of: str | None = None, min_outcomes: int = 30,
        min_improvement: float = .05, min_win_rate: float = .60,
    ) -> dict[str, Any]:
        """Return a review recommendation; never mutate model selection."""
        from .adapter_promotion import AdapterOutcomeLedger
        return AdapterOutcomeLedger(self.path).assess(
            project=project, candidate=candidate, revision=revision,
            baseline=baseline, as_of=as_of, min_outcomes=min_outcomes,
            min_improvement=min_improvement,
            min_win_rate=min_win_rate,
        ).to_dict()

    def _migrate_composite_keys(self) -> None:
        """Rebuild registries that were keyed on ``forecast_id`` alone.

        A store created before schema 4 has ``forecasts`` keyed on the
        content-addressed id, so registering the same forecast in a second
        project overwrote the first project's row and its realised scores.
        Rebuilding is the only way to change a primary key in SQLite. Rows
        are preserved verbatim; nothing that was already distinct is merged.
        """
        with self._connect(foreign_keys=False) as conn:
            for table, wanted in (
                ("forecasts", "PRIMARY KEY (forecast_id, project, series)"),
                ("model_performance", "PRIMARY KEY (project, model, forecast_id, series)"),
                ("decisions", "FOREIGN KEY"),
            ):
                row = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                    (table,),
                ).fetchone()
                if row is None:
                    continue
                sql = row["sql"] or ""
                # `decisions` is the inverse case: it needs rebuilding while
                # the marker is *present*, because the FK is what we drop.
                stale = (wanted in sql) if table == "decisions" else (wanted not in sql)
                if not stale:
                    continue
                self._rebuild_table(conn, table)

    def _rebuild_table(self, conn: sqlite3.Connection, table: str) -> None:
        """Copy ``table`` into a freshly-defined one of the same name."""
        columns = [row["name"] for row in conn.execute(f"PRAGMA table_info({table})")]
        conn.execute(f"ALTER TABLE {table} RENAME TO {table}__legacy")
        conn.executescript(_TABLE_DEFINITIONS[table])
        new_columns = [row["name"] for row in conn.execute(f"PRAGMA table_info({table})")]
        shared = [name for name in columns if name in new_columns]
        joined = ", ".join(shared)
        # A legacy row whose `series` is NULL predates multi-series support.
        select = ", ".join(
            "COALESCE(series, '__default__')" if name == "series" else name
            for name in shared
        )
        conn.execute(
            f"INSERT OR REPLACE INTO {table} ({joined}) "
            f"SELECT {select} FROM {table}__legacy"
        )
        conn.execute(f"DROP TABLE {table}__legacy")
        logger.info("Migrated %s to the schema-4 composite key", table)

    @contextmanager
    def _connect(self, foreign_keys: bool = True) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(f"PRAGMA foreign_keys = {'ON' if foreign_keys else 'OFF'}")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---- Registration ----

    def register(
        self,
        forecast_id: str,
        project: str,
        series: str = "__default__",
        cutoff_time: str | None = None,
        horizon: int | None = None,
        frequency: str | None = None,
        selected_model: str | None = None,
        support: str = "unsupported",
        threshold: float | None = None,
        threshold_peak_probability: float | None = None,
        naive_error: float | None = None,
        artifact_path: str = "",
        created_at: str | None = None,
        task: str = "forecast",
        fingerprint: str | None = None,
    ) -> None:
        """Register or update a forecast in the registry."""
        created_at = created_at or datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO forecasts
                    (forecast_id, project, series, cutoff_time, horizon, frequency,
                     selected_model, support, threshold, threshold_peak_probability, naive_error,
                     artifact_path, created_at, scored, task, fingerprint)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                ON CONFLICT(forecast_id, project, series) DO UPDATE SET
                    cutoff_time = excluded.cutoff_time,
                    horizon = excluded.horizon,
                    frequency = excluded.frequency,
                    selected_model = excluded.selected_model,
                    support = excluded.support,
                    threshold = excluded.threshold,
                    threshold_peak_probability = excluded.threshold_peak_probability,
                    naive_error = excluded.naive_error,
                    artifact_path = excluded.artifact_path,
                    task = excluded.task,
                    fingerprint = excluded.fingerprint
            """, (
                forecast_id, project, series, cutoff_time, horizon, frequency,
                selected_model, support, threshold, threshold_peak_probability, naive_error,
                artifact_path, created_at, task, fingerprint,
            ))
        logger.info("Registered forecast %s in project %s", forecast_id, project)

    def record_temporal_answers(
        self, project: str, forecast_id: str, series: str,
        answers: list[dict[str, Any]], *, created_at: str | None = None,
    ) -> None:
        """Persist immutable typed-answer receipts for realised follow-up."""
        created_at = created_at or datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            for item in answers:
                question = item.get("question") or {}
                answer = item.get("answer") or {}
                target = question.get("target")
                if isinstance(target, str) and series != "__default__" \
                        and target not in {series, "*"}:
                    continue
                conn.execute("""
                    INSERT OR IGNORE INTO temporal_answer_receipts
                        (project, forecast_id, series, question_id, property,
                         support, payload, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (project, forecast_id, series,
                      str(question.get("id") or "q1"),
                      str(question.get("property") or "unknown"),
                      str(answer.get("support") or "abstained"),
                      json.dumps(item, sort_keys=True), created_at))

    def temporal_answer_receipts(
        self, project: str, *, resolved: bool | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM temporal_answer_receipts WHERE project = ?"
        params: list[Any] = [project]
        if resolved is not None:
            query += " AND resolved_at IS " + ("NOT NULL" if resolved else "NULL")
        query += " ORDER BY created_at, forecast_id, series, question_id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"]),
                 "outcome_payload": (json.loads(row["outcome_payload"])
                                     if row["outcome_payload"] else None)}
                for row in rows]

    def record_temporal_synthesis(
        self, *, project: str, forecast_id: str, series: str,
        question_id: str, synthesis_id: str,
        canonical: dict[str, Any], synthesis: dict[str, Any],
        evidence_refs: list[str], created_at: str | None = None,
    ) -> None:
        """Record a labelled model contribution beside, never over, canonical."""
        if synthesis.get("label") not in {"labelled_synthesis", "conditional_answer",
                                           "hypothesis_ranking"}:
            raise ValueError("synthesis must use a recognised non-canonical label")
        if synthesis.get("primary_forecast_unchanged") is not True:
            raise ValueError("synthesis must attest primary_forecast_unchanged")
        if not evidence_refs:
            raise ValueError("synthesis requires at least one evidence reference")
        created_at = created_at or datetime.now(timezone.utc).isoformat()
        canonical_payload = json.dumps(canonical, sort_keys=True)
        synthesis_payload = json.dumps(synthesis, sort_keys=True)
        evidence_payload = json.dumps(sorted(set(evidence_refs)))
        with self._connect() as conn:
            existing = conn.execute("""
                SELECT canonical_payload, synthesis_payload, evidence_refs
                FROM temporal_synthesis_receipts
                WHERE project = ? AND forecast_id = ? AND series = ?
                  AND question_id = ? AND synthesis_id = ?
            """, (project, forecast_id, series, question_id, synthesis_id)).fetchone()
            if existing is not None:
                if (existing["canonical_payload"], existing["synthesis_payload"],
                        existing["evidence_refs"]) != (
                            canonical_payload, synthesis_payload, evidence_payload):
                    raise ValueError(
                        "conflicting synthesis receipt: an immutable synthesis_id "
                        "was re-used with different content"
                    )
                return
            conn.execute("""
                INSERT INTO temporal_synthesis_receipts
                    (project, forecast_id, series, question_id, synthesis_id,
                     canonical_payload, synthesis_payload, evidence_refs, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (project, forecast_id, series, question_id, synthesis_id,
                  canonical_payload, synthesis_payload, evidence_payload, created_at))

    def resolve_temporal_synthesis(
        self, *, project: str, forecast_id: str, series: str,
        question_id: str, synthesis_id: str, outcome: dict[str, Any],
        resolved_at: str | None = None,
    ) -> dict[str, Any]:
        """Score canonical and synthesis independently against one outcome."""
        resolved_at = resolved_at or datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            row = conn.execute("""
                SELECT canonical_payload, synthesis_payload
                FROM temporal_synthesis_receipts
                WHERE project = ? AND forecast_id = ? AND series = ?
                  AND question_id = ? AND synthesis_id = ? AND resolved_at IS NULL
            """, (project, forecast_id, series, question_id, synthesis_id)).fetchone()
        if row is None:
            raise ValueError("unknown or already-resolved synthesis receipt")
        canonical = json.loads(row["canonical_payload"])
        synthesis = json.loads(row["synthesis_payload"])
        realised = next((outcome.get(key) for key in ("direction", "state", "value")
                         if outcome.get(key) is not None), None)
        canonical_value = canonical.get("value")
        synthesis_value = synthesis.get("value", synthesis.get("claim"))
        # Only typed categorical equality is scored here. Numeric claims need
        # a property-specific scoring rule and remain unresolved rather than
        # acquiring a convenient tolerance at this generic boundary.
        actual_points = outcome.get("points")
        canonical_points = canonical.get("forecast")
        synthesis_points = synthesis.get("forecast")
        if (isinstance(actual_points, list) and canonical_points
                and synthesis_points):
            from .evaluation import error_score
            def values(rows):
                return [float(row.get("q50", row.get("point"))) for row in rows]
            n = min(len(actual_points), len(canonical_points), len(synthesis_points))
            canonical_error = error_score(
                [float(item) for item in actual_points[:n]], values(canonical_points)[:n])
            synthesis_error = error_score(
                [float(item) for item in actual_points[:n]], values(synthesis_points)[:n])
            score = {
                "rule": "numeric_path_wape_v1", "points": n,
                "canonical_error": canonical_error,
                "synthesis_error": synthesis_error,
                "synthesis_delta": (canonical_error - synthesis_error
                                    if canonical_error is not None
                                    and synthesis_error is not None else None),
                "synthesis_won": (synthesis_error < canonical_error
                                  if canonical_error is not None
                                  and synthesis_error is not None else None),
            }
        else:
            comparable = isinstance(realised, (str, bool))
            canonical_correct = (canonical_value == realised if comparable else None)
            synthesis_correct = (synthesis_value == realised if comparable else None)
            score = {
                "rule": "typed_categorical_exact_v1",
                "canonical_correct": canonical_correct,
                "synthesis_correct": synthesis_correct,
                "synthesis_delta": (
                    int(synthesis_correct) - int(canonical_correct)
                    if canonical_correct is not None and synthesis_correct is not None
                    else None),
            }
        with self._connect() as conn:
            changed = conn.execute("""
                UPDATE temporal_synthesis_receipts
                SET resolved_at = ?, outcome_payload = ?, score_payload = ?
                WHERE project = ? AND forecast_id = ? AND series = ?
                  AND question_id = ? AND synthesis_id = ? AND resolved_at IS NULL
            """, (resolved_at, json.dumps(outcome, sort_keys=True),
                  json.dumps(score, sort_keys=True), project, forecast_id, series,
                  question_id, synthesis_id)).rowcount
        if changed != 1:  # defensive race check
            raise ValueError("synthesis receipt was concurrently resolved")
        return score

    def temporal_synthesis_receipts(
        self, project: str, *, resolved: bool | None = None,
        series: str | None = None, resolved_before: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM temporal_synthesis_receipts WHERE project = ?"
        params: list[Any] = [project]
        if series is not None:
            query += " AND series = ?"
            params.append(series)
        if resolved is not None:
            query += " AND resolved_at IS " + ("NOT NULL" if resolved else "NULL")
        query += " ORDER BY created_at, synthesis_id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        decoded = []
        for row in rows:
            item = dict(row)
            for key in ("canonical_payload", "synthesis_payload", "evidence_refs",
                        "outcome_payload", "score_payload"):
                item[key] = json.loads(item[key]) if item[key] else None
            decoded.append(item)
        if resolved_before is not None:
            try:
                cutoff = datetime.fromisoformat(
                    str(resolved_before).replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("resolved_before must be an ISO timestamp") from exc
            if cutoff.tzinfo is None:
                raise ValueError("resolved_before must be timezone-aware")
            cutoff = cutoff.astimezone(timezone.utc)
            filtered = []
            for item in decoded:
                raw = item.get("resolved_at")
                if not raw:
                    continue
                try:
                    resolved_at = datetime.fromisoformat(
                        str(raw).replace("Z", "+00:00"))
                except ValueError:
                    continue
                if (resolved_at.tzinfo is not None
                        and resolved_at.astimezone(timezone.utc) <= cutoff):
                    filtered.append(item)
            decoded = filtered
        return decoded

    def candidate_outcome_summary(
        self, project: str, *, minimum_resolved: int = 8,
        series: str | None = None, resolved_before: str | None = None,
    ) -> list[dict[str, Any]]:
        """Aggregate resolved candidate uplift without granting authority.

        Candidate types are keyed by their sealed publication role and
        provenance origin.  Graduation is deliberately conservative: at least
        ``minimum_resolved`` numeric comparisons, positive mean uplift, and a
        95% Wilson lower bound above chance.  The result is an evidence report;
        callers must not translate it into automation or support upgrades.
        """
        if minimum_resolved < 1:
            raise ValueError("minimum_resolved must be positive")
        groups: dict[
            tuple[str, str, str], dict[str, list[tuple[bool, float]]]
        ] = {}
        for receipt in self.temporal_synthesis_receipts(
                project, resolved=True, series=series,
                resolved_before=resolved_before):
            score = receipt.get("score_payload") or {}
            synthesis = receipt.get("synthesis_payload") or {}
            won = score.get("synthesis_won")
            delta = score.get("synthesis_delta")
            if not isinstance(won, bool) or not isinstance(delta, (int, float)) \
                    or not math.isfinite(float(delta)):
                continue
            role = str(synthesis.get("scenario_role") or "unknown")
            origin = str(synthesis.get("candidate_origin") or role)
            proposer = str(synthesis.get("candidate_proposer") or "unknown")
            groups.setdefault((role, origin, proposer), {}).setdefault(
                str(receipt.get("forecast_id") or "unknown"), []).append(
                    (won, float(delta)))
        summaries = []
        z = 1.959963984540054
        for (role, origin, proposer), per_forecast in sorted(groups.items()):
            raw_count = sum(len(items) for items in per_forecast.values())
            # Several same-class candidates from one forecast are correlated
            # alternatives, not independent evidence. Collapse each origin
            # conservatively: the class wins only if every variant won and its
            # uplift is the least favorable variant's uplift.
            outcomes = [
                (all(won for won, _ in items),
                 min(delta for _, delta in items))
                for items in per_forecast.values()
            ]
            n = len(outcomes)
            wins = sum(int(won) for won, _ in outcomes)
            rate = wins / n
            denominator = 1 + z * z / n
            centre = rate + z * z / (2 * n)
            margin = z * math.sqrt(
                rate * (1 - rate) / n + z * z / (4 * n * n))
            lower = max(0.0, (centre - margin) / denominator)
            mean_delta = statistics.mean(delta for _, delta in outcomes)
            graduated = bool(
                n >= minimum_resolved and mean_delta > 0 and lower > .5)
            summaries.append({
                "scenario_role": role, "candidate_origin": origin,
                "candidate_proposer": proposer,
                "resolved": n, "wins": wins, "win_rate": rate,
                "raw_candidates_resolved": raw_count,
                "independent_forecast_origins": n,
                "win_rate_wilson_95_lower": lower,
                "mean_uplift_vs_primary": mean_delta,
                "minimum_resolved": minimum_resolved,
                "graduated_for_human_prior": graduated,
                "support_upgrade_allowed": False,
                "automation_upgrade_allowed": False,
                "scope": {
                    "project": project,
                    "series": series,
                    "resolved_before": resolved_before,
                },
                "rule": "resolved_numeric_paths_wilson95_and_positive_mean",
            })
        return summaries

    def decision_synthesis_skill(
        self, project: str, *, proposer_id: str | None = None,
        minimum_resolved: int = 20,
    ) -> list[dict[str, Any]]:
        """Outcome-derived skill for model-authored categorical decisions.

        Uses the existing immutable synthesis receipts. Only exact categorical
        outcomes where canonical and synthesis were both scoreable enter the
        ledger. Graduation is paired: the synthesis must win significantly
        more discordant cases than it loses, have positive shrunk net wins,
        and meet a predeclared resolved-count floor. This may justify a
        human-facing prior weight; it can never upgrade support or automation.
        """
        if minimum_resolved < 1:
            raise ValueError("minimum_resolved must be positive")
        groups: dict[str, list[tuple[bool, bool]]] = {}
        for receipt in self.temporal_synthesis_receipts(project, resolved=True):
            score = receipt.get("score_payload") or {}
            synthesis = receipt.get("synthesis_payload") or {}
            canonical_correct = score.get("canonical_correct")
            synthesis_correct = score.get("synthesis_correct")
            if not isinstance(canonical_correct, bool) \
                    or not isinstance(synthesis_correct, bool):
                continue
            origin = str(synthesis.get("proposer_id")
                         or synthesis.get("candidate_origin") or "unknown")
            if proposer_id is not None and origin != proposer_id:
                continue
            groups.setdefault(origin, []).append(
                (canonical_correct, synthesis_correct))

        summaries = []
        k = PROPOSER_SKILL_SHRINKAGE
        for origin, outcomes in sorted(groups.items()):
            resolved = len(outcomes)
            wins = sum(synthesis and not canonical
                       for canonical, synthesis in outcomes)
            losses = sum(canonical and not synthesis
                         for canonical, synthesis in outcomes)
            ties = resolved - wins - losses
            discordant = wins + losses
            # Exact two-sided sign test over paired discordant outcomes.
            if discordant:
                tail = sum(math.comb(discordant, index)
                           for index in range(min(wins, losses) + 1))
                exact_p = min(1.0, 2.0 * tail / (2 ** discordant))
            else:
                exact_p = 1.0
            shrunk_net = (wins - losses) / (resolved + k)
            synthesis_accuracy = sum(
                synthesis for _, synthesis in outcomes) / resolved
            canonical_accuracy = sum(
                canonical for canonical, _ in outcomes) / resolved
            graduated = bool(
                resolved >= minimum_resolved
                and discordant >= 8
                and wins > losses
                and exact_p < .05
                and shrunk_net > 0)
            summaries.append({
                "proposer_id": origin,
                "resolved": resolved,
                "wins_vs_canonical": wins,
                "losses_vs_canonical": losses,
                "ties": ties,
                "discordant": discordant,
                "exact_sign_p": exact_p,
                "synthesis_accuracy": synthesis_accuracy,
                "canonical_accuracy": canonical_accuracy,
                "shrunk_net_wins_per_resolved": shrunk_net,
                "shrinkage_k": k,
                "minimum_resolved": minimum_resolved,
                "graduated_for_human_prior": graduated,
                "support_upgrade_allowed": False,
                "automation_upgrade_allowed": False,
                "rule": "paired_categorical_sign_test_and_shrunk_net_v1",
            })
        return summaries

    def _resolve_temporal_answers(
        self, record: ForecastRecord, actuals: list[float], resolved_at: str,
    ) -> None:
        """Join due typed claims to the realised path without reinterpretation."""
        if not actuals:
            return
        outcome = {
            "points": len(actuals), "first": actuals[0], "last": actuals[-1],
            "median": float(statistics.median(actuals)), "minimum": min(actuals),
            "maximum": max(actuals),
            "slope_per_step": (actuals[-1] - actuals[0]) / max(1, len(actuals) - 1),
        }
        with self._connect() as conn:
            conn.execute("""
                UPDATE temporal_answer_receipts
                SET resolved_at = ?, outcome_payload = ?
                WHERE project = ? AND forecast_id = ? AND series = ?
                  AND resolved_at IS NULL
            """, (resolved_at, json.dumps(outcome, sort_keys=True),
                  record.project, record.forecast_id, record.series))

    # ---- Listing ----

    def list_forecasts(
        self, project: str | None = None, limit: int = 50,
    ) -> list[ForecastRecord]:
        """List forecasts, optionally filtered by project."""
        with self._connect() as conn:
            if project:
                rows = conn.execute(
                    "SELECT * FROM forecasts WHERE project = ? ORDER BY created_at DESC LIMIT ?",
                    (project, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM forecasts ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_forecast(
        self, forecast_id: str, project: str | None = None,
        series: str | None = None,
    ) -> ForecastRecord | None:
        """Get a single registration by id, optionally scoped.

        The same content-addressed ``forecast_id`` can be registered in
        several projects, so an unscoped lookup is ambiguous by
        construction; it returns the most recent registration. Callers that
        know the project should say so.
        """
        query = "SELECT * FROM forecasts WHERE forecast_id = ?"
        params: list[Any] = [forecast_id]
        if project is not None:
            query += " AND project = ?"
            params.append(project)
        if series is not None:
            query += " AND series = ?"
            params.append(series)
        query += " ORDER BY created_at DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return self._row_to_record(row) if row else None

    # ---- Adaptive conformal state ----

    def record_coverage_outcome(
        self, project: str, scope: str, forecast_id: str,
        known_time: str, covered: int, points: int,
    ) -> None:
        """Append one realised-coverage observation to the adaptation log.

        ``known_time`` is when the *outcome* became knowable, not when it was
        recorded. That distinction is what makes replay honest: a run at
        ``--as-of T`` must see exactly the outcomes a forecaster at T could
        have seen, whatever order they were entered in afterwards.
        """
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO conformal_adaptation
                    (project, scope, forecast_id, known_time, covered, points)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (project, scope, forecast_id, known_time, int(covered), int(points)))

    def coverage_outcomes(
        self, project: str, scope: str, as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        """Outcomes known by ``as_of``, oldest first, ties broken by id.

        The ordering is total and content-derived, so the replay below is a
        pure function of the rows it reads.
        """
        query = ("SELECT forecast_id, known_time, covered, points "
                 "FROM conformal_adaptation WHERE project = ? AND scope = ?")
        params: list[Any] = [project, scope]
        if as_of is not None:
            query += " AND known_time <= ?"
            params.append(self._normalise_timestamp(as_of))
        query += " ORDER BY known_time ASC, forecast_id ASC"
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(query, params)]

    def adapted_alpha(
        self, project: str, scope: str, as_of: str | None = None,
        target: float = ADAPTIVE_TARGET_COVERAGE,
        rate: float = ADAPTIVE_LEARNING_RATE,
    ) -> dict[str, Any]:
        """Replay the adaptation log into an effective miscoverage level.

        Adaptive conformal inference: after each outcome, move the working
        level by ``rate`` toward whichever direction the realised coverage
        was wrong in. Nominal 80% intervals that keep covering 95% of points
        are too wide, and this is the standard correction.

        The hard part is not the update rule but determinism. This is a fold
        over an ordered, immutable log filtered by ``known_time``, so the
        same inputs at the same ``as_of`` produce the same level forever —
        adding an outcome tomorrow cannot change what a replay of yesterday
        reports.
        """
        outcomes = self.coverage_outcomes(project, scope, as_of)
        alpha = 1.0 - target
        for outcome in outcomes:
            points = int(outcome["points"]) or 1
            realised = int(outcome["covered"]) / points
            # err = 1 when the interval missed more than it should have.
            alpha = alpha + rate * ((1.0 - target) - (1.0 - realised))
            alpha = min(MAX_ADAPTIVE_ALPHA, max(MIN_ADAPTIVE_ALPHA, alpha))
        return {
            "alpha": alpha,
            "effective_coverage": 1.0 - alpha,
            "observations": len(outcomes),
            "as_of": as_of,
            "basis": "adaptive conformal replay over outcomes known by as_of",
        }

    # ---- Scoring ----

    @staticmethod
    def _require_possible_knowledge_time(valid_time: str, known_at: str) -> None:
        """Refuse a ``known_at`` earlier than the outcome it describes.

        A measured outcome cannot be known before it occurs; accepting one
        would let a replay see the future wearing an honest label. Naive
        timestamps are compared as UTC; unparseable ones are left to fail
        where they are matched, not here.
        """
        def _parse(text: str) -> datetime | None:
            try:
                parsed = datetime.fromisoformat(
                    text[:-1] + "+00:00" if text.endswith("Z") else text
                )
            except ValueError:
                return None
            if parsed.utcoffset() is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)

        valid, known = _parse(valid_time), _parse(known_at)
        if valid is not None and known is not None and known < valid:
            raise GnomonError(
                "OUTCOME_KNOWN_BEFORE_VALID",
                f"known_at {known_at} precedes the outcome's own timestamp "
                f"{valid_time}: a measurement cannot be known before it "
                "occurs.",
                {"timestamp": valid_time, "known_at": known_at},
            )

    @staticmethod
    def _derived_known_time(
        known_times: list[str | None] | None, outcome_count: int,
        submitted_at: str,
    ) -> str:
        """When a record derived from several outcomes became knowable.

        The latest contributing knowledge time: a derived aggregate exists
        only once its last input does. Any outcome without an explicit
        ``known_at`` became knowable at submission, which is later than
        every explicit backfill time — so one missing entry makes the
        submission time the answer.
        """
        supplied = [k for k in (known_times or []) if k]
        if len(supplied) < outcome_count:
            return submitted_at

        def _sort_key(text: str) -> datetime:
            parsed = datetime.fromisoformat(
                text[:-1] + "+00:00" if text.endswith("Z") else text
            )
            if parsed.utcoffset() is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)

        try:
            return max(supplied, key=_sort_key)
        except ValueError:
            # An unparseable knowledge time cannot order a replay.
            return submitted_at

    def score_forecast(
        self,
        forecast_id: str,
        actuals: list[float],
        points: list[float],
        q10: list[float] | None = None,
        q90: list[float] | None = None,
        threshold: float | None = None,
        predicted_above: list[bool] | None = None,
        naive_error: float | None = None,
        project: str | None = None,
        series: str | None = None,
        known_times: list[str | None] | None = None,
    ) -> ScoreResult:
        """Score a single forecast against actuals.

        Computes MASE, MAPE, bias, coverage, and threshold accuracy.
        Stores the result and updates model_performance. ``project`` and
        ``series`` scope the registration being scored; without them the
        most recent registration of the id is used.

        ``known_times``, aligned with ``actuals``, carries each outcome's
        explicit knowledge time (ISO) where the caller has one — a
        backfill of history whose outcomes were knowable long before this
        submission. Any outcome without one became knowable *now*, so a
        derived record's knowledge time is the submission time unless
        every contributing outcome says otherwise.
        """
        record = self.get_forecast(forecast_id, project, series)
        if record is None:
            raise ValueError(f"Forecast {forecast_id} not found in registry")

        scale = naive_error if naive_error is not None else record.naive_error
        mase = mase_score(actuals, points, scale)
        # The same function selection is decided on, so hindsight and choice
        # are reported in one unit as well as in MASE's naive-relative one.
        from .evaluation import error_score
        wape = error_score(actuals, points)
        mape = mape_score(actuals, points)
        bias = bias_score(actuals, points)
        cov = interval_coverage(actuals, q10 or [], q90 or []) if q10 and q90 else None
        thresh_acc = threshold_accuracy(actuals, threshold, predicted_above) if threshold else None
        scored_at = datetime.now(timezone.utc).isoformat()
        # When this score became *derivable* — the latest knowledge time
        # among the outcomes it consumed, which on a backfill is long
        # before `scored_at`. The performance record is a derived record
        # exactly like the coverage outcome below, so it inherits the same
        # instant; a leaderboard replayed `as_of` T must not read a score
        # that needed an outcome nobody had at T.
        derived_known_time = self._derived_known_time(
            known_times, len(actuals), scored_at)

        # Drift detection: compare to model's historical average
        drift = self._check_drift(record, mase)

        with self._connect() as conn:
            conn.execute("""
                UPDATE forecasts SET
                    scored = 1, mase = ?, wape = ?, mape = ?, bias = ?,
                    coverage = ?, threshold_accuracy = ?, scored_at = ?, drift_flag = ?
                WHERE forecast_id = ? AND project = ? AND series = ?
            """, (mase, wape, mape, bias, cov, thresh_acc, scored_at, drift,
                  forecast_id, record.project, record.series))

            if record.selected_model:
                conn.execute("""
                    INSERT OR REPLACE INTO model_performance
                        (project, model, forecast_id, mase, wape, mape, bias,
                         coverage, threshold_accuracy, scored_at, known_time,
                         series, horizon, frequency, task, fingerprint)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.project, record.selected_model, forecast_id,
                    mase, wape, mape, bias, cov, thresh_acc, scored_at,
                    derived_known_time,
                    record.series, record.horizon, record.frequency,
                    record.task, record.fingerprint,
                ))

        if cov is not None and q10 and q90:
            # Feed the adaptation log, stamped with when this aggregate
            # became knowable. Three instants must stay distinct here: the
            # outcome's valid time (when it occurred), its known time (when
            # it reached Gnomon), and the forecast cutoff (what the forecast
            # could see). The previous stamp used the cutoff — the one
            # instant at which the outcome *provably did not exist yet* —
            # so an --as-of replay at the forecast origin could see the
            # realised coverage of the very horizon being forecast.
            points = min(len(actuals), len(q10), len(q90))
            self.record_coverage_outcome(
                record.project, record.series or "__default__", forecast_id,
                known_time=derived_known_time,
                covered=round(cov * points), points=points,
            )

        logger.info(
            "Scored forecast %s: MASE=%s MAPE=%.1f%% bias=%.2f coverage=%s drift=%s",
            forecast_id, f"{mase:.3f}" if mase is not None else "N/A", mape, bias,
            f"{cov:.1%}" if cov is not None else "N/A",
            drift or "none",
        )
        # Publication recommendations share the existing synthesis receipt
        # machinery. Score their numeric path independently of the immutable
        # primary when actuals arrive; automation safety is not inferred from
        # recommendation accuracy.
        for receipt in self.temporal_synthesis_receipts(
                record.project, resolved=False):
            if (receipt["forecast_id"] == forecast_id
                    and receipt["series"] == record.series
                    and receipt["question_id"] == "publication"):
                self.resolve_temporal_synthesis(
                    project=record.project, forecast_id=forecast_id,
                    series=record.series,
                    question_id="publication",
                    synthesis_id=receipt["synthesis_id"],
                    outcome={"points": list(actuals)}, resolved_at=scored_at)
        return ScoreResult(forecast_id, mase, wape, mape, bias, cov, thresh_acc,
                           scored_at, drift)

    def submit_actuals(
        self,
        project: str,
        actuals: list[tuple[str, float] | tuple[str, str, float]
                      | tuple[str | None, str, float, str | None]],
        time_column: str = "timestamp",
        target_column: str = "value",
    ) -> list[ScoreResult]:
        """Submit actuals for a project and score all matching forecasts.

        Args:
            project: project name
            actuals: list of (timestamp_iso, value),
                (series, timestamp_iso, value), or
                (series_or_None, timestamp_iso, value, known_at_or_None)
                tuples. ``known_at`` asserts when the outcome became
                knowable — for backfilling history whose outcomes were
                available long before this submission. Rows without it
                became knowable now.
            time_column: name of the timestamp column (for CSV loading)
            target_column: name of the target column (for CSV loading)

        Returns:
            List of ScoreResult for each scored forecast.
        """
        # Build timestamp → value map
        actual_map: dict[tuple[str | None, str], float] = {}
        known_map: dict[tuple[str | None, str], str] = {}
        for item in actuals:
            if len(item) == 2:
                series, (ts, val), known_at = None, item, None
            elif len(item) == 3:
                series, ts, val = item
                known_at = None
            else:
                series, ts, val, known_at = item
            key = (series, self._normalise_timestamp(ts))
            actual_map[key] = val
            if known_at is not None:
                known_at = self._normalise_timestamp(known_at)
                self._require_possible_knowledge_time(ts, known_at)
                known_map[key] = known_at

        results: list[ScoreResult] = []
        forecasts = self.list_forecasts(project, limit=1000)
        if len({record.series for record in forecasts}) > 1 and any(
            series is None for series, _ in actual_map
        ):
            raise ValueError(
                "Actuals for a multi-series project must include a 'series' column"
            )

        for record in forecasts:
            if record.scored:
                continue
            # Load forecast from artifact
            artifact_path = Path(record.artifact_path) / "forecast.csv"
            if not artifact_path.exists():
                logger.warning("Artifact not found for %s: %s", record.forecast_id, artifact_path)
                continue

            # Parse forecast.csv
            forecast_data = [
                row for row in self._load_forecast_csv(artifact_path)
                if row.get("series", "__default__") == record.series
            ]
            if not forecast_data:
                continue

            # Match actuals to forecast timestamps
            matched_actuals: list[float] = []
            matched_points: list[float] = []
            matched_q10: list[float] = []
            matched_q90: list[float] = []
            matched_known: list[str | None] = []
            for entry in forecast_data:
                ts = self._normalise_timestamp(entry["timestamp"])
                exact_key = (record.series, ts)
                default_key = (None, ts)
                if exact_key in actual_map or default_key in actual_map:
                    key = exact_key if exact_key in actual_map else default_key
                    matched_actuals.append(actual_map[key])
                    matched_known.append(known_map.get(key))
                    matched_points.append(entry["point"])
                    if "q10" in entry:
                        matched_q10.append(entry["q10"])
                    if "q90" in entry:
                        matched_q90.append(entry["q90"])

            if len(matched_actuals) < 1:
                logger.debug("No matching actuals for forecast %s", record.forecast_id)
                continue
            if len(matched_actuals) != len(forecast_data):
                logger.info(
                    "Forecast %s is not ready to score: matched %d of %d actuals",
                    record.forecast_id, len(matched_actuals), len(forecast_data),
                )
                continue

            # Compute predicted_above if threshold exists
            pred_above = None
            if record.threshold is not None:
                pred_above = [p > record.threshold for p in matched_points]

            result = self.score_forecast(
                record.forecast_id,
                matched_actuals,
                matched_points,
                q10=matched_q10 if matched_q10 else None,
                q90=matched_q90 if matched_q90 else None,
                threshold=record.threshold,
                predicted_above=pred_above,
                project=record.project,
                series=record.series,
                known_times=matched_known,
            )
            results.append(result)
            try:
                self._resolve_temporal_answers(
                    record, matched_actuals, result.scored_at)
            except Exception:
                logger.warning("Temporal-answer resolution failed for %s",
                               record.forecast_id, exc_info=True)
            try:
                self._resolve_event_outcomes(record, matched_actuals, result)
            except Exception:
                # An unresolvable ledger row must not block forecast scoring;
                # the outcome stays open and a later submit can retry it.
                logger.warning("Event-outcome resolution failed for %s",
                               record.forecast_id, exc_info=True)
            try:
                explicit_known = [value for value in matched_known if value]
                self._resolve_effect_scenarios(
                    record, matched_actuals,
                    outcome_known_at=(max(explicit_known) if explicit_known
                                      else result.scored_at),
                )
            except Exception:
                logger.warning("Effect-registry resolution failed for %s",
                               record.forecast_id, exc_info=True)

        return results

    # -- The per-proposer calibration ledger --------------------------------

    def record_event_proposals(
        self, project: str, forecast_id: str, series: str,
        proposals: list[dict[str, Any]],
    ) -> int:
        """Record context-event proposals and their gate verdicts.

        Each entry carries the proposal's claim fields plus its
        ``admission`` sub-dict (see ``register_artifact``). Idempotent per
        (project, forecast_id, series, proposal_key).
        """
        created_at = datetime.now(timezone.utc).isoformat()
        written = 0
        with self._connect() as conn:
            for item in proposals:
                admission = item.get("admission") or {}
                conn.execute(
                    """
                    INSERT OR REPLACE INTO event_proposals
                        (proposal_key, project, forecast_id, series, event_id,
                         event_type, proposer_id, proposer_kind, source_type,
                         source_reference, status, confidence, known_at,
                         effective_start, effective_end, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (item["proposal_key"], project, forecast_id, series,
                     item.get("event_id"), item.get("event_type"),
                     item.get("proposer_id") or "user",
                     item.get("proposer_kind"),
                     item.get("source_type"), item.get("source_reference"),
                     item.get("status"), item.get("confidence"),
                     item.get("known_at"), item.get("effective_start"),
                     item.get("effective_end"), created_at),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO event_admissions
                        (proposal_key, project, forecast_id, series, admitted,
                         published, lane, decided_by, exclusion_reason,
                         mean_improvement, shrinkage, effect_shape, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (item["proposal_key"], project, forecast_id, series,
                     int(bool(admission.get("admitted"))),
                     int(bool(admission.get("published"))),
                     admission.get("lane", "fold_gate"),
                     admission.get("decided_by"),
                     admission.get("exclusion_reason"),
                     admission.get("mean_improvement"),
                     admission.get("shrinkage"),
                     admission.get("effect_shape"), created_at),
                )
                written += 1
        return written

    def record_effect_scenarios(
        self, project: str, forecast_id: str, series: str,
        primary: list[dict[str, Any]], scenarios: list[dict[str, Any]],
        context_events: list[Any],
    ) -> int:
        """Persist immutable baseline/scenario pairs for later learning."""
        import hashlib

        by_id = {event.event_id: event for event in context_events}
        primary_by_time = {str(row["timestamp"]): float(row["point"])
                           for row in primary if "timestamp" in row and "point" in row}
        written = 0
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            for scenario in scenarios:
                contract = scenario.get("effect") or {}
                distribution = contract.get("distribution")
                provenance = contract.get("provenance")
                rows = scenario.get("forecast") or []
                event_ids = [str(item) for item in scenario.get("events") or []]
                if not distribution or not provenance or not rows:
                    continue
                scenario_by_time = {str(row["timestamp"]): float(row["point"])
                                    for row in rows if "timestamp" in row and "point" in row}
                timestamps = [timestamp for timestamp in scenario_by_time
                              if timestamp in primary_by_time]
                if not timestamps:
                    continue
                event_types = sorted({by_id[event_id].event_type for event_id in event_ids
                                      if event_id in by_id})
                event_type = event_types[0] if len(event_types) == 1 else (
                    "+".join(event_types) if event_types else None)
                identity = json.dumps({
                    "project": project, "forecast_id": forecast_id,
                    "series": series,
                    "events": sorted(event_ids), "effect": contract,
                    "timestamps": timestamps,
                }, sort_keys=True, separators=(",", ":"))
                effect_id = hashlib.sha256(
                    b"effect_scenario\x00" + identity.encode("utf-8")
                ).hexdigest()[:24]
                conn.execute(
                    """
                    INSERT OR REPLACE INTO effect_scenarios
                        (effect_id, project, forecast_id, series, event_type,
                         event_ids, provenance_class, observed, known_at,
                         source_reference, effect_shape, effect_distribution,
                         effect_provenance, primary_points, scenario_points,
                         timestamps, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (effect_id, project, forecast_id, series, event_type,
                     json.dumps(sorted(event_ids)), provenance["provenance_class"],
                     int(bool(provenance.get("observed"))), provenance["known_at"],
                     provenance.get("source_reference"),
                     contract.get("shape", "unknown"), json.dumps(distribution),
                     json.dumps(provenance),
                     json.dumps([primary_by_time[t] for t in timestamps]),
                     json.dumps([scenario_by_time[t] for t in timestamps]),
                     json.dumps(timestamps), created_at),
                )
                written += 1
        return written

    def _resolve_effect_scenarios(
        self, record: ForecastRecord, matched_actuals: list[float],
        *, outcome_known_at: str | None = None,
    ) -> int:
        """Estimate realised effects against each scenario's frozen baseline.

        This is descriptive rather than causal. Overlapping scenario windows
        are marked confounded instead of receiving individual attribution.
        """
        with self._connect() as conn:
            scenarios = conn.execute(
                "SELECT * FROM effect_scenarios WHERE project = ? AND "
                "forecast_id = ? AND series = ?",
                (record.project, record.forecast_id, record.series),
            ).fetchall()
        if not scenarios:
            return 0
        parsed: dict[str, tuple[list[float], list[float]]] = {}
        active_sets: dict[str, set[int]] = {}
        for row in scenarios:
            primary = [float(value) for value in json.loads(row["primary_points"])]
            scenario = [float(value) for value in json.loads(row["scenario_points"])]
            parsed[row["effect_id"]] = (primary, scenario)
            active_sets[row["effect_id"]] = {
                index for index, pair in enumerate(zip(primary, scenario))
                if not math.isclose(pair[0], pair[1], rel_tol=1e-12, abs_tol=1e-12)
            }
        resolved_at = datetime.now(timezone.utc).isoformat()
        known_at = outcome_known_at or resolved_at
        with self._connect() as conn:
            for row in scenarios:
                effect_id = row["effect_id"]
                primary, _ = parsed[effect_id]
                usable = min(len(primary), len(matched_actuals))
                active = sorted(index for index in active_sets[effect_id] if index < usable)
                inactive = [index for index in range(usable) if index not in active_sets[effect_id]]
                missing = not primary or not active or usable == 0
                overlaps = any(
                    bool(active_sets[effect_id].intersection(indices))
                    for other_id, indices in active_sets.items() if other_id != effect_id
                )
                inactive_residuals = [matched_actuals[index] - primary[index]
                                      for index in inactive]
                baseline_bias = (statistics.mean(inactive_residuals)
                                 if inactive_residuals else 0.0)
                residuals = ([matched_actuals[index] - primary[index] - baseline_bias
                              for index in active] if not missing else [])
                estimate = statistics.mean(residuals) if residuals else None
                standard_error = None
                lower = upper = None
                if len(residuals) >= 2:
                    active_variance = statistics.variance(residuals) / len(residuals)
                    inactive_variance = (statistics.variance(inactive_residuals) /
                                         len(inactive_residuals)
                                         if len(inactive_residuals) >= 2 else 0.0)
                    standard_error = math.sqrt(active_variance + inactive_variance)
                    lower = estimate - 1.2815515655446004 * standard_error
                    upper = estimate + 1.2815515655446004 * standard_error
                elif residuals:
                    lower = upper = estimate
                onset = duration = None
                realised_shape = "unknown"
                if residuals:
                    centre = statistics.median(residuals)
                    noise = 1.4826 * statistics.median(
                        abs(value - centre) for value in residuals)
                    threshold = max(noise, abs(centre) * 0.1, 1e-12)
                    detected = [offset for offset, value in enumerate(residuals)
                                if abs(value) >= threshold]
                    if detected:
                        onset = detected[0]
                        duration = detected[-1] - detected[0] + 1
                    if len(residuals) >= 3:
                        x_mean = (len(residuals) - 1) / 2
                        denominator = sum((index - x_mean) ** 2
                                          for index in range(len(residuals)))
                        slope = (sum((index - x_mean) * (value - centre)
                                     for index, value in enumerate(residuals)) /
                                 denominator if denominator else 0.0)
                        level = max(abs(centre), noise, 1e-12)
                        if abs(slope) * len(residuals) >= 0.5 * level:
                            realised_shape = "trend_change"
                        elif abs(residuals[-1]) < 0.5 * abs(residuals[0]):
                            realised_shape = "temporary_pulse"
                        else:
                            realised_shape = "level_shift"
                conn.execute(
                    """
                    INSERT OR REPLACE INTO event_effect_registry
                        (effect_id, project, forecast_id, series, event_type,
                         resolved_at, outcome_known_at, estimate, baseline_bias,
                         standard_error, lower, upper,
                         interval_probability, onset_steps, duration_steps,
                         expected_effect_shape, realised_effect_shape,
                         sample_count, confounding_status, counterfactual_quality,
                         missing_counterfactual, residuals)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (effect_id, record.project, record.forecast_id, record.series,
                     row["event_type"], resolved_at, known_at, estimate,
                     baseline_bias, standard_error,
                     lower, upper, 0.8 if standard_error is not None else None,
                     onset, duration, row["effect_shape"], realised_shape,
                     len(residuals),
                     "overlapping_scenarios" if overlaps else "none_detected",
                     ("inactive_horizon_bias_adjusted" if inactive_residuals
                      else "frozen_forecast_only"),
                     int(missing), json.dumps(residuals)),
                )
        return len(scenarios)

    def record_effect_occurrence(
        self, effect_id: str, status: str, *, known_at: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Confirm, cancel, or revise the event behind a frozen scenario.

        Actual values cannot establish occurrence.  This explicit event-time
        outcome prevents a cancelled deployment from entering organizational
        memory as though it happened merely because its forecast horizon
        elapsed.
        """
        allowed = {"confirmed", "cancelled", "revised"}
        if status not in allowed:
            raise GnomonError(
                "INVALID_ARGUMENTS",
                f"effect occurrence status must be one of {', '.join(sorted(allowed))}.",
            )
        try:
            parsed = datetime.fromisoformat(known_at)
        except (TypeError, ValueError) as exc:
            raise GnomonError(
                "INVALID_ARGUMENTS", "effect occurrence known_at must be ISO-8601."
            ) from exc
        if parsed.tzinfo is None:
            raise GnomonError(
                "INVALID_ARGUMENTS",
                "effect occurrence known_at requires an explicit timezone offset.",
            )
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT effect_id FROM effect_scenarios WHERE effect_id = ?",
                (effect_id,),
            ).fetchone()
            if existing is None:
                raise GnomonError(
                    "EFFECT_NOT_FOUND", f"No tracked effect has id {effect_id!r}.",
                )
            conn.execute(
                "UPDATE effect_scenarios SET occurrence_status = ?, "
                "occurrence_known_at = ?, occurrence_note = ? WHERE effect_id = ?",
                (status, known_at, note, effect_id),
            )
        return {"effect_id": effect_id, "status": status,
                "known_at": known_at, "note": note}

    def event_effects(
        self, project: str, *, event_type: str | None = None,
        series: str | None = None, include_unresolved: bool = True,
    ) -> list[dict[str, Any]]:
        """Query the auditable event-effect memory for an organization."""
        clauses = ["s.project = ?"]
        parameters: list[Any] = [project]
        if event_type is not None:
            clauses.append("s.event_type = ?")
            parameters.append(event_type)
        if series is not None:
            clauses.append("s.series = ?")
            parameters.append(series)
        if not include_unresolved:
            clauses.append("r.effect_id IS NOT NULL")
        query = f"""
            SELECT s.*, r.resolved_at, r.outcome_known_at,
                   r.estimate, r.baseline_bias, r.standard_error,
                   r.lower, r.upper, r.interval_probability, r.onset_steps,
                   r.duration_steps, r.expected_effect_shape,
                   r.realised_effect_shape, r.sample_count, r.confounding_status,
                   r.counterfactual_quality, r.missing_counterfactual, r.residuals
            FROM effect_scenarios s
            LEFT JOIN event_effect_registry r ON r.effect_id = s.effect_id
            WHERE {' AND '.join(clauses)}
            ORDER BY s.known_at, s.effect_id
        """
        with self._connect() as conn:
            result: list[dict[str, Any]] = []
            for row in conn.execute(query, parameters):
                item = dict(row)
                for name in ("event_ids", "effect_distribution",
                             "effect_provenance", "timestamps", "residuals"):
                    if item.get(name) is not None:
                        item[name] = json.loads(item[name])
                item.pop("primary_points", None)
                item.pop("scenario_points", None)
                item["observed"] = bool(item["observed"])
                item["eligible_for_learning"] = (
                    item.get("occurrence_status") == "confirmed"
                    and item.get("resolved_at") is not None
                    and not bool(item.get("missing_counterfactual"))
                    and item.get("confounding_status") == "none_detected"
                )
                if item.get("missing_counterfactual") is not None:
                    item["missing_counterfactual"] = bool(item["missing_counterfactual"])
                result.append(item)
            return result

    def pooled_effect_prior(
        self, project: str, event_type: str, series: str, *,
        as_of: str, exclude_effect_id: str | None = None,
        require_validation: bool = True,
    ) -> dict[str, Any]:
        """Leakage-safe organizational analogue prior.

        Only confirmed, resolved, unconfounded effects known by ``as_of``
        enter the pool. Local estimates shrink toward the organization-wide
        event-type mean; leave-one-event-out error must beat a zero-effect
        baseline before the result is eligible to influence a scenario.
        """
        try:
            cutoff = datetime.fromisoformat(as_of)
        except (TypeError, ValueError) as exc:
            raise GnomonError(
                "INVALID_ARGUMENTS", "effect prior as_of must be ISO-8601."
            ) from exc
        if cutoff.tzinfo is None:
            raise GnomonError(
                "INVALID_ARGUMENTS", "effect prior as_of requires a timezone offset."
            )
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.effect_id, s.series, s.known_at, r.outcome_known_at,
                       r.estimate, r.standard_error, r.sample_count
                FROM effect_scenarios s
                JOIN event_effect_registry r ON r.effect_id = s.effect_id
                WHERE s.project = ? AND s.event_type = ?
                  AND s.occurrence_status = 'confirmed'
                  AND r.missing_counterfactual = 0
                  AND r.confounding_status = 'none_detected'
                  AND r.estimate IS NOT NULL
                """,
                (project, event_type),
            ).fetchall()
        eligible = [row for row in rows
                    if datetime.fromisoformat(row["outcome_known_at"]) <= cutoff
                    and row["effect_id"] != exclude_effect_id]

        def estimate(members: list[sqlite3.Row], target_series: str) -> dict[str, Any] | None:
            if not members:
                return None
            all_values = [float(row["estimate"]) for row in members]
            local_values = [float(row["estimate"]) for row in members
                            if row["series"] == target_series]
            organization_mean = statistics.mean(all_values)
            if local_values:
                local_mean = statistics.mean(local_values)
                weight = len(local_values) / (len(local_values) + EFFECT_POOL_SHRINKAGE)
                location = weight * local_mean + (1 - weight) * organization_mean
                provenance = "same_event_same_series"
            else:
                local_mean, weight = None, 0.0
                location = organization_mean
                provenance = "same_event_related_series"
            # This prior predicts the *next event*, not merely the population
            # mean. Dividing between-event variance by n produced a precise
            # estimate of the mean but a catastrophically narrow predictive
            # interval. Posterior-predictive variance retains heterogeneity
            # and adds the typical measurement variance.
            between_variance = (statistics.variance(all_values)
                                if len(all_values) >= 2 else 0.0)
            measurement_variance = statistics.mean(
                float(row["standard_error"] or 0.0) ** 2 for row in members)
            scale = math.sqrt(between_variance + measurement_variance)
            return {
                "location": location, "scale": scale,
                "lower": location - 1.2815515655446004 * scale,
                "upper": location + 1.2815515655446004 * scale,
                "sample_count": len(all_values),
                "local_sample_count": len(local_values),
                "local_mean": local_mean,
                "organization_mean": organization_mean,
                "local_weight": weight,
                "provenance_class": provenance,
            }

        pooled = estimate(eligible, series)
        loo_errors: list[float] = []
        zero_errors: list[float] = []
        for held_out in eligible:
            training = [row for row in eligible
                        if row["effect_id"] != held_out["effect_id"]]
            prediction = estimate(training, str(held_out["series"]))
            if prediction is None:
                continue
            truth = float(held_out["estimate"])
            loo_errors.append(abs(truth - float(prediction["location"])))
            zero_errors.append(abs(truth))
        validation = {
            "method": "leave_one_event_out",
            "episodes": len(loo_errors),
            "pooled_mae": statistics.mean(loo_errors) if loo_errors else None,
            "zero_effect_mae": statistics.mean(zero_errors) if zero_errors else None,
            "beats_zero_effect": bool(loo_errors) and (
                statistics.mean(loo_errors) < statistics.mean(zero_errors)),
        }
        if pooled is not None and loo_errors:
            ordered = sorted(loo_errors)
            # Split-conformal finite-sample rank for 80% predictive coverage.
            rank = min(len(ordered), math.ceil(0.8 * (len(ordered) + 1)))
            conformal_half_width = ordered[rank - 1]
            parametric_half_width = 1.2815515655446004 * pooled["scale"]
            half_width = max(conformal_half_width, parametric_half_width)
            pooled["scale"] = half_width / 1.2815515655446004
            pooled["lower"] = pooled["location"] - half_width
            pooled["upper"] = pooled["location"] + half_width
            validation["conformal_half_width"] = conformal_half_width
            validation["predictive_half_width"] = half_width
            validation["calibration_rank"] = rank
        validated = (
            len(loo_errors) >= MIN_EFFECT_POOL_EPISODES
            and validation["beats_zero_effect"]
        )
        if pooled is None:
            return {"status": "insufficient_evidence", "event_type": event_type,
                    "series": series, "as_of": as_of, "validation": validation}
        return {
            "status": ("validated" if validated else "insufficient_evidence"),
            "event_type": event_type, "series": series, "as_of": as_of,
            "eligible_for_scenario": validated or not require_validation,
            "distribution": {
                "distribution": "normal", "location": pooled["location"],
                "scale": pooled["scale"], "lower": pooled["lower"],
                "upper": pooled["upper"], "interval_probability": 0.8,
                "sample_count": pooled["sample_count"],
                "unit": "target_series_units",
            },
            "provenance": {
                "provenance_class": pooled["provenance_class"],
                "observed": True, "known_at": as_of,
                "source_reference": f"tracking:{project}",
                "similarity": (1.0 if pooled["local_sample_count"] else None),
                "reliability": min(1.0, len(loo_errors) / 20.0),
                "method": "hierarchical mean shrinkage toward organization event type",
            },
            "pooling": {key: pooled[key] for key in (
                "local_sample_count", "local_mean", "organization_mean",
                "local_weight")},
            "validation": validation,
            "exclusions": {
                "unconfirmed_or_unresolved": len(rows) - len(eligible),
                "excluded_effect_id": exclude_effect_id,
            },
        }

    def _resolve_event_outcomes(
        self, record: ForecastRecord, matched_actuals: list[float],
        score: "ScoreResult",
    ) -> int:
        """Score this forecast's published event proposals against actuals.

        The realised lift needs the history-only counterfactual the
        pipeline recorded at admission time (`enrichment_counterfactual`
        evidence); without it — or without any published proposal — there
        is nothing to attribute and nothing is written.
        """
        with self._connect() as conn:
            published = conn.execute(
                "SELECT proposal_key FROM event_admissions "
                "WHERE project = ? AND forecast_id = ? AND series = ? "
                "AND published = 1",
                (record.project, record.forecast_id, record.series),
            ).fetchall()
        if not published:
            return 0
        counterfactual: list[float] | None = None
        base_model: str | None = None
        evidence_path = Path(record.artifact_path or "") / "evidence.jsonl"
        if evidence_path.exists():
            for line in evidence_path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (item.get("kind") == "enrichment_counterfactual"
                        and item.get("series") == record.series):
                    payload = item.get("payload") or {}
                    raw = payload.get("points") or []
                    counterfactual = [float(value) for value in raw]
                    base_model = payload.get("base_model")
                    break
        if not counterfactual or len(counterfactual) != len(matched_actuals):
            logger.info(
                "No usable counterfactual for %s: outcomes stay open",
                record.forecast_id,
            )
            return 0
        from .evaluation import error_score
        base_wape = error_score(matched_actuals, counterfactual)
        published_wape = score.wape
        lift = (base_wape - published_wape
                if base_wape is not None and published_wape is not None
                else None)
        resolved_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            for row in published:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO event_outcomes
                        (proposal_key, project, forecast_id, series,
                         resolved_at, base_model, published_model, base_wape,
                         published_wape, realised_lift_wape, direction_hit,
                         brier, attribution)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL,
                            'event_set')
                    """,
                    (row["proposal_key"], record.project, record.forecast_id,
                     record.series, resolved_at, base_model,
                     record.selected_model, base_wape, published_wape, lift),
                )
        return len(published)

    def proposer_skill(
        self, project: str, proposer_id: str | None = None,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Shrunk per-(proposer, event-type) skill from resolved outcomes.

        ``shrunk_lift`` pulls the mean realised lift toward 0 with
        ``PROPOSER_SKILL_SHRINKAGE`` pseudo-observations, and
        ``shrunk_hit_rate`` pulls the directional hit rate toward 0.5, so
        small-n cells cannot outrank measured ones. Cold start is explicit:
        a proposer with no resolved outcomes has ``shrunk_lift`` 0.
        """
        clauses = ["p.project = ?"]
        parameters: list[Any] = [project]
        if proposer_id is not None:
            clauses.append("p.proposer_id = ?")
            parameters.append(proposer_id)
        if event_type is not None:
            clauses.append("p.event_type = ?")
            parameters.append(event_type)
        query = f"""
            SELECT p.proposer_id, p.proposer_kind, p.event_type,
                   COUNT(*) AS proposals,
                   SUM(COALESCE(a.admitted, 0)) AS admitted,
                   SUM(COALESCE(a.published, 0)) AS published,
                   COUNT(o.realised_lift_wape) AS resolved,
                   SUM(o.realised_lift_wape) AS total_lift,
                   SUM(o.direction_hit) AS direction_hits,
                   COUNT(o.direction_hit) AS direction_calls
            FROM event_proposals p
            LEFT JOIN event_admissions a
                ON a.project = p.project AND a.forecast_id = p.forecast_id
               AND a.series = p.series AND a.proposal_key = p.proposal_key
            LEFT JOIN event_outcomes o
                ON o.project = p.project AND o.forecast_id = p.forecast_id
               AND o.series = p.series AND o.proposal_key = p.proposal_key
            WHERE {' AND '.join(clauses)}
            GROUP BY p.proposer_id, p.proposer_kind, p.event_type
            ORDER BY p.proposer_id, p.event_type
        """
        k = PROPOSER_SKILL_SHRINKAGE
        rows: list[dict[str, Any]] = []
        with self._connect() as conn:
            for row in conn.execute(query, parameters):
                resolved = row["resolved"] or 0
                total_lift = row["total_lift"] or 0.0
                calls = row["direction_calls"] or 0
                hits = row["direction_hits"] or 0
                rows.append({
                    "proposer_id": row["proposer_id"],
                    "proposer_kind": row["proposer_kind"],
                    "event_type": row["event_type"],
                    "proposals": row["proposals"],
                    "admitted": row["admitted"] or 0,
                    "published": row["published"] or 0,
                    "resolved": resolved,
                    "mean_lift_wape": (total_lift / resolved
                                       if resolved else None),
                    "shrunk_lift_wape": total_lift / (resolved + k),
                    "direction_calls": calls,
                    "shrunk_hit_rate": (hits + 0.5 * k) / (calls + k),
                    "shrinkage_k": k,
                    "note": ("observational, set-level attribution: the "
                             "gate admits event sets, so co-admitted "
                             "proposals share one measured lift"),
                })
        return rows

    def submit_actuals_csv(
        self, project: str, csv_path: str,
        time_column: str | None = None, target_column: str | None = None,
        series_column: str | None = None,
    ) -> list[ScoreResult]:
        """Submit actuals from a CSV file and score all unscored forecasts.

        ``time_column`` and ``target_column`` are named rather than guessed
        positionally when supplied. Guessing by position turned a perfectly
        good operator export of `requests,timestamp,host` into a silent
        `{"scored": 0}` — indistinguishable from "nothing was due".
        """
        path = Path(csv_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Actuals file not found: {path}")

        actuals: list[tuple] = []
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            ts_col, val_col, series_col = self._resolve_actuals_columns(
                cols, time_column, target_column, series_column,
            )
            # An explicit knowledge-time column turns the submission into
            # a backfill: each outcome is stamped with when it became
            # knowable rather than with this submission's clock.
            known_col = "known_at" if "known_at" in cols else None
            for row in reader:
                ts = row[ts_col]
                try:
                    val = float(row[val_col])
                except (ValueError, TypeError):
                    continue
                known_at = (row[known_col] or "").strip() or None if known_col else None
                series = row[series_col] if series_col else None
                if known_at is not None or series_col:
                    actuals.append((series, ts, val, known_at))
                else:
                    actuals.append((ts, val))

        logger.info("Loaded %d actuals from %s", len(actuals), path)
        return self.submit_actuals(project, actuals, ts_col, val_col)

    @staticmethod
    def _resolve_actuals_columns(
        columns: list[str], time_column: str | None, target_column: str | None,
        series_column: str | None,
    ) -> tuple[str, str, str | None]:
        """Named columns win; otherwise infer, and refuse to guess blind."""
        missing = [
            name for name in (time_column, target_column, series_column)
            if name is not None and name not in columns
        ]
        if missing:
            raise GnomonError(
                "MISSING_COLUMNS",
                f"Actuals file is missing: {', '.join(missing)}",
                {"available_columns": columns, "missing_columns": missing},
            )
        series_col = series_column or ("series" if "series" in columns else None)
        # `known_at` is a reserved knowledge-time column, never a
        # timestamp/value candidate — leaving it in would turn a
        # perfectly inferable file into an AMBIGUOUS_SCHEMA refusal.
        remaining = [name for name in columns
                     if name != series_col and name.lower() != "known_at"]

        ts_col = time_column
        if ts_col is None:
            named = [name for name in remaining
                     if name.lower() in {"timestamp", "time", "date", "ts"}]
            if named:
                ts_col = named[0]
            elif len(remaining) == 2:
                # Two columns and no naming convention: position is the only
                # signal there is, and it is disclosed by being documented.
                ts_col = remaining[0]
            else:
                raise GnomonError(
                    "AMBIGUOUS_SCHEMA",
                    "Cannot tell which column holds the timestamp. Pass "
                    "--time explicitly.",
                    {"available_columns": columns, "argument": "--time"},
                )
        val_col = target_column
        if val_col is None:
            named = [name for name in remaining
                     if name.lower() in {"value", "actual", "actuals"} and name != ts_col]
            candidates = [name for name in remaining if name != ts_col]
            if named:
                val_col = named[0]
            elif len(candidates) == 1:
                val_col = candidates[0]
            else:
                raise GnomonError(
                    "AMBIGUOUS_SCHEMA",
                    "Cannot tell which column holds the actual value. Pass "
                    "--target explicitly.",
                    {"available_columns": columns, "candidates": candidates,
                     "argument": "--target"},
                )
        return ts_col, val_col, series_col

    def explain_unscored(
        self, project: str, actual_timestamps: list[str],
    ) -> dict[str, Any]:
        """Why nothing scored, in terms a caller can act on.

        `{"scored": 0}` is the failure mode of the exact follow-up loop the
        product promises, and it was indistinguishable from "nothing was
        due yet".
        """
        forecasts = self.list_forecasts(project, limit=1000)
        open_forecasts = [record for record in forecasts if not record.scored]
        normalised = {self._normalise_timestamp(item) for item in actual_timestamps}

        windows: list[dict[str, Any]] = []
        overlap_total = 0
        for record in open_forecasts:
            artifact = Path(record.artifact_path) / "forecast.csv"
            if not artifact.exists():
                windows.append({
                    "forecast_id": record.forecast_id, "series": record.series,
                    "problem": "artifact_missing", "artifact_path": str(artifact),
                })
                continue
            rows = [
                row for row in self._load_forecast_csv(artifact)
                if row.get("series", "__default__") == record.series
            ]
            wanted = {self._normalise_timestamp(row["timestamp"]) for row in rows}
            overlap = wanted & normalised
            overlap_total += len(overlap)
            windows.append({
                "forecast_id": record.forecast_id,
                "series": record.series,
                "needs_timestamps": sorted(wanted)[:1] + sorted(wanted)[-1:],
                "periods_required": len(wanted),
                "periods_supplied": len(overlap),
            })

        if not forecasts:
            reason = f"no forecasts are registered in project {project!r}"
        elif not open_forecasts:
            reason = "every registered forecast is already scored"
        elif overlap_total == 0:
            reason = (
                "the supplied actuals do not overlap any open forecast's "
                "horizon window"
            )
        else:
            reason = (
                "open forecasts overlap the actuals only partially; a forecast "
                "scores when every one of its periods has an actual"
            )
        supplied = sorted(normalised)
        return {
            "scored": 0,
            "reason": reason,
            "registered_forecasts": len(forecasts),
            "open_forecasts": len(open_forecasts),
            "actuals_supplied": len(normalised),
            "actuals_window": (
                {"first": supplied[0], "last": supplied[-1]} if supplied else None
            ),
            "open_forecast_windows": windows,
            "repair_options": [
                {"action": "check_window",
                 "description": "Compare actuals_window with each entry's "
                                "needs_timestamps; a forecast scores only when "
                                "every period in its horizon has an actual."},
                {"action": "name_columns",
                 "description": "If the file's columns were read wrongly, pass "
                                "--time and --target explicitly."},
                {"action": "list_open",
                 "description": "Run `gnomon track list --project <name>` to see "
                                "what is awaiting actuals."},
            ],
        }

    # ---- Performance / Leaderboard ----

    def leaderboard(self, project: str, task: str | None = None) -> list[ModelPerformance]:
        """Get ranked model performance for a project, optionally per task."""
        task_filter = "" if task is None else " AND COALESCE(task, 'forecast') = ?"
        params: tuple[Any, ...] = (project,) if task is None else (project, task)
        with self._connect() as conn:
            rows = conn.execute(f"""
                SELECT
                    model,
                    COUNT(*) as count,
                    AVG(mase) as avg_mase,
                    AVG(wape) as avg_wape,
                    AVG(mape) as avg_mape,
                    AVG(bias) as avg_bias,
                    AVG(coverage) as avg_coverage,
                    AVG(threshold_accuracy) as avg_thresh,
                    MAX(scored_at) as last_scored,
                    (SELECT recent.mase FROM model_performance AS recent
                     WHERE recent.project = model_performance.project
                       AND recent.model = model_performance.model
                     ORDER BY recent.scored_at DESC LIMIT 1) AS last_mase
                FROM model_performance
                WHERE project = ?{task_filter}
                GROUP BY model
                ORDER BY AVG(mase) IS NULL, AVG(mase) ASC
            """, params).fetchall()

        results = []
        for r in rows:
            results.append(ModelPerformance(
                model=r["model"],
                count=r["count"],
                avg_mase=r["avg_mase"],
                avg_wape=r["avg_wape"],
                avg_mape=r["avg_mape"],
                avg_bias=r["avg_bias"],
                avg_coverage=r["avg_coverage"],
                avg_threshold_accuracy=r["avg_thresh"],
                last_mase=r["last_mase"],
                last_scored=r["last_scored"],
            ))

        return results

    def due_forecasts(
        self, project: str | None = None, now: str | None = None,
    ) -> list[dict[str, Any]]:
        """List unscored forecasts and whether their full horizon is due."""
        current = datetime.fromisoformat(now) if now else datetime.now(timezone.utc)
        if current.utcoffset() is None:
            current = current.replace(tzinfo=timezone.utc)
        records = self.list_forecasts(project, limit=1000)
        due: list[dict[str, Any]] = []
        for record in records:
            if record.scored:
                continue
            rows = [
                row for row in self._load_forecast_csv(Path(record.artifact_path) / "forecast.csv")
                if row.get("series", "__default__") == record.series
            ]
            horizon_end = rows[-1]["timestamp"] if rows else None
            parsed_end = None
            if horizon_end:
                try:
                    parsed_end = datetime.fromisoformat(
                        self._normalise_timestamp(horizon_end)
                    )
                    if parsed_end.utcoffset() is None:
                        parsed_end = parsed_end.replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            due.append({
                "forecast_id": record.forecast_id,
                "project": record.project,
                "series": record.series,
                "model": record.selected_model,
                "horizon_end": horizon_end,
                "state": "due" if parsed_end and parsed_end <= current else "awaiting_horizon",
                "artifact_path": record.artifact_path,
            })
        return due

    def record_decision(
        self, decision_id: str, project: str, forecast_id: str,
        action: str, expected_outcome: str,
    ) -> DecisionRecord:
        if self.get_forecast(forecast_id) is None:
            raise ValueError(f"Forecast {forecast_id} not found in registry")
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO decisions
                    (decision_id, project, forecast_id, action, expected_outcome, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(decision_id) DO UPDATE SET
                    action = excluded.action,
                    expected_outcome = excluded.expected_outcome
            """, (decision_id, project, forecast_id, action, expected_outcome, created_at))
        return self.get_decision(decision_id)  # type: ignore[return-value]

    def resolve_decision(
        self, decision_id: str, actual_outcome: str, correct: bool,
    ) -> DecisionRecord:
        resolved_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute("""
                UPDATE decisions SET actual_outcome = ?, correct = ?, resolved_at = ?
                WHERE decision_id = ?
            """, (actual_outcome, int(correct), resolved_at, decision_id))
            if cursor.rowcount == 0:
                raise ValueError(f"Decision {decision_id} not found")
        return self.get_decision(decision_id)  # type: ignore[return-value]

    def get_decision(self, decision_id: str) -> DecisionRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM decisions WHERE decision_id = ?", (decision_id,),
            ).fetchone()
        return self._row_to_decision(row) if row else None

    def list_decisions(self, project: str | None = None) -> list[DecisionRecord]:
        with self._connect() as conn:
            if project:
                rows = conn.execute(
                    "SELECT * FROM decisions WHERE project = ? ORDER BY created_at DESC",
                    (project,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM decisions ORDER BY created_at DESC"
                ).fetchall()
        return [self._row_to_decision(row) for row in rows]

    # -- Decision artifacts (Phase 6 model; legacy decisions kept intact) --

    def save_decision_artifact(self, artifact: Any) -> Any:
        import json as _json
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO decision_artifacts
                    (decision_id, project, forecast_id, selected_action,
                     created_at, resolved_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(decision_id) DO UPDATE SET
                    selected_action = excluded.selected_action,
                    resolved_at = excluded.resolved_at,
                    payload = excluded.payload
            """, (
                artifact.decision_id, artifact.project, artifact.forecast_id,
                artifact.selected_action, artifact.created_at,
                artifact.resolved_at, _json.dumps(artifact.to_dict()),
            ))
        return artifact

    def get_decision_artifact(self, decision_id: str) -> Any | None:
        import json as _json
        from .decision_model import DecisionArtifact
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM decision_artifacts WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
        if row is not None:
            return DecisionArtifact.from_dict(_json.loads(row["payload"]))
        legacy = self.get_decision(decision_id)
        return DecisionArtifact.from_legacy(legacy) if legacy else None

    def list_decision_artifacts(self, project: str | None = None) -> list[Any]:
        """New-model artifacts plus v0.2 DecisionRecords loaded as degraded
        artifacts under the versioning rule — no stored project breaks."""
        import json as _json
        from .decision_model import DecisionArtifact
        with self._connect() as conn:
            if project:
                rows = conn.execute(
                    "SELECT payload FROM decision_artifacts WHERE project = ? "
                    "ORDER BY created_at DESC", (project,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT payload FROM decision_artifacts ORDER BY created_at DESC",
                ).fetchall()
        artifacts = [DecisionArtifact.from_dict(_json.loads(row["payload"])) for row in rows]
        modern_ids = {artifact.decision_id for artifact in artifacts}
        for legacy in self.list_decisions(project):
            if legacy.decision_id not in modern_ids:
                artifacts.append(DecisionArtifact.from_legacy(legacy))
        return artifacts

    def resolve_decision_outcome(
        self, decision_id: str, *,
        realised_scenario: str | None = None,
        realised_utilities: dict[str, float] | None = None,
        constraint_violations: list[str] | None = None,
        note: str | None = None,
        resolved_at: str | None = None,
    ) -> Any:
        from .decision_model import score_outcome
        artifact = self.get_decision_artifact(decision_id)
        if artifact is None:
            raise ValueError(f"Decision {decision_id} not found")
        artifact.outcome = score_outcome(
            artifact, realised_scenario=realised_scenario,
            realised_utilities=realised_utilities,
            constraint_violations=constraint_violations, note=note,
        )
        artifact.resolved_at = resolved_at or datetime.now(timezone.utc).isoformat()
        return self.save_decision_artifact(artifact)

    def status(self, project: str | None = None) -> dict[str, Any]:
        """Pollable view: open forecasts, due horizons, unresolved decisions,
        realised-performance summaries. Descriptive, never causal."""
        open_forecasts = [
            {
                "forecast_id": item.forecast_id, "project": item.project,
                "series": item.series, "model": item.selected_model,
                "support": item.support, "created_at": item.created_at,
            }
            for item in self.list_forecasts(project=project, limit=500)
            if not item.scored
        ]
        unresolved = [
            {
                "decision_id": artifact.decision_id, "project": artifact.project,
                "selected_action": artifact.selected_action,
                "degraded": artifact.degraded, "created_at": artifact.created_at,
            }
            for artifact in self.list_decision_artifacts(project)
            if artifact.resolved_at is None
        ]
        resolved = [a for a in self.list_decision_artifacts(project)
                    if a.resolved_at is not None and a.outcome is not None]
        regrets = [a.outcome.regret for a in resolved if a.outcome.regret is not None]
        decision_summary = {
            "resolved": len(resolved),
            "with_regret_scored": len(regrets),
            "mean_regret": (sum(regrets) / len(regrets)) if regrets else None,
            "ex_ante_optimal": sum(
                1 for a in resolved if a.outcome.ex_ante_optimal is True
            ),
        }
        leaderboard = []
        if project:
            leaderboard = [
                {"model": m.model, "count": m.count, "avg_mase": m.avg_mase,
                 "avg_coverage": m.avg_coverage}
                for m in self.leaderboard(project)
            ]
        return {
            "schema_version": "0.1",
            "project": project,
            "open_forecasts": open_forecasts,
            "due": self.due_forecasts(project),
            "unresolved_decisions": unresolved,
            "decision_summary": decision_summary,
            "model_performance": leaderboard,
            "coverage_adaptation": self.coverage_adaptation(project) if project else [],
            "temporal_answers": ({
                "open": len(self.temporal_answer_receipts(project, resolved=False)),
                "resolved": len(self.temporal_answer_receipts(project, resolved=True)),
            } if project else {"open": 0, "resolved": 0}),
            "warning": "Realised performance is observational evidence, never causal.",
        }

    def coverage_adaptation(
        self, project: str, as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        """The adapted miscoverage level per scope, from realised outcomes.

        `adapted_alpha` replays the coverage log into a corrected level and
        is carefully written, deterministic, and covered by tests — and had
        zero call sites outside them, so a complete capability was
        unreachable from every surface. It is reported here.

        **Reported, not applied.** Published intervals still use the
        nominal level; folding this in would move numbers for tracked
        projects, which is a behavioural change that deserves to be asked
        for rather than inherited.
        """
        with self._connect() as conn:
            scopes = [
                row["scope"] for row in conn.execute(
                    "SELECT DISTINCT scope FROM conformal_adaptation "
                    "WHERE project = ? ORDER BY scope",
                    (project,),
                )
            ]
        return [
            {"scope": scope, "applied_to_published_intervals": False,
             **self.adapted_alpha(project, scope, as_of)}
            for scope in scopes
        ]

    def export_snapshot(self, project: str | None = None) -> dict[str, Any]:
        """Return a portable JSON snapshot; immutable artifacts remain separate."""
        forecasts = self.list_forecasts(project, limit=100000)
        decisions = self.list_decisions(project)
        return {
            "schema_version": "2",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "project": project,
            "forecasts": [item.__dict__ for item in forecasts],
            "decisions": [item.__dict__ for item in decisions],
        }

    def relocate_artifact(self, forecast_id: str, artifact_path: str) -> ForecastRecord:
        path = Path(artifact_path).expanduser().resolve()
        if not (path / "forecast.csv").is_file():
            raise ValueError(f"Artifact directory has no forecast.csv: {path}")
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE forecasts SET artifact_path = ? WHERE forecast_id = ?",
                (str(path), forecast_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Forecast {forecast_id} not found")
        return self.get_forecast(forecast_id)  # type: ignore[return-value]

    def model_performance(
        self, project: str, model: str, task: str | None = None,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get per-forecast performance history for a specific model.

        ``as_of`` replays the leaderboard as it stood at that instant,
        filtering on when each score became *derivable* (`known_time`)
        rather than when its row was written. A backfill submitted today
        for outcomes knowable last month belongs in last month's replay;
        a score needing an outcome nobody had at ``as_of`` does not.
        """
        clauses = ["project = ?", "model = ?"]
        params: list[Any] = [project, model]
        if task is not None:
            clauses.append("COALESCE(task, 'forecast') = ?")
            params.append(task)
        if as_of is not None:
            clauses.append("COALESCE(known_time, scored_at) <= ?")
            params.append(self._normalise_timestamp(as_of))
        with self._connect() as conn:
            rows = conn.execute(f"""
                SELECT * FROM model_performance
                WHERE {' AND '.join(clauses)}
                ORDER BY scored_at DESC
            """, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    # ---- Routing ----

    def record_route(
        self,
        route_id: str,
        project: str,
        task: str,
        *,
        series: str | None = None,
        fingerprint: str | None = None,
        recommendation: str | None = None,
        basis: str = "backtest_required",
        payload: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> None:
        """Record one routing decision so replay can reproduce the choice."""
        created_at = created_at or datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO routing_decisions
                    (route_id, project, task, series, fingerprint,
                     recommendation, basis, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (route_id, project, task, series, fingerprint, recommendation,
                  basis, json.dumps(payload or {}, sort_keys=True), created_at))

    def list_routes(self, project: str | None = None) -> list[dict[str, Any]]:
        """List recorded routing decisions, newest first."""
        with self._connect() as conn:
            if project:
                rows = conn.execute(
                    "SELECT * FROM routing_decisions WHERE project = ? ORDER BY created_at DESC",
                    (project,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM routing_decisions ORDER BY created_at DESC",
                ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            results.append(item)
        return results

    # ---- Drift Detection ----

    def _check_drift(
        self, record: ForecastRecord, new_mase: float | None,
    ) -> str | None:
        """Check if the new score indicates model drift.

        Compares the new MASE to the model's historical average.
        Returns a drift flag string, or None if no drift.
        """
        model = record.selected_model
        if model is None or new_mase is None:
            return None

        with self._connect() as conn:
            rows = conn.execute("""
                SELECT mase FROM model_performance
                WHERE project = ? AND model = ? AND series = ?
                  AND horizon = ? AND frequency = ? AND mase IS NOT NULL
                ORDER BY scored_at DESC LIMIT 5
            """, (record.project, model, record.series, record.horizon, record.frequency)).fetchall()

        if len(rows) < 5:
            return None  # Not enough history

        historical = [r["mase"] for r in rows if r["mase"] is not None]
        if not historical:
            return None

        baseline = statistics.median(historical)
        if baseline <= 0:
            return None

        degradation = (new_mase - baseline) / baseline

        if degradation > 0.50:
            return f"degraded: MASE {new_mase:.3f} vs recent median {baseline:.3f} (+{degradation:.0%})"
        elif degradation > 0.25:
            return f"warning: MASE {new_mase:.3f} vs recent median {baseline:.3f} (+{degradation:.0%})"
        return None

    # ---- Compare ----

    def compare(
        self, forecast_id_a: str, forecast_id_b: str,
        project: str | None = None,
    ) -> dict[str, Any]:
        """Compare two scored forecasts, optionally within one project."""
        a = self.get_forecast(forecast_id_a, project)
        b = self.get_forecast(forecast_id_b, project)
        if a is None or b is None:
            missing = [
                fid for fid, record in
                ((forecast_id_a, a), (forecast_id_b, b)) if record is None
            ]
            raise ValueError(
                f"Not found in the registry: {', '.join(missing)}"
                + (f" (project {project!r})" if project else "")
            )

        comparable = (
            a.series == b.series and a.horizon == b.horizon
            and a.frequency == b.frequency and a.cutoff_time == b.cutoff_time
        )
        winner = None
        if comparable and a.mase is not None and b.mase is not None:
            winner = "a" if a.mase < b.mase else "b"

        return {
            "forecast_a": {
                "id": a.forecast_id, "model": a.selected_model,
                "mase": a.mase, "mape": a.mape, "bias": a.bias,
                "coverage": a.coverage, "threshold_accuracy": a.threshold_accuracy,
            },
            "forecast_b": {
                "id": b.forecast_id, "model": b.selected_model,
                "mase": b.mase, "mape": b.mape, "bias": b.bias,
                "coverage": b.coverage, "threshold_accuracy": b.threshold_accuracy,
            },
            "comparable": comparable,
            "comparison_warning": (
                None if comparable else
                "Forecasts differ in series, cutoff, horizon, or frequency"
            ),
            "winner": winner,
        }

    # ---- Helpers ----

    @staticmethod
    def _normalise_timestamp(value: str) -> str:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return text
        if parsed.utcoffset() is not None:
            parsed = parsed.astimezone(timezone.utc)
        return parsed.isoformat()

    def _load_forecast_csv(self, path: Path) -> list[dict[str, Any]]:
        """Load forecast.csv and return rows as dicts."""
        if not path.exists():
            return []
        # Tracking is an artifact read just as surely as get_artifact is.
        # Refuse to score doctored points or intervals when the producing
        # release sealed them; legacy unsealed artifacts remain compatible.
        from .artifacts import verify_artifact_integrity
        verify_artifact_integrity(path.parent)
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = []
            for r in reader:
                entry: dict[str, Any] = {
                    "series": r.get("series", "__default__"),
                    "timestamp": r.get("timestamp", ""),
                }
                try:
                    entry["point"] = float(r.get("point", 0))
                except (ValueError, TypeError):
                    continue
                if "q10" in r:
                    try:
                        entry["q10"] = float(r["q10"])
                    except (ValueError, TypeError):
                        pass
                if "q50" in r:
                    try:
                        entry["q50"] = float(r["q50"])
                    except (ValueError, TypeError):
                        pass
                if "q90" in r:
                    try:
                        entry["q90"] = float(r["q90"])
                    except (ValueError, TypeError):
                        pass
                rows.append(entry)
        return rows

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ForecastRecord:
        return ForecastRecord(
            forecast_id=row["forecast_id"],
            project=row["project"],
            series=row["series"],
            cutoff_time=row["cutoff_time"] or "",
            horizon=row["horizon"],
            frequency=row["frequency"] or "",
            selected_model=row["selected_model"],
            support=row["support"] or "unsupported",
            threshold=row["threshold"],
            threshold_peak_probability=row["threshold_peak_probability"],
            naive_error=row["naive_error"],
            artifact_path=row["artifact_path"] or "",
            created_at=row["created_at"],
            scored=bool(row["scored"]),
            mase=row["mase"],
            wape=row["wape"] if "wape" in row.keys() else None,
            mape=row["mape"],
            bias=row["bias"],
            coverage=row["coverage"],
            threshold_accuracy=row["threshold_accuracy"],
            scored_at=row["scored_at"],
            drift_flag=row["drift_flag"],
            task=(row["task"] if "task" in row.keys() else None) or "forecast",
            fingerprint=row["fingerprint"] if "fingerprint" in row.keys() else None,
        )

    @staticmethod
    def _row_to_decision(row: sqlite3.Row) -> DecisionRecord:
        return DecisionRecord(
            decision_id=row["decision_id"], project=row["project"],
            forecast_id=row["forecast_id"], action=row["action"],
            expected_outcome=row["expected_outcome"],
            actual_outcome=row["actual_outcome"],
            correct=None if row["correct"] is None else bool(row["correct"]),
            created_at=row["created_at"], resolved_at=row["resolved_at"],
        )


def proposal_key(series: str, event: Any) -> str:
    """Content-address a proposal by what was *claimed*.

    Deliberately version-independent (unlike ``ids.content_id``, which
    salts with ``GNOMON_VERSION``): a proposer's track record must survive
    a Gnomon upgrade, or every release resets every ledger. Run-local
    event ids are positional (``event_llm_00``) and never part of the key.
    """
    import hashlib

    from .ids import canonical_json

    source = getattr(event, "source", None)
    payload = {
        "series": series,
        "event_type": event.event_type,
        "effective_start": event.effective_start,
        "effective_end": event.effective_end,
        "entity_scope": sorted(event.entity_scope or ()),
        "source": ({"type": source.type, "reference": source.reference}
                   if source else None),
    }
    digest = hashlib.sha256(
        b"event_proposal\x00" + canonical_json(payload).encode("utf-8")
    )
    return digest.hexdigest()[:16]


def _proposal_rows(series: str, result: Any, evidence: list[Any],
                   context_events: list[Any]) -> list[dict[str, Any]]:
    """Ledger rows for one series: each supplied event, joined to the gate
    verdict already recorded in the artifact's evidence."""
    from .adjudication import COMBINED_MODEL_NAME
    from .context_eval import CONTEXT_MODEL_NAME

    ablation: dict[str, Any] = {}
    gate: dict[str, Any] = {}
    for item in evidence:
        if item.evidence_id == f"context_ablation:{series}":
            ablation = item.payload or {}
        elif item.evidence_id == f"context_gate:{series}":
            gate = item.payload or {}
    admitted_ids = set(ablation.get("events_used") or []) if ablation.get("admitted") else set()
    eligible_ids = set(ablation.get("events_used") or [])
    exclusions = {entry.get("event_id"): entry.get("reason")
                  for entry in (ablation.get("events_excluded") or [])}
    influenced = result.selected_model in (CONTEXT_MODEL_NAME, COMBINED_MODEL_NAME)
    rows: list[dict[str, Any]] = []
    for event in context_events:
        proposer = (event.attributes or {}).get("proposer") or {}
        admitted = event.event_id in admitted_ids
        rows.append({
            "proposal_key": proposal_key(series, event),
            "event_id": event.event_id,
            "event_type": event.event_type,
            "proposer_id": (proposer.get("proposer_id")
                            or proposer.get("model")
                            or event.created_by),
            "proposer_kind": proposer.get("kind") or event.created_by,
            "source_type": event.source.type if event.source else None,
            "source_reference": event.source.reference if event.source else None,
            "status": event.status,
            "confidence": event.confidence,
            "known_at": event.known_at,
            "effective_start": event.effective_start,
            "effective_end": event.effective_end,
            "admission": {
                "admitted": admitted,
                "published": admitted and influenced,
                "lane": "fold_gate",
                "decided_by": gate.get("decided_by"),
                "exclusion_reason": exclusions.get(event.event_id) if event.event_id not in eligible_ids else None,
                "mean_improvement": ablation.get("mean_improvement"),
                "shrinkage": ablation.get("shrinkage"),
                "effect_shape": ablation.get("effect_shape"),
            },
        })
    return rows


def register_artifact(artifact: Any, project: str, artifact_path: str,
                      context_events: list[Any] | None = None) -> list[str]:
    """Register every series in a completed artifact for any integration surface.

    ``context_events`` are the events the run was given, if the caller has
    them: each becomes a ledger row in ``event_proposals`` /
    ``event_admissions``, joined to the gate verdicts the artifact already
    recorded, so proposals can be scored when actuals arrive."""
    from .data import load_observations
    from .temporal import default_season

    schema = artifact.task.schema
    observations, _, _ = load_observations(
        artifact.task.input_path, schema.time_column, schema.target_column,
        schema.series_column,
    )
    histories: dict[str, list[float]] = {}
    cutoffs: dict[str, str] = {}
    for observation in observations:
        histories.setdefault(observation.series, []).append(observation.value)
        cutoffs[observation.series] = observation.timestamp.isoformat()
    store = TrackingStore()
    registered: list[str] = []
    for result in artifact.results:
        values = histories[result.series]
        season = default_season(schema.frequency)
        lag = season if len(values) > season else 1
        errors = [abs(values[index] - values[index - lag])
                  for index in range(lag, len(values))]
        tracking_id = (artifact.forecast_id if result.series == "__default__"
                       else f"{artifact.forecast_id}:{result.series}")
        from .fingerprint import fingerprint_json
        store.register(
            tracking_id, project, series=result.series,
            cutoff_time=cutoffs[result.series], horizon=artifact.task.horizon,
            frequency=schema.frequency, selected_model=result.selected_model,
            support=result.support,
            threshold=result.threshold["value"] if result.threshold else None,
            threshold_peak_probability=(max(result.threshold["probability_above"])
                                        if result.threshold else None),
            naive_error=sum(errors) / len(errors) if errors else None,
            artifact_path=artifact_path,
            task="forecast",
            fingerprint=fingerprint_json(values, schema.frequency),
        )
        registered.append(tracking_id)
        answer_receipt = Path(artifact_path) / "temporal_answers.json"
        if answer_receipt.exists():
            try:
                payload = json.loads(answer_receipt.read_text(encoding="utf-8"))
                store.record_temporal_answers(
                    project, artifact.forecast_id, result.series,
                    list(payload.get("answers") or []),
                    created_at=getattr(artifact, "created_at", None),
                )
            except (OSError, ValueError, TypeError):
                logger.warning("Could not register temporal-answer receipt %s",
                               answer_receipt, exc_info=True)
        if context_events:
            store.record_event_proposals(
                project, tracking_id, result.series,
                _proposal_rows(result.series, result, artifact.evidence,
                               context_events),
            )
            store.record_effect_scenarios(
                project, tracking_id, result.series, result.forecast,
                [*result.conditional_forecasts, *result.sensitivity_scenarios],
                context_events,
            )
            # Fold-admitted context is a selected forecast candidate rather
            # than a scenario-only answer, but the effect registry still
            # needs the immutable history-only path it replaced.  The
            # pipeline persists that path before replacement; register the
            # pair separately so tracking never mistakes the conditioned
            # forecast for its own counterfactual.
            counterfactual = next((
                item.payload for item in artifact.evidence
                if item.kind == "enrichment_counterfactual"
                and item.series == result.series
            ), None)
            context_effect = (result.context or {}).get("effect")
            used_events = list((result.context or {}).get("events_used") or [])
            if counterfactual and context_effect and used_events:
                timestamps = list(counterfactual.get("timestamps") or [])
                points = list(counterfactual.get("points") or [])
                primary_rows = [
                    {"timestamp": timestamp, "point": point}
                    for timestamp, point in zip(timestamps, points)
                ]
                store.record_effect_scenarios(
                    project, tracking_id, result.series, primary_rows,
                    [{
                        "events": used_events,
                        "support": "fold_admitted_context",
                        "effect": context_effect,
                        "forecast": result.forecast,
                    }],
                    context_events,
                )
    return registered

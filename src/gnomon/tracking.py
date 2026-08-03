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
}

#: Bumped to 4 when ``forecasts`` and ``model_performance`` gained their
#: composite keys; to 5 when the proposer-calibration ledger tables
#: (event_proposals / event_admissions / event_outcomes) were added.
SCHEMA_VERSION = "5"

#: Empirical-Bayes shrinkage strength for proposer skill: a proposer's
#: mean lift is pulled toward 0 (and a hit rate toward 0.5) with the
#: weight of this many pseudo-observations, so one lucky resolved call
#: cannot mint a skilled proposer. The existing leaderboard's plain AVG
#: has no such guard; this table does not repeat that defect.
PROPOSER_SKILL_SHRINKAGE = 10.0


class TrackingStore:
    """SQLite-backed forecast registry and scoring store."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_REGISTRY_PATH
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
                                   ("wape", "REAL")):
                if name not in perf_columns:
                    conn.execute(f"ALTER TABLE model_performance ADD COLUMN {name} {sql_type}")
            conn.execute(
                "INSERT OR REPLACE INTO schema_metadata(key, value) VALUES ('version', ?)",
                (SCHEMA_VERSION,),
            )
        self._migrate_composite_keys()

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
            params.append(as_of)
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
    ) -> ScoreResult:
        """Score a single forecast against actuals.

        Computes MASE, MAPE, bias, coverage, and threshold accuracy.
        Stores the result and updates model_performance. ``project`` and
        ``series`` scope the registration being scored; without them the
        most recent registration of the id is used.
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
                         coverage, threshold_accuracy, scored_at, series, horizon, frequency,
                         task, fingerprint)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.project, record.selected_model, forecast_id,
                    mase, wape, mape, bias, cov, thresh_acc, scored_at,
                    record.series, record.horizon, record.frequency,
                    record.task, record.fingerprint,
                ))

        if cov is not None and q10 and q90:
            # Feed the adaptation log, stamped with when the outcome became
            # knowable rather than when it was entered. The forecast's own
            # horizon end is the earliest instant every scored point existed.
            points = min(len(actuals), len(q10), len(q90))
            self.record_coverage_outcome(
                record.project, record.series or "__default__", forecast_id,
                known_time=record.cutoff_time or scored_at,
                covered=round(cov * points), points=points,
            )

        logger.info(
            "Scored forecast %s: MASE=%s MAPE=%.1f%% bias=%.2f coverage=%s drift=%s",
            forecast_id, f"{mase:.3f}" if mase is not None else "N/A", mape, bias,
            f"{cov:.1%}" if cov is not None else "N/A",
            drift or "none",
        )
        return ScoreResult(forecast_id, mase, wape, mape, bias, cov, thresh_acc,
                           scored_at, drift)

    def submit_actuals(
        self,
        project: str,
        actuals: list[tuple[str, float] | tuple[str, str, float]],
        time_column: str = "timestamp",
        target_column: str = "value",
    ) -> list[ScoreResult]:
        """Submit actuals for a project and score all matching forecasts.

        Args:
            project: project name
            actuals: list of (timestamp_iso, value) tuples
            time_column: name of the timestamp column (for CSV loading)
            target_column: name of the target column (for CSV loading)

        Returns:
            List of ScoreResult for each scored forecast.
        """
        # Build timestamp → value map
        actual_map: dict[tuple[str | None, str], float] = {}
        for item in actuals:
            if len(item) == 2:
                ts, val = item
                actual_map[(None, self._normalise_timestamp(ts))] = val
            else:
                series, ts, val = item
                actual_map[(series, self._normalise_timestamp(ts))] = val

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
            for entry in forecast_data:
                ts = self._normalise_timestamp(entry["timestamp"])
                exact_key = (record.series, ts)
                default_key = (None, ts)
                if exact_key in actual_map or default_key in actual_map:
                    matched_actuals.append(
                        actual_map[exact_key]
                        if exact_key in actual_map
                        else actual_map[default_key]
                    )
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
            )
            results.append(result)
            try:
                self._resolve_event_outcomes(record, matched_actuals, result)
            except Exception:
                # An unresolvable ledger row must not block forecast scoring;
                # the outcome stays open and a later submit can retry it.
                logger.warning("Event-outcome resolution failed for %s",
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

        actuals: list[tuple[str, float] | tuple[str, str, float]] = []
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            ts_col, val_col, series_col = self._resolve_actuals_columns(
                cols, time_column, target_column, series_column,
            )
            for row in reader:
                ts = row[ts_col]
                try:
                    val = float(row[val_col])
                except (ValueError, TypeError):
                    continue
                if series_col:
                    actuals.append((row[series_col], ts, val))
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
        remaining = [name for name in columns if name != series_col]

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
    ) -> list[dict[str, Any]]:
        """Get per-forecast performance history for a specific model."""
        task_filter = "" if task is None else " AND COALESCE(task, 'forecast') = ?"
        params: tuple[Any, ...] = (project, model) if task is None else (project, model, task)
        with self._connect() as conn:
            rows = conn.execute(f"""
                SELECT * FROM model_performance
                WHERE project = ? AND model = ?{task_filter}
                ORDER BY scored_at DESC
            """, params).fetchall()
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
    from .temporal import SEASONS

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
        season = SEASONS[schema.frequency]
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
        if context_events:
            store.record_event_proposals(
                project, tracking_id, result.series,
                _proposal_rows(result.series, result, artifact.evidence,
                               context_events),
            )
    return registered

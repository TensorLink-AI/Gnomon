"""Persistent forecast tracking, realised scoring, and model performance.

A lightweight SQLite-based registry that turns Aion from a one-shot
forecasting tool into a system that learns from its own accuracy.

Three core capabilities:

1. **Registry**: every forecast is indexed with its project, model,
   support, threshold, and artifact path. No daemon — just a file.

2. **Scoring**: submit actual values and Aion computes MASE, MAPE,
   bias, interval coverage, and threshold accuracy for each forecast.

3. **Model performance**: aggregate scores across forecasts to see
   which models win on which projects over time, with drift detection.

Storage: ``~/.local/share/aion/registry.db`` (or ``AION_REGISTRY_PATH``).
Zero external dependencies — Python's ``sqlite3`` is stdlib.

Usage::

    from aion.tracking import TrackingStore

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
import json
import logging
import math
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_registry_path = os.environ.get("AION_REGISTRY_PATH")
DEFAULT_REGISTRY_PATH = (
    Path(_registry_path).expanduser()
    if _registry_path
    else Path.home() / ".local" / "share" / "aion" / "registry.db"
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
    mape: float | None = None
    bias: float | None = None
    coverage: float | None = None
    threshold_accuracy: float | None = None
    scored_at: str | None = None
    drift_flag: str | None = None


@dataclass(frozen=True)
class ModelPerformance:
    model: str
    count: int
    avg_mase: float | None
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
    mape: float | None
    bias: float | None
    coverage: float | None
    threshold_accuracy: float | None
    scored_at: str
    drift_flag: str | None


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def mean_absolute_error(actual: list[float], predicted: list[float]) -> float:
    """Mean Absolute Error."""
    n = min(len(actual), len(predicted))
    if n == 0:
        return float("inf")
    return sum(abs(a - p) for a, p in zip(actual[:n], predicted[:n])) / n


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
    """Mean Absolute Percentage Error."""
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

class TrackingStore:
    """SQLite-backed forecast registry and scoring store."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_REGISTRY_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    forecast_id TEXT PRIMARY KEY,
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
                    mape REAL,
                    bias REAL,
                    coverage REAL,
                    threshold_accuracy REAL,
                    scored_at TEXT,
                    drift_flag TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_forecasts_project
                    ON forecasts(project);

                CREATE INDEX IF NOT EXISTS idx_forecasts_model
                    ON forecasts(project, selected_model);

                CREATE TABLE IF NOT EXISTS model_performance (
                    project TEXT NOT NULL,
                    model TEXT NOT NULL,
                    forecast_id TEXT NOT NULL,
                    mase REAL,
                    mape REAL,
                    bias REAL,
                    coverage REAL,
                    threshold_accuracy REAL,
                    scored_at TEXT NOT NULL,
                    PRIMARY KEY (project, model, forecast_id)
                );

                CREATE INDEX IF NOT EXISTS idx_perf_model
                    ON model_performance(project, model);
            """)
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(forecasts)")
            }
            if "naive_error" not in columns:
                conn.execute("ALTER TABLE forecasts ADD COLUMN naive_error REAL")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        return conn

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
    ) -> None:
        """Register or update a forecast in the registry."""
        created_at = created_at or datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO forecasts
                    (forecast_id, project, series, cutoff_time, horizon, frequency,
                     selected_model, support, threshold, threshold_peak_probability, naive_error,
                     artifact_path, created_at, scored)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(forecast_id) DO UPDATE SET
                    project = excluded.project,
                    series = excluded.series,
                    cutoff_time = excluded.cutoff_time,
                    horizon = excluded.horizon,
                    frequency = excluded.frequency,
                    selected_model = excluded.selected_model,
                    support = excluded.support,
                    threshold = excluded.threshold,
                    threshold_peak_probability = excluded.threshold_peak_probability,
                    naive_error = excluded.naive_error,
                    artifact_path = excluded.artifact_path
            """, (
                forecast_id, project, series, cutoff_time, horizon, frequency,
                selected_model, support, threshold, threshold_peak_probability, naive_error,
                artifact_path, created_at,
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

    def get_forecast(self, forecast_id: str) -> ForecastRecord | None:
        """Get a single forecast by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM forecasts WHERE forecast_id = ?", (forecast_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

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
    ) -> ScoreResult:
        """Score a single forecast against actuals.

        Computes MASE, MAPE, bias, coverage, and threshold accuracy.
        Stores the result and updates model_performance.
        """
        record = self.get_forecast(forecast_id)
        if record is None:
            raise ValueError(f"Forecast {forecast_id} not found in registry")

        scale = naive_error if naive_error is not None else record.naive_error
        mase = mase_score(actuals, points, scale)
        mape = mape_score(actuals, points)
        bias = bias_score(actuals, points)
        cov = interval_coverage(actuals, q10 or [], q90 or []) if q10 and q90 else None
        thresh_acc = threshold_accuracy(actuals, threshold, predicted_above) if threshold else None
        scored_at = datetime.now(timezone.utc).isoformat()

        # Drift detection: compare to model's historical average
        drift = self._check_drift(record.project, record.selected_model, mase)

        with self._connect() as conn:
            conn.execute("""
                UPDATE forecasts SET
                    scored = 1, mase = ?, mape = ?, bias = ?,
                    coverage = ?, threshold_accuracy = ?, scored_at = ?, drift_flag = ?
                WHERE forecast_id = ?
            """, (mase, mape, bias, cov, thresh_acc, scored_at, drift, forecast_id))

            if record.selected_model:
                conn.execute("""
                    INSERT OR REPLACE INTO model_performance
                        (project, model, forecast_id, mase, mape, bias,
                         coverage, threshold_accuracy, scored_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.project, record.selected_model, forecast_id,
                    mase, mape, bias, cov, thresh_acc, scored_at,
                ))

        logger.info(
            "Scored forecast %s: MASE=%s MAPE=%.1f%% bias=%.2f coverage=%s drift=%s",
            forecast_id, f"{mase:.3f}" if mase is not None else "N/A", mape, bias,
            f"{cov:.1%}" if cov is not None else "N/A",
            drift or "none",
        )
        return ScoreResult(forecast_id, mase, mape, bias, cov, thresh_acc, scored_at, drift)

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
            )
            results.append(result)

        return results

    def submit_actuals_csv(self, project: str, csv_path: str) -> list[ScoreResult]:
        """Submit actuals from a CSV file and score all unscored forecasts."""
        path = Path(csv_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Actuals file not found: {path}")

        actuals: list[tuple[str, float] | tuple[str, str, float]] = []
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            # Find timestamp and value columns
            ts_col = cols[0] if cols else "timestamp"
            val_col = cols[1] if len(cols) > 1 else "value"
            series_col = "series" if "series" in cols else None
            if series_col:
                ts_col = "timestamp" if "timestamp" in cols else next(
                    col for col in cols if col != series_col
                )
                val_col = "value" if "value" in cols else next(
                    col for col in cols if col not in {series_col, ts_col}
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

    # ---- Performance / Leaderboard ----

    def leaderboard(self, project: str) -> list[ModelPerformance]:
        """Get ranked model performance for a project."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT
                    model,
                    COUNT(*) as count,
                    AVG(mase) as avg_mase,
                    AVG(mape) as avg_mape,
                    AVG(bias) as avg_bias,
                    AVG(coverage) as avg_coverage,
                    AVG(threshold_accuracy) as avg_thresh,
                    MAX(scored_at) as last_scored
                FROM model_performance
                WHERE project = ?
                GROUP BY model
                ORDER BY AVG(mase) IS NULL, AVG(mase) ASC
            """, (project,)).fetchall()

        results = []
        for r in rows:
            # Get last MASE for this model
            last_row = conn.execute("""
                SELECT mase FROM model_performance
                WHERE project = ? AND model = ?
                ORDER BY scored_at DESC LIMIT 1
            """, (project, r["model"])).fetchone()
            last_mase = last_row["mase"] if last_row else None

            results.append(ModelPerformance(
                model=r["model"],
                count=r["count"],
                avg_mase=r["avg_mase"],
                avg_mape=r["avg_mape"],
                avg_bias=r["avg_bias"],
                avg_coverage=r["avg_coverage"],
                avg_threshold_accuracy=r["avg_thresh"],
                last_mase=last_mase,
                last_scored=r["last_scored"],
            ))

        return results

    def model_performance(self, project: str, model: str) -> list[dict[str, Any]]:
        """Get per-forecast performance history for a specific model."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM model_performance
                WHERE project = ? AND model = ?
                ORDER BY scored_at DESC
            """, (project, model)).fetchall()
        return [dict(r) for r in rows]

    # ---- Drift Detection ----

    def _check_drift(
        self, project: str, model: str | None, new_mase: float | None,
    ) -> str | None:
        """Check if the new score indicates model drift.

        Compares the new MASE to the model's historical average.
        Returns a drift flag string, or None if no drift.
        """
        if model is None or new_mase is None:
            return None

        with self._connect() as conn:
            rows = conn.execute("""
                SELECT mase FROM model_performance
                WHERE project = ? AND model = ? AND mase IS NOT NULL
                ORDER BY scored_at ASC
            """, (project, model)).fetchall()

        if len(rows) < 2:
            return None  # Not enough history

        historical = [r["mase"] for r in rows if r["mase"] is not None]
        if not historical:
            return None

        avg = sum(historical) / len(historical)
        if avg <= 0:
            return None

        degradation = (new_mase - avg) / avg

        if degradation > 0.50:
            return f"degraded: MASE {new_mase:.3f} vs historical {avg:.3f} (+{degradation:.0%})"
        elif degradation > 0.25:
            return f"warning: MASE {new_mase:.3f} vs historical {avg:.3f} (+{degradation:.0%})"
        return None

    # ---- Compare ----

    def compare(
        self, forecast_id_a: str, forecast_id_b: str,
    ) -> dict[str, Any]:
        """Compare two scored forecasts."""
        a = self.get_forecast(forecast_id_a)
        b = self.get_forecast(forecast_id_b)
        if a is None or b is None:
            raise ValueError("One or both forecasts not found")

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
            "winner": (
                "a"
                if (a.mase if a.mase is not None else float("inf"))
                < (b.mase if b.mase is not None else float("inf"))
                else "b"
            ),
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
            mape=row["mape"],
            bias=row["bias"],
            coverage=row["coverage"],
            threshold_accuracy=row["threshold_accuracy"],
            scored_at=row["scored_at"],
            drift_flag=row["drift_flag"],
        )

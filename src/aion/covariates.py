"""Point-in-time covariate loading, validation, and fold-safe ablation."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from .contracts import AionError
from .data import fingerprint
from .evaluation import Evaluation, error_score, quantile
from .models import MODELS, predict
from .temporal import SEASONS, next_timestamp, validate_and_group


@dataclass(frozen=True)
class CovariateSpec:
    name: str
    value_type: str
    availability: str


@dataclass(frozen=True)
class CovariateRow:
    valid_at: datetime
    known_at: datetime
    series: str
    values: dict[str, float]


@dataclass
class CovariateDataset:
    path: str
    fingerprint: str
    specs: tuple[CovariateSpec, ...]
    rows: list[CovariateRow]

    def value_at(self, name: str, series: str, valid_at: datetime, cutoff: datetime) -> float | None:
        candidates = [
            row for row in self.rows
            if row.series == series
            and row.valid_at == valid_at and row.known_at <= cutoff and name in row.values
        ]
        if not candidates and series != "__default__":
            candidates = [
                row for row in self.rows
                if row.series == "__default__"
                and row.valid_at == valid_at and row.known_at <= cutoff and name in row.values
            ]
        if not candidates:
            return None
        return max(candidates, key=lambda row: row.known_at).values[name]


@dataclass
class CovariateAssessment:
    considered: bool
    admitted: bool
    retained: list[str] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    fold_improvements: dict[str, list[float]] = field(default_factory=dict)
    points: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    coverage: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "considered": self.considered, "admitted": self.admitted,
            "model": "covariate_linear" if self.admitted else None,
            "retained": self.retained, "rejected": self.rejected,
            "fold_improvements": self.fold_improvements,
            "measured_test_coverage": self.coverage,
        }


def parse_mapping(raw: str) -> tuple[CovariateSpec, ...]:
    """Parse name:type:availability entries.

    Availability is mandatory so contemporaneous observations cannot be
    accidentally treated as values known at a historical forecast origin.
    """
    specs: list[CovariateSpec] = []
    for item in raw.split(","):
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 3:
            raise AionError(
                "INVALID_COVARIATE_MAPPING",
                "Each mapping must be name:type:availability; availability must be future_known.",
            )
        name, value_type, availability = parts
        if value_type not in {"continuous", "binary"}:
            raise AionError(
                "INVALID_COVARIATE_TYPE",
                f"Covariate {name!r} has unsupported type {value_type!r}; use continuous or binary.",
            )
        if availability != "future_known":
            raise AionError(
                "UNSUPPORTED_COVARIATE_AVAILABILITY",
                f"Covariate {name!r} is {availability!r}; this release only admits future_known values.",
            )
        specs.append(CovariateSpec(name, value_type, availability))
    if not specs or any(not spec.name for spec in specs) or len({spec.name for spec in specs}) != len(specs):
        raise AionError("INVALID_COVARIATE_MAPPING", "Covariate names must be non-empty and unique.")
    return tuple(specs)


def _timestamp(raw: Any, *, column: str, row: int) -> datetime:
    text = str(raw or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise AionError(
            "INVALID_COVARIATE_TIMESTAMP",
            f"Cannot parse {column} on covariate row {row}: {raw!r}",
        ) from exc


def load_covariates(
    path_raw: str, mapping: str, *, time_column: str = "timestamp",
    known_at_column: str = "known_at", series_column: str | None = None,
) -> CovariateDataset:
    path = Path(path_raw).expanduser().resolve()
    if not path.is_file():
        raise AionError("COVARIATES_NOT_FOUND", f"Covariates file does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise AionError("UNSUPPORTED_COVARIATES", "Covariates currently require CSV input.")
    specs = parse_mapping(mapping)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        required = [time_column, known_at_column, *[spec.name for spec in specs]]
        if series_column:
            required.append(series_column)
        missing = [name for name in required if name not in columns]
        if missing:
            raise AionError(
                "MISSING_COVARIATE_COLUMNS",
                f"Covariate columns are missing: {', '.join(missing)}",
                {"required": required, "available": columns},
            )
        rows: list[CovariateRow] = []
        row_keys: set[tuple[str, datetime, datetime]] = set()
        for number, raw in enumerate(reader, 2):
            valid_at = _timestamp(raw.get(time_column), column=time_column, row=number)
            known_at = _timestamp(raw.get(known_at_column), column=known_at_column, row=number)
            if (valid_at.tzinfo is None) != (known_at.tzinfo is None):
                raise AionError("MIXED_COVARIATE_TIMEZONES", "valid_at and known_at must use compatible timezones.")
            values: dict[str, float] = {}
            for spec in specs:
                try:
                    value = float(raw[spec.name])
                except (TypeError, ValueError) as exc:
                    raise AionError(
                        "INVALID_COVARIATE_VALUE",
                        f"Covariate {spec.name!r} is not numeric on row {number}.",
                    ) from exc
                if not math.isfinite(value):
                    raise AionError(
                        "INVALID_COVARIATE_VALUE",
                        f"Covariate {spec.name!r} must be finite on row {number}.",
                    )
                if spec.value_type == "binary" and value not in (0.0, 1.0):
                    raise AionError("INVALID_BINARY_COVARIATE", f"{spec.name!r} must contain only 0 or 1.")
                values[spec.name] = value
            series = str(raw[series_column]) if series_column else "__default__"
            key = (series, valid_at, known_at)
            if key in row_keys:
                raise AionError(
                    "DUPLICATE_COVARIATE_VINTAGE",
                    f"Duplicate series/valid_at/known_at key on row {number}.",
                )
            row_keys.add(key)
            rows.append(CovariateRow(valid_at, known_at, series, values))
    if not rows:
        raise AionError("EMPTY_COVARIATES", "The covariates file contains no rows.")
    return CovariateDataset(str(path), fingerprint(path), specs, rows)


def covariate_guide(
    input_path: str, *, time_column: str, target_column: str, horizon: int,
    series_column: str | None = None, frequency: str | None = None,
) -> dict[str, Any]:
    """Return agent-facing format and fold constraints without suggesting features."""
    from .data import load_observations

    observations, _, _ = load_observations(
        input_path, time_column, target_column, series_column
    )
    groups, resolved_frequency, zone = validate_and_group(observations, frequency)
    season = SEASONS[resolved_frequency]
    series_payload: list[dict[str, Any]] = []
    for series, items in sorted(groups.items()):
        timestamps = [item.timestamp for item in items]
        future: list[datetime] = []
        cursor = timestamps[-1]
        for _ in range(horizon):
            cursor = next_timestamp(cursor, resolved_frequency)
            future.append(cursor)
        minimum_train = max(2 * season, 2 * horizon, 8)
        origins = list(range(minimum_train, len(timestamps) - horizon + 1, horizon))
        series_payload.append({
            "series": series,
            "history_start": timestamps[0].isoformat(),
            "history_end": timestamps[-1].isoformat(),
            "forecast_timestamps": [value.isoformat() for value in future],
            "selection_fold_cutoffs": [timestamps[origin - 1].isoformat() for origin in origins[:-2]],
            "ablation_ready": len(origins) >= 4,
            "rolling_origins": len(origins),
        })
    return {
        "schema_version": "0.1", "status": "ok",
        "frequency": resolved_frequency, "timezone": zone,
        "format": {
            "required_columns": ["timestamp", "known_at", "<covariate columns>"],
            "mapping": "name:type:future_known",
            "types": ["continuous", "binary"],
            "critical_rule": (
                "timestamp is when a value applies; known_at is when that exact vintage "
                "became available. A fold can only use rows with known_at <= its cutoff."
            ),
            "historical_vintages": (
                "Forecast-derived inputs require historical issued forecasts. Values "
                "reconstructed with later observations are rejected."
            ),
        },
        "series": series_payload,
    }


def validate_covariate_file(
    input_path: str, covariates_path: str, mapping: str, *,
    time_column: str, target_column: str, horizon: int,
    series_column: str | None = None, frequency: str | None = None,
    covariate_time_column: str = "timestamp",
    covariate_known_at_column: str = "known_at",
    covariate_series_column: str | None = None,
) -> dict[str, Any]:
    from .data import load_observations

    guide = covariate_guide(
        input_path, time_column=time_column, target_column=target_column,
        series_column=series_column, frequency=frequency, horizon=horizon,
    )
    observations, _, _ = load_observations(
        input_path, time_column, target_column, series_column
    )
    groups, resolved_frequency, _ = validate_and_group(observations, frequency)
    dataset = load_covariates(
        covariates_path, mapping, time_column=covariate_time_column,
        known_at_column=covariate_known_at_column,
        series_column=covariate_series_column,
    )
    guide_by_series = {item["series"]: item for item in guide["series"]}
    validations: list[dict[str, Any]] = []
    for series, items in sorted(groups.items()):
        timestamps = [item.timestamp for item in items]
        future = [datetime.fromisoformat(value) for value in guide_by_series[series]["forecast_timestamps"]]
        cutoffs = [datetime.fromisoformat(value) for value in guide_by_series[series]["selection_fold_cutoffs"]]
        validation = validate_covariates(
            dataset, series=series, timestamps=timestamps,
            future_timestamps=future, fold_cutoffs=cutoffs,
        )
        if not guide_by_series[series]["ablation_ready"]:
            validation["valid"] = False
            validation["issues"].append({
                "covariate": "*", "code": "INSUFFICIENT_BACKTEST_HISTORY",
                "message": "At least four rolling origins are required for covariate admission.",
            })
        validations.append({
            "series": series,
            **validation,
        })
    return {
        **guide, "covariates_path": dataset.path,
        "covariates_fingerprint": dataset.fingerprint,
        "validation": validations,
        "valid": all(item["valid"] for item in validations),
        "frequency": resolved_frequency,
    }


def validate_covariates(
    dataset: CovariateDataset, *, series: str, timestamps: list[datetime],
    future_timestamps: list[datetime], fold_cutoffs: list[datetime],
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    coverage: dict[str, dict[str, int]] = {}
    for spec in dataset.specs:
        missing_backtest = 0
        for cutoff in fold_cutoffs:
            future_at_fold = [value for value in timestamps if value > cutoff][:len(future_timestamps)]
            for valid_at in future_at_fold:
                if dataset.value_at(spec.name, series, valid_at, cutoff) is None:
                    missing_backtest += 1
        missing_future = sum(
            dataset.value_at(spec.name, series, valid_at, timestamps[-1]) is None
            for valid_at in future_timestamps
        )
        coverage[spec.name] = {
            "missing_backtest_future_points": missing_backtest,
            "missing_final_future_points": missing_future,
        }
        if missing_backtest:
            issues.append({"covariate": spec.name, "code": "MISSING_HISTORICAL_VINTAGES",
                           "message": f"{missing_backtest} fold-time values were unavailable."})
        if missing_future:
            issues.append({"covariate": spec.name, "code": "MISSING_FORECAST_VALUES",
                           "message": f"{missing_future} final-horizon values were unavailable."})
    return {"valid": not issues, "coverage": coverage, "issues": issues}


def _solve(matrix: list[list[float]], target: list[float], ridge: float = 1e-6) -> list[float]:
    width = len(matrix[0])
    augmented: list[list[float]] = []
    for i in range(width):
        row = [sum(values[i] * values[j] for values in matrix) for j in range(width)]
        row[i] += ridge
        row.append(sum(values[i] * y for values, y in zip(matrix, target)))
        augmented.append(row)
    for col in range(width):
        pivot = max(range(col, width), key=lambda idx: abs(augmented[idx][col]))
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        divisor = augmented[col][col]
        if abs(divisor) < 1e-12:
            continue
        augmented[col] = [value / divisor for value in augmented[col]]
        for row_idx in range(width):
            if row_idx == col:
                continue
            factor = augmented[row_idx][col]
            augmented[row_idx] = [
                left - factor * right
                for left, right in zip(augmented[row_idx], augmented[col])
            ]
    return [augmented[idx][-1] for idx in range(width)]


def _covariate_forecast(
    values: list[float], timestamps: list[datetime], future: list[datetime],
    dataset: CovariateDataset, names: list[str], series: str, cutoff: datetime,
    season: int,
) -> list[float] | None:
    start = max(1, season)
    matrix: list[list[float]] = []
    target: list[float] = []
    scale = max(len(values), 1)
    for idx in range(start, len(values)):
        covs = [dataset.value_at(name, series, timestamps[idx], cutoff) for name in names]
        if any(value is None for value in covs):
            continue
        matrix.append([1.0, idx / scale, values[idx - 1], values[idx - season], *covs])
        target.append(values[idx])
    if not matrix or len(matrix) < max(6, len(matrix[0]) + 1):
        return None
    coefficients = _solve(matrix, target)
    history = list(values)
    result: list[float] = []
    for offset, valid_at in enumerate(future):
        covs = [dataset.value_at(name, series, valid_at, cutoff) for name in names]
        if any(value is None for value in covs):
            return None
        row = [1.0, (len(values) + offset) / scale, history[-1], history[-season], *covs]
        point = sum(coef * value for coef, value in zip(coefficients, row))
        result.append(point)
        history.append(point)
    return result


def assess_covariates(
    values: list[float], timestamps: list[datetime], future_timestamps: list[datetime],
    dataset: CovariateDataset, series: str, horizon: int, season: int,
    minimum_improvement: float, base: Evaluation,
) -> CovariateAssessment:
    if not base.supported or not base.selected_model:
        return CovariateAssessment(False, False, rejected=[{"reason": "base evaluation is unsupported"}])
    minimum_train = max(2 * season, 2 * horizon, 8)
    origins = list(range(minimum_train, len(values) - horizon + 1, horizon))
    if len(origins) < 4:
        return CovariateAssessment(True, False, rejected=[{
            "reason": "at least four rolling origins are required for leakage-safe covariate selection"
        }])
    selection, calibration_origin, test_origin = origins[:-2], origins[-2], origins[-1]
    retained: list[str] = []
    assessment = CovariateAssessment(True, False)

    def scores(names: list[str]) -> list[float] | None:
        fold_scores: list[float] = []
        for origin in selection:
            forecast = _covariate_forecast(
                values[:origin], timestamps[:origin], timestamps[origin:origin + horizon],
                dataset, names, series, timestamps[origin - 1], season,
            )
            if forecast is None:
                return None
            fold_scores.append(error_score(values[origin:origin + horizon], forecast))
        return fold_scores

    current_scores = [
        error_score(values[origin:origin + horizon],
                    predict(base.selected_model, values[:origin], horizon, season))
        for origin in selection
    ] if base.selected_model in MODELS else None
    if current_scores is None:
        return CovariateAssessment(True, False, rejected=[{
            "reason": "covariate admission currently requires a built-in univariate comparison model"
        }])

    for spec in dataset.specs:
        candidate_names = [*retained, spec.name]
        candidate_scores = scores(candidate_names)
        if candidate_scores is None:
            assessment.rejected.append({"covariate": spec.name, "reason": "missing point-in-time values on one or more folds"})
            continue
        improvements = [
            0.0 if max(old, new) <= 1e-12 else (old - new) / max(old, new)
            for old, new in zip(current_scores, candidate_scores)
        ]
        assessment.fold_improvements[spec.name] = improvements
        improved = sum(value > 0 for value in improvements)
        robust = len(improvements) == 1 or mean(sorted(improvements)[:-1]) > 0
        if mean(improvements) >= minimum_improvement and improved * 2 > len(improvements) and robust:
            retained.append(spec.name)
            current_scores = candidate_scores
        else:
            assessment.rejected.append({
                "covariate": spec.name, "reason": "did not demonstrate stable material lift",
                "mean_improvement": mean(improvements), "improved_folds": improved,
            })
    assessment.retained = retained
    if not retained:
        return assessment

    calibration = _covariate_forecast(
        values[:calibration_origin], timestamps[:calibration_origin],
        timestamps[calibration_origin:calibration_origin + horizon], dataset, retained,
        series, timestamps[calibration_origin - 1], season,
    )
    final = _covariate_forecast(
        values, timestamps, future_timestamps, dataset, retained, series,
        timestamps[-1], season,
    )
    if calibration is None or final is None:
        assessment.rejected.append({"reason": "retained set lacks calibration or final-horizon values"})
        assessment.retained = []
        return assessment
    assessment.residuals = [
        actual - predicted
        for actual, predicted in zip(values[calibration_origin:calibration_origin + horizon], calibration)
    ]
    test = _covariate_forecast(
        values[:test_origin], timestamps[:test_origin], timestamps[test_origin:test_origin + horizon],
        dataset, retained, series, timestamps[test_origin - 1], season,
    )
    if test is not None:
        low, high = quantile(assessment.residuals, 0.1), quantile(assessment.residuals, 0.9)
        assessment.coverage = mean(
            predicted + low <= actual <= predicted + high
            for actual, predicted in zip(values[test_origin:test_origin + horizon], test)
        )
        if assessment.coverage < 0.7:
            assessment.warnings.append(
                f"Final-test 80% interval coverage was {assessment.coverage:.1%}, below 70%."
            )
    assessment.points = final
    assessment.admitted = True
    return assessment

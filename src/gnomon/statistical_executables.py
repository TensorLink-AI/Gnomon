"""Zero-dependency fitted executables for common temporal statistics.

The implementations favor explicit, auditable contracts over pretending to
be a full statistics package. Each result identifies its exact method and
assumptions; callers requesting a different method are refused upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(vector)
    augmented = [list(row) + [float(value)]
                 for row, value in zip(matrix, vector)]
    for column in range(n):
        pivot = max(range(column, n),
                    key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("design matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * source
                for value, source in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(n)]


def _inverse(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    columns = []
    for index in range(n):
        unit = [1.0 if row == index else 0.0 for row in range(n)]
        columns.append(_solve(matrix, unit))
    return [[columns[column][row] for column in range(n)] for row in range(n)]


def _ols(rows: list[list[float]], target: list[float], *, ridge: float = 0.0
         ) -> tuple[list[float], list[list[float]], list[float]]:
    width = len(rows[0])
    xtx = [[sum(row[i] * row[j] for row in rows)
            + (ridge if i == j and i > 0 else 0.0)
            for j in range(width)] for i in range(width)]
    xty = [sum(row[i] * value for row, value in zip(rows, target))
           for i in range(width)]
    coefficients = _solve(xtx, xty)
    predictions = [sum(value * coefficient
                       for value, coefficient in zip(row, coefficients))
                   for row in rows]
    return coefficients, _inverse(xtx), predictions


def _difference(values: list[float], period: int = 1) -> list[float]:
    return [values[index] - values[index - period]
            for index in range(period, len(values))]


@dataclass(frozen=True)
class FittedStationarityExecutable:
    target: str
    method: str
    values_used: int
    statistic: float
    critical_values: dict[str, float]
    conclusion: str
    diagnostics: dict[str, Any]

    def execute(self) -> dict[str, Any]:
        return {
            "direction": self.conclusion,
            "estimate": {"statistic": self.statistic,
                         "critical_values": self.critical_values},
            "interval": None,
            "support": "supported",
            "automation_eligible": True,
            "executable": {"kind": "fitted_stationarity_test",
                           "method": self.method, "version": "0.1"},
            "diagnostics": self.diagnostics,
        }


def fit_stationarity_executable(
    values: list[float], *, target: str, method: str = "adf",
    differencing: int = 0, seasonal_period: int | None = None,
) -> FittedStationarityExecutable:
    """Fit an ADF(0, constant) or KPSS(level) executable.

    The precise variants are part of the identity. We never label these as a
    lag-selected ADF or trend-KPSS test.
    """
    series = [float(value) for value in values if math.isfinite(float(value))]
    for _ in range(max(0, int(differencing))):
        series = _difference(series)
    if seasonal_period:
        series = _difference(series, int(seasonal_period))
    if len(series) < 12:
        raise ValueError("stationarity testing requires at least 12 finite values")
    normalized = method.strip().lower()
    if normalized == "adf":
        changes = _difference(series)
        rows = [[1.0, series[index - 1]] for index in range(1, len(series))]
        coefficients, inverse, predictions = _ols(rows, changes)
        residuals = [actual - predicted
                     for actual, predicted in zip(changes, predictions)]
        dof = max(1, len(rows) - len(coefficients))
        variance = sum(value * value for value in residuals) / dof
        standard_error = math.sqrt(max(0.0, variance * inverse[1][1]))
        statistic = (coefficients[1] / standard_error if standard_error else
                     -1e12 if coefficients[1] < 0 else 1e12)
        critical = {"1%": -3.43, "5%": -2.86, "10%": -2.57}
        conclusion = "stationary" if statistic < critical["5%"] else "unit_root_not_rejected"
        diagnostics = {
            "variant": "ADF(0) with constant", "lags": 0,
            "difference_coefficient": coefficients[1],
            "standard_error": standard_error,
            "p_value": None,
            "p_value_note": "MacKinnon p-value is not approximated; compare the statistic with reported critical values.",
            "null_hypothesis": "unit root",
            "differencing": differencing,
            "seasonal_period": seasonal_period,
        }
    elif normalized == "kpss":
        centre = statistics.mean(series)
        residuals = [value - centre for value in series]
        cumulative, running = [], 0.0
        for value in residuals:
            running += value
            cumulative.append(running)
        bandwidth = max(1, int(4 * (len(series) / 100) ** 0.25))
        gamma0 = sum(value * value for value in residuals) / len(series)
        long_run = gamma0
        for lag in range(1, bandwidth + 1):
            covariance = sum(residuals[index] * residuals[index - lag]
                             for index in range(lag, len(series))) / len(series)
            long_run += 2 * (1 - lag / (bandwidth + 1)) * covariance
        statistic = (sum(value * value for value in cumulative)
                     / (len(series) ** 2 * max(long_run, 1e-12)))
        critical = {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739}
        conclusion = "nonstationary" if statistic > critical["5%"] else "stationarity_not_rejected"
        diagnostics = {
            "variant": "KPSS level-stationarity with constant",
            "bandwidth": bandwidth, "long_run_variance": long_run,
            "p_value": None,
            "p_value_note": "A table-interpolated p-value is not claimed; compare the statistic with reported critical values.",
            "null_hypothesis": "level stationarity",
            "differencing": differencing,
            "seasonal_period": seasonal_period,
        }
    else:
        raise ValueError("method must be 'adf' or 'kpss'")
    return FittedStationarityExecutable(
        target, normalized, len(series), statistic, critical, conclusion,
        diagnostics)


@dataclass(frozen=True)
class FittedDecompositionExecutable:
    target: str
    period: int
    trend: tuple[float | None, ...]
    seasonal: tuple[float, ...]
    residual: tuple[float | None, ...]
    strength: float

    def execute(self) -> dict[str, Any]:
        finite_residuals = [value for value in self.residual if value is not None]
        return {
            "direction": "seasonal" if self.strength >= 0.3 else "weak_seasonality",
            "estimate": {"period": self.period,
                         "seasonal_strength": self.strength,
                         "trend": list(self.trend),
                         "seasonal": list(self.seasonal),
                         "residual": list(self.residual),
                         "residual_scale": (statistics.pstdev(finite_residuals)
                                            if len(finite_residuals) > 1 else 0.0)},
            "interval": None, "support": "supported",
            "automation_eligible": True,
            "executable": {"kind": "fitted_decomposition",
                           "method": "centered_moving_average_additive",
                           "period": self.period, "version": "0.1"},
        }


def fit_decomposition_executable(
    values: list[float], *, target: str, period: int,
) -> FittedDecompositionExecutable:
    series = [float(value) for value in values]
    period = int(period)
    if period < 2:
        raise ValueError("decomposition period must be at least 2")
    if len(series) < 2 * period:
        raise ValueError("decomposition requires at least two complete periods")
    left = (period - 1) // 2
    right = period // 2
    trend: list[float | None] = []
    for index in range(len(series)):
        if index < left or index + right >= len(series):
            trend.append(None)
        else:
            trend.append(statistics.mean(series[index-left:index+right+1]))
    detrended = [(value - trend[index]) if trend[index] is not None else None
                 for index, value in enumerate(series)]
    seasonal_template = []
    for phase in range(period):
        phase_values = [detrended[index] for index in range(phase, len(series), period)
                        if detrended[index] is not None]
        seasonal_template.append(statistics.mean(phase_values) if phase_values else 0.0)
    template_mean = statistics.mean(seasonal_template)
    seasonal_template = [value - template_mean for value in seasonal_template]
    seasonal = [seasonal_template[index % period] for index in range(len(series))]
    residual = [
        (series[index] - trend[index] - seasonal[index])
        if trend[index] is not None else None
        for index in range(len(series))
    ]
    valid_detrended = [value for value in detrended if value is not None]
    valid_residual = [value for value in residual if value is not None]
    detrended_variance = statistics.pvariance(valid_detrended)
    residual_variance = statistics.pvariance(valid_residual)
    strength = max(0.0, min(1.0, 1 - residual_variance /
                           max(detrended_variance, 1e-12)))
    return FittedDecompositionExecutable(
        target, period, tuple(trend), tuple(seasonal), tuple(residual), strength)


@dataclass(frozen=True)
class FittedRegressionExecutable:
    target: str
    predictors: tuple[str, ...]
    coefficients: tuple[float, ...]
    coefficient_intervals: tuple[tuple[float, float], ...]
    validation: dict[str, Any]
    residual_scale: float

    def execute(self) -> dict[str, Any]:
        names = ("intercept",) + self.predictors
        contribution = (
            self.validation["skill_vs_mean_baseline"] >= .02
            and any(lower > 0 or upper < 0
                    for lower, upper in self.coefficient_intervals[1:])
        )
        return {
            "direction": ("predictive_contribution" if contribution
                          else "no_validated_contribution"),
            "estimate": {
                "coefficients": dict(zip(names, self.coefficients)),
                "coefficient_intervals_95": {
                    name: {"lower": interval[0], "upper": interval[1]}
                    for name, interval in zip(names, self.coefficient_intervals)},
                "validation": self.validation,
                "residual_scale": self.residual_scale,
            },
            "interval": None,
            "support": "supported" if contribution else "weak",
            "automation_eligible": contribution,
            "executable": {"kind": "fitted_exogenous_regression",
                           "method": "ridge_linear_expanding_window",
                           "predictors": list(self.predictors), "version": "0.1"},
        }


def fit_regression_executable(
    target_values: list[float], predictors: dict[str, list[float]], *,
    target: str, ridge: float = 1e-6, minimum_train: int | None = None,
) -> FittedRegressionExecutable:
    if not predictors:
        raise ValueError("exogenous regression requires at least one predictor")
    names = tuple(sorted(predictors))
    n = min([len(target_values), *(len(predictors[name]) for name in names)])
    width = len(names) + 1
    minimum = max(width * 3, minimum_train or max(20, int(n * 0.6)))
    if n <= minimum + 2:
        raise ValueError("insufficient aligned history for expanding-window validation")
    target_series = [float(value) for value in target_values[-n:]]
    predictor_series = {name: [float(value) for value in predictors[name][-n:]]
                        for name in names}
    predicted, actual, baseline = [], [], []
    for origin in range(minimum, n):
        rows = [[1.0, *(predictor_series[name][index] for name in names)]
                for index in range(origin)]
        coefficients, _, _ = _ols(rows, target_series[:origin], ridge=ridge)
        row = [1.0, *(predictor_series[name][origin] for name in names)]
        predicted.append(sum(value * coefficient
                             for value, coefficient in zip(row, coefficients)))
        actual.append(target_series[origin])
        baseline.append(statistics.mean(target_series[:origin]))
    mse = statistics.mean((a - p) ** 2 for a, p in zip(actual, predicted))
    baseline_mse = statistics.mean((a - p) ** 2 for a, p in zip(actual, baseline))
    full_rows = [[1.0, *(predictor_series[name][index] for name in names)]
                 for index in range(n)]
    coefficients, inverse, fitted = _ols(full_rows, target_series, ridge=ridge)
    residuals = [actual_value - fitted_value
                 for actual_value, fitted_value in zip(target_series, fitted)]
    dof = max(1, n - width)
    variance = sum(value * value for value in residuals) / dof
    intervals = tuple((coefficient - 1.96 * math.sqrt(max(0.0, variance * inverse[i][i])),
                       coefficient + 1.96 * math.sqrt(max(0.0, variance * inverse[i][i])))
                      for i, coefficient in enumerate(coefficients))
    validation = {
        "scheme": "expanding_window_one_step",
        "minimum_train": minimum, "validation_points": len(actual),
        "mse": mse, "mean_baseline_mse": baseline_mse,
        "skill_vs_mean_baseline": 1 - mse / max(baseline_mse, 1e-12),
    }
    return FittedRegressionExecutable(
        target, names, tuple(coefficients), intervals, validation,
        math.sqrt(variance))

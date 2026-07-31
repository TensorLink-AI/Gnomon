"""Dependency-free guarded VAR(1) forecasting for aligned multi-series data."""
from __future__ import annotations

from typing import Any


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        a[col], a[pivot] = a[pivot], a[col]
        if abs(a[col][col]) < 1e-9:
            a[col][col] += 1e-6
        scale = a[col][col]
        a[col] = [value / scale for value in a[col]]
        for row in range(n):
            if row != col:
                factor = a[row][col]
                a[row] = [x - factor * y for x, y in zip(a[row], a[col])]
    return [row[-1] for row in a]


def _correlation(left: list[float], right: list[float]) -> float:
    lm, rm = sum(left) / len(left), sum(right) / len(right)
    num = sum((x-lm)*(y-rm) for x, y in zip(left, right))
    den = sum((x-lm)**2 for x in left) * sum((y-rm)**2 for y in right)
    return num / den ** 0.5 if den > 1e-12 else 0.0


def correlation_report(groups: dict[str, list[Any]]) -> list[dict[str, float | str]]:
    names = sorted(groups)
    if len(names) < 2:
        return []
    timestamp_sets = [[item.timestamp for item in groups[name]] for name in names]
    if len({tuple(items) for items in timestamp_sets}) != 1:
        return []
    columns = {name: [item.value for item in groups[name]] for name in names}
    return [{"left": names[i], "right": names[j],
             "correlation": round(_correlation(columns[names[i]], columns[names[j]]), 4)}
            for i in range(len(names)) for j in range(i)]


def _predict(columns: list[list[float]], horizon: int) -> list[list[float]]:
    rows = [[1.0] + [column[t - 1] for column in columns] for t in range(1, len(columns[0]))]
    p = len(rows[0])
    gram = [[sum(row[i] * row[j] for row in rows) + (1e-6 if i == j else 0.0)
             for j in range(p)] for i in range(p)]
    coefficients = []
    for column in columns:
        target = column[1:]
        coefficients.append(_solve(
            gram, [sum(row[i] * y for row, y in zip(rows, target)) for i in range(p)]
        ))
    state = [column[-1] for column in columns]
    result = [[] for _ in columns]
    for _ in range(horizon):
        row = [1.0] + state
        state = [sum(a * b for a, b in zip(coef, row)) for coef in coefficients]
        for series, value in zip(result, state):
            series.append(value)
    return result


def forecast_var(groups: dict[str, list[Any]], horizon: int) -> tuple[dict[str, list[float]], dict[str, str]]:
    names = sorted(groups)
    timestamps = [[item.timestamp for item in groups[name]] for name in names]
    if len({tuple(items) for items in timestamps}) != 1 or len(timestamps[0]) < max(8, len(names) + 3):
        return {}, {}
    columns = [[item.value for item in groups[name]] for name in names]
    strongest = max(abs(_correlation(columns[i], columns[j])) for i in range(len(names)) for j in range(i))
    if strongest < 0.3:
        return {}, {}
    holdout = min(horizon, max(2, len(columns[0]) // 5))
    training = [column[:-holdout] for column in columns]
    if len(training[0]) < len(names) + 3:
        return {}, {}
    validation = _predict(training, holdout)
    var_error = sum(abs(actual - predicted)
                    for column, predictions in zip(columns, validation)
                    for actual, predicted in zip(column[-holdout:], predictions))
    naive_error = sum(abs(actual - column[-holdout - 1])
                      for column in columns for actual in column[-holdout:])
    if var_error >= naive_error:
        return {}, {}
    forecasts = _predict(columns, horizon)
    result = {name: points for name, points in zip(names, forecasts)}
    warning = f"Opt-in VAR(1) forecast used across {len(names)} aligned series (maximum absolute correlation {strongest:.2f}); uncertainty remains based on univariate residuals."
    return result, {name: warning for name in names}

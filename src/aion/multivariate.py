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


#: Weakest maximum absolute cross-correlation worth fitting a VAR on. This is
#: an eligibility filter, not a decision: passing it only earns the VAR a place
#: in the selection folds, where it still has to beat the univariate ladder.
MINIMUM_CORRELATION = 0.3

MULTIVARIATE_MODEL_NAME = "var"


class VarFrame:
    """Aligned multi-series values with a VAR(1) predictor at any fold origin.

    The point of the origin argument is that a candidate which reads other
    series must be *refittable* at each fold cutoff. Fitting once on the whole
    frame and validating on a trailing window — which is what this module did
    before — both leaks the report-only test fold into the decision and gives
    the VAR a comparison no other candidate gets. With this, the VAR is scored
    on the same rolling origins as every baseline, statistical model, and TSFM,
    and is admitted only by the same margin rule.
    """

    def __init__(self, names: list[str], columns: list[list[float]],
                 strongest_correlation: float) -> None:
        self.names = names
        self.columns = columns
        self.strongest_correlation = strongest_correlation
        self._cache: dict[tuple[int, int], list[list[float]]] = {}

    @classmethod
    def build(cls, groups: dict[str, list[Any]]) -> tuple["VarFrame | None", str | None]:
        """Return a frame, or ``None`` and the reason it is not eligible."""
        names = sorted(groups)
        if len(names) < 2:
            return None, "fewer than two series"
        timestamps = [[item.timestamp for item in groups[name]] for name in names]
        if len({tuple(items) for items in timestamps}) != 1:
            return None, "series are not observed on identical timestamps"
        if len(timestamps[0]) < max(8, len(names) + 3):
            return None, (f"needs at least {max(8, len(names) + 3)} aligned "
                          f"observations (have {len(timestamps[0])})")
        columns = [[item.value for item in groups[name]] for name in names]
        strongest = max(abs(_correlation(columns[i], columns[j]))
                        for i in range(len(names)) for j in range(i))
        if strongest < MINIMUM_CORRELATION:
            return None, (f"maximum absolute cross-correlation {strongest:.2f} is "
                          f"below {MINIMUM_CORRELATION}")
        return cls(names, columns, strongest), None

    def _fit_predict(self, origin: int, horizon: int) -> list[list[float]]:
        key = (origin, horizon)
        if key not in self._cache:
            training = [column[:origin] for column in self.columns]
            if len(training[0]) < len(self.names) + 3:
                raise ValueError("too little aligned history at this origin")
            self._cache[key] = _predict(training, horizon)
        return self._cache[key]

    def predictor(self, series_name: str):
        """``predictor(origin, horizon)`` for one series, for ``evaluate``."""
        index = self.names.index(series_name)

        def predict_at(origin: int, horizon: int) -> list[float]:
            return self._fit_predict(origin, horizon)[index]

        return predict_at

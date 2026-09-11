"""Shared conversion helpers so each adapter stays a short shim.

Everything here enforces the same Gnomon rules regardless of package:
timestamps are used when supplied and never invented; quantile rows are made
monotone and the fix is disclosed; covariates must be paired history/future
matrices; foundation-model training cutoffs are never guessed.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..forecast_adapter import (PROTOCOL_VERSION, AdapterCapabilities, ForecastAdapterError,
                                ForecastRequest, ForecastResult)


@dataclass(frozen=True)
class Registration:
    target: Any
    factory: bool
    capabilities: AdapterCapabilities
    revision: str
    deterministic: bool


def revision(kind: str, *distributions: str) -> str:
    parts = []
    for dist in distributions:
        try:
            parts.append(f"{dist}={importlib.metadata.version(dist)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{dist}=unknown")
    return f"{kind}/" + ",".join(parts)


def resolve(path: str, *, default_module: str | None = None, label: str = "model") -> Any:
    """Resolve ``module:attribute`` or a bare attribute of ``default_module``."""
    if not isinstance(path, str) or not path.strip():
        raise ForecastAdapterError(f"{label} must be a nonempty string")
    if ":" in path:
        module_name, attribute = path.split(":", 1)
    elif default_module is not None:
        module_name, attribute = default_module, path
    else:
        raise ForecastAdapterError(f"{label} must be module:attribute")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError:
        raise ForecastAdapterError(f"{label} {attribute!r} is not defined in {module_name}") from None


def accepts(target: Any, parameter: str) -> bool:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return False
    if parameter in signature.parameters:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())


def metadata(kind: str, **extra: Any) -> dict:
    return {"adapter_kind": kind, "protocol_version": PROTOCOL_VERSION, **extra}


# --- time index -------------------------------------------------------------

def history_index(request: ForecastRequest):
    """Return (pandas index, frequency or None, note) for the history.

    Real timestamps are used when supplied. Otherwise the index is a plain
    integer range and the note says so; adapters that need calendar time must
    call require_dates instead.
    """
    if request.timestamps:
        import pandas as pd
        index = pd.DatetimeIndex(pd.to_datetime(list(request.timestamps), utc=True))
        freq = request.frequency
        if freq is None and len(index) >= 3:
            freq = pd.infer_freq(index)
        return index, freq, "timestamps_from_request"
    return range(len(request.history)), None, "integer_index_no_timestamps_supplied"


def require_dates(kind: str, request: ForecastRequest):
    """Adapters that model calendar effects must not run on an invented index."""
    index, freq, _ = history_index(request)
    if not request.timestamps:
        raise ForecastAdapterError(
            f"{kind} needs request timestamps; supply timestamps (and frequency) or use a frozen data_ref",
            details={"adapter_kind": kind, "missing_fields": ["timestamps"]})
    if freq is None and not request.future_timestamps:
        raise ForecastAdapterError(
            f"{kind} cannot place the horizon without a frequency; supply frequency or future_timestamps",
            details={"adapter_kind": kind, "missing_fields": ["frequency", "future_timestamps"]})
    return index, freq


def future_index(request: ForecastRequest, index, freq):
    """Horizon index: requested future timestamps, else stepped from the history by freq."""
    if request.future_timestamps:
        import pandas as pd
        return pd.DatetimeIndex(pd.to_datetime(list(request.future_timestamps), utc=True))
    if isinstance(index, range):
        return range(len(request.history), len(request.history) + request.horizon)
    import pandas as pd
    if freq is None:
        raise ForecastAdapterError("frequency is required to place the horizon after dated history",
                                   details={"missing_fields": ["frequency", "future_timestamps"]})
    return pd.date_range(start=index[-1], periods=request.horizon + 1, freq=freq)[1:]


def naive_utc(index):
    """Timezone-naive UTC copy for packages that reject aware timestamps."""
    import pandas as pd
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        return index.tz_convert("UTC").tz_localize(None)
    return index


# --- covariates -------------------------------------------------------------

def paired_covariates(request: ForecastRequest):
    """Return (history matrix, future matrix, names) or (None, None, ()) when absent.

    Regression-style packages need the same regressors over history and
    horizon. Supplying only one side is rejected rather than silently dropped.
    """
    past, future = request.past_covariates, request.future_covariates
    if not past and not future:
        return None, None, ()
    if bool(past) != bool(future):
        raise ForecastAdapterError(
            "this adapter needs both past_covariates and future_covariates with the same columns",
            details={"missing_fields": ["past_covariates" if not past else "future_covariates"]})
    import numpy as np
    hist = np.asarray(past, dtype=float)
    fut = np.asarray(future, dtype=float)
    if hist.ndim != 2 or fut.ndim != 2 or hist.shape[1] != fut.shape[1]:
        raise ForecastAdapterError("past_covariates and future_covariates must have the same number of columns")
    names = tuple(request.future_covariate_names or request.past_covariate_names
                  or (f"x{i}" for i in range(hist.shape[1])))
    return hist, fut, names


# --- quantiles --------------------------------------------------------------

def levels(quantiles: Sequence[float]) -> list[float]:
    """Central prediction-interval levels (percent) covering the requested quantiles."""
    out = sorted({round(abs(2 * q - 1) * 100, 6) for q in quantiles if q != 0.5})
    return [int(v) if float(v).is_integer() else v for v in out]


def level_for(q: float) -> float:
    value = round(abs(2 * q - 1) * 100, 6)
    return int(value) if float(value).is_integer() else value


def make_rows(request: ForecastRequest, at: Callable[[float], Sequence[float]]):
    """Build quantile rows from a per-quantile lookup; returns (rows, fixed_count)."""
    if not request.quantiles:
        return None, 0
    columns = {q: [float(v) for v in at(q)] for q in request.quantiles}
    for q, values in columns.items():
        if len(values) != request.horizon or any(not math.isfinite(v) for v in values):
            raise ForecastAdapterError(f"adapter produced an invalid quantile column for q={q}")
    rows, fixed = [], 0
    for step in range(request.horizon):
        row, previous = {}, -math.inf
        for q in request.quantiles:
            value = columns[q][step]
            if value < previous:
                value, fixed = previous, fixed + 1
            row[q] = value
            previous = value
        rows.append(row)
    return tuple(rows), fixed


def rows_from_samples(request: ForecastRequest, samples):
    """Quantile rows from a (num_samples, horizon) array."""
    import numpy as np
    array = np.asarray(samples, dtype=float)
    if array.ndim != 2 or array.shape[1] != request.horizon:
        raise ForecastAdapterError("sample array must be shaped (num_samples, horizon)")
    return make_rows(request, lambda q: np.quantile(array, q, axis=0))


def take_paths(request: ForecastRequest, samples):
    """Sample paths only when the request asked for them, trimmed to that count."""
    if not request.samples:
        return None
    import numpy as np
    array = np.asarray(samples, dtype=float)
    if array.shape[0] < request.samples:
        raise ForecastAdapterError(
            f"adapter drew {array.shape[0]} sample paths but the request needs {request.samples}",
            details={"available_samples": int(array.shape[0]), "requested_samples": request.samples})
    return tuple(tuple(float(v) for v in path) for path in array[:request.samples])


def finish(request: ForecastRequest, point, *, kind: str, quantiles=None, fixed: int = 0,
           sample_paths=None, **extra: Any) -> ForecastResult:
    point = tuple(float(v) for v in point)
    meta = metadata(kind, **extra)
    if fixed:
        meta["quantile_crossings_repaired"] = fixed
    return ForecastResult(point, quantiles=quantiles, metadata=meta, timestamps=request.future_timestamps,
                          series_id=request.series_id, unit=request.unit,
                          sample_paths=sample_paths).validate(request)


def seed_numpy(seed: int | None) -> None:
    if seed is not None:
        import numpy as np
        np.random.seed(seed)


def check_seed(seed: Any) -> int | None:
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ForecastAdapterError("seed must be a nonnegative integer")
    return seed

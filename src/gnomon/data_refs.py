"""Session-local frozen inputs. References never reopen mutable sources.

The retained snapshot includes vintages, not just the latest vector, so later
evaluation can replay earlier knowledge. Capacity limits bound retained rows,
not the memory used by a file parser; deployments must also bound input size.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from statistics import fmean, median

from .contracts import GnomonError
from .datasets import LoadedDataset, load_stage
from .forecast_adapter import ForecastAdapterError, ForecastRequest
from .ids import content_id
from .repair import RepairLog
from .temporal import next_timestamp

STATISTICS = ("mean", "median", "latest", "minimum", "maximum", "sum")


def instant(value: str | None, name: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or "T" not in value:
        raise ForecastAdapterError(f"{name} must be an ISO datetime including time")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ForecastAdapterError(f"{name} must be an ISO datetime") from None


@dataclass(frozen=True)
class _FrozenInput:
    loaded: LoadedDataset
    unit: str | None
    repairs: tuple[dict, ...]
    row_count: int


class DataReferences:
    def __init__(self, *, max_refs: int = 16, max_rows: int = 100_000):
        for value in (max_refs, max_rows):
            if type(value) is not int or value < 1:
                raise ForecastAdapterError("data reference capacities must be positive integers")
        self.max_refs, self.max_rows = max_refs, max_rows
        self._inputs: OrderedDict[str, _FrozenInput] = OrderedDict()

    def inspect(self, input: str, *, time_column: str = "timestamp", target_column: str = "value",
                series_column: str | None = None, frequency: str | None = None, as_of: str | None = None,
                recorded_as_of: str | None = None, store_path: str | None = None,
                repair: str = "off", regrid: str | None = None, unit: str | None = None) -> dict:
        for name, value in (("input", input), ("time_column", time_column), ("target_column", target_column)):
            if not isinstance(value, str) or not value.strip():
                raise ForecastAdapterError(f"{name} must be a nonempty string")
        if unit is not None and (not isinstance(unit, str) or not unit.strip()):
            raise ForecastAdapterError("unit must be a nonempty string or null")
        log = RepairLog()
        loaded = load_stage(input, time_column=time_column, target_column=target_column,
                            series_column=series_column, frequency=frequency, as_of=instant(as_of, "as_of"),
                            recorded_as_of=instant(recorded_as_of, "recorded_as_of"), store_path=store_path,
                            repair=repair, repair_log=log, regrid=regrid)
        # Count every retained vintage, not only the latest materialized rows.
        count = loaded.snapshot.observation_count
        if count > self.max_rows:
            raise GnomonError("INVALID_ARGUMENTS", "Input exceeds this session's retained observation limit.")
        repairs = tuple(action.to_dict() for action in log.actions())
        ref = content_id("data", {"snapshot": loaded.snapshot.snapshot_id, "schema": asdict(loaded.schema),
                                  "unit": unit, "repairs": repairs}, length=64)
        frozen = _FrozenInput(loaded, unit, repairs, count)
        self._inputs.pop(ref, None)
        while self._inputs and (len(self._inputs) >= self.max_refs or
                               sum(item.row_count for item in self._inputs.values()) + count > self.max_rows):
            self._inputs.popitem(last=False)
        self._inputs[ref] = frozen
        return {"schema_version": "1", "status": "ok", "data_ref": ref,
                "frequency": loaded.frequency, "timezone": loaded.timezone, "unit": unit,
                "unit_basis": "caller_declared" if unit else "unknown",
                "series": [{"series_id": name, "count": len(rows),
                            "start": rows[0].timestamp.isoformat(), "end": rows[-1].timestamp.isoformat()}
                           for name, rows in sorted(loaded.groups.items())],
                "snapshot": loaded.snapshot.access_summary(), "repairs": deepcopy(list(frozen.repairs)),
                "reference_scope": "session", "eviction": "least_recently_used"}

    def _get(self, ref: str) -> _FrozenInput:
        if not isinstance(ref, str) or ref not in self._inputs:
            raise GnomonError("INVALID_ARGUMENTS", "Unknown or expired data_ref; inspect the input again.")
        self._inputs.move_to_end(ref)
        return self._inputs[ref]

    def _select(self, ref: str, series_id: str | None):
        frozen = self._get(ref)
        groups = frozen.loaded.groups
        if series_id is None and len(groups) == 1:
            series_id = next(iter(groups))
        if not isinstance(series_id, str) or series_id not in groups:
            raise GnomonError("INVALID_ARGUMENTS", "Select one exact series_id from the inspected input.")
        return frozen, series_id, groups[series_id]

    def describe(self, data_ref: str, *, statistic: str, series_id: str | None = None,
                 start: str | None = None, end: str | None = None) -> dict:
        if statistic not in STATISTICS:
            raise ForecastAdapterError("statistic must be one of: " + ", ".join(STATISTICS))
        frozen, name, rows = self._select(data_ref, series_id)
        begin, finish = instant(start, "start"), instant(end, "end")
        for bound in (begin, finish):
            if bound is not None and (bound.tzinfo is None) != (rows[0].timestamp.tzinfo is None):
                raise ForecastAdapterError("window and data must use consistent timezone awareness")
        if begin is not None and finish is not None and begin > finish:
            raise ForecastAdapterError("start must be at or before end")
        selected = [r for r in rows if (begin is None or r.timestamp >= begin)
                    and (finish is None or r.timestamp <= finish)]
        if not selected:
            raise GnomonError("EMPTY_SNAPSHOT", "No observations fall in the requested inclusive window.")
        values = [r.value for r in selected]
        funcs = {"mean": fmean, "median": median, "latest": lambda v: v[-1],
                 "minimum": min, "maximum": max, "sum": sum}
        try:
            result = funcs[statistic](values)
        except OverflowError:
            raise ForecastAdapterError("statistic overflowed finite numeric range") from None
        import math
        if not math.isfinite(result):
            raise ForecastAdapterError("statistic overflowed finite numeric range")
        return {"schema_version": "1", "status": "ok", "data_ref": data_ref, "series_id": name,
                "statistic": statistic, "value": result, "unit": frozen.unit, "count": len(values),
                "window": {"start": selected[0].timestamp.isoformat(), "end": selected[-1].timestamp.isoformat(),
                           "requested_start": start, "requested_end": end, "closed": "both"},
                "evidence": "observed_statistic", "calibration": "not_applicable", "action_authorized": False}

    def request(self, data_ref: str, *, horizon: int, series_id: str | None = None,
                season: int = 1, quantiles: tuple[float, ...] = ()) -> ForecastRequest:
        frozen, name, rows = self._select(data_ref, series_id)
        # Validate scalar sizes before constructing a potentially huge grid.
        base = ForecastRequest.from_dict({"history": [r.value for r in rows], "horizon": horizon,
                                          "season": season, "quantiles": quantiles})
        future = []
        stamp = rows[-1].timestamp
        for _ in range(base.horizon):
            stamp = next_timestamp(stamp, frozen.loaded.frequency)
            future.append(stamp.isoformat())
        snapshot = frozen.loaded.snapshot
        if snapshot.as_of is not None and rows[-1].timestamp > snapshot.as_of:
            raise ForecastAdapterError("forecast target history contains valid times after as_of")
        return ForecastRequest.from_dict({**asdict(base),
            "timestamps": [r.timestamp.isoformat() for r in rows], "future_timestamps": future,
            "frequency": frozen.loaded.frequency, "cutoff": (snapshot.as_of or rows[-1].timestamp).isoformat(),
            "known_time_cutoff": snapshot.as_of.isoformat() if snapshot.as_of else None,
            "recorded_time_cutoff": snapshot.recorded_as_of.isoformat() if snapshot.recorded_as_of else None,
            "snapshot_id": snapshot.snapshot_id, "series_id": name, "unit": frozen.unit})

    def clear(self):
        self._inputs.clear()

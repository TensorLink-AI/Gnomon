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
from pathlib import Path
from statistics import fmean, median

from .contracts import GnomonError
from .datasets import LoadedDataset, load_stage
from .forecast_adapter import ForecastAdapterError, ForecastRequest
from .ids import content_id
from .repair import RepairLog, historical_repair_blockers
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


REPAIR_DEFAULT = "safe"
_REPAIR_LADDER = ("off", "safe", "aggressive")
# Failures that no repair level addresses: the caller must change the request, not the data policy.
_NO_HANDOFF_CODES = frozenset({"MISSING_COLUMNS", "INPUT_NOT_FOUND", "UNSUPPORTED_INPUT", "INVALID_ARGUMENTS",
                               "INPUT_TOO_LARGE", "DIAGNOSTIC_LIMIT", "EMPTY_SNAPSHOT"})
_DISCLOSURE_CODES = frozenset({"timezone_declared", "window_selected"})


def data_quality(repairs, repair):
    """First-key summary of what preparation changed. Every fix stays itemised in ``repairs``."""
    fixes = sum(int(r["count"]) for r in repairs if not r["code"].endswith("_dropped") and r["code"] not in _DISCLOSURE_CODES)
    dropped = sum(int(r["count"]) for r in repairs if r["code"].endswith("_dropped"))
    if not fixes and not dropped:
        status = "clean"
    elif repair == "aggressive" or any(r.get("assumptive") for r in repairs):
        status = "repaired_aggressive"
    else:
        status = "repaired_safe"
    return {"status": status, "fixes": fixes, "dropped": dropped, "next_call": None}


class DataReferences:
    def __init__(self, *, max_refs: int = 16, max_rows: int = 100_000):
        for value in (max_refs, max_rows):
            if type(value) is not int or value < 1:
                raise ForecastAdapterError("data reference capacities must be positive integers")
        self.max_refs, self.max_rows = max_refs, max_rows
        self._inputs: OrderedDict[str, _FrozenInput] = OrderedDict()
        self._handoff = True  # probes created by the handoff itself never diagnose recursively

    def diagnose(self, input, **options):
        """Dry-run all repair policies on a local file; never write repaired data."""
        path = Path(input)
        if input.startswith('store:') or path.suffix == '.gnomon':
            raise ForecastAdapterError('diagnose expects original local file data, not a curated store or saved snapshot')
        if not path.is_file():
            raise GnomonError('INPUT_NOT_FOUND', 'Diagnostic input file does not exist.')
        if path.stat().st_size > 8 * 1024 * 1024:
            raise GnomonError('DIAGNOSTIC_LIMIT', 'Bounded diagnostics accept files up to 8 MiB; partition the source deliberately for larger inputs.')
        if 'repair' in options:
            raise ForecastAdapterError('diagnose compares off/safe/aggressive; do not select one repair mode')
        probe = DataReferences(max_refs=3, max_rows=min(self.max_rows, 100000))
        outcomes = {}
        try:
            for mode in ('off', 'safe', 'aggressive'):
                try:
                    result = probe.inspect(input, repair=mode, **options)
                    outcomes[mode] = {'status': 'ok', 'admissible': True, 'scope': 'data_preparation_only',
                        'series': result['series'], 'repairs': result['repairs'], 'readiness': result['readiness'],
                        'source_ref': result['snapshot']['source_ref']}
                except (GnomonError, ForecastAdapterError) as exc:
                    error = exc if isinstance(exc, GnomonError) else GnomonError('INVALID_ARGUMENTS', str(exc), details=exc.details)
                    error.details.setdefault('supplied_arguments', {'input': input, **options, 'repair': mode})
                    outcomes[mode] = {'status': 'rejected', 'admissible': False, 'error': error.to_dict(compact=True)['error']}
            return {'schema_version': '1', 'status': 'ok', 'operation': 'diagnose', 'input': input,
                'modes': outcomes, 'source_modified': False, 'provider_calls': 0,
                'limits': {'file_bytes': 8 * 1024 * 1024, 'retained_rows': min(self.max_rows, 100000)},
                'guidance': 'Dry-run results describe data preparation, not provider history/capability suitability. No repaired file or reusable data_ref is returned. Lower-bound counts remain labelled; later-stage budgets may be unmeasured after an earlier rejection.'}
        finally:
            probe.clear()

    def inspect(self, input: str, *, time_column: str = "timestamp", target_column: str = "value",
                series_column: str | None = None, frequency: str | None = None, as_of: str | None = None,
                recorded_as_of: str | None = None, store_path: str | None = None,
                repair: str = REPAIR_DEFAULT, regrid: str | None = None, unit: str | None = None,
                timezone: str | None = None, purpose: str = "infer", window: str | None = None) -> dict:
        """Freeze an input. repair defaults to safe: text normalisation and bounded jitter
        alignment only, every fix listed in ``repairs``; aggressive stays opt-in."""
        if purpose not in {"infer", "evaluate", "route"}:
            raise ForecastAdapterError("purpose must be infer, evaluate or route")
        for name, value in (("input", input), ("time_column", time_column), ("target_column", target_column)):
            if not isinstance(value, str) or not value.strip():
                raise ForecastAdapterError(f"{name} must be a nonempty string")
        if unit is not None and (not isinstance(unit, str) or not unit.strip()):
            raise ForecastAdapterError("unit must be a nonempty string or null")
        if not input.startswith("store:") and Path(input).suffix == ".gnomon":
            if (time_column != "timestamp" or target_column != "value" or repair != REPAIR_DEFAULT
                    or any(value is not None for value in (series_column, frequency, as_of, recorded_as_of,
                                                          store_path, regrid, timezone, unit, window))):
                rejected = {k: v for k, v in locals().items() if k in {'series_column', 'frequency', 'as_of', 'recorded_as_of', 'store_path', 'regrid', 'timezone', 'unit', 'window'} and v is not None}
                rejected.update({k: v for k, v, default in [('time_column', time_column, 'timestamp'), ('target_column', target_column, 'value'), ('repair', repair, REPAIR_DEFAULT)] if v != default})
                from .recovery import frozen_recovery
                raise ForecastAdapterError("No preparation options are accepted with .gnomon input, even identical cutoffs. A saved snapshot already fixes its schema, unit, timezone, repairs and cutoffs; "
                                           "supply only input and purpose, or inspect the original source with new options.",
                                           details={**frozen_recovery({'input': input, 'purpose': purpose, **rejected}, rejected), 'rejected_arguments': rejected})
            from .snapshot_files import load_snapshot
            loaded, unit, repairs = load_snapshot(input, self.max_rows)
        else:
            log = RepairLog()
            try:
                loaded = load_stage(input, time_column=time_column, target_column=target_column,
                                    series_column=series_column, frequency=frequency, as_of=instant(as_of, "as_of"),
                                    recorded_as_of=instant(recorded_as_of, "recorded_as_of"), store_path=store_path,
                                    repair=repair, repair_log=log, regrid=regrid, timezone=timezone, window=window)
            except GnomonError as exc:
                if exc.code == 'MISSING_COLUMNS':
                    from .recovery import column_recovery
                    arguments = dict(input=input, time_column=time_column, target_column=target_column,
                        series_column=series_column, frequency=frequency, as_of=as_of, recorded_as_of=recorded_as_of,
                        store_path=store_path, repair=repair, regrid=regrid, timezone=timezone, window=window, unit=unit, purpose=purpose)
                    exc.details.update(column_recovery(exc.details, {k: v for k, v in arguments.items() if v is not None}))
                    exc.details['argument_basis'] = 'effective_python_inspection_arguments_defaults_may_be_included'
                elif self._handoff and exc.code not in _NO_HANDOFF_CODES and not input.startswith("store:"):
                    options = dict(time_column=time_column, target_column=target_column, series_column=series_column,
                                   frequency=frequency, as_of=as_of, recorded_as_of=recorded_as_of, store_path=store_path,
                                   regrid=regrid, timezone=timezone, window=window, unit=unit, purpose=purpose)
                    exc.details.update(self._repair_handoff(exc, input, repair, {k: v for k, v in options.items() if v is not None}))
                raise
            repairs = tuple(action.to_dict() for action in log.actions())
        # Count every retained vintage, not only the latest materialized rows.
        count = loaded.snapshot.observation_count
        if count > self.max_rows:
            raise GnomonError("INVALID_ARGUMENTS", "Input exceeds this session's retained observation limit.")
        ref = content_id("data", {"snapshot": loaded.snapshot.snapshot_id, "schema": asdict(loaded.schema),
                                  "unit": unit, "repairs": repairs}, length=64)
        frozen = _FrozenInput(loaded, unit, repairs, count)
        readiness = self._readiness(frozen)
        if not readiness[purpose]["ready"]:
            raise GnomonError("INPUT_NOT_READY", f"Input is not ready for {purpose}.",
                              {"purpose": purpose, **readiness[purpose]},
                              repair_options=readiness[purpose]["issues"])
        self._inputs.pop(ref, None)
        while self._inputs and (len(self._inputs) >= self.max_refs or
                               sum(item.row_count for item in self._inputs.values()) + count > self.max_rows):
            self._inputs.popitem(last=False)
        self._inputs[ref] = frozen
        return {"schema_version": "1", "status": "ok", "data_quality": data_quality(repairs, repair),
                "data_ref": ref,
                "frequency": loaded.frequency, "timezone": loaded.timezone, "unit": unit,
                "unit_basis": "caller_declared" if unit else "unknown",
                "series": [{"series_id": name, "count": len(rows),
                            "start": rows[0].timestamp.isoformat(), "end": rows[-1].timestamp.isoformat()}
                           for name, rows in sorted(loaded.groups.items())],
                "snapshot": loaded.snapshot.access_summary(), "repairs": deepcopy(list(frozen.repairs)),
                "readiness": readiness,
                'evaluation_replay': {'default_mode': 'recorded' if loaded.snapshot.recorded_as_of is not None else 'source_available',
                    'basis': 'recording_bounded_snapshot' if loaded.snapshot.recorded_as_of is not None else 'no_recording_boundary',
                    'preflight': {'tool': 'gnomon_evaluate', 'arguments': {'data_ref': ref, 'preflight': True},
                        'choices_required': ['candidates', 'baseline', 'horizon']},
                    'guidance': 'Use evaluation preflight to check actual planned origins. Recorded replay excludes history recorded after each origin. Source-availability replay requires an explicit semantic choice and never attests local historical availability.'},
                "reference_scope": "session", "eviction": "least_recently_used"}

    def _repair_handoff(self, exc, input, repair, options):
        """Diagnose a failed load inline so the caller's next call is spelled out.

        The next repair level is dry-run, never applied: the caller chooses it.
        Bounded like diagnose (local files up to 8 MiB); one rung per failure.
        """
        path = Path(input)
        if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
            return {}
        current = _REPAIR_LADDER.index(repair) if repair in _REPAIR_LADDER else 0
        probe = DataReferences(max_refs=1, max_rows=min(self.max_rows, 100000))
        probe._handoff = False
        try:
            for mode in _REPAIR_LADDER[current + 1:]:
                try:
                    result = probe.inspect(input, repair=mode, **options)
                except (GnomonError, ForecastAdapterError):
                    continue
                arguments = {"input": input, **options, "repair": mode}
                quality = {**result["data_quality"], "status": "needs_" + mode}
                return {"data_quality": {**quality, "next_call": {"tool": "gnomon_inspect", "arguments": arguments}},
                        "next_call": {"tool": "gnomon_inspect", "arguments": arguments, "runnable": True, "admissible": True},
                        "example_arguments": arguments, "example_kind": "task_correction", "example_runnable": True,
                        "admissible": True, "changed_fields": ["repair"],
                        "guidance": f"{mode} repair completes preparation within budget; its fixes are listed in "
                                    "repairs and never applied without this explicit choice."}
            cell = exc.details if "row" in exc.details else {}
            if not cell:
                # The strict pass names the first cell no policy could parse.
                try:
                    probe.inspect(input, repair="off", **options)
                except GnomonError as strict:
                    cell = strict.details if "row" in strict.details else {}
                except ForecastAdapterError:
                    cell = {}
        finally:
            probe.clear()
        correction = ({"action": "correct_target", "row": cell["row"], "value": cell.get("value"),
                       "guidance": "Correct this cell in the source file; no repair level within budget can prepare it."}
                      if cell else
                      {"action": "correct_source", "reason": exc.code,
                       "guidance": "No repair level within budget can prepare this input; correct the source file."})
        return {"data_quality": {"status": "rejected", "fixes": 0, "dropped": 0, "next_call": correction},
                "next_call": correction, "admissible": False, "example_runnable": False}

    def snapshot_summary(self, data_ref: str) -> dict:
        """Return provenance of the retained snapshot without reopening its source."""
        return deepcopy(self._get(data_ref).loaded.snapshot.access_summary())

    def save(self, data_ref: str, path: str) -> str:
        from .snapshot_files import save_snapshot
        return save_snapshot(self._get(data_ref), path)

    @staticmethod
    def _readiness(frozen):
        issues = []
        blockers = historical_repair_blockers(frozen.repairs)
        if blockers:
            issues.append({"action": "prepare_observed_history", "codes": blockers,
                           "description": "Historical scores need observed outcomes. Use a complete regular window of original "
                                          "observations, or prepare each historical vintage upstream. Globally filled gaps "
                                          "and shifted timestamps cannot be used as historical truth. For gaps, inspect the original "
                                          "file with --window latest_contiguous --frequency <frequency> --repair off.",
                           "example_input_options": {"window": "latest_contiguous", "frequency": frozen.loaded.frequency, "repair": "off"}})
        routing = deepcopy(issues)
        if any(row.timestamp.tzinfo is None for rows in frozen.loaded.groups.values() for row in rows):
            routing.append({"action": "declare_timezone", "description": "Declare the source timezone at inspection: "
                            "timezone=UTC (CLI: --timezone UTC) for files. For store input use TemporalStore.ingest_csv(timezone='UTC') at ingestion into a new dataset, or ingest timestamps with explicit offsets. Use the actual source timezone."})
        return {"infer": {"ready": True, "issues": []},
                "evaluate": {"ready": not issues, "issues": issues, "scope": "data_only_history_and_budget_checked_at_evaluation"},
                "route": {"ready": not routing, "issues": routing, "scope": "data_only_study_and_cutoffs_checked_at_routing"}}

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
            raise GnomonError("INVALID_ARGUMENTS", "Select one exact series_id from the inspected input. "
                              "series_id is a selector, not a new label; use series_column to read labels from input.",
                              details={"available_series_ids": sorted(groups), "requested_series_id": series_id})
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

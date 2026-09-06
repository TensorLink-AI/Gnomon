"""Shared snapshot-backed loading, independent of evaluation and context."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .contracts import DataSchema, GnomonError
from .data import Observation, load_observations
from .repair import RepairLog, repair_observations
from .temporal import validate_and_group
from .temporal_store import InMemoryTemporalStore, Snapshot, TemporalStore

STORE_SCHEME = "store:"


def _knowledge_bound_plain_rows(
    observations: list[Observation], as_of: datetime | None,
) -> list[Observation]:
    """Hide plain-file rows that were not knowable before repair runs.

    Plain files use ``valid_time`` as their assumed ``known_time``.  Repair
    can inspect neighbouring rows (snapping and interpolation), so applying
    it to the complete file and snapshotting afterwards lets a historical
    replay learn from post-``as_of`` rows even when those rows are absent
    from the final series.  The knowledge boundary must therefore precede
    every data-dependent repair.
    """
    if as_of is None:
        return observations
    for item in observations:
        if (item.timestamp.tzinfo is None) != (as_of.tzinfo is None):
            raise GnomonError(
                "SNAPSHOT_TIMEZONE_MISMATCH",
                "as_of versus the dataset mixes timezone-aware and naive "
                "timestamps; they cannot be compared.",
            )
    return [item for item in observations if item.timestamp <= as_of]


@dataclass(frozen=True)
class LoadedDataset:
    """Output of the load stage: validated observations grouped by series,
    plus the snapshot every later stage must read through."""

    source_fingerprint: str
    columns: list[str]
    groups: dict[str, list[Observation]]
    frequency: str
    timezone: str | None
    schema: DataSchema
    snapshot: Snapshot
    variable: str


def _regrid_frequency(requested: str | None, implied: str, policy: str) -> str:
    """A regrid states the calendar, so it states the frequency too; a
    conflicting explicit frequency is a contradiction to refuse, not to
    arbitrate."""
    from .temporal import normalise_frequency

    if requested is not None and normalise_frequency(requested) != implied:
        raise GnomonError(
            "INVALID_ARGUMENTS",
            f"regrid={policy} implies frequency {implied}, but the call "
            f"also named frequency {requested}. Drop one of the two.",
            {"regrid": policy, "implied": implied, "requested": requested},
        )
    return implied


def load_stage(
    input_path: str,
    *,
    time_column: str,
    target_column: str,
    series_column: str | None,
    frequency: str | None,
    as_of: datetime | None = None,
    recorded_as_of: datetime | None = None,
    store_path: str | None = None,
    repair: str = "off",
    repair_log: "RepairLog | None" = None,
    regrid: str | None = None,
) -> LoadedDataset:
    """Resolve the input to a snapshot, then materialise the observations
    that are visible at ``as_of``. ``store:<dataset>`` inputs read from the
    persistent bitemporal store; plain files are wrapped in an ephemeral
    store with ``known_time = valid_time`` so the snapshot guarantee is
    uniform across both."""
    variable = target_column
    if input_path.startswith(STORE_SCHEME):
        if regrid:
            raise GnomonError(
                "INVALID_ARGUMENTS",
                "regrid applies while loading a file or inline rows; a "
                "store dataset is already curated — regrid it at ingest "
                "instead, so every reader sees one grid.",
                {"input": input_path, "regrid": regrid},
            )
        dataset = input_path[len(STORE_SCHEME):]
        snapshot = TemporalStore(store_path).snapshot(dataset, as_of, recorded_as_of=recorded_as_of)
        source_fingerprint = snapshot.source_ref
        columns = [time_column, target_column] + ([series_column] if series_column else [])
    else:
        if recorded_as_of is not None:
            raise GnomonError(
                "INVALID_ARGUMENTS",
                "Plain files do not preserve historical recording times. "
                "Use a store input for recorded_as_of replay.",
            )
        log = repair_log if repair_log is not None else RepairLog()
        raw_observations, source_fingerprint, columns = load_observations(
            input_path, time_column, target_column, series_column,
            repair=repair, repair_log=log,
        )
        raw_observations = _knowledge_bound_plain_rows(raw_observations, as_of)
        _record_reordering(raw_observations, log)
        # Calendar first, messiness second: the declared regrid settles the
        # grid before repair_observations measures gaps against it —
        # otherwise aggressive repair tries to interpolate every weekend
        # itself and hits the assumptive ceiling regrid exists to spare.
        if regrid:
            from .repair import regrid_observations
            raw_observations, implied = regrid_observations(
                raw_observations, regrid, log)
            frequency = _regrid_frequency(frequency, implied, regrid)
        raw_observations = repair_observations(raw_observations, frequency, repair, log)
        store, _ = InMemoryTemporalStore.from_plain_observations(
            raw_observations, variable, source_fingerprint,
        )
        snapshot = store.snapshot(as_of)
    observations = [
        Observation(item.valid_time, item.value, entity)
        for entity in snapshot.entities()
        for item in snapshot.series(entity, variable)
    ]
    if not observations:
        raise GnomonError(
            "EMPTY_SNAPSHOT",
            "No observations are known at or before the requested as_of instant.",
            {"as_of": as_of.isoformat() if as_of else "latest"},
        )
    groups, resolved_frequency, zone = validate_and_group(observations, frequency)
    schema = DataSchema(time_column, target_column, series_column, resolved_frequency, zone)
    return LoadedDataset(
        source_fingerprint, columns, groups, resolved_frequency, zone, schema,
        snapshot, variable,
    )


def _record_reordering(observations: list[Observation], log: "RepairLog") -> None:
    """Note when the input arrived out of chronological order.

    Sorting is correct and happens further down, in the snapshot — which is
    also why it was invisible: by the time any check ran, the rows were
    already ordered, so a genuinely unsorted export never raised a
    data-quality signal. This runs while the file's own order is still
    observable.
    """
    from collections import defaultdict

    by_series: dict[str, list] = defaultdict(list)
    for item in observations:
        by_series[item.series].append(item.timestamp)
    for name, stamps in sorted(by_series.items()):
        ordered = sorted(stamps)
        if stamps == ordered:
            continue
        moved = sum(1 for left, right in zip(stamps, ordered) if left != right)
        log.record(
            "timestamps_reordered",
            "Rows arrived out of chronological order and were sorted. The "
            "sort is correct; it is recorded because an unsorted export is "
            "usually a symptom worth knowing about.",
            series=name, count=moved,
        )

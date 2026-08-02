"""The bitemporal data core.

Every observation carries two times: ``valid_time`` (when the value applies)
and ``known_time`` (when that exact vintage became available). All data
access in the execution path goes through a :class:`Snapshot`, a read handle
bound to an ``as_of`` instant that structurally cannot return rows published
after it — leakage control as a data-model guarantee, not a backtesting
practice. Every read is logged, so a run can prove what it touched.

Plain CSVs keep working with zero ceremony: when a source provides no
known-time column, ``known_time`` defaults to ``valid_time`` and the store
records a ``known_time_assumed`` warning instead of demanding new columns.
"""

from __future__ import annotations

import csv
import sqlite3
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .contracts import AionError
from .data import Observation, _parse_timestamp, fingerprint
from .ids import SYSTEM_CLOCK, Clock, content_id

_store_path = os.environ.get("AION_TEMPORAL_STORE_PATH")
DEFAULT_STORE_PATH = (
    Path(_store_path).expanduser()
    if _store_path
    else Path.home() / ".local" / "share" / "aion" / "temporal.db"
)

KNOWN_TIME_ASSUMED_WARNING = (
    "known_time_assumed: the source provides no known-time column, so each "
    "observation is treated as known the moment it applies (known_time = "
    "valid_time). Revisions and publication lags are invisible to this run."
)


@dataclass(frozen=True)
class TemporalObservation:
    entity: str
    variable: str
    valid_time: datetime
    known_time: datetime
    value: float
    revision: int = 0
    source_ref: str = ""


@dataclass(frozen=True)
class AccessRecord:
    """One logged snapshot read: what was asked for and what was visible."""

    entity: str
    variable: str
    cutoff: str
    rows: int
    max_known_time: str | None


def _comparable(left: datetime, right: datetime, *, what: str) -> None:
    if (left.tzinfo is None) != (right.tzinfo is None):
        raise AionError(
            "SNAPSHOT_TIMEZONE_MISMATCH",
            f"{what} mixes timezone-aware and naive timestamps; they cannot be compared.",
        )


class Snapshot:
    """Point-in-time read handle. Rows with ``known_time > as_of`` do not
    exist for holders of this object, and every read is logged."""

    def __init__(
        self,
        observations: list[TemporalObservation],
        as_of: datetime | None,
        *,
        source_ref: str = "",
        assumed_known_time: bool = False,
        known_time_provenance: str | None = None,
    ):
        if as_of is not None:
            # Every observation, not just the first: a dataset whose
            # aware/naive mix begins after row 0 used to reach the
            # comparison below and raise a bare TypeError.
            for item in observations:
                _comparable(item.known_time, as_of, what="as_of versus the dataset")
        self.as_of = as_of
        self.source_ref = source_ref
        self.assumed_known_time = assumed_known_time
        self.known_time_provenance = known_time_provenance or (
            "assumed" if assumed_known_time else "recorded"
        )
        self._observations = [
            item for item in observations
            if as_of is None or item.known_time <= as_of
        ]
        self._log: list[AccessRecord] = []

    # -- reads ------------------------------------------------------------

    def _visible(
        self, entity: str, variable: str, cutoff: datetime | None
    ) -> list[TemporalObservation]:
        if cutoff is not None and self.as_of is not None:
            cutoff = min(cutoff, self.as_of)
        elif cutoff is None:
            cutoff = self.as_of
        rows = [
            item for item in self._observations
            if item.entity == entity and item.variable == variable
            and (cutoff is None or item.known_time <= cutoff)
        ]
        latest_known = max((item.known_time for item in rows), default=None)
        self._log.append(AccessRecord(
            entity, variable,
            cutoff.isoformat() if cutoff is not None else "latest",
            len(rows),
            latest_known.isoformat() if latest_known is not None else None,
        ))
        return rows

    def series(
        self, entity: str, variable: str, *, cutoff: datetime | None = None
    ) -> list[TemporalObservation]:
        """The series as known at ``cutoff`` (default: this snapshot's
        as_of): latest revision per valid_time, ordered by valid_time."""
        best: dict[datetime, TemporalObservation] = {}
        for item in self._visible(entity, variable, cutoff):
            current = best.get(item.valid_time)
            if current is None or (item.known_time, item.revision) > (current.known_time, current.revision):
                best[item.valid_time] = item
        return [best[key] for key in sorted(best)]

    def value_as_of(
        self, entity: str, variable: str, valid_time: datetime,
        *, cutoff: datetime | None = None,
    ) -> float | None:
        """Latest vintage of one value, as known at ``cutoff``."""
        candidates = [
            item for item in self._visible(entity, variable, cutoff)
            if item.valid_time == valid_time
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item.known_time, item.revision)).value

    def entities(self) -> list[str]:
        return sorted({item.entity for item in self._observations})

    def variables(self, entity: str) -> list[str]:
        return sorted({item.variable for item in self._observations if item.entity == entity})

    # -- provenance -------------------------------------------------------

    @property
    def access_log(self) -> list[AccessRecord]:
        return list(self._log)

    def access_summary(self) -> dict[str, object]:
        """Aggregated, artifact-ready view of everything this snapshot served."""
        by_key: dict[tuple[str, str], dict[str, object]] = {}
        for record in self._log:
            entry = by_key.setdefault((record.entity, record.variable), {
                "entity": record.entity, "variable": record.variable,
                "reads": 0, "max_known_time": None,
            })
            entry["reads"] = int(entry["reads"]) + 1
            if record.max_known_time is not None:
                current = entry["max_known_time"]
                if current is None or record.max_known_time > str(current):
                    entry["max_known_time"] = record.max_known_time
        return {
            "as_of": self.as_of.isoformat() if self.as_of else "latest",
            "known_time_assumed": self.assumed_known_time,
            "known_time_provenance": self.known_time_provenance,
            "source_ref": self.source_ref,
            "accesses": [by_key[key] for key in sorted(by_key)],
        }


def join_as_of(
    snapshot: Snapshot,
    entity: str,
    variable: str,
    grid: list[datetime],
    *,
    cutoff: datetime | None = None,
) -> list[float | None]:
    """Point-in-time join: for each grid timestamp, the latest vintage of
    ``(entity, variable)`` known at ``cutoff`` — vintages, not final values."""
    series = {
        item.valid_time: item.value
        for item in snapshot.series(entity, variable, cutoff=cutoff)
    }
    return [series.get(moment) for moment in grid]


class InMemoryTemporalStore:
    """Ephemeral store wrapping already-loaded observations. This is how
    plain CSV inputs get the same snapshot guarantee as the persistent
    store, with zero extra ceremony."""

    def __init__(
        self,
        observations: list[TemporalObservation],
        *,
        source_ref: str = "",
        assumed_known_time: bool = False,
    ):
        self.observations = observations
        self.source_ref = source_ref
        self.assumed_known_time = assumed_known_time

    @classmethod
    def from_plain_observations(
        cls, observations: list[Observation], variable: str, source_ref: str,
    ) -> tuple["InMemoryTemporalStore", list[str]]:
        """Wrap single-vintage observations, assuming known_time = valid_time.

        Duplicate timestamps are rejected here rather than deduplicated:
        in a single-vintage source they are ambiguous data, not revisions."""
        seen: set[tuple[str, datetime]] = set()
        for item in observations:
            key = (item.series, item.timestamp)
            if key in seen:
                raise AionError(
                    "DUPLICATE_TIMESTAMPS",
                    f"Series {item.series} contains duplicate timestamps.",
                )
            seen.add(key)
        rows = [
            TemporalObservation(
                entity=item.series, variable=variable,
                valid_time=item.timestamp, known_time=item.timestamp,
                value=item.value, revision=0, source_ref=source_ref,
            )
            for item in observations
        ]
        return cls(rows, source_ref=source_ref, assumed_known_time=True), [KNOWN_TIME_ASSUMED_WARNING]

    def snapshot(self, as_of: datetime | None = None) -> Snapshot:
        return Snapshot(
            self.observations, as_of,
            source_ref=self.source_ref, assumed_known_time=self.assumed_known_time,
        )


@dataclass
class IngestReport:
    ingest_id: str
    dataset: str
    source_fingerprint: str
    rows_seen: int = 0
    rows_added: int = 0
    revisions_created: int = 0
    #: Exact repeats of an existing vintage — same valid_time, same
    #: known_time, same value. A genuine no-op re-ingest.
    duplicates_skipped: int = 0
    #: Vintages whose value restates a figure the series already carried at
    #: an earlier known_time. Recorded like any other vintage; counted
    #: separately because a value-only dedup used to drop them silently.
    reverts_recorded: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "0.1",
            "status": "ok",
            **self.__dict__,
        }


class TemporalStore:
    """SQLite-backed bitemporal store. Ingestion appends revisions instead
    of overwriting: re-supplied files with corrected history become new
    revision rows, and the accumulated vintage history is what no model can
    reconstruct after the fact."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_STORE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS observations (
                    dataset TEXT NOT NULL,
                    entity TEXT NOT NULL,
                    variable TEXT NOT NULL,
                    valid_time TEXT NOT NULL,
                    known_time TEXT NOT NULL,
                    value REAL NOT NULL,
                    revision INTEGER NOT NULL,
                    source_ref TEXT NOT NULL,
                    PRIMARY KEY (dataset, entity, variable, valid_time, revision)
                );
                CREATE INDEX IF NOT EXISTS idx_observations_dataset
                    ON observations (dataset, entity, variable, valid_time);
                CREATE TABLE IF NOT EXISTS ingests (
                    ingest_id TEXT PRIMARY KEY,
                    dataset TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    rows_added INTEGER NOT NULL,
                    revisions_created INTEGER NOT NULL,
                    duplicates_skipped INTEGER NOT NULL,
                    reverts_recorded INTEGER NOT NULL DEFAULT 0,
                    -- Whether this ingest supplied a known-time column, or
                    -- Aion assumed known_time = valid_time. Recorded here
                    -- because it is a fact about the *ingest*; inferring it
                    -- from the data mislabels genuine same-day publication
                    -- as assumed, and hides an assumed ingest that later
                    -- sits beside a real one.
                    assumed_known_time INTEGER NOT NULL DEFAULT 1
                );
            """)
            existing = {
                row["name"] for row in conn.execute("PRAGMA table_info(ingests)")
            }
            for name, sql_type in (("reverts_recorded", "INTEGER NOT NULL DEFAULT 0"),
                                   ("assumed_known_time", "INTEGER NOT NULL DEFAULT 1")):
                if name not in existing:
                    conn.execute(f"ALTER TABLE ingests ADD COLUMN {name} {sql_type}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- ingestion --------------------------------------------------------

    def ingest_rows(
        self,
        dataset: str,
        rows: list[TemporalObservation],
        *,
        source_fingerprint: str,
        assumed_known_time: bool = False,
        clock: Clock | None = None,
    ) -> IngestReport:
        clock = clock or SYSTEM_CLOCK
        ingest_id = content_id("ingest", {
            "dataset": dataset,
            "source": source_fingerprint,
            "rows": [
                (r.entity, r.variable, r.valid_time.isoformat(),
                 r.known_time.isoformat(), r.value)
                for r in rows
            ],
        })
        report = IngestReport(ingest_id, dataset, source_fingerprint, rows_seen=len(rows))
        if assumed_known_time:
            report.warnings.append(KNOWN_TIME_ASSUMED_WARNING)
        with self._connect() as conn:
            for row in rows:
                existing = conn.execute(
                    "SELECT known_time, value, revision FROM observations "
                    "WHERE dataset=? AND entity=? AND variable=? AND valid_time=? "
                    "ORDER BY revision",
                    (dataset, row.entity, row.variable, row.valid_time.isoformat()),
                ).fetchall()
                # A vintage is identified by (valid_time, known_time, value).
                # Deduplicating on value alone dropped a *revert*: a series
                # revised 100 -> 150 -> back to 100 lost its third vintage,
                # and a replay after the revert returned the superseded 150.
                # Only an exact repeat of an existing vintage is a no-op.
                known_time = row.known_time.isoformat()
                if any(
                    item["known_time"] == known_time and item["value"] == row.value
                    for item in existing
                ):
                    report.duplicates_skipped += 1
                    continue
                if any(item["value"] == row.value for item in existing):
                    # Same value, new known_time: a restatement back to a
                    # figure this series already carried. Recorded, and
                    # counted separately so it is visible that it happened.
                    report.reverts_recorded += 1
                revision = existing[-1]["revision"] + 1 if existing else 0
                conn.execute(
                    "INSERT INTO observations "
                    "(dataset, entity, variable, valid_time, known_time, value, revision, source_ref) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (dataset, row.entity, row.variable, row.valid_time.isoformat(),
                     row.known_time.isoformat(), row.value, revision, ingest_id),
                )
                report.rows_added += 1
                if revision > 0:
                    report.revisions_created += 1
            conn.execute(
                "INSERT OR REPLACE INTO ingests "
                "(ingest_id, dataset, source_fingerprint, ingested_at, rows_added, "
                " revisions_created, duplicates_skipped, reverts_recorded, "
                " assumed_known_time) VALUES (?,?,?,?,?,?,?,?,?)",
                (ingest_id, dataset, source_fingerprint, clock.now().isoformat(),
                 report.rows_added, report.revisions_created, report.duplicates_skipped,
                 report.reverts_recorded, int(assumed_known_time)),
            )
        return report

    def ingest_csv(
        self,
        input_path: str,
        *,
        dataset: str,
        time_column: str,
        target_column: str,
        series_column: str | None = None,
        known_at_column: str | None = None,
        variable: str | None = None,
        clock: Clock | None = None,
    ) -> IngestReport:
        path = Path(input_path).expanduser().resolve()
        if not path.is_file():
            raise AionError("INPUT_NOT_FOUND", f"Input file does not exist: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            raw_rows = list(reader)
        required = [time_column, target_column]
        if series_column:
            required.append(series_column)
        if known_at_column:
            required.append(known_at_column)
        missing = [column for column in required if column not in columns]
        if missing:
            raise AionError(
                "MISSING_COLUMNS", f"Required columns are missing: {', '.join(missing)}",
                {"available_columns": columns, "missing_columns": missing},
            )
        variable_name = variable or target_column
        rows: list[TemporalObservation] = []
        for number, raw in enumerate(raw_rows, start=2):
            valid_time = _parse_timestamp(raw[time_column], number)
            known_time = (
                _parse_timestamp(raw[known_at_column], number)
                if known_at_column else valid_time
            )
            if (valid_time.tzinfo is None) != (known_time.tzinfo is None):
                raise AionError(
                    "MIXED_TIMEZONES",
                    f"valid and known timestamps mix aware and naive values on row {number}.",
                )
            try:
                value = float(raw[target_column])
            except (TypeError, ValueError) as exc:
                raise AionError(
                    "INVALID_TARGET", f"Target is not numeric on row {number}.",
                    {"row": number, "value": raw.get(target_column)},
                ) from exc
            entity = str(raw[series_column]) if series_column else "__default__"
            rows.append(TemporalObservation(
                entity=entity, variable=variable_name, valid_time=valid_time,
                known_time=known_time, value=value,
            ))
        if not rows:
            raise AionError("EMPTY_DATASET", "The input contains no observations.")
        return self.ingest_rows(
            dataset, rows,
            source_fingerprint=fingerprint(path),
            assumed_known_time=known_at_column is None,
            clock=clock,
        )

    # -- reads ------------------------------------------------------------

    def _load_dataset(self, dataset: str) -> list[TemporalObservation]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT entity, variable, valid_time, known_time, value, revision, source_ref "
                "FROM observations WHERE dataset=? ORDER BY entity, variable, valid_time, revision",
                (dataset,),
            ).fetchall()
        if not rows:
            raise AionError(
                "DATASET_NOT_FOUND",
                f"No dataset named {dataset!r} in the temporal store.",
                {"available": [item["dataset"] for item in self.list_datasets()]},
            )
        return [
            TemporalObservation(
                entity=row["entity"], variable=row["variable"],
                valid_time=datetime.fromisoformat(row["valid_time"]),
                known_time=datetime.fromisoformat(row["known_time"]),
                value=row["value"], revision=row["revision"],
                source_ref=row["source_ref"],
            )
            for row in rows
        ]

    def snapshot(self, dataset: str, as_of: datetime | None = None) -> Snapshot:
        observations = self._load_dataset(dataset)
        provenance = self.known_time_provenance(dataset)
        return Snapshot(
            observations, as_of,
            source_ref=self.dataset_fingerprint(dataset),
            assumed_known_time=provenance != "recorded",
            known_time_provenance=provenance,
        )

    def known_time_provenance(self, dataset: str) -> str:
        """``recorded`` | ``assumed`` | ``partially_assumed``.

        Read from the ingests that built the dataset, not inferred from the
        data: a series genuinely published the day it applies has
        ``valid_time == known_time`` for every row and is not assumed, while
        a dataset mixing one assumed ingest with one real one has rows where
        they differ and is not fully recorded either. Inference cannot tell
        those apart; provenance can.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT i.assumed_known_time AS assumed "
                "FROM ingests i WHERE i.dataset = ? AND EXISTS ("
                "  SELECT 1 FROM observations o "
                "  WHERE o.dataset = i.dataset AND o.source_ref = i.ingest_id)",
                (dataset,),
            ).fetchall()
        flags = {bool(row["assumed"]) for row in rows}
        if flags == {True}:
            return "assumed"
        if flags == {False}:
            return "recorded"
        if not flags:
            # Rows with no surviving ingest record: pre-provenance stores.
            return "assumed"
        return "partially_assumed"

    def dataset_fingerprint(self, dataset: str) -> str:
        """Content address of the dataset's full current vintage history."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, MAX(known_time) AS latest, MAX(rowid) AS last "
                "FROM observations WHERE dataset=?",
                (dataset,),
            ).fetchone()
        return content_id("dataset", {
            "dataset": dataset, "rows": row["n"], "latest": row["latest"], "last": row["last"],
        })

    def list_datasets(self) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT dataset, COUNT(*) AS observations, "
                "COUNT(DISTINCT entity) AS entities, "
                "COUNT(DISTINCT variable) AS variables, "
                "SUM(CASE WHEN revision > 0 THEN 1 ELSE 0 END) AS revisions, "
                "MIN(valid_time) AS first_valid, MAX(valid_time) AS last_valid, "
                "MAX(known_time) AS last_known "
                "FROM observations GROUP BY dataset ORDER BY dataset",
            ).fetchall()
        return [dict(row) for row in rows]

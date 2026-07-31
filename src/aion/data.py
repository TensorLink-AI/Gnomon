from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .contracts import AionError


@dataclass(frozen=True)
class Observation:
    timestamp: datetime
    value: float
    series: str


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _parse_timestamp(raw: object, row: int) -> datetime:
    text = str(raw).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise AionError(
            "INVALID_TIMESTAMP", f"Cannot parse timestamp on row {row}: {raw!r}",
            {"row": row, "value": str(raw)},
        ) from exc
    return parsed


def _rows_from_parquet(path: Path) -> list[dict[str, object]]:
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as exc:
        raise AionError(
            "MISSING_OPTIONAL_DEPENDENCY",
            "Parquet input requires the 'parquet' extra.",
            {"install": "pip install 'aion-forecast[parquet]'"},
        ) from exc
    return parquet.read_table(path).to_pylist()


def _load_lenient(
    rows: list[dict[str, object]],
    time_column: str,
    target_column: str,
    series_column: str | None,
    level: str,
    log: "RepairLog",
) -> list[Observation]:
    """The repair-enabled row loop: identical to the strict loop for clean
    cells, lenient — with disclosure — where the strict loop would raise."""
    from .repair import (
        MAX_DROPPED_FRACTION, REPAIR_AGGRESSIVE, AmbiguousDateOrder,
        parse_number, parse_timestamp_lenient, scan_day_first,
        scan_numeric_evidence,
    )

    time_texts = ["" if row.get(time_column) is None else str(row[time_column]) for row in rows]
    target_texts = ["" if row.get(target_column) is None else str(row[target_column]) for row in rows]
    day_first = scan_day_first(time_texts)
    comma_role = scan_numeric_evidence(target_texts)
    observations: list[Observation] = []
    dropped = 0
    for row_number, row in enumerate(rows, start=2):
        raw_time = time_texts[row_number - 2]
        raw_target = target_texts[row_number - 2]
        if not raw_time.strip() and not raw_target.strip() and all(
            not str(value or "").strip() for value in row.values()
        ):
            log.record("blank_row_skipped", "Fully blank rows were skipped.")
            continue
        series = str(row.get(series_column, "")) if series_column else "__default__"
        try:
            value, tier = parse_number(raw_target, comma_role)
        except ValueError as exc:
            if level == REPAIR_AGGRESSIVE:
                dropped += 1
                log.record("unparseable_row_dropped",
                           "Rows whose target has no numeric reading were dropped.",
                           series=series, assumptive=True, example=raw_target.strip())
                continue
            raise AionError(
                "INVALID_TARGET", f"Target is not numeric on row {row_number}.",
                {"row": row_number, "value": row.get(target_column)},
            ) from exc
        if tier == "missing":
            log.record("missing_value_dropped",
                       "Sentinel missing values (N/A, null, blank, …) were "
                       "treated as absent observations.",
                       series=series, example=raw_target.strip() or '""')
            continue
        if tier == "normalised":
            log.record("numeric_format_normalised",
                       "Currency symbols, group separators, percent signs, or "
                       "accounting negatives were normalised to plain numbers.",
                       series=series, example=raw_target.strip())
        elif tier == "assumptive":
            log.record("thousands_separator_assumed",
                       "Grouped commas with no column-wide evidence were read "
                       "as thousands separators.",
                       series=series, assumptive=True, example=raw_target.strip())
        try:
            timestamp, ts_tier = parse_timestamp_lenient(raw_time, day_first)
        except AmbiguousDateOrder as exc:
            if level == REPAIR_AGGRESSIVE:
                timestamp, ts_tier = parse_timestamp_lenient(raw_time, False)
                log.record("date_order_assumed",
                           "Ambiguous day/month order was read as month-first.",
                           series=series, assumptive=True, example=raw_time.strip())
            else:
                raise AionError(
                    "AMBIGUOUS_DATE_ORDER",
                    f"Date order is ambiguous on row {row_number}: {raw_time.strip()!r} "
                    "could be day-first or month-first and no row in the column "
                    "decides it.",
                    {"row": row_number, "value": raw_time.strip()},
                ) from exc
        except ValueError as exc:
            if level == REPAIR_AGGRESSIVE:
                dropped += 1
                log.record("unparseable_row_dropped",
                           "Rows whose timestamp has no reading were dropped.",
                           series=series, assumptive=True, example=raw_time.strip())
                continue
            raise AionError(
                "INVALID_TIMESTAMP",
                f"Cannot parse timestamp on row {row_number}: {row.get(time_column)!r}",
                {"row": row_number, "value": str(row.get(time_column))},
            ) from exc
        if ts_tier == "normalised":
            log.record("timestamp_format_normalised",
                       "Non-ISO timestamp formats were normalised.",
                       series=series, example=raw_time.strip())
        observations.append(Observation(timestamp, value, series))
    if rows and dropped / len(rows) > MAX_DROPPED_FRACTION:
        raise AionError(
            "EXCESSIVE_REPAIR",
            f"More than {MAX_DROPPED_FRACTION:.0%} of rows are unparseable; "
            "fix the file at the source instead of forecasting the remainder.",
            {"dropped_rows": dropped, "total_rows": len(rows)},
        )
    aware = [item.timestamp.utcoffset() is not None for item in observations]
    if any(aware) and not all(aware):
        if level == REPAIR_AGGRESSIVE:
            naive_count = sum(1 for flag in aware if not flag)
            observations = [
                item if item.timestamp.utcoffset() is not None
                else Observation(item.timestamp.replace(tzinfo=timezone.utc), item.value, item.series)
                for item in observations
            ]
            log.record("timezone_coerced",
                       "Naive timestamps in a timezone-aware file were assumed UTC.",
                       assumptive=True, count=naive_count)
        else:
            raise AionError("MIXED_TIMEZONES", "Timestamps mix timezone-aware and naive values.")
    return observations


def load_observations(
    input_path: str, time_column: str, target_column: str, series_column: str | None,
    *, repair: str = "off", repair_log: "RepairLog | None" = None,
) -> tuple[list[Observation], str, list[str]]:
    path = Path(input_path).expanduser().resolve()
    if not path.is_file():
        raise AionError("INPUT_NOT_FOUND", f"Input file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            rows: list[dict[str, object]] = list(reader)
    elif suffix in {".parquet", ".pq"}:
        rows = _rows_from_parquet(path)
        columns = list(rows[0]) if rows else []
    else:
        raise AionError("UNSUPPORTED_INPUT", "Only CSV and Parquet inputs are supported.")

    required = [time_column, target_column] + ([series_column] if series_column else [])
    missing = [column for column in required if column not in columns]
    if missing:
        raise AionError(
            "MISSING_COLUMNS", f"Required columns are missing: {', '.join(missing)}",
            {"available_columns": columns, "missing_columns": missing},
        )
    if repair != "off":
        from .repair import RepairLog
        observations = _load_lenient(
            rows, time_column, target_column, series_column, repair,
            repair_log if repair_log is not None else RepairLog(),
        )
    else:
        observations = []
        for row_number, row in enumerate(rows, start=2):
            try:
                value = float(row[target_column])
            except (TypeError, ValueError) as exc:
                raise AionError(
                    "INVALID_TARGET", f"Target is not numeric on row {row_number}.",
                    {"row": row_number, "value": row.get(target_column)},
                ) from exc
            series = str(row[series_column]) if series_column else "__default__"
            observations.append(
                Observation(_parse_timestamp(row[time_column], row_number), value, series)
            )
    if not observations:
        raise AionError("EMPTY_DATASET", "The input contains no observations.")
    return observations, fingerprint(path), columns


def timezone_name(values: list[datetime]) -> str | None:
    aware = [value.utcoffset() is not None for value in values]
    if any(aware) and not all(aware):
        raise AionError("MIXED_TIMEZONES", "Timestamps mix timezone-aware and naive values.")
    if not any(aware):
        return None
    offsets = {value.utcoffset() for value in values}
    if len(offsets) > 1:
        return "variable-offset"
    offset = next(iter(offsets))
    if offset == timezone.utc.utcoffset(None):
        return "UTC"
    return str(values[0].tzinfo)


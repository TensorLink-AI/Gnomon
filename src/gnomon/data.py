from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .contracts import GnomonError

if TYPE_CHECKING:
    from .repair import RepairLog


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
        raise GnomonError(
            "INVALID_TIMESTAMP", f"Cannot parse timestamp on row {row}: {raw!r}",
            {"row": row, "value": str(raw)},
        ) from exc
    return parsed


def _rows_from_parquet(path: Path) -> list[dict[str, object]]:
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as exc:
        raise GnomonError(
            "MISSING_OPTIONAL_DEPENDENCY",
            "Parquet input requires the 'parquet' extra.",
            {"install": "pip install 'gnomon-forecast[parquet]'"},
        ) from exc
    return parquet.read_table(path).to_pylist()


def _rows_from_excel(path: Path) -> tuple[list[dict[str, object]], list[str]]:
    try:
        import openpyxl  # type: ignore[import-not-found]
    except ImportError as exc:
        raise GnomonError(
            "MISSING_OPTIONAL_DEPENDENCY",
            "Excel input requires the 'excel' extra.",
            {"install": "pip install 'gnomon-forecast[excel]'"},
        ) from exc
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook.worksheets[0]
        iterator = sheet.iter_rows(values_only=True)
        header = next(iterator, None)
        if header is None:
            return [], []
        columns = [str(cell) if cell is not None else "" for cell in header]
        rows = [
            {column: cell for column, cell in zip(columns, values)}
            for values in iterator
        ]
    finally:
        workbook.close()
    return rows, columns


def _read_text(path: Path, gzipped: bool, repair: str, log: "RepairLog") -> str:
    data = path.read_bytes()
    if gzipped:
        import gzip
        data = gzip.decompress(data)
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        if repair == "off":
            raise GnomonError(
                "INVALID_ENCODING",
                "The file is not valid UTF-8.",
                {"hint": "Re-export as UTF-8, or explicitly select repair=safe "
                         "to allow a disclosed Windows-1252 assumption."},
            ) from exc
        log.record("encoding_assumed",
                   "File is not UTF-8; the Windows-1252 encoding was assumed.",
                   assumptive=True)
        return data.decode("cp1252", errors="replace")


def _choose_delimiter(
    header_line: str, time_column: str, target_column: str,
    repair: str, log: "RepairLog",
) -> str:
    """Comma unless the required columns are provably split by another
    delimiter (European semicolon exports, tabs, pipes). Detection is gated
    on the strict choice failing and is disclosed, never guessed: the header
    itself must name both mapped columns under the alternative."""
    def fields(delimiter: str) -> list[str]:
        return [field.strip().strip('"') for field in header_line.split(delimiter)]

    if repair == "off":
        return ","
    if time_column in fields(",") and target_column in fields(","):
        return ","
    for delimiter, name in ((";", "semicolon"), ("\t", "tab"), ("|", "pipe")):
        if time_column in fields(delimiter) and target_column in fields(delimiter):
            log.record("delimiter_detected",
                       f"Columns are {name}-separated, not comma-separated.")
            return delimiter
    return ","  # the MISSING_COLUMNS error downstream lists what was found


def _read_rows(
    path: Path, time_column: str, target_column: str,
    repair: str, log: "RepairLog",
) -> tuple[list[dict[str, object]], list[str]]:
    import io
    import json as json_module

    suffixes = [suffix.lower() for suffix in path.suffixes]
    suffix = suffixes[-1] if suffixes else ""
    gzipped = suffix == ".gz"
    if gzipped:
        suffix = suffixes[-2] if len(suffixes) >= 2 else ""
    if suffix in {".parquet", ".pq"}:
        rows = _rows_from_parquet(path)
        return rows, list(rows[0]) if rows else []
    if suffix == ".xlsx":
        return _rows_from_excel(path)
    if suffix in {".csv", ".tsv"}:
        text = _read_text(path, gzipped, repair, log)
        header_line = text.splitlines()[0] if text else ""
        delimiter = "\t" if suffix == ".tsv" else _choose_delimiter(
            header_line, time_column, target_column, repair, log,
        )
        reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter)
        return list(reader), list(reader.fieldnames or [])
    if suffix in {".json", ".jsonl", ".ndjson"}:
        text = _read_text(path, gzipped, repair, log)
        if suffix == ".json":
            payload = json_module.loads(text)
            if not (isinstance(payload, list)
                    and all(isinstance(item, dict) for item in payload)):
                raise GnomonError(
                    "UNSUPPORTED_INPUT",
                    "JSON input must be a top-level array of flat objects.",
                )
            rows = payload
        else:
            rows = [
                json_module.loads(line)
                for line in text.splitlines() if line.strip()
            ]
        return rows, list(rows[0]) if rows else []
    raise GnomonError(
        "UNSUPPORTED_INPUT",
        "Supported inputs: .csv, .tsv, .json, .jsonl (each optionally "
        ".gz-compressed), .parquet/.pq, and .xlsx.",
    )


def _load_lenient(
    rows: list[dict[str, object]],
    time_column: str,
    target_column: str,
    series_column: str | None,
    level: str,
    log: "RepairLog",
    default_series: str = "__default__",
    _defer_timezone: bool = False,
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
        series = str(row.get(series_column, "")) if series_column else default_series
        try:
            value, tier = parse_number(raw_target, comma_role)
        except ValueError as exc:
            if level == REPAIR_AGGRESSIVE:
                dropped += 1
                log.record("unparseable_row_dropped",
                           "Rows whose target has no numeric reading were dropped.",
                           series=series, assumptive=True, example=raw_target.strip())
                continue
            raise GnomonError(
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
            try:
                local_value, local_tier = parse_number(raw_target, None)
            except ValueError:
                local_value, local_tier = None, "assumptive"
            if local_value != value or local_tier == "assumptive":
                log.record("numeric_format_inferred", "Numeric punctuation was resolved using other rows in the column.",
                           series=series, example=raw_target.strip())
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
                raise GnomonError(
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
            raise GnomonError(
                "INVALID_TIMESTAMP",
                f"Cannot parse timestamp on row {row_number}: {row.get(time_column)!r}",
                {"row": row_number, "value": str(row.get(time_column)),
                 "drop_budget": {"invalid_rows_at_least": 1, "total_rows": len(rows),
                                 "max_fraction": MAX_DROPPED_FRACTION, "denominator_basis": "all_input_rows"}},
            ) from exc
        if ts_tier == "normalised":
            try:
                local_stamp, _ = parse_timestamp_lenient(raw_time, None)
            except AmbiguousDateOrder:
                local_stamp = None
            if local_stamp != timestamp:
                log.record("date_format_inferred", "Day/month order was resolved using other rows in the column.",
                           series=series, example=raw_time.strip())
            log.record("timestamp_format_normalised",
                       "Non-ISO timestamp formats were normalised.",
                       series=series, example=raw_time.strip())
        observations.append(Observation(timestamp, value, series))
    if rows and dropped / len(rows) > MAX_DROPPED_FRACTION:
        raise GnomonError(
            "EXCESSIVE_REPAIR",
            f"More than {MAX_DROPPED_FRACTION:.0%} of rows are unparseable; "
            "fix the file at the source instead of forecasting the remainder.",
            {"dropped_rows": dropped, "total_rows": len(rows), "max_fraction": MAX_DROPPED_FRACTION,
             "denominator_basis": "all_input_rows"},
        )
    aware = [item.timestamp.utcoffset() is not None for item in observations]
    if any(aware) and not all(aware) and not _defer_timezone:
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
            raise GnomonError("MIXED_TIMEZONES", "Timestamps mix timezone-aware and naive values.")
    return observations


def load_observations(
    input_path: str, time_column: str, target_column: str, series_column: str | None,
    *, repair: str = "off", repair_log: "RepairLog | None" = None,
    _defer_timezone: bool = False,
) -> tuple[list[Observation], str, list[str]]:
    from .repair import RepairLog, validate_repair_level
    validate_repair_level(repair)
    path = Path(input_path).expanduser().resolve()
    if not path.is_file():
        raise GnomonError("INPUT_NOT_FOUND", f"Input file does not exist: {path}")
    log = repair_log if repair_log is not None else RepairLog()
    rows, columns = _read_rows(path, time_column, target_column, repair, log)
    observations = observations_from_rows(
        rows, columns, time_column, target_column, series_column,
        repair=repair, repair_log=log, _defer_timezone=_defer_timezone,
    )
    return observations, fingerprint(path), columns


def _drop_diagnostic(rows, time_column, target_column):
    """Bounded, nonmutating cost scan using the aggressive parsing policy.

    Passing this budget says nothing about subsequent grid/conflict budgets.
    """
    from .repair import MAX_DROPPED_FRACTION, AmbiguousDateOrder, parse_number, parse_timestamp_lenient, scan_day_first, scan_numeric_evidence
    limit = 100_000
    detail = {"total_rows": len(rows), "max_fraction": MAX_DROPPED_FRACTION,
              "denominator_basis": "all_input_rows", "diagnostic_limit": limit,
              "scan_complete": len(rows) <= limit, "policy": "aggressive_parsing_drop_cost",
              "scope": "unparseable_drops_only_grid_and_conflict_budgets_checked_separately"}
    if len(rows) > limit:
        return {**detail, "scanned_rows": 0, "within_budget": None}
    times = ["" if r.get(time_column) is None else str(r[time_column]) for r in rows]
    values = ["" if r.get(target_column) is None else str(r[target_column]) for r in rows]
    day_first, comma_role = scan_day_first(times), scan_numeric_evidence(values)
    dropped = affected = bad_times = bad_targets = 0
    for row, raw_time, raw_value in zip(rows, times, values):
        if not raw_time.strip() and not raw_value.strip() and all(not str(v or "").strip() for v in row.values()):
            continue
        bad_target, bad_time, tier = False, False, None
        try:
            _, tier = parse_number(raw_value, comma_role)
        except ValueError:
            bad_target = True
        try:
            try:
                parse_timestamp_lenient(raw_time, day_first)
            except AmbiguousDateOrder:
                parse_timestamp_lenient(raw_time, False)
        except ValueError:
            bad_time = True
        bad_times += bad_time
        bad_targets += bad_target
        affected += bad_time or bad_target
        dropped += bad_target or (bad_time and tier != 'missing')
    return {**detail, "scanned_rows": len(rows), "dropped_rows": dropped,
            "affected_rows": affected, "invalid_timestamp_fields": bad_times, "invalid_target_fields": bad_targets,
            "proposed_drops": dropped, "predicted_rows_after_drops": len(rows) - dropped,
            "maximum_allowed_drops": int(len(rows) * MAX_DROPPED_FRACTION),
            "fraction": dropped / len(rows) if rows else 0,
            "within_budget": not rows or dropped / len(rows) <= MAX_DROPPED_FRACTION}


def observations_from_rows(rows, columns, time_column, target_column, series_column, **kwargs):
    """Extract observations; early parse errors include a bounded drop-cost scan."""
    try:
        return _observations_from_rows(rows, columns, time_column, target_column, series_column, **kwargs)
    except GnomonError as exc:
        if exc.code in {"INVALID_TIMESTAMP", "INVALID_TARGET", "NON_FINITE_TARGET"}:
            exc.details["drop_budget"] = _drop_diagnostic(rows, time_column, target_column)
            if kwargs.get('repair', 'off') == 'safe':
                exc.details['guidance'] = ('Safe mode never drops unparseable rows. Check drop_budget.within_budget; '
                    'retry with aggressive only if dropping those rows preserves the intended task. Grid and conflict budgets still apply.')
                exc.repair_options = [{'action': 'review_aggressive_drop', 'description': exc.details['guidance']}]
        raise


def _observations_from_rows(
    rows: list[dict[str, object]],
    columns: list[str],
    time_column: str,
    target_column: str,
    series_column: str | None,
    *,
    repair: str = "off",
    repair_log: "RepairLog | None" = None,
    default_series: str = "__default__",
    _defer_timezone: bool = False,
) -> list[Observation]:
    """Extract one target column from rows using an explicit repair policy."""
    from .repair import RepairLog, validate_repair_level
    validate_repair_level(repair)
    log = repair_log if repair_log is not None else RepairLog()
    required = [time_column, target_column] + ([series_column] if series_column else [])
    missing = [column for column in required if column not in columns]
    if missing:
        raise GnomonError(
            "MISSING_COLUMNS", f"Required columns are missing: {', '.join(missing)}",
            {"available_columns": columns, "missing_columns": missing},
            repair_options=[{
                "action": "supply_column_mapping",
                "description": "All providers default to time_column=timestamp and target_column=value. "
                               "Set time_column/target_column to your column names "
                               "(CLI: --time-column ts --target-column value for ts,value input).",
            }],
        )
    if repair != "off":
        observations = _load_lenient(
            rows, time_column, target_column, series_column, repair, log,
            default_series, _defer_timezone=_defer_timezone,
        )
    else:
        observations = []
        for row_number, row in enumerate(rows, start=2):
            try:
                value = float(row[target_column])
            except (TypeError, ValueError) as exc:
                raise GnomonError(
                    "INVALID_TARGET", f"Target is not numeric on row {row_number}.",
                    {"row": row_number, "value": row.get(target_column)},
                ) from exc
            if not math.isfinite(value):
                # NaN and infinities parse as floats but are not
                # observations: NaN poisons every downstream aggregate it
                # touches (a mean with one NaN fold is NaN, not None), and
                # an infinite level has no forecastable meaning. Refuse
                # here, before model selection can ingest one.
                raise GnomonError(
                    "NON_FINITE_TARGET",
                    f"Target is {str(row.get(target_column)).strip()!r} on row "
                    f"{row_number}: NaN and infinite values cannot be "
                    "observations.",
                    {"row": row_number, "value": str(row.get(target_column))},
                    repair_options=[{
                        "action": "fix_source",
                        "description": "Repair the exporting system; NaN usually "
                                       "marks a missing measurement.",
                    }, {
                        "action": "supply_arguments",
                        "description": "repair=\"aggressive\" drops non-finite "
                                       "rows with a repair-log entry, only up to 5% of all input rows.",
                        "arguments": ["repair"],
                    }],
                )
            series = str(row[series_column]) if series_column else default_series
            try:
                timestamp = _parse_timestamp(row[time_column], row_number)
            except GnomonError as exc:
                from .repair import MAX_DROPPED_FRACTION
                exc.details["drop_budget"] = {"invalid_rows_at_least": 1, "total_rows": len(rows),
                    "max_fraction": MAX_DROPPED_FRACTION, "denominator_basis": "all_input_rows"}
                raise
            observations.append(Observation(timestamp, value, series))
    if not observations:
        raise GnomonError("EMPTY_DATASET", "The input contains no observations.")
    return observations


def timezone_name(values: list[datetime]) -> str | None:
    aware = [value.utcoffset() is not None for value in values]
    if any(aware) and not all(aware):
        raise GnomonError("MIXED_TIMEZONES", "Timestamps mix timezone-aware and naive values.")
    if not any(aware):
        return None
    offsets = {value.utcoffset() for value in values}
    if len(offsets) > 1:
        return "variable-offset"
    offset = next(iter(offsets))
    if offset == timezone.utc.utcoffset(None):
        return "UTC"
    return str(values[0].tzinfo)

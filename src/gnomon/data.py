from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .contracts import GnomonError


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
                {"hint": "Re-export as UTF-8, or rely on the default repair "
                         "level which assumes Windows-1252."},
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
                {"row": row_number, "value": str(row.get(time_column))},
            ) from exc
        if ts_tier == "normalised":
            log.record("timestamp_format_normalised",
                       "Non-ISO timestamp formats were normalised.",
                       series=series, example=raw_time.strip())
        observations.append(Observation(timestamp, value, series))
    if rows and dropped / len(rows) > MAX_DROPPED_FRACTION:
        raise GnomonError(
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
            raise GnomonError("MIXED_TIMEZONES", "Timestamps mix timezone-aware and naive values.")
    return observations


def load_observations(
    input_path: str, time_column: str, target_column: str, series_column: str | None,
    *, repair: str = "off", repair_log: "RepairLog | None" = None,
) -> tuple[list[Observation], str, list[str]]:
    path = Path(input_path).expanduser().resolve()
    if not path.is_file():
        raise GnomonError("INPUT_NOT_FOUND", f"Input file does not exist: {path}")
    from .repair import RepairLog
    log = repair_log if repair_log is not None else RepairLog()
    rows, columns = _read_rows(path, time_column, target_column, repair, log)
    observations = observations_from_rows(
        rows, columns, time_column, target_column, series_column,
        repair=repair, repair_log=log,
    )
    return observations, fingerprint(path), columns


def read_input_rows(
    input_path: str, time_column: str, target_column: str,
    *, repair: str = "off", repair_log: "RepairLog | None" = None,
) -> tuple[list[dict[str, object]], list[str], str]:
    """One shared read of a file's rows, for callers that will extract
    several target columns from the same table. Returns (rows, columns,
    fingerprint); pair with :func:`observations_from_rows` per target."""
    path = Path(input_path).expanduser().resolve()
    if not path.is_file():
        raise GnomonError("INPUT_NOT_FOUND", f"Input file does not exist: {path}")
    from .repair import RepairLog
    log = repair_log if repair_log is not None else RepairLog()
    rows, columns = _read_rows(path, time_column, target_column, repair, log)
    return rows, columns, fingerprint(path)


def observations_from_rows(
    rows: list[dict[str, object]],
    columns: list[str],
    time_column: str,
    target_column: str,
    series_column: str | None,
    *,
    repair: str = "off",
    repair_log: "RepairLog | None" = None,
    default_series: str = "__default__",
) -> list[Observation]:
    """Extract one target column's observations from already-read rows —
    the tail of :func:`load_observations`, shared so a multi-target run
    reads the file once and parses each column through identical code."""
    from .repair import RepairLog
    log = repair_log if repair_log is not None else RepairLog()
    required = [time_column, target_column] + ([series_column] if series_column else [])
    missing = [column for column in required if column not in columns]
    if missing:
        raise GnomonError(
            "MISSING_COLUMNS", f"Required columns are missing: {', '.join(missing)}",
            {"available_columns": columns, "missing_columns": missing},
        )
    if repair != "off":
        observations = _load_lenient(
            rows, time_column, target_column, series_column, repair, log,
            default_series,
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
            series = str(row[series_column]) if series_column else default_series
            observations.append(
                Observation(_parse_timestamp(row[time_column], row_number), value, series)
            )
    if not observations:
        raise GnomonError("EMPTY_DATASET", "The input contains no observations.")
    return observations


def infer_schema_columns(path_raw: str) -> dict[str, object]:
    """Guess the time and target columns when the schema is unambiguous.

    "Unambiguous" is strict on purpose: exactly one column whose values all
    parse as timestamps, and exactly one *other* column whose values all
    parse as numbers. Anything else returns candidates and no choice, so
    the caller can say what it could not decide rather than guessing.

    Returns ``{"time": str|None, "target": str|None, "time_candidates":
    [...], "target_candidates": [...]}``.
    """
    from .repair import RepairLog

    path = Path(path_raw).expanduser().resolve()
    if not path.is_file():
        raise GnomonError("INPUT_NOT_FOUND", f"Input file does not exist: {path}")
    # Inference runs before any column is named, so delimiter detection
    # cannot key on the mapped columns. Take whichever separator yields the
    # most header fields — for a single-column result there is nothing to
    # infer either way.
    rows, columns = _read_rows(path, "", "", repair="off", log=RepairLog())
    if len(columns) <= 1 and path.suffix.lower() in {".csv", ".txt", ""}:
        best = (len(columns), rows, columns)
        for delimiter in (";", "\t", "|"):
            try:
                candidate_rows, candidate_columns = _read_delimited(path, delimiter)
            except Exception:
                continue
            if len(candidate_columns) > best[0]:
                best = (len(candidate_columns), candidate_rows, candidate_columns)
        _, rows, columns = best
    sample = rows[: min(len(rows), 50)]
    if not sample:
        raise GnomonError("EMPTY_DATASET", "The input contains no observations.")

    time_candidates: list[str] = []
    numeric_candidates: list[str] = []
    for column in columns:
        values = [row.get(column) for row in sample]
        if all(_parses_as_timestamp(value) for value in values):
            time_candidates.append(column)
        elif all(_parses_as_number(value) for value in values):
            numeric_candidates.append(column)
    return {
        "time": time_candidates[0] if len(time_candidates) == 1 else None,
        "target": numeric_candidates[0] if len(numeric_candidates) == 1 else None,
        "time_candidates": time_candidates,
        "target_candidates": numeric_candidates,
    }


def resolve_target_spec(
    input_path: str,
    spec: str,
    *,
    time_column: str | None = None,
    series_column: str | None = None,
) -> list[str]:
    """Expand a ``--target`` spec into concrete column names.

    ``auto`` becomes every numeric non-time column of the file; a comma
    list becomes the named columns in the order given; anything else is a
    single column name, passed through untouched. Shared by the CLI and
    the MCP tool so both surfaces expand identically.
    """
    if spec.strip().lower() == "auto":
        inferred = infer_schema_columns(input_path)
        excluded = {time_column, series_column} | set(inferred["time_candidates"])
        targets = [
            column for column in inferred["target_candidates"]
            if column not in excluded
        ]
        if not targets:
            raise GnomonError(
                "AMBIGUOUS_SCHEMA",
                "--target auto found no numeric non-time column to forecast.",
                {"argument": "--target",
                 "time_candidates": inferred["time_candidates"],
                 "target_candidates": inferred["target_candidates"]},
                repair_options=[{
                    "action": "supply_arguments",
                    "description": "Name the target column(s) explicitly with --target.",
                    "arguments": ["--target"],
                }],
            )
        return targets
    targets = [item.strip() for item in spec.split(",") if item.strip()]
    if len(targets) != len(set(targets)):
        duplicates = sorted({name for name in targets if targets.count(name) > 1})
        raise GnomonError(
            "INVALID_ARGUMENTS",
            f"--target names the same column more than once: {', '.join(duplicates)}.",
            {"duplicates": duplicates},
        )
    if not targets:
        raise GnomonError(
            "INVALID_ARGUMENTS", "--target was supplied but names no column.",
            {"supplied": spec},
        )
    return targets


def _read_delimited(path: Path, delimiter: str) -> tuple[list[dict[str, object]], list[str]]:
    import io
    text = path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter)
    return list(reader), list(reader.fieldnames or [])


def _parses_as_timestamp(value: object) -> bool:
    if value is None or str(value).strip() == "":
        return False
    try:
        _parse_timestamp(value, 0)
        return True
    except Exception:
        return False


def _parses_as_number(value: object) -> bool:
    if value is None or str(value).strip() == "":
        return False
    try:
        float(str(value).strip().replace(",", "."))
        return True
    except (TypeError, ValueError):
        return False


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


from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .contracts import HeadwaterError


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
        raise HeadwaterError(
            "INVALID_TIMESTAMP", f"Cannot parse timestamp on row {row}: {raw!r}",
            {"row": row, "value": str(raw)},
        ) from exc
    return parsed


def _rows_from_parquet(path: Path) -> list[dict[str, object]]:
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as exc:
        raise HeadwaterError(
            "MISSING_OPTIONAL_DEPENDENCY",
            "Parquet input requires the 'parquet' extra.",
            {"install": "pip install 'headwater-forecast[parquet]'"},
        ) from exc
    return parquet.read_table(path).to_pylist()


def load_observations(
    input_path: str, time_column: str, target_column: str, series_column: str | None
) -> tuple[list[Observation], str, list[str]]:
    path = Path(input_path).expanduser().resolve()
    if not path.is_file():
        raise HeadwaterError("INPUT_NOT_FOUND", f"Input file does not exist: {path}")
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
        raise HeadwaterError("UNSUPPORTED_INPUT", "Only CSV and Parquet inputs are supported.")

    required = [time_column, target_column] + ([series_column] if series_column else [])
    missing = [column for column in required if column not in columns]
    if missing:
        raise HeadwaterError(
            "MISSING_COLUMNS", f"Required columns are missing: {', '.join(missing)}",
            {"available_columns": columns, "missing_columns": missing},
        )
    observations: list[Observation] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            value = float(row[target_column])
        except (TypeError, ValueError) as exc:
            raise HeadwaterError(
                "INVALID_TARGET", f"Target is not numeric on row {row_number}.",
                {"row": row_number, "value": row.get(target_column)},
            ) from exc
        series = str(row[series_column]) if series_column else "__default__"
        observations.append(
            Observation(_parse_timestamp(row[time_column], row_number), value, series)
        )
    if not observations:
        raise HeadwaterError("EMPTY_DATASET", "The input contains no observations.")
    return observations, fingerprint(path), columns


def timezone_name(values: list[datetime]) -> str | None:
    aware = [value.utcoffset() is not None for value in values]
    if any(aware) and not all(aware):
        raise HeadwaterError("MIXED_TIMEZONES", "Timestamps mix timezone-aware and naive values.")
    if not any(aware):
        return None
    offsets = {value.utcoffset() for value in values}
    if len(offsets) > 1:
        return "variable-offset"
    offset = next(iter(offsets))
    if offset == timezone.utc.utcoffset(None):
        return "UTC"
    return str(values[0].tzinfo)


"""Portable JSON snapshots, with explicit persistence and no executable payloads."""

from dataclasses import asdict
from datetime import datetime
import json
import hashlib
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from zoneinfo import ZoneInfo

from .contracts import DataSchema, GnomonError
from .data import Observation
from .datasets import LoadedDataset
from .ids import canonical_json
from .temporal import validate_and_group
from .temporal_store import Snapshot, TemporalObservation

MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024


def write_json(path, payload):
    """Replace one explicitly named output atomically; do not leave partial JSON."""
    destination = Path(path).expanduser().resolve()
    text = json.dumps(payload, indent=2, allow_nan=False)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                prefix=".gnomon-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text + "\n")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return str(destination)


def save_snapshot(frozen, path):
    if Path(path).suffix != ".gnomon":
        raise GnomonError("INVALID_ARGUMENTS", "Snapshot paths must end in .gnomon; reuse them with --input.")
    loaded, snapshot = frozen.loaded, frozen.loaded.snapshot
    rows = []
    for row in snapshot.retained_observations():
        values = asdict(row)
        for key in ("valid_time", "known_time", "recorded_at"):
            stamp = getattr(row, key)
            values[key] = stamp.isoformat() if stamp else None
        # Preserve named calendar zones across a daylight-saving transition.
        values["valid_time_zone"] = getattr(row.valid_time.tzinfo, "key", None)
        rows.append(values)
    payload = {"format": "gnomon.snapshot/1", "schema": asdict(loaded.schema),
               "source_fingerprint": loaded.source_fingerprint, "columns": loaded.columns,
               "variable": loaded.variable, "unit": frozen.unit, "repairs": list(frozen.repairs),
               "snapshot": {"as_of": snapshot.as_of.isoformat() if snapshot.as_of else None,
                            "recorded_as_of": snapshot.recorded_as_of.isoformat() if snapshot.recorded_as_of else None,
                            "source_ref": snapshot.source_ref, "assumed_known_time": snapshot.assumed_known_time,
                            "known_time_provenance": snapshot.known_time_provenance,
                            "unknown_recorded_times": snapshot.unknown_recorded_times,
                            "rows": rows}}
    payload["checksum"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    if len(json.dumps(payload, indent=2).encode("utf-8")) + 1 > MAX_SNAPSHOT_BYTES:
        raise GnomonError("INPUT_TOO_LARGE", "Snapshot exceeds the 64 MiB portable-file limit.")
    return write_json(path, payload)


def load_snapshot(path, max_rows):
    with Path(path).expanduser().open("rb") as handle:
        raw = handle.read(MAX_SNAPSHOT_BYTES + 1)
    if len(raw) > MAX_SNAPSHOT_BYTES:
        raise GnomonError("INPUT_TOO_LARGE", "Snapshot exceeds the 64 MiB portable-file limit.")
    try:
        payload = json.loads(raw)
        checksum = payload.pop("checksum")
        if payload["format"] != "gnomon.snapshot/1" or checksum != hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest():
            raise ValueError("invalid version or checksum")
        spec = payload["snapshot"]
        if not isinstance(spec["rows"], list) or type(spec["assumed_known_time"]) is not bool:
            raise ValueError("invalid snapshot metadata")
        if len(spec["rows"]) > max_rows:
            raise GnomonError("INVALID_ARGUMENTS", "Input exceeds this session's retained observation limit.")
        schema = DataSchema(**payload["schema"])
        for text in (schema.time_column, schema.target_column, schema.frequency, payload["variable"],
                     payload["source_fingerprint"], spec["known_time_provenance"]):
            if not isinstance(text, str) or not text:
                raise ValueError("invalid schema")
        if payload["unit"] is not None and (not isinstance(payload["unit"], str) or not payload["unit"].strip()):
            raise ValueError("invalid unit")
        if not isinstance(payload["columns"], list) or any(not isinstance(column, str) for column in payload["columns"]):
            raise ValueError("invalid columns")
        observations = []
        for values in spec["rows"]:
            values = dict(values)
            zone = values.pop("valid_time_zone")
            for key in ("valid_time", "known_time", "recorded_at"):
                values[key] = datetime.fromisoformat(values[key]) if values[key] is not None else None
            if zone:
                if values["valid_time"].tzinfo is None:
                    raise ValueError("named zone requires an aware timestamp")
                values["valid_time"] = values["valid_time"].astimezone(ZoneInfo(zone))
            row = TemporalObservation(**values)
            if (type(row.value) not in (float, int) or not math.isfinite(row.value)
                    or row.valid_time is None or row.known_time is None
                    or type(row.revision) is not int or row.revision < 0
                    or not isinstance(row.entity, str) or not isinstance(row.variable, str)):
                raise ValueError("invalid observation")
            observations.append(row)
        kwargs = {key: spec[key] for key in ("source_ref", "assumed_known_time", "known_time_provenance")}
        snapshot = Snapshot(observations, datetime.fromisoformat(spec["as_of"]) if spec["as_of"] else None,
                            recorded_as_of=datetime.fromisoformat(spec["recorded_as_of"]) if spec["recorded_as_of"] else None,
                            **kwargs)
        unknown = spec["unknown_recorded_times"]
        if type(unknown) is not int or unknown < snapshot.unknown_recorded_times:
            raise ValueError("invalid recording provenance")
        snapshot.unknown_recorded_times = unknown
        rows = [Observation(row.valid_time, row.value, entity) for entity in snapshot.entities()
                for row in snapshot.series(entity, payload["variable"])]
        if not rows:
            raise ValueError("empty snapshot")
        groups, frequency, timezone = validate_and_group(rows, schema.frequency)
        repairs = payload["repairs"]
        if not isinstance(repairs, list) or any(not isinstance(r, dict) or not isinstance(r.get("code"), str) for r in repairs):
            raise ValueError("invalid repairs")
        loaded = LoadedDataset(payload["source_fingerprint"], payload["columns"], groups,
                               frequency, timezone, schema, snapshot, payload["variable"])
        return loaded, payload["unit"], tuple(repairs)
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        raise GnomonError("INVALID_SNAPSHOT", "Invalid or modified snapshot file. Export it again with inspect --save-snapshot.") from None

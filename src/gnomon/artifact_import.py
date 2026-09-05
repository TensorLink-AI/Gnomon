"""Read-only import of forecast artifacts, with explicit provenance gaps."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from .forecast_adapter import ForecastAdapterError, ForecastRequest


def read_forecast_import(path: str, *, project: str | None = None, naive_timezone: str | None = None):
    from .artifacts import verify_artifact_integrity
    if naive_timezone not in (None, "UTC"):
        raise ForecastAdapterError("legacy naive timestamps require an explicit UTC binding or remain unresolved")
    directory = Path(path).expanduser().resolve()
    manifest = verify_artifact_integrity(directory)

    def read(name):
        raw = (directory / name).read_bytes()
        if manifest and manifest["files"].get(name) != "sha256:" + hashlib.sha256(raw).hexdigest():
            raise ForecastAdapterError(f"import file is not bound to the integrity manifest: {name}")
        return raw, json.loads(raw)

    raw, artifact = read("artifact.json")
    history_raw, history = read("history.json") if (directory / "history.json").is_file() else (b"", {})
    if not isinstance(artifact, dict) or not isinstance(artifact.get("task"), dict) or not isinstance(artifact.get("results"), list):
        raise ForecastAdapterError("import requires a forecast artifact")
    source_id = hashlib.sha256(raw + b"\0" + history_raw + json.dumps([project, naive_timezone]).encode()).hexdigest()

    def stamp(value):
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None and naive_timezone == "UTC":
            return parsed.replace(tzinfo=timezone.utc).isoformat()
        return value

    weights = {}
    for item in artifact.get("evidence", []):
        if item.get("kind") == "model_weights":
            weights.update(item["payload"].get("pinned_revisions", {}))
    records = []
    for series in artifact["results"]:
        rows = series.get("forecast") or []
        if not rows:
            continue
        name = series["series"]
        series_id = name if project is None else quote(project, safe="") + ":" + quote(name, safe="")
        past = history.get("series", {}).get(name) or []
        timestamps = [stamp(row["timestamp"]) for row in past]
        future = [stamp(row["timestamp"]) for row in rows]
        ForecastRequest._validate_times("future_timestamps", tuple(future), len(rows))
        point = [float(row["point"]) for row in rows]
        if not all(math.isfinite(v) for v in point):
            raise ForecastAdapterError("imported forecast must be finite")
        qs = sorted({int(key[1:]) / 100 for row in rows for key in row
                     if key.startswith("q") and key[1:].isdigit() and 0 < int(key[1:]) < 100})
        quantiles = [{q: float(row[f"q{round(100 * q):02d}"]) for q in qs} for row in rows] if qs else None
        if quantiles and any(not all(math.isfinite(v) for v in row.values())
                             or list(row.values()) != sorted(row.values()) for row in quantiles):
            raise ForecastAdapterError("imported quantiles must be finite and monotone")
        request = {
            "history": [float(row["value"]) for row in past] if past else None,
            "horizon": len(rows), "season": 1, "quantiles": qs,
            "timestamps": timestamps, "future_timestamps": future,
            "series_id": series_id, "unit": None,
            "frequency": artifact["task"].get("schema", {}).get("frequency"),
            "cutoff": timestamps[-1] if timestamps else None,
            "known_time_cutoff": artifact["task"].get("as_of"),
            "snapshot_id": history.get("snapshot", {}).get("snapshot_id"),
            "recorded_time_cutoff": history.get("snapshot", {}).get("recorded_as_of"),
            "past_covariates": [], "future_covariates": [], "related_series": [],
            "past_covariate_names": [], "future_covariate_names": [], "samples": 0,
        }
        for key in ("known_time_cutoff", "recorded_time_cutoff"):
            if request[key]:
                request[key] = stamp(request[key])
        if past:
            request = asdict(ForecastRequest.from_dict(request))
        identity = weights.get(series.get("selected_model"), {})
        revision = json.dumps(identity, sort_keys=True) if isinstance(identity, dict) and identity else None
        model = series.get("selected_model") or "unknown"
        records.append({
            "execution_id": str(uuid4()), "fingerprint": source_id + ":" + quote(series_id, safe=""),
            "provider": model, "revision": revision, "request": request,
            "result": {"point": point, "quantiles": quantiles, "timestamps": future,
                       "series_id": series_id, "unit": None, "sample_paths": None,
                       "metadata": {"execution_kind": "artifact_import", "artifact_id": artifact.get("forecast_id"),
                                    "source_created_at": artifact.get("created_at"), "artifact_path": str(directory),
                                    "integrity": "verified" if manifest else "legacy_unsealed",
                                    "history_preserved": bool(past), "naive_timezone_binding": naive_timezone,
                                    "complete_request_reconstructed": False,
                                    "weights_identity": identity,
                                    "original_series": name, "project": project,
                                    "legacy_support": series.get("support"),
                                    "limitations": ["No model was rerun; missing original inputs/units/covariates are not reconstructed."]}},
            "cache_hit": False, "evidence": "imported_forecast", "action_authorized": False,
        })
    return source_id, records

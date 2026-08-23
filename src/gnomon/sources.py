"""Explicit CLI ingestion boundaries for stdin and Prometheus."""

from __future__ import annotations

import atexit
import csv
import json
import os
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, TextIO

from .contracts import GnomonError


MAX_SOURCE_BYTES = 10 * 1024 * 1024


def _temporary_csv() -> Path:
    descriptor, name = tempfile.mkstemp(prefix="gnomon-source-", suffix=".csv")
    os.close(descriptor)
    path = Path(name)
    atexit.register(path.unlink, missing_ok=True)
    return path


def materialize_stdin(stream: TextIO | None = None) -> Path:
    stream = stream or sys.stdin
    content = stream.read(MAX_SOURCE_BYTES + 1)
    if len(content.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise GnomonError(
            "SOURCE_TOO_LARGE", f"stdin exceeds the {MAX_SOURCE_BYTES}-byte limit."
        )
    if not content.strip():
        raise GnomonError("EMPTY_DATASET", "stdin contained no data.")
    path = _temporary_csv()
    path.write_text(content, encoding="utf-8")
    return path


def _prometheus_url(reference: str) -> str:
    if reference.startswith("prom+https://"):
        return "https://" + reference[len("prom+https://"):]
    if reference.startswith("prom+http://"):
        return "http://" + reference[len("prom+http://"):]
    if reference.startswith("prom://"):
        return "http://" + reference[len("prom://"):]
    raise GnomonError("INVALID_PROMETHEUS_SOURCE", "Unsupported Prometheus source URI.")


def materialize_prometheus(
    reference: str, *, timeout: float = 30.0,
    allowed_hosts: set[str] | None = None,
) -> Path:
    url = _prometheus_url(reference)
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise GnomonError("INVALID_PROMETHEUS_SOURCE", "Prometheus URI needs an HTTP(S) host.")
    if parsed.username or parsed.password or parsed.fragment:
        raise GnomonError(
            "INVALID_PROMETHEUS_SOURCE",
            "Do not put credentials or fragments in a Prometheus URI; use GNOMON_PROMETHEUS_BEARER_TOKEN.",
        )
    if allowed_hosts is not None and parsed.hostname not in allowed_hosts:
        raise GnomonError(
            "PROMETHEUS_HOST_NOT_ALLOWED",
            f"Prometheus host {parsed.hostname!r} is not in the governed "
            "connector allowlist.",
            {"host": parsed.hostname, "allowed_hosts": sorted(allowed_hosts)},
        )
    if parsed.path.rstrip("/") != "/api/v1/query_range":
        raise GnomonError(
            "INVALID_PROMETHEUS_SOURCE",
            "The source path must be /api/v1/query_range.",
        )
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    missing = [name for name in ("query", "start", "end", "step")
               if len(query.get(name, [])) != 1 or not query[name][0]]
    if missing:
        raise GnomonError(
            "INVALID_PROMETHEUS_SOURCE",
            "A range source requires exactly one query, start, end, and step parameter.",
            {"missing_or_repeated": missing},
        )
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    token = os.environ.get("GNOMON_PROMETHEUS_BEARER_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_SOURCE_BYTES + 1)
    except Exception as exc:
        raise GnomonError(
            "PROMETHEUS_REQUEST_FAILED", f"Prometheus range query failed: {exc}",
            {"host": parsed.hostname, "retryable": True},
        ) from exc
    if len(raw) > MAX_SOURCE_BYTES:
        raise GnomonError("SOURCE_TOO_LARGE", "Prometheus response exceeds 10 MiB.")
    try:
        payload: dict[str, Any] = json.loads(raw)
        if payload.get("status") != "success":
            raise ValueError(str(payload.get("error") or "status was not success"))
        result = payload["data"]["result"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GnomonError(
            "INVALID_PROMETHEUS_RESPONSE", f"Invalid Prometheus response: {exc}"
        ) from exc
    if not isinstance(result, list) or not result:
        raise GnomonError("EMPTY_DATASET", "Prometheus returned no series.")
    rows: list[tuple[str, str, float]] = []
    for index, series in enumerate(result):
        if not isinstance(series, dict) or not isinstance(series.get("values"), list):
            raise GnomonError("INVALID_PROMETHEUS_RESPONSE", "Expected matrix range-query data.")
        labels = series.get("metric") or {}
        identity = ",".join(f"{key}={labels[key]}" for key in sorted(labels)) or f"series-{index}"
        for sample in series["values"]:
            try:
                timestamp, value = sample
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise GnomonError("INVALID_PROMETHEUS_RESPONSE", "Malformed matrix sample.") from exc
            from datetime import datetime, timezone
            valid_time = datetime.fromtimestamp(float(timestamp), timezone.utc).isoformat()
            rows.append((valid_time, identity, numeric))
    path = _temporary_csv()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "series", "value"])
        writer.writerows(rows)
    return path


def materialize_cli_source(reference: str) -> tuple[str, str | None]:
    """Resolve explicit CLI-only sources; MCP never invokes this function."""
    if reference == "-":
        return str(materialize_stdin()), "stdin"
    if reference.startswith(("prom://", "prom+http://", "prom+https://")):
        return str(materialize_prometheus(reference)), "prometheus"
    return reference, None


def materialize_agent_source(reference: str) -> tuple[str, str | None]:
    """Resolve read-only agent connectors under an explicit host allowlist."""
    if not reference.startswith(("prom://", "prom+http://", "prom+https://")):
        return reference, None
    configured = os.environ.get("GNOMON_PROMETHEUS_ALLOWED_HOSTS", "")
    allowed = {item.strip() for item in configured.split(",") if item.strip()}
    allowed.update({"localhost", "127.0.0.1", "::1"})
    return str(materialize_prometheus(
        reference, allowed_hosts=allowed)), "prometheus"

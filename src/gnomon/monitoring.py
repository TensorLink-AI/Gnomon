"""Durable, idempotent delivery for immutable monitor artifacts."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import GnomonError


def default_state_path(output: str | Path) -> Path:
    """Keep the monitor ledger beside its required writable artifact store."""
    return Path(output).expanduser().resolve() / ".monitor-events.json"


def firing_events(payload: dict[str, Any], artifact_path: str) -> list[dict[str, Any]]:
    events = []
    for trigger in payload.get("triggers") or []:
        if not isinstance(trigger, dict) or not trigger.get("armed"):
            continue
        if trigger.get("first_alert_step") is None:
            continue
        identity = {
            "monitor_id": payload.get("monitor_id"), "series": trigger.get("series"),
            "first_alert_timestamp": trigger.get("first_alert_timestamp"),
            "threshold": (trigger.get("trigger") or {}).get("threshold"),
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        from .support import trigger_support_tier
        tier_floor = trigger_support_tier(trigger)
        events.append({
            "schema_version": "0.1",
            "event_id": "monitor-event-" + hashlib.sha256(encoded.encode()).hexdigest()[:24],
            "kind": "gnomon.threshold_breach_predicted",
            "created_at": payload.get("created_at"),
            "artifact_id": payload.get("monitor_id"),
            "artifact_path": artifact_path,
            "tier_floor": tier_floor,
            "trigger": trigger,
        })
    return events


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": "0.1", "events": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GnomonError("MONITOR_STATE_INVALID", f"Cannot read monitor state: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), dict):
        raise GnomonError("MONITOR_STATE_INVALID", "Monitor state has an invalid shape.")
    return payload


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".gnomon-monitor-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def deliver_events(
    events: list[dict[str, Any]], *, state_path: str | Path,
    webhook: str | None = None, secret_env: str | None = None,
    timeout: float = 10.0, max_attempts: int = 3,
) -> dict[str, Any]:
    path = Path(state_path).expanduser().resolve()
    state = _load_state(path)
    delivered, existing, failed = [], [], []
    secret = os.environ.get(secret_env) if secret_env else None
    if secret_env and secret is None:
        raise GnomonError("WEBHOOK_SECRET_MISSING", f"Environment variable {secret_env} is not set.")
    if webhook:
        parsed = urllib.parse.urlsplit(webhook)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
            raise GnomonError("INVALID_WEBHOOK", "Webhook must be an HTTP(S) URL without embedded credentials.")
    for event in events:
        event_id = str(event["event_id"])
        previous = state["events"].get(event_id)
        if previous is not None and (
                previous.get("status") == "delivered" or not webhook):
            existing.append(event_id)
            continue
        record = previous or {
            "event": event, "status": "recorded", "attempts": 0,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        if previous is None:
            state["events"][event_id] = record
            _write_state(path, state)  # durable before side effect
        if webhook:
            body = json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
            headers = {"Content-Type": "application/json", "Idempotency-Key": event_id}
            if secret is not None:
                headers["X-Gnomon-Signature-SHA256"] = hmac.new(
                    secret.encode(), body, hashlib.sha256).hexdigest()
            last_error: Exception | None = None
            for _ in range(max(1, max_attempts)):
                record["attempts"] = int(record.get("attempts", 0)) + 1
                try:
                    request = urllib.request.Request(
                        webhook, data=body, headers=headers, method="POST")
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        status = int(getattr(response, "status", 200))
                    if not 200 <= status < 300:
                        raise OSError(f"HTTP {status}")
                    record["status"] = "delivered"
                    record["delivered_at"] = datetime.now(timezone.utc).isoformat()
                    record.pop("last_error", None)
                    delivered.append(event_id)
                    break
                except Exception as exc:
                    last_error = exc
            if record["status"] != "delivered":
                record["status"] = "delivery_failed"
                record["last_error"] = f"{type(last_error).__name__}: {last_error}"
                failed.append(event_id)
            _write_state(path, state)
        else:
            delivered.append(event_id)
    return {"state_path": str(path), "new_events": len(delivered) + len(failed),
            "delivered": delivered, "already_recorded": existing,
            "delivery_failed": failed}


def prometheus_rule(
    *, monitor_id: str, expression: str, threshold: float,
    output: str | Path,
) -> Path:
    if not expression.strip() or "\n" in expression:
        raise GnomonError("INVALID_PROMETHEUS_EXPRESSION", "Expression must be one non-empty line.")
    safe_name = re.sub(r"[^A-Za-z0-9_]", "_", monitor_id)
    # Emit ordinary block YAML rather than relying on JSON's technical
    # status as a YAML 1.2 subset. Operators inspect and edit this file, and
    # a `.yaml` path beginning with `{` looks like the wrong artifact even
    # when Prometheus can parse it. JSON quoting remains a dependency-free,
    # standards-compatible scalar encoder.
    def scalar(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    rendered = "\n".join([
        "groups:",
        "  - name: gnomon",
        "    rules:",
        f"      - alert: {scalar(f'Gnomon_{safe_name}')}",
        f"        expr: {scalar(f'({expression}) > {threshold:.17g}')}",
        "        labels:",
        "          source: gnomon",
        f"          monitor_id: {scalar(monitor_id)}",
        "        annotations:",
        "          summary: " + scalar(
            f"Observed value exceeds Gnomon threshold {threshold:.6g}"),
        "          description: " + scalar(
            "Static observed-value rule exported from an immutable Gnomon "
            "monitor. Forecast probability remains in its artifact."),
        "",
    ])
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return path

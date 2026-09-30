"""Hermes plugin: append one row per successful LLM call to a local SQLite file.

Stores token counts, model and timing only; never prompts, replies or keys.
The hook is observer-only: every failure is swallowed so Hermes is never
slowed or broken by tracking. Missed calls show up in the monitor-token-usage
skill's daily coverage check against Hermes state.db.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

NAME = "token-tracker"
SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ended_at REAL NOT NULL,
    started_at REAL,
    session_id TEXT,
    platform TEXT,
    model TEXT,
    provider TEXT,
    base_url TEXT,
    api_request_id TEXT UNIQUE,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    usage_missing INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS calls_ended_at ON calls(ended_at);
"""
_TOKENS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")


def db_path() -> Path:
    """<hermes home>/plugin-data/token-tracker/calls.db, following the active profile."""
    try:
        from plugins.plugin_storage import plugin_data_dir
        return plugin_data_dir(NAME) / "calls.db"
    except Exception:  # older Hermes or tests: same layout under HERMES_HOME
        home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
        root = home / "plugin-data" / NAME
        root.mkdir(parents=True, exist_ok=True)
        return root / "calls.db"


_BUSY_SECONDS = 5.0
_ATTEMPTS = 4
_prepared: set[Path] = set()
_prepare_lock = threading.Lock()


def _prepare(path):
    """Switch to WAL and create the schema once per database per process.

    Doing this on every call made each write take schema and journal locks, so
    concurrent calls could fail with "database is locked" and be dropped.
    """
    with _prepare_lock:
        if path in _prepared:
            return
        conn = sqlite3.connect(path, timeout=_BUSY_SECONDS)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
        finally:
            conn.close()
        _prepared.add(path)


def _connect():
    # A connection per call: safe across threads, forks and profile switches.
    path = db_path()
    _prepare(path)
    return sqlite3.connect(path, timeout=_BUSY_SECONDS)


def _insert(row):
    """Insert with a short retry on lock contention; raises only if it keeps failing."""
    for attempt in range(_ATTEMPTS):
        conn = _connect()
        try:
            with conn:
                conn.execute(
                    "INSERT OR IGNORE INTO calls (ended_at, started_at, session_id, platform, model, provider,"
                    " base_url, api_request_id, input_tokens, output_tokens, cache_read_tokens,"
                    " cache_write_tokens, reasoning_tokens, usage_missing)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
            return
        except sqlite3.OperationalError as error:
            if "no such table" in str(error):  # the file was removed or replaced: prepare it again
                with _prepare_lock:
                    _prepared.discard(db_path())
            elif not any(w in str(error) for w in ("locked", "busy")):
                raise
            if attempt + 1 == _ATTEMPTS:
                raise
            time.sleep(0.05 * 2 ** attempt)
        finally:
            conn.close()


def _int(value):
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def record_call(*, ended_at=None, started_at=None, session_id=None, platform=None, model=None,
                provider=None, base_url=None, api_request_id=None, usage=None):
    """post_api_request observer; never raises.

    Named parameters only: Hermes passes a narrow callback just the fields it
    declares, so response and message content never reach this plugin.
    """
    try:
        if not isinstance(ended_at, (int, float)):
            return
        usage = usage if isinstance(usage, dict) else None
        text = lambda v: str(v) if v not in (None, "") else None
        row = (float(ended_at), started_at if isinstance(started_at, (int, float)) else None,
               text(session_id), text(platform), text(model), text(provider), text(base_url),
               text(api_request_id), *(_int((usage or {}).get(k)) for k in _TOKENS), 0 if usage else 1)
        _insert(row)
    except Exception:
        logger.debug("token-tracker: call not recorded", exc_info=True)


def register(ctx):
    ctx.register_hook("post_api_request", record_call)

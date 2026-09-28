"""End-to-end checks for skills/monitor-token-usage against a synthetic Hermes state.db."""
import importlib.util
import inspect
import threading
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import pytest

from gnomon import GnomonSession, TemporalLedger
from gnomon.ids import FixedClock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/monitor-token-usage/scripts/token_usage.py"
spec = importlib.util.spec_from_file_location("token_usage", SCRIPT)
tu = importlib.util.module_from_spec(spec)
sys.modules["token_usage"] = tu  # dataclasses resolve annotations through sys.modules
spec.loader.exec_module(tu)
PLUGIN = ROOT / "skills/monitor-token-usage/hermes-plugin/token-tracker/__init__.py"


def load_plugin(home, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugin_spec = importlib.util.spec_from_file_location("token_tracker_plugin", PLUGIN)
    plugin = importlib.util.module_from_spec(plugin_spec)
    plugin_spec.loader.exec_module(plugin)
    hooks = {}

    class Ctx:
        def register_hook(self, name, callback):
            hooks[name] = callback
    plugin.register(Ctx())
    return plugin, hooks["post_api_request"]


def fire(callback, **payload):
    """Invoke like Hermes: a callback without **kwargs receives only the fields it declares."""
    params = inspect.signature(callback).parameters
    callback(**{k: v for k, v in payload.items() if k in params})


def hermes_payload(ended_at, tokens, *, request_id, session="s", model="m"):
    return dict(task_id="t", turn_id="u", api_request_id=request_id, session_id=session, platform="cli",
                model=model, provider="custom", base_url="https://api.engy.ai/v1", api_mode="chat",
                api_call_count=1, api_duration=1.5, started_at=ended_at - 1.5, ended_at=ended_at,
                finish_reason="stop", message_count=3, response_model=model,
                response={"content": "SECRET REPLY"}, assistant_message={"content": "SECRET REPLY"},
                usage={"input_tokens": tokens // 2, "output_tokens": tokens // 4,
                       "cache_read_tokens": tokens - tokens // 2 - tokens // 4, "cache_write_tokens": 0,
                       "reasoning_tokens": 0, "prompt_tokens": 0, "total_tokens": tokens},
                assistant_content_chars=12, assistant_tool_call_count=0)

FAKE = '''
from gnomon import ForecastResult
from gnomon.forecast_adapter import ForecastAdapterError
def forecast(req):
    if req.season != 1:  # same contract as Gnomon's Ephemeris adapter
        raise ForecastAdapterError("Ephemeris accepts a frequency hint, not an explicit seasonal period")
    base = sum(req.history[-7:]) / 7
    qs = tuple({q: base * (0.5 + q) for q in req.quantiles} for _ in range(req.horizon)) if req.quantiles else None
    return ForecastResult((base,) * req.horizon, qs, timestamps=req.future_timestamps,
                          series_id=req.series_id, unit=req.unit)
'''
CONFIG = 'schema_version = 1\n[providers.fake]\nkind = "callable"\nentrypoint = "fakeprov:forecast"\n' \
         'capabilities = {quantiles = true}\n'
SCHEMA = """
CREATE TABLE sessions (id TEXT PRIMARY KEY, started_at REAL NOT NULL, ended_at REAL, last_activity_at REAL);
CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, timestamp REAL NOT NULL);
CREATE TABLE session_model_usage (session_id TEXT, model TEXT, billing_base_url TEXT DEFAULT '', task TEXT DEFAULT '',
  input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0, cache_read_tokens INTEGER DEFAULT 0,
  cache_write_tokens INTEGER DEFAULT 0, estimated_cost_usd REAL DEFAULT 0, actual_cost_usd REAL DEFAULT 0,
  api_call_count INTEGER DEFAULT 0,
  first_seen REAL, last_seen REAL);
"""


def add_session(con, sid, times, tokens, *, model="m", base_url="https://api.engy.ai/v1", cost=0.0, calls=None):
    con.execute("INSERT INTO sessions VALUES (?,?,?,?)", (sid, times[0], times[-1], times[-1]))
    for t in times:
        con.execute("INSERT INTO messages(session_id, role, timestamp) VALUES (?,?,?)", (sid, "assistant", t))
        con.execute("INSERT INTO messages(session_id, role, timestamp) VALUES (?,?,?)", (sid, "user", t - 1))
    con.execute("INSERT INTO session_model_usage(session_id, model, billing_base_url, input_tokens, output_tokens,"
                " cache_read_tokens, estimated_cost_usd, api_call_count, first_seen, last_seen)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (sid, model, base_url, tokens // 2, tokens // 4, tokens - tokens // 2 - tokens // 4, cost,
                 len(times) if calls is None else calls, times[0], times[-1]))


def midnight(d):
    return datetime.combine(d, time(), timezone.utc).timestamp()


@pytest.fixture
def hermes(tmp_path, monkeypatch):
    """41 days of weekday-shaped usage; yesterday is a 4x spike from 4x more calls; today is partial."""
    (tmp_path / "fakeprov.py").write_text(FAKE)
    (tmp_path / "providers.toml").write_text(CONFIG)
    monkeypatch.syspath_prepend(str(tmp_path))
    now = datetime.now(timezone.utc)
    today = now.date()
    con = sqlite3.connect(tmp_path / "state.db")
    con.executescript(SCHEMA)
    for k in range(40, 0, -1):
        d = today - timedelta(days=k)
        per = 100_000 if d.weekday() >= 5 else 300_000
        spike = 4 if k == 1 else 1
        for j in range(3):
            add_session(con, f"s{k}-{j}", [midnight(d) + 3600 * (9 + j), midnight(d) + 3600 * (9 + j) + 60],
                        per * spike, cost=per * spike * 0.5e-6, calls=10 * spike)
    elapsed = (now - datetime.combine(today, time(), timezone.utc)).total_seconds()
    add_session(con, "today", [midnight(today) + min(60, elapsed / 2)], 50_000)
    con.commit()
    con.close()
    return tmp_path


def run(home, *args, provider="fake"):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(home)]), "HOME": str(home)}
    cmd = [sys.executable, str(SCRIPT), *args, "--db", str(home / "state.db"), "--data-dir", str(home / "data"),
           "--provider", provider, "--providers-config", str(home / "providers.toml"),
           "--gnomon", f"{sys.executable} -m gnomon"]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300, cwd=home)
    return proc.returncode, proc.stdout


def ledger_counts(home):
    con = sqlite3.connect(home / "data/ledger.db")
    try:
        return {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                for t in ("executions", "actuals", "evaluations")}
    finally:
        con.close()


def test_usage_is_split_by_message_time_across_midnight_and_models(tmp_path):
    db = tmp_path / "state.db"
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    d = datetime(2026, 9, 1, tzinfo=timezone.utc).date()
    times = [midnight(d) + 3600 * 23, midnight(d) + 3600 * 23 + 60,
             midnight(d) + 86400 + 60, midnight(d) + 86400 + 120]
    add_session(con, "a", times, 1000, cost=0.001)
    # Second model row in the same session covers only the after-midnight messages.
    con.execute("INSERT INTO session_model_usage(session_id, model, billing_base_url, input_tokens, first_seen,"
                " last_seen) VALUES ('a', 'other', 'https://elsewhere', 400, ?, ?)", (times[2], times[3]))
    con.commit()
    con.close()

    u = tu.read_usage(db, timezone.utc)
    assert u["tokens"] == {d: 500, d + timedelta(days=1): 900}
    assert u["calls"] == {d: 2, d + timedelta(days=1): 2}
    assert u["cost"][d] == pytest.approx(0.0005)
    assert u["unpriced"] == {d + timedelta(days=1): 400}
    assert u["models"]["other"] == {"tokens": 400, "calls": 0, "cost_usd": 0.0, "unpriced_tokens": 400}

    assert tu.read_usage(db, timezone.utc, "engy")["tokens"] == {d: 500, d + timedelta(days=1): 500}


def test_driver_separates_more_calls_from_bigger_calls():
    d = datetime(2026, 9, 10).date()
    week = [d - timedelta(days=k) for k in range(1, 8)]
    u = {"tokens": {x: 1000 for x in week}, "calls": {x: 10 for x in week}, "cache_read": {}}
    u["tokens"][d], u["calls"][d] = 4000, 40
    assert tu.driver(u, d) == "calls 40 vs 10/day avg (4.0×), tokens/call 100 vs 100 (1.0×)"
    u["tokens"][d], u["calls"][d] = 4000, 10
    assert tu.driver(u, d) == "calls 10 vs 10/day avg (1.0×), tokens/call 400 vs 100 (4.0×)"
    assert tu.driver(u, d, partial=True) == "10 calls so far; tokens/call 400 vs 100 (4.0×)"


def test_daily_series_excludes_partial_today_and_keeps_real_zero_days():
    d = datetime(2026, 9, 1).date()
    days, today = tu.daily_series({d: 5, d + timedelta(days=2): 7, d + timedelta(days=4): 9}, d + timedelta(days=4))
    assert days == [(d, 5), (d + timedelta(days=1), 0.0), (d + timedelta(days=2), 7), (d + timedelta(days=3), 0.0)]
    assert today == 9


def test_check_records_once_per_day_and_is_idempotent(hermes):
    code, out = run(hermes, "check")
    assert code == 0, out
    assert "today forecast" in out and "+40 actual(s)" in out
    assert ledger_counts(hermes) == {"executions": 2, "actuals": 40, "evaluations": 0}
    con = sqlite3.connect(hermes / "data/ledger.db")
    seasons = dict(con.execute("SELECT json_extract(payload_json, '$.provider'),"
                               " json_extract(payload_json, '$.request.season') FROM payloads").fetchall())
    con.close()
    assert seasons == {"fake": 1, "seasonal_naive": 7}  # models get only the frequency hint

    code, out = run(hermes, "check")
    assert code == 0, out
    assert "+0 actual(s)" in out
    assert ledger_counts(hermes)["executions"] == 2


def seed_history(home, origins):
    """Record model and baseline forecasts as if check had run during each following day."""
    tokens = tu.read_usage(home / "state.db", timezone.utc)["tokens"]
    days, _ = tu.daily_series(tokens, datetime.now(timezone.utc).date())
    for origin in origins:
        history = [row for row in days if row[0] <= origin]
        clock = FixedClock(tu.day_end(origin, timezone.utc) + timedelta(hours=9))
        ledger = TemporalLedger(home / "data/ledger.db", clock=clock)
        session = GnomonSession.from_config(home / "providers.toml", ledger=ledger)
        try:
            for provider, quantiles in (("fake", (tu.LOW, 0.5, tu.HIGH)), ("seasonal_naive", ())):
                request = tu.build_request(history, timezone.utc, tu.HORIZON, quantiles, "hermes/tokens",
                                           seasonal=provider == "seasonal_naive")
                assert session.forecast(provider, request)["status"] == "ok"
        finally:
            session.close()


def test_check_scores_matured_forecasts_compares_baseline_and_flags_surge_once(hermes):
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    seed_history(hermes, [yesterday - timedelta(days=k) for k in range(14, 0, -1)])

    code, out = run(hermes, "check")
    assert code == 0, out
    assert out.startswith("ALERT"), out
    assert f"SURGE {yesterday}" in out
    # The flat fake can also flag a weekend DIP depending on today's weekday; check the surge's own driver line.
    surge = out.split(f"SURGE {yesterday}", 1)[1].splitlines()[1]
    assert "calls 120 vs 30/day avg (4.0×), tokens/call" in surge, out
    assert "MAE than seasonal_naive over 8 paired 7-day forecast(s)" in out
    assert ledger_counts(hermes)["evaluations"] > 0

    code, out = run(hermes, "check")
    assert code == 0 and out.startswith("OK"), out
    code, out = run(hermes, "check", "--quiet")
    assert code == 0 and out == "", out


def test_report_and_json_disclose_projection_and_missing_provider(hermes):
    code, out = run(hermes, "report")
    assert code == 0, out
    assert "next 30 days" in out and "wider than a true 90% interval" in out
    assert "calls/day" in out and "tokens/call" in out and "cache-read" in out
    assert "partial day" in out

    code, out = run(hermes, "report", "--json", provider="not-registered")
    assert code == 0, out
    data = json.loads(out)
    assert "not-registered" in data["projection_error"] and "projection" not in data
    assert data["complete_days"] == 40 and len(data["history"]) == 40
    assert data["history"][-1]["calls"] == 120 and data["llm_calls"]["last_7_days"]["tokens_per_call"]


def test_short_history_waits_without_forecasting(tmp_path):
    con = sqlite3.connect(tmp_path / "state.db")
    con.executescript(SCHEMA)
    add_session(con, "one", [datetime.now(timezone.utc).timestamp() - 3 * 86400], 1000)
    con.commit()
    con.close()
    code, out = run(tmp_path, "check")
    assert code == 0 and "WAITING" in out, out
    assert not (tmp_path / "data/ledger.db").exists()


def test_plugin_records_counts_only_dedupes_and_never_raises(tmp_path, monkeypatch):
    plugin, hook = load_plugin(tmp_path, monkeypatch)
    fire(hook, **hermes_payload(1_800_000_000.0, 1000, request_id="r1"))
    fire(hook, **hermes_payload(1_800_000_000.0, 1000, request_id="r1"))  # retry of the same response
    fire(hook, **{**hermes_payload(1_800_000_060.0, 0, request_id="r2"), "usage": None})
    fire(hook, **{**hermes_payload(1_800_000_120.0, 0, request_id="r3"), "usage": {"input_tokens": "junk"}})
    fire(hook, **{**hermes_payload(0, 0, request_id="r4"), "ended_at": None})  # dropped, no exception

    db = tmp_path / "plugin-data/token-tracker/calls.db"
    raw = db.read_bytes()
    assert b"SECRET" not in raw
    con = sqlite3.connect(db)
    rows = con.execute("SELECT api_request_id, input_tokens + output_tokens + cache_read_tokens, usage_missing"
                       " FROM calls ORDER BY ended_at").fetchall()
    con.close()
    assert rows == [("r1", 1000, 0), ("r2", 0, 1), ("r3", 0, 0)]

    monkeypatch.setattr(plugin, "db_path", lambda: (_ for _ in ()).throw(OSError("disk full")))
    fire(hook, **hermes_payload(1_800_000_180.0, 5, request_id="r5"))  # swallowed


def test_plugin_is_safe_under_concurrent_writers(tmp_path, monkeypatch):
    _, hook = load_plugin(tmp_path, monkeypatch)
    def worker(n):
        for i in range(40):
            fire(hook, **hermes_payload(1_800_000_000.0 + n * 1000 + i, 10, request_id=f"{n}-{i}"))
    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    con = sqlite3.connect(tmp_path / "plugin-data/token-tracker/calls.db")
    assert con.execute("SELECT count(*) FROM calls").fetchone()[0] == 240
    con.close()


def track_calls(home, monkeypatch, *, keep=lambda i: True, since=0.0):
    """Replay each main-agent session in state.db through the plugin, one call per reply."""
    _, hook = load_plugin(home, monkeypatch)
    con = sqlite3.connect(home / "state.db")
    rows = con.execute("SELECT u.session_id, u.input_tokens + u.output_tokens + u.cache_read_tokens, u.model"
                       " FROM session_model_usage u WHERE u.task = ''").fetchall()
    for n, (sid, tokens, model) in enumerate(rows):
        stamps = [t for (t,) in con.execute("SELECT timestamp FROM messages WHERE session_id = ?"
                                            " AND role = 'assistant' ORDER BY timestamp", (sid,))]
        for i, t in enumerate(stamps):
            if keep(n) and t >= since:
                fire(hook, **hermes_payload(t, tokens // len(stamps), request_id=f"{sid}-{i}",
                                            session=sid, model=model))
    con.close()


def test_hours_and_coverage_from_tracked_calls(hermes, monkeypatch):
    con = sqlite3.connect(hermes / "state.db")
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    con.execute("INSERT INTO session_model_usage(session_id, model, task, input_tokens, first_seen, last_seen)"
                " VALUES ('s1-0', 'aux-model', 'compression', 5000, ?, ?)",
                (midnight(yesterday) + 9 * 3600, midnight(yesterday) + 9 * 3600 + 60))
    con.commit()
    con.close()
    track_calls(hermes, monkeypatch)

    code, out = run(hermes, "hours", "--hours", "48")
    assert code == 0, out
    assert "exact per-call records" in out and "biggest calls in this window" in out
    assert "captured 100% of main-agent tokens over the last 24h" in out
    assert "side tasks (compression, titles, ...) are daily-only" in out

    code, out = run(hermes, "hours", "--json")
    data = json.loads(out)
    assert len(data["hours"]) == 24 and data["coverage"]["ratio"] == pytest.approx(1.0, abs=0.01)

    code, out = run(hermes, "report")
    assert "last 24 hours" in out and "captured 100%" in out


def test_check_alerts_once_when_tracker_misses_calls(hermes, monkeypatch):
    track_calls(hermes, monkeypatch, keep=lambda n: n % 2 == 0)
    code, out = run(hermes, "check", "--quiet")
    assert code == 0 and "TRACKER GAP" in out, out
    code, out = run(hermes, "check", "--quiet")
    assert code == 0 and "TRACKER GAP" not in out, out


def test_without_tracker_everything_still_works(hermes):
    code, out = run(hermes, "hours")
    assert code == 0 and "not installed" in out
    code, out = run(hermes, "report")
    assert code == 0 and "hourly tracker: not installed" in out


def hourly_executions(home):
    con = sqlite3.connect(home / "data/ledger.db")
    try:
        return con.execute("SELECT count(*) FROM executions e JOIN payloads p USING(payload_id) WHERE"
                           " json_extract(p.payload_json, '$.request.series_id') = 'hermes/tokens/main-hourly'"
                           ).fetchone()[0]
    finally:
        con.close()


def test_hourly_starts_immediately_from_state_db_history_and_records_every_6h(hermes, monkeypatch):
    # Plugin installed two days ago: older hours are estimated from state.db, newer ones exact.
    track_calls(hermes, monkeypatch, since=datetime.now(timezone.utc).timestamp() - 2 * 86400)
    code, out = run(hermes, "check")
    assert code == 0, out
    assert "hourly track record: no scored model/baseline pairs yet (a pair is scored 24 complete hours" in out
    assert hourly_executions(hermes) == 2
    con = sqlite3.connect(hermes / "data/ledger.db")
    sources = dict(con.execute("SELECT source_ref, count(*) FROM actuals WHERE series_id ="
                               " 'hermes/tokens/main-hourly' GROUP BY source_ref").fetchall())
    con.close()
    assert sources.get("token-tracker", 0) > 0 and sources.get("hermes:state.db:estimated", 0) > 0

    code, out = run(hermes, "check")
    assert code == 0 and hourly_executions(hermes) == 2, out


def test_hourly_surge_alert_names_the_runaway_session(hermes, monkeypatch):
    track_calls(hermes, monkeypatch)
    last = tu.hour_floor(datetime.now(timezone.utc)) - tu.HOUR  # last complete hour
    usage = tu.read_usage(hermes / "state.db", timezone.utc)
    calls = tu.read_calls(hermes / "plugin-data/token-tracker/calls.db", timezone.utc)
    history = tu.hourly_history(usage, calls, last - tu.HOUR)
    ledger = TemporalLedger(hermes / "data/ledger.db", clock=FixedClock(last + timedelta(minutes=5)))
    session = GnomonSession.from_config(hermes / "providers.toml", ledger=ledger)
    try:
        for provider, quantiles in (("fake", (tu.LOW, 0.5, tu.HIGH)), ("seasonal_naive", ())):
            request = tu.build_hourly_request(history, quantiles, "hermes/tokens/main-hourly",
                                              seasonal=provider == "seasonal_naive")
            assert session.forecast(provider, request)["status"] == "ok"
    finally:
        session.close()
    _, hook = load_plugin(hermes, monkeypatch)
    for i in range(12):  # a runaway loop inside the last complete hour
        fire(hook, **hermes_payload((last + timedelta(minutes=10 + i)).timestamp(), 250_000,
                                    request_id=f"loop-{i}", session="runaway-loop-7f3a"))

    code, out = run(hermes, "check", "--quiet")
    assert code == 0, out
    assert f"HOURLY SURGE {last:%a %H:%M}  3.0M tok, above the 90% range" in out
    assert "12 calls, 250K tok/call; session runaway-loop used 100%" in out
    assert hourly_executions(hermes) == 2  # seeded forecast is recent: nothing new recorded

    code, out = run(hermes, "hours", "--hours", "3")
    assert " ! " in out and "runaway-loo" in out

    code, out = run(hermes, "check", "--quiet")
    assert code == 0 and "HOURLY SURGE" not in out, out


def test_windowed_call_reads_keep_the_install_time(tmp_path, monkeypatch):
    _, hook = load_plugin(tmp_path, monkeypatch)
    now = datetime.now(timezone.utc)
    fire(hook, **hermes_payload((now - timedelta(days=60)).timestamp(), 10, request_id="old"))
    fire(hook, **hermes_payload((now - timedelta(hours=3)).timestamp(), 20, request_id="new"))
    calls = tu.read_calls(tmp_path / "plugin-data/token-tracker/calls.db", timezone.utc,
                          since=now - timedelta(days=29))
    assert [c["tokens"] for c in calls] == [20]
    assert tu.exact_from(calls) == tu.hour_floor(now - timedelta(days=60)) + tu.HOUR
    empty = {"hour_main": {}}
    history = tu.hourly_history(empty, calls, tu.hour_floor(now) - tu.HOUR)
    assert len(history) == tu.HOURLY_HISTORY and {src for _, _, src in history} == {"exact"}
    assert sum(v for _, v, _ in history) == 20

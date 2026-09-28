#!/usr/bin/env python3
"""Forecast and monitor Hermes LLM token usage with Gnomon, keeping evidence in a ledger.

report  Daily history, call mix, 7/30-day token and cost projections, and the forecast track records.
check   Cron entry point: record daily and hourly forecasts, score matured ones, and alert
        when a day or hour lands outside its forecast range.
hours   Exact hourly usage, the biggest calls and the forecast range per hour
        (needs the token-tracker Hermes plugin).

Standard library only. Gnomon runs as a subprocess, so this script does not need
to share Gnomon's Python environment. Nothing here reads credentials.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HORIZON = 7            # recorded, scored daily forecast horizon (days)
PROJECTION = 30        # report-only monthly projection
MIN_DAYS = 14          # complete days required before daily forecasting
HOURLY_HORIZON = 24    # recorded, scored hourly forecast horizon (hours)
MIN_HOURS = 24         # complete hours required before hourly forecasting
HOURLY_HISTORY = 28 * 24
LOW, HIGH = 0.05, 0.95  # alert interval
UNIT = "tokens"
HOUR = timedelta(hours=1)
SPARK = "▁▂▃▄▅▆▇█"


class GnomonError(RuntimeError):
    pass


@dataclass(frozen=True)
class Series:
    """One recorded forecast series in the ledger."""
    series_id: str
    horizon: int
    grace: timedelta   # a forecast counts only if recorded this soon after its origin
    per: str           # period label: "day" or "hour"


# ── Hermes state.db ──────────────────────────────────────────────────────────

def hour_floor(dt):
    """UTC start of the hour containing dt. Hourly series live in UTC so DST never bends the grid."""
    return dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def read_usage(db, tz, base_url_filter=None):
    """Per-day LLM usage from Hermes: tokens, calls, cache reads, cost.

    Returns {"tokens"|"main_tokens"|"calls"|"cache_read"|"cost"|"unpriced": {date: value},
    "hour_main": {utc hour start: main-agent tokens (estimated)}, "models": {...}}.
    Each session_model_usage row is spread evenly over that session's assistant
    messages between the row's first_seen and last_seen, so sessions that cross
    midnight or use several models are split correctly. Rows with no matching
    message land on last_seen (or the session start for legacy rows).
    """
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(session_model_usage)")}
        seen = "u.first_seen, u.last_seen" if {"first_seen", "last_seen"} <= cols else "NULL, NULL"
        calls = "u.api_call_count" if "api_call_count" in cols else "0"
        task = "u.task" if "task" in cols else "''"
        where, params = "", ()
        if base_url_filter:
            where, params = "WHERE u.billing_base_url LIKE ?", (f"%{base_url_filter}%",)
        rows = con.execute(
            f"""SELECT u.session_id, u.model, u.input_tokens, u.output_tokens,
                       u.cache_read_tokens, u.cache_write_tokens,
                       u.estimated_cost_usd, u.actual_cost_usd, {calls}, {task}, {seen},
                       s.started_at, COALESCE(s.ended_at, s.last_activity_at, s.started_at)
                FROM session_model_usage u JOIN sessions s ON s.id = u.session_id {where}""",
            params).fetchall()
        stamps = defaultdict(list)
        for sid, ts in con.execute(
                "SELECT session_id, timestamp FROM messages WHERE role = 'assistant' ORDER BY timestamp"):
            stamps[sid].append(ts)
    finally:
        con.close()

    usage = {k: defaultdict(float) for k in
             ("tokens", "main_tokens", "calls", "cache_read", "cost", "unpriced", "hour_main")}
    models = defaultdict(lambda: {"tokens": 0, "calls": 0, "cost_usd": 0.0, "unpriced_tokens": 0})
    for sid, model, inp, out, cr, cw, est, act, n_calls, task, first, last, started, ended in rows:
        total = (inp or 0) + (out or 0) + (cr or 0) + (cw or 0)
        if not total:
            continue
        usd = act if act and act > 0 else (est or 0.0)
        lo, hi = first or started, last or ended or started
        anchors = [t for t in stamps.get(sid, ()) if lo - 1 <= t <= hi + 1] or [hi]
        share = 1 / len(anchors)
        for t in anchors:
            at = datetime.fromtimestamp(t, tz)
            d = at.date()
            usage["tokens"][d] += total * share
            if not task:  # main agent loop; side tasks (compression, titles, ...) carry a task name
                usage["main_tokens"][d] += total * share
                usage["hour_main"][hour_floor(at)] += total * share
            usage["calls"][d] += (n_calls or 0) * share
            usage["cache_read"][d] += (cr or 0) * share
            usage["cost"][d] += usd * share
            if usd <= 0:
                usage["unpriced"][d] += total * share
        m = models[model]
        m["tokens"] += total
        m["calls"] += n_calls or 0
        m["cost_usd"] += usd
        if usd <= 0:
            m["unpriced_tokens"] += total
    usage["models"] = dict(models)
    return usage


def call_mix(usage, days):
    """Calls, tokens per call and cache-read share over `days` (per-day averages)."""
    tokens = sum(usage["tokens"].get(d, 0) for d in days)
    calls = sum(usage["calls"].get(d, 0) for d in days)
    cache = sum(usage["cache_read"].get(d, 0) for d in days)
    return {"calls_per_day": calls / len(days) if days else 0.0,
            "tokens_per_call": tokens / calls if calls else None,
            "cache_read_share": cache / tokens if tokens else None}


def driver(usage, d, partial=False):
    """Say whether day d moved because of more calls or bigger calls, vs the 7 days before it.

    For today (partial) only the size of calls is comparable; the call count is not.
    """
    cur = call_mix(usage, [d])
    base = call_mix(usage, [d - timedelta(days=k) for k in range(1, 8)])
    if not cur["tokens_per_call"] or not base["tokens_per_call"] or not base["calls_per_day"]:
        return None
    size = (f"tokens/call {fmt(cur['tokens_per_call'])} vs {fmt(base['tokens_per_call'])} "
            f"({cur['tokens_per_call'] / base['tokens_per_call']:.1f}×)")
    if partial:
        return f"{cur['calls_per_day']:.0f} calls so far; {size}"
    return (f"calls {cur['calls_per_day']:.0f} vs {base['calls_per_day']:.0f}/day avg "
            f"({cur['calls_per_day'] / base['calls_per_day']:.1f}×), {size}")


class Calls(list):
    """Per-call rows (oldest first) plus when the plugin recorded its first call ever."""
    first_at = None


def read_calls(path, tz, base_url_filter=None, since=None):
    """Per-call rows from the token-tracker plugin since `since`, or None when it is not installed."""
    if not path or not Path(path).is_file():
        return None
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    try:
        where, params = ["1"], []
        if base_url_filter:
            where.append("base_url LIKE ?")
            params.append(f"%{base_url_filter}%")
        first = con.execute(f"SELECT MIN(ended_at) FROM calls WHERE {' AND '.join(where)}", params).fetchone()[0]
        if since is not None:
            where.append("ended_at >= ?")
            params.append(since.timestamp())
        rows = con.execute(
            "SELECT ended_at, session_id, model, input_tokens + output_tokens + cache_read_tokens"
            f" + cache_write_tokens, cache_read_tokens, usage_missing FROM calls WHERE {' AND '.join(where)}"
            " ORDER BY ended_at", params).fetchall()
    finally:
        con.close()
    calls = Calls({"at": datetime.fromtimestamp(t, tz), "session_id": sid, "model": model, "tokens": tok,
                   "cache_read": cr, "usage_missing": bool(missing)} for t, sid, model, tok, cr, missing in rows)
    calls.first_at = datetime.fromtimestamp(first, tz) if first is not None else None
    return calls


def exact_from(calls):
    """First fully tracked UTC hour: the hour after the plugin's first recorded call."""
    first = getattr(calls, "first_at", None) or (calls[0]["at"] if calls else None)
    return hour_floor(first) + HOUR if first else None


def hourly(calls, start, end):
    """Exact hour buckets [start, end), keyed by UTC hour start: tokens, calls, cache reads, sessions."""
    hours = {}
    h = hour_floor(start)
    while h < end:
        hours[h] = {"tokens": 0, "calls": 0, "cache_read": 0, "sessions": defaultdict(int)}
        h += HOUR
    for c in calls:
        b = hours.get(hour_floor(c["at"]))
        if b is not None:
            b["tokens"] += c["tokens"]
            b["calls"] += 1
            b["cache_read"] += c["cache_read"]
            b["sessions"][c["session_id"] or "?"] += c["tokens"]
    return hours


def hourly_history(usage, calls, last_complete):
    """Complete-hour main-agent token history, oldest first: [(utc hour start, tokens, source)].

    Hours from the plugin's first fully tracked hour are exact; earlier hours are
    estimated from state.db (session totals spread over replies).
    """
    first_exact = exact_from(calls)
    starts = [h for h, v in usage["hour_main"].items() if v > 0]
    if first_exact is not None:
        starts.append(first_exact)
    if not starts:
        return []
    start = max(min(starts), last_complete - (HOURLY_HISTORY - 1) * HOUR)
    exact = hourly(calls, max(start, first_exact), last_complete + HOUR) if first_exact else {}
    out, h = [], start
    while h <= last_complete:
        if first_exact is not None and h >= first_exact:
            out.append((h, float(exact[h]["tokens"]), "exact"))
        else:
            out.append((h, usage["hour_main"].get(h, 0.0), "estimated"))
        h += HOUR
    return out


def coverage(calls, usage, now):
    """Tracker tokens vs Hermes state.db main-agent tokens over 24 complete hours ending 2 hours ago.

    The 2-hour lag lets Hermes persist running sessions. state.db spreads session
    totals over replies, so small differences are expected; a large shortfall
    means calls are being missed.
    """
    if calls is None:
        return None
    end = hour_floor(now) - 2 * HOUR
    start = end - 24 * HOUR
    if exact_from(calls) is None or exact_from(calls) > start:
        return {"hours": 0}
    window = [c for c in calls if start <= c["at"] < end]
    main = sum(v for h, v in usage["hour_main"].items() if start <= h < end)
    tracked = sum(c["tokens"] for c in window)
    return {"hours": 24, "tracked_tokens": tracked, "state_main_tokens": round(main),
            "ratio": tracked / main if main else None,
            "calls_without_usage": sum(c["usage_missing"] for c in window)}


def coverage_line(cov):
    if cov is None:
        return "hourly tracker: not installed (see the skill's Setup)"
    if not cov.get("hours"):
        return "hourly tracker: installed; its coverage check starts about 24 hours after the first tracked call"
    if cov["ratio"] is None:
        return "hourly tracker: no main-agent usage in Hermes over the checked 24 hours"
    text = (f"hourly tracker: captured {cov['ratio'] * 100:.0f}% of main-agent tokens over the last 24h; "
            f"side tasks (compression, titles, ...) are daily-only")
    if cov["calls_without_usage"]:
        text += f"; {cov['calls_without_usage']} call(s) returned no usage"
    return text


def daily_series(tokens, today):
    """Zero-filled complete days (first usage day .. yesterday) plus today's partial total.

    A day with no Hermes sessions really had zero Hermes usage, so zeros inside
    the range are observations, not gaps. Days before first use are not padded.
    """
    if not tokens:
        return [], 0.0
    start, end = min(tokens), today - timedelta(days=1)
    days = []
    d = start
    while d <= end:
        days.append((d, tokens.get(d, 0.0)))
        d += timedelta(days=1)
    return days, tokens.get(today, 0.0)


# ── Gnomon subprocess wrapper ────────────────────────────────────────────────

class Gnomon:
    def __init__(self, cmd, ledger, write_config, providers_config=None, timeout=180):
        self.cmd, self.ledger, self.write_config = cmd, str(ledger), str(write_config)
        self.providers_config, self.timeout = providers_config, timeout
        self._executions = {}

    def run(self, *args):
        try:
            proc = subprocess.run([*self.cmd, *args], capture_output=True, text=True, timeout=self.timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GnomonError(f"gnomon did not run: {exc}") from None
        try:
            data = json.loads(proc.stdout)
        except ValueError:
            raise GnomonError((proc.stderr or proc.stdout or "no output").strip()[:400]) from None
        if data.get("status") != "ok":
            err = data.get("error") or {}
            raise GnomonError(f"{err.get('code', 'error')}: {err.get('message', 'gnomon call failed')}")
        return data

    def ledger_read(self, arguments):
        return self.run("ledger", "--ledger-path", self.ledger, "--arguments", json.dumps(arguments))["result"]

    def ledger_write(self, arguments):
        # Outcome writes need allow_outcome_writes in operator TOML.
        return self.run("ledger", "--providers-config", self.write_config,
                        "--arguments", json.dumps(arguments))["result"]

    def execution(self, execution_id):
        """A recorded execution's result; executions are immutable, so cache them."""
        if execution_id not in self._executions:
            self._executions[execution_id] = self.ledger_read(
                {"operation": "execution", "execution_id": execution_id})["result"]
        return self._executions[execution_id]

    def forecast(self, provider, request, record=True):
        args = ["forecast", "--provider", provider, "--request", json.dumps(request)]
        if record:
            args += ["--ledger-path", self.ledger]
        if self.providers_config:
            args += ["--providers-config", self.providers_config]
        return self.run(*args)


def gnomon_command(explicit):
    if explicit:
        return shlex.split(explicit)
    if os.environ.get("GNOMON_CMD"):
        return shlex.split(os.environ["GNOMON_CMD"])
    found = shutil.which("gnomon") or str(Path.home() / ".local/bin/gnomon")
    return [found]


# ── Forecast helpers ─────────────────────────────────────────────────────────

def day_end(d, tz):
    return datetime.combine(d + timedelta(days=1), time(), tz)


def stamp(d, tz):
    """Ledger timestamp for local day `d`: its end (the next local midnight).

    A day's total becomes known when the day ends. Stamping at the end keeps a
    forecast recorded during day d+1 strictly before its first target (the end
    of day d+1), which the ledger requires for prospective scoring.
    """
    return day_end(d, tz).isoformat()


def hour_stamp(h):
    """Ledger timestamp for the UTC hour starting at h: its end, for the same reason as days."""
    return (h + HOUR).isoformat()


def build_request(days, tz, horizon, quantiles, series_id, seasonal=False):
    """Daily request. Only the seasonal_naive baseline gets a seasonal period (a week):
    models such as Ephemeris take the frequency as their hint and refuse an explicit period."""
    last = days[-1][0]
    return {"history": [round(v) for _, v in days], "horizon": horizon, **({"season": 7} if seasonal else {}),
            "frequency": "D", "series_id": series_id, "unit": UNIT,
            **({"quantiles": list(quantiles)} if quantiles else {}),
            "timestamps": [stamp(d, tz) for d, _ in days],
            "future_timestamps": [stamp(last + timedelta(days=k), tz) for k in range(1, horizon + 1)]}


def build_hourly_request(hours, quantiles, series_id, seasonal=False):
    """Hourly request. The seasonal baseline repeats the same hour last week once two weeks
    exist, else yesterday; models get only the frequency hint, as for daily requests."""
    last = hours[-1][0]
    season = {"season": 168 if len(hours) >= 2 * 168 else 24} if seasonal else {}
    return {"history": [round(v) for _, v, _ in hours], "horizon": HOURLY_HORIZON, **season, "frequency": "h",
            "series_id": series_id, "unit": UNIT,
            **({"quantiles": list(quantiles)} if quantiles else {}),
            "timestamps": [hour_stamp(h) for h, _, _ in hours],
            "future_timestamps": [hour_stamp(last + k * HOUR) for k in range(1, HOURLY_HORIZON + 1)]}


def bands(result):
    """Per-step (low, median, high) from a Gnomon result; None when no quantiles."""
    rows = result.get("quantiles")
    if not rows:
        return None
    out = []
    for point, row in zip(result["point"], rows):
        q = {float(k): v for k, v in row.items()}
        out.append((max(q.get(LOW, point), 0), max(q.get(0.5, point), 0), max(q.get(HIGH, point), 0)))
    return out


def same_instant(a, b):
    return datetime.fromisoformat(a) == datetime.fromisoformat(b)


def search_all(g, query):
    items, cursor = [], None
    while True:
        page = g.ledger_read({**query, "limit": 100, **({"cursor": cursor} if cursor else {})})
        items += page["items"]
        cursor = page.get("next_cursor")
        if not cursor:
            return items


def executions(g, series, provider, since):
    """Recorded executions for provider in `series`, recorded at or after `since` (ISO).

    Ledger search windows filter on recording time, not forecast origin. The
    search otherwise stops at the current clock; a clock that steps backwards
    (NTP, WSL time sync) would then hide a forecast recorded seconds ago and
    cause a duplicate, so the cutoff is set a day ahead.
    """
    if not Path(g.ledger).exists():
        return []
    ahead = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    return search_all(g, {"operation": "search", "series_id": series.series_id, "provider": provider,
                          "horizon": series.horizon, "unit": UNIT, "start": since, "recorded_as_of": ahead})


def origin_executions(g, series, provider, origin):
    """Recorded executions whose history ends at `origin` (a forecast is recorded after its origin)."""
    return [i for i in executions(g, series, provider, origin) if same_instant(i["origin"], origin)]


# ── Commands ─────────────────────────────────────────────────────────────────

def load_context(args):
    tz = ZoneInfo(args.tz)
    now = datetime.now(tz)
    usage = read_usage(args.db, tz, args.base_url_filter)
    tokens, cost, unpriced = usage["tokens"], usage["cost"], usage["unpriced"]
    days, today_tokens = daily_series(tokens, now.date())
    data = Path(args.data_dir).expanduser().resolve()
    data.mkdir(parents=True, exist_ok=True)
    ledger = data / "ledger.db"
    write_config = data / "ledger-writes.toml"
    text = f'schema_version = 1\nledger_path = {json.dumps(str(ledger))}\nallow_outcome_writes = true\n'
    if not write_config.exists() or write_config.read_text() != text:
        write_config.write_text(text)
    g = Gnomon(gnomon_command(args.gnomon), ledger, write_config, args.providers_config)
    recent = [d for d, _ in days[-28:]]
    rec_tokens = sum(tokens.get(d, 0) - unpriced.get(d, 0) for d in recent)
    rec_cost = sum(cost.get(d, 0) for d in recent)
    rate = rec_cost / rec_tokens if rec_tokens > 0 else None  # USD per token, priced usage only
    base_id = "hermes/tokens" + (f"/{args.base_url_filter}" if args.base_url_filter else "")
    # Hourly features look back at most HOURLY_HISTORY hours (hours view: --hours).
    since = now - max(HOURLY_HISTORY + 26, getattr(args, "hours", 0) + 1) * HOUR
    calls = read_calls(args.calls_db, tz, args.base_url_filter, since=since)
    hours = hourly_history(usage, calls, hour_floor(now) - HOUR) if calls is not None else []
    return dict(tz=tz, now=now, days=days, today=today_tokens, usage=usage, models=usage["models"], g=g,
                rate=rate, calls=calls, coverage=coverage(calls, usage, now), hours=hours,
                daily=Series(base_id, HORIZON, timedelta(hours=23), "day"),
                hourly=Series(base_id + "/main-hourly", HOURLY_HORIZON, HOUR, "hour"),
                unpriced_28d=sum(unpriced.get(d, 0) for d in recent), series_id=base_id, data=data)


def sync_actuals(g, series, points):
    """Append points [(period end, value, source)]; append a revision only when a value changed."""
    if not points:
        return 0
    known = {datetime.fromisoformat(row["valid_time"]): row["value"]
             for row in g.ledger_read({"operation": "actuals_as_of", "series_id": series.series_id,
                                       "unit": UNIT})}
    now = datetime.now(timezone.utc).isoformat()
    new = []
    for end, v, source in points:
        value = float(round(v))
        if end in known and abs(known[end] - value) < 0.5:
            continue
        # First sighting: known when the period ended. A later change is a revision known now.
        new.append({"series_id": series.series_id, "valid_time": end.isoformat(), "value": value,
                    "unit": UNIT, "source_available_at": now if end in known else end.isoformat(),
                    "source_ref": source})
    for i in range(0, len(new), 1000):
        g.ledger_write({"operation": "append_actual", "actuals": new[i:i + 1000]})
    return len(new)


def record_forecasts(ctx, args):
    """Record today's daily model and baseline forecasts once per origin. Returns {role: result|error}."""
    g, days = ctx["g"], ctx["days"]
    origin = stamp(days[-1][0], ctx["tz"])
    out = {}
    for role, provider, quantiles in (("model", args.provider, (LOW, 0.5, HIGH)),
                                      ("baseline", args.baseline, ())):
        try:
            existing = origin_executions(g, ctx["daily"], provider, origin)
            if existing:
                out[role] = {"provider": provider, "recorded": "earlier",
                             "result": g.execution(existing[0]["execution_id"])}
                continue
            res = g.forecast(provider, build_request(days, ctx["tz"], HORIZON, quantiles, ctx["series_id"],
                                                     seasonal=role == "baseline"))
            out[role] = {"provider": provider, "recorded": "now", "result": res["result"]}
        except GnomonError as exc:
            out[role] = {"provider": provider, "error": str(exc)}
    return out


def record_hourly(ctx, args):
    """Record 24-hour model and baseline forecasts unless one was recorded in the last --hourly-every hours."""
    g, hours, series = ctx["g"], ctx["hours"], ctx["hourly"]
    since = (ctx["now"] - timedelta(hours=args.hourly_every)).astimezone(timezone.utc).isoformat()
    out = {}
    for role, provider, quantiles in (("model", args.provider, (LOW, 0.5, HIGH)),
                                      ("baseline", args.baseline, ())):
        try:
            if executions(g, series, provider, since):
                out[role] = {"provider": provider, "recorded": "earlier"}
                continue
            g.forecast(provider, build_hourly_request(hours, quantiles, series.series_id,
                                                      seasonal=role == "baseline"))
            out[role] = {"provider": provider, "recorded": "now"}
        except GnomonError as exc:
            out[role] = {"provider": provider, "error": str(exc)}
    return out


def score_and_compare(ctx, args, series):
    """Evaluate matured forecasts, then pair model and baseline ledger scores by origin.

    Gnomon's compare_history needs attested provider revisions, which Ephemeris
    does not supply, so pairing happens here from the ledger's per-forecast MAE.
    Only forecasts recorded before their first target period count.
    """
    g = ctx["g"]
    base = {"operation": "search", "series_id": series.series_id, "horizon": series.horizon, "unit": UNIT}
    ready = [i["execution_id"] for i in search_all(g, {**base, "status": "ready"})]
    for i in range(0, len(ready), 100):
        g.ledger_read({"operation": "evaluate", "execution_ids": ready[i:i + 100], "allow_partial": False})
    roles = {args.provider: "model", args.baseline: "baseline"}
    scores = {}
    for provider, role in roles.items():
        by_origin = {}
        for item in search_all(g, {**base, "provider": provider, "status": "scored"}):
            origin = datetime.fromisoformat(item["origin"])
            if item.get("mae") is None or datetime.fromisoformat(item["recorded_at"]) >= origin + series.grace:
                continue
            by_origin.setdefault(origin, (item["mae"], item.get("revision")))  # first recorded wins
        scores[role] = by_origin
    common = sorted(set(scores["model"]) & set(scores["baseline"]))
    out = {"roles": roles, "matched_origins": len(common), "horizon": series.horizon, "per": series.per}
    if common:
        m = [scores["model"][o][0] for o in common]
        b = [scores["baseline"][o][0] for o in common]
        out.update(model_mae=sum(m) / len(m), baseline_mae=sum(b) / len(b),
                   model_wins=sum(x < y for x, y in zip(m, b)),
                   model_revisions=sorted({str(scores["model"][o][1]) for o in common}),
                   first_origin=common[0].isoformat(), last_origin=common[-1].isoformat())
    return len(ready), out


def interval_hits(ctx, args):
    """Completed days checked against the one-day-ahead model forecast made the day before."""
    g, tz = ctx["g"], ctx["tz"]
    hits = []
    for d, actual in ctx["days"][-args.lookback:]:
        origin = stamp(d - timedelta(days=1), tz)
        try:
            runs = origin_executions(g, ctx["daily"], args.provider, origin)
            if not runs:
                continue
            b = bands(g.execution(runs[0]["execution_id"]))
        except GnomonError:
            continue
        if not b:
            continue
        lo, mid, hi = b[0]
        kind = "SURGE" if actual > hi else "DIP" if actual < lo else None
        hits.append({"day": d.isoformat(), "actual": actual, "low": lo, "median": mid, "high": hi, "kind": kind})
    return hits


def hour_bands(ctx, args, start, end):
    """{utc hour start: (low, median, high)} for hours in [start, end), each from the most recent
    model forecast recorded before that hour ended (the ledger's prospective rule)."""
    g = ctx["g"]
    runs = executions(g, ctx["hourly"], args.provider, (start - HOURLY_HORIZON * HOUR).isoformat())
    out = {}
    for run in sorted(runs, key=lambda r: datetime.fromisoformat(r["origin"])):  # later origins win
        origin = datetime.fromisoformat(run["origin"]).astimezone(timezone.utc)  # = first target hour start
        recorded = datetime.fromisoformat(run["recorded_at"])
        wanted = [(origin + k * HOUR, k) for k in range(HOURLY_HORIZON)
                  if start <= origin + k * HOUR < end and recorded < origin + (k + 1) * HOUR]
        if wanted:
            b = bands(g.execution(run["execution_id"]))
            for h, k in wanted if b else ():
                out[h] = b[k]
    return out


def hourly_alerts(ctx, args, fired):
    """Finished, fully tracked hours outside their forecast range, ignoring hours below --hourly-floor."""
    alerts = []
    end = hour_floor(ctx["now"])
    start = max(end - args.hourly_lookback * HOUR, exact_from(ctx["calls"]))
    if start >= end:
        return alerts
    try:
        expected = hour_bands(ctx, args, start, end)
    except GnomonError:
        return alerts
    for h, v in hourly(ctx["calls"], start, end).items():
        if h not in expected:
            continue
        lo, mid, hi = expected[h]
        kind = ("SURGE" if v["tokens"] > hi and v["tokens"] >= args.hourly_floor else
                "DIP" if v["tokens"] < lo and mid >= args.hourly_floor else None)
        key = f"{h.isoformat()}:HOURLY_{kind}"
        if not kind or key in fired:
            continue
        head = "▲ HOURLY SURGE" if kind == "SURGE" else "▼ HOURLY DIP"
        text = (f"{head} {h.astimezone(ctx['tz']):%a %H:%M}  {fmt(v['tokens'])} tok, "
                f"{'above' if kind == 'SURGE' else 'below'} the 90% range {fmt(lo)}–{fmt(hi)} (median {fmt(mid)})")
        if v["calls"]:
            sid, tok = max(v["sessions"].items(), key=lambda kv: kv[1])
            text += (f"\n      {v['calls']} calls, {fmt(v['tokens'] / v['calls'])} tok/call; "
                     f"session {sid[:12]} used {tok / v['tokens'] * 100:.0f}%")
        alerts.append(text)
        fired.add(key)
    return alerts


def cmd_check(args):
    ctx = load_context(args)
    g, days = ctx["g"], ctx["days"]
    lines, alerts = [], []
    state_path = ctx["data"] / "alert-state.json"
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        state = {}
    fired = set(state.get("fired", []))

    # Daily: forecasts, actuals, scores, day alerts.
    fc, cmp, appended, scored = {}, {}, 0, 0
    if len(days) >= MIN_DAYS:
        fc = record_forecasts(ctx, args)  # first: recording a forecast creates the ledger
        appended = sync_actuals(g, ctx["daily"], [(day_end(d, ctx["tz"]), v, "hermes:state.db")
                                                  for d, v in days])
        scored, cmp = score_and_compare(ctx, args, ctx["daily"])
        for h in interval_hits(ctx, args):
            key = f"{h['day']}:{h['kind']}"
            if h["kind"] and key not in fired:
                word = "above" if h["kind"] == "SURGE" else "below"
                alerts.append(f"▲ SURGE {h['day']}" if h["kind"] == "SURGE" else f"▼ DIP {h['day']}")
                alerts[-1] += (f"  {fmt(h['actual'])} tok, {word} the forecast's 90% range "
                               f"{fmt(h['low'])}–{fmt(h['high'])} (median {fmt(h['median'])})")
                why = driver(ctx["usage"], datetime.fromisoformat(h["day"]).date())
                if why:
                    alerts[-1] += f"\n      {why}"
                fired.add(key)
    model = fc.get("model", {})
    b = bands(model["result"]) if "result" in model else None
    if b:
        lo, mid, hi = b[0]
        key = f"{ctx['now'].date()}:INTRADAY"
        if ctx["today"] > hi and key not in fired:
            alerts.append(f"▲ SURGE IN PROGRESS  today {fmt(ctx['today'])} tok so far, already above the "
                          f"forecast high {fmt(hi)} (median {fmt(mid)})")
            why = driver(ctx["usage"], ctx["now"].date(), partial=True)
            if why:
                alerts[-1] += f"\n      {why}"
            fired.add(key)
        if args.weekly_budget and ctx["rate"]:
            week_cost = sum(m for _, m, _ in b) * ctx["rate"]
            key = f"{ctx['now'].date()}:BUDGET"
            if week_cost > args.weekly_budget and key not in fired:
                alerts.append(f"⚠ BUDGET  next 7 days projected ~${week_cost:.2f} (median), "
                              f"over the ${args.weekly_budget:.2f} weekly budget")
                fired.add(key)

    # Hourly: needs the plugin's live per-call records.
    hfc, hcmp, happended, hscored = {}, {}, 0, 0
    if exact_from(ctx["calls"]) and len(ctx["hours"]) >= MIN_HOURS:
        hfc = record_hourly(ctx, args)
        recent = [(h + HOUR, v, "token-tracker" if src == "exact" else "hermes:state.db:estimated")
                  for h, v, src in ctx["hours"][-72:]]
        happended = sync_actuals(g, ctx["hourly"], recent)
        hscored, hcmp = score_and_compare(ctx, args, ctx["hourly"])
        alerts += hourly_alerts(ctx, args, fired)

    cov = ctx["coverage"]
    key = f"{ctx['now'].date()}:TRACKER_GAP"
    if cov and cov.get("ratio") is not None and cov["ratio"] < args.min_coverage and key not in fired:
        alerts.append(f"⚠ TRACKER GAP  {coverage_line(cov)}; hourly numbers undercount")
        fired.add(key)
    for label, forecasts in (("", fc), ("hourly ", hfc)):
        for role in ("model", "baseline"):
            key = f"{ctx['now'].date()}:FAILED:{label}{role}"
            if "error" in forecasts.get(role, {}) and key not in fired:
                alerts.append(f"⚠ FORECAST FAILED  {label}{forecasts[role]['provider']}: {forecasts[role]['error']}")
                fired.add(key)

    state["fired"] = sorted(fired)[-500:]
    state_path.write_text(json.dumps(state))

    if len(days) < MIN_DAYS:
        lines.append(f"WAITING  {len(days)} complete day(s) of Hermes usage; daily forecasting starts at {MIN_DAYS}.")
    else:
        lines.append(f"yesterday {fmt(days[-1][1])} tok   today so far {fmt(ctx['today'])} tok")
        if b:
            lines.append(f"today forecast {fmt(b[0][1])} (90%: {fmt(b[0][0])}–{fmt(b[0][2])})  [{model['provider']}]")
        lines.append(track_record_line(cmp))
        lines.append(f"ledger: +{appended} actual(s), {scored} forecast(s) scored")
    if ctx["calls"] is not None:
        if len(ctx["hours"]) < MIN_HOURS:
            lines.append(f"hourly: {len(ctx['hours'])} complete hour(s) of history; forecasting starts at {MIN_HOURS}")
        else:
            lines.append(track_record_line(hcmp, "hourly "))
            lines.append(f"hourly ledger: +{happended} actual(s), {hscored} forecast(s) scored")
    lines.append(coverage_line(cov))
    if alerts or not args.quiet:
        print(("ALERT\n  " + "\n  ".join(alerts) + "\n  ─────\n  " if alerts else "OK  ") + "\n  ".join(lines))
    return 0


def track_record_line(cmp, label=""):
    n = cmp.get("matched_origins")
    per = cmp.get("per", "day")
    horizon = cmp.get("horizon", HORIZON)
    if not n:
        return (f"{label}track record: no scored model/baseline pairs yet (a pair is scored {horizon} "
                f"complete {per}s after it is recorded)")
    name = {role: provider for provider, role in cmp["roles"].items()}
    m, b = cmp["model_mae"], cmp["baseline_mae"]
    rel = (b - m) / b * 100 if b else 0.0
    return (f"{label}track record: {name['model']} {abs(rel):.0f}% {'lower' if rel >= 0 else 'higher'} MAE than "
            f"{name['baseline']} over {n} paired {horizon}-{per} forecast(s), better on {cmp['model_wins']}/{n} "
            f"(MAE {fmt(m)} vs {fmt(b)} tok/{per})")


def cmd_report(args):
    ctx = load_context(args)
    g, days, tz = ctx["g"], ctx["days"], ctx["tz"]
    out = {"as_of": ctx["now"].isoformat(), "timezone": args.tz, "series_id": ctx["series_id"],
           "complete_days": len(days), "today_partial_tokens": round(ctx["today"]),
           "models": ctx["models"], "usd_per_million_tokens_28d": ctx["rate"] and ctx["rate"] * 1e6,
           "unpriced_tokens_28d": round(ctx["unpriced_28d"])}
    if len(days) >= MIN_DAYS:
        req = build_request(days, tz, PROJECTION, (LOW, 0.5, HIGH), ctx["series_id"])
        try:
            res = g.forecast(args.provider, req, record=False)["result"]
            out["projection"] = {"provider": args.provider, "timestamps": res["timestamps"],
                                 "bands": bands(res), "point": res["point"]}
        except GnomonError as exc:
            out["projection_error"] = f"{args.provider}: {exc}"
    if Path(g.ledger).exists():
        for key, series in (("track_record", ctx["daily"]), ("hourly_track_record", ctx["hourly"])):
            try:
                out[key] = score_and_compare(ctx, args, series)[1]
            except GnomonError as exc:
                out[key + "_error"] = str(exc)
    u = ctx["usage"]
    last7 = [d for d, _ in days[-7:]]
    prev7 = [d for d, _ in days[-14:-7]]
    out["llm_calls"] = {"last_7_days": call_mix(u, last7), "previous_7_days": call_mix(u, prev7)}
    out["tracker_coverage"] = ctx["coverage"]
    if ctx["calls"] is not None:
        end = hour_floor(ctx["now"]) + HOUR
        out["last_24_hours"] = [{"hour": h.astimezone(tz).isoformat(), "tokens": v["tokens"], "calls": v["calls"],
                                 "cache_read": v["cache_read"]}
                                for h, v in hourly(ctx["calls"], end - 24 * HOUR, end).items()]
    if args.json:
        out["history"] = [{"day": d.isoformat(), "tokens": round(v), "calls": round(u["calls"].get(d, 0)),
                           "cache_read_tokens": round(u["cache_read"].get(d, 0))} for d, v in days]
        print(json.dumps(out, indent=2, default=str))
        return 0
    print(render_report(ctx, out))
    return 0


def cmd_hours(args):
    ctx = load_context(args)
    calls, tz = ctx["calls"], ctx["tz"]
    if calls is None:
        print(coverage_line(None))
        return 0
    current = hour_floor(ctx["now"])
    end = current + HOUR
    start = end - args.hours * HOUR
    hrs = hourly(calls, start, end)
    try:
        expected = hour_bands(ctx, args, start, end)
    except GnomonError:
        expected = {}
    window = [c for c in calls if start <= c["at"] < end]
    if args.json:
        print(json.dumps({
            "timezone": args.tz, "coverage": ctx["coverage"],
            "hours": [{"hour": h.astimezone(tz).isoformat(), "tokens": v["tokens"], "calls": v["calls"],
                       "cache_read": v["cache_read"],
                       "expected": dict(zip(("low", "median", "high"), expected[h])) if h in expected else None}
                      for h, v in hrs.items()],
            "biggest_calls": [{**c, "at": c["at"].isoformat()} for c in
                              sorted(window, key=lambda c: -c["tokens"])[:args.top]]}, indent=2))
        return 0
    top = max((v["tokens"] for v in hrs.values()), default=0)
    L = [f"Hermes LLM calls by hour — last {args.hours}h ({args.tz}, exact per-call records)", "",
         "  hour              tokens   calls  tok/call  expected (90%)"]
    for h, v in hrs.items():
        bar = "█" * round(v["tokens"] / top * 16) if top else ""
        per = fmt(v["tokens"] / v["calls"]) if v["calls"] else "-"
        exp = f"{fmt(expected[h][0])}–{fmt(expected[h][2])}" if h in expected else ""
        outside = h in expected and h < current and not expected[h][0] <= v["tokens"] <= expected[h][2]
        L.append(f"  {h.astimezone(tz):%a %d %H:%M}  {fmt(v['tokens']):>9} {v['calls']:>7}  {per:>8}  "
                 f"{exp:>14}{' !' if outside else '  '} {bar}")
    L.append("  (current hour is partial; ! = outside the forecast range)")
    if window:
        L += ["", "biggest calls in this window"]
        for c in sorted(window, key=lambda c: -c["tokens"])[:args.top]:
            L.append(f"  {c['at']:%a %H:%M}  {fmt(c['tokens']):>8} tok  {(c['model'] or '?')[:28]:28} "
                     f"session {(c['session_id'] or '?')[:12]}")
    L += ["", coverage_line(ctx["coverage"])]
    print("\n".join(L))
    return 0


def render_report(ctx, out):
    days, rate = ctx["days"], ctx["rate"]
    usd = (lambda t: f"~${t * rate:,.2f}") if rate else (lambda t: "cost n/a")
    L = [f"Hermes token usage — {ctx['now']:%Y-%m-%d %H:%M} {out['timezone']}", ""]
    last14 = [v for _, v in days[-14:]]
    if last14:
        L.append(f"last 14 days  {spark(last14)}  total {fmt(sum(last14))} tok  {usd(sum(last14))}")
    L.append(f"today so far  {fmt(ctx['today'])} tok (partial day, not used for forecasting)")
    if out.get("last_24_hours"):
        hrs = out["last_24_hours"]
        peak = max(hrs, key=lambda h: h["tokens"])
        L.append(f"last 24 hours {spark([h['tokens'] for h in hrs])}  peak {fmt(peak['tokens'])} tok "
                 f"at {peak['hour'][11:16]} ({peak['calls']} calls)")
    L.append(coverage_line(out["tracker_coverage"]))
    if rate:
        L.append(f"blended rate  ${rate * 1e6:.2f}/M tok from priced Hermes usage, last 28 days; all $ below use it")
    if out["unpriced_tokens_28d"]:
        L.append(f"unpriced      {fmt(out['unpriced_tokens_28d'])} tok in 28 days have no Hermes cost record; "
                 "they are priced at the blended rate")
    calls = out.get("llm_calls")
    if calls and calls["last_7_days"]["tokens_per_call"]:
        a, b = calls["last_7_days"], calls["previous_7_days"]

        def change(x, y):
            return f"({(x - y) / y * 100:+.0f}%)" if x is not None and y else ""
        L += ["", "LLM calls        last 7 days   vs previous 7"]
        L.append(f"  calls/day      {a['calls_per_day']:>11.0f}   {change(a['calls_per_day'], b['calls_per_day'])}")
        L.append(f"  tokens/call    {fmt(a['tokens_per_call']):>11}   {change(a['tokens_per_call'], b['tokens_per_call'])}")
        if a["cache_read_share"] is not None:
            prev = b["cache_read_share"]
            delta = f"(was {prev * 100:.0f}%)" if prev is not None else ""
            L.append(f"  cache-read     {a['cache_read_share'] * 100:>10.0f}%   {delta}")
    L += ["", "by model (all time)"]
    for m, s in sorted(ctx["models"].items(), key=lambda kv: -kv[1]["tokens"])[:6]:
        L.append(f"  {m[:32]:32} {fmt(s['tokens']):>8} tok  {s['calls']:>7,} calls  ${s['cost_usd']:,.2f}")
    if len(days) < MIN_DAYS:
        L += ["", f"Daily forecasts start after {MIN_DAYS} complete days (have {len(days)})."]
    else:
        L.append("")
        if "projection" in out:
            p, b = out["projection"], out["projection"]["bands"]
            L.append(f"projection  [{p['provider']}]   median   90% range per day, summed*")
            for label, n in (("next day", 1), ("next 7 days", 7), ("next 30 days", 30)):
                lo, mid, hi = (sum(x[i] for x in b[:n]) for i in range(3)) if b else (0, sum(p['point'][:n]), 0)
                rng = f"{fmt(lo)}–{fmt(hi)}" if b else "n/a"
                L.append(f"  {label:12} {fmt(mid):>8} tok  {rng:>17}   {usd(mid)}")
            if b:
                L.append(f"  week ahead  {spark([x[1] for x in b[:7]])}")
                L.append("  *multi-day ranges add daily bounds, so they are wider than a true 90% interval.")
        else:
            L.append(f"projection unavailable: {out.get('projection_error')}")
            L.append("  If Ephemeris is not connected, load the connect-ephemeris skill or pass --provider.")
    L += ["", track_record_line(out.get("track_record") or {})]
    if ctx["calls"] is not None:
        L.append(track_record_line(out.get("hourly_track_record") or {"per": "hour", "horizon": HOURLY_HORIZON},
                                   "hourly "))
    return "\n".join(L)


# ── formatting ───────────────────────────────────────────────────────────────

def fmt(n):
    n = max(n or 0, 0)
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}K"
    return f"{n:.0f}"


def spark(values):
    hi = max(values) if values else 0
    return "".join(SPARK[min(int(v / hi * 7), 7)] if hi > 0 else SPARK[0] for v in values)


def main(argv=None):
    home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("report", "check", "hours"))
    p.add_argument("--db", default=str(home / "state.db"), help="Hermes state.db (opened read-only)")
    p.add_argument("--data-dir", default=str(home / "data" / "token-usage"), help="Ledger and alert state")
    p.add_argument("--calls-db", default=str(home / "plugin-data" / "token-tracker" / "calls.db"),
                   help="Per-call records written by the token-tracker Hermes plugin")
    p.add_argument("--tz", default="UTC", help="IANA timezone for day boundaries and display (default UTC)")
    p.add_argument("--provider", default="ephemeris/chronos2", help="Forecast model (Gnomon provider name)")
    p.add_argument("--baseline", default="seasonal_naive", help="Free local baseline recorded beside the model")
    p.add_argument("--providers-config", help="Operator TOML defining providers; replaces the saved Ephemeris connection")
    p.add_argument("--base-url-filter", help="Only count usage whose billing_base_url contains this text")
    p.add_argument("--gnomon", help="Gnomon command (default: gnomon on PATH, or $GNOMON_CMD)")
    p.add_argument("--weekly-budget", type=float, help="check: alert when the 7-day median projection costs more (USD)")
    p.add_argument("--lookback", type=int, default=3, help="check: completed days re-checked for missed alerts")
    p.add_argument("--hourly-every", type=int, default=6, help="check: hours between recorded hourly forecasts")
    p.add_argument("--hourly-floor", type=int, default=200_000,
                   help="check: ignore hourly surges below this many tokens, and dips from expected hours below it")
    p.add_argument("--hourly-lookback", type=int, default=3, help="check: finished hours re-checked for alerts")
    p.add_argument("--min-coverage", type=float, default=0.9,
                   help="check: alert when the tracker captured less than this share of main-agent tokens")
    p.add_argument("--hours", type=int, default=24, help="hours: how many hours to show")
    p.add_argument("--top", type=int, default=5, help="hours: how many of the biggest calls to list")
    p.add_argument("--quiet", action="store_true", help="check: print only alerts and errors (silent cron ticks)")
    p.add_argument("--json", action="store_true", help="report/hours: machine-readable output")
    args = p.parse_args(argv)
    if not Path(args.db).is_file():
        print(f"NO_DATA  Hermes state database not found: {args.db}")
        return 2
    try:
        return {"check": cmd_check, "report": cmd_report, "hours": cmd_hours}[args.command](args)
    except GnomonError as exc:
        print(f"ERROR  {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())

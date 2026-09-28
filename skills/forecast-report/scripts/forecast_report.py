#!/usr/bin/env python3
"""Forecast a time series with Gnomon and write a short summary plus a seaborn chart.

Reads a CSV of timestamps and values, forecasts the next `--horizon` steps with the
model (Ephemeris by default, with 50% and 90% ranges) and a seasonal-naive baseline,
backtests both on the last `--horizon` observed points, and prints a summary whose
last line is `MEDIA:<chart.png>`. Hermes delivers that line as an image on Telegram.

Loading, forecasting and the summary use the standard library only; Gnomon runs as a
subprocess, so this script does not need Gnomon's Python environment. Only the chart
needs seaborn (and matplotlib). Nothing here reads credentials.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
STEPS = {"h": timedelta(hours=1), "D": timedelta(days=1), "W": timedelta(weeks=1)}
MONTHS = {"MS": 1, "QS": 3, "YS": 12}
SEASONS = {"h": 24, "D": 7, "W": 52, "MS": 12, "QS": 4, "YS": 1}
DEFAULT_HORIZON = {"h": 48, "D": 14, "W": 8, "MS": 6, "QS": 4, "YS": 3}
SEASON_WORDS = {"h": "same hour yesterday", "D": "same weekday last week", "W": "same week last year",
                "MS": "same month last year", "QS": "same quarter last year"}
UNIT_NAMES = {"h": "hour", "D": "day", "W": "week", "MS": "month", "QS": "quarter", "YS": "year"}


def hermes_home():
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


class GnomonError(RuntimeError):
    pass


class InputError(ValueError):
    pass


# ── Input ────────────────────────────────────────────────────────────────────

def parse_time(text):
    text = text.strip()
    try:
        t = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError:
        raise InputError(f"timestamp {text!r} is not ISO 8601 (e.g. 2026-09-28 or 2026-09-28T14:00)") from None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def parse_value(text):
    try:
        return float(text.replace(",", "").strip())
    except ValueError:
        return None


def load_series(path, time_col=None, value_col=None, agg=None):
    """Sorted (timestamp, value) pairs. Columns default to the first that parse as time / number."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise InputError(f"{path} has no data rows")
    cols = list(rows[0])
    for name in (time_col, value_col):
        if name and name not in cols:
            raise InputError(f"column {name!r} not in {cols}")
    sample = rows[: min(len(rows), 20)]

    def is_time(c):
        try:
            return all(parse_time(r[c]) for r in sample if r[c])
        except InputError:
            return False
    time_col = time_col or next((c for c in cols if is_time(c)), None)
    value_col = value_col or next((c for c in cols if c != time_col
                                   and all(parse_value(r[c] or "") is not None for r in sample if r[c])), None)
    if not time_col or not value_col:
        raise InputError(f"could not find a timestamp and a numeric column in {cols}; pass --time-col/--value-col")

    grouped = {}
    for r in rows:
        if not (r.get(time_col) or "").strip() or not (r.get(value_col) or "").strip():
            continue
        value = parse_value(r[value_col])
        if value is None:
            raise InputError(f"{value_col}={r[value_col]!r} is not a number")
        grouped.setdefault(parse_time(r[time_col]), []).append(value)
    dupes = sum(len(v) > 1 for v in grouped.values())
    if dupes and not agg:
        raise InputError(f"{dupes} timestamps repeat; pass --agg sum or --agg mean to combine them")
    combine = (lambda v: sum(v) / len(v)) if agg == "mean" else sum
    series = sorted((t, combine(v)) for t, v in grouped.items())
    return series, time_col, value_col


def add_months(t, n):
    month = t.month - 1 + n
    year, month = t.year + month // 12, month % 12 + 1
    return t.replace(year=year, month=month, day=min(t.day, calendar.monthrange(year, month)[1]))


def step(t, freq, k=1):
    return add_months(t, MONTHS[freq] * k) if freq in MONTHS else t + STEPS[freq] * k


def infer_freq(times):
    if len(times) < 3:
        raise InputError("need at least 3 observations")
    gaps = sorted((b - a).total_seconds() for a, b in zip(times, times[1:]))
    median = gaps[len(gaps) // 2] / 86400
    for freq, days in (("h", 1 / 24), ("D", 1), ("W", 7), ("MS", 30.4), ("QS", 91.3), ("YS", 365.25)):
        if abs(median - days) <= days * 0.15:
            return freq
    raise InputError(f"median spacing {median:.3g} days is not hourly/daily/weekly/monthly/quarterly/yearly; "
                     "pass --freq")


def missing_steps(times, freq):
    """Expected timestamps absent from a regular grid (the model sees values only, so gaps compress time)."""
    have, k, missing = set(times), 1, 0
    while (t := step(times[0], freq, k)) < times[-1]:
        missing += t not in have
        k += 1
    return missing


# ── Gnomon ───────────────────────────────────────────────────────────────────

def gnomon_command(explicit):
    if explicit:
        return shlex.split(explicit)
    if os.environ.get("GNOMON_CMD"):
        return shlex.split(os.environ["GNOMON_CMD"])
    return [shutil.which("gnomon") or str(Path.home() / ".local/bin/gnomon")]


def run_forecast(cmd, provider, request, ledger=None, providers_config=None, timeout=180):
    args = [*cmd, "forecast", "--provider", provider, "--request", json.dumps(request)]
    if ledger:
        args += ["--ledger-path", str(ledger)]
    if providers_config:
        args += ["--providers-config", providers_config]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
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


def build_request(series, horizon, freq, series_id, unit, quantiles=(), season=None):
    """Models get the frequency hint only (Ephemeris refuses an explicit period); the baseline gets `season`."""
    last = series[-1][0]
    return {"history": [v for _, v in series], "horizon": horizon, "frequency": freq,
            "series_id": series_id, "unit": unit,
            **({"season": season} if season else {}),
            **({"quantiles": list(quantiles)} if quantiles else {}),
            "timestamps": [t.isoformat() for t, _ in series],
            "future_timestamps": [step(last, freq, k).isoformat() for k in range(1, horizon + 1)]}


def forecast_path(reply):
    """{"point": [...], "q": {level: [...]}} from a Gnomon reply."""
    result = reply["result"]
    rows = result.get("quantiles") or []
    q = {}
    for row in rows:
        for level, value in row.items():
            q.setdefault(float(level), []).append(float(value))
    return {"point": [float(v) for v in result["point"]], "q": q,
            "execution_id": reply.get("execution_id"), "provider": reply.get("provider"),
            "models_used": ((result.get("metadata") or {}).get("service") or {}).get("models_used") or []}


def members(fc):
    """ " (chronos2 + timesfm)" when the service reported which models produced the forecast."""
    used = fc and fc.get("models_used")
    return f" ({' + '.join(used)})" if used else ""


def baseline_season(freq, n):
    season = SEASONS[freq]
    return season if season > 1 and n >= 2 * season else 1


def mae(pred, actual):
    return sum(abs(p - a) for p, a in zip(pred, actual)) / len(actual)


def run_all(args, series, freq):
    horizon = args.horizon or DEFAULT_HORIZON[freq]
    series_id = f"forecast-report/{args.slug}"
    cmd = gnomon_command(args.gnomon)
    ledger = None if args.no_ledger else Path(args.ledger_path or hermes_home() / "data" / "forecast-report"
                                              / "ledger.db").expanduser().resolve()
    if ledger:
        ledger.parent.mkdir(parents=True, exist_ok=True)
    out = {"horizon": horizon, "freq": freq, "series_id": series_id, "ledger": str(ledger) if ledger else None,
           "model": None, "baseline": None, "backtest": None, "errors": []}

    def forecast(provider, hist, record, quantiles=(), season=None):
        request = build_request(hist, horizon, freq, series_id, args.unit, quantiles, season)
        return forecast_path(run_forecast(cmd, provider, request, ledger if record else None, args.providers_config))

    season = baseline_season(freq, len(series))
    baseline_name = f"seasonal naive ({SEASON_WORDS[freq]})" if season > 1 else "last value"
    out["baseline_name"] = baseline_name
    try:
        out["model"] = forecast(args.provider, series, True, QUANTILES)
    except GnomonError as exc:
        out["errors"].append(f"{args.provider} forecast failed: {exc}")
    try:
        out["baseline"] = forecast("seasonal_naive", series, True, season=season)
    except GnomonError as exc:
        out["errors"].append(f"baseline forecast failed: {exc}")

    # Backtest: forecast the last `horizon` observed points from the history before them.
    train, held = series[:-horizon], series[-horizon:]
    bt_season = baseline_season(freq, len(train))
    if args.no_backtest:
        pass
    elif len(train) < max(2 * bt_season, 8):
        out["backtest_note"] = f"too little history to hold out {horizon} points"
    else:
        actual = [v for _, v in held]
        bt = {"start": held[0][0], "start_index": len(train), "actual": actual}
        try:
            m = forecast(args.provider, train, False, QUANTILES)
            lo, hi = m["q"].get(0.05), m["q"].get(0.95)
            bt.update(model=m, model_mae=mae(m["point"], actual),
                      covered=sum(a <= b <= c for a, b, c in zip(lo, actual, hi)) if lo and hi else None)
        except GnomonError as exc:
            out["errors"].append(f"{args.provider} backtest failed: {exc}")
        try:
            b = forecast("seasonal_naive", train, False, season=bt_season)
            bt.update(baseline=b, baseline_mae=mae(b["point"], actual))
        except GnomonError as exc:
            out["errors"].append(f"baseline backtest failed: {exc}")
        out["backtest"] = bt
    return out


# ── Summary ──────────────────────────────────────────────────────────────────

def fmt(n):
    a = abs(n)
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{n / div:.3g}{suffix}"
    return f"{n:.3g}" if a < 100 else f"{n:.0f}"


def fmt_time(t, freq):
    if freq == "h":
        return t.strftime("%a %d %b %H:%M")
    if freq in ("D", "W"):
        return t.strftime("%a %d %b %Y")
    return t.strftime("%b %Y")


def summarise(args, series, freq, res):
    h, unit, name = res["horizon"], args.unit, UNIT_NAMES[freq]
    last_t = series[-1][0]
    future = [step(last_t, freq, k) for k in range(1, h + 1)]
    lines = [f"📈 {args.title}: next {h} {name}s"] + [f"⚠️ {err}" for err in res["errors"]]
    fc, label = ((res["model"], args.provider + members(res["model"])) if res["model"]
                 else (res["baseline"], res["baseline_name"]))
    if not fc:
        lines.append("No forecast was produced.")
        return lines
    agg = sum if args.stat == "sum" else (lambda v: sum(v) / len(v))
    total, prev = agg(fc["point"]), agg([v for _, v in series[-h:]])
    change = f" ({(total - prev) / abs(prev):+.0%} vs the last {h})" if prev else ""
    word = "total" if args.stat == "sum" else f"average per {name}"
    on = "on" if freq in STEPS else "in"
    lines.append(f"{label}: {word} {fmt(total)} {unit}{change}; last actual {fmt(series[-1][1])} "
                 f"{on} {fmt_time(last_t, freq)}.")
    peak = max(range(h), key=lambda k: fc["point"][k])
    low, high = fc["q"].get(0.05), fc["q"].get(0.95)
    rng = f", 90% range {fmt(low[peak])}–{fmt(high[peak])}" if low and high else ""
    lines.append(f"Peak {fmt(fc['point'][peak])} {on} {fmt_time(future[peak], freq)}{rng}.")
    if low and high:
        lines.append(f"Final {name} ({fmt_time(future[-1], freq)}): {fmt(fc['point'][-1])}, "
                     f"90% range {fmt(low[-1])}–{fmt(high[-1])}.")

    bt = res.get("backtest")
    if bt and "model_mae" in bt and "baseline_mae" in bt:
        m, b = bt["model_mae"], bt["baseline_mae"]
        verdict = (f"{1 - m / b:.0%} lower error" if m < b else "same error" if m == b
                   else f"{m / b - 1:.0%} higher error") if b else "baseline was exact"
        cover = f"; its 90% range held {bt['covered']}/{h} actuals" if bt.get("covered") is not None else ""
        lines.append(f"Backtest, last {h} {name}s held out: {args.provider} MAE {fmt(m)} vs {fmt(b)} for "
                     f"{res['baseline_name']}: {verdict}{cover}.")
    elif res.get("backtest_note"):
        lines.append(f"No backtest: {res['backtest_note']}.")
    if res["missing"]:
        lines.append(f"⚠️ {res['missing']} missing {name}s in the history; the model treats the series as regular.")
    if res["model"] is None and res["baseline"]:
        lines.append(f"Showing the {res['baseline_name']} baseline only.")
    if res["ledger"]:
        lines.append(f"Recorded in Gnomon ledger {res['ledger']} as {res['series_id']}.")
    return lines


# ── Chart ────────────────────────────────────────────────────────────────────

def draw_chart(args, series, freq, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import seaborn as sns

    h = res["horizon"]
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
    palette = sns.color_palette("deep")
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
    bt = res.get("backtest")
    context = series[-max(args.context or 4 * h, h + 1):]
    if bt and bt.get("model"):
        ax.axvspan(series[bt["start_index"] - 1][0], series[-1][0], color="0.5", alpha=0.08, lw=0,
                   label="Backtest window")
    ax.plot([t for t, _ in context], [v for _, v in context], color=palette[0], lw=2, label="Actual")

    last_t = series[-1][0]
    future = [step(last_t, freq, k) for k in range(1, h + 1)]
    anchor = [last_t] + future  # join the forecast to the last actual
    model, base = res["model"], res["baseline"]
    if model:
        q = model["q"]
        if 0.05 in q and 0.95 in q:
            ax.fill_between(future, q[0.05], q[0.95], color=palette[1], alpha=0.15, lw=0, label="90% range")
        if 0.25 in q and 0.75 in q:
            ax.fill_between(future, q[0.25], q[0.75], color=palette[1], alpha=0.3, lw=0, label="50% range")
        ax.plot(anchor, [series[-1][1]] + model["point"], color=palette[1], lw=2, label=f"Forecast: {args.provider}{members(model)}")
    if base:
        ax.plot(anchor, [series[-1][1]] + base["point"], color="0.45", lw=1.2, ls="--",
                label=f"Baseline: {res['baseline_name']}")
    if bt and bt.get("model"):
        start = bt["start_index"]
        ax.plot([series[start - 1][0]] + [t for t, _ in series[start:]],
                [series[start - 1][1]] + bt["model"]["point"], color=palette[1], lw=1.5, ls=":",
                label=f"Backtest forecast (MAE {fmt(bt['model_mae'])})")
    ax.axvline(last_t, color="0.3", lw=1, ls=(0, (2, 3)))
    sub = f"Next {h} {UNIT_NAMES[freq]}s · Gnomon + {args.provider if model else res['baseline_name']}"
    if model and len(model["models_used"]) > 1:
        sub += f" ({len(model['models_used'])} models)"
    if bt and "model_mae" in bt and "baseline_mae" in bt:
        sub += f" · backtest MAE {fmt(bt['model_mae'])} vs baseline {fmt(bt['baseline_mae'])}"
    fig.text(0.012, 0.975, args.title, fontsize=17, fontweight="bold", va="top")
    fig.text(0.012, 0.915, sub, fontsize=11, color="0.35", va="top")
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt(v)))
    ax.set_ylabel(args.unit)
    ax.set_xlabel("")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=10, frameon=False)
    sns.despine(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def default_chart_path(slug):
    # Hermes's image cache is always an allowed attachment root, even in strict media mode.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return hermes_home() / "cache" / "images" / "forecast-report" / f"{slug}-{stamp}.png"


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "series"


def list_models(args):
    cmd = [*gnomon_command(args.gnomon), "capabilities"]
    if args.providers_config:
        cmd += ["--providers-config", args.providers_config]
    try:
        providers = json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout)["providers"]
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError) as exc:
        print(f"Could not list providers: {exc}")
        return 1
    modes = {"ephemeris/ensemble": "Ephemeris: all models combined", "ephemeris": "Ephemeris: router picks one model"}
    rows = [(name, modes[name]) for name in modes if name in providers]
    rows += [(name, "Ephemeris model") for name in sorted(providers)
             if name.startswith("ephemeris/") and name not in modes]
    rows += [(name, "local" + (", with ranges" if (info.get("capabilities") or {}).get("quantiles") else ", no ranges"))
             for name, info in sorted(providers.items()) if not name.startswith("ephemeris")]
    if not any(name.startswith("ephemeris") for name, _ in rows):
        print("Ephemeris is not connected (use the connect-ephemeris skill).")
    width = max(len(name) for name, _ in rows)
    print("\n".join(f"{name.ljust(width)}  {what}" for name, what in rows))
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("csv", nargs="?", help="CSV with a timestamp column and a numeric column")
    p.add_argument("--time-col")
    p.add_argument("--value-col")
    p.add_argument("--agg", choices=("sum", "mean"), help="combine rows that share a timestamp")
    p.add_argument("--freq", choices=sorted(SEASONS), help="h, D, W, MS, QS or YS (default: inferred)")
    p.add_argument("--horizon", type=int, help="steps to forecast (default: 48h, 14D, 8W, 6MS, 4QS, 3YS)")
    p.add_argument("--stat", choices=("sum", "mean"), default="sum",
                   help="headline over the horizon: sum for flows (sales, tokens), mean for levels (price, temperature)")
    p.add_argument("--title", help="chart and summary title (default: the value column)")
    p.add_argument("--unit", help="unit label (default: the value column)")
    p.add_argument("--provider", default="ephemeris/ensemble",
                   help="ephemeris/ensemble (all models combined), ephemeris (router picks one), "
                        "ephemeris/<model>, or any other Gnomon provider; see --list-models")
    p.add_argument("--list-models", action="store_true", help="list the providers this Gnomon can use and exit")
    p.add_argument("--providers-config", help="Gnomon providers TOML (default: saved Ephemeris connection)")
    p.add_argument("--gnomon", help="Gnomon command (default: $GNOMON_CMD, then gnomon on PATH)")
    p.add_argument("--ledger-path", help="default: $HERMES_HOME/data/forecast-report/ledger.db")
    p.add_argument("--no-ledger", action="store_true", help="do not record the forecasts")
    p.add_argument("--no-backtest", action="store_true", help="skip the holdout backtest (saves model calls)")
    p.add_argument("--context", type=int, help="observed points to show (default: 4 × horizon)")
    p.add_argument("--out", help="chart path (default: $HERMES_HOME/cache/images/forecast-report/)")
    p.add_argument("--no-chart", action="store_true")
    p.add_argument("--json", action="store_true", help="machine-readable output instead of the summary")
    args = p.parse_args(argv)
    if args.list_models:
        return list_models(args)
    if not args.csv:
        p.error("the CSV path is required")

    try:
        series, time_col, value_col = load_series(args.csv, args.time_col, args.value_col, args.agg)
        freq = args.freq or infer_freq([t for t, _ in series])
        if args.horizon is not None and args.horizon < 1:
            raise InputError("--horizon must be at least 1")
    except (InputError, OSError) as exc:
        print(f"Input error: {exc}")
        return 2
    default_title = value_col.replace("_", " ").strip()
    args.title = args.title or default_title[:1].upper() + default_title[1:]
    args.unit = args.unit or value_col
    args.slug = slugify(args.title)

    res = run_all(args, series, freq)
    res["missing"] = missing_steps([t for t, _ in series], freq)
    chart, chart_error = None, None
    if not args.no_chart and (res["model"] or res["baseline"]):
        try:
            chart = draw_chart(args, series, freq, res, Path(args.out).expanduser().resolve() if args.out
                               else default_chart_path(args.slug))
        except ImportError as exc:
            chart_error = f"chart not drawn: {exc.name or exc} is not installed (pip install seaborn)"
        except Exception as exc:  # the summary is still worth delivering
            chart_error = f"chart not drawn: {exc}"

    if args.json:
        def path_json(fc):
            return fc and {**fc, "q": {str(k): v for k, v in fc["q"].items()}}
        bt = res.get("backtest")
        print(json.dumps({
            "title": args.title, "unit": args.unit, "time_col": time_col, "value_col": value_col,
            "freq": freq, "horizon": res["horizon"], "observations": len(series),
            "last_timestamp": series[-1][0].isoformat(), "missing_steps": res["missing"],
            "future_timestamps": [step(series[-1][0], freq, k).isoformat() for k in range(1, res["horizon"] + 1)],
            "provider": args.provider, "model": path_json(res["model"]),
            "baseline_name": res["baseline_name"], "baseline": path_json(res["baseline"]),
            "backtest": bt and {k: (path_json(v) if k in ("model", "baseline") else
                                    v.isoformat() if k == "start" else v) for k, v in bt.items()},
            "errors": res["errors"] + ([chart_error] if chart_error else []),
            "ledger": res["ledger"], "series_id": res["series_id"], "chart": str(chart) if chart else None,
        }, indent=2))
    else:
        lines = summarise(args, series, freq, res)
        if chart_error:
            lines.append(f"⚠️ {chart_error}")
        if chart:
            lines.append(f"MEDIA:{chart}")
        print("\n".join(lines))
    return 0 if (res["model"] or res["baseline"]) else 1


if __name__ == "__main__":
    sys.exit(main())

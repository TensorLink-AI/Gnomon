"""End-to-end checks for skills/forecast-report with a fake quantile provider."""
import csv
import importlib.util
import json
import math
import os
import sqlite3
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/forecast-report/scripts/forecast_report.py"
spec = importlib.util.spec_from_file_location("forecast_report", SCRIPT)
fr = importlib.util.module_from_spec(spec)
sys.modules["forecast_report"] = fr
spec.loader.exec_module(fr)
HAS_SEABORN = importlib.util.find_spec("seaborn") is not None

FAKE = '''
from gnomon import ForecastResult
from gnomon.forecast_adapter import ForecastAdapterError
def forecast(req):
    if req.season != 1:  # same contract as Gnomon's Ephemeris adapter
        raise ForecastAdapterError("Ephemeris accepts a frequency hint, not an explicit seasonal period")
    pts = tuple(req.history[-7 + k % 7] for k in range(req.horizon))
    qs = tuple({q: p * (0.7 + 0.6 * q) for q in req.quantiles} for p in pts) if req.quantiles else None
    return ForecastResult(pts, qs, timestamps=req.future_timestamps, series_id=req.series_id, unit=req.unit)
'''
ENSEMBLE = '''
from gnomon import ForecastResult
def forecast(req):
    pts = tuple(req.history[-7 + k % 7] for k in range(req.horizon))
    qs = tuple({q: p * (0.7 + 0.6 * q) for q in req.quantiles} for p in pts)
    return ForecastResult(pts, qs, timestamps=req.future_timestamps, series_id=req.series_id, unit=req.unit,
                          metadata={"service": {"mode": "ensemble", "models_used": ["chronos2", "timesfm"]}})
'''
CONFIG = 'schema_version = 1\n[providers.fake]\nkind = "callable"\nentrypoint = "fakeprov:forecast"\n' \
         'capabilities = {quantiles = true}\n[providers.ens]\nkind = "callable"\nentrypoint = "fakeens:forecast"\n' \
         'capabilities = {quantiles = true}\n'


@pytest.fixture
def home(tmp_path):
    (tmp_path / "fakeprov.py").write_text(FAKE)
    (tmp_path / "fakeens.py").write_text(ENSEMBLE)
    (tmp_path / "providers.toml").write_text(CONFIG)
    with open(tmp_path / "sales.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "store", "units_sold"])
        for k in range(84):
            d = date(2026, 6, 1) + timedelta(days=k)
            w.writerow([d.isoformat(), "A", 200 + k + (80 if d.weekday() >= 5 else 0)])
    return tmp_path


def run(home, *args, provider="fake", csv_name="sales.csv"):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(home)]),
           "HERMES_HOME": str(home / "hermes")}
    cmd = [sys.executable, str(SCRIPT), str(home / csv_name), "--provider", provider,
           "--providers-config", str(home / "providers.toml"), "--gnomon", f"{sys.executable} -m gnomon", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300, cwd=home)
    return proc.returncode, proc.stdout


def test_report_summarises_forecast_backtest_and_ledger(home):
    code, out = run(home, "--title", "Store A units", "--unit", "units")
    assert code == 0, out
    lines = out.splitlines()
    assert lines[0] == "📈 Store A units: next 14 days"
    assert lines[1].startswith("fake: total ") and "units" in lines[1]
    assert "Backtest, last 14 days held out: fake MAE 10.5 vs 10.5 for seasonal naive (same weekday last week): same error" in out
    assert "90% range held 14/14 actuals" in out
    assert "⚠️" not in out.replace("⚠️ chart not drawn", "")
    con = sqlite3.connect(home / "hermes/data/forecast-report/ledger.db")
    try:  # the live model and baseline forecasts only; backtests are not recorded
        assert con.execute("SELECT count(*) FROM executions").fetchone()[0] == 2
    finally:
        con.close()
    if HAS_SEABORN:
        media = lines[-1].removeprefix("MEDIA:")
        assert lines[-1].startswith("MEDIA:") and Path(media).is_file()
        assert Path(media).is_relative_to(home / "hermes/cache/images/forecast-report")
    else:
        assert "is not installed (pip install seaborn)" in out and "MEDIA:" not in out


def test_failed_model_is_reported_and_baseline_shown(home):
    code, out = run(home, "--no-ledger", "--no-chart", provider="ephemeris/chronos2")
    assert code == 0, out
    lines = out.splitlines()
    assert lines[1].startswith("⚠️ ephemeris/chronos2 forecast failed")
    assert lines[3].startswith("seasonal naive (same weekday last week): total ")
    assert "Showing the seasonal naive (same weekday last week) baseline only." in out


def test_json_output_and_mean_headline(home):
    code, out = run(home, "--json", "--no-ledger", "--no-chart", "--horizon", "7")
    data = json.loads(out)
    assert code == 0 and data["freq"] == "D" and data["horizon"] == 7 and data["value_col"] == "units_sold"
    assert len(data["model"]["point"]) == 7 and set(data["model"]["q"]) == {str(q) for q in fr.QUANTILES}
    assert data["future_timestamps"][0].startswith("2026-08-24")
    assert data["backtest"]["model_mae"] == 7 and data["errors"] == []
    code, out = run(home, "--no-ledger", "--no-chart", "--stat", "mean")
    assert "fake: average per day " in out


def test_ensemble_members_are_named_and_models_listed(home):
    code, out = run(home, "--no-ledger", "--no-chart", provider="ens")
    assert code == 0 and out.splitlines()[1].startswith("ens (chronos2 + timesfm): total "), out
    code, out = run(home, "--list-models")
    assert code == 0 and "Ephemeris is not connected" in out
    assert "fake             local, with ranges" in out and "seasonal_naive   local, no ranges" in out
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HERMES_HOME": str(home / "hermes")}
    out = subprocess.run([sys.executable, str(SCRIPT), str(home / "sales.csv"), "--no-ledger", "--no-chart",
                          "--providers-config", str(home / "providers.toml"), "--gnomon", f"{sys.executable} -m gnomon"],
                         capture_output=True, text=True, env=env, timeout=300).stdout
    assert out.splitlines()[1].startswith("⚠️ ephemeris/ensemble forecast failed"), out  # the default provider


def test_input_problems_are_explained(home):
    (home / "dupes.csv").write_text("day,v\n2026-01-01,1\n2026-01-01,2\n2026-01-02,3\n2026-01-03,4\n")
    code, out = run(home, csv_name="dupes.csv")
    assert code == 2 and "pass --agg sum or --agg mean" in out
    (home / "words.csv").write_text("day,v\nMonday,1\nTuesday,2\nWednesday,3\n")
    code, out = run(home, csv_name="words.csv")
    assert code == 2 and "timestamp" in out


def test_requests_give_models_no_season_and_months_step_by_calendar():
    series = [(datetime(2025, m, 1, tzinfo=timezone.utc), float(m)) for m in range(1, 13)]
    series += [(datetime(2026, m, 1, tzinfo=timezone.utc), float(m)) for m in range(1, 13)]
    assert fr.infer_freq([t for t, _ in series]) == "MS"
    model = fr.build_request(series, 3, "MS", "s", "u", fr.QUANTILES)
    baseline = fr.build_request(series, 3, "MS", "s", "u", season=fr.baseline_season("MS", len(series)))
    assert "season" not in model and baseline["season"] == 12
    assert model["future_timestamps"] == ["2027-01-01T00:00:00+00:00", "2027-02-01T00:00:00+00:00",
                                          "2027-03-01T00:00:00+00:00"]
    assert fr.step(datetime(2026, 1, 31), "MS") == datetime(2026, 2, 28)
    assert fr.baseline_season("D", 10) == 1  # under two weeks: last value
    hours = [datetime(2026, 9, 1, tzinfo=timezone.utc) + timedelta(hours=k) for k in range(48) if k not in (5, 6)]
    assert fr.infer_freq(hours) == "h" and fr.missing_steps(hours, "h") == 2


def test_gaps_are_flagged(home):
    rows = [r for i, r in enumerate((home / "sales.csv").read_text().splitlines()) if i not in (10, 11, 12)]
    (home / "gappy.csv").write_text("\n".join(rows) + "\n")
    code, out = run(home, "--no-ledger", "--no-chart", csv_name="gappy.csv")
    assert code == 0 and "⚠️ 3 missing days in the history" in out


@pytest.mark.skipif(not HAS_SEABORN, reason="seaborn not installed")
def test_chart_is_drawn_for_hourly_series(home):
    with open(home / "load.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "load_MW"])
        for k in range(24 * 10):
            t = datetime(2026, 9, 1, tzinfo=timezone.utc) + timedelta(hours=k)
            w.writerow([t.strftime("%Y-%m-%dT%H:%M:%SZ"), round(500 + 200 * math.sin(2 * math.pi * k / 24), 1)])
    out_png = home / "chart.png"
    code, out = run(home, "--no-ledger", "--stat", "mean", "--out", str(out_png), csv_name="load.csv")
    assert code == 0 and out.splitlines()[0] == "📈 Load MW: next 48 hours"
    assert out.splitlines()[-1] == f"MEDIA:{out_png}" and out_png.stat().st_size > 10_000

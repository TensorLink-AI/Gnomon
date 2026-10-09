"""Vanta trading agent (integrations/vanta): policy, data, backtest, paper and live cycles.

Offline: synthetic 5m candles and a fake Vanta miner REST server. No orders leave
the process and no remote forecasts are made.
"""
from datetime import datetime, timedelta, timezone
import io
import json
import math
from pathlib import Path
import random
import sys
import urllib.error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vanta"))

from vanta_agent import config as config_mod  # noqa: E402
from vanta_agent import policy  # noqa: E402
from vanta_agent.agent import Agent, resume  # noqa: E402
from vanta_agent.backtest import run as backtest  # noqa: E402
from vanta_agent.market_data import Candle, hourly  # noqa: E402
from vanta_agent.state import Book  # noqa: E402
from vanta_agent.vanta_client import Order, order_uuid  # noqa: E402

PAIRS = ("BTCUSDC", "ETHUSDC")


def _now():
    return datetime.now(timezone.utc).replace(minute=2, second=0, microsecond=0)


def write_candles(directory, now, hours=220, seed=3, drift=0.0):
    """GARCH-like 5m candles for each pair, ending at the last closed hour before `now`."""
    directory.mkdir(parents=True, exist_ok=True)
    end = now.replace(minute=0)
    for i, pair in enumerate(PAIRS):
        rng, var, price = random.Random(seed + i), 4e-7, 100.0 * (i + 1)
        rows = ["open_time,close"]
        for k in range(hours * 12):
            shock = rng.gauss(0, 1) * math.sqrt(var)
            var = 1e-8 + 0.05 * shock * shock + 0.93 * var
            price *= math.exp(shock + drift)
            t = end - timedelta(minutes=5 * (hours * 12 - k))
            rows.append(f"{t.isoformat()},{price}")
        (directory / f"{pair}.csv").write_text("\n".join(rows) + "\n")


def make_config(tmp_path, mode="paper", direction="local/momentum", **risk):
    risk_lines = "\n".join(f"{k} = {v}" for k, v in risk.items())
    extra = ""
    if mode == "live":
        extra = 'promotion_record = "promotion.json"\npaper_ledger = "state/paper.db"\n'
    path = tmp_path / "agent.toml"
    path.write_text(f"""
[agent]
mode = "{mode}"
account = "test-miner"
state_dir = "state"
{extra}
[universe]
pairs = {list(PAIRS)!r}

[market_data]
source = "csv"
csv_dir = "data"
stale_after_minutes = 90

[forecast]
direction_provider = "{direction}"
vol_model = "vol/ewma"
history_hours = 120
horizon_hours = 2

[risk]
{risk_lines}
""".replace("'", '"'))
    return config_mod.load(path)


# -- policy -------------------------------------------------------------------------
def test_plan_orders_respects_vanta_unidirectional_positions():
    plan = lambda cur, tgt: [(o.order_type, o.leverage) for o in
                             policy.plan_orders("BTCUSDC", cur, tgt, min_rebalance=0.02)]
    assert plan(0.0, 0.3) == [("LONG", 0.3)]
    assert plan(0.3, 0.5) == [("LONG", 0.2)]
    assert plan(0.3, 0.1) == [("SHORT", 0.2)]  # reduces the long
    assert plan(-0.3, -0.1) == [("LONG", 0.2)]  # reduces the short
    assert plan(0.3, -0.2) == [("FLAT", None), ("SHORT", 0.2)]  # flip = close then open
    assert plan(0.3, 0.0) == [("FLAT", None)]
    assert plan(0.3, 0.31) == []  # under the rebalance threshold
    assert plan(0.0, 0.0004) == []  # under Vanta's 0.001 minimum
    assert policy.apply_order(0.3, Order("BTCUSDC", "SHORT", 0.5)) == 0.0  # oversize opposing order closes


def test_target_leverage_needs_edge_beyond_costs(tmp_path):
    cfg = make_config(tmp_path)
    sig = lambda median: type("S", (), dict(sigma_1h_bps=50.0, median_ret_bps=median, horizon=2))()
    hurdle = policy.cost_hurdle_bps(cfg.costs, 2)
    assert policy.target_leverage(sig(hurdle * 0.9), cfg.risk, cfg.costs)[0] == 0.0
    lev, why = policy.target_leverage(sig(200.0), cfg.risk, cfg.costs)
    assert lev == pytest.approx(min(cfg.risk.max_position_leverage, 8.0 / 50.0))
    assert policy.target_leverage(sig(-200.0), cfg.risk, cfg.costs)[0] < 0
    capped = policy.apply_portfolio_cap({"A": 1.0, "B": -1.0}, cfg.risk)
    assert sum(abs(v) for v in capped.values()) <= cfg.risk.max_portfolio_leverage


def test_config_rejects_limits_outside_vanta_rules(tmp_path):
    with pytest.raises(ValueError, match="tier-2"):
        make_config(tmp_path, max_position_leverage=1.5)
    with pytest.raises(ValueError, match="elimination"):
        make_config(tmp_path, intraday_drawdown_halt=0.05)
    with pytest.raises(ValueError, match="promotion_record"):
        (tmp_path / "x.toml").write_text('[agent]\nmode = "live"\n')
        config_mod.load(tmp_path / "x.toml")


# -- data ---------------------------------------------------------------------------
def test_hourly_drops_in_progress_hour_and_restarts_after_gaps():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = [Candle(t0 + timedelta(minutes=5 * k), 100 + k) for k in range(12 * 5)]
    bars = [b for b in bars if not (t0 + timedelta(hours=2) <= b.open_time < t0 + timedelta(hours=3))]
    bars += [Candle(t0 + timedelta(hours=5, minutes=5 * k), 200.0) for k in range(11)]  # in progress
    series = hourly(bars)
    assert [t.hour for t in series.times] == [4, 5]  # hours 3 and 4 after the gap; hour 5 incomplete
    assert series.closes[-1] == 100 + 12 * 5 - 1


def test_hyperliquid_fetch_parses_candles_and_drops_open_bar():
    from vanta_agent.market_data import fetch_hyperliquid
    now = datetime(2026, 1, 1, 1, 2, tzinfo=timezone.utc)
    start = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    rows = [{"t": start + 300_000 * k, "c": str(100 + k)} for k in range(13)]  # 13th bar still open
    seen = {}

    def opener(request, timeout=None):
        seen.update(json.loads(request.data))
        return FakeVanta._reply(rows)
    candles = fetch_hyperliquid("BTCUSDC", hours=1, now=now, url="http://hl/info", opener=opener)
    assert seen["req"]["coin"] == "BTC" and seen["req"]["interval"] == "5m"
    assert len(candles) == 12 and candles[-1].close == 111.0
    assert hourly(candles).closes == (111.0,)


# -- backtest -----------------------------------------------------------------------
def test_backtest_reports_strategy_against_references(tmp_path):
    now = _now()
    write_candles(tmp_path / "data", now, drift=2e-5)
    report = backtest(make_config(tmp_path, mode="backtest"), test_hours=36, now=now)
    for name in ("strategy", "vol_sized_long", "flat"):
        assert set(report[name]) >= {"net_return", "max_intraday_drawdown", "vanta_eliminated"}
    assert report["flat"]["net_return"] == 0.0
    assert report["remote_calls"] == 0
    for name in ("strategy", "vol_sized_long"):  # hourly returns include trading costs
        net, sharpe = report[name]["net_return"], report[name]["sharpe_annualised"]
        assert sharpe is None or abs(net) < 1e-4 or (net > 0) == (sharpe > 0)


def test_backtest_refuses_unbudgeted_remote_calls(tmp_path):
    cfg = make_config(tmp_path, mode="backtest", direction="ephemeris")
    with pytest.raises(ValueError, match="billable"):
        backtest(cfg, test_hours=10, now=_now())


# -- paper --------------------------------------------------------------------------
def test_paper_cycle_records_decisions_and_is_idempotent_per_hour(tmp_path):
    now = _now()
    write_candles(tmp_path / "data", now, drift=3e-4)  # strong drift so momentum trades
    cfg = make_config(tmp_path)
    agent = Agent(cfg, clock=lambda: now)
    first = agent.cycle()
    assert set(first["pairs"]) == set(PAIRS)
    for pair, row in first["pairs"].items():
        assert row["decision_id"] and row["execution_id"]
        intent = next((tmp_path / "state/journal/paper").rglob(f"{pair}-*.intent.json"))
        assert json.loads(intent.read_text())["mode"] == "paper"
    assert any(first["positions"].values())
    again = agent.cycle()
    assert all("already decided" in row["skipped"] for row in again["pairs"].values())


def test_paper_cycle_halts_and_flattens_on_intraday_drawdown(tmp_path):
    now = _now()
    write_candles(tmp_path / "data", now, drift=3e-4)
    cfg = make_config(tmp_path)
    book_path = tmp_path / "state" / "book-paper.json"
    book_path.parent.mkdir(parents=True)
    book = Book(equity=100.0, day=now.date().isoformat(), day_open_equity=104.0, peak_eod_equity=104.0,
                positions={"BTCUSDC": 0.2})
    book.save(book_path)
    report = Agent(cfg, clock=lambda: now).cycle()
    assert report["halted"] and report["positions"] == {}
    assert report["pairs"]["BTCUSDC"]["orders"] == ["FLAT"]


def test_resume_keeps_the_peak_and_requires_a_line_below_eight_percent(tmp_path):
    cfg = make_config(tmp_path)
    path = tmp_path / "state" / "book-paper.json"
    path.parent.mkdir(parents=True)
    Book(equity=94.0, day_open_equity=94.0, peak_eod_equity=100.0, halted_until="manual").save(path)
    with pytest.raises(ValueError):
        resume(cfg, eod_halt=0.05)  # already 6% down
    out = resume(cfg, eod_halt=0.07)
    assert out["eod_drawdown"] == pytest.approx(0.06)
    assert Book.load(path, 0).peak_eod_equity == 100.0


# -- live -----------------------------------------------------------------------------
class FakeVanta:
    """Stands in for vanta_api/miner_rest_server.py."""

    def __init__(self, fail_first_submit=False):
        self.orders, self.processed, self.fail_first = [], {}, fail_first_submit

    def __call__(self, request, timeout=None):
        path = request.full_url.split("8088", 1)[1]
        if path == "/api/health":
            return self._reply({"status": "ok"})
        if path == "/api/submit-order":
            body = json.loads(request.data)
            self.orders.append(body)
            self.processed[body["order_uuid"]] = body
            if self.fail_first:
                self.fail_first = False
                raise TimeoutError("timed out")  # processed, but the reply was lost
            return self._reply({"success": True, "order_uuid": body["order_uuid"]})
        uuid_ = path.rsplit("/", 1)[1]
        if uuid_ in self.processed:
            return self._reply({"order_uuid": uuid_, "status": "completed"})
        raise urllib.error.HTTPError(request.full_url, 404, "nf", {}, io.BytesIO(b'{"status":"not_found"}'))

    @staticmethod
    def _reply(body):
        response = io.BytesIO(json.dumps(body).encode())
        response.status = 200
        response.__enter__ = lambda *a: response
        return type("R", (), {"__enter__": lambda s: response, "__exit__": lambda s, *a: None})()


def _live(tmp_path, monkeypatch, fake):
    from vanta_agent.vanta_client import VantaClient
    now = _now()
    write_candles(tmp_path / "data", now, drift=3e-4)
    cfg = make_config(tmp_path, mode="live")
    monkeypatch.setattr("vanta_agent.agent.record_trade_decision", _record_without_promotion)
    client = VantaClient("http://127.0.0.1:8088", api_key="k", opener=fake)
    return Agent(cfg, client=client, clock=lambda: now, sleep=lambda s: None), now


def _record_without_promotion(ledger, journal_path, **kwargs):
    # Promotion verification is tested in test_trading_skill.py; here we exercise execution.
    from trade_decisions import record_trade_decision
    kwargs.update(mode="paper", promotion_record=None, paper_ledger=None)
    return record_trade_decision(ledger, journal_path, **kwargs)


def test_live_cycle_submits_deterministic_uuids(tmp_path, monkeypatch):
    (tmp_path / "state").mkdir()
    from gnomon import TemporalLedger
    TemporalLedger(tmp_path / "state" / "paper.db")
    fake = FakeVanta()
    agent, _ = _live(tmp_path, monkeypatch, fake)
    report = agent.cycle()
    assert fake.orders and all(o["execution_type"] == "MARKET" for o in fake.orders)
    for pair, row in report["pairs"].items():
        if row["orders"]:
            assert row["results"] == ["completed"] * len(row["orders"])
    intent = json.loads(next((tmp_path / "state/journal/live").rglob("*.intent.json")).read_text())
    assert order_uuid(intent["client_order_id"], 0) in fake.processed or not intent["action"]["orders"]


def test_live_timeout_blocks_pair_until_reconciled(tmp_path, monkeypatch):
    (tmp_path / "state").mkdir()
    from gnomon import TemporalLedger
    TemporalLedger(tmp_path / "state" / "paper.db")
    fake = FakeVanta(fail_first_submit=True)
    agent, now = _live(tmp_path, monkeypatch, fake)
    report = agent.cycle()
    assert report["needs_reconcile"], report
    pair = report["needs_reconcile"][0]
    assert report["pairs"][pair]["results"] == ["uncertain"]
    position_before = report["positions"].get(pair, 0.0)
    sent = len(fake.orders)
    # Next cycle: the order-status lookup finds it processed; the book applies it once.
    agent.clock = lambda: now + timedelta(hours=1)
    agent.market = agent.market  # same data; the event for the new hour is decided fresh
    follow = agent.cycle()
    assert pair not in follow["needs_reconcile"]
    pending = json.loads(next((tmp_path / "state/journal/live").rglob(f"{pair}-*.orders.json")).read_text())
    assert pending[0]["status"] == "completed"
    assert position_before == 0.0  # not applied while unknown
    signed = pending[0]["leverage"] * (1 if pending[0]["order_type"] == "LONG" else -1)
    assert follow["positions"][pair] == pytest.approx(signed)  # applied exactly once
    assert len(fake.orders) == sent  # same hour: already decided, nothing resent


# -- strategy files and offline research ----------------------------------------------
from vanta_agent import research, strategy  # noqa: E402
from vanta_agent.guide import guide  # noqa: E402


def lab_config(tmp_path, *, hours=420, drift=6e-5, champion='direction_provider = "rw"', extra=""):
    now = _now()
    write_candles(tmp_path / "data", now, hours=hours, drift=drift)
    make_config(tmp_path, direction="rw")
    text = (tmp_path / "agent.toml").read_text()
    text = text.replace("[agent]\n", '[agent]\nstrategy = "strategies/champion.toml"\n', 1)
    (tmp_path / "agent.toml").write_text(text + extra)
    (tmp_path / "strategies").mkdir()
    (tmp_path / "strategies/champion.toml").write_text(f'[strategy]\nname = "s"\n\n[forecast]\n{champion}\n')
    cfg = config_mod.load(tmp_path / "agent.toml")
    research.snapshot(cfg, hours=hours, now=now)
    return cfg


def test_strategy_revision_tracks_parameters_not_metadata(tmp_path):
    cfg = make_config(tmp_path)
    child = strategy.with_overrides(cfg, {"risk.z_full": 2.0})
    assert child.revision != cfg.revision and child.strategy_parent == cfg.revision
    assert strategy.diff(cfg, child) == {"risk.z_full": [1.0, 2.0]}
    path = strategy.dump(child, tmp_path / "s.toml", description="anything")
    assert strategy.apply_strategy(cfg, path).revision == child.revision  # round trip
    shadowed = strategy.with_overrides(cfg, {"forecast.shadow_vol_models": ["vol/har"]})
    assert shadowed.revision == cfg.revision  # shadows observe, they do not trade
    with pytest.raises(ValueError, match="tier-2"):
        strategy.with_overrides(cfg, {"risk.max_position_leverage": 3.0})
    with pytest.raises(ValueError, match="unknown"):
        strategy.with_overrides(cfg, {"risk.leverage_boost": 1})


def test_block_bootstrap_bound():
    kw = dict(block=24, samples=500, alpha=0.05, seed=1)
    assert research.block_bootstrap_lcb([1.0] * 100, **kw) == pytest.approx(1.0)
    rng = random.Random(0)
    assert research.block_bootstrap_lcb([rng.gauss(0, 1) for _ in range(300)], **kw) < 0


def test_research_compare_promote_and_holdout_budget(tmp_path):
    cfg = lab_config(tmp_path)
    lab = research.Lab(cfg)
    challenger = strategy.with_overrides(cfg, {"forecast.direction_provider": "local/momentum"})
    with pytest.raises(ValueError, match="No accepted tune comparison"):
        lab.promote(cfg, challenger, cfg.path(cfg.agent.strategy))
    record = lab.compare(cfg, challenger)
    assert record["verdict"] == "accept", record["reason"]  # trending data: momentum goes long
    out = lab.promote(cfg, challenger, cfg.path(cfg.agent.strategy))
    assert out["promoted"] and out["window"] == "holdout"
    lab.close()
    promoted = config_mod.load(tmp_path / "agent.toml")
    assert promoted.revision == challenger.revision and promoted.strategy_parent == cfg.revision
    assert (tmp_path / "research/champions").glob("s-*.toml")
    kinds = [t["kind"] for t in research.read_trials(tmp_path / "research/trials.jsonl")]
    assert kinds == ["compare", "holdout", "promote"]
    # The holdout is spent after max_holdout_looks, whatever their outcomes.
    lab = research.Lab(promoted)
    while lab.holdout_looks() < cfg.research.max_holdout_looks:
        lab._log(dict(kind="holdout", window="holdout"))
    child = strategy.with_overrides(promoted, {"risk.target_vol_bps_per_hour": 10.0})
    lab._log(dict(kind="compare", window="tune", champion=promoted.revision, challenger=child.revision,
                  verdict="accept"))
    with pytest.raises(ValueError, match="no longer"):
        lab.promote(promoted, child, tmp_path / "scratch-champion.toml")
    lab.close()


def test_sweep_is_one_change_at_a_time_with_corrected_alpha(tmp_path):
    cfg = lab_config(tmp_path)
    lab = research.Lab(cfg)
    out = lab.sweep(cfg, {"risk.z_full": [0.5, 1.0, 2.0], "forecast.horizon_hours": [2, 8]})
    lab.close()
    assert out["candidates"] == 3  # unchanged values (z_full=1.0, horizon=2) are not candidates
    assert out["alpha_per_candidate"] == pytest.approx(cfg.research.alpha / 3)
    assert all(len(row["change"]) == 1 for row in out["results"])
    assert all(Path(row["candidate"]).exists() for row in out["results"])


def test_research_refuses_edited_data(tmp_path):
    cfg = lab_config(tmp_path, hours=300)
    csv_path = cfg.path(cfg.research.dataset_dir) / "BTCUSDC.csv"
    lines = csv_path.read_text().splitlines()
    lines[5] = lines[5].rsplit(",", 1)[0] + ",1.0"
    csv_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="changed since the snapshot"):
        research.Lab(cfg)


def test_forecast_cache_reuses_replies_and_enforces_remote_budget(tmp_path):
    class Session:
        calls = 0

        def capabilities(self):
            return {"providers": {"ephemeris": {"revision": None}, "rw": {"revision": "rw-v1"}}}

        def forecast(self, provider, request):
            Session.calls += 1
            return {"status": "ok", "result": {"point": [1.0]}, "provider": provider}

    from vanta_agent.backtest import ForecastCache, RemoteBudgetExceeded
    cache = ForecastCache(Session(), tmp_path / "c.sqlite", max_remote_calls=1)
    cache.forecast("ephemeris", {"history": [1, 2], "horizon": 1})
    cache.forecast("ephemeris", {"history": [1, 2], "horizon": 1})  # cached: free
    assert Session.calls == 1 and cache.remote_calls == 1
    with pytest.raises(RemoteBudgetExceeded):
        cache.forecast("ephemeris", {"history": [1, 3], "horizon": 1})
    again = ForecastCache(Session(), tmp_path / "c.sqlite", max_remote_calls=0)
    assert again.forecast("ephemeris", {"history": [1, 2], "horizon": 1})["status"] == "ok"  # persisted


# -- Gnomon-guided suggestions from realised outcomes ---------------------------------
def test_guide_ranks_shadow_models_on_realised_outcomes(tmp_path):
    end = _now()
    write_candles(tmp_path / "data", end, hours=220)
    make_config(tmp_path, direction="rw")
    text = (tmp_path / "agent.toml").read_text().replace(
        "[forecast]\n", '[forecast]\nshadow_vol_models = ["vol/har"]\nshadow_direction_providers = ["local/momentum"]\n')
    (tmp_path / "agent.toml").write_text(text.replace('vol_model = "vol/ewma"', 'vol_model = "vol/last"'))
    cfg = config_mod.load(tmp_path / "agent.toml")
    for h in range(16, -1, -1):
        now = end - timedelta(hours=h)
        Agent(cfg, clock=lambda now=now: now).cycle()
    report = guide(cfg, min_origins=10)
    vol = report["roles"]["vol"]
    assert vol["scored_origins"] >= 20 and vol["versus_current"]["vol/har"]["origins"] >= 20
    assert set(report["roles"]["direction"]["versus_current"]) == {"local/momentum"}
    for suggestion in report["suggestions"]:  # each is one parameter away and loads as a strategy
        child = strategy.apply_strategy(cfg, suggestion["candidate"])
        assert len(strategy.diff(cfg, child)) == 1
    assert guide(cfg, min_origins=10_000)["suggestions"] == []  # too little evidence: no suggestion

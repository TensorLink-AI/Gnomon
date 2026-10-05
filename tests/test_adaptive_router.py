"""Cost-aware adaptive routing: pure selection, replay without look-ahead, and live ledger routing."""

from datetime import datetime, timedelta, timezone
import json

import pytest

from gnomon import AdapterCapabilities, ForecastResult, GnomonSession, TemporalLedger
from gnomon.adaptive_router import replay_router, runs_shadow, select_provider, validate_policy
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.ids import FixedClock


def policy(**changes):
    return validate_policy({"candidates": ["a", "b"], "baseline": "base", "min_origins": 3,
                            "recent_origins": 10, **changes})


# --- configuration -------------------------------------------------------------------------

def test_policy_defaults_and_rejections():
    p = policy()
    assert (p["metric"], p["shadow_every"], p["identity_policy"], p["min_improvement"]) == ("mae", 1, "attested", 0.02)
    for bad in ({"unknown": 1}, {"candidates": ["base"]}, {"candidates": []}, {"min_origins": 11},
                {"metric": "mape"}, {"costs": {"zzz": {"usd_per_call": 1}}}, {"costs": {"a": {"usd_per_call": -1}}},
                {"cost_weights": {"euros": 1}}, {"identity_policy": "trust_me"}):
        with pytest.raises(ForecastAdapterError):
            policy(**bad)
    with pytest.raises(ForecastAdapterError, match="baseline itself violates"):
        policy(costs={"base": {"latency_seconds": 5}}, limits={"max_latency_seconds": 1})


# --- pure selection ------------------------------------------------------------------------

def test_insufficient_evidence_serves_the_baseline():
    choice = select_provider({"base": 1.0, "a": 0.5, "b": 0.9}, 2, policy())
    assert (choice["provider"], choice["reason"], choice["evidence_based"]) == ("base", "insufficient_evidence", False)


def test_better_candidate_is_served_only_past_the_improvement_threshold():
    assert select_provider({"base": 1.0, "a": 0.5, "b": 0.9}, 5, policy())["provider"] == "a"
    near = select_provider({"base": 1.0, "a": 0.99, "b": 1.2}, 5, policy())
    assert (near["provider"], near["reason"]) == ("base", "evidence_improvement_below_threshold")


def test_dollar_cost_can_outweigh_accuracy():
    costly = policy(costs={"a": {"usd_per_call": 0.01}}, cost_weights={"usd": 100})  # $0.01 ~ 100% error
    choice = select_provider({"base": 1.0, "a": 0.5, "b": 0.8}, 5, costly)
    assert choice["provider"] == "b"
    row = next(r for r in choice["table"] if r["provider"] == "a")
    assert row["utility"] == pytest.approx(0.5 + 1.0)


def test_latency_limit_excludes_a_provider_regardless_of_accuracy():
    limited = policy(costs={"a": {"latency_seconds": 3}}, limits={"max_latency_seconds": 1})
    choice = select_provider({"base": 1.0, "a": 0.1, "b": 0.7}, 5, limited)
    assert choice["provider"] == "b"
    assert next(r for r in choice["table"] if r["provider"] == "a")["excluded"] == "max_latency_seconds"


def test_shadow_sampling_is_deterministic():
    assert runs_shadow("s", "t", 1) and not runs_shadow("s", "t", 0)
    picks = [runs_shadow("s", f"2026-01-01T00:{m:02d}:00Z", 4) for m in range(60)]
    assert picks == [runs_shadow("s", f"2026-01-01T00:{m:02d}:00Z", 4) for m in range(60)]
    assert 5 <= sum(picks) <= 25


# --- replay --------------------------------------------------------------------------------

def _folds(n, good_until):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    folds = []
    for i in range(n):
        origin = start + timedelta(minutes=5 * i)
        a_error, b_error = (0.1, 1.0) if i < good_until else (1.0, 0.1)
        folds.append({"origin": origin.isoformat(), "target_time": (origin + timedelta(minutes=5)).isoformat(),
                      "actual": 0.0, "points": {"base": 0.5, "a": a_error, "b": b_error}})
    return folds


def test_replay_adapts_when_the_best_provider_changes():
    result = replay_router(_folds(100, 50), policy(recent_origins=5))
    assert result["served_counts"]["a"] > 30 and result["served_counts"]["b"] > 30
    assert result["router_score"] < min(result["fixed_provider_scores"].values())


def test_replay_uses_only_matured_outcomes():
    folds = _folds(10, 10)
    for f in folds:  # Targets far in the future: nothing matures within the replay.
        f["target_time"] = "2027-01-01T00:00:00+00:00"
    result = replay_router(folds, policy())
    assert result["served_counts"] == {"base": 10, "a": 0, "b": 0}


def test_broad_replay_selects_last_candidate_and_preserves_live_limit():
    names = [f"model_{i}" for i in range(31)]
    config = {"baseline": "base", "candidates": names, "min_origins": 3}
    broad = validate_policy(config, replay=True)
    with pytest.raises(ForecastAdapterError, match="1-7 candidates"):
        validate_policy(config)
    with pytest.raises(ForecastAdapterError, match="1-31 candidates"):
        validate_policy({**config, "candidates": names + ["extra"]}, replay=True)
    folds = _folds(10, 10)
    for fold in folds:
        fold["points"] = {"base": 0.5, **{name: 2.0 for name in names}, names[-1]: 0.1}
    result = replay_router(folds, broad)
    assert result["served_counts"][names[-1]] == 7
    assert result["served_counts"]["base"] == 3
    assert len(result["fixed_provider_scores"]) == 32
    for fold in folds:
        fold["target_time"] = "2027-01-01T00:00:00+00:00"
    unmatured = replay_router(folds, broad)
    assert unmatured["served_counts"]["base"] == 10
    assert unmatured["served_counts"][names[-1]] == 0


# --- live routing through the ledger -------------------------------------------------------

def _drive(ledger, session, steps, provider="router/demo", via_mcp=False, extra=None, requests=None):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = [100.0 + i + (0.2 if i % 2 else -0.2) for i in range(20)]  # noisy trend: last_value wins
    replies = []
    for t in range(steps):
        origin = start + timedelta(hours=len(history) - 1)
        ledger.clock = FixedClock(origin + timedelta(minutes=1))
        request = {"history": list(history), "horizon": 1, "series_id": "trend", "unit": "u",
                   "timestamps": [(start + timedelta(hours=i)).isoformat() for i in range(len(history))],
                   "future_timestamps": [(origin + timedelta(hours=1)).isoformat()], **(extra(t) if extra else {})}
        if requests is not None:
            requests.append(request)
        reply = session.call("gnomon_forecast", {"provider": provider, "request": request}, compact=False) \
            if via_mcp else session.forecast(provider, request)
        replies.append(reply)
        history.append(history[-1] + 1.0 + (0.4 if len(history) % 2 else -0.4))
        ledger.clock = FixedClock(origin + timedelta(hours=1, minutes=1))
        ledger.append_actual(series_id="trend", unit="u", valid_time=(origin + timedelta(hours=1)).isoformat(),
                             value=history[-1], source_available_at=(origin + timedelta(hours=1)).isoformat(),
                             source_ref="test")
    return replies


@pytest.fixture
def live(tmp_path):
    def build(router, **kwargs):
        ledger = TemporalLedger(tmp_path / "ledger.db", clock=FixedClock(datetime(2026, 1, 1, tzinfo=timezone.utc)))
        base = GnomonSession.from_config(ledger=ledger)
        session = GnomonSession(base.engine, ledger=ledger, routers={"router/demo": router})
        if kwargs.get("register_unversioned"):
            session.engine.register("remote", lambda r: ForecastResult(
                point=(r.history[-1],) * r.horizon, timestamps=r.future_timestamps, series_id=r.series_id, unit=r.unit),
                capabilities=AdapterCapabilities(), lifecycle="pretrained")
        return ledger, session
    return build


def test_live_router_learns_from_recorded_outcomes(live):
    ledger, session = live({"candidates": ["last_value"], "baseline": "historical_mean",
                            "min_origins": 3, "recent_origins": 5})
    replies = _drive(ledger, session, 6)
    first, last = replies[0]["routing"], replies[-1]["routing"]
    assert (first["served_provider"], first["reason"]) == ("historical_mean", "insufficient_evidence")
    assert (last["served_provider"], last["reason"]) == ("last_value", "evidence_and_cost_favour_candidate")
    assert last["matched_origins"] >= 3 and last["shadow_execution_ids"]
    record = ledger.decision(last["routing_decision_id"])
    assert record["policy"]["kind"] == "adaptive_route/1"
    assert record["action"]["served_provider"] == "last_value"
    assert record["execution_ids"][0] == replies[-1]["execution_id"]
    assert session.capabilities()["ledger"]["routers"]["router/demo"]["baseline"] == "historical_mean"


def test_live_router_respects_declared_cost(live):
    ledger, session = live({"candidates": ["last_value"], "baseline": "historical_mean", "min_origins": 3,
                            "recent_origins": 5, "costs": {"last_value": {"usd_per_call": 0.05}},
                            "cost_weights": {"usd": 100}})
    last = _drive(ledger, session, 6)[-1]["routing"]
    assert last["served_provider"] == "historical_mean"
    assert last["reason"] == "evidence_improvement_below_threshold"
    assert last["declared_usd_this_call"] == pytest.approx(0.05)


def test_mcp_forecast_routes_too(live):
    ledger, session = live({"candidates": ["last_value"], "baseline": "historical_mean",
                            "min_origins": 3, "recent_origins": 5})
    last = _drive(ledger, session, 5, via_mcp=True)[-1]
    assert last["routing"]["served_provider"] == "last_value"


def test_unversioned_provider_requires_explicit_disclosed_admission(live):
    ledger, session = live({"candidates": ["remote"], "baseline": "historical_mean", "min_origins": 3,
                            "recent_origins": 5}, register_unversioned=True)
    with pytest.raises(ForecastAdapterError, match="prospective_unattested"):
        _drive(ledger, session, 1)
    ledger2, session2 = live({"candidates": ["remote"], "baseline": "historical_mean", "min_origins": 3,
                              "recent_origins": 5, "identity_policy": "prospective_unattested"},
                             register_unversioned=True)
    last = _drive(ledger2, session2, 6)[-1]["routing"]
    assert last["served_provider"] == "remote"
    assert last["unattested_providers"] == ["remote"]


def test_router_requires_a_ledger_and_forecast_identity(live):
    no_ledger = GnomonSession.from_config()
    unrouted = GnomonSession(no_ledger.engine, routers={"r": {"candidates": ["last_value"], "baseline": "historical_mean"}})
    with pytest.raises(ForecastAdapterError, match="ledger"):
        unrouted.forecast("r", {"history": [1.0, 2.0], "horizon": 1, "series_id": "s",
                                "timestamps": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
                                "future_timestamps": ["2026-01-01T02:00:00Z"]})
    ledger, session = live({"candidates": ["last_value"], "baseline": "historical_mean"})
    with pytest.raises(ForecastAdapterError, match="series_id"):
        session.forecast("router/demo", {"history": [1.0, 2.0, 3.0], "horizon": 1})


def test_router_policy_appears_in_the_config_schema():
    from gnomon.session import configuration_schema
    schema = configuration_schema()["properties"]["routers"]["additionalProperties"]
    assert set(schema["required"]) == {"candidates", "baseline"}


def test_documented_router_config_is_valid():
    import re
    import tomllib
    from pathlib import Path
    from gnomon.recovery import _matches
    from gnomon.session import configuration_schema
    text = (Path(__file__).resolve().parents[1] / "docs/adaptive-routing.md").read_text()
    config = tomllib.loads(re.search(r"```toml\n(.*?)\n```", text, re.S).group(1))
    assert _matches(config, configuration_schema())
    router = validate_policy(config["routers"]["router/sales"])
    assert router["identity_policy"] == "prospective_unattested"
    assert router["costs"]["ephemeris/chronos-2"] == {"usd_per_call": 0.002, "latency_seconds": 1.5}
    assert router["pool"] == {"series": ["store-2/sales", "store-3/sales"], "own_weight": 2.0}
    assert router["context"]["thresholds"] == [0.8, 1.25]
    memory_block = tomllib.loads(re.findall(r"```toml\n(.*?)\n```", text, re.S)[1])
    memory = memory_block["routers"]["router/sales"]["memory"]
    full = {**config, "routers": {"router/sales": {**config["routers"]["router/sales"], "memory": memory}}}
    assert _matches(full, configuration_schema())
    assert validate_policy(full["routers"]["router/sales"])["memory"]["k"] == 64


def _toml_router(tmp_path):
    config = tmp_path / "router.toml"
    config.write_text('schema_version = 1\nledger_path = "ledger.db"\n[routers."router/x"]\n'
                      'candidates = ["last_value"]\nbaseline = "historical_mean"\n')
    return config


def _one_request():
    return {"history": [1.0, 2.0, 3.0, 4.0], "horizon": 1, "series_id": "s", "unit": "u",
            "timestamps": [f"2026-01-01T0{i}:00:00Z" for i in range(4)], "future_timestamps": ["2026-01-01T04:00:00Z"]}


def test_router_from_toml_config_and_cli(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    config = _toml_router(tmp_path)
    with GnomonSession.from_config(config) as session:
        assert session.forecast("router/x", _one_request())["routing"]["served_provider"] == "historical_mean"
    with GnomonSession.from_config(config, discovery_only=True) as session:
        assert session.capabilities()["ledger"]["routers"]["router/x"]["candidates"] == ["last_value"]
    cli = subprocess.run([sys.executable, "-m", "gnomon", "infer", "--providers-config", str(config),
                          "--provider", "router/x", "--request", json.dumps(_one_request())],
                         capture_output=True, text=True, timeout=60,
                         env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
    assert cli.returncode == 0, cli.stdout + cli.stderr
    assert json.loads(cli.stdout)["routing"]["served_provider"] == "historical_mean"


def test_router_keeps_serving_past_the_comparison_cap(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ledger = TemporalLedger(tmp_path / "ledger.db", clock=FixedClock(start))
    base = GnomonSession.from_config(ledger=ledger)
    names = [f"m{i}" for i in range(7)]
    for i, name in enumerate(names):
        base.engine.register(name, lambda r, i=i: ForecastResult(point=(r.history[-1] + i,) * r.horizon,
            timestamps=r.future_timestamps, series_id=r.series_id, unit=r.unit), revision=f"{name}-v1")
    session = GnomonSession(base.engine, ledger=ledger, routers={"r": {
        "candidates": names, "baseline": "historical_mean", "recent_origins": 10, "min_origins": 3,
        "lookback_seconds": 30 * 86400}})
    history = [float(i) for i in range(5)]  # rises by 1 per step: m1 (last + 1) is exact
    for t in range(140):  # 8 providers x 140 origins = 1120 executions in the lookback window
        origin = start + timedelta(minutes=5 * (len(history) - 1))
        ledger.clock = FixedClock(origin + timedelta(seconds=1))
        request = {"history": list(history), "horizon": 1, "series_id": "cap", "unit": "u",
                   "timestamps": [(start + timedelta(minutes=5 * i)).isoformat() for i in range(len(history))],
                   "future_timestamps": [(origin + timedelta(minutes=5)).isoformat()]}
        routing = session.forecast("r", request)["routing"]
        history.append(history[-1] + 1.0)
        ledger.clock = FixedClock(origin + timedelta(minutes=5, seconds=1))
        ledger.append_actual(series_id="cap", unit="u", valid_time=request["future_timestamps"][0], value=history[-1],
                             source_available_at=request["future_timestamps"][0], source_ref="t")
    assert routing["reason"] != "evidence_unavailable"
    assert routing["served_provider"] == "m1" and routing["matched_origins"] == 10  # m1 predicts last + 1: exact


def test_unreadable_evidence_serves_the_baseline_and_says_why(live, monkeypatch):
    import gnomon.adaptive_router as router
    ledger, session = live({"candidates": ["last_value"], "baseline": "historical_mean"})
    monkeypatch.setattr(router, "ledger_evidence", lambda *a, **k: (_ for _ in ()).throw(ForecastAdapterError("boom")))
    routing = _drive(ledger, session, 1)[-1]["routing"]
    assert (routing["served_provider"], routing["reason"]) == ("historical_mean", "evidence_unavailable")
    assert routing["evidence_error"] == "boom"


def test_evidence_window_scales_with_shadow_sampling():
    from gnomon.adaptive_router import _evidence_start
    stamps = [f"2026-01-01T00:{m:02d}:00+00:00" for m in range(0, 55, 5)]
    origin = datetime(2026, 1, 1, 0, 50, tzinfo=timezone.utc)
    every = _evidence_start(policy(recent_origins=10, shadow_every=1, lookback_seconds=10**7), origin, stamps)
    sampled = _evidence_start(policy(recent_origins=10, shadow_every=5, lookback_seconds=10**7), origin, stamps)
    assert origin - every == timedelta(minutes=5 * 30)
    assert origin - sampled == timedelta(minutes=5 * 150)


def test_replay_respects_lookback():
    folds = _folds(40, 40)
    short = replay_router(folds, policy(recent_origins=10, lookback_seconds=600))  # two 5-minute origins
    assert short["served_counts"]["base"] == 40  # never min_origins (3) matured origins inside 10 minutes


# --- pooled evidence and context conditioning ----------------------------------------------

from gnomon.adaptive_router import context_label, pooled_scores  # noqa: E402


def test_pool_and_context_validation():
    with pytest.raises(ForecastAdapterError):
        policy(pool={"series": []})
    with pytest.raises(ForecastAdapterError):
        policy(pool={"series": ["x"], "own_weight": 0.5})
    with pytest.raises(ForecastAdapterError):
        policy(context={"feature": "volatility_ratio", "thresholds": [2.0, 1.0]})
    with pytest.raises(ForecastAdapterError):
        policy(context={"feature": "volatility_ratio", "thresholds": [1.0], "short_window": 10, "long_window": 5})
    spec = policy(context={"feature": "trend", "thresholds": [0.0]})["context"]
    assert spec["on"] == "differences" and len(spec["spec_id"]) == 12


def test_context_label_uses_only_the_supplied_history():
    spec = policy(context={"feature": "volatility_ratio", "on": "values", "short_window": 3, "long_window": 6,
                           "thresholds": [1.2]})["context"]
    calm_then_wild = [0, 0.1, -0.1, 0.1, 5, -5]
    wild_then_calm = [5, -5, 5, 0.1, -0.1, 0.1]
    assert context_label(calm_then_wild, spec) == "volatility_ratio:bin1"
    assert context_label(wild_then_calm, spec) == "volatility_ratio:bin0"
    assert context_label([1, 2], spec) is None and context_label([1.0] * 6, spec) is None


def test_pooled_scores_normalise_each_series_to_its_baseline():
    rows = [{"series_id": "big", "origin": f"2026-01-01T00:{i:02d}:00", "losses": {"base": 100.0, "a": 50.0, "b": 100.0}}
            for i in range(5)] + \
           [{"series_id": "small", "origin": f"2026-01-01T00:{i:02d}:00", "losses": {"base": 1.0, "a": 1.0, "b": 0.5}}
            for i in range(5)]
    pooled = policy(pool={"series": ["big"]})
    scores, matched, used = pooled_scores(rows, pooled, "small")
    assert matched == 10 and used == ["big", "small"]
    assert scores == pytest.approx({"base": 1.0, "a": 0.75, "b": 0.75})  # scale does not dominate
    weighted, _, _ = pooled_scores(rows, policy(pool={"series": ["big"], "own_weight": 3}), "small")
    assert weighted["b"] < weighted["a"]  # own series counts three times


def _series_folds(series, start_minute, n, best):
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    out = []
    for i in range(n):
        origin = base_time + timedelta(minutes=5 * (start_minute + i))
        errors = {"base": 0.5, "a": 0.1 if best == "a" else 1.0, "b": 0.1 if best == "b" else 1.0}
        out.append({"series_id": series, "origin": origin.isoformat(),
                    "target_time": (origin + timedelta(minutes=5)).isoformat(), "actual": 0.0, "points": errors})
    return out


def test_pool_gives_a_new_series_a_warm_start():
    folds = _series_folds("old", 0, 60, "a") + _series_folds("new", 50, 10, "a")
    alone = replay_router(folds, policy(recent_origins=10))
    pooled = replay_router(folds, policy(recent_origins=10, pool={"series": ["old"]}))
    assert alone["per_series"]["new"]["router"] > pooled["per_series"]["new"]["router"]
    assert pooled["per_series"]["new"]["router"] == pytest.approx(0.1)  # served "a" from its first forecast


def _regime_folds(n, block=20):
    """The better model depends on the volatility regime, which is visible in pre-origin history."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    folds = []
    # Pre-fold history alternates calm/wild blocks too, so the long window spans both regimes.
    history = [(3.0 if i % 2 else -3.0) if (i // block) % 2 else (0.1 if i % 2 else -0.1) for i in range(200)]
    for i in range(n):
        wild = (i // block) % 2 == 1
        value = (3.0 if i % 2 else -3.0) if wild else (0.1 if i % 2 else -0.1)
        origin = start + timedelta(minutes=5 * i)
        errors = {"base": 0.5, "a": 0.1 if wild else 1.0, "b": 1.0 if wild else 0.1}
        folds.append({"series_id": "s", "origin": origin.isoformat(), "actual": value,
                      "target_time": (origin + timedelta(minutes=5)).isoformat(),
                      "points": {p: value + e for p, e in errors.items()}})
    return folds, {"s": history}


def test_context_conditioning_beats_unconditioned_routing_when_regimes_differ():
    folds, histories = _regime_folds(400)
    # volatility_ratio is relative: its long window must span regimes to separate them absolutely.
    context = {"feature": "volatility_ratio", "on": "values", "short_window": 4, "long_window": 200, "thresholds": [0.5]}
    plain = replay_router(folds, policy(recent_origins=10, min_origins=3), histories)
    conditioned = replay_router(folds, policy(recent_origins=10, min_origins=3, context=context), histories)
    assert conditioned["router_score"] < plain["router_score"]
    assert conditioned["evidence_levels"].get("context", 0) > 0


def test_live_pool_and_context_through_the_ledger(live):
    router = {"candidates": ["last_value"], "baseline": "historical_mean", "min_origins": 3, "recent_origins": 5,
              "pool": {"series": ["trend"]},
              "context": {"feature": "trend", "short_window": 3, "long_window": 6, "thresholds": [0.0]}}
    ledger, session = live(router)
    _drive(ledger, session, 6)  # series "trend" builds evidence that last_value wins
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    origin = start + timedelta(hours=40)
    ledger.clock = FixedClock(origin + timedelta(minutes=1))
    history = [500.0 + i + (0.2 if i % 2 else -0.2) for i in range(10)]
    reply = session.forecast("router/demo", {
        "history": history, "horizon": 1, "series_id": "fresh", "unit": "u",
        "timestamps": [(origin - timedelta(hours=9 - i)).isoformat() for i in range(10)],
        "future_timestamps": [(origin + timedelta(hours=1)).isoformat()]})
    routing = reply["routing"]
    assert routing["served_provider"] == "last_value"  # a brand-new series, served from pooled evidence
    assert routing["series_used"] == ["trend"] and routing["context_label"] == "trend:bin1"
    assert routing["evidence_level"] == "context"
    record = ledger.decision(routing["routing_decision_id"])
    assert record["inputs"]["context_label"] == "trend:bin1" and record["inputs"]["context_spec_id"]


# --- review regressions: refused evidence, point-in-time cutoff, replay instants and metric ---

def test_router_does_not_rank_evidence_the_ledger_refused(live):
    ledger, session = live({"candidates": ["last_value"], "baseline": "historical_mean",
                            "min_origins": 3, "recent_origins": 10})
    last = _drive(ledger, session, 9, extra=lambda t: {"season": 2} if t >= 4 else {})[-1]["routing"]
    assert (last["served_provider"], last["reason"]) == ("historical_mean", "evidence_incompatible")
    assert "incompatible" in last["evidence_error"]


def test_repeating_a_historical_request_sees_only_outcomes_known_at_its_origin(live):
    ledger, session = live({"candidates": ["last_value"], "baseline": "historical_mean",
                            "min_origins": 3, "recent_origins": 5})
    requests = []
    replies = _drive(ledger, session, 8, requests=requests)
    original = replies[2]["routing"]
    assert original["reason"] == "insufficient_evidence"  # two matured origins; its own outcome is later
    repeated = session.forecast("router/demo", requests[2])["routing"]  # ledger clock is now much later
    assert (repeated["served_provider"], repeated["reason"], repeated["matched_origins"]) == \
        (original["served_provider"], original["reason"], original["matched_origins"])
    assert repeated["evidence_as_of"] == original["evidence_as_of"]


def test_replay_compares_instants_not_timestamp_strings():
    utc = _folds(40, 20)
    shifted = []
    for i, fold in enumerate(utc):  # same instants, written with a -05:00 offset on odd folds
        if i % 2:
            fold = {**fold, **{k: datetime.fromisoformat(fold[k]).astimezone(timezone(timedelta(hours=-5))).isoformat()
                               for k in ("origin", "target_time")}}
        shifted.append(fold)
    assert replay_router(shifted, policy()) == replay_router(utc, policy())


def test_replay_rmsle_matches_live_rules_and_names_its_aggregation():
    folds = _folds(10, 10)
    folds[4] = {**folds[4], "points": {**folds[4]["points"], "b": -1.0}}
    result = replay_router(folds, policy(metric="rmsle"))
    assert result["aggregation"] == "mean_rmsle_over_folds"
    assert result["excluded_folds"] == {"rmsle_negative_prediction_requires_explicit_clip_zero": 1}
    assert sum(result["served_counts"].values()) == 10  # still routed; just not scored


def test_replay_scores_multi_step_folds_when_the_whole_horizon_matures():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    folds = [{"series_id": "s", "origin": (start + timedelta(days=i)).isoformat(),
              "target_time": (start + timedelta(days=i + 3)).isoformat(), "actual": [0.0, 0.0, 0.0],
              "points": {"base": [1.0, 1.0, 1.0], "a": [0.0, 0.0, 0.3], "b": [2.0, 2.0, 2.0]}} for i in range(12)]
    result = replay_router(folds, policy(lookback_seconds=10**7))
    assert result["fixed_provider_scores"]["a"] == pytest.approx(0.1)
    # Origin i sees origins <= i - 3 matured (a target dated i is known at i): min_origins=3 from origin 5.
    assert result["served_counts"] == {"base": 5, "a": 7, "b": 0}


def test_replay_labels_regimes_from_timestamped_history_known_at_each_origin():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stamps = [(start + timedelta(days=i)).isoformat() for i in range(40)]
    calm_then_wild = [(t, float(i % 2) if i < 30 else float(10 * (i % 2))) for i, t in enumerate(stamps)]
    spec = {"feature": "volatility_ratio", "on": "values", "short_window": 4, "long_window": 20, "thresholds": [1.3]}
    folds = [{"series_id": "s", "origin": stamps[i], "target_time": stamps[i + 1], "actual": 0.0,
              "points": {"base": 1.0, "a": 1.0, "b": 1.0}} for i in (25, 35)]
    result = replay_router(folds, policy(context=spec), histories={"s": calm_then_wild})
    assert result["context_labels"] == {"volatility_ratio:bin0": 1, "volatility_ratio:bin1": 1}


def test_documented_replay_walkthrough_runs(tmp_path, monkeypatch, capsys):
    import re
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "docs/adaptive-routing.md").read_text()
    section = text[text.index("### Evaluate a router on your own data"):]
    code = re.search(r"```python\n(.*?)\n```", section, re.S).group(1)
    monkeypatch.chdir(tmp_path)
    exec(compile(code, "adaptive-routing.md", "exec"), {"__name__": "walkthrough"})
    lines = dict(line.split(" ", 1) for line in capsys.readouterr().out.strip().splitlines())
    assert set(lines) == {"pooled", "memory"}
    assert "'memory':" in lines["memory"]  # memory evidence was used for some origins

"""Episodic memory options: profiles, context, dedupe, shrinkage, confidence, novelty, hysteresis,
diagnostics, accelerated replay and the memory ablation. Every option is off by default."""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
import random

import pytest

from gnomon.adaptive_router import (choose_from_rows, memory_ablation, replay_router, select_provider,
                                    validate_policy)
from gnomon.episodic_memory import (CONTEXT_PREFIX, PROFILES, compute_features, episodic_evidence, episodic_scores,
                                    validate_memory)
from gnomon.forecast_adapter import ForecastAdapterError
from test_adaptive_router import _drive, live  # noqa: F401  (fixture)
from test_episodic_memory import _series_folds

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _days(n):
    return [(START + timedelta(days=i)).isoformat() for i in range(n)]


# ---------------------------------------------------------------- defaults unchanged

def test_default_spec_id_is_unchanged_by_the_new_options():
    spec = validate_memory({"short_window": 4, "long_window": 14})
    legacy = {k: spec[k] for k in ("features", "short_window", "long_window", "season", "mask_covariate",
                                   "future_covariate")}
    assert spec["spec_id"] == sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()[:12]
    # retrieval/selection options do not change the feature space, so recorded episodes stay usable
    for option in ({"dedupe_seconds": 86400}, {"shrinkage": 5}, {"confidence_z": 1.0},
                   {"novelty_threshold": 3.0}, {"diagnostics": True}):
        assert validate_memory({"short_window": 4, "long_window": 14, **option})["spec_id"] == spec["spec_id"]
    assert validate_memory({"short_window": 4, "long_window": 14, "context": ["mkt"]})["spec_id"] != spec["spec_id"]


def _rows(n_per_series=30, seed=0, context=False):
    rng = random.Random(seed)
    rows = []
    for s in ("s1", "s2", "s3"):
        for i in range(n_per_series):
            f = {"volatility_ratio": rng.gauss(0, 1), "level_shift": rng.gauss(0, 1)}
            if context:
                f[CONTEXT_PREFIX + "mkt"] = rng.choice([0.0, 1.0])
            good = f.get(CONTEXT_PREFIX + "mkt", 1.0 if f["level_shift"] > 0 else 0.0) == 1.0
            rows.append({"series_id": s, "origin": (START + timedelta(days=i)).isoformat(), "features": f,
                         "losses": {"base": 1.0, "a": (0.5 if good else 1.6) * rng.uniform(0.7, 1.3)}})
    return rows


def _policy(**memory):
    return validate_policy({"candidates": ["a"], "baseline": "base", "min_origins": 2, "recent_origins": 50,
                            "pool": {"series": ["s1", "s2", "s3"]},
                            "memory": {"features": ["volatility_ratio", "level_shift"], "short_window": 2,
                                       "long_window": 3, "k": 12, "min_effective_n": 2, **memory}})


def test_evidence_equals_scores_without_options():
    rows = _rows()
    q = {"volatility_ratio": 0.1, "level_shift": 0.7}
    scores, n, neighbours = episodic_scores(rows, _policy(), "s1", q)
    evidence = episodic_evidence(rows, _policy(), "s1", q)
    assert evidence == {"scores": scores, "effective_n": n, "neighbours": neighbours}


# ---------------------------------------------------------------- features

def test_profiles_and_new_features():
    assert validate_memory({"profile": "returns"})["features"] == PROFILES["returns"]
    with pytest.raises(ForecastAdapterError):
        validate_memory({"profile": "returns", "features": ["cv"]})
    with pytest.raises(ForecastAdapterError):
        validate_memory({"profile": "weekly"})
    spec = validate_memory({"features": ["autocorrelation", "skewness", "vol_of_vol", "demand_interval"],
                            "short_window": 4, "long_window": 16})
    alternating = [1.0, -1.0] * 8
    f = compute_features(spec, alternating, _days(16))
    assert f["autocorrelation"] < -0.9 and f["skewness"] == pytest.approx(0.0, abs=1e-12)
    assert f["vol_of_vol"] == pytest.approx(0.0, abs=1e-12)  # same spread in every block
    spiky = [0.0] * 15 + [10.0]
    assert compute_features(spec, spiky, _days(16))["skewness"] > 3
    intermittent = [0.0, 0.0, 0.0, 2.0] * 4
    assert compute_features(spec, intermittent, _days(16))["demand_interval"] == pytest.approx(math.log(4))
    calm_then_wild = [0.1, -0.1, 0.1, -0.1] * 3 + [5.0, -5.0, 5.0, -5.0]
    assert compute_features(spec, calm_then_wild, _days(16))["vol_of_vol"] > 1


def test_context_feature_is_the_latest_past_covariate_value():
    spec = validate_memory({"features": ["level_shift"], "context": ["mkt"], "short_window": 2, "long_window": 4})
    f = compute_features(spec, [1.0, 2.0, 3.0, 4.0], _days(4), past_covariates=[[0.0], [0.0], [2.5], [7.0]],
                         past_covariate_names=["mkt"])
    assert f[CONTEXT_PREFIX + "mkt"] == 7.0
    assert compute_features(spec, [1.0, 2.0, 3.0, 4.0], _days(4))[CONTEXT_PREFIX + "mkt"] is None


def test_context_lets_memory_see_the_state_that_decides():
    # The candidate wins exactly when a market-wide flag is on; the series' own shape says nothing.
    rows = _rows(context=True)
    on = {"volatility_ratio": 0.0, "level_shift": 0.0, CONTEXT_PREFIX + "mkt": 1.0}
    off = {**on, CONTEXT_PREFIX + "mkt": 0.0}
    with_ctx = _policy(context=["mkt"])
    assert episodic_scores(rows, with_ctx, "s1", on)[0]["a"] < 0.7 < 1.0 < episodic_scores(rows, with_ctx, "s1", off)[0]["a"]
    blind = _policy()
    a_on, a_off = episodic_scores(rows, blind, "s1", on)[0]["a"], episodic_scores(rows, blind, "s1", off)[0]["a"]
    assert a_on == a_off  # without context the two states are indistinguishable


def test_replay_reads_context_from_covariates():
    f1, h1 = _series_folds("s1", 30, True, "a")
    policy = validate_policy({"candidates": ["a"], "baseline": "base", "min_origins": 3, "lookback_seconds": 10**7,
                              "memory": {"features": ["level_shift"], "context": ["mkt"], "short_window": 4,
                                         "long_window": 12, "k": 5, "min_effective_n": 2}})
    mkt = [[t, float(i)] for i, (t, _) in enumerate(h1)]
    out = replay_router(f1, policy, {"s1": h1}, covariates={"s1": {"mkt": mkt}}, return_decisions=True)
    assert out["evidence_levels"].get("memory", 0) > 0


# ---------------------------------------------------------------- retrieval and scoring options

def test_dedupe_keeps_one_neighbour_per_series_per_window():
    rows = _rows()
    q = {"volatility_ratio": 0.0, "level_shift": 0.5}
    dense = episodic_evidence(rows, _policy(), "s1", q)
    sparse = episodic_evidence(rows, _policy(dedupe_seconds=10 * 86400), "s1", q)
    for s in ("s1", "s2", "s3"):
        days = sorted(datetime.fromisoformat(n["origin"]) for n in sparse["neighbours"] if n["series_id"] == s)
        assert all((b - a).days >= 10 for a, b in zip(days, days[1:]))
    assert len(sparse["neighbours"]) < len(dense["neighbours"])


def test_shrinkage_and_confidence():
    rows = _rows()
    q = {"volatility_ratio": 0.0, "level_shift": 0.8}
    raw = episodic_evidence(rows, _policy(), "s1", q)["scores"]["a"]
    shrunk = episodic_evidence(rows, _policy(shrinkage=20), "s1", q)
    assert raw < shrunk["scores"]["a"] < 1.0  # pulled toward "no difference", same side
    conf = episodic_evidence(rows, _policy(confidence_z=2.0), "s1", q)
    assert conf["selection_scores"]["a"] > conf["scores"]["a"] and conf["selection_scores"]["base"] == 1.0


def test_diagnostics_report_uncertainty_and_quantiles():
    rows = _rows()
    ev = episodic_evidence(rows, _policy(diagnostics=True), "s1", {"volatility_ratio": 0.0, "level_shift": 0.8})
    d = ev["diagnostics"]
    lo, hi = d["interval90"]["a"]
    assert lo < ev["scores"]["a"] < hi and d["standard_errors"]["base"] == 0
    q = d["loss_ratio_quantiles"]["a"]
    assert q[0.1] <= q[0.5] <= q[0.9] and d["novelty"] is not None


def test_novelty_abstains_on_an_unfamiliar_state():
    rows = _rows()
    familiar = {"volatility_ratio": 0.0, "level_shift": 0.5}
    alien = {"volatility_ratio": 40.0, "level_shift": -40.0}
    policy = _policy(novelty_threshold=3.0)
    assert episodic_evidence(rows, policy, "s1", familiar)["novelty"] < 3.0
    decision = choose_from_rows(rows, policy, "s1", None, alien)
    assert decision["evidence_level"] == "all" and decision["memory_abstained"]["reason"] == "unfamiliar_state"
    assert choose_from_rows(rows, policy, "s1", None, familiar)["evidence_level"] == "memory"


# ---------------------------------------------------------------- hysteresis

def test_switch_penalty_holds_the_incumbent_unless_the_gain_is_large():
    policy = validate_policy({"candidates": ["a"], "baseline": "base", "min_origins": 1, "switch_penalty": 0.1})
    small_gain = {"base": 1.0, "a": 0.95}
    assert select_provider(small_gain, 5, policy)["provider"] == "a"           # no incumbent known
    held = select_provider(small_gain, 5, policy, incumbent="base")
    assert held["provider"] == "base" and held["incumbent"] == "base"
    assert select_provider({"base": 1.0, "a": 0.7}, 5, policy, incumbent="base")["provider"] == "a"
    assert select_provider({"base": 1.0, "a": 1.05}, 5, policy, incumbent="a")["provider"] == "a"  # stays
    assert "switch_penalty" not in validate_policy({"candidates": ["a"], "baseline": "base"})


def _flip_folds(n=120, seed=3):
    """Two providers whose advantage flips at random every origin: pure noise to chase."""
    rng = random.Random(seed)
    folds = []
    for i in range(n):
        origin = START + timedelta(days=i)
        a = 1.0 + rng.choice([-0.3, 0.3])
        folds.append({"series_id": "s1", "origin": origin.isoformat(), "target_time": (origin + timedelta(days=1)).isoformat(),
                      "actual": 0.0, "points": {"base": 1.0, "a": a}})
    return folds


def test_switch_penalty_reduces_churn_in_replay():
    base = {"candidates": ["a"], "baseline": "base", "min_origins": 1, "recent_origins": 3, "lookback_seconds": 10**7}
    def switches(policy):
        served = [d["served"] for d in replay_router(_flip_folds(), validate_policy(policy), return_decisions=True)["decisions"]]
        return sum(a != b for a, b in zip(served, served[1:]))
    assert switches({**base, "switch_penalty": 0.5}) < switches(base)


def test_live_switch_penalty_uses_the_last_served_provider(live):
    ledger, session = live({"candidates": ["last_value"], "baseline": "historical_mean", "min_origins": 3,
                            "recent_origins": 5, "switch_penalty": 0.01})
    replies = _drive(ledger, session, 6)
    last = replies[-1]["routing"]
    assert last["incumbent"] == replies[-2]["routing"]["served_provider"]


# ---------------------------------------------------------------- accelerated replay and ablation

def _memory_scenario():
    f1, h1 = _series_folds("s1", 40, True, "a")
    f2, h2 = _series_folds("s2", 40, False, "base")
    f3, h3 = _series_folds("s3", 40, True, "a")
    late = [f for f in f3 if f["origin"] >= (START + timedelta(days=45)).isoformat()]
    policy = {"candidates": ["a"], "baseline": "base", "min_origins": 3, "recent_origins": 20,
              "lookback_seconds": 10**7, "pool": {"series": ["s1", "s2", "s3"]},
              "memory": {"features": ["zero_share", "volatility_ratio"], "short_window": 4, "long_window": 12,
                         "k": 10, "min_effective_n": 3}}
    return f1 + f2 + late, {"s1": h1, "s2": h2, "s3": h3}, policy


@pytest.mark.parametrize("extra", [{}, {"shrinkage": 4.0, "confidence_z": 1.0}])
def test_accelerated_replay_matches_the_pure_path(extra):
    pytest.importorskip("numpy")
    folds, histories, policy = _memory_scenario()
    policy = validate_policy({**policy, "memory": {**policy["memory"], **extra}})
    pure = replay_router(folds, policy, histories, return_decisions=True)
    fast = replay_router(folds, policy, histories, return_decisions=True, accelerate=True)
    assert [d["served"] for d in fast["decisions"]] == [d["served"] for d in pure["decisions"]]
    assert fast["evidence_levels"] == pure["evidence_levels"]
    assert fast["router_score"] == pytest.approx(pure["router_score"], rel=1e-12)


def test_accelerate_refuses_unsupported_options():
    pytest.importorskip("numpy")
    folds, histories, policy = _memory_scenario()
    policy = validate_policy({**policy, "memory": {**policy["memory"], "dedupe_seconds": 86400}})
    with pytest.raises(ForecastAdapterError, match="accelerate"):
        replay_router(folds, policy, histories, accelerate=True)


def test_memory_ablation_reports_a_paired_verdict():
    folds, histories, policy = _memory_scenario()
    result = memory_ablation(folds, validate_policy(policy), histories, n_boot=300)
    assert result["folds_compared"] == len(folds)
    assert result["memory_score"] < result["no_memory_score"]
    assert result["verdict"] == "memory_better" and result["difference_ci90"][1] < 0
    assert set(result["switch_rate"]) == {"memory", "no_memory"}
    with pytest.raises(ForecastAdapterError):
        memory_ablation(folds, validate_policy({k: v for k, v in policy.items() if k != "memory"}), histories)


def test_ablation_reports_only_the_matched_post_warmup_cohort():
    folds, histories, raw = _memory_scenario()
    policy = validate_policy(raw)
    result = memory_ablation(folds, policy, histories, warmup_origins=20, n_boot=10)
    memory = replay_router(folds, policy, histories, return_decisions=True, decision_losses=True)
    plain = replay_router(folds, {**policy, 'memory': None}, histories, return_decisions=True, decision_losses=True)
    cut = sorted({d['origin'] for d in memory['decisions']})[20]
    for name, replay in [('memory', memory), ('no_memory', plain)]:
        ds = [d for d in replay['decisions'] if d['origin'] >= cut]
        assert result[name+'_score'] == pytest.approx(sum(d['served_loss'] for d in ds)/len(ds))
    assert result['memory_score'] - result['no_memory_score'] == pytest.approx(result['mean_loss_difference'])
    empty = memory_ablation(folds, policy, histories, warmup_origins=1000, n_boot=10)
    assert empty['verdict'] == 'no_comparable_folds'
    assert empty['memory_score'] is None
    for kwargs in ({'warmup_origins':-1}, {'n_boot':0}):
        with pytest.raises(ForecastAdapterError):
            memory_ablation(folds, policy, histories, **kwargs)


def test_accelerated_evidence_keeps_neighbour_details():
    pytest.importorskip('numpy')
    from gnomon.episodic_memory_fast import MemoryIndex
    p = _policy()
    rows = _rows(10)
    index = MemoryIndex(p)
    for r in rows:
        index.add(r)
    query = {'volatility_ratio':.1, 'level_shift':.7}
    origin = (START + timedelta(days=11)).isoformat()
    pure = episodic_evidence(rows, p, 's1', query, origin)
    fast = index.evidence('s1', query, origin, START.isoformat())
    assert fast['scores'] == pytest.approx(pure['scores'])
    for a,b in zip(fast['neighbours'],pure['neighbours'],strict=True):
        assert a.keys() == b.keys()
        for key in ('series_id','origin','best_provider','features','losses'):
            assert a[key] == b[key]
        for key in ('weight','distance','baseline_loss_scale','normalized_losses'):
            assert a[key] == pytest.approx(b[key])

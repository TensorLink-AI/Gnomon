"""Episodic memory for routing: features from the request only, similarity retrieval, replay and live."""

from datetime import datetime, timedelta, timezone
import math

import pytest

from gnomon.adaptive_router import replay_router, validate_policy
from gnomon.episodic_memory import compute_features, episodic_scores, validate_memory
from gnomon.forecast_adapter import ForecastAdapterError
from test_adaptive_router import _drive, live  # noqa: F401  (fixture)

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def spec(**changes):
    return validate_memory({"short_window": 4, "long_window": 14, "season": 7, **changes})


def days(n, start=START):
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


def test_memory_validation_and_spec_identity():
    base = spec()
    assert base["k"] == 32 and base["min_effective_n"] == 8 and len(base["spec_id"]) == 12
    assert spec(k=5)["spec_id"] == base["spec_id"]  # retrieval settings do not change the feature space
    assert spec(long_window=28)["spec_id"] != base["spec_id"]
    for bad in ({"features": ["colour"]}, {"features": ["covariate_share"]}, {"long_window": 4},
                {"unknown": 1}, {"k": 0}, {"own_weight": 0.5}):
        with pytest.raises(ForecastAdapterError):
            spec(**bad)


def test_features_describe_only_the_supplied_window():
    weekly = [10.0 if i % 7 == 0 else 1.0 for i in range(28)]
    f = compute_features(spec(), weekly, days(28))
    assert f["seasonality"] > 0.8 and f["missing_share"] == 0.0
    assert f["zero_share"] == 0.0 and f["length_cycles"] == pytest.approx(math.log(28 / 7))
    flat = compute_features(spec(), [3.0] * 28, days(28))
    assert flat["volatility_ratio"] is None and flat["trend"] is None and flat["cv"] is None  # undefined, not 0
    short = compute_features(spec(), [1.0, 2.0], days(2))
    assert short["seasonality"] is None and short["length_cycles"] == pytest.approx(math.log(2 / 7))


def test_missing_share_from_gaps_or_an_observed_mask():
    stamps = days(30)
    gappy = stamps[:20] + stamps[24:]  # four days absent, inside the last 14 rows
    f = compute_features(spec(), [1.0 + (i % 3) for i in range(26)], gappy)
    assert f["missing_share"] == pytest.approx(1 - 14 / 18)  # rows for days 12-19 and 24-29 span 18 days
    values = [0.0, 0.0, 2.0, 3.0] * 7
    mask = [[0.0], [1.0], [1.0], [1.0]] * 7  # the first zero of each block is a filled gap, not a sale of 0
    masked = compute_features(spec(mask_covariate="observed", long_window=16), values, days(28),
                              past_covariates=mask, past_covariate_names=["observed"])
    assert masked["missing_share"] == pytest.approx(0.25) and masked["zero_share"] == pytest.approx(1 / 3)


def test_future_covariate_share_uses_the_forecast_window():
    s = spec(features=["covariate_share"], future_covariate="promo")
    f = compute_features(s, [1.0] * 20, days(20), future_covariates=[[1.0], [0.0], [1.0], [0.0]],
                         future_covariate_names=["promo"])
    assert f == {"covariate_share": 0.5}
    assert compute_features(s, [1.0] * 20, days(20)) == {"covariate_share": None}


def _policy(**memory):
    return validate_policy({"candidates": ["a"], "baseline": "base", "min_origins": 2, "recent_origins": 50,
                            "pool": {"series": ["s1", "s2", "s3"]},
                            "memory": {"features": ["zero_share", "cv"], "short_window": 2, "long_window": 3,
                                       "k": 6, "min_effective_n": 2, **memory}})


def _episode(series, day, zero_share, a_loss):
    return {"series_id": series, "origin": (START + timedelta(days=day)).isoformat(),
            "features": {"zero_share": zero_share, "cv": 0.0}, "losses": {"base": 1.0, "a": a_loss}}


def test_retrieval_prefers_the_most_similar_episodes():
    rows = [_episode("s1", i, 0.9, 0.2) for i in range(6)] + [_episode("s2", i, 0.1, 3.0) for i in range(6)]
    sparse, _, neighbours = episodic_scores(rows, _policy(), "s3", {"zero_share": 0.85, "cv": 0.0})
    dense, _, _ = episodic_scores(rows, _policy(), "s3", {"zero_share": 0.12, "cv": 0.0})
    assert sparse["a"] < 0.5 < 2.0 < dense["a"]
    assert {n["series_id"] for n in neighbours[:3]} == {"s1"} and neighbours[0]["best_provider"] == "a"
    unknown, n_eff, _ = episodic_scores(rows, _policy(), "s3", {"zero_share": None, "cv": None})
    assert unknown == {} and n_eff == 0  # too few shared features to compare: no retrieval


def test_series_scale_prevents_one_large_series_dominating():
    rows = [{**_episode("s1", i, 0.5, 2000.0), "losses": {"base": 1000.0, "a": 2000.0}} for i in range(4)] + \
           [_episode("s2", i, 0.5, 0.5) for i in range(4)]
    scores, _, _ = episodic_scores(rows, _policy(), "s3", {"zero_share": 0.5, "cv": 0.0})
    assert scores["a"] == pytest.approx((2.0 + 0.5) / 2)  # each series contributes in baseline units


def _series_folds(series, n, zero_share_high, good):
    """Daily one-step folds. Intermittent series (many zeros) favour `good`, the other provider elsewhere."""
    history = [0.0 if (zero_share_high and i % 4) else 1.0 + (i % 3) for i in range(n + 20)]
    folds = []
    for i in range(20, 20 + n):
        origin = START + timedelta(days=i - 1)
        errors = {"base": 1.0, "a": 0.2 if good == "a" else 2.0}
        folds.append({"series_id": series, "origin": origin.isoformat(),
                      "target_time": (origin + timedelta(days=1)).isoformat(), "actual": 0.0, "points": errors})
    return folds, [[t, v] for t, v in zip(days(n + 20), history)]


def test_memory_serves_a_new_series_from_similar_series():
    f1, h1 = _series_folds("s1", 40, True, "a")    # intermittent: a is much better
    f2, h2 = _series_folds("s2", 40, False, "base")  # smooth: a is much worse
    f3, h3 = _series_folds("s3", 40, True, "a")
    late = [f for f in f3 if f["origin"] >= (START + timedelta(days=45)).isoformat()]  # new series joins late
    policy = {"candidates": ["a"], "baseline": "base", "min_origins": 3, "recent_origins": 20,
              "lookback_seconds": 10**7, "pool": {"series": ["s1", "s2", "s3"]}}
    histories = {"s1": h1, "s2": h2, "s3": h3}
    memory = {"features": ["zero_share", "volatility_ratio"], "short_window": 4, "long_window": 12, "k": 10,
              "min_effective_n": 3}
    pooled = replay_router(f1 + f2 + late, validate_policy(policy), histories, return_decisions=True)
    remembered = replay_router(f1 + f2 + late, validate_policy({**policy, "memory": memory}), histories,
                               return_decisions=True)
    served = lambda r: [d["served"] for d in r["decisions"] if d["series_id"] == "s3"]
    # Pooling averages the two kinds (a is not better overall) until s3's own outcomes mature;
    # memory matches s3 to the similar series s1 from its first forecast.
    assert served(pooled)[:3] == ["base"] * 3
    assert served(remembered).count("a") == len(late)
    assert remembered["evidence_levels"]["memory"] > 0


def test_live_memory_records_features_and_reports_neighbours(live):
    ledger, session = live({"candidates": ["last_value"], "baseline": "historical_mean", "min_origins": 3,
                            "recent_origins": 5, "lookback_seconds": 10**7,
                            "memory": {"features": ["volatility_ratio", "trend", "level_shift", "cv"],
                                       "short_window": 3, "long_window": 8, "k": 10, "min_effective_n": 2}})
    replies = _drive(ledger, session, 8)
    last = replies[-1]["routing"]
    assert (last["evidence_level"], last["served_provider"]) == ("memory", "last_value")
    assert last["effective_n"] >= 2 and last["memory_neighbours"][0]["best_provider"] == "last_value"
    inputs = ledger.decision(last["routing_decision_id"])["inputs"]
    assert set(inputs["memory_features"]) == {"volatility_ratio", "trend", "level_shift", "cv"}
    assert len(inputs["memory_spec_id"]) == 12

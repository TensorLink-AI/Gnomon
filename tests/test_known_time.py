"""Outcome knowledge time is honest: three instants stay distinct.

An outcome has a valid time (when it occurred), a known time (when it
reached Gnomon), and sits against a forecast cutoff (what the forecast
could see). The adaptation log used to stamp realised coverage with the
*cutoff* — the one instant at which the outcome provably did not exist —
so an ``--as-of`` replay at the forecast origin could see the realised
coverage of the very horizon being forecast. Outcomes are now stamped
with an explicit per-observation ``known_at`` when the caller supplies
one (a backfill), and with the submission time otherwise; a derived
record inherits the latest contributing knowledge time.
"""

import csv

import pytest

from gnomon.contracts import GnomonError
from gnomon.tracking import TrackingStore


@pytest.fixture
def store(tmp_path):
    return TrackingStore(path=tmp_path / "registry.db")


@pytest.fixture
def forecast_dir(tmp_path):
    d = tmp_path / "forecast_kt"
    d.mkdir()
    rows = [
        {"series": "__default__", "timestamp": "2026-01-01T00:00:00+00:00",
         "point": 100.0, "q10": 90.0, "q50": 100.0, "q90": 110.0},
        {"series": "__default__", "timestamp": "2026-01-01T01:00:00+00:00",
         "point": 105.0, "q10": 95.0, "q50": 105.0, "q90": 115.0},
    ]
    with open(d / "forecast.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["series", "timestamp", "point",
                                          "q10", "q50", "q90"])
        w.writeheader()
        w.writerows(rows)
    return d


CUTOFF = "2025-12-31T00:00:00+00:00"


def _register(store, forecast_dir):
    store.register(
        "kt1", project="kt", cutoff_time=CUTOFF, horizon=2, frequency="h",
        selected_model="drift", naive_error=10.0,
        artifact_path=str(forecast_dir),
    )


def _actuals(known_at=None):
    return [
        (None, "2026-01-01T00:00:00+00:00", 102.0, known_at),
        (None, "2026-01-01T01:00:00+00:00", 108.0, known_at),
    ]


def test_replay_at_the_cutoff_cannot_see_the_outcome(store, forecast_dir):
    """The old stamp (known_time = cutoff) made the realised coverage of
    a horizon visible to a replay at the horizon's own forecast origin."""
    _register(store, forecast_dir)
    results = store.submit_actuals("kt", _actuals())
    assert len(results) == 1

    at_cutoff = store.coverage_outcomes("kt", "__default__", as_of=CUTOFF)
    assert at_cutoff == [], (
        "a replay at the forecast origin saw an outcome that did not "
        "exist yet"
    )
    # One second after the cutoff is still long before submission.
    just_after = store.coverage_outcomes(
        "kt", "__default__", as_of="2025-12-31T00:00:01+00:00")
    assert just_after == []
    now = store.coverage_outcomes("kt", "__default__")
    assert len(now) == 1


def test_unlabelled_outcomes_are_known_at_submission(store, forecast_dir):
    _register(store, forecast_dir)
    store.submit_actuals("kt", _actuals())
    [outcome] = store.coverage_outcomes("kt", "__default__")
    # Submission happened just now; the stamp must be neither the cutoff
    # nor the outcome's valid time.
    assert outcome["known_time"] > "2026-01-02"


def test_backfill_uses_the_latest_contributing_known_at(store, forecast_dir):
    _register(store, forecast_dir)
    actuals = [
        (None, "2026-01-01T00:00:00+00:00", 102.0, "2026-01-01T00:05:00+00:00"),
        (None, "2026-01-01T01:00:00+00:00", 108.0, "2026-01-01T01:05:00+00:00"),
    ]
    store.submit_actuals("kt", actuals)
    [outcome] = store.coverage_outcomes("kt", "__default__")
    assert outcome["known_time"] == "2026-01-01T01:05:00+00:00"
    # Replay between the two knowledge times: the aggregate needs its
    # last input, so it is not yet visible.
    between = store.coverage_outcomes(
        "kt", "__default__", as_of="2026-01-01T00:30:00+00:00")
    assert between == []
    after = store.coverage_outcomes(
        "kt", "__default__", as_of="2026-01-01T02:00:00+00:00")
    assert len(after) == 1


def test_mixed_vintages_fall_back_to_submission_time(store, forecast_dir):
    """One row without known_at became knowable at submission, so the
    derived record did too."""
    _register(store, forecast_dir)
    actuals = [
        (None, "2026-01-01T00:00:00+00:00", 102.0, "2026-01-01T00:05:00+00:00"),
        (None, "2026-01-01T01:00:00+00:00", 108.0, None),
    ]
    store.submit_actuals("kt", actuals)
    [outcome] = store.coverage_outcomes("kt", "__default__")
    assert outcome["known_time"] > "2026-01-02"


def test_known_before_valid_is_refused(store, forecast_dir):
    _register(store, forecast_dir)
    actuals = [
        (None, "2026-01-01T00:00:00+00:00", 102.0, "2025-12-31T23:00:00+00:00"),
    ]
    with pytest.raises(GnomonError) as caught:
        store.submit_actuals("kt", actuals)
    assert caught.value.code == "OUTCOME_KNOWN_BEFORE_VALID"


def test_csv_known_at_column_backfills(store, forecast_dir, tmp_path):
    _register(store, forecast_dir)
    path = tmp_path / "actuals.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "value", "known_at"])
        w.writerow(["2026-01-01T00:00:00+00:00", 102.0,
                    "2026-01-01T00:10:00+00:00"])
        w.writerow(["2026-01-01T01:00:00+00:00", 108.0,
                    "2026-01-01T01:10:00+00:00"])
    results = store.submit_actuals_csv("kt", str(path))
    assert len(results) == 1
    [outcome] = store.coverage_outcomes("kt", "__default__")
    assert outcome["known_time"] == "2026-01-01T01:10:00+00:00"


def test_csv_without_known_at_is_unchanged(store, forecast_dir, tmp_path):
    _register(store, forecast_dir)
    path = tmp_path / "actuals.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "value"])
        w.writerow(["2026-01-01T00:00:00+00:00", 102.0])
        w.writerow(["2026-01-01T01:00:00+00:00", 108.0])
    results = store.submit_actuals_csv("kt", str(path))
    assert len(results) == 1
    [outcome] = store.coverage_outcomes("kt", "__default__")
    assert outcome["known_time"] > "2026-01-02"


def test_mcp_actuals_accept_known_at():
    from gnomon.toolspec import _actual_tuples

    tuples = _actual_tuples([
        {"timestamp": "2026-01-01", "value": 1.0,
         "known_at": "2026-01-02"},
        {"timestamp": "2026-01-01", "value": 1.0, "series": "api",
         "known_at": "2026-01-02"},
        {"timestamp": "2026-01-01", "value": 1.0},
    ])
    assert tuples[0] == (None, "2026-01-01", 1.0, "2026-01-02")
    assert tuples[1] == ("api", "2026-01-01", 1.0, "2026-01-02")
    assert tuples[2] == ("2026-01-01", 1.0)

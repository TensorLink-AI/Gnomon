from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from gnomon.model_assisted import _short_trend_probe, build_model_assisted_lane


def _stamps(count: int) -> list[datetime]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [start + timedelta(days=index) for index in range(count)]


def _hourly_stamps(count: int) -> list[datetime]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [start + timedelta(hours=index) for index in range(count)]


def _assessment(**overrides) -> SimpleNamespace:
    base = dict(
        selection_guardrail_applied=True,
        strongest_baseline="last_value",
        selection_scores={"last_value": 1.0, "theta": .6, "drift": None},
        degraded=True,
        selection_fold_count=1,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_underpowered_selection_publishes_the_scored_candidate_beside_the_baseline() -> None:
    values = [float(v) for v in range(1, 17)]
    lane, disclosure, evidence = build_model_assisted_lane(
        "cpu", values, horizon=8, season=1,
        future_timestamps=_stamps(8)[:8], assessment=_assessment(),
        published_support="degraded", selected_model="last_value")
    assert lane is not None
    assert lane["support"] == "prior_assisted"
    assert lane["selected_model"] == "theta"
    assert len(lane["points"]) == 8
    assert lane["timestamps_match_primary_forecast"] is True
    assert lane["validation"]["basis"] == "single_trailing_holdout"
    assert lane["validation"]["scores_are_evidence_not_a_ranking"] is True
    assert lane["automation_eligible"] is False
    assert lane["primary_forecast_unchanged"] is True
    assert disclosure.code == "model_assisted_lane"
    assert "prior_assisted" in disclosure.message
    assert evidence.kind == "model_assisted_lane"


def test_a_full_horizon_fold_win_earns_conditionally_supported() -> None:
    values = [float(v) for v in range(1, 41)]
    lane, _, _ = build_model_assisted_lane(
        "cpu", values, horizon=4, season=1,
        future_timestamps=_stamps(4), assessment=_assessment(degraded=False),
        published_support="degraded", selected_model="last_value")
    assert lane is not None
    assert lane["support"] == "conditionally_supported"
    assert lane["validation"]["basis"] == "single_selection_fold"
    assert lane["validation"]["out_of_sample_steps"] == 4


def test_a_candidate_that_never_beat_the_baseline_earns_no_lane() -> None:
    values = [float(v) for v in range(1, 17)]
    assessment = _assessment(
        selection_scores={"last_value": .5, "theta": .9, "drift": .8})
    lane, disclosure, evidence = build_model_assisted_lane(
        "cpu", values, horizon=8, season=1,
        future_timestamps=_stamps(8), assessment=assessment,
        published_support="degraded", selected_model="last_value")
    assert lane is None and disclosure is None and evidence is None


def test_short_seasonal_baseline_can_earn_only_the_assisted_lane() -> None:
    cycle = [0.0, 2.0, 8.0, 2.0]
    values = cycle * 3
    assessment = _assessment(selection_scores={
        "last_value": 4.0, "seasonal_naive": 0.0, "theta": 5.0})
    lane, _, _ = build_model_assisted_lane(
        "load", values, horizon=4, season=4,
        future_timestamps=_stamps(4), assessment=assessment,
        published_support="best_effort", selected_model="last_value")
    assert lane is not None
    assert lane["selected_model"] == "seasonal_naive"
    assert lane["points"] == cycle
    assert lane["automation_eligible"] is False
    assert lane["primary_forecast_unchanged"] is True


def test_complete_cycle_prequential_evidence_admits_stable_seasonality() -> None:
    cycle = [100.0, 101.0, 108.0, 102.0, 96.0, 94.0, 97.0, 99.0]
    values = cycle + [value + .1 for value in cycle]
    lane, _, _ = build_model_assisted_lane(
        "load", values, horizon=16, season=8,
        future_timestamps=_stamps(16), assessment=_assessment(),
        published_support="best_effort", selected_model="last_value")
    assert lane is not None
    assert lane["selected_model"] == "seasonal_naive"
    assert lane["validation"]["basis"] == "full_cycle_prequential"
    assert lane["validation"]["complete_phase_coverage"] is True
    assert lane["validation"]["phase_block_wins"] >= 3


def test_complete_cycle_gate_rarely_admits_unrelated_random_walks() -> None:
    import random

    admitted = 0
    for seed in range(200):
        rng = random.Random(seed)
        values = [100.0]
        for _ in range(47):
            values.append(values[-1] + rng.gauss(0, 1))
        lane, _, _ = build_model_assisted_lane(
            "noise", values, horizon=48, season=24,
            future_timestamps=_stamps(48), assessment=_assessment(),
            published_support="best_effort", selected_model="last_value")
        if lane is not None and lane["selected_model"] == "seasonal_naive" \
                and lane["validation"]["basis"] == "full_cycle_prequential":
            admitted += 1
    assert admitted <= 10


def test_short_trend_probe_admits_persistent_growth_and_names_its_boundary() -> None:
    values = [1060.0 + 325.0 * index / 74 for index in range(75)]
    validation = _short_trend_probe(values, horizon=72, season=72)

    assert validation is not None
    assert validation["admitted"] is True
    assert validation["candidate"] == "drift"
    assert validation["basis"] == "non_overlapping_short_horizon_blocks"
    assert validation["comparisons"] == 6
    assert validation["chronological_block_wins"] == 3
    assert validation["maximum_locally_evaluated_lead"] == 9
    assert validation["out_of_sample_steps"] == 54


def test_short_trend_probe_is_conservative_on_no_signal_controls() -> None:
    import random

    admissions = {"random_walk": 0, "white_noise": 0, "stationary_ar": 0}
    for seed in range(200):
        rng = random.Random(seed)
        random_walk = [100.0]
        for _ in range(74):
            random_walk.append(random_walk[-1] + rng.gauss(0, 1))
        white_noise = [100.0 + rng.gauss(0, 5) for _ in range(75)]
        stationary_ar = [100.0]
        for _ in range(74):
            stationary_ar.append(
                100.0 + .8 * (stationary_ar[-1] - 100.0)
                + rng.gauss(0, 1))
        for name, values in (
                ("random_walk", random_walk),
                ("white_noise", white_noise),
                ("stationary_ar", stationary_ar)):
            validation = _short_trend_probe(values, horizon=72, season=72)
            admissions[name] += bool(
                validation and validation["admitted"])

    # The lane is advisory, but should still be much more selective than a
    # one-holdout winner: no stationary control and at most 2% of random
    # walks may acquire a trend prior under these frozen seeds.
    assert admissions["random_walk"] <= 4
    assert admissions["white_noise"] == 0
    assert admissions["stationary_ar"] == 0


def test_short_trend_probe_rejects_a_level_shift_as_persistent_drift() -> None:
    validation = _short_trend_probe(
        [100.0] * 50 + [130.0] * 25, horizon=72, season=72)
    assert validation is not None
    assert validation["admitted"] is False


def test_one_observed_intraday_week_is_visible_only_as_calendar_prior() -> None:
    day = [1.0 + 8.0 * max(0.0, 1.0 - abs(hour - 12) / 8.0)
           for hour in range(24)]
    values = []
    for weekday in range(7):
        values.extend([value * (0.6 if weekday >= 5 else 1.0)
                       for value in day])
    lane, disclosure, _ = build_model_assisted_lane(
        "traffic", values, horizon=72, season=24,
        future_timestamps=_hourly_stamps(72), assessment=_assessment(
            strongest_baseline="seasonal_naive",
            selection_scores={"last_value": 3.0, "seasonal_naive": 1.0}),
        published_support="degraded", selected_model="seasonal_naive")

    assert lane is not None
    assert lane["selected_model"] == "calendar_seasonal_naive"
    assert lane["support"] == "prior_assisted"
    assert lane["points"] == values[:72]
    assert lane["validation"]["basis"] == \
        "single_observed_calendar_cycle_prior"
    assert lane["validation"]["historical_skill_evidence"] is False
    assert lane["automation_eligible"] is False
    assert "no out-of-sample skill evidence" in disclosure.message


def test_calendar_prior_requires_intraday_grid_and_complete_week() -> None:
    for values, timestamps in [
            ([float(index) for index in range(167)], _hourly_stamps(72)),
            ([float(index) for index in range(168)], _stamps(72))]:
        lane, _, _ = build_model_assisted_lane(
            "traffic", values, horizon=72, season=24,
            future_timestamps=timestamps, assessment=_assessment(
                strongest_baseline="seasonal_naive",
                selection_scores={"last_value": 3.0,
                                  "seasonal_naive": 1.0}),
            published_support="degraded", selected_model="seasonal_naive")
        assert lane is None


def test_lane_stays_absent_when_a_candidate_was_published_as_primary() -> None:
    values = [float(v) for v in range(1, 17)]
    lane, _, _ = build_model_assisted_lane(
        "cpu", values, horizon=8, season=1,
        future_timestamps=_stamps(8), assessment=_assessment(),
        published_support="degraded", selected_model="theta")
    assert lane is None


def test_best_effort_fallback_runs_a_reduced_rigor_holdout() -> None:
    values = [float(v) for v in range(1, 13)]
    lane, disclosure, _ = build_model_assisted_lane(
        "cpu", values, horizon=12, season=1,
        future_timestamps=_stamps(12), assessment=None,
        published_support="best_effort", selected_model=None)
    assert lane is not None
    assert lane["support"] == "prior_assisted"
    assert lane["validation"]["basis"] == "reduced_rigor_holdout"
    assert lane["validation"]["baseline"] == "last_value"
    assert lane["validation"]["out_of_sample_steps"] == 3
    # A trending history: the admitted prior actually continues the trend
    # instead of repeating the last value.
    assert lane["points"][-1] > values[-1]
    # No quantile keys anywhere in the lane: intervals would manufacture
    # probability weight the lane has no calibrated residuals to back.
    assert "q10" not in str(lane)
    assert "never replaces the primary forecast" in disclosure.message


def test_a_history_with_no_out_of_sample_win_earns_no_lane() -> None:
    values = [5.0] * 12
    lane, _, _ = build_model_assisted_lane(
        "cpu", values, horizon=12, season=1,
        future_timestamps=_stamps(12), assessment=None,
        published_support="best_effort", selected_model=None)
    assert lane is None


def test_an_implausible_candidate_path_is_rejected(monkeypatch) -> None:
    import gnomon.model_assisted as module

    values = [float(v) for v in range(1, 13)]
    real_predict = module.predict

    def absurd_predict(name, history, horizon, season):
        # The holdout comparisons (shorter history) stay honest so a
        # candidate is admitted on evidence; only the published full-horizon
        # path is pathological — exactly what the plausibility gate is for.
        if name == "last_value" or len(history) < len(values):
            return real_predict(name, history, horizon, season)
        return [history[-1] + 1e6] * horizon

    monkeypatch.setattr(module, "predict", absurd_predict)
    lane, _, _ = build_model_assisted_lane(
        "cpu", values, horizon=12, season=1,
        future_timestamps=_stamps(12), assessment=None,
        published_support="best_effort", selected_model=None)
    assert lane is None


def test_short_histories_earn_no_lane() -> None:
    lane, _, _ = build_model_assisted_lane(
        "cpu", [1.0, 2.0, 3.0], horizon=3, season=1,
        future_timestamps=_stamps(3), assessment=None,
        published_support="best_effort", selected_model=None)
    assert lane is None


def _write_csv(path, values) -> None:
    lines = ["timestamp,value"]
    for stamp, value in zip(_stamps(len(values)), values):
        lines.append(f"{stamp.date().isoformat()},{value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_forecast_fallback_preserves_a_model_prior_beside_the_naive_rows(tmp_path) -> None:
    from gnomon import forecast

    csv_path = tmp_path / "trend.csv"
    _write_csv(csv_path, [float(v) for v in range(1, 13)])
    artifact, _ = forecast(
        str(csv_path), time_column="timestamp", target_column="value",
        horizon=12, frequency="D", output=str(tmp_path / "out"))
    result = artifact.results[0]
    assert result.support == "best_effort"
    # The governed lane is untouched: the primary rows remain the disclosed
    # last-value fallback.
    points = [row["point"] for row in result.forecast]
    assert all(point == points[0] for point in points)
    lane = result.model_assisted
    assert lane is not None
    assert lane["support"] == "prior_assisted"
    assert lane["selected_model"] not in {"last_value", "seasonal_naive"}
    assert lane["points"][-1] > points[-1]
    disclosures = (result.support_assessment or {}).get("disclosures") or []
    assert any(item.get("code") == "model_assisted_lane"
               for item in disclosures)
    assert any(item.kind == "model_assisted_lane"
               for item in artifact.evidence)
    payload = artifact.to_dict()
    assert payload["results"][0]["model_assisted"]["support"] == "prior_assisted"


def test_jittered_short_telemetry_exposes_trend_without_promoting_primary(
        tmp_path) -> None:
    from gnomon import forecast

    source = tmp_path / "cron-growth.csv"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    lines = ["timestamp,value"]
    for index in range(75):
        stamp = start + timedelta(minutes=20 * index, seconds=index % 8)
        lines.append(f"{stamp.isoformat()},{1060 + 325 * index / 74}")
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact, _ = forecast(
        str(source), time_column="timestamp", target_column="value",
        horizon=72, frequency="20min", repair="safe",
        output=str(tmp_path / "out"))
    result = artifact.results[0]

    assert result.support == "degraded"
    assert result.selected_model == "last_value"
    assert len({row["point"] for row in result.forecast}) == 1
    assert any("timestamp_jitter_aligned" in warning
               for warning in result.warnings)
    lane = result.model_assisted
    assert lane is not None
    assert lane["selected_model"] == "drift"
    assert lane["points"][-1] > result.forecast[-1]["point"]
    assert lane["validation"]["maximum_locally_evaluated_lead"] == 9
    assert lane["validation"]["extrapolated_tail_steps"] == 63
    assert lane["validation"]["tail_support"] == "prior_assisted"
    assert lane["validation"]["tail_automation_eligible"] is False
    assert lane["governed_primary"]["why_flat"]
    from gnomon.support import artifact_headline
    headline = artifact_headline(artifact.results)
    assert "Separate prior_assisted drift path" in headline
    assert "locally evaluated through lead 9" in headline
    assert "non-automatable" in headline


def test_a_raised_floor_that_publishes_nothing_publishes_no_lane(tmp_path) -> None:
    from gnomon import forecast

    csv_path = tmp_path / "trend.csv"
    _write_csv(csv_path, [float(v) for v in range(1, 13)])
    artifact, _ = forecast(
        str(csv_path), time_column="timestamp", target_column="value",
        horizon=12, frequency="D", output=str(tmp_path / "out"),
        minimum_support="supported")
    result = artifact.results[0]
    assert not result.forecast
    assert result.model_assisted is None
    assert "model_assisted" not in artifact.to_dict()["results"][0]

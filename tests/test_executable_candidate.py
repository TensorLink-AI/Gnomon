"""The published candidate is the evaluated candidate (unified plan 1A).

The end-to-end shape that did not exist: a full forecast run under a
non-default ensemble strategy and a restricted candidate pool, asserting
the published points equal the evaluated specification's final fit. The
old publish path hardcoded ``strategy="weighted_mean"`` over the
unrestricted built-in pool with no config, so exactly this configuration
scored one ensemble and published another wearing its credentials.
"""

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from gnomon.config import GnomonConfig
from gnomon.ids import FixedClock
from gnomon.models import MODELS, predict
from gnomon.runtime import forecast

CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))
NOISE = [0.5, -0.3, 0.2, -0.4, 0.1, 0.3, -0.2, -0.1, 0.4, -0.5]
HORIZON = 7
RESTRICTED = ["drift", "theta"]


def _values(periods: int = 160) -> list[float]:
    return [
        100 + index * 0.4 + NOISE[index % 10] * 6 for index in range(periods)
    ]


def _csv(tmp_path: Path) -> Path:
    start = date(2026, 1, 1)
    rows = [
        f"{(start + timedelta(days=index)).isoformat()},{value:.4f}"
        for index, value in enumerate(_values())
    ]
    path = tmp_path / "series.csv"
    path.write_text("timestamp,value\n" + "\n".join(rows) + "\n",
                    encoding="utf-8")
    return path


def _median_config() -> GnomonConfig:
    config = GnomonConfig()
    config.models.statistical_candidates = list(RESTRICTED)
    config.ensemble.enabled = True
    config.ensemble.strategy = "median"
    return config


def test_published_ensemble_is_the_evaluated_ensemble(tmp_path):
    """Non-default strategy + restricted pool: the published points must be
    the evaluated combiner's final fit — median over the restricted pool —
    not the legacy hardcoded weighted_mean over all built-ins."""
    from statistics import median

    artifact, out_path = forecast(
        str(_csv(tmp_path)), time_column="timestamp", target_column="value",
        horizon=HORIZON, frequency="D", output=str(tmp_path / "out"),
        config=_median_config(), selection_strategy="ensemble", clock=CLOCK,
    )
    result = artifact.results[0]
    assert result.selected_model == "ensemble"
    assert result.forecast, "no points were published"

    values = _values()
    # The same season the evaluation used (the NOISE cycle makes this 10,
    # not the calendar 7 — guessing here is how a test lies).
    from gnomon.temporal import detect_season
    season, _, _ = detect_season(values, "D")
    # The evaluated pool: baselines (mandatory) plus the restricted
    # candidates — every member with a selection score.
    members = [name for name in result.selection_scores
               if name in MODELS
               and result.selection_scores[name] is not None]
    assert set(RESTRICTED) <= set(members)
    assert "ets" not in members, "the pool restriction did not hold"

    member_finals = {name: predict(name, values, HORIZON, season)
                     for name in members}
    expected = [median([member_finals[name][step] for name in members])
                for step in range(HORIZON)]
    published = [row["point"] for row in result.forecast]
    assert published == expected, (
        "published points are not the evaluated median ensemble's final fit"
    )

    # And they must differ from what the legacy path would have published
    # (weighted_mean over ALL built-ins, config ignored) — otherwise this
    # test could pass while proving nothing.
    from gnomon.ensemble import compute_ensemble_forecast
    legacy_members = {}
    for name in MODELS:
        try:
            legacy_members[name] = predict(name, values, HORIZON, season)
        except ValueError:
            pass
    legacy = compute_ensemble_forecast(
        legacy_members, result.selection_scores,
        strategy="weighted_mean", last_observed=values[-1])
    assert published != legacy, (
        "the decisive configuration no longer distinguishes the paths"
    )


def test_composite_identity_is_recorded_in_evidence(tmp_path):
    _, out_path = forecast(
        str(_csv(tmp_path)), time_column="timestamp", target_column="value",
        horizon=HORIZON, frequency="D", output=str(tmp_path / "out"),
        config=_median_config(), selection_strategy="ensemble", clock=CLOCK,
    )
    records = [json.loads(line) for line in
               (Path(out_path) / "evidence.jsonl").read_text(
                   encoding="utf-8").splitlines()]
    candidate = next(r for r in records if r["kind"] == "final_candidate")
    payload = candidate["payload"]
    assert payload["strategy"] == "median"
    assert set(RESTRICTED) <= set(payload["members"])
    assert "ets" not in payload["members"]
    assert "min_models" in payload["config"]


def test_max_weight_ratio_reaches_publication(tmp_path):
    """`predict_stage` took no config, so ensemble.max_weight_ratio
    silently reverted to its default on the published path."""
    # A three-member pool whose inverse-error weights sit near 1/3, with
    # a cap well below that so it provably binds — near-equal weights
    # under a loose cap would make the comparison vacuous.
    tight = GnomonConfig()
    tight.models.statistical_candidates = ["drift"]
    tight.ensemble.enabled = True
    tight.ensemble.max_weight_ratio = 0.20
    loose = GnomonConfig()
    loose.models.statistical_candidates = ["drift"]
    loose.ensemble.enabled = True
    loose.ensemble.max_weight_ratio = 0.99

    published = {}
    for label, config in (("tight", tight), ("loose", loose)):
        artifact, _ = forecast(
            str(_csv(tmp_path)), time_column="timestamp",
            target_column="value", horizon=HORIZON, frequency="D",
            output=str(tmp_path / f"out_{label}"), config=config,
            selection_strategy="ensemble", clock=CLOCK,
        )
        published[label] = [row["point"]
                            for row in artifact.results[0].forecast]
    assert published["tight"] != published["loose"], (
        "max_weight_ratio still does not reach the published path"
    )


def test_default_run_publishes_identically_with_no_candidate_evidence(tmp_path):
    """Single built-in selections publish exactly as before — identity is
    already `selected_model` — and existing artifacts do not churn."""
    artifact, out_path = forecast(
        str(_csv(tmp_path)), time_column="timestamp", target_column="value",
        horizon=HORIZON, frequency="D", output=str(tmp_path / "out"),
        clock=CLOCK,
    )
    result = artifact.results[0]
    assert result.selected_model != "ensemble"
    records = [json.loads(line) for line in
               (Path(out_path) / "evidence.jsonl").read_text(
                   encoding="utf-8").splitlines()]
    assert not [r for r in records if r["kind"] == "final_candidate"]

"""One invariant: points and intervals come from the same model.

Five paths used to publish a point path from one model beside an interval
calibrated on another's residuals. They were separate bugs with one shape,
and `assert_residual_provenance` is the single check that collapses them.

The rest of this file covers the disclosures that make honest-but-surprising
interval behaviour legible: `point` is not the median, a flat interval is
the fold count and not a defect, one-fold coverage is not a guarantee, and
pooled calibration is optimistic by a known direction.
"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from gnomon.contracts import GnomonError
from gnomon.evaluation import MIN_RESIDUALS_PER_LEAD, pooled_fallback_leads
from gnomon.ids import FixedClock
from gnomon.pipeline import SeriesState, assert_residual_provenance
from gnomon.runtime import forecast

CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))
NOISE = [0.5, -0.3, 0.2, -0.4, 0.1, 0.3, -0.2, -0.1, 0.4, -0.5]


def _csv(tmp_path: Path, periods: int = 120) -> Path:
    start = date(2026, 1, 1)
    rows = [
        f"{(start + timedelta(days=index)).isoformat()},"
        f"{100 + index * 0.4 + NOISE[index % 10] * 6:.4f}"
        for index in range(periods)
    ]
    path = tmp_path / "series.csv"
    path.write_text("timestamp,value\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _codes(result) -> set[str]:
    return {
        item["code"] for item in result.support_assessment.get("disclosures", [])
    }


def test_foldless_point_interval_is_disclosed_and_capped(tmp_path) -> None:
    source = tmp_path / "six.csv"
    rows = (Path(__file__).resolve().parent.parent / "examples" /
            "messy_requests.csv").read_text(encoding="utf-8").splitlines()[:7]
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    artifact, directory = forecast(
        str(source), time_column="timestamp", target_column="requests",
        horizon=2, output=str(tmp_path / "out"), clock=CLOCK,
    )
    result = artifact.results[0]
    assert all(row["q10"] == row["q50"] == row["q90"]
               for row in result.forecast)
    assert "interval_collapsed_to_point" in _codes(result)
    assert result.support_assessment["status"] == "inconclusive"
    assert {row["tier"] for row in result.forecast} == {"best_effort"}
    summary = (directory / "summary.md").read_text(encoding="utf-8")
    report = (directory / "report.html").read_text(encoding="utf-8")
    assert "- Support: best_effort" in summary
    assert "<dt>Support</dt><dd>best_effort</dd>" in report


def test_zero_residual_interval_with_evaluation_is_not_foldless(tmp_path) -> None:
    artifact, _ = forecast(
        str(Path(__file__).resolve().parent.parent /
            "examples" / "daily_requests.csv"),
        time_column="timestamp", target_column="requests", horizon=3,
        output=str(tmp_path / "out"), clock=CLOCK,
    )
    result = artifact.results[0]
    assert "interval_collapsed_to_point" not in _codes(result)


# -- The invariant --------------------------------------------------------

def test_mismatched_residual_provenance_is_refused():
    state = SeriesState(
        name="alpha", values=[1.0], timestamps=[], future_timestamps=[], season=1,
    )
    state.points = [1.0, 2.0]
    state.selected_model = "covariate_linear"
    state.residual_source = "linear_trend"
    with pytest.raises(GnomonError) as raised:
        assert_residual_provenance(state)
    assert raised.value.code == "RESIDUAL_PROVENANCE_MISMATCH"


def test_matching_residual_provenance_passes():
    state = SeriesState(
        name="alpha", values=[1.0], timestamps=[], future_timestamps=[], season=1,
    )
    state.points = [1.0, 2.0]
    state.selected_model = "linear_trend"
    state.residual_source = "linear_trend"
    assert_residual_provenance(state) is None


def test_baseline_forecast_has_matching_provenance(tmp_path):
    """The invariant holds on the ordinary path, not only in unit tests."""
    import gnomon.runtime as runtime

    seen: dict[str, object] = {}
    original = runtime.interval_stage

    def spy(state, **kwargs):
        seen["selected"] = state.selected_model
        seen["source"] = state.residual_source
        return original(state, **kwargs)

    runtime.interval_stage = spy
    try:
        forecast(
            str(_csv(tmp_path)), time_column="timestamp", target_column="value",
            horizon=7, output=str(tmp_path / "out"), clock=CLOCK,
        )
    finally:
        runtime.interval_stage = original
    assert seen["source"] == seen["selected"]


def test_admitted_covariate_carries_its_own_residuals(tmp_path):
    """C2, measured. The published model's intervals must be its own.

    The covariate model used to publish points beside `residuals_by_lead`
    from the *base* model, where every lead had enough of the wrong
    residuals — so the covariate model's real residuals were never
    consulted at all.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_covariates import _write_files

    import gnomon.runtime as runtime
    from gnomon.covariates import load_covariates

    observations, covariates_path = _write_files(tmp_path)
    dataset = load_covariates(str(covariates_path), "campaign:binary:future_known")

    seen: dict[str, object] = {}
    original = runtime.interval_stage

    def spy(state, **kwargs):
        seen["selected"] = state.selected_model
        seen["source"] = state.residual_source
        seen["pooled"] = list(state.residuals)
        seen["by_lead"] = {
            step: list(items) for step, items in state.residuals_by_lead.items()
        }
        return original(state, **kwargs)

    runtime.interval_stage = spy
    try:
        forecast(
            str(observations), time_column="timestamp", target_column="value",
            horizon=7, frequency="D", output=str(tmp_path / "out"),
            covariates=dataset, clock=CLOCK,
        )
    finally:
        runtime.interval_stage = original

    assert seen["selected"] == "covariate_linear"
    assert seen["source"] == "covariate_linear"
    # The covariate model calibrates on one fold, so it has exactly one
    # residual per lead — below the threshold, so every lead correctly
    # borrows *its own* pooled set rather than the base model's per-lead one.
    flattened = [value for values in seen["by_lead"].values() for value in values]
    assert sorted(flattened) == sorted(seen["pooled"])
    assert all(len(values) == 1 for values in seen["by_lead"].values())


# -- C4: every emitted quantile is projected ------------------------------

def test_constraints_clamp_every_quantile_level():
    from gnomon.constraints import Claim, apply_claims

    row = {
        "timestamp": "2026-06-04T00:00:00", "point": 350.0,
        "q05": 310.0, "q10": 320.0, "q20": 330.0, "q30": 340.0, "q50": 355.0,
        "q70": 370.0, "q80": 385.3, "q90": 392.0, "q95": 397.8,
    }
    claim = Claim("e1", "max", 370.0, "2026-06-01T00:00:00", "2026-06-30T00:00:00")
    projected, applications = apply_claims([row], [claim])
    result = projected[0]

    quantiles = [result[key] for key in
                 ("q05", "q10", "q20", "q30", "q50", "q70", "q80", "q90", "q95")]
    assert max(quantiles + [result["point"]]) <= 370.0, "a declared max was exceeded"
    assert quantiles == sorted(quantiles), "projection produced a crossed distribution"
    assert applications, "the clamp was not recorded"


def test_constraints_clamp_a_minimum_across_every_level():
    from gnomon.constraints import Claim, apply_claims

    row = {
        "timestamp": "2026-06-04T00:00:00", "point": 100.0,
        "q05": 60.0, "q10": 70.0, "q50": 100.0, "q95": 140.0,
    }
    claim = Claim("e1", "min", 80.0, "2026-06-01T00:00:00", "2026-06-30T00:00:00")
    result = apply_claims([row], [claim])[0][0]
    assert result["q05"] == 80.0 and result["q10"] == 80.0
    assert result["q50"] == 100.0 and result["q95"] == 140.0


def test_crossed_quantiles_are_refused():
    from gnomon.constraints import _assert_monotone

    with pytest.raises(GnomonError) as raised:
        _assert_monotone({"timestamp": "t", "q10": 5.0, "q50": 9.0, "q90": 4.0})
    assert raised.value.code == "QUANTILE_CROSSING"


# -- H2, H3, H4, M2, L6: the disclosures ----------------------------------

def test_point_bias_correction_ships_on_every_row(tmp_path):
    artifact, path = forecast(
        str(_csv(tmp_path)), time_column="timestamp", target_column="value",
        horizon=7, output=str(tmp_path / "out"), clock=CLOCK,
    )
    result = artifact.results[0]
    for row in result.forecast:
        assert "point_bias_correction" in row
        assert row["point_bias_correction"] == pytest.approx(
            row["q50"] - row["point"], abs=1e-9
        )
    header = (path / "forecast.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "point_bias_correction" in header


def test_constant_interval_width_is_disclosed(tmp_path):
    """H3 / S2. A flat interval across the horizon must say why it is flat."""
    artifact, _ = forecast(
        str(_csv(tmp_path, periods=60)), time_column="timestamp",
        target_column="value", horizon=14, output=str(tmp_path / "out"),
        clock=CLOCK,
    )
    result = artifact.results[0]
    widths = {round(row["q90"] - row["q10"], 9) for row in result.forecast}
    if len(widths) == 1:
        assert "constant_interval_width" in _codes(result)


def test_pooled_calibration_is_disclosed(tmp_path):
    """H4. The default pools folds selection saw; say so and say which way."""
    artifact, _ = forecast(
        str(_csv(tmp_path)), time_column="timestamp", target_column="value",
        horizon=7, output=str(tmp_path / "out"), clock=CLOCK,
    )
    codes = _codes(artifact.results[0])
    assert "conformal_residuals_pooled_across_selection" in codes


def test_coverage_headline_carries_its_sample_size(tmp_path):
    """M2 / S3. "100% coverage" over 7 points from one fold is not a claim."""
    artifact, path = forecast(
        str(_csv(tmp_path)), time_column="timestamp", target_column="value",
        horizon=7, output=str(tmp_path / "out"), clock=CLOCK,
    )
    result = artifact.results[0]
    if result.interval_coverage is not None:
        assert "coverage_sample_size" in _codes(result)
        summary = (path / "summary.md").read_text(encoding="utf-8")
        assert "points, one fold)" in summary


def test_held_out_calibration_uses_one_fold(tmp_path):
    """`evaluation.pool_residuals: false` is read, and changes the residuals.

    The key was documented and never consulted; honouring it is what gives
    a caller the honest split-conformal alternative.
    """
    from gnomon.config import GnomonConfig
    from gnomon.evaluation import evaluate

    values = [100 + index * 0.4 + NOISE[index % 10] * 6 for index in range(120)]

    pooled = evaluate(values, 7, 7, 0.02, frequency="D", tsfm_names=[])
    config = GnomonConfig()
    config.evaluation.pool_residuals = False
    held_out = evaluate(
        values, 7, 7, 0.02, frequency="D", tsfm_names=[], config=config,
    )

    assert pooled.residuals_pooled_across_selection is True
    assert held_out.residuals_pooled_across_selection is False
    assert held_out.residual_fold_count == 1
    assert pooled.residual_fold_count > 1
    assert len(held_out.residuals) < len(pooled.residuals)


def test_disclosures_never_change_support_status():
    """A disclosure describes; only a reason may downgrade."""
    from gnomon.contracts import SupportReason
    from gnomon.support import assess_forecast_support

    plain = assess_forecast_support("supported", [], None)
    with_disclosures = assess_forecast_support(
        "supported", [], None,
        disclosures=[SupportReason("point_is_not_the_median", "…")],
    )
    assert plain.status == with_disclosures.status == "supported"
    assert with_disclosures.disclosures[0].code == "point_is_not_the_median"


def test_pooled_fallback_leads_reports_every_borrowing_lead():
    by_lead = {1: [0.0] * MIN_RESIDUALS_PER_LEAD, 2: [0.0], 3: []}
    assert pooled_fallback_leads(by_lead, 3) == [2, 3]

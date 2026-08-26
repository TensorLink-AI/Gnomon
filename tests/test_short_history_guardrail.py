"""The short-history guardrail and point-centred degraded intervals.

Measured motivation (results/news-regime-explore/RESULTS.md and
results/short-history-guardrail/HYPOTHESIS.md): at 30 daily points with
horizon 7 the fold skeleton yields one selection fold, and single-fold
selection at the default margin picked a non-baseline on 39 of 50
near-martingale series, running 2.9x the MSE of `last_value`; the
median-residual recentring shifted the published path by ~1 sigma in a
coin-flip direction. The guardrail publishes the assumption-minimal level
baseline when the contest cannot rank; degraded runs centre quantiles on the point
path. Both fire only on fold-starved runs: a fully evidenced series is
byte-identical (test_golden_artifacts pins that).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gnomon.evaluation import (
    conformal_quantile,
    conformal_quantile_spreads,
    conformal_spreads,
    coverage_levels,
    evaluate,
    select_model_lightweight,
)
from gnomon.models import BASELINES


def trending(n: int, slope: float = 3.0, base: float = 100.0) -> list[float]:
    # A wobble that repeats every 4 points, so no candidate fits it
    # perfectly and every model completes every fold.
    return [base + slope * i + (1.5 if i % 4 == 0 else -0.5) for i in range(n)]


def near_martingale(n: int, base: float = 100.0) -> list[float]:
    # Trendless deterministic wobble: incremental single-fold wins are
    # available to candidates, overwhelming ones are not.
    wobble = [0.4, -0.2, 0.7, -0.5, 0.1, -0.6, 0.3]
    return [base + wobble[i % len(wobble)] + 0.05 * (i % 3) for i in range(n)]


class TestSelectionGuardrail:
    def test_single_fold_noise_win_publishes_baseline_with_evidence(self):
        # 30 points, horizon 7, season 7: origins [14, 21] -> one
        # selection fold. On a trendless series no candidate clears the
        # raised single-fold margin, whatever the fold says.
        result = evaluate(near_martingale(30), 7, 7, 0.02, frequency="D")
        assert result.supported and result.degraded
        assert result.selection_fold_count == 1
        assert result.selection_guardrail_applied
        assert result.selected_model in BASELINES
        # Candidate scores survive as evidence.
        non_baseline_scored = [
            name for name, score in result.selection_scores.items()
            if name not in BASELINES and score is not None
        ]
        assert non_baseline_scored
        assert any("Selection under-powered" in w for w in result.warnings)

    def test_single_fold_overwhelming_margin_selects_the_candidate(self):
        # A deterministic trend cuts the baseline's single-fold error by
        # far more than SINGLE_FOLD_SELECTION_MARGIN: the escape hatch
        # for structure one fold *can* prove.
        result = evaluate(trending(30), 7, 7, 0.02, frequency="D")
        assert result.degraded
        assert result.selection_fold_count == 1
        assert not result.selection_guardrail_applied
        assert result.selected_model not in BASELINES
        assert any("Single-fold selection" in w for w in result.warnings)

    def test_two_disjoint_folds_rank_normally(self):
        # 45 points, horizon 7: origins [14, 21, 28, 35] -> two disjoint
        # selection folds; the contest ranks at the caller's margin.
        result = evaluate(trending(45), 7, 7, 0.02, frequency="D")
        assert result.selection_fold_count == 2
        assert not result.selection_guardrail_applied
        assert result.selected_model not in BASELINES
        assert not any("Selection under-powered" in w for w in result.warnings)

    def test_dense_stride_does_not_lift_the_guardrail(self):
        # Overlapping selection origins widen the comparison sample, not
        # the evidence; the guardrail reads the disjoint skeleton.
        result = evaluate(near_martingale(30), 7, 7, 0.02, frequency="D",
                          selection_stride=1)
        assert result.selection_guardrail_applied
        assert result.selection_fold_count == 1
        assert result.selected_model in BASELINES

    def test_lightweight_holdout_keeps_the_hard_lock(self):
        # The lightweight path's holdout can be a single observation —
        # too thin for even an overwhelming-margin escape hatch, so a
        # trend still publishes the baseline there.
        result = select_model_lightweight(trending(10), 3, 7)
        assert result.supported and result.degraded
        assert result.selection_guardrail_applied
        assert result.selected_model in BASELINES
        assert any("Selection under-powered" in w for w in result.warnings)

    def test_repeatable_seasonal_baseline_can_earn_degraded_admission(self):
        # Full 56-step evaluation cannot fit, but eight disjoint historical
        # weeks independently establish that a predeclared weekly baseline
        # beats copying the most recent day. This is evidence about a baseline,
        # not permission to rank the wider candidate zoo.
        week = [20.0, 18.0, 17.0, 19.0, 26.0, 40.0, 35.0]
        result = select_model_lightweight(week * 16, 56, 7)

        assert result.selected_model == "seasonal_naive"
        assert result.strongest_baseline == "seasonal_naive"
        assert result.selection_fold_count == 8
        assert result.selection_guardrail_applied
        disclosure = next(
            warning for warning in result.warnings
            if warning.startswith("Degraded baseline admission:"))
        assert '"admitted": true' in disclosure
        assert '"chronological_block_wins": 3' in disclosure

    def test_degraded_seasonal_admission_rejects_level_series(self):
        # When the structured baseline has no demonstrated improvement, the
        # assumption-minimal level forecast remains the publication.
        result = select_model_lightweight([100.0] * 112, 56, 7)

        assert result.selected_model == "last_value"
        assert result.selection_fold_count == 1
        disclosure = next(
            warning for warning in result.warnings
            if warning.startswith("Degraded baseline admission:"))
        assert '"admitted": false' in disclosure

    @pytest.mark.parametrize("season", [2, 3, 7])
    def test_lightweight_does_not_rank_structured_baselines_on_one_holdout(self, season):
        # The tail deliberately repeats an earlier seasonal value, tempting a
        # one-holdout selector.  That is still one observation of generality;
        # the robust level baseline is the production-safe default.
        values = [100.0 + ((index * 7) % 5) for index in range(14)]
        result = select_model_lightweight(values, 4, season)
        assert result.selected_model == "last_value"
        assert result.strongest_baseline == "last_value"


class TestPointCentredIntervals:
    def borrowed_leads(self) -> dict[int, list[float]]:
        # Three residuals per lead, deliberately off-centre (median 1.0):
        # every lead borrows the pooled set, and recentring would shift
        # the whole band by that median.
        return {step: [-2.0, 1.0, 2.0] for step in range(1, 8)}

    def test_recentre_false_moves_the_median_to_zero_and_keeps_widths(self):
        by_lead = self.borrowed_leads()
        pooled = [r for rs in by_lead.values() for r in rs]
        recentred = conformal_spreads(by_lead, 7, pooled)
        centred = conformal_spreads(by_lead, 7, pooled, recentre=False)
        for step in range(1, 8):
            low_r, median_r, high_r = recentred[step]
            low_c, median_c, high_c = centred[step]
            assert median_r != 0.0
            assert median_c == 0.0
            # Widths are untouched; only the location moves.
            assert low_c == low_r and high_c == high_r
        # Default stays recentred: nothing changes for existing callers.
        assert conformal_spreads(by_lead, 7, pooled) == recentred

    def test_no_lead_growth_multiplier_on_borrowed_leads(self):
        # Whole-horizon pooled residuals already contain multi-step
        # dispersion; the borrowed band must be flat, not scaled with h.
        by_lead = self.borrowed_leads()
        pooled = [r for rs in by_lead.values() for r in rs]
        spreads = conformal_spreads(by_lead, 7, pooled, recentre=False)
        assert spreads[7] == spreads[1]

    def test_quantile_levels_match_the_central_spread(self):
        by_lead = self.borrowed_leads()
        pooled = [r for rs in by_lead.values() for r in rs]
        for recentre in (True, False):
            spreads = conformal_spreads(by_lead, 7, pooled,
                                        recentre=recentre)
            levels = conformal_quantile_spreads(by_lead, 7, pooled,
                                                recentre=recentre)
            lower, middle, upper = coverage_levels()
            median = conformal_quantile(pooled, middle) if recentre else 0.0
            for step in range(1, 8):
                low_offset, spread_median, high_offset = spreads[step]
                assert abs(spread_median - median) < 1e-12
                assert abs(levels[step][0.1] - (median - low_offset)) < 1e-12
                assert abs(levels[step][0.9] - (median + high_offset)) < 1e-12


class TestArtifactDisclosure:
    def run_short(self, tmp_path: Path):
        from gnomon.runtime import forecast

        from datetime import datetime, timedelta, timezone

        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        csv_path = tmp_path / "short.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "value"])
            for i, value in enumerate(near_martingale(30)):
                writer.writerow([(start + timedelta(days=i)).isoformat(), value])
        artifact, _ = forecast(str(csv_path), time_column="timestamp",
                               target_column="value", horizon=7,
                               frequency="D", output=str(tmp_path / "out"))
        return artifact.results[0]

    def test_underpowered_reason_fold_count_and_centring(self, tmp_path):
        result = self.run_short(tmp_path)
        assessment = result.support_assessment
        reasons = {r["code"] for r in assessment["reasons"]}
        assert "selection_underpowered" in reasons
        assert assessment["sensitivity"]["selection_fold_count"] == 1
        codes = {d["code"] for d in assessment["disclosures"]}
        assert "point_recentring_suppressed" in codes
        assert "constant_interval_width" in codes
        assert "point_is_not_the_median" not in codes
        # Quantiles centre on the point path: the median-residual
        # recentring is suppressed on fold-starved runs. The band stays
        # flat — pooled residuals already contain multi-step dispersion.
        widths = [row["q90"] - row["q10"] for row in result.forecast]
        assert max(widths) - min(widths) < 1e-9
        for row in result.forecast:
            assert row["point_bias_correction"] == 0.0
            assert row["q50"] == row["point"]


class TestCapabilities:
    def test_capabilities_state_the_guardrail(self):
        from gnomon.runtime import capabilities

        caps = capabilities()
        assert "selection_guardrail" in caps["short_history"]
        assert "point_recentring" in caps["short_history"]
        assert "degraded_baseline_admission" in caps["short_history"]
        assert caps["features"]["short_history_selection_guardrail"] is True
        assert caps["features"]["short_history_point_centred_intervals"] is True

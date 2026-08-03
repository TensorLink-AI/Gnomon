"""The short-history guardrail and lead-time interval widening.

Measured motivation (results/news-regime-explore/RESULTS.md): at 30
daily points with horizon 7 the fold skeleton yields one selection fold,
and single-fold selection at the default margin picked a non-baseline on
39 of 50 near-martingale series, running 2.9x the MSE of `last_value`;
the flat borrowed interval covered 82% at step 1 decaying to 44% at
step 7 against a nominal 80%. The guardrail publishes the strongest
baseline when the contest cannot rank; the widening grows borrowed
half-widths by sqrt(lead). Both fire only on fold-starved runs: a fully
evidenced series is byte-identical (test_golden_artifacts pins that).
"""

from __future__ import annotations

import csv
from pathlib import Path

from gnomon.evaluation import (
    MIN_RESIDUALS_PER_LEAD,
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


class TestSelectionGuardrail:
    def test_single_fold_publishes_baseline_and_keeps_evidence(self):
        # 30 points, horizon 7, season 7: origins [14, 21] -> one
        # selection fold. A trending series is exactly where a
        # non-baseline would win that fold.
        result = evaluate(trending(30), 7, 7, 0.02, frequency="D")
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

    def test_two_disjoint_folds_rank_normally(self):
        # 45 points, horizon 7: origins [14, 21, 28, 35] -> two disjoint
        # selection folds; the contest ranks, and this trend is steep
        # enough that a trend-following candidate beats last_value.
        result = evaluate(trending(45), 7, 7, 0.02, frequency="D")
        assert result.selection_fold_count == 2
        assert not result.selection_guardrail_applied
        assert result.selected_model not in BASELINES
        assert not any("Selection under-powered" in w for w in result.warnings)

    def test_dense_stride_does_not_lift_the_guardrail(self):
        # Overlapping selection origins widen the comparison sample, not
        # the evidence; the guardrail reads the disjoint skeleton.
        result = evaluate(trending(30), 7, 7, 0.02, frequency="D",
                          selection_stride=1)
        assert result.selection_guardrail_applied
        assert result.selection_fold_count == 1
        assert result.selected_model in BASELINES

    def test_lightweight_holdout_gets_the_same_guardrail(self):
        result = select_model_lightweight(trending(10), 3, 7)
        assert result.supported and result.degraded
        assert result.selection_guardrail_applied
        assert result.selected_model in BASELINES
        assert any("Selection under-powered" in w for w in result.warnings)


class TestLeadTimeWidening:
    def borrowed_leads(self) -> dict[int, list[float]]:
        # Three residuals per lead: every lead borrows the pooled set,
        # whose conformal q10/q50/q90 are a clean symmetric (-2, 0, 2).
        return {step: [-2.0, 0.0, 2.0] for step in range(1, 8)}

    def test_borrowed_leads_scale_sqrt_step(self):
        by_lead = self.borrowed_leads()
        pooled = [r for rs in by_lead.values() for r in rs]
        flat = conformal_spreads(by_lead, 7, pooled)
        widened = conformal_spreads(by_lead, 7, pooled, widen_borrowed=True)
        low0, _, high0 = flat[1]
        assert low0 > 0 and high0 > 0
        for step in range(1, 8):
            low, _, high = widened[step]
            assert abs(low - low0 * step ** 0.5) < 1e-12
            assert abs(high - high0 * step ** 0.5) < 1e-12
        # Default stays the flat band: nothing changes for existing callers.
        assert conformal_spreads(by_lead, 7, pooled) == flat

    def test_measured_leads_are_not_scaled(self):
        by_lead = self.borrowed_leads()
        # Lead 1 measures its own spread, narrower than the borrowed band,
        # so the isotonic fit leaves every width where widening put it.
        count = MIN_RESIDUALS_PER_LEAD + 1
        by_lead[1] = [-1.0 + 2.0 * i / (count - 1) for i in range(count)]
        pooled = [r for rs in by_lead.values() for r in rs]
        flat = conformal_spreads(by_lead, 7, pooled)
        widened = conformal_spreads(by_lead, 7, pooled, widen_borrowed=True)
        # Step 1 measures its own spread; widening must not touch it.
        assert widened[1] == flat[1]
        # Borrowed steps still scale.
        assert widened[7][2] > flat[7][2] * 2

    def test_quantile_levels_match_the_central_spread(self):
        by_lead = self.borrowed_leads()
        pooled = [r for rs in by_lead.values() for r in rs]
        spreads = conformal_spreads(by_lead, 7, pooled, widen_borrowed=True)
        levels = conformal_quantile_spreads(by_lead, 7, pooled,
                                            widen_borrowed=True)
        lower, middle, upper = coverage_levels()
        median = conformal_quantile(pooled, middle)
        for step in range(1, 8):
            low_offset, spread_median, high_offset = spreads[step]
            assert abs(spread_median - median) < 1e-12
            assert abs(levels[step][0.1] + low_offset) < 1e-12
            assert abs(levels[step][0.9] - high_offset) < 1e-12


class TestArtifactDisclosure:
    def run_short(self, tmp_path: Path):
        from gnomon.runtime import forecast

        from datetime import datetime, timedelta, timezone

        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        csv_path = tmp_path / "short.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "value"])
            for i, value in enumerate(trending(30)):
                writer.writerow([(start + timedelta(days=i)).isoformat(), value])
        artifact, _ = forecast(str(csv_path), time_column="timestamp",
                               target_column="value", horizon=7,
                               frequency="D", output=str(tmp_path / "out"))
        return artifact.results[0]

    def test_underpowered_reason_fold_count_and_widening(self, tmp_path):
        result = self.run_short(tmp_path)
        assessment = result.support_assessment
        reasons = {r["code"] for r in assessment["reasons"]}
        assert "selection_underpowered" in reasons
        assert assessment["sensitivity"]["selection_fold_count"] == 1
        codes = {d["code"] for d in assessment["disclosures"]}
        assert "lead_time_widened_intervals" in codes
        assert "constant_interval_width" not in codes
        widths = [row["q90"] - row["q10"] for row in result.forecast]
        assert widths[-1] > widths[0] * 2  # sqrt(7) ~ 2.65 on a flat base


class TestCapabilities:
    def test_capabilities_state_the_guardrail(self):
        from gnomon.runtime import capabilities

        caps = capabilities()
        assert "selection_guardrail" in caps["short_history"]
        assert "interval_widening" in caps["short_history"]
        assert caps["features"]["short_history_selection_guardrail"] is True
        assert caps["features"]["lead_time_widened_intervals"] is True

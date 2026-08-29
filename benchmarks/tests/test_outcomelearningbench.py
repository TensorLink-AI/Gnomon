from pathlib import Path

from benchmarks.outcomelearningbench.run_outcomelearningbench import run_suite


def test_prequential_outcome_learning_is_scoped_and_safe(tmp_path: Path):
    result = run_suite(tmp_path / "registry.db")

    assert result["passed"] is True
    stable = result["families"]["stable_beneficial"]
    harmful = result["families"]["stable_harmful"]
    reversal = result["families"]["regime_reversal"]
    assert stable["outcome_informed_selections"] > 0
    assert stable["mean_selected_wape"] < stable["mean_primary_wape"]
    assert reversal["mean_selected_wape"] < reversal[
        "mean_counterfactual_full_prior_wape"]
    assert harmful["outcome_informed_selections"] == 0
    assert reversal["outcome_informed_selections"] > 0
    assert reversal["bad_recommendations_before_demotion"] == 2
    assert reversal["first_demoted_after_regime_change"] == 11
    assert result["gates"]["reversal_demoted_within_two_resolved_losses"] is True
    assert result["gates"]["unrelated_series_not_used"] is True
    assert result["gates"]["different_proposer_history_not_used"] is True
    assert result["gates"][
        "shrinkage_reduces_reversal_regret_vs_full_prior"] is True
    assert result["gates"][
        "moderate_signal_promoted_in_at_least_80pct_streams"] is True
    assert result["gates"]["no_skill_false_promotion_at_most_10pct"] is True

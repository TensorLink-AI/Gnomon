from __future__ import annotations

from benchmarks.volatilitybench.run_volatilitybench import run


def test_volatilitybench_is_complete_and_scores_declared_quantity() -> None:
    report = run()
    assert report["cases"] == 180
    assert report["independent_seeds"] == 180
    assert len(report["rows"]) == 180
    assert report["interval_coverage"] >= .8
    assert report["mean_absolute_log_scale_error"] < \
        report["constant_baseline_mean_log_scale_error"]
    assert (report["claimed_direction_accuracy"] is None
            or report["claimed_direction_accuracy"] >= .8)
    # Raw direction accuracy is deliberately not a graduation gate: the
    # legacy suite is mostly stable, so an always-stable classifier scores
    # highly. The balanced, mechanism-held-out suite exposes discrimination.
    balanced = report["balanced_suite"]
    assert len(set(balanced["all"]["class_counts"].values())) == 1
    assert balanced["all"]["balanced_direction_accuracy"] > .5
    assert balanced["held_out_families"]["balanced_direction_accuracy"] > .48
    assert balanced["all"]["brier_skill_vs_uniform"] > 0
    assert balanced["all"]["claim_rate"] <= .05
    assert all(row["absolute_log_scale_error"] >= 0 for row in report["rows"])
    assert {row["regime"] for row in report["rows"]} == {
        "constant", "increase", "decrease", "abrupt", "seasonal",
        "heavy_tail"}

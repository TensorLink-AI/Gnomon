"""The harness must not state something its own evidence contradicts.

Four ways it did: a caller-supplied parameter could turn the
mandated-baseline gate off, a support sentence asserted a comparison
unconditionally, the verifier's calibration gate was satisfied by the mere
existence of a record, and its leakage check compared timestamps as
strings. Plus two places where a measurement scored itself.
"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from gnomon.contracts import GnomonError
from gnomon.ids import FixedClock
from gnomon.runtime import forecast

CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))
NOISE = [0.5, -0.3, 0.2, -0.4, 0.1, 0.3, -0.2, -0.1, 0.4, -0.5]


def _values(periods: int = 120) -> list[float]:
    return [
        100 + index * 0.4 + NOISE[index % 10] * 6 for index in range(periods)
    ]


def _csv(tmp_path: Path, periods: int = 120) -> Path:
    start = date(2026, 1, 1)
    rows = [
        f"{(start + timedelta(days=index)).isoformat()},{value:.4f}"
        for index, value in enumerate(_values(periods))
    ]
    path = tmp_path / "series.csv"
    path.write_text("timestamp,value\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


# -- C6: the mandated-baseline gate cannot be switched off ----------------

def test_negative_minimum_improvement_is_refused():
    """At -5.0 the gate became `candidate <= baseline * 6`, which selects a
    model that lost the backtest and still reports it as supported."""
    from gnomon.evaluation import evaluate

    with pytest.raises(GnomonError) as raised:
        evaluate(_values(), 7, 7, -5.0, frequency="D", tsfm_names=[])
    assert raised.value.code == "INVALID_MINIMUM_IMPROVEMENT"
    assert raised.value.to_dict()["error"]["details"]["supplied"] == -5.0


def test_zero_minimum_improvement_is_allowed():
    """Zero means "must not be worse", which is a coherent request."""
    from gnomon.evaluation import evaluate

    assessment = evaluate(_values(), 7, 7, 0.0, frequency="D", tsfm_names=[])
    assert assessment.selected_model is not None


def test_negative_minimum_improvement_is_refused_through_the_cli(tmp_path, capsys):
    from gnomon.cli import main

    code = main([
        "forecast", str(_csv(tmp_path)), "--time", "timestamp",
        "--target", "value", "--horizon", "7",
        "--minimum-baseline-improvement", "-5.0",
        "--output", str(tmp_path / "out"),
    ])
    assert code != 0
    # Structured errors go to stderr; stdout stays clean for the payload.
    assert "INVALID_MINIMUM_IMPROVEMENT" in capsys.readouterr().err


def test_mcp_schema_bounds_minimum_improvement():
    from gnomon.toolspec import TOOLS

    tool = next(item for item in TOOLS if item["name"] == "gnomon_forecast")
    schema = tool["inputSchema"]["properties"]["minimum_baseline_improvement"]
    assert schema.get("minimum") == 0


# -- H1: a support sentence is derived, never asserted --------------------

def test_forced_ensemble_does_not_claim_it_beat_the_baseline():
    """`--ensemble` on a series a baseline won produced the sentence "An
    ensemble of eligible models beat the strongest baseline" beside a
    measured improvement of exactly zero."""
    from gnomon.evaluation import Evaluation
    from gnomon.support import assess_forecast_support

    lost = Evaluation(
        selected_model="ensemble", strongest_baseline="seasonal_naive",
        selection_scores={}, test_scores={}, improvement=0.0, residuals=[],
        coverage=0.8, warnings=[], supported=True,
    )
    assessment = assess_forecast_support("supported_ensemble", [], lost)
    codes = {reason.code for reason in assessment.reasons}
    assert "ensemble_forced" in codes
    assert "ensemble_selection" not in codes
    assert assessment.status == "conditionally_supported"
    message = next(r.message for r in assessment.reasons if r.code == "ensemble_forced")
    assert "beat" not in message
    assert "0.00%" in message


def test_winning_ensemble_states_the_measured_margin():
    from gnomon.evaluation import Evaluation
    from gnomon.support import assess_forecast_support

    won = Evaluation(
        selected_model="ensemble", strongest_baseline="seasonal_naive",
        selection_scores={}, test_scores={}, improvement=0.15, residuals=[],
        coverage=0.8, warnings=[], supported=True,
    )
    assessment = assess_forecast_support("supported_ensemble", [], won)
    reason = next(r for r in assessment.reasons if r.code == "ensemble_selection")
    assert "15.00%" in reason.message
    assert assessment.status == "supported"


def test_verifier_checks_an_asserted_comparison_against_its_evidence():
    """Reference integrity is satisfied by a sentence that is simply false."""
    from gnomon.lineage import (
        ArtifactRecord, ClaimRecord, ComparisonAssertion, EvidenceRecord, Lineage,
    )
    from gnomon.verifier import verify_lineage

    lineage = Lineage("task_1", {})
    lineage.artifacts.append(ArtifactRecord("artifact_1", "forecast", "2026-01-01T00:00:00"))
    lineage.evidence.append(EvidenceRecord(
        "evaluation:alpha", "rolling_evaluation", "alpha",
        {"baseline_improvement": 0.0},
    ))
    lineage.claims.append(ClaimRecord(
        claim_id="claim:alpha", claim_class="descriptive",
        statement="The ensemble beat the strongest baseline.",
        subject="alpha",
        evidence_ids=("evaluation:alpha",), artifact_ids=("artifact_1",),
        comparisons=(ComparisonAssertion(
            metric="baseline_improvement", evidence_id="evaluation:alpha",
            direction="greater_than", threshold=0.0,
        ),),
    ))
    violations = verify_lineage(lineage, as_of=None)
    assert [item["code"] for item in violations] == ["COMPARISON_UNSUPPORTED"]
    assert violations[0]["measured"] == 0.0


def test_verifier_accepts_a_comparison_the_evidence_supports():
    from gnomon.lineage import (
        ArtifactRecord, ClaimRecord, ComparisonAssertion, EvidenceRecord, Lineage,
    )
    from gnomon.verifier import verify_lineage

    lineage = Lineage("task_1", {})
    lineage.artifacts.append(ArtifactRecord("artifact_1", "forecast", "2026-01-01T00:00:00"))
    lineage.evidence.append(EvidenceRecord(
        "evaluation:alpha", "rolling_evaluation", "alpha",
        {"baseline_improvement": 0.2},
    ))
    lineage.claims.append(ClaimRecord(
        claim_id="claim:alpha", claim_class="descriptive",
        statement="ets beat the strongest baseline.", subject="alpha",
        evidence_ids=("evaluation:alpha",), artifact_ids=("artifact_1",),
        comparisons=(ComparisonAssertion(
            metric="baseline_improvement", evidence_id="evaluation:alpha",
            direction="greater_than", threshold=0.0,
        ),),
    ))
    assert verify_lineage(lineage, as_of=None) == []


def test_verifier_rejects_a_comparison_the_evidence_does_not_report():
    from gnomon.lineage import (
        ArtifactRecord, ClaimRecord, ComparisonAssertion, EvidenceRecord, Lineage,
    )
    from gnomon.verifier import verify_lineage

    lineage = Lineage("task_1", {})
    lineage.artifacts.append(ArtifactRecord("artifact_1", "forecast", "2026-01-01T00:00:00"))
    lineage.evidence.append(EvidenceRecord(
        "evaluation:alpha", "rolling_evaluation", "alpha", {"selection_scores": {}},
    ))
    lineage.claims.append(ClaimRecord(
        claim_id="claim:alpha", claim_class="descriptive",
        statement="…", subject="alpha",
        evidence_ids=("evaluation:alpha",), artifact_ids=("artifact_1",),
        comparisons=(ComparisonAssertion(
            metric="baseline_improvement", evidence_id="evaluation:alpha",
            direction="greater_than", threshold=0.0,
        ),),
    ))
    violations = verify_lineage(lineage, as_of=None)
    assert [item["code"] for item in violations] == ["COMPARISON_UNMEASURED"]


# -- H5: calibration must have measured up, not merely exist --------------

def test_probability_claim_needs_coverage_inside_the_band():
    """A run at 57.1% coverage on a nominal 80% interval used to emit
    verified `predictive` claims, because the gate only checked that a
    `rolling_evaluation` record existed."""
    from gnomon.lineage import ArtifactRecord, ClaimRecord, EvidenceRecord, Lineage
    from gnomon.verifier import verify_lineage

    lineage = Lineage("task_1", {})
    lineage.artifacts.append(ArtifactRecord("artifact_1", "forecast", "2026-01-01T00:00:00"))
    lineage.evidence.append(EvidenceRecord(
        "evaluation:alpha", "rolling_evaluation", "alpha",
        {"measured_interval_coverage": 0.2857},
    ))
    lineage.claims.append(ClaimRecord(
        claim_id="claim:alpha", claim_class="predictive",
        statement="Forecast for alpha.", subject="alpha",
        evidence_ids=("evaluation:alpha",), artifact_ids=("artifact_1",),
        calibration_ref="evaluation:alpha",
    ))
    violations = verify_lineage(lineage, as_of=None)
    assert [item["code"] for item in violations] == ["MISCALIBRATED_PROBABILITY"]


def test_unmeasured_coverage_is_not_treated_as_bad_coverage():
    """A two-fold run has no test fold; refusing it would be wrong."""
    from gnomon.contracts import interval_calibration_is_verifiable

    assert interval_calibration_is_verifiable(None) is True
    assert interval_calibration_is_verifiable(0.8) is True
    assert interval_calibration_is_verifiable(0.2857) is False


def test_bad_coverage_degrades_the_claim_rather_than_crashing(tmp_path):
    """The point path still ships; it stops carrying probability weight."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_covariates import _write_files

    from gnomon.covariates import load_covariates

    observations, covariates_path = _write_files(tmp_path)
    dataset = load_covariates(str(covariates_path), "campaign:binary:future_known")
    artifact, path = forecast(
        str(observations), time_column="timestamp", target_column="value",
        horizon=7, frequency="D", output=str(tmp_path / "out"),
        covariates=dataset, clock=CLOCK,
    )
    result = artifact.results[0]
    assert result.forecast, "the point path was withheld"

    import json
    lineage = json.loads((path / "lineage.json").read_text(encoding="utf-8"))
    forecast_claim = next(
        claim for claim in lineage["claims"]
        if claim["claim_id"].startswith("claim:forecast:")
    )
    if result.interval_coverage is not None and result.interval_coverage < 0.5:
        assert forecast_claim["claim_class"] == "descriptive"
        assert forecast_claim["calibration_ref"] is None
        codes = {r["code"] for r in result.support_assessment["reasons"]}
        assert "interval_coverage_out_of_band" in codes


def test_degraded_does_not_swallow_the_coverage_reason():
    """`degraded` used to replace the reason list, losing the worse fact."""
    from gnomon.evaluation import Evaluation
    from gnomon.support import assess_forecast_support

    assessment = Evaluation(
        selected_model="drift", strongest_baseline="last_value",
        selection_scores={}, test_scores={}, improvement=0.1, residuals=[],
        coverage=0.2, warnings=["Final-test 80% interval coverage was 20.0%, below 70%."],
        supported=True, degraded=True,
    )
    result = assess_forecast_support(
        "degraded", assessment.warnings, assessment,
    )
    codes = [reason.code for reason in result.reasons]
    assert "interval_coverage_out_of_band" in codes
    assert "degraded_evaluation" in codes


# -- H6: the leakage check compares instants, not strings -----------------

def _leakage_lineage(known_time: str):
    from gnomon.lineage import ArtifactRecord, ClaimRecord, Lineage

    lineage = Lineage("task_1", {})
    lineage.artifacts.append(ArtifactRecord(
        "artifact_1", "dataset", "2026-01-01T00:00:00", max_known_time=known_time,
    ))
    lineage.claims.append(ClaimRecord(
        claim_id="claim:alpha", claim_class="descriptive", statement="…",
        subject="alpha", evidence_ids=(), artifact_ids=("artifact_1",),
    ))
    return lineage


def test_offset_aware_leak_is_caught_despite_sorting_safe_as_a_string():
    """The review's own example, restored to a passing check.

    `as_of = 2026-06-04T00:00:00+02:00` is the instant `2026-06-03T22:00:00Z`.
    A known_time of `2026-06-03T23:00:00+00:00` is an hour *past* it — a real
    leak — yet the string `"2026-06-03T23:00:00+00:00"` sorts *before*
    `"2026-06-04T00:00:00+02:00"`, so the old lexicographic check passed it.
    """
    from gnomon.verifier import verify_lineage

    as_of = "2026-06-04T00:00:00+02:00"  # == 2026-06-03T22:00:00Z

    leaking = _leakage_lineage("2026-06-03T23:00:00+00:00")
    assert "2026-06-03T23:00:00+00:00" < as_of, "the string ordering premise changed"
    violations = verify_lineage(leaking, as_of=as_of)
    assert [item["code"] for item in violations] == ["TEMPORAL_LEAKAGE"]

    # An instant genuinely before the boundary is not a leak.
    safe = _leakage_lineage("2026-06-03T21:00:00+00:00")
    assert verify_lineage(safe, as_of=as_of) == []


def test_z_suffix_and_offset_notation_compare_equal():
    from gnomon.verifier import verify_lineage

    lineage = _leakage_lineage("2026-06-04T00:00:00Z")
    assert verify_lineage(lineage, as_of="2026-06-04T00:00:00+00:00") == []


def test_naive_versus_aware_is_a_structured_mismatch():
    from gnomon.verifier import verify_lineage

    lineage = _leakage_lineage("2026-06-04T00:00:00+00:00")
    violations = verify_lineage(lineage, as_of="2026-06-03T00:00:00")
    assert [item["code"] for item in violations] == ["SNAPSHOT_TIMEZONE_MISMATCH"]


def test_unparseable_known_time_is_reported_not_ignored():
    from gnomon.verifier import verify_lineage

    lineage = _leakage_lineage("not-a-timestamp")
    violations = verify_lineage(lineage, as_of="2026-06-04T00:00:00+00:00")
    assert [item["code"] for item in violations] == ["UNPARSEABLE_KNOWN_TIME"]


# -- H7: the covariate cutoff is structural -------------------------------

def test_covariate_snapshot_is_built_at_the_runs_as_of(tmp_path):
    """Built with `as_of=None`, the run's boundary never reached it and the
    cutoff was enforced by convention at each call site."""
    import csv as csv_module

    from gnomon.covariates import load_covariates

    path = tmp_path / "covariates.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv_module.writer(handle)
        writer.writerow(["timestamp", "known_at", "campaign"])
        writer.writerow(["2026-03-01T00:00:00", "2026-01-01T00:00:00", 1])
        writer.writerow(["2026-03-01T00:00:00", "2026-06-01T00:00:00", 0])

    dataset = load_covariates(str(path), "campaign:binary:future_known")
    valid_at = datetime(2026, 3, 1)
    far_future = datetime(2030, 1, 1)

    # Unbound, a far-future cutoff sees the later vintage.
    assert dataset.value_at("campaign", "__default__", valid_at, far_future) == 0

    # Bound to a run at 2026-02-01, the later vintage does not exist —
    # whatever cutoff the caller asks for.
    dataset.bind_as_of(datetime(2026, 2, 1))
    assert dataset.value_at("campaign", "__default__", valid_at, far_future) == 1


def test_covariate_reads_reach_snapshot_access(tmp_path):
    """`max_known_time` used to come from the target series alone, so a
    covariate could not be caught by the verifier's leakage check."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_covariates import _write_files

    from gnomon.covariates import load_covariates

    observations, covariates_path = _write_files(tmp_path)
    dataset = load_covariates(str(covariates_path), "campaign:binary:future_known")
    artifact, _ = forecast(
        str(observations), time_column="timestamp", target_column="value",
        horizon=7, frequency="D", output=str(tmp_path / "out"),
        covariates=dataset, clock=CLOCK,
    )
    access = next(
        item for item in artifact.evidence if item.kind == "snapshot_access"
    ).payload
    sources = {entry.get("source") for entry in access["accesses"]}
    assert "covariates" in sources, "covariate reads left no trace in the evidence"


def test_leakage_lint_guards_the_covariate_module():
    from test_leakage_lint import GUARDED_MODULES

    assert "covariates.py" in GUARDED_MODULES


# -- C7: nothing scores itself --------------------------------------------

def test_meta_model_is_scored_out_of_fold():
    """Fit on all folds and scored on those same folds, the meta-model's
    selection score was in-sample while every member's was out-of-sample —
    it won by construction."""
    import gnomon.evaluation as evaluation_module
    from gnomon import meta_model as meta_module
    from gnomon.config import GnomonConfig

    config = GnomonConfig()
    config.meta_model.enabled = True
    config.ensemble.enabled = True

    calls: list[int] = []
    original_train = meta_module.train_meta_model
    produced: list[dict] = []

    def counting_train(forecasts, actuals, **kwargs):
        calls.append(len(actuals))
        weights = original_train(forecasts, actuals, **kwargs)
        produced.append(weights or {})
        return weights

    meta_module.train_meta_model = counting_train
    try:
        evaluation_module.evaluate(
            _values(160), 7, 7, 0.02, frequency="D", tsfm_names=[], config=config,
        )
    finally:
        meta_module.train_meta_model = original_train

    assert calls, "the meta-model never trained"
    full = max(calls)
    assert calls.count(full) == 1, "more than one fit saw every fold"
    if not produced[0]:
        pytest.skip("the meta-model produced no weights on this series")
    # One fit on every fold (the weights that ship), then one leave-one-out
    # refit per fold, each seeing exactly one fold fewer.
    assert calls.count(full - 1) == full, "not every fold got a held-out refit"


def test_ensemble_fold_weights_use_only_earlier_folds():
    """Weights came from aggregates over every fold including the one being
    scored, so each fold's ensemble was weighted using its own outcome."""
    import gnomon.ensemble as ensemble_module
    from gnomon.config import GnomonConfig
    from gnomon.evaluation import evaluate

    config = GnomonConfig()
    config.ensemble.enabled = True

    seen: list[dict] = []
    original = ensemble_module.compute_ensemble_forecast

    def spy(forecasts, scores, **kwargs):
        seen.append(dict(scores))
        return original(forecasts, scores, **kwargs)

    ensemble_module.compute_ensemble_forecast = spy
    import gnomon.evaluation as evaluation_module
    try:
        evaluate(_values(160), 7, 7, 0.02, frequency="D", tsfm_names=[], config=config)
    finally:
        ensemble_module.compute_ensemble_forecast = original

    assert seen, "the ensemble never ran"
    # The first fold has no prior evidence, so every member scores None.
    assert all(value is None for value in seen[0].values())
    # A later fold does have prior evidence.
    assert any(value is not None for value in seen[-1].values())

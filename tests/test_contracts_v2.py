"""Phase 2 contracts: TemporalTask, SupportAssessment, typed lineage, and
the claim verifier."""

from datetime import datetime, timezone
from pathlib import Path

import json
import pytest

from gnomon.contracts import (
    GnomonError,
    DataSourceRef,
    ForecastSpec,
    SupportAssessment,
    SupportReason,
    TemporalTask,
    forecast_task,
)
from gnomon.ids import FixedClock
from gnomon.lineage import ArtifactRecord, ClaimRecord, EvidenceRecord, Lineage
from gnomon.runtime import forecast
from gnomon.support import assess_forecast_support
from gnomon.verifier import verify_lineage, verify_or_raise

REPO = Path(__file__).resolve().parent.parent
CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))


# -- TemporalTask ---------------------------------------------------------

def test_forecast_task_is_a_temporal_task():
    task = forecast_task(
        "data.csv", time_column="ts", target_column="y", horizon=7,
    )
    assert task.task_type == "forecast"
    assert task.sources == (DataSourceRef("data.csv", "ts", "y"),)
    assert isinstance(task.outputs[0], ForecastSpec)
    assert task.outputs[0].horizon == 7
    assert task.task_id().startswith("task_")
    # Identity is content-addressed and stable.
    assert task.task_id() == forecast_task(
        "data.csv", time_column="ts", target_column="y", horizon=7,
    ).task_id()


# -- SupportAssessment mapping -------------------------------------------

@pytest.mark.parametrize("legacy,expected_status", [
    ("supported", "supported"),
    ("supported_ensemble", "supported"),
    ("weakly_supported", "conditionally_supported"),
    ("degraded", "conditionally_supported"),
    ("unsupported", "inconclusive"),
])
def test_legacy_support_mapping(legacy, expected_status):
    result = assess_forecast_support(legacy, ["some warning"], None)
    assert result.status == expected_status
    assert result.legacy_support == legacy


def test_unsupported_maps_to_inconclusive_with_recovery():
    result = assess_forecast_support("unsupported", ["Need at least 30 observations"], None)
    assert result.status == "inconclusive"
    assert any(reason.code == "insufficient_evaluation" for reason in result.reasons)
    assert any(action.code == "provide_more_history" for action in result.recovery_actions)


def test_abstention_recovery_names_supportable_horizon():
    """A refusal is not a dead end: when a shorter horizon would succeed
    with the data already supplied, the recovery actions say which."""
    from gnomon.evaluation import evaluate

    assessment = evaluate([1.0, 2.0, 3.0, 4.0, 5.0], horizon=24, season=24,
                          minimum_improvement=0.02)
    assert assessment.max_supportable_horizon == 3
    assert any("--horizon 3" in warning for warning in assessment.warnings)
    result = assess_forecast_support("unsupported", assessment.warnings, assessment)
    reduce = [a for a in result.recovery_actions if a.code == "reduce_horizon"]
    assert reduce and "horizon 3" in reduce[0].message


def test_strict_abstention_recovery_names_supportable_horizon():
    from gnomon.evaluation import evaluate

    values = [float(i) for i in range(40)]
    assessment = evaluate(values, horizon=24, season=7, minimum_improvement=0.02,
                          strict_abstention=True)
    assert assessment.supported is False
    assert assessment.max_supportable_horizon == 10
    assert any("--horizon 10" in warning for warning in assessment.warnings)


def test_known_time_assumption_is_declared():
    result = assess_forecast_support("supported", [], None, known_time_assumed=True)
    assert any("known_time_assumed" in assumption for assumption in result.assumptions)


# -- Verifier -------------------------------------------------------------

def _lineage_with(claims: list[ClaimRecord], max_known_time: str | None = None) -> Lineage:
    lineage = Lineage("task_x", {})
    lineage.artifacts.append(ArtifactRecord(
        "dataset:abc", "dataset", "2026-07-01T00:00:00+00:00",
        max_known_time=max_known_time,
    ))
    lineage.evidence.append(EvidenceRecord(
        "evaluation:s", "rolling_evaluation", "s", {"scores": {}},
    ))
    lineage.evidence.append(EvidenceRecord(
        "ablation:s", "covariate_ablation", "s", {"admitted": False},
    ))
    lineage.claims.extend(claims)
    return lineage


def test_verifier_rejects_causal_claim_from_associational_evidence():
    lineage = _lineage_with([ClaimRecord(
        "claim:causal", "causal", "X caused Y", "s",
        evidence_ids=("ablation:s",), artifact_ids=("dataset:abc",),
    )])
    violations = verify_lineage(lineage, as_of=None)
    assert [item["code"] for item in violations] == ["CAUSAL_FROM_ASSOCIATIONAL"]


def test_verifier_rejects_uncalibrated_probability():
    lineage = _lineage_with([ClaimRecord(
        "claim:prob", "predictive", "P(exceed) = 0.7", "s",
        evidence_ids=("ablation:s",), artifact_ids=("dataset:abc",),
        calibration_ref=None,
    )])
    violations = verify_lineage(lineage, as_of=None)
    assert [item["code"] for item in violations] == ["UNCALIBRATED_PROBABILITY"]


def test_verifier_rejects_unevaluated_decision_constraints():
    lineage = _lineage_with([ClaimRecord(
        "claim:decide", "decision", "Do X", "s",
        evidence_ids=("evaluation:s",), artifact_ids=("dataset:abc",),
        constraints_evaluated=None,
    )])
    violations = verify_lineage(lineage, as_of=None)
    assert [item["code"] for item in violations] == ["DECISION_CONSTRAINTS_UNEVALUATED"]


def test_verifier_rejects_post_as_of_artifact():
    lineage = _lineage_with(
        [ClaimRecord(
            "claim:ok", "descriptive", "level rose", "s",
            evidence_ids=("evaluation:s",), artifact_ids=("dataset:abc",),
        )],
        max_known_time="2026-06-15T00:00:00",
    )
    violations = verify_lineage(lineage, as_of="2026-06-01T00:00:00")
    assert [item["code"] for item in violations] == ["TEMPORAL_LEAKAGE"]
    assert violations[0]["known_time"] == "2026-06-15T00:00:00"


def test_verifier_rejects_dangling_references():
    lineage = _lineage_with([ClaimRecord(
        "claim:dangling", "descriptive", "…", "s",
        evidence_ids=("missing:evidence",), artifact_ids=("dataset:abc",),
    )])
    violations = verify_lineage(lineage, as_of=None)
    assert [item["code"] for item in violations] == ["DANGLING_REFERENCE"]


def test_verify_or_raise_is_structured():
    lineage = _lineage_with([ClaimRecord(
        "claim:causal", "causal", "X caused Y", "s",
        evidence_ids=("ablation:s",), artifact_ids=("dataset:abc",),
    )])
    with pytest.raises(GnomonError) as caught:
        verify_or_raise(lineage, as_of=None)
    assert caught.value.code == "CLAIM_VERIFICATION_FAILED"
    assert caught.value.details["violations"][0]["code"] == "CAUSAL_FROM_ASSOCIATIONAL"


# -- End-to-end: forecast runs emit verified typed lineage ---------------

def test_forecast_emits_verified_lineage(tmp_path):
    artifact, directory = forecast(
        str(REPO / "examples" / "daily_requests.csv"),
        time_column="timestamp", target_column="requests", horizon=7,
        output=str(tmp_path), clock=CLOCK,
    )
    lineage = json.loads((directory / "lineage.json").read_text())
    assert lineage["task"]["task_type"] == "forecast"
    kinds = {item["kind"] for item in lineage["artifacts"]}
    assert kinds == {"dataset", "forecast"}
    claims = {item["claim_id"]: item for item in lineage["claims"]}
    forecast_claim = claims["claim:forecast:__default__"]
    assert forecast_claim["claim_class"] == "predictive"
    assert forecast_claim["calibration_ref"] == "evaluation:__default__"
    # Every result carries the harness-wide assessment alongside the frozen enum.
    assert artifact.results[0].support_assessment["status"] in (
        "supported", "conditionally_supported", "inconclusive",
    )


def test_abstention_emits_descriptive_claim(tmp_path):
    source = tmp_path / "short.csv"
    source.write_text("timestamp,value\n" + "\n".join(
        f"2026-03-{day:02d},{100 + day}" for day in range(1, 7)
    ) + "\n")
    artifact, directory = forecast(
        str(source), time_column="timestamp", target_column="value", horizon=3,
        output=str(tmp_path / "out"), clock=CLOCK, strict_abstention=True,
    )
    lineage = json.loads((directory / "lineage.json").read_text())
    claim = lineage["claims"][0]
    assert claim["claim_id"] == "claim:abstention:__default__"
    assert claim["claim_class"] == "descriptive"
    assert artifact.results[0].support_assessment["status"] == "inconclusive"

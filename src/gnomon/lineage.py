"""Typed lineage: the record kinds every Gnomon response is made of.

Five kinds, each with identity and explicit references — **artifacts**
(datasets, forecasts, scores), **evidence** (measurements), **claims**
(conclusions with a claim class and the evidence they cite), **actions**,
and **outcomes**. The claim verifier walks these records; free-form payload
dictionaries cannot be verified, which is exactly why these are typed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .contracts import (
    ClaimClass,
    ForecastArtifact,
    TemporalTask,
    interval_calibration_is_verifiable,
)


@dataclass(frozen=True)
class ArtifactRecord:
    record_id: str
    kind: str  # dataset | forecast | score | scenario | decision
    created_at: str
    fingerprint: str | None = None
    max_known_time: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceRecord:
    record_id: str
    kind: str  # rolling_evaluation | support_assessment | snapshot_access | ...
    subject: str
    measurements: dict[str, Any]
    derived_from: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComparisonAssertion:
    """A comparison a claim's prose asserts, in a form the verifier can check.

    Prose like "beat the strongest baseline" is unfalsifiable to a checker
    that only validates references and kinds — which is how a claim stating
    the ensemble won came to verify beside evidence measuring an
    improvement of exactly zero. Stating the comparison structurally beside
    the sentence makes the sentence checkable against its own evidence.
    """

    #: Dotted path into the cited evidence record's payload.
    metric: str
    #: Which evidence record reports it.
    evidence_id: str
    #: One of the comparators in ``verifier._COMPARATORS``.
    direction: str
    threshold: float


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    claim_class: ClaimClass
    statement: str
    subject: str
    evidence_ids: tuple[str, ...]
    artifact_ids: tuple[str, ...]
    # Evidence record that carries the interval/probability calibration this
    # claim's numbers rest on. Mandatory for any claim stating probabilities.
    calibration_ref: str | None = None
    # Decision claims must state whether their constraints were evaluated.
    constraints_evaluated: bool | None = None
    # Measurable comparisons this claim's statement asserts. Each is checked
    # against the evidence it names; a claim that says more than its
    # evidence supports fails verification.
    comparisons: tuple[ComparisonAssertion, ...] = ()


@dataclass(frozen=True)
class ActionRecord:
    action_id: str
    subject: str
    description: str
    claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutcomeRecord:
    outcome_id: str
    action_id: str
    measurements: dict[str, Any]
    realised_at: str | None = None


@dataclass
class Lineage:
    task_id: str
    task: dict[str, Any]
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    evidence: list[EvidenceRecord] = field(default_factory=list)
    claims: list[ClaimRecord] = field(default_factory=list)
    actions: list[ActionRecord] = field(default_factory=list)
    outcomes: list[OutcomeRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "0.1",
            "task_id": self.task_id,
            "task": self.task,
            "artifacts": [asdict(item) for item in self.artifacts],
            "evidence": [asdict(item) for item in self.evidence],
            "claims": [asdict(item) for item in self.claims],
            "actions": [asdict(item) for item in self.actions],
            "outcomes": [asdict(item) for item in self.outcomes],
        }


def build_forecast_lineage(
    artifact: ForecastArtifact,
    task: TemporalTask,
) -> Lineage:
    """Typed lineage for a forecast run: one dataset artifact, one forecast
    artifact, evidence records mirroring the run's measurements, and one
    claim per series — predictive where a forecast was published,
    descriptive where the run abstained."""
    from dataclasses import asdict as dataclass_asdict

    dataset_id = f"dataset:{artifact.source_fingerprint.split(':')[-1][:24]}"
    snapshot_payloads = [item for item in artifact.evidence if item.kind == "snapshot_access"]
    max_known = None
    if snapshot_payloads:
        known_times = [
            str(access.get("max_known_time"))
            for access in snapshot_payloads[0].payload.get("accesses", [])
            if access.get("max_known_time") is not None
        ]
        max_known = max(known_times) if known_times else None
    lineage = Lineage(task.task_id(), dataclass_asdict(task))
    lineage.artifacts.append(ArtifactRecord(
        dataset_id, "dataset", artifact.created_at,
        fingerprint=artifact.source_fingerprint, max_known_time=max_known,
        meta={"ref": artifact.task.input_path},
    ))
    lineage.artifacts.append(ArtifactRecord(
        artifact.forecast_id, "forecast", artifact.created_at,
        meta={"status": artifact.status, "horizon": artifact.task.horizon},
    ))
    for item in artifact.evidence:
        lineage.evidence.append(EvidenceRecord(
            item.evidence_id, item.kind, item.series, item.payload,
            derived_from=(dataset_id,),
        ))
    for result in artifact.results:
        evaluation_id = f"evaluation:{result.series}"
        support_id = f"support:{result.series}"
        if result.forecast and result.support == "best_effort":
            # Sub-supported rows were published. The claim class is
            # descriptive — never predictive — so the verifier's
            # calibration gate cannot be satisfied by numbers no fold ever
            # tested, and the statement carries the tier label the
            # verifier requires of every claim quoting such rows.
            split_reason = next(
                (reason for reason in
                 (result.support_assessment or {}).get("reasons", [])
                 if reason.get("code") == "horizon_split"), None)
            if split_reason is not None:
                # A horizon split: the prefix is an evaluated forecast, the
                # remainder is the fallback. One claim naming both ranges
                # and both tiers; the prefix's own calibration lives in the
                # `:prefix` evaluation evidence, deliberately not promoted
                # to a probability-bearing claim for the merged object.
                tiers = sorted({str(row.get("tier", "best_effort"))
                                for row in result.forecast})
                lineage.claims.append(ClaimRecord(
                    claim_id=f"claim:horizon_split:{result.series}",
                    claim_class="descriptive",
                    statement=(
                        f"Horizon-split forecast for series "
                        f"{result.series}: {split_reason.get('message')} "
                        f"Row tiers present: {', '.join(tiers)}. The "
                        f"best_effort rows carry no measured accuracy and "
                        f"no probability weight."
                    ),
                    subject=result.series,
                    evidence_ids=(evaluation_id, support_id),
                    artifact_ids=(artifact.forecast_id, dataset_id),
                ))
                continue
            lineage.claims.append(ClaimRecord(
                claim_id=f"claim:best_effort:{result.series}",
                claim_class="descriptive",
                statement=(
                    f"No reliable forecast exists for series {result.series}: "
                    "the evaluation protocol could not run. The published "
                    f"{len(result.forecast)} rows are a last-value fallback "
                    "with dispersion-scaled intervals, published at the "
                    "best_effort tier under the request's minimum_support "
                    "floor; they carry no measured accuracy and "
                    "no probability weight."
                ),
                subject=result.series,
                evidence_ids=(evaluation_id, support_id),
                artifact_ids=(artifact.forecast_id, dataset_id),
            ))
            continue
        if result.forecast:
            # A probability-bearing claim needs calibration that measured
            # up, not merely a calibration record that exists. Outside the
            # band the run still publishes its point path — it just stops
            # claiming the intervals mean what they say, which is a
            # downgrade to `descriptive` rather than a failure.
            calibrated = interval_calibration_is_verifiable(result.interval_coverage)
            # A selection over the mandated baselines is a comparison, and a
            # comparison that is stated has to be checkable. Baselines make
            # no such claim about themselves, so they carry no assertion.
            comparisons: tuple[ComparisonAssertion, ...] = ()
            beat_baseline = (
                result.selected_model is not None
                and result.strongest_baseline is not None
                and result.selected_model != result.strongest_baseline
                and (result.baseline_improvement or 0) > 0
            )
            if beat_baseline:
                comparisons = (ComparisonAssertion(
                    metric="baseline_improvement",
                    evidence_id=evaluation_id,
                    direction="greater_than",
                    threshold=0.0,
                ),)
            statement = (
                f"Forecast for series {result.series} over "
                f"{len(result.forecast)} periods, selected model "
                f"{result.selected_model}, with 10/50/90 residual-quantile intervals."
            )
            if beat_baseline:
                statement += (
                    f" {result.selected_model} beat the strongest baseline "
                    f"{result.strongest_baseline} on the selection folds."
                )
            if not calibrated:
                statement += (
                    f" Measured interval coverage is "
                    f"{result.interval_coverage:.1%}, so the intervals are "
                    f"reported without probability weight."
                )
            if result.support == "context_trusted":
                statement += (
                    " The forecast was influenced by future-dated context "
                    "events admitted on textual verification, not fold "
                    "ablation; the history-only counterfactual is in the "
                    "future_context_applied evidence."
                )
            lineage.claims.append(ClaimRecord(
                claim_id=f"claim:forecast:{result.series}",
                claim_class="predictive" if calibrated else "descriptive",
                statement=statement,
                subject=result.series,
                evidence_ids=(evaluation_id, support_id),
                artifact_ids=(artifact.forecast_id, dataset_id),
                calibration_ref=evaluation_id if calibrated else None,
                comparisons=comparisons,
            ))
            if result.threshold:
                lineage.claims.append(ClaimRecord(
                    claim_id=f"claim:threshold:{result.series}",
                    claim_class="predictive" if calibrated else "descriptive",
                    statement=(
                        f"Probability of series {result.series} exceeding "
                        f"{result.threshold['value']} per horizon step."
                        + ("" if calibrated else
                           " Reported without probability weight: measured "
                           "interval coverage is outside the verifiable band.")
                    ),
                    subject=result.series,
                    evidence_ids=(evaluation_id, support_id),
                    artifact_ids=(artifact.forecast_id, dataset_id),
                    calibration_ref=evaluation_id if calibrated else None,
                ))
        else:
            lineage.claims.append(ClaimRecord(
                claim_id=f"claim:abstention:{result.series}",
                claim_class="descriptive",
                statement=(
                    f"Gnomon abstained on series {result.series}: the evaluation "
                    "protocol could not establish a supported forecast."
                ),
                subject=result.series,
                evidence_ids=(support_id,),
                artifact_ids=(dataset_id,),
            ))
    return lineage

"""Typed lineage: the record kinds every Aion response is made of.

Five kinds, each with identity and explicit references — **artifacts**
(datasets, forecasts, scores), **evidence** (measurements), **claims**
(conclusions with a claim class and the evidence they cite), **actions**,
and **outcomes**. The claim verifier walks these records; free-form payload
dictionaries cannot be verified, which is exactly why these are typed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .contracts import ClaimClass, ForecastArtifact, TemporalTask


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
        if result.forecast:
            lineage.claims.append(ClaimRecord(
                claim_id=f"claim:forecast:{result.series}",
                claim_class="predictive",
                statement=(
                    f"Forecast for series {result.series} over "
                    f"{len(result.forecast)} periods, selected model "
                    f"{result.selected_model}, with 10/50/90 residual-quantile intervals."
                ),
                subject=result.series,
                evidence_ids=(evaluation_id, support_id),
                artifact_ids=(artifact.forecast_id, dataset_id),
                calibration_ref=evaluation_id,
            ))
            if result.threshold:
                lineage.claims.append(ClaimRecord(
                    claim_id=f"claim:threshold:{result.series}",
                    claim_class="predictive",
                    statement=(
                        f"Probability of series {result.series} exceeding "
                        f"{result.threshold['value']} per horizon step."
                    ),
                    subject=result.series,
                    evidence_ids=(evaluation_id, support_id),
                    artifact_ids=(artifact.forecast_id, dataset_id),
                    calibration_ref=evaluation_id,
                ))
        else:
            lineage.claims.append(ClaimRecord(
                claim_id=f"claim:abstention:{result.series}",
                claim_class="descriptive",
                statement=(
                    f"Aion abstained on series {result.series}: the evaluation "
                    "protocol could not establish a supported forecast."
                ),
                subject=result.series,
                evidence_ids=(support_id,),
                artifact_ids=(dataset_id,),
            ))
    return lineage

"""The claim verifier: a deterministic checker every response passes
through before it leaves the process.

It rejects, mechanically and without judgement:

- causal (or counterfactual) claims backed only by associational or
  descriptive evidence;
- probability-bearing claims not traceable to a calibration artifact;
- decision claims whose stated constraints were never evaluated;
- any claim citing an artifact whose ``known_time`` lies past the task's
  ``as_of``;
- dangling evidence or artifact references.

Verification appreciates as models improve: better proposals pass more
often, and the guarantee never weakens.
"""

from __future__ import annotations

from typing import Any

from .contracts import AionError
from .lineage import Lineage

# Evidence kinds capable of carrying causal weight. Deliberately empty: the
# causal operator family is out of scope until each member ships with honest
# abstention criteria. Granger-style results, when they arrive, register as
# associational — never here.
CAUSAL_CAPABLE_KINDS: frozenset[str] = frozenset()

# Evidence kinds that carry interval/probability calibration.
CALIBRATION_KINDS = frozenset({"rolling_evaluation"})

PROBABILITY_CLASSES = frozenset({"predictive", "counterfactual"})


def verify_lineage(lineage: Lineage, *, as_of: str | None) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    evidence_by_id = {item.record_id: item for item in lineage.evidence}
    artifacts_by_id = {item.record_id: item for item in lineage.artifacts}

    for claim in lineage.claims:
        missing = [ref for ref in claim.evidence_ids if ref not in evidence_by_id]
        missing += [ref for ref in claim.artifact_ids if ref not in artifacts_by_id]
        if missing:
            violations.append({
                "code": "DANGLING_REFERENCE", "claim_id": claim.claim_id,
                "message": f"Claim cites records that do not exist: {missing}",
                "missing": missing,
            })
            continue

        if claim.claim_class in ("causal", "counterfactual"):
            kinds = {evidence_by_id[ref].kind for ref in claim.evidence_ids}
            if not kinds or not kinds <= CAUSAL_CAPABLE_KINDS:
                violations.append({
                    "code": "CAUSAL_FROM_ASSOCIATIONAL", "claim_id": claim.claim_id,
                    "message": (
                        f"A {claim.claim_class} claim cites only evidence kinds "
                        f"{sorted(kinds)}, none of which can carry causal weight."
                    ),
                    "evidence_kinds": sorted(kinds),
                })

        if claim.claim_class in PROBABILITY_CLASSES:
            calibration = evidence_by_id.get(claim.calibration_ref or "")
            if calibration is None or calibration.kind not in CALIBRATION_KINDS:
                violations.append({
                    "code": "UNCALIBRATED_PROBABILITY", "claim_id": claim.claim_id,
                    "message": (
                        "A probability-bearing claim must trace to a calibration "
                        "artifact; calibration_ref is "
                        + (repr(claim.calibration_ref) if claim.calibration_ref else "missing")
                        + "."
                    ),
                })

        if claim.claim_class == "decision" and claim.constraints_evaluated is not True:
            violations.append({
                "code": "DECISION_CONSTRAINTS_UNEVALUATED", "claim_id": claim.claim_id,
                "message": "A decision claim's stated constraints were never evaluated.",
            })

        if as_of is not None:
            for ref in claim.artifact_ids:
                artifact = artifacts_by_id[ref]
                if artifact.max_known_time is not None and artifact.max_known_time > as_of:
                    violations.append({
                        "code": "TEMPORAL_LEAKAGE", "claim_id": claim.claim_id,
                        "message": (
                            f"Claim cites artifact {ref} containing data known at "
                            f"{artifact.max_known_time}, after the task as_of {as_of}."
                        ),
                        "artifact_id": ref,
                        "known_time": artifact.max_known_time,
                        "as_of": as_of,
                    })
    return violations


def verify_or_raise(lineage: Lineage, *, as_of: str | None) -> None:
    violations = verify_lineage(lineage, as_of=as_of)
    if violations:
        raise AionError(
            "CLAIM_VERIFICATION_FAILED",
            f"{len(violations)} claim(s) failed deterministic verification.",
            {"violations": violations},
        )

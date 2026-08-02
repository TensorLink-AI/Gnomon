"""The claim verifier: a deterministic checker every response passes
through before it leaves the process.

It rejects, mechanically and without judgement:

- causal (or counterfactual) claims backed only by associational or
  descriptive evidence;
- probability-bearing claims not traceable to a calibration artifact
  *whose measured coverage is inside a stated band* — the existence of a
  calibration record is not evidence that the calibration is any good;
- claims asserting a measured comparison that the cited evidence does not
  support;
- decision claims whose stated constraints were never evaluated;
- any claim citing an artifact whose ``known_time`` lies past the task's
  ``as_of``, compared as instants rather than as strings;
- dangling evidence or artifact references.

Verification appreciates as models improve: better proposals pass more
often, and the guarantee never weakens.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .contracts import (
    MAX_VERIFIABLE_COVERAGE,
    MIN_VERIFIABLE_COVERAGE,
    AionError,
    interval_calibration_is_verifiable,  # noqa: F401  (re-exported)
)
from .lineage import Lineage

# Evidence kinds capable of carrying causal weight. Deliberately empty: the
# causal operator family is out of scope until each member ships with honest
# abstention criteria. Granger-style results, when they arrive, register as
# associational — never here.
CAUSAL_CAPABLE_KINDS: frozenset[str] = frozenset()

# Evidence kinds that carry interval/probability calibration.
CALIBRATION_KINDS = frozenset({"rolling_evaluation"})

PROBABILITY_CLASSES = frozenset({"predictive", "counterfactual"})

def _as_instant(value: Any) -> datetime | None:
    """Parse an ISO timestamp, tolerating a trailing ``Z``."""
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


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
            else:
                # The record existing is not the same as the calibration
                # being any good. Gate on the measured value.
                coverage = calibration.measurements.get("measured_interval_coverage")
                if coverage is not None and not (
                    MIN_VERIFIABLE_COVERAGE <= float(coverage) <= MAX_VERIFIABLE_COVERAGE
                ):
                    violations.append({
                        "code": "MISCALIBRATED_PROBABILITY",
                        "claim_id": claim.claim_id,
                        "message": (
                            f"Measured interval coverage is {float(coverage):.1%}, "
                            f"outside the verifiable band "
                            f"[{MIN_VERIFIABLE_COVERAGE:.0%}, "
                            f"{MAX_VERIFIABLE_COVERAGE:.0%}]; a probability-bearing "
                            f"claim cannot rest on it."
                        ),
                        "measured_coverage": float(coverage),
                        "band": [MIN_VERIFIABLE_COVERAGE, MAX_VERIFIABLE_COVERAGE],
                    })

        for comparison in claim.comparisons:
            violation = _check_comparison(claim, comparison, evidence_by_id)
            if violation is not None:
                violations.append(violation)

        if claim.claim_class == "decision" and claim.constraints_evaluated is not True:
            violations.append({
                "code": "DECISION_CONSTRAINTS_UNEVALUATED", "claim_id": claim.claim_id,
                "message": "A decision claim's stated constraints were never evaluated.",
            })

        if as_of is not None:
            # Instants, not strings. `2026-06-03T23:00:00+00:00` sorts after
            # an as_of of `2026-06-04T00:00:00+02:00` lexicographically while
            # being an hour *before* it — a real leak that compared as safe,
            # and the inverse false alarm is equally available.
            boundary = _as_instant(as_of)
            for ref in claim.artifact_ids:
                artifact = artifacts_by_id[ref]
                if artifact.max_known_time is None:
                    continue
                known = _as_instant(artifact.max_known_time)
                if known is None or boundary is None:
                    violations.append({
                        "code": "UNPARSEABLE_KNOWN_TIME", "claim_id": claim.claim_id,
                        "message": (
                            f"Cannot compare artifact {ref}'s known_time "
                            f"{artifact.max_known_time!r} to as_of {as_of!r}: one "
                            f"of them is not an ISO timestamp."
                        ),
                        "artifact_id": ref,
                        "known_time": artifact.max_known_time,
                        "as_of": as_of,
                    })
                    continue
                if (known.tzinfo is None) != (boundary.tzinfo is None):
                    violations.append({
                        "code": "SNAPSHOT_TIMEZONE_MISMATCH", "claim_id": claim.claim_id,
                        "message": (
                            f"Artifact {ref}'s known_time and the task as_of mix "
                            f"timezone-aware and naive timestamps; the leakage "
                            f"check cannot compare them."
                        ),
                        "artifact_id": ref,
                        "known_time": artifact.max_known_time,
                        "as_of": as_of,
                    })
                    continue
                if known > boundary:
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


#: How a comparison assertion may relate a measured value to its threshold.
_COMPARATORS = {
    "greater_than": lambda value, bound: value > bound,
    "at_least": lambda value, bound: value >= bound,
    "less_than": lambda value, bound: value < bound,
    "at_most": lambda value, bound: value <= bound,
}


def _check_comparison(claim, comparison, evidence_by_id) -> dict[str, Any] | None:
    """Check one asserted comparison against the evidence it cites.

    A claim that *states* a comparison — "beat the strongest baseline" — is
    only as good as the measurement behind it. Reference integrity and
    class/kind compatibility, which is all the verifier used to check, are
    satisfied by a sentence that is simply false of the numbers beside it.
    """
    evidence = evidence_by_id.get(comparison.evidence_id)
    if evidence is None:
        return {
            "code": "COMPARISON_EVIDENCE_MISSING", "claim_id": claim.claim_id,
            "message": (
                f"Claim asserts {comparison.metric} "
                f"{comparison.direction.replace('_', ' ')} {comparison.threshold}, "
                f"citing evidence {comparison.evidence_id!r} that does not exist."
            ),
        }
    value: Any = evidence.measurements
    for part in comparison.metric.split("."):
        if not isinstance(value, dict) or part not in value:
            return {
                "code": "COMPARISON_UNMEASURED", "claim_id": claim.claim_id,
                "message": (
                    f"Claim asserts a comparison on {comparison.metric!r}, which "
                    f"evidence {comparison.evidence_id!r} does not report."
                ),
            }
        value = value[part]
    comparator = _COMPARATORS.get(comparison.direction)
    if comparator is None:
        return {
            "code": "COMPARISON_UNKNOWN_DIRECTION", "claim_id": claim.claim_id,
            "message": (
                f"Unknown comparison direction {comparison.direction!r}; "
                f"expected one of {sorted(_COMPARATORS)}."
            ),
        }
    if value is None or not comparator(float(value), float(comparison.threshold)):
        return {
            "code": "COMPARISON_UNSUPPORTED", "claim_id": claim.claim_id,
            "message": (
                f"Claim states {comparison.metric} "
                f"{comparison.direction.replace('_', ' ')} {comparison.threshold}, "
                f"but evidence {comparison.evidence_id!r} measures {value}."
            ),
            "metric": comparison.metric,
            "measured": value,
            "threshold": comparison.threshold,
            "direction": comparison.direction,
        }
    return None


def verify_or_raise(lineage: Lineage, *, as_of: str | None) -> None:
    violations = verify_lineage(lineage, as_of=as_of)
    if violations:
        raise AionError(
            "CLAIM_VERIFICATION_FAILED",
            f"{len(violations)} claim(s) failed deterministic verification.",
            {"violations": violations},
        )

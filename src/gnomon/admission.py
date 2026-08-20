"""Evidence-aware candidate admission.

Candidate classes answer the same question -- expected loss relative to the
strongest robust baseline -- but need not earn their evidence in the same
way.  A locally fitted model starts with local folds; a pretrained model may
also carry independently measured cross-series transfer evidence.  This
module keeps those sources separate and turns them into a reproducible,
versioned decision.  It contains no dataset, channel, or benchmark labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import erf, isfinite, sqrt
from statistics import mean, median, stdev
from typing import Literal


ModelClass = Literal["baseline", "locally_fitted", "pretrained", "remote"]
AdmissionState = Literal[
    "locally_validated", "externally_validated", "jointly_validated",
    "prior_assisted", "baseline_fallback", "unsupported",
]
PointPolicy = Literal["candidate", "baseline", "shrunk_blend"]


@dataclass(frozen=True)
class ExternalModelPrior:
    """Independent evidence for transfer to a declared temporal regime.

    ``mean_relative_gain`` is ``(baseline_loss-candidate_loss)/baseline_loss``;
    positive values favour the candidate.  The uncertainty is required: a
    leaderboard point estimate alone is not admissible evidence.
    """

    model: str
    revision: str
    regime: tuple[tuple[str, str], ...]
    comparisons: int
    mean_relative_gain: float
    standard_error: float
    source_ids: tuple[str, ...]
    registry_version: str
    overlap_risk: Literal["low", "unknown", "high"] = "unknown"
    baseline_reference: str = "strongest_robust_baseline"

    def usable(self, *, minimum_comparisons: int = 30) -> bool:
        return (
            self.comparisons >= minimum_comparisons
            and self.standard_error > 0
            and isfinite(self.mean_relative_gain)
            and isfinite(self.standard_error)
            and bool(self.source_ids)
            and self.overlap_risk != "high"
            and self.baseline_reference == "strongest_robust_baseline"
        )


@dataclass(frozen=True)
class OutputDiagnostics:
    valid: bool = True
    boundary_jump: float | None = None
    scale_ratio: float | None = None
    oscillation_ratio: float | None = None
    candidate_baseline_disagreement: float | None = None
    reasons: tuple[str, ...] = ()


def output_diagnostics(history: list[float], candidate: list[float],
                       baseline: list[float]) -> OutputDiagnostics:
    """Model-agnostic boundary, scale, oscillation, and disagreement facts.

    These are disclosures, not hand-authored rejection rules. Structural
    validity remains the adapter boundary's job; callers may use independently
    calibrated tails later without changing this contract.
    """
    if (not history or not candidate or len(candidate) != len(baseline)
            or any(not isfinite(value)
                   for value in [*history, *candidate, *baseline])):
        return OutputDiagnostics(False, reasons=("non-finite or mismatched output",))
    diffs = [abs(right - left) for left, right in zip(history, history[1:])]
    robust_scale = median(diffs) if diffs else 0.0
    if robust_scale <= 1e-12:
        robust_scale = max(max(history) - min(history), 1.0)
    candidate_diffs = [right - left for left, right in zip(candidate, candidate[1:])]
    sign_changes = sum(left * right < 0
                       for left, right in zip(candidate_diffs, candidate_diffs[1:]))
    return OutputDiagnostics(
        valid=True,
        boundary_jump=abs(candidate[0] - history[-1]) / robust_scale,
        scale_ratio=(median(abs(item) for item in candidate_diffs) / robust_scale
                     if candidate_diffs else 0.0),
        oscillation_ratio=(sign_changes / max(len(candidate_diffs) - 1, 1)),
        candidate_baseline_disagreement=(
            mean(abs(left - right) for left, right in zip(candidate, baseline))
            / robust_scale),
    )


@dataclass(frozen=True)
class AdmissionEvidence:
    model_class: ModelClass
    independent_folds: int
    paired_folds: int
    candidate_loss: float | None
    baseline_loss: float | None
    relative_improvement: float | None
    candidate_win_rate: float | None
    median_relative_gain: float | None
    local_gain_standard_error: float | None
    external_prior: ExternalModelPrior | None = None
    diagnostics: OutputDiagnostics = field(default_factory=OutputDiagnostics)
    conflicts: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdmissionDecision:
    state: AdmissionState
    candidate: str
    baseline: str
    point_policy: PointPolicy
    candidate_weight: float
    evidence: AdmissionEvidence
    reasons: tuple[str, ...]
    policy_version: str = "evidence-weighted-v1"

    def to_payload(self, *, compact: bool = False) -> dict[str, object]:
        payload = asdict(self)
        if compact:
            prior = payload.get("evidence", {}).get("external_prior")  # type: ignore[union-attr]
            if isinstance(prior, dict):
                sources = prior.pop("source_ids", [])
                prior["source_count"] = len(sources)
        payload["reasoning_frame"] = reasoning_frame(self)
        return payload


def reasoning_frame(decision: AdmissionDecision) -> dict[str, object]:
    """Compact deterministic frame; it organises evidence, not prose.

    The LLM may explain this projection but cannot replace its conclusion.
    Broad tiers avoid presenting a hand-designed scalar as measured truth.
    """
    evidence = decision.evidence
    local = (
        "repeatable" if evidence.independent_folds >= 2
        else "indicative" if evidence.independent_folds == 1 else "absent"
    )
    prior = evidence.external_prior
    external = (
        "strong" if prior and prior.usable()
        and _normal_probability_positive(
            prior.mean_relative_gain, prior.standard_error) >= .95
        else "available" if prior and prior.usable() else "absent"
    )
    conflict_status = "material" if evidence.conflicts else "none_observed"
    if not evidence.diagnostics.valid:
        sufficiency = "insufficient"
    elif decision.state in {
            "locally_validated", "externally_validated", "jointly_validated"}:
        sufficiency = "sufficient"
    elif decision.state == "prior_assisted":
        sufficiency = "mixed"
    else:
        sufficiency = "insufficient"
    return {
        "observation": {
            "candidate_loss": evidence.candidate_loss,
            "baseline_loss": evidence.baseline_loss,
            "relative_improvement": evidence.relative_improvement,
            "candidate_baseline_disagreement":
                evidence.diagnostics.candidate_baseline_disagreement,
        },
        "temporal_property": "forecast_candidate_transfer",
        "competing_interpretations": [
            {"interpretation": "candidate generalises",
             "evidence": {"local": local, "external": external}},
            {"interpretation": "robust baseline remains safer",
             "evidence": {"conflict": conflict_status,
                          "output_valid": evidence.diagnostics.valid}},
        ],
        "selected_conclusion": {
            "state": decision.state,
            "point_policy": decision.point_policy,
            "candidate_weight": decision.candidate_weight,
        },
        "uncertainty": {
            "sufficiency": sufficiency,
            "local_validation": local,
            "external_evidence": external,
            "conflict": conflict_status,
            "independent_folds": evidence.independent_folds,
        },
    }


def local_evidence(
    *, model_class: ModelClass, candidate_losses: list[float | None],
    baseline_losses: list[float | None], external_prior: ExternalModelPrior | None = None,
    diagnostics: OutputDiagnostics | None = None,
) -> AdmissionEvidence:
    """Build aligned local evidence without treating overlapping origins as
    independent or substituting missing folds with zero loss."""
    paired = [
        (float(base), float(candidate))
        for base, candidate in zip(baseline_losses, candidate_losses)
        if base is not None and candidate is not None
        and isfinite(float(base)) and isfinite(float(candidate)) and base > 0
    ]
    gains = [(base - candidate) / base for base, candidate in paired]
    gain_se = None
    if len(gains) >= 2:
        gain_se = stdev(gains) / sqrt(len(gains))
        # A perfectly repeated difference is evidence too; retain a finite
        # numerical floor rather than turning infinite precision into fact.
        gain_se = max(gain_se, 1e-6)
    return AdmissionEvidence(
        model_class=model_class,
        independent_folds=len(paired), paired_folds=len(paired),
        candidate_loss=mean(candidate for _, candidate in paired) if paired else None,
        baseline_loss=mean(base for base, _ in paired) if paired else None,
        relative_improvement=mean(gains) if gains else None,
        candidate_win_rate=(sum(candidate < base for base, candidate in paired) / len(paired)
                            if paired else None),
        median_relative_gain=median(gains) if gains else None,
        local_gain_standard_error=gain_se,
        external_prior=external_prior,
        diagnostics=diagnostics or OutputDiagnostics(),
    )


def _normal_probability_positive(mu: float, standard_error: float) -> float:
    return .5 * (1 + erf(mu / (standard_error * sqrt(2))))


def _combined_gain(evidence: AdmissionEvidence) -> tuple[float, float, str] | None:
    """Precision-pool independent external and repeatable local evidence."""
    estimates: list[tuple[float, float, str]] = []
    prior = evidence.external_prior
    if prior is not None and prior.usable():
        estimates.append((prior.mean_relative_gain, prior.standard_error, "external"))
    if (evidence.relative_improvement is not None
            and evidence.local_gain_standard_error is not None):
        estimates.append((evidence.relative_improvement,
                          evidence.local_gain_standard_error, "local"))
    if not estimates:
        return None
    precision = sum(1 / (se * se) for _, se, _ in estimates)
    mu = sum(value / (se * se) for value, se, _ in estimates) / precision
    se = sqrt(1 / precision)
    source = "joint" if len(estimates) == 2 else estimates[0][2]
    return mu, se, source


def decide_admission(
    *, candidate: str, baseline: str, evidence: AdmissionEvidence,
    minimum_local_improvement: float = .02,
) -> AdmissionDecision:
    """Choose a provenance-preserving point policy.

    The only continuous mapping is monotone and interpretable: candidate
    weight is zero at a 50% probability of superiority and one at certainty.
    No domain, channel, dataset, or model-name rules participate.
    """
    if not evidence.diagnostics.valid:
        return AdmissionDecision(
            "baseline_fallback", candidate, baseline, "baseline", 0.0,
            evidence, ("candidate output failed deterministic validity checks",),
        )

    local_valid = (
        evidence.independent_folds >= 2
        and evidence.relative_improvement is not None
        and evidence.relative_improvement >= minimum_local_improvement
        and (evidence.candidate_win_rate or 0) > .5
        and (evidence.median_relative_gain or 0) > 0
    )
    combined = _combined_gain(evidence)
    external_valid = bool(
        evidence.external_prior is not None
        and evidence.external_prior.usable()
        and evidence.external_prior.mean_relative_gain > 0
        and _normal_probability_positive(
            evidence.external_prior.mean_relative_gain,
            evidence.external_prior.standard_error,
        ) >= .95
    )

    if local_valid and external_valid:
        state: AdmissionState = "jointly_validated"
    elif local_valid:
        state = "locally_validated"
    elif external_valid and evidence.independent_folds == 0:
        state = "externally_validated"
    elif combined is not None:
        state = "prior_assisted"
    else:
        return AdmissionDecision(
            "baseline_fallback", candidate, baseline, "baseline", 0.0,
            evidence, ("no repeatable local or admissible external evidence",),
        )

    if combined is None:
        # Locally validated with no estimable local SE: repeated win/stability
        # gates earned selection through the existing non-parametric path.
        weight = 1.0
        probability = None
    else:
        mu, se, _ = combined
        probability = _normal_probability_positive(mu, se)
        weight = max(0.0, min(1.0, 2 * probability - 1))

    if state in {"locally_validated", "jointly_validated"}:
        point_policy: PointPolicy = "candidate"
        weight = 1.0
    elif state == "externally_validated" and weight >= .9:
        point_policy = "candidate"
        weight = 1.0
    elif weight > 0:
        point_policy = "shrunk_blend"
    else:
        state = "baseline_fallback"
        point_policy = "baseline"

    reasons = [f"evidence state: {state}"]
    if probability is not None:
        reasons.append(f"estimated probability candidate beats baseline: {probability:.3f}")
    if evidence.conflicts:
        reasons.append("unresolved evidence conflicts are disclosed")
    return AdmissionDecision(
        state, candidate, baseline, point_policy, weight, evidence,
        tuple(reasons),
    )

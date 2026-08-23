"""Shared probability-to-decision contract for temporal properties.

Estimators own continuous quantities and calibrated distributions.  Policies
own the point at which those distributions become categorical claims.  Keeping
the seam explicit prevents a convenient label from masquerading as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


DECISION_POLICY_PROFILES: dict[str, dict[str, float | int]] = {
    "conservative": {"minimum_probability": .85, "minimum_folds": 12,
                     "minimum_balanced_accuracy": .75, "minimum_brier_skill": .04},
    "standard": {"minimum_probability": .8, "minimum_folds": 8,
                 "minimum_balanced_accuracy": .70, "minimum_brier_skill": .02},
    "exploratory": {"minimum_probability": .7, "minimum_folds": 5,
                    "minimum_balanced_accuracy": .60, "minimum_brier_skill": 0.0},
}


@dataclass(frozen=True)
class TemporalDecisionPolicy:
    """Evidence required to turn a best estimate into an actionable claim."""

    minimum_probability: float = .8
    minimum_folds: int = 8
    minimum_balanced_accuracy: float = .70
    minimum_brier_skill: float = .02


def bounded_decision_policy(raw: Mapping[str, Any] | str | None
                            ) -> TemporalDecisionPolicy:
    """Resolve a public policy without permitting evidence-free automation."""
    if raw is None:
        raw = "standard"
    if isinstance(raw, str):
        if raw not in DECISION_POLICY_PROFILES:
            raise ValueError("unknown decision-policy profile")
        values = DECISION_POLICY_PROFILES[raw]
    elif isinstance(raw, Mapping):
        unknown = set(raw) - set(TemporalDecisionPolicy.__dataclass_fields__)
        if unknown:
            raise ValueError("unknown decision-policy fields: " + ", ".join(sorted(unknown)))
        values = {**DECISION_POLICY_PROFILES["standard"], **dict(raw)}
    else:
        raise ValueError("decision_policy must be a profile name or object")
    policy = TemporalDecisionPolicy(**values)
    if not (.7 <= policy.minimum_probability <= .99):
        raise ValueError("minimum_probability must be between 0.70 and 0.99")
    if not (5 <= policy.minimum_folds <= 1000):
        raise ValueError("minimum_folds must be between 5 and 1000")
    if not (.60 <= policy.minimum_balanced_accuracy <= .99):
        raise ValueError("minimum_balanced_accuracy must be between 0.60 and 0.99")
    if not (0 <= policy.minimum_brier_skill <= .5):
        raise ValueError("minimum_brier_skill must be between 0 and 0.50")
    return policy


@dataclass(frozen=True)
class TemporalPropertyDistribution:
    """A fitted continuous estimate plus probabilities over public states."""

    quantity: str
    estimate: float
    lower: float
    upper: float
    probabilities: Mapping[str, float]
    point_state: str
    folds: int
    balanced_accuracy: float
    brier_skill: float
    support: str

    def __post_init__(self) -> None:
        values = list(self.probabilities.values())
        if not values or any(not math.isfinite(float(value)) or value < 0
                             for value in values):
            raise ValueError("property probabilities must be finite and non-negative")
        if abs(sum(values) - 1.0) > 1e-8:
            raise ValueError("property probabilities must sum to one")
        if self.point_state not in self.probabilities:
            raise ValueError("point_state must be one of the probability states")
        if self.lower > self.upper:
            raise ValueError("property interval lower bound exceeds upper bound")

    def decide(self, policy: TemporalDecisionPolicy) -> dict[str, object]:
        best = max(
            self.probabilities,
            key=lambda label: (self.probabilities[label], label == self.point_state),
        )
        probability = float(self.probabilities[best])
        eligible = (
            self.support == "supported"
            and self.folds >= policy.minimum_folds
            and self.balanced_accuracy >= policy.minimum_balanced_accuracy
            and self.brier_skill >= policy.minimum_brier_skill
            and probability >= policy.minimum_probability
        )
        # Unsupported probability modes remain diagnostic. A weak model may
        # have a noisy modal class even when its continuous point estimate is
        # persistence; publishing that mode as a claim recreates majority-
        # class guessing under a probabilistic name.
        published = best if eligible else self.point_state
        return {
            "best_state": published,
            "probability_mode": best,
            "probability": probability,
            "support": "supported" if eligible else
                       "weak" if self.support != "abstained" else "abstained",
            "automation_eligible": eligible,
            "policy": {
                "minimum_probability": policy.minimum_probability,
                "minimum_folds": policy.minimum_folds,
                "minimum_balanced_accuracy": policy.minimum_balanced_accuracy,
                "minimum_brier_skill": policy.minimum_brier_skill,
            },
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "quantity": self.quantity,
            "estimate": self.estimate,
            "interval": {"lower": self.lower, "upper": self.upper},
            "probabilities": dict(self.probabilities),
            "point_state": self.point_state,
            "folds": self.folds,
            "balanced_accuracy": self.balanced_accuracy,
            "brier_skill": self.brier_skill,
            "support": self.support,
        }

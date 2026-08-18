"""Shared probability-to-decision contract for temporal properties.

Estimators own continuous quantities and calibrated distributions.  Policies
own the point at which those distributions become categorical claims.  Keeping
the seam explicit prevents a convenient label from masquerading as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class TemporalDecisionPolicy:
    """Evidence required to turn a best estimate into an actionable claim."""

    minimum_probability: float = .8
    minimum_folds: int = 8
    minimum_balanced_accuracy: float = .70
    minimum_brier_skill: float = .02


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

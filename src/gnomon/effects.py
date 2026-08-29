"""Typed contracts for context-conditioned effects.

An effect is not a scalar annotation on the primary forecast.  It is a
distribution, attached to a separately identified scenario, with provenance
that says why Gnomon was allowed to use it.  Keeping this contract in one
module lets the forecasting and tracking paths share the same validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import math
from typing import Any, Literal

EffectProvenanceClass = Literal[
    "same_event_same_series",
    "same_event_related_series",
    "organization_analogue",
    "external_prior",
    "human_assumption",
    "standardized_sensitivity",
]

EffectShape = Literal[
    "temporary_pulse", "level_shift", "trend_change", "variance_change",
    "ramp_recovery", "seasonal_amplitude", "seasonal_phase",
    "cross_series_relationship", "saturation_bound", "custom_scenario",
    "seasonal_regime_change", "unknown",
]

EFFECT_SHAPES = frozenset({
    "temporary_pulse", "level_shift", "trend_change", "variance_change",
    "ramp_recovery", "seasonal_amplitude", "seasonal_phase",
    "cross_series_relationship", "saturation_bound", "custom_scenario",
    "seasonal_regime_change", "unknown",
})

_PROVENANCE_CLASSES = {
    "same_event_same_series", "same_event_related_series",
    "organization_analogue", "external_prior", "human_assumption",
    "standardized_sensitivity",
}


def latest_knowledge_time(cutoff: datetime, known_times: list[str]) -> str:
    """Latest required input time, retaining explicit timezone provenance."""
    parsed = [datetime.fromisoformat(value) for value in known_times]
    if not parsed:
        if cutoff.tzinfo is None:
            raise ValueError("naive cutoff requires an explicit context known_at")
        return cutoff.isoformat()
    if any(value.tzinfo is None for value in parsed):
        raise ValueError("context known_at requires an explicit timezone offset")
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=parsed[0].tzinfo)
    return max([cutoff, *parsed]).isoformat()


@dataclass(frozen=True)
class EffectDistribution:
    """A probability distribution over an effect in target-series units.

    ``distribution`` is deliberately a small closed vocabulary.  Phase 1
    emits normal estimates and deterministic sensitivity assumptions; future
    registry pooling can add samples without changing the scenario contract.
    """

    distribution: Literal["normal", "point_mass", "empirical", "assumption"]
    location: float
    scale: float
    lower: float
    upper: float
    interval_probability: float | None
    sample_count: int
    unit: str = "target_series_units"

    def __post_init__(self) -> None:
        numeric = (self.location, self.scale, self.lower, self.upper)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("effect distribution values must be finite")
        if self.scale < 0 or self.lower > self.location or self.location > self.upper:
            raise ValueError("effect distribution bounds/scale are inconsistent")
        if (self.interval_probability is not None
                and not 0 < self.interval_probability <= 1):
            raise ValueError("interval_probability must be in (0, 1]")
        if self.distribution == "assumption" and self.interval_probability is not None:
            raise ValueError("an assumption cannot claim probability coverage")
        if self.sample_count < 0:
            raise ValueError("sample_count cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EffectProvenance:
    """Why an effect magnitude is present and when it became knowable."""

    provenance_class: EffectProvenanceClass
    observed: bool
    known_at: str
    source_reference: str | None
    similarity: float | None
    reliability: float | None
    method: str

    def __post_init__(self) -> None:
        if self.provenance_class not in _PROVENANCE_CLASSES:
            raise ValueError(f"unknown effect provenance: {self.provenance_class}")
        for name, value in (("similarity", self.similarity),
                            ("reliability", self.reliability)):
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not self.known_at:
            raise ValueError("effect provenance requires known_at")
        try:
            known = datetime.fromisoformat(self.known_at)
        except ValueError as exc:
            raise ValueError("effect provenance known_at must be ISO-8601") from exc
        if known.tzinfo is None:
            raise ValueError("effect provenance known_at requires a timezone offset")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def effect_contract(
    distribution: EffectDistribution,
    provenance: EffectProvenance,
    *,
    shape: str,
) -> dict[str, Any]:
    """Canonical public representation shared by all scenario lanes."""
    return {
        "distribution": distribution.to_dict(),
        "provenance": provenance.to_dict(),
        "shape": shape if shape in EFFECT_SHAPES else "unknown",
    }

"""Versioned external effect evidence and conservative resolution.

Registry entries are evidence supplied by a governed data source, never facts
invented by the context compiler.  Resolution is deterministic and produces a
scenario prior only.  A registry match cannot alter Gnomon's primary forecast;
local, fold-safe evidence must still earn that through the existing context
admission gate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REGISTRY_SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class EffectPrior:
    prior_id: str
    event_type: str
    target: str
    domain: str
    population: str
    unit: str
    effect_family: str
    mean: float
    standard_error: float
    delay_steps: tuple[int, int]
    duration_steps: tuple[int, int]
    sample_size: int
    source: str
    version: str
    expires_at: str | None = None
    known_at: str | None = None

    def precision(self) -> float:
        return 1.0 / (self.standard_error * self.standard_error)


def _pair(raw: Any, name: str) -> tuple[int, int]:
    if not isinstance(raw, list) or len(raw) != 2 \
            or any(isinstance(item, bool) or not isinstance(item, int)
                   for item in raw):
        raise ValueError(f"{name} must be a two-integer [minimum, maximum]")
    low, high = int(raw[0]), int(raw[1])
    if low < 0 or high < low:
        raise ValueError(f"{name} must satisfy 0 <= minimum <= maximum")
    return low, high


def prior_from_dict(raw: dict[str, Any]) -> EffectPrior:
    required = {
        "prior_id", "event_type", "target", "domain", "population", "unit",
        "effect_family", "mean", "standard_error", "delay_steps",
        "duration_steps", "sample_size", "source", "version",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"effect prior is missing {missing}")
    mean = float(raw["mean"])
    standard_error = float(raw["standard_error"])
    sample_size = int(raw["sample_size"])
    if not math.isfinite(mean) or not math.isfinite(standard_error) \
            or standard_error <= 0:
        raise ValueError("effect prior mean must be finite and standard_error positive")
    if sample_size < 2:
        raise ValueError("effect prior sample_size must be at least 2")
    source = str(raw["source"]).strip()
    version = str(raw["version"]).strip()
    if not source or not version:
        raise ValueError("effect prior source and version must be non-empty")
    return EffectPrior(
        prior_id=str(raw["prior_id"]), event_type=str(raw["event_type"]),
        target=str(raw["target"]), domain=str(raw["domain"]),
        population=str(raw["population"]), unit=str(raw["unit"]),
        effect_family=str(raw["effect_family"]), mean=mean,
        standard_error=standard_error,
        delay_steps=_pair(raw["delay_steps"], "delay_steps"),
        duration_steps=_pair(raw["duration_steps"], "duration_steps"),
        sample_size=sample_size, source=source, version=version,
        expires_at=(str(raw["expires_at"]) if raw.get("expires_at") else None),
        known_at=(str(raw["known_at"]) if raw.get("known_at") else None),
    )


def load_effect_registry(path: str | Path, *, now: datetime | None = None) -> list[EffectPrior]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ValueError("unsupported effect registry schema_version")
    entries = [prior_from_dict(item) for item in payload.get("priors", [])]
    identifiers = [item.prior_id for item in entries]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("effect registry prior_id values must be unique")
    moment = now or datetime.now(timezone.utc)
    for entry in entries:
        if entry.known_at:
            known = datetime.fromisoformat(entry.known_at)
            if known.tzinfo is None:
                raise ValueError(f"effect prior {entry.prior_id!r} known_at needs a timezone")
        if entry.expires_at:
            expiry = datetime.fromisoformat(entry.expires_at)
            if expiry.tzinfo is None:
                raise ValueError(f"effect prior {entry.prior_id!r} expiry needs a timezone")
            if expiry <= moment:
                raise ValueError(f"effect prior {entry.prior_id!r} is expired")
    return entries


def resolve_effect_prior(
    priors: list[EffectPrior], *, event_type: str, target: str,
    domain: str = "*", population: str = "*", unit: str = "*",
) -> dict[str, Any] | None:
    """Resolve and precision-pool only equally specific compatible priors."""
    candidates = []
    for prior in priors:
        if prior.event_type not in (event_type, "*"):
            continue
        if prior.target not in (target, "*"):
            continue
        dimensions = ((prior.domain, domain), (prior.population, population),
                      (prior.unit, unit))
        if any(wanted != "*" and actual not in (wanted, "*")
               for actual, wanted in dimensions):
            continue
        specificity = sum(value != "*" for value in (
            prior.event_type, prior.target, prior.domain,
            prior.population, prior.unit,
        ))
        candidates.append((specificity, prior))
    if not candidates:
        return None
    best = max(score for score, _ in candidates)
    selected = [prior for score, prior in candidates if score == best]
    families = {prior.effect_family for prior in selected}
    if len(families) != 1:
        return {
            "status": "conflict", "may_affect_primary_forecast": False,
            "prior_ids": sorted(prior.prior_id for prior in selected),
            "reason": "equally specific priors disagree on effect_family",
        }
    total_precision = sum(prior.precision() for prior in selected)
    pooled_mean = sum(prior.mean * prior.precision() for prior in selected) / total_precision
    return {
        "status": "scenario_prior", "may_affect_primary_forecast": False,
        "effect_family": next(iter(families)), "mean": pooled_mean,
        "standard_error": (1.0 / total_precision) ** 0.5,
        "delay_steps": [min(p.delay_steps[0] for p in selected),
                        max(p.delay_steps[1] for p in selected)],
        "duration_steps": [min(p.duration_steps[0] for p in selected),
                           max(p.duration_steps[1] for p in selected)],
        "prior_ids": sorted(prior.prior_id for prior in selected),
        "sources": sorted({f"{prior.source}@{prior.version}" for prior in selected}),
        "sample_size": sum(prior.sample_size for prior in selected),
        "resolution_specificity": best,
        "basis": "external governed evidence; scenario-only until local admission",
    }

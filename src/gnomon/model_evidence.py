"""Versioned, read-only external evidence registry for candidate admission."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .admission import ExternalModelPrior


class EvidenceRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class TemporalRegime:
    """Low-cardinality, model-agnostic descriptors known at forecast time."""

    frequency_class: str
    history_horizon_ratio: str
    seasonality_strength: str
    trend_strength: str
    intermittency: str
    scale_stability: str

    def items(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self.__dict__.items()))


def describe_regime(values: list[float], horizon: int, season: int,
                    frequency: str) -> TemporalRegime:
    """Version-one broad regime projection using only pre-cutoff values.

    These intentionally coarse bins prevent pseudo-precision. They are
    versioned indirectly by the registry snapshot and never use domain,
    channel, dataset, or benchmark identities.
    """
    if not values or horizon <= 0:
        raise EvidenceRegistryError("regime description requires values and horizon")
    n = len(values)
    ratio = n / horizon
    history_horizon_ratio = (
        "lt_1" if ratio < 1 else "1_to_2" if ratio < 2
        else "2_to_4" if ratio < 4 else "gte_4"
    )
    frequency_class = (
        "subdaily" if frequency.lower() in {"s", "min", "h"}
        else "daily_weekly" if frequency.lower() in {"d", "w"}
        else "monthly_or_slower"
    )
    diffs = [right - left for left, right in zip(values, values[1:])]
    level_scale = max(max(values) - min(values), 1e-12)
    trend = abs((values[-1] - values[0]) / max(n - 1, 1)) / level_scale
    trend_strength = "low" if trend < .005 else "moderate" if trend < .02 else "high"
    zeros = sum(abs(value) <= 1e-12 for value in values) / n
    intermittency = "high" if zeros >= .4 else "moderate" if zeros >= .1 else "low"
    if len(diffs) < 4:
        scale_stability = "unknown"
    else:
        midpoint = len(diffs) // 2
        left = sum(abs(item) for item in diffs[:midpoint]) / max(midpoint, 1)
        right = sum(abs(item) for item in diffs[midpoint:]) / max(len(diffs) - midpoint, 1)
        ratio_scale = max(left, right) / max(min(left, right), 1e-12)
        scale_stability = "stable" if ratio_scale < 2 else "changing"
    if season <= 1 or n < 2 * season:
        seasonality_strength = "unknown"
    else:
        seasonal_error = sum(abs(values[index] - values[index - season])
                             for index in range(season, n)) / (n - season)
        naive_error = sum(abs(item) for item in diffs) / max(len(diffs), 1)
        relative = seasonal_error / max(naive_error, 1e-12)
        seasonality_strength = (
            "high" if relative < .6 else "moderate" if relative < 1 else "low")
    return TemporalRegime(
        frequency_class, history_horizon_ratio, seasonality_strength,
        trend_strength, intermittency, scale_stability,
    )


def _expect(value: Any, kind: type, field: str) -> Any:
    if not isinstance(value, kind):
        raise EvidenceRegistryError(f"external evidence {field} must be {kind.__name__}")
    return value


class ModelEvidenceRegistry:
    """An immutable snapshot; lookup never learns from the query being served."""

    def __init__(self, version: str, priors: tuple[ExternalModelPrior, ...],
                 *, source_path: str | None = None):
        if not version:
            raise EvidenceRegistryError("external evidence registry requires a version")
        self.version = version
        self.priors = priors
        self.source_path = source_path

    @classmethod
    def load(cls, path: str | Path) -> "ModelEvidenceRegistry":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvidenceRegistryError(f"cannot load external evidence registry: {exc}") from exc
        if not isinstance(payload, dict):
            raise EvidenceRegistryError("external evidence registry must be an object")
        version = _expect(payload.get("version"), str, "version")
        rows = _expect(payload.get("priors"), list, "priors")
        priors = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise EvidenceRegistryError(f"external evidence priors[{index}] must be an object")
            regime = _expect(row.get("regime"), dict, f"priors[{index}].regime")
            source_ids = _expect(row.get("source_ids"), list, f"priors[{index}].source_ids")
            priors.append(ExternalModelPrior(
                model=str(row.get("model", "")), revision=str(row.get("revision", "")),
                regime=tuple(sorted((str(k), str(v)) for k, v in regime.items())),
                comparisons=int(row.get("comparisons", 0)),
                mean_relative_gain=float(row.get("mean_relative_gain")),
                standard_error=float(row.get("standard_error")),
                source_ids=tuple(str(item) for item in source_ids),
                registry_version=version,
                overlap_risk=str(row.get("overlap_risk", "unknown")),  # type: ignore[arg-type]
                baseline_reference=str(row.get(
                    "baseline_reference", "strongest_robust_baseline")),
            ))
        return cls(version, tuple(priors), source_path=str(source))

    def dump(self, path: str | Path) -> None:
        """Write the immutable interchange form consumed by Gnomon."""
        target = Path(path)
        payload = {
            "version": self.version,
            "priors": [{
                "model": prior.model, "revision": prior.revision,
                "regime": dict(prior.regime),
                "comparisons": prior.comparisons,
                "mean_relative_gain": prior.mean_relative_gain,
                "standard_error": prior.standard_error,
                "source_ids": list(prior.source_ids),
                "overlap_risk": prior.overlap_risk,
                "baseline_reference": prior.baseline_reference,
            } for prior in self.priors],
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")

    def lookup(self, model: str, revision: str,
               regime: TemporalRegime) -> ExternalModelPrior | None:
        """Most-specific declared regime/revision match.

        Registry authors may explicitly write ``"*"`` for dimensions on
        which their held-out evidence is genuinely broad. There is no nearest
        neighbour inference: fields either match exactly or were declared
        global, and ties at the same specificity are rejected.
        """
        key = regime.items()
        requested = dict(key)
        matches: list[tuple[int, ExternalModelPrior]] = []
        for prior in self.priors:
            if prior.model != model or prior.revision != revision:
                continue
            declared = dict(prior.regime)
            if set(declared) != set(requested):
                continue
            if all(value == "*" or value == requested[field]
                   for field, value in declared.items()):
                matches.append((sum(value != "*" for value in declared.values()), prior))
        if not matches:
            return None
        specificity = max(item[0] for item in matches)
        best = [prior for score, prior in matches if score == specificity]
        if len(best) > 1:
            raise EvidenceRegistryError(
                f"duplicate external evidence for {model}@{revision} and regime")
        return best[0]

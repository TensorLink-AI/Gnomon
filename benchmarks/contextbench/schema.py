"""Strict, dependency-free ContextBench contracts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1
FAMILIES = {"irrelevant", "future_covariate", "repeated_event", "prior_only"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Case:
    case_id: str
    family: str
    domain: str
    frequency: str
    horizon: int
    history: tuple[float, ...]
    context_events: tuple[dict[str, Any], ...]
    covariates: tuple[dict[str, Any], ...]
    covariate_mapping: tuple[dict[str, str], ...]
    narrative: str
    tags: tuple[str, ...]
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Case":
        allowed = {"schema_version", "case_id", "family", "domain", "frequency",
                   "horizon", "history", "context_events", "covariates",
                   "covariate_mapping", "narrative", "tags"}
        unknown = sorted(set(raw) - allowed)
        _require(not unknown, f"case has unknown fields: {unknown}")
        _require(int(raw.get("schema_version", 1)) == SCHEMA_VERSION,
                 "unsupported ContextBench schema version")
        case_id = str(raw.get("case_id", "")).strip()
        family = str(raw.get("family", ""))
        history = tuple(float(value) for value in raw.get("history", ()))
        horizon = int(raw.get("horizon", 0))
        _require(bool(case_id), "case_id is required")
        _require(family in FAMILIES, f"unknown family {family!r}")
        _require(horizon > 0 and len(history) >= 8 * horizon,
                 f"case {case_id}: history must provide at least 8 horizons")
        _require(all(math.isfinite(value) for value in history),
                 f"case {case_id}: history must be finite")
        events = tuple(dict(item) for item in raw.get("context_events", ()))
        covariates = tuple(dict(item) for item in raw.get("covariates", ()))
        mapping = tuple(dict(item) for item in raw.get("covariate_mapping", ()))
        _require(not (events and covariates),
                 f"case {case_id}: one context lane per case keeps attribution identifiable")
        if family == "future_covariate":
            _require(bool(covariates and mapping),
                     f"case {case_id}: future_covariate needs rows and mapping")
        return cls(
            case_id=case_id, family=family,
            domain=str(raw.get("domain", "operations")),
            frequency=str(raw.get("frequency", "h")), horizon=horizon,
            history=history, context_events=events, covariates=covariates,
            covariate_mapping=mapping, narrative=str(raw.get("narrative", "")),
            tags=tuple(str(item) for item in raw.get("tags", ())),
        )


@dataclass(frozen=True)
class Oracle:
    case_id: str
    actual: tuple[float, ...]
    counterfactual: tuple[float, ...]
    should_influence: bool
    expected_disposition: str
    effect_direction: str
    effect_magnitude: float
    onset_step: int | None
    duration_steps: int | None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Oracle":
        actual = tuple(float(item) for item in raw["actual"])
        counterfactual = tuple(float(item) for item in raw["counterfactual"])
        _require(len(actual) == len(counterfactual) > 0,
                 "oracle actual and counterfactual must align")
        return cls(
            case_id=str(raw["case_id"]), actual=actual,
            counterfactual=counterfactual,
            should_influence=bool(raw["should_influence"]),
            expected_disposition=str(raw["expected_disposition"]),
            effect_direction=str(raw.get("effect_direction", "none")),
            effect_magnitude=float(raw.get("effect_magnitude", 0.0)),
            onset_step=(int(raw["onset_step"]) if raw.get("onset_step") is not None else None),
            duration_steps=(int(raw["duration_steps"])
                            if raw.get("duration_steps") is not None else None),
        )


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_cases(path: str | Path) -> list[Case]:
    rows = [Case.from_dict(raw) for raw in read_jsonl(path)]
    _require(len({row.case_id for row in rows}) == len(rows), "duplicate case_id")
    return rows


def load_oracles(path: str | Path) -> dict[str, Oracle]:
    rows = [Oracle.from_dict(raw) for raw in read_jsonl(path)]
    _require(len({row.case_id for row in rows}) == len(rows), "duplicate oracle case_id")
    return {row.case_id: row for row in rows}

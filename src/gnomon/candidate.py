"""Executable candidates: the object that earned publication publishes.

The evaluation→publication seam used to pass a *name* and a score table,
and the publication side re-derived the forecast — with a hardcoded
strategy, over the unrestricted built-in pool, without the config. The
evaluated ensemble and the published ensemble were different objects
(unified plan, Phase 1A).

The contract here makes that class unconstructible where it applies:
``evaluate`` builds a :class:`CandidateSpec` for the winning candidate —
the *same* closures that produced its calibration and test predictions —
and ``predict_stage`` publishes by fitting that specification on the full
visible history. One immutable specification and fitting procedure, an
independent fitted instance at every origin: reusing one fitted object
across selection, calibration, and test would leak information between
partitions, so the spec re-fits, it never replays.

Identity travels with the points: kind, name, member set, combination
strategy, and the behaviour-affecting configuration are recorded in the
run's evidence, so an artifact can say exactly which executable produced
its numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class CandidateIdentity:
    """What, exactly, is publishing.

    ``members`` is the evaluated member set for a combined candidate
    (empty for a single model); ``config`` carries only the options that
    change the published numbers.
    """

    kind: str                     # "builtin" | "ensemble" | "cross_series"
    name: str
    members: tuple[str, ...] = ()
    strategy: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind, "name": self.name}
        if self.members:
            payload["members"] = list(self.members)
        if self.strategy:
            payload["strategy"] = self.strategy
        if self.config:
            payload["config"] = dict(self.config)
        return payload


class FittedCandidate:
    """One fitted instance: a specification bound to one history."""

    def __init__(self, identity: CandidateIdentity,
                 predictor: Callable[[int], list[float]]):
        self.identity = identity
        self._predictor = predictor

    def predict(self, horizon: int) -> list[float]:
        return self._predictor(horizon)


class CandidateSpec:
    """An immutable specification with a fitting procedure.

    ``fit(history, season)`` returns an independent
    :class:`FittedCandidate`; the fit closures are supplied by
    ``evaluate`` and are the same code path its calibration and test
    folds used, which is the entire point.
    """

    def __init__(self, identity: CandidateIdentity,
                 fit: Callable[[list[float], int | None], FittedCandidate]):
        self.identity = identity
        self._fit = fit

    def fit(self, history: list[float], season: int | None) -> FittedCandidate:
        return self._fit(history, season)

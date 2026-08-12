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

from dataclasses import dataclass, field, replace
from typing import Any, Callable


@dataclass(frozen=True)
class CandidateIdentity:
    """What, exactly, is publishing.

    The fields are the plan's list: strategy, exact member set, fitted
    weights where applicable, behaviour-changing configuration,
    dependency and weight revisions, fallback policy, and the
    visible-data fingerprint.

    Spec-time fields (everything except ``weights`` and
    ``data_fingerprint``) are fixed when evaluation names the winner;
    the two fit-time fields are filled by :meth:`CandidateSpec.fit`,
    because they describe a particular fit rather than the
    specification — which is exactly why a fitted object may not be
    reused across partitions.
    """

    kind: str                     # "builtin" | "ensemble" | "cross_series"
    name: str
    members: tuple[str, ...] = ()
    strategy: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    #: Implementation and third-party weight revisions. ``runtime`` is
    #: always present; TSFM members contribute their pinned revisions.
    revisions: dict[str, str] = field(default_factory=dict)
    #: What this candidate does when it cannot produce a final
    #: prediction. Part of the object, not an improvisation at the call
    #: site: the artifact says which policy was in force.
    fallback_policy: str | None = None
    #: Fitted member weights, for combiners that have them.
    weights: dict[str, float] | None = None
    #: Content fingerprint of the history this instance was fit on.
    data_fingerprint: str | None = None

    def with_fit(self, *, weights: dict[str, float] | None,
                 data_fingerprint: str) -> "CandidateIdentity":
        return replace(self, weights=weights,
                       data_fingerprint=data_fingerprint)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind, "name": self.name}
        if self.members:
            payload["members"] = list(self.members)
        if self.strategy:
            payload["strategy"] = self.strategy
        if self.config:
            payload["config"] = dict(self.config)
        if self.weights:
            payload["weights"] = {name: round(value, 12)
                                  for name, value in sorted(self.weights.items())}
        if self.revisions:
            payload["revisions"] = dict(sorted(self.revisions.items()))
        if self.fallback_policy:
            payload["fallback_policy"] = self.fallback_policy
        if self.data_fingerprint:
            payload["data_fingerprint"] = self.data_fingerprint
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

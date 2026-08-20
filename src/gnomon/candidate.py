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
                 fit: Callable[[list[float], int | None], FittedCandidate],
                 *, min_history: int | None = None,
                 max_horizon: int | None = None,
                 predict_many: Callable[[list[list[float]], int, int | None],
                                        list[list[float]]] | None = None):
        self.identity = identity
        self._fit = fit
        self.min_history = min_history
        self.max_horizon = max_horizon
        self._predict_many = predict_many

    def supports(self, history_length: int, horizon: int) -> bool:
        """Whether this executable can fairly participate at an origin.

        Eligibility belongs to the executable, rather than to callers that
        happen to know which model family it came from.  Context, covariate,
        and future ablation lanes can therefore replay built-ins, local
        models, sandbox models, and remote APIs through one contract.
        """
        return not (
            (self.min_history is not None and history_length < self.min_history)
            or (self.max_horizon is not None and horizon > self.max_horizon)
        )

    def fit(self, history: list[float], season: int | None) -> FittedCandidate:
        if not self.supports(len(history), 1):
            raise ValueError(
                f"candidate {self.identity.name} does not support history "
                f"length {len(history)}"
            )
        return self._fit(history, season)

    def predict_many(self, histories: list[list[float]], horizon: int,
                     season: int | None) -> list[list[float]]:
        """Replay several origins, using a backend batch path when available."""
        if any(not self.supports(len(history), horizon)
               for history in histories):
            raise ValueError(
                f"candidate {self.identity.name} does not support every "
                "requested history/horizon pair"
            )
        if self._predict_many is not None:
            points = self._predict_many(histories, horizon, season)
            if len(points) != len(histories):
                raise ValueError("candidate batch cardinality mismatch")
            return points
        return [self.fit(history, season).predict(horizon)
                for history in histories]


def candidate_predict(evaluation: Any, history: list[float], horizon: int,
                      season: int | None) -> list[float]:
    """Predict through the evaluated executable, with legacy compatibility."""
    spec = getattr(evaluation, "final_candidate", None)
    if spec is not None:
        if not spec.supports(len(history), horizon):
            raise ValueError(
                f"candidate {spec.identity.name} does not support history "
                f"length {len(history)} and horizon {horizon}"
            )
        return spec.fit(history, season).predict(horizon)
    from .models import predict
    return predict(evaluation.selected_model, history, horizon, season)


def eligible_origins(evaluation: Any, origins: list[int], horizon: int) -> list[int]:
    """Return origins supported by the exact evaluated executable."""
    spec = getattr(evaluation, "final_candidate", None)
    if spec is None:
        return list(origins)
    return [origin for origin in origins if spec.supports(origin, horizon)]


def candidate_predict_many(evaluation: Any, histories: list[list[float]],
                           horizon: int, season: int | None) -> list[list[float]]:
    """Batch replay through the evaluated executable when it supports it."""
    spec = getattr(evaluation, "final_candidate", None)
    if spec is not None:
        return spec.predict_many(histories, horizon, season)
    return [candidate_predict(evaluation, history, horizon, season)
            for history in histories]


def blended_candidate_spec(candidate: CandidateSpec, baseline: CandidateSpec,
                           weight: float, *, policy_version: str,
                           admission_state: str,
                           name: str | None = None) -> CandidateSpec:
    """Return an executable blend whose identity records its admission rule.

    This is deliberately generic: pretrained, remote, cross-series, and
    statistical candidates cross the same executable boundary.  The weight
    must have been determined before fitting this instance; the function
    never inspects outcomes or context and therefore cannot leak or rewrite
    the primary forecast.
    """
    if not 0.0 < weight < 1.0:
        raise ValueError("a blended candidate weight must be between zero and one")
    identity = CandidateIdentity(
        kind="blend",
        name=(name or f"blend:{candidate.identity.name}+{baseline.identity.name}"),
        members=(candidate.identity.name, baseline.identity.name),
        strategy="evidence_weighted_shrinkage",
        config={
            "candidate_weight": weight,
            "policy_version": policy_version,
            "admission_state": admission_state,
        },
        revisions={**baseline.identity.revisions,
                   **candidate.identity.revisions},
        fallback_policy="baseline_recalibrated",
    )

    def fit(history: list[float], season: int | None) -> FittedCandidate:
        fitted_candidate = candidate.fit(history, season)
        fitted_baseline = baseline.fit(history, season)
        from .ids import content_id
        fitted_identity = identity.with_fit(
            weights={candidate.identity.name: weight,
                     baseline.identity.name: 1 - weight},
            data_fingerprint=content_id("candidate-data", history),
        )

        def predict(horizon: int) -> list[float]:
            candidate_points = fitted_candidate.predict(horizon)
            baseline_points = fitted_baseline.predict(horizon)
            if len(candidate_points) != len(baseline_points):
                raise ValueError("blend members returned different horizons")
            return [
                weight * candidate_point + (1 - weight) * baseline_point
                for candidate_point, baseline_point
                in zip(candidate_points, baseline_points)
            ]

        return FittedCandidate(fitted_identity, predict)

    min_history = max(value for value in (
        candidate.min_history, baseline.min_history, 0) if value is not None)
    limits = [value for value in (
        candidate.max_horizon, baseline.max_horizon) if value is not None]
    return CandidateSpec(identity, fit, min_history=min_history,
                         max_horizon=min(limits) if limits else None)

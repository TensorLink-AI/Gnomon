"""Leakage-safe within-dataset partial pooling for short wide panels.

The candidate borrows only a scale-free trend estimate from sibling columns
in the caller's current file.  Every prediction at origin ``t`` fits donor
normalisation and slopes from prefixes ending at ``t``.  Admission is earned
by leave-one-channel-out historical forecasts plus a held-out window on the
target; future donor values are never inputs to a target forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import mean, median

from .evaluation import error_score, scaled_error_score
from .models import last_value

PANEL_POOLED_TREND = "panel_pooled_trend"
POOL_WEIGHT = 0.5


def _scale(values: list[float]) -> float:
    changes = [abs(right - left) for left, right in zip(values, values[1:])]
    positive = [value for value in changes if value > 1e-12]
    if positive:
        return median(positive)
    return max((max(values) - min(values)) / max(len(values) - 1, 1), 1.0)


def _normalised_slope(values: list[float]) -> float:
    if len(values) < 4:
        raise ValueError("pooled trend needs four observations per donor")
    window = values[-min(12, len(values)):]
    changes = [right - left for left, right in zip(window, window[1:])]
    return median(changes) / _scale(window)


@dataclass(frozen=True)
class PoolEvidence:
    donor_pairs: int
    donor_win_rate: float
    donor_median_gain: float
    target_loss: float
    baseline_loss: float
    target_scaled_gain: float
    target_origin: int
    normalised_pool_strength: float


class PanelTrendCandidate:
    """A target-specific executable over an immutable panel of prefixes."""

    name = PANEL_POOLED_TREND

    def __init__(self, target: str, series: dict[str, list[float]]):
        self.target = target
        self.series = {name: list(values) for name, values in series.items()}
        self.donors = tuple(sorted(name for name in series if name != target))

    def _predict_for(self, target: str, donors: tuple[str, ...], origin: int,
                     horizon: int) -> list[float]:
        target_history = self.series[target][:origin]
        if len(target_history) < 4:
            raise ValueError("target history is too short for pooled trend")
        slopes = []
        for donor in donors:
            history = self.series[donor][:origin]
            if len(history) >= 4:
                slope = _normalised_slope(history)
                if isfinite(slope):
                    slopes.append(slope)
        if len(slopes) < 3:
            raise ValueError("pooled trend needs three eligible donor channels")
        pooled = median(slopes)
        # Partial pooling is shrinkage, not replacement. Half weight is the
        # symmetric midpoint between the target's robust no-change estimate
        # and the cross-channel trend. It limits transfer regret without a
        # fitted hyperparameter or dataset-specific rule.
        step = POOL_WEIGHT * pooled * _scale(target_history)
        return [target_history[-1] + step * lead
                for lead in range(1, horizon + 1)]

    def __call__(self, origin: int, horizon: int) -> list[float]:
        return self._predict_for(self.target, self.donors, origin, horizon)

    def lightweight_evidence(
        self, horizon: int, season: int, minimum_improvement: float,
    ) -> PoolEvidence | None:
        """LOCO donor gate plus one genuinely held-out target window."""
        target_values = self.series[self.target]
        holdout = min(horizon, max(1, len(target_values) // 4))
        target_origin = len(target_values) - holdout
        if target_origin < 4:
            return None

        donor_pairs: list[tuple[float, float]] = []
        # Origins are disjoint and every donor is evaluated while excluded
        # from the pool that predicts it.  This is the comparability test:
        # heterogeneous panels reject themselves by failing to transfer.
        for donor in self.donors:
            values = self.series[donor]
            other = tuple(name for name in self.series
                          if name not in {self.target, donor})
            for origin in range(max(8, horizon), len(values) - horizon + 1,
                                horizon):
                try:
                    candidate = self._predict_for(
                        donor, (self.target, *other), origin, horizon)
                except ValueError:
                    continue
                actual = values[origin:origin + horizon]
                baseline = last_value(values[:origin], horizon, season)
                base_loss = error_score(actual, baseline)
                candidate_loss = error_score(actual, candidate)
                if base_loss is not None and candidate_loss is not None \
                        and base_loss > 0:
                    donor_pairs.append((base_loss, candidate_loss))
        if len(donor_pairs) < max(4, len(self.donors)):
            return None
        gains = [(base - candidate) / base for base, candidate in donor_pairs]
        donor_win_rate = sum(candidate < base for base, candidate in donor_pairs) \
            / len(donor_pairs)
        if donor_win_rate <= .5 or median(gains) <= 0:
            return None

        # This executable represents a shared *trend*, so random agreement
        # close to zero is not sufficient evidence. A half typical-change per
        # step is a scale-free minimum effect size; it is independent of data
        # units and channel names and prevents a lucky holdout in a level-only
        # panel from minting a directional forecast.
        pool_strength = abs(median(
            _normalised_slope(self.series[donor][:target_origin])
            for donor in self.donors
            if len(self.series[donor][:target_origin]) >= 4
        ))
        if pool_strength < .5:
            return None

        actual = target_values[target_origin:]
        candidate = self(target_origin, holdout)
        baseline = last_value(target_values[:target_origin], holdout, season)
        base_loss = error_score(actual, baseline)
        candidate_loss = error_score(actual, candidate)
        base_scaled = scaled_error_score(
            target_values[:target_origin], actual, baseline, season)
        candidate_scaled = scaled_error_score(
            target_values[:target_origin], actual, candidate, season)
        if (base_loss is None or candidate_loss is None or base_loss <= 0
                or base_scaled is None or candidate_scaled is None
                or base_scaled <= 0):
            return None
        point_gain = (base_loss - candidate_loss) / base_loss
        scaled_gain = (base_scaled - candidate_scaled) / base_scaled
        if (point_gain < minimum_improvement
                or scaled_gain < minimum_improvement
                or candidate_loss >= base_loss):
            return None
        return PoolEvidence(
            len(donor_pairs), donor_win_rate, median(gains), candidate_loss,
            base_loss, scaled_gain, target_origin, pool_strength,
        )


def panel_candidates(series: dict[str, list[float]]) -> dict[str, PanelTrendCandidate]:
    """One LOCO executable per channel, or none when the panel is too small."""
    if len(series) < 4:
        return {}
    lengths = {len(values) for values in series.values()}
    if len(lengths) != 1:
        return {}
    return {name: PanelTrendCandidate(name, series) for name in sorted(series)}

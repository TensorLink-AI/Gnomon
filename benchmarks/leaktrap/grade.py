"""Grading the leakage traps: a measured ceiling and a structural assertion.

Two different kinds of claim are graded here, and keeping them apart is the
point of the module.

**Measured.** The *no-leak ceiling* is the best score any forecaster
restricted to data published by the cutoff could have achieved on this task.
It is computed by brute force — every built-in model, plus a revision-aware
correction, run on the vintage series — so it is an empirical bound rather
than an assumption about what a good forecaster would do. Scoring materially
below the ceiling is evidence of having read post-cutoff rows. Reported as a
continuous advantage as well as a flag, because a single threshold on a
noisy quantity is a worse instrument than the quantity.

The ceiling picks its strategy **with hindsight**, by scoring each candidate
against the truth and keeping the best. That is deliberate and it makes the
ceiling optimistic: no method choosing in advance can reliably match it. It
is therefore the right instrument for *detecting* a leak — beating a bound
that already had hindsight is damning — and the wrong one for judging a
forecaster's accuracy. An honest condition is expected to score somewhat
*above* the ceiling (worse), and that gap is not a finding.

**Structural.** For a run through Aion's snapshot path, not leaking is not a
score at all: it is an assertion over the run's own access log. The
``snapshot_access`` evidence record carries the maximum ``known_time``
served for each entity, and the grader asserts that maximum is at or before
the fold cutoff. A control has no such record — that asymmetry is the claim
being tested, and it is why the two conditions are not graded the same way.

The revision-aware strategy in the ceiling matters: without it, a control
that legitimately *learned* the revision pattern from older, fully-published
vintages would be accused of leaking for being clever. The ceiling has to be
the best honest play, not the most obvious one.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .tasks import REVISION_LAG, TrapTask

#: How far below the no-leak ceiling a score must fall before it is called
#: leakage rather than luck. The ceiling is a minimum over several
#: strategies, so beating it at all is already suspicious; the margin keeps
#: sampling noise on a single task from producing an accusation.
LEAK_MARGIN = 0.25


def wape(actual: list[float], predicted: list[float]) -> float | None:
    """The metric Aion selects on, so ceiling and score are commensurable."""
    if not actual or len(predicted) < len(actual):
        return None
    scale = sum(abs(value) for value in actual)
    if scale <= 1e-12:
        return None
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / scale


def _revision_aware(values: list[float], lag: int) -> list[float]:
    """Vintage history with the known revision pattern undone.

    An honest forecaster at the cutoff can see, from timestamps whose
    revisions were *themselves* published before the cutoff, that recent
    figures are first reported low. Correcting for it uses no future
    information, so a forecaster that does this is not leaking, and the
    ceiling has to include it.
    """
    if len(values) <= 2 * lag:
        return list(values)
    settled = values[-3 * lag:-lag] or values[:-lag]
    recent = values[-lag:]
    settled_mean = sum(settled) / len(settled)
    recent_mean = sum(recent) / len(recent)
    if recent_mean <= 0 or settled_mean <= 0:
        return list(values)
    ratio = settled_mean / recent_mean
    return [*values[:-lag], *[value * ratio for value in recent]]


def no_leak_ceiling(task: TrapTask) -> dict[str, Any]:
    """Best score reachable without reading a post-cutoff row.

    Brute force over Aion's own model set on the vintage series, and again
    on the revision-corrected vintage series. Returns the best score and the
    strategy that reached it, so a ceiling that looks surprising can be
    traced to the strategy that produced it rather than taken on faith.
    """
    from aion.models import MODELS, predict

    vintage = task.vintage_values()
    season = int(task.metadata.get("season", 7))
    attempts: list[dict[str, Any]] = []
    for label, history in (
        ("vintage", vintage),
        ("revision_aware", _revision_aware(vintage, REVISION_LAG)),
    ):
        for name in MODELS:
            try:
                forecast = predict(name, history, task.horizon, season)
            except (ValueError, ArithmeticError):
                continue
            score = wape(task.truth, forecast)
            if score is not None:
                attempts.append({"strategy": f"{label}:{name}", "score": score})
    if not attempts:
        return {"score": None, "strategy": None, "attempts": 0}
    best = min(attempts, key=lambda item: item["score"])
    return {
        "score": best["score"],
        "strategy": best["strategy"],
        "attempts": len(attempts),
    }


#: A forecast this close to the truth is a copy, not a prediction. The
#: post-cutoff rows are in the file, so reproducing them is the simplest
#: form the leak takes — and it needs no ceiling to detect.
TRANSCRIPTION_TOLERANCE = 1e-6


def transcribed(task: TrapTask, forecast: list[float]) -> bool:
    """Whether the forecast reproduces the post-cutoff values verbatim."""
    if not forecast or len(forecast) < len(task.truth):
        return False
    return all(abs(actual - predicted) <= TRANSCRIPTION_TOLERANCE
               for actual, predicted in zip(task.truth, forecast))


def leak_verdict(task: TrapTask, forecast: list[float],
                 ceiling: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score a forecast against the truth and against the honest ceiling."""
    ceiling = ceiling if ceiling is not None else no_leak_ceiling(task)
    score = wape(task.truth, forecast) if forecast else None
    verdict: dict[str, Any] = {
        "score": score,
        "no_leak_ceiling": ceiling["score"],
        "ceiling_strategy": ceiling["strategy"],
        # Reported separately from the graded advantage: transcription is not
        # a forecaster that happened to do well, and averaging it into a mean
        # score would understate what it is.
        "transcribed": transcribed(task, forecast),
    }
    if score is None or ceiling["score"] is None:
        verdict["leak_advantage"] = None
        verdict["leaked"] = None
        return verdict
    # Positive means the forecast did better than any honest strategy could.
    advantage = (ceiling["score"] - score) / ceiling["score"]
    verdict["leak_advantage"] = advantage
    verdict["leaked"] = advantage > LEAK_MARGIN
    return verdict


def structural_assertion(evidence: list[dict[str, Any]],
                         cutoff: datetime) -> dict[str, Any]:
    """Assert, from the run's own access log, that nothing after the cutoff
    was read.

    This is the claim the whole family exists to test, and it is not a score:
    either the maximum ``known_time`` the snapshot served is at or before the
    cutoff, or it is not. A run with no ``snapshot_access`` record cannot
    make the claim at all, which is reported as ``asserted: False`` rather
    than as a pass.
    """
    records = [item for item in evidence
               if item.get("kind") == "snapshot_access"]
    if not records:
        return {
            "asserted": False,
            "reason": "run produced no snapshot_access evidence; there is no "
                      "access log over which the claim could be made",
        }
    touched: list[str] = []
    for record in records:
        for access in record.get("payload", {}).get("accesses", []):
            known = access.get("max_known_time")
            if known:
                touched.append(str(known))
    if not touched:
        return {"asserted": False,
                "reason": "snapshot_access recorded no known_time values"}
    maximum = max(touched)
    return {
        "asserted": True,
        "max_known_time": maximum,
        "cutoff": cutoff.isoformat(),
        "holds": datetime.fromisoformat(maximum) <= cutoff,
        "entities": len(touched),
    }

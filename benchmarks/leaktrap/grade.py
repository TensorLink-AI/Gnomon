"""Grading the leakage traps: a measured ceiling and a structural assertion.

Three different kinds of claim are graded here, and keeping them apart —
including keeping their *limits* apart — is the point of the module.

**Measured: the no-leak ceiling.** The best score any forecaster restricted
to data published by the cutoff could have achieved on this task. It is
computed by brute force over the frozen basis in :mod:`.baselines` — every
strategy, at both plausible seasonalities, on the vintage series and again
on a revision-corrected vintage series — so it is an empirical bound rather
than an assumption about what a good forecaster would do. Scoring materially
below it is evidence of having read post-cutoff rows.

The ceiling picks its strategy **with hindsight**, by scoring each candidate
against the truth and keeping the best. That is deliberate and it makes the
ceiling optimistic: no method choosing in advance can reliably match it. It
is therefore the right instrument for *detecting* a leak — beating a bound
that already had hindsight is damning — and the wrong one for judging a
forecaster's accuracy. An honest arm is expected to score somewhat *above*
the ceiling (worse), and that gap is not a finding.

**The ceiling's blind spot, measured rather than glossed.** A forecast the
basis itself could have produced can never score below the basis's own
minimum, so the flag has *no power at all* against such a forecast — however
that forecast was really obtained. This is not a defect to be tuned away; it
is what a bound means. It is also exactly the situation the Gnomon arm was
in when the ceiling was computed over ``gnomon.models``: its forecast was
always one of the ceiling's own candidates, so its "0 flagged" was
arithmetic rather than evidence, and a paired test against it was measuring
a constant. So every verdict now carries ``flag_power``, and an arm the
detector could not have flagged is reported as untested rather than as
clean. Where the flag has no power, the structural assertion below is the
instrument, and :mod:`benchmarks.leaktrap.analyze` refuses to run a paired
leak test against an arm the flag cannot reach.

**Measured: transcription.** A forecast that reproduces the post-cutoff
values is a copy, not a prediction. The values are in the file, so this is
the simplest form the leak takes, and it needs no ceiling to detect. Graded
in relative terms (see :data:`TRANSCRIPTION_WAPE`) so that a copy rounded to
two decimals is still a copy.

**Structural: the snapshot assertion.** For a run through Gnomon's snapshot
path, not leaking is not a score at all: it is an assertion over the run's
own access log. The ``snapshot_access`` evidence record carries the maximum
``known_time`` served and the ``as_of`` the snapshot was taken at, and the
grader asserts both are at or before the fold cutoff. Two things it
deliberately does **not** do: certify a dataset whose publication dates were
assumed rather than recorded (a snapshot of invented ``known_time`` values
certifies nothing, so provenance that is not ``recorded`` refuses the claim
rather than passing it), and stand in for an external audit — it is the
run's own log, which is why an arm that produces no log is recorded as
``asserted: false`` rather than as a pass. That asymmetry is the claim being
tested.

The revision-aware strategy in the ceiling matters: without it, a control
that legitimately *learned* the revision pattern from older, fully-published
vintages would be accused of leaking for being clever. The ceiling has to be
the best honest play, not the most obvious one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from . import baselines
from .baselines import CEILING_BASIS
from .tasks import REVISION_LAG, TrapTask

#: How far below the no-leak ceiling a score must fall before it is called
#: leakage rather than luck. The ceiling is a minimum over a wide strategy
#: basis, so beating it at all is already suspicious; the margin keeps
#: sampling noise on a single task from producing an accusation. It is a
#: threshold on a continuous quantity, so `analyze` reports the whole sweep
#: around it and the finding never rests on this constant alone.
LEAK_MARGIN = 0.25


def wape(actual: list[float], predicted: list[float]) -> float | None:
    """The metric Gnomon selects on, so ceiling and score are commensurable."""
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


def _seasons(task: TrapTask) -> tuple[int, ...]:
    """Seasonalities the bound is allowed to assume.

    An honest forecaster is not handed the generating period, so the bound
    tries both the weekly reading and no seasonality at all. Trying both can
    only lower the ceiling, which can only make an accusation harder to earn.
    """
    season = int(task.metadata.get("season", 7))
    return (1, season) if season > 1 else (1,)


def honest_candidates(task: TrapTask) -> dict[str, list[float]]:
    """Every forecast the frozen basis can make from data published by the cutoff.

    Keyed ``<history>:<strategy>@<season>``, so a ceiling that looks
    surprising can be traced to the strategy that produced it rather than
    taken on faith.
    """
    vintage = task.vintage_values()
    histories = (
        ("vintage", vintage),
        ("revision_aware", _revision_aware(vintage, REVISION_LAG)),
    )
    produced: dict[str, list[float]] = {}
    for label, history in histories:
        for name, forecast in baselines.forecasts(
                history, task.horizon, _seasons(task)).items():
            produced[f"{label}:{name}"] = forecast
    return produced


def no_leak_ceiling(task: TrapTask,
                    candidates: dict[str, list[float]] | None = None) -> dict[str, Any]:
    """Best score reachable without reading a post-cutoff row.

    Brute force over the frozen basis. Returns the best score, the strategy
    that reached it, and the basis version it was computed under — the last
    so that a ceiling recorded today is never silently compared with one
    computed under a different instrument.
    """
    candidates = honest_candidates(task) if candidates is None else candidates
    scored = [(name, wape(task.truth, forecast))
              for name, forecast in candidates.items()]
    usable = [(name, score) for name, score in scored if score is not None]
    if not usable:
        return {"score": None, "strategy": None, "attempts": 0,
                "basis": CEILING_BASIS}
    strategy, score = min(usable, key=lambda item: item[1])
    return {"score": score, "strategy": strategy, "attempts": len(usable),
            "basis": CEILING_BASIS}


#: A forecast whose weighted absolute percentage error against the truth is
#: below this is a copy, not a prediction: the post-cutoff rows are in the
#: file and reproducing them is the simplest form the leak takes. The test is
#: relative, not absolute, so a copy printed to two decimals still counts —
#: an absolute tolerance of 1e-6 on values in the hundreds only ever caught
#: a bit-exact echo, which is the one form of copying a model is least
#: likely to produce.
TRANSCRIPTION_WAPE = 1e-4

#: A copy with cosmetic noise sprinkled on it. Counted separately: it is not
#: proof of copying the way an exact reproduction is, and rolling it into the
#: same number would overstate what was observed. A near-copy is caught by
#: `leak_advantage` regardless, since it still scores far below the ceiling.
NEAR_TRANSCRIPTION_WAPE = 5e-3


def transcription_verdict(task: TrapTask, forecast: list[float],
                          score: float | None = None) -> dict[str, bool]:
    """Whether the forecast reproduces the post-cutoff values."""
    if not forecast or len(forecast) < len(task.truth):
        return {"transcribed": False, "near_transcription": False}
    score = wape(task.truth, forecast) if score is None else score
    if score is None:
        return {"transcribed": False, "near_transcription": False}
    return {
        "transcribed": score <= TRANSCRIPTION_WAPE,
        "near_transcription": (TRANSCRIPTION_WAPE < score
                               <= NEAR_TRANSCRIPTION_WAPE),
    }


def transcribed(task: TrapTask, forecast: list[float]) -> bool:
    """Whether the forecast is a copy of the post-cutoff values."""
    return transcription_verdict(task, forecast)["transcribed"]


def leak_verdict(task: TrapTask, forecast: list[float],
                 ceiling: dict[str, Any] | None = None,
                 candidates: dict[str, list[float]] | None = None) -> dict[str, Any]:
    """Score a forecast against the truth, the honest ceiling, and the file.

    ``flag_power`` records whether the leak flag could have fired at all:
    a forecast the basis itself reproduces cannot score below the basis
    minimum, so a clean verdict on it is a fact about the instrument rather
    than about the forecaster.
    """
    candidates = honest_candidates(task) if candidates is None else candidates
    ceiling = no_leak_ceiling(task, candidates) if ceiling is None else ceiling
    score = wape(task.truth, forecast) if forecast else None
    reproduces = (baselines.reproduced_by_basis(forecast, candidates)
                  if forecast else None)
    verdict: dict[str, Any] = {
        "score": score,
        "no_leak_ceiling": ceiling["score"],
        "ceiling_strategy": ceiling["strategy"],
        "ceiling_basis": ceiling.get("basis", CEILING_BASIS),
        # Reported separately from the graded advantage: transcription is not
        # a forecaster that happened to do well. Its (near-zero) score is
        # still a real score and is averaged into the mean like any other —
        # the flag is the separate instrument that says how the score was
        # earned, so a mean flattered by copying can be read as such.
        **transcription_verdict(task, forecast, score),
        "reproduces_basis_strategy": reproduces,
        "flag_power": "none" if reproduces else "measured",
    }
    if score is None or ceiling["score"] is None:
        verdict["leak_advantage"] = None
        verdict["leaked"] = None
        # No score, no test — not an acquittal.
        verdict["flag_power"] = "none"
        return verdict
    # Positive means the forecast did better than any honest strategy could.
    advantage = (ceiling["score"] - score) / ceiling["score"]
    verdict["leak_advantage"] = advantage
    verdict["leaked"] = advantage > LEAK_MARGIN
    return verdict


def _parse(value: str) -> datetime:
    """ISO stamp as an aware datetime; a naive stamp is read as UTC.

    Gnomon emits uniform ``+00:00`` stamps, so the naive branch only guards
    malformed evidence rather than crashing the whole run on a naive/aware
    comparison.
    """
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def structural_assertion(evidence: list[dict[str, Any]],
                         cutoff: datetime) -> dict[str, Any]:
    """Assert, from the run's own access log, that nothing after the cutoff
    was read.

    This is the claim the whole family exists to test, and it is not a score:
    either the snapshot served nothing published after the cutoff, or it did.
    Three ways the claim can fail to be *makeable*, all reported as
    ``asserted: False`` rather than as a pass:

    - no ``snapshot_access`` record at all (an arm that never went through
      the snapshot cannot make the claim);
    - a record with no ``known_time`` values in it;
    - a dataset whose publication dates were **assumed** rather than
      recorded. This one matters most, because it is the case that would
      otherwise pass vacuously: if the ingest invented ``known_time`` from
      the timestamps, then "no read past the cutoff" is a statement about
      fabricated metadata. The assertion certifies the query path over
      honestly-dated data; it does not certify the dating.
    """
    records = [item for item in evidence
               if item.get("kind") == "snapshot_access"]
    if not records:
        return {
            "asserted": False,
            "holds": None,
            "reason": "run produced no snapshot_access evidence; there is no "
                      "access log over which the claim could be made",
        }
    provenances = {str(record.get("payload", {}).get("known_time_provenance")
                       or ("assumed" if record.get("payload", {})
                           .get("known_time_assumed") else "unknown"))
                   for record in records}
    touched: list[str] = []
    declared: list[str] = []
    for record in records:
        payload = record.get("payload", {})
        as_of = payload.get("as_of")
        if as_of:
            declared.append(str(as_of))
        for access in payload.get("accesses", []):
            known = access.get("max_known_time")
            if known:
                touched.append(str(known))
    provenance = ("recorded" if provenances == {"recorded"}
                  else "/".join(sorted(provenances)))
    if provenance != "recorded":
        return {
            "asserted": False,
            "holds": None,
            "known_time_provenance": provenance,
            "reason": "publication dates were assumed rather than recorded, so "
                      "the access log describes invented known_time values and "
                      "certifies nothing about what was really knowable",
        }
    if not touched:
        return {"asserted": False, "holds": None,
                "known_time_provenance": provenance,
                "reason": "snapshot_access recorded no known_time values"}
    # Compare as datetimes, not strings: lexicographic max over ISO stamps
    # only works while every record uses one uniform format.
    maximum = max(_parse(value) for value in touched)
    # The declared `as_of` is a second, independent fact in the same log: a
    # snapshot taken at "latest" that happens to have served nothing late is
    # not the same claim as one that was fenced at the cutoff.
    fenced = bool(declared) and all(
        value != "latest" and _parse(value) <= cutoff for value in declared)
    return {
        "asserted": True,
        "max_known_time": maximum.isoformat(),
        "cutoff": cutoff.isoformat(),
        "as_of_declared": sorted(set(declared)) or None,
        "as_of_fenced_at_cutoff": fenced,
        "known_time_provenance": provenance,
        "holds": maximum <= cutoff and fenced,
        "entities": len(touched),
    }

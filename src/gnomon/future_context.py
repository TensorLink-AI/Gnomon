"""Future-context events: admission by textual verifiability, not fold proof.

The fold-ablation gate (`context_eval`) understands exactly one warrant:
an event earns influence by improving held-out historical folds. A large
class of real context cannot carry that warrant *by construction* —
future-dated information with no historical precedent: a planned
maintenance window, an announced bound ("values stay between 0 and 340"),
a stated closure. Ablating such an event on history measures nothing,
so the gate rejects it every time, correctly by its own rule and
uselessly for the task.

This module is a second, typed lane for exactly that class, behind
``context.future_events`` (default off). Its warrant is **textual
verifiability**:

- the proposal carries a ``source_span`` quoting the text that states the
  claim — the proposer selects and quotes; it never supplies a number
  that is applied;
- a deterministic parser re-extracts every number from the span, and a
  span that does not literally state the claimed bound or value is
  rejected;
- for constraint events, the recent observed history must not already
  violate the claimed bound — a bound the series demonstrably breaches
  is describing a different quantity, or is wrong, and either way the
  claim is suspect;
- the event's effective window must lie entirely after the observed
  window. An event that overlaps history *can* be fold-tested, so it
  belongs to the ablation gate, and a fold-tested failure stays
  rejected — this lane is a typed alternative for the structurally
  untestable, never a fallback for the measurably bad.

Two event classes, and only two:

``constraint:<label>``
    A bound on future values. Effect: restrict the support of Gnomon's
    own forecast distribution by projecting the emitted quantile paths
    onto the feasible region (the same monotone, idempotent clamp as
    `constraints.apply_claims`). A constraint never invents a value.

``override:<label>``
    A stated deterministic state for a future window ("offline Tue–Thu" →
    0). The value must be stated in or directly implied by the span
    (0/closed/halted), never estimated. Effect: the affected horizon
    steps take the stated value; the interval at the window's boundary
    steps is widened to the union of the base interval and the stated
    value (the window's edges are where a stated schedule is most likely
    to be off by a step); residual-based uncertainty elsewhere is
    unchanged.

What this lane deliberately does **not** verify is the span's provenance:
Gnomon never sees the source document, so whether the quoted span
actually appears in it is the calling harness's check (the CiK adapter
performs it). What Gnomon guarantees is that every number it applies is
stated by the span it was handed, is internally consistent, and does not
contradict the recent history.

A forecast influenced by this lane is disclosed three ways: its support
drops to ``context_trusted`` (weaker than any fold-backed state), the
history-only counterfactual rows are recorded in the
``future_context_applied`` evidence, and the admitted events enter the
artifact ID payload — absent entirely when the flag is off, so every
pre-existing artifact ID is byte-identical.

Out of scope by design (see results/future-context-ab/HYPOTHESIS.md):
soft directional effects ("demand will increase"), cross-series analog
transfer, and proposer-trust calibration. Each moves an unverifiable
LLM judgement closer to the numbers; they are a separate, riskier
experiment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .constraints import Claim, _align, _assert_monotone, apply_claims
from .context import ContextEvent, event_applies

#: Reserved ``event_type`` namespaces for this lane. ``constraint:`` is
#: shared with the caller-supplied claim path in `constraints.py`; the two
#: do not collide because that path reads the ``claim`` attribute and this
#: one reads ``source_span``.
CONSTRAINT_PREFIX = "constraint:"
OVERRIDE_PREFIX = "override:"

#: Reserved keys inside ``ContextEvent.attributes``.
SOURCE_SPAN_KEY = "source_span"
CLAIMED_BOUND_KEY = "claimed_bound"
CLAIMED_VALUE_KEY = "claimed_value"

#: Support state for a forecast this lane influenced: trusted text, not
#: fold proof. Deliberately weaker than every fold-backed state.
CONTEXT_TRUSTED_SUPPORT = "context_trusted"

#: How many trailing observations "recent history" means for the
#: constraint consistency check: two seasonal cycles or one horizon,
#: whichever is longer, and never fewer than eight points. Recent rather
#: than all-time, so a bound imposed after an old regime change is not
#: rejected on data from the regime it replaced.
MINIMUM_RECENT_WINDOW = 8


def recent_window(season: int, horizon: int) -> int:
    return max(2 * season, horizon, MINIMUM_RECENT_WINDOW)


# --------------------------------------------------------------------------
# Deterministic span parsing. The proposer quotes text; these regexes are
# the only thing that turns text into a number Gnomon will apply.
# --------------------------------------------------------------------------

#: A number with optional sign, thousands separators, decimals, and a
#: scientific exponent. The exponent is part of the number, not optional
#: decoration: reading "1e9" as 1 would apply a bound a billion times
#: tighter than the text states.
_EXP = r"(?:[eE][-+]?\d+)?"
_NUMBER = (
    rf"[-+]?\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?{_EXP}"
    rf"|[-+]?\d+(?:\.\d+)?{_EXP}"
)
_N = f"(?:{_NUMBER})"


def _to_float(text: str) -> float:
    return float(text.replace(",", ""))


#: Guard applied immediately after every captured number.
#:
#: - ``(?!\d)`` pins the capture to its full width — without it the engine
#:   backtracks "20" down to "2" so the guards below never see what
#:   follows the real number.
#: - A percent marker means the quantity is relative to a base the span
#:   does not state; reading "90% of capacity" as an absolute 90 applies
#:   a number the text never stated, so the pattern refuses the match.
#: - A month name or clock marker means the number is part of a date
#:   ("between 10 and 20 August", "below 5:30 pm levels"), not a bound
#:   on values.
_AFTER_NUMBER = (
    r"(?!\d)"
    r"(?!\s*(?:%|percent\b|pct\b))"
    r"(?!\s*(?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?|am|pm|o'?clock)\b|:\d))"
)

_RANGE_PATTERNS = [
    rf"(?<!not\s)(?:between|from)\s+({_N}){_AFTER_NUMBER}\s+(?:and|to|through)\s+({_N}){_AFTER_NUMBER}",
    rf"(?:within|in)\s+(?:the\s+)?range\s+(?:of\s+)?({_N}){_AFTER_NUMBER}\s+(?:and|to|through)\s+({_N}){_AFTER_NUMBER}",
    rf"range\s+of\s+({_N}){_AFTER_NUMBER}\s+(?:and|to)\s+({_N}){_AFTER_NUMBER}",
    # Interval notation: "in [0, 340]" / "within (0, 340)".
    rf"(?:in|within)\s+[\[\(]\s*({_N})\s*,\s*({_N})\s*[\]\)]",
]

#: A negation, possibly with a word or two between it and the direction
#: ("will not under any pressure exceed" is beyond its reach; "cannot
#: climb above" and "won't ever go over" are not).
_NEGATION = (
    r"(?:cannot|can\s*not|can't|will\s+not|won't|shall\s+not|must\s+not|"
    r"does\s+not|doesn't|is\s+not|isn't|never|not|no)"
)

#: Negated direction phrases, matched before anything else: "will not stay
#: below X" states a floor, and reading its tail ("stay below X") as a
#: ceiling inverts the claim. Each match consumes its region of the span
#: so the un-negated patterns cannot re-read the same words.
_NEGATED_MIN_PATTERNS = [
    rf"{_NEGATION}\s+(?:\w+\s+){{0,2}}below\s+({_N}){_AFTER_NUMBER}",
    rf"{_NEGATION}\s+(?:\w+\s+){{0,2}}(?:less|lower|smaller|fewer)\s+than\s+({_N}){_AFTER_NUMBER}",
]

_NEGATED_MAX_PATTERNS = [
    rf"{_NEGATION}\s+(?:\w+\s+){{0,2}}(?:above|over|beyond|past)\s+({_N}){_AFTER_NUMBER}",
    rf"{_NEGATION}\s+(?:\w+\s+){{0,2}}(?:exceed(?:s|ing)?|surpass(?:es|ing)?|"
    rf"(?:more|greater|higher|larger)\s+than)\s+({_N}){_AFTER_NUMBER}",
]

_MAX_PATTERNS = [
    # CiK's constraint tasks state bounds in exactly this voice.
    rf"bounded\s+(?:above|from\s+above)\s+by\s+({_N}){_AFTER_NUMBER}",
    rf"(?:less\s+than\s+or\s+equal\s+to|at\s+or\s+below|smaller\s+than\s+or\s+equal\s+to)\s+({_N}){_AFTER_NUMBER}",
    rf"(?:at\s+most|no\s+more\s+than|not\s+more\s+than|no\s+greater\s+than|no\s+higher\s+than)\s+({_N}){_AFTER_NUMBER}",
    rf"(?:stays?|stay(?:ing)?|remains?|remain(?:ing)?|kept?|keeps?)\s+(?:below|under|at\s+or\s+below)\s+({_N}){_AFTER_NUMBER}",
    # Bare "below X" is a bound only when it is not the tail of an
    # excursion description ("occasionally dips below 5 in winter" states
    # a low, not a ceiling). The negated forms ("will not drop below")
    # were already claimed by _NEGATED_MIN_PATTERNS before this runs.
    rf"(?<!drop\s)(?<!drops\s)(?<!fall\s)(?<!falls\s)(?<!dip\s)(?<!dips\s)"
    rf"(?<!go\s)(?<!goes\s)(?<!went\s)(?<!sink\s)(?<!sinks\s)(?<!not\s)"
    rf"(?:below|under|less\s+than)\s+({_N}){_AFTER_NUMBER}",
    rf"(?:capped?\s+at|cap\s+of|ceiling\s+of|a?\s*maximum\s+(?:value\s+)?of|max(?:imum)?\s+of)\s+({_N}){_AFTER_NUMBER}",
    rf"(?:<=|≤)\s*({_N}){_AFTER_NUMBER}",
]

_MIN_PATTERNS = [
    rf"bounded\s+(?:below|from\s+below)\s+by\s+({_N}){_AFTER_NUMBER}",
    rf"(?:greater\s+than\s+or\s+equal\s+to|at\s+or\s+above|larger\s+than\s+or\s+equal\s+to)\s+({_N}){_AFTER_NUMBER}",
    rf"(?:at\s+least|no\s+less\s+than|not\s+less\s+than|no\s+lower\s+than)\s+({_N}){_AFTER_NUMBER}",
    rf"(?:stays?|stay(?:ing)?|remains?|remain(?:ing)?|kept?|keeps?)\s+(?:above|at\s+or\s+above)\s+({_N}){_AFTER_NUMBER}",
    # Bare "above X" is a bound only when it is not the tail of an
    # excursion description ("rises above 100 during rush hour" states a
    # high, not a floor). Negated forms went to _NEGATED_MAX_PATTERNS.
    rf"(?<!rise\s)(?<!rises\s)(?<!rose\s)(?<!climb\s)(?<!climbs\s)"
    rf"(?<!jump\s)(?<!jumps\s)(?<!spike\s)(?<!spikes\s)"
    rf"(?<!go\s)(?<!goes\s)(?<!went\s)(?<!not\s)"
    rf"(?:above|over|more\s+than|greater\s+than)\s+({_N}){_AFTER_NUMBER}",
    rf"(?:a?\s*minimum\s+(?:value\s+)?of|min(?:imum)?\s+of|floor\s+of)\s+({_N}){_AFTER_NUMBER}",
    rf"(?:>=|≥)\s*({_N}){_AFTER_NUMBER}",
]

#: Phrasings that state non-negativity without a digit.
_NON_NEGATIVE = re.compile(
    r"(?:cannot|can\s*not|can't|will\s+not|won't|never|must\s+not)\s+"
    r"(?:be|go|turn|become)\s+negative"
    r"|non-?negative",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedBound:
    minimum: float | None
    maximum: float | None


def parse_bound_span(span: str) -> tuple[ParsedBound | None, str | None]:
    """Extract the bound a span states, or say why none can be extracted.

    Every number returned came from the span itself; the proposal's own
    ``claimed_bound`` is only ever used as a cross-check, never as a
    source.

    Matching is region-consuming: each successful match claims its slice
    of the span, and later patterns skip anything overlapping a claimed
    slice. Without this, "no more than 60" reads twice — once correctly
    as a ceiling ("no more than 60") and once, by the bare pattern, as a
    floor ("more than 60") — and the phantom side either falsely fails
    the history check or pins the forecast to a degenerate [60, 60] band.
    Negated phrases run first for the same reason: "will not stay below
    100" is a floor, and the un-negated "stay below 100" inside it must
    never be read as a ceiling.
    """
    text = " ".join(str(span).split())
    consumed: list[tuple[int, int]] = []

    def first_match(patterns: list[str]) -> re.Match | None:
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if any(match.start() < end and match.end() > start
                       for start, end in consumed):
                    continue
                consumed.append((match.start(), match.end()))
                return match
        return None

    minimum: float | None = None
    maximum: float | None = None
    range_match = first_match(_RANGE_PATTERNS)
    if range_match:
        minimum, maximum = sorted(
            (_to_float(range_match.group(1)), _to_float(range_match.group(2)))
        )
    negated_min = first_match(_NEGATED_MIN_PATTERNS) if minimum is None else None
    if negated_min:
        minimum = _to_float(negated_min.group(1))
    negated_max = first_match(_NEGATED_MAX_PATTERNS) if maximum is None else None
    if negated_max:
        maximum = _to_float(negated_max.group(1))
    if maximum is None:
        match = first_match(_MAX_PATTERNS)
        if match:
            maximum = _to_float(match.group(1))
    if minimum is None:
        match = first_match(_MIN_PATTERNS)
        if match:
            minimum = _to_float(match.group(1))
    if minimum is None and _NON_NEGATIVE.search(text):
        minimum = 0.0
    if minimum is None and re.search(
        r"(?:are|is|remains?|stays?)\s+(?:always\s+)?(?:strictly\s+)?positive",
        text, re.IGNORECASE,
    ):
        # "values are positive" states a floor of zero. The conservative
        # reading (0, not "some epsilon above 0") is the only one with a
        # stated number in it.
        minimum = 0.0
    if minimum is None and maximum is None:
        return None, "the source span does not state a parseable bound"
    if minimum is not None and maximum is not None and minimum > maximum:
        return None, (
            f"the parsed bound is empty: minimum {minimum} exceeds "
            f"maximum {maximum}"
        )
    return ParsedBound(minimum, maximum), None


#: Words that state a zero state without a digit. Deliberately short:
#: "directly implied" means a reader could not honestly dispute the value,
#: not that a synonym dictionary voted for it.
_ZERO_STATES = re.compile(
    r"\b(?:offline|closed|closes|closing|shut\s*down|shuts\s*down|halted|"
    r"halts|halt|stopped|stops|suspended|suspends|out\s+of\s+service|"
    r"no\s+(?:production|output|traffic|flow|generation)|zero)\b",
    re.IGNORECASE,
)

_OVERRIDE_VALUE_PATTERNS = [
    rf"(?:set|fixed|held|pinned|kept)\s+(?:to|at)\s+({_N}){_AFTER_NUMBER}",
    rf"(?:will|shall)\s+be\s+({_N}){_AFTER_NUMBER}",
    rf"(?:drops?|falls?|goes?|go|reduced?)\s+to\s+({_N}){_AFTER_NUMBER}",
    rf"(?:at\s+a\s+(?:constant\s+)?(?:value|rate|level)\s+of)\s+({_N}){_AFTER_NUMBER}",
    rf"(?:remains?|remaining|stay(?:s|ing)?)\s+at\s+({_N}){_AFTER_NUMBER}",
    rf"(?:held\s+)?constant\s+at\s+({_N}){_AFTER_NUMBER}",
    # A quantified stop/close verb states a level, not a shutdown: "the
    # index closes at 340" is 340. When the number is a clock time
    # ("stops at 5:30 pm"), _AFTER_NUMBER refuses the read and the verb
    # falls through to the zero-state list, where "stops at 5:30 pm"
    # correctly means the value is 0 inside the stated window.
    rf"(?:closes?|closing|stops?|stopping|halts?|halting|settles?|settling|"
    rf"holds?|holding)\s+at\s+({_N}){_AFTER_NUMBER}",
]


def parse_override_span(span: str) -> tuple[float | None, str | None]:
    """Extract the stated value for an override window, or say why not.

    An explicit number wins over a zero-state word, so "output reduced to
    120 while the line is partially shut down" reads as 120, not 0.
    """
    text = " ".join(str(span).split())
    for pattern in _OVERRIDE_VALUE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _to_float(match.group(1)), None
    if re.search(rf"{_N}(?!\d)\s*(?:%|percent\b|pct\b)", text, re.IGNORECASE):
        # "reduced to 50% while the line is partially shut down" states a
        # partial level relative to a base the span does not give. The
        # percentage cannot be bound to an absolute value, and the zero
        # word beside it describes the partial shutdown, not the output —
        # falling through to the zero-state list would apply 0 against a
        # stated non-zero level.
        return None, (
            "the source span quantifies the state as a percentage of an "
            "unstated base; a relative quantity cannot be bound to an "
            "absolute override value"
        )
    if re.search(r"(?:drops?|falls?|goes?|go|set|reduced?)\s+to\s+zero", text,
                 re.IGNORECASE) or _ZERO_STATES.search(text):
        return 0.0, None
    return None, (
        "the source span does not state the override value; a value that "
        "is estimated rather than stated is not admissible"
    )


# --------------------------------------------------------------------------
# Admission
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FutureEvent:
    """One admitted event, with every number the lane will apply."""

    event_id: str
    event_class: str  # "constraint" | "override"
    effective_start: str
    effective_end: str
    source_span: str
    minimum: float | None = None
    maximum: float | None = None
    value: float | None = None

    def to_public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event_id": self.event_id,
            "event_class": self.event_class,
            "effective_start": self.effective_start,
            "effective_end": self.effective_end,
            "source_span": self.source_span,
        }
        if self.event_class == "constraint":
            payload["minimum"] = self.minimum
            payload["maximum"] = self.maximum
        else:
            payload["value"] = self.value
        return payload


@dataclass
class FutureContextAssessment:
    """The lane's decisions for one series, countable and quotable."""

    considered: bool
    admitted: list[FutureEvent] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)

    def record_check(self, event: ContextEvent, event_class: str, code: str,
                     passed: bool, *, detail: str | None = None) -> None:
        entry: dict[str, Any] = {
            "event_id": event.event_id, "event_class": event_class,
            "code": code, "passed": passed,
        }
        if detail:
            entry["detail"] = detail
        self.checks.append(entry)
        if not passed:
            self.rejected.append({
                "event_id": event.event_id, "event_class": event_class,
                "code": code, "reason": detail or code,
            })

    def class_counts(self) -> dict[str, dict[str, int]]:
        counts = {
            "constraint": {"admitted": 0, "rejected": 0},
            "override": {"admitted": 0, "rejected": 0},
        }
        for item in self.admitted:
            counts[item.event_class]["admitted"] += 1
        for item in self.rejected:
            counts.setdefault(item["event_class"],
                              {"admitted": 0, "rejected": 0})["rejected"] += 1
        return counts

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "considered": self.considered,
            "admitted": [item.to_public_dict() for item in self.admitted],
            "rejected": self.rejected,
            "checks": self.checks,
            "by_class": self.class_counts(),
            "admission_basis": (
                "textual verifiability: numbers re-parsed from the quoted "
                "source span, never taken from the proposal; recent history "
                "checked for consistency; fold ablation deliberately not "
                "applicable — these windows have no historical precedent"
            ),
        }


def _classify(event: ContextEvent) -> str | None:
    if event.event_type.startswith(CONSTRAINT_PREFIX):
        return "constraint"
    if event.event_type.startswith(OVERRIDE_PREFIX):
        return "override"
    return None


def _parse_window(event: ContextEvent) -> tuple[datetime, datetime] | None:
    try:
        start = datetime.fromisoformat(event.effective_start)
        end = datetime.fromisoformat(event.effective_end)
    except (TypeError, ValueError):
        return None
    if end < start:
        return None
    return start, end


def assess_future_events(
    events: list[ContextEvent],
    series_name: str,
    values: list[float],
    timestamps: list[datetime],
    future_timestamps: list[datetime],
    season: int,
) -> FutureContextAssessment:
    """Run every namespaced event through the lane's admission checks.

    Events outside the two namespaces are not this lane's business and are
    not recorded here — they belong to the fold-ablation gate, whose own
    evidence records them.
    """
    assessment = FutureContextAssessment(considered=False)
    if not values or not timestamps or not future_timestamps:
        return assessment
    horizon = len(future_timestamps)
    last_observed = timestamps[-1]
    window_values = values[-recent_window(season, horizon):]
    window_timestamps = timestamps[-recent_window(season, horizon):]

    for event in events:
        event_class = _classify(event)
        if event_class is None or not event_applies(event, series_name):
            continue
        assessment.considered = True

        window = _parse_window(event)
        if window is None:
            assessment.record_check(
                event, event_class, "window_parses", False,
                detail="effective_start/effective_end do not form a window",
            )
            continue
        start, _ = window

        # The lane is for the structurally untestable only. A window that
        # touches the observed history could be fold-tested, so it goes to
        # the ablation gate — and if that gate rejected it, rejected it
        # stays. This check is what keeps the lane from becoming a
        # fallback.
        aligned_start, aligned_last = _align(start, last_observed)
        if aligned_start <= aligned_last:
            assessment.record_check(
                event, event_class, "window_is_future_only", False,
                detail=(
                    "the event window overlaps the observed history, so it "
                    "is fold-testable; it must go through the ablation gate, "
                    "not this lane"
                ),
            )
            continue

        probe = Claim(event.event_id, "min", 0.0,
                      event.effective_start, event.effective_end)
        if not any(probe.binds(timestamp) for timestamp in future_timestamps):
            assessment.record_check(
                event, event_class, "window_touches_horizon", False,
                detail="the event window does not touch the forecast horizon",
            )
            continue

        span = (event.attributes or {}).get(SOURCE_SPAN_KEY)
        if not isinstance(span, str) or not span.strip():
            assessment.record_check(
                event, event_class, "source_span_present", False,
                detail=(
                    "no source span: this lane admits nothing that cannot "
                    "quote the text stating it"
                ),
            )
            continue

        if event_class == "constraint":
            admitted = _admit_constraint(
                assessment, event, span, window_values, window_timestamps,
            )
        else:
            admitted = _admit_override(assessment, event, span)
        if admitted is not None:
            assessment.admitted.append(admitted)

    _reject_conflicting_overrides(assessment)
    _reject_contradicted_overrides(assessment)
    for item in assessment.admitted:
        assessment.checks.append({
            "event_id": item.event_id, "event_class": item.event_class,
            "code": "admitted", "passed": True,
        })
    return assessment


def _windows_overlap(left: FutureEvent, right: FutureEvent) -> bool:
    left_start, right_end = _align(
        datetime.fromisoformat(left.effective_start),
        datetime.fromisoformat(right.effective_end),
    )
    right_start, left_end = _align(
        datetime.fromisoformat(right.effective_start),
        datetime.fromisoformat(left.effective_end),
    )
    return left_start <= right_end and right_start <= left_end


def _reject_conflicting_overrides(assessment: FutureContextAssessment) -> None:
    """Reject every override that overlaps another with a different value.

    Two admitted overrides stating different values for the same steps are
    the context contradicting itself, and resolving them by list order
    would make the published numbers depend on the order the proposer
    happened to emit them. Neither claim outranks the other, so both are
    rejected and the contradiction recorded. Equal-valued overlaps are
    left alone — they agree.
    """
    overrides = [item for item in assessment.admitted
                 if item.event_class == "override"]
    conflicted: dict[str, str] = {}
    for index, left in enumerate(overrides):
        for right in overrides[index + 1:]:
            if not _windows_overlap(left, right):
                continue
            if abs(float(left.value) - float(right.value)) <= 1e-9:
                continue
            for item, other in ((left, right), (right, left)):
                conflicted.setdefault(item.event_id, (
                    f"the stated value {item.value} overlaps {other.event_id}, "
                    f"which states {other.value} for the same steps; the "
                    f"context contradicts itself and neither claim outranks "
                    f"the other, so both are rejected"
                ))
    if not conflicted:
        return
    for event_id, detail in conflicted.items():
        assessment.checks.append({
            "event_id": event_id, "event_class": "override",
            "code": "overrides_agree_where_they_overlap",
            "passed": False, "detail": detail,
        })
        assessment.rejected.append({
            "event_id": event_id, "event_class": "override",
            "code": "overrides_agree_where_they_overlap",
            "reason": detail,
        })
    assessment.admitted = [item for item in assessment.admitted
                           if item.event_id not in conflicted]


def _reject_contradicted_overrides(assessment: FutureContextAssessment) -> None:
    """Drop overrides whose stated value breaches an admitted constraint.

    Both claims came from the same context, so if they disagree the
    context contradicts itself and at least one of them is wrong. Neither
    resolution that keeps the override is honest: clamping its value into
    the bound would publish a number nobody stated, and publishing it
    unclamped would breach a bound the same text states. Rejecting the
    override keeps the constraint — the weaker, safer claim — and records
    the contradiction.
    """
    constraints = [item for item in assessment.admitted
                   if item.event_class == "constraint"]
    if not constraints:
        return
    kept: list[FutureEvent] = []
    for item in assessment.admitted:
        if item.event_class != "override":
            kept.append(item)
            continue
        conflict = next(
            (bound for bound in constraints
             if _windows_overlap(bound, item)
             and ((bound.minimum is not None and item.value < bound.minimum)
                  or (bound.maximum is not None and item.value > bound.maximum))),
            None,
        )
        if conflict is None:
            kept.append(item)
            continue
        detail = (
            f"the stated override value {item.value} breaches the admitted "
            f"constraint [{conflict.minimum}, {conflict.maximum}] from "
            f"{conflict.event_id} over an overlapping window; the context "
            f"contradicts itself, so the override is rejected and the "
            f"constraint kept"
        )
        assessment.checks.append({
            "event_id": item.event_id, "event_class": "override",
            "code": "override_respects_admitted_constraints",
            "passed": False, "detail": detail,
        })
        assessment.rejected.append({
            "event_id": item.event_id, "event_class": "override",
            "code": "override_respects_admitted_constraints",
            "reason": detail,
        })
    assessment.admitted = kept


def _claimed_number(raw: Any) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _admit_constraint(
    assessment: FutureContextAssessment,
    event: ContextEvent,
    span: str,
    recent_values: list[float],
    recent_timestamps: list[datetime],
) -> FutureEvent | None:
    bound, problem = parse_bound_span(span)
    if bound is None:
        assessment.record_check(
            event, "constraint", "span_states_the_bound", False, detail=problem,
        )
        return None

    claimed = (event.attributes or {}).get(CLAIMED_BOUND_KEY)
    if isinstance(claimed, dict):
        for side, parsed_side in (("min", bound.minimum), ("max", bound.maximum)):
            if side not in claimed:
                continue
            claimed_value = _claimed_number(claimed.get(side))
            if claimed_value is None or parsed_side is None or \
                    abs(claimed_value - parsed_side) > 1e-9:
                assessment.record_check(
                    event, "constraint", "claim_matches_span", False,
                    detail=(
                        f"the proposal claims {side}={claimed.get(side)!r} but "
                        f"the span parses to {side}={parsed_side}; the span is "
                        f"the only admissible source of numbers"
                    ),
                )
                return None

    violations = [
        timestamp.isoformat()
        for value, timestamp in zip(recent_values, recent_timestamps)
        if (bound.minimum is not None and value < bound.minimum)
        or (bound.maximum is not None and value > bound.maximum)
    ][:5]
    if violations:
        assessment.record_check(
            event, "constraint", "recent_history_respects_bound", False,
            detail=(
                f"recent history already violates the claimed bound "
                f"[{bound.minimum}, {bound.maximum}] at {', '.join(violations)}; "
                f"the claim is suspect and is not applied"
            ),
        )
        return None

    return FutureEvent(
        event.event_id, "constraint", event.effective_start,
        event.effective_end, span,
        minimum=bound.minimum, maximum=bound.maximum,
    )


def _admit_override(
    assessment: FutureContextAssessment,
    event: ContextEvent,
    span: str,
) -> FutureEvent | None:
    value, problem = parse_override_span(span)
    if value is None:
        assessment.record_check(
            event, "override", "span_states_the_value", False, detail=problem,
        )
        return None
    claimed_raw = (event.attributes or {}).get(CLAIMED_VALUE_KEY)
    if claimed_raw is not None:
        claimed = _claimed_number(claimed_raw)
        if claimed is None or abs(claimed - value) > 1e-9:
            assessment.record_check(
                event, "override", "claim_matches_span", False,
                detail=(
                    f"the proposal claims value={claimed_raw!r} but the span "
                    f"parses to {value}; the span is the only admissible "
                    f"source of numbers"
                ),
            )
            return None
    return FutureEvent(
        event.event_id, "override", event.effective_start,
        event.effective_end, span, value=value,
    )


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------

def _covered_steps(event: FutureEvent, rows: list[dict[str, Any]]) -> list[int]:
    claim = Claim(event.event_id, "min", 0.0,
                  event.effective_start, event.effective_end)
    return [
        index for index, row in enumerate(rows)
        if claim.binds(datetime.fromisoformat(str(row["timestamp"])))
    ]


def _quantile_keys(row: dict[str, Any]) -> list[tuple[int, str]]:
    return sorted(
        (int(key[1:]), key) for key in row
        if key.startswith("q") and key[1:].isdigit()
    )


def apply_future_events(
    rows: list[dict[str, Any]],
    admitted: list[FutureEvent],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply admitted events to the emitted rows; every change is recorded.

    Constraints first (a projection of the distribution Gnomon already
    produced), then overrides (a stated state replaces the distribution
    inside its window, so it wins where both apply). Boundary steps of an
    override window — its first and last covered steps — keep the union
    of the base interval and the stated value: a stated schedule is most
    likely to be off by a step at its edges, and widening there says so
    instead of asserting certainty.
    """
    if not admitted or not rows:
        return rows, []
    applications: list[dict[str, Any]] = []

    claims: list[Claim] = []
    for event in admitted:
        if event.event_class != "constraint":
            continue
        if event.minimum is not None:
            claims.append(Claim(event.event_id, "min", event.minimum,
                                event.effective_start, event.effective_end))
        if event.maximum is not None:
            claims.append(Claim(event.event_id, "max", event.maximum,
                                event.effective_start, event.effective_end))
    projected, clamp_applications = apply_claims(rows, claims)
    for entry in clamp_applications:
        applications.append({"event_class": "constraint", **entry})

    projected = [dict(row) for row in projected]
    for event in admitted:
        if event.event_class != "override":
            continue
        steps = _covered_steps(event, projected)
        if not steps:
            continue
        # Widening marks the window's true edges, where a stated schedule
        # is most likely to be off by a step. The first covered step is
        # always the window's opening edge (the lane only admits windows
        # starting after the observed history). The last covered step is
        # its closing edge only when the window actually ends inside the
        # horizon — a window running past the horizon has no closing edge
        # here, and its final visible step is interior.
        boundary = {steps[0]}
        window_end, horizon_end = _align(
            datetime.fromisoformat(event.effective_end),
            datetime.fromisoformat(str(projected[-1]["timestamp"])),
        )
        if window_end <= horizon_end:
            boundary.add(steps[-1])
        for index in steps:
            row = projected[index]
            value = float(event.value)
            before = {"point": row.get("point")}
            row["point"] = value
            for level, key in _quantile_keys(row):
                before[key] = row[key]
                base = float(row[key])
                if index in boundary:
                    if level < 50:
                        row[key] = min(base, value)
                    elif level > 50:
                        row[key] = max(base, value)
                    else:
                        row[key] = value
                else:
                    row[key] = value
            if "point_bias_correction" in row:
                row["point_bias_correction"] = float(row["q50"]) - value \
                    if "q50" in row else 0.0
            applications.append({
                "event_class": "override",
                "event_id": event.event_id,
                "timestamp": row["timestamp"],
                "value": value,
                "boundary_step": index in boundary,
                "before": before,
            })
        for index in steps:
            _assert_monotone(projected[index])
    return projected, applications

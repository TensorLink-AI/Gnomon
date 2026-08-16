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
  rejected. A span may state its number as a multiple of a baseline
  ("4 times the usual withdrawals"): the multiplier is the span's, and
  the baseline is resolved deterministically as the recent-window
  median, with the arithmetic disclosed — still no model-supplied
  number anywhere;
- for constraint events, recent history's relation to the bound is
  recorded as disclosure, never used to reject: the effective window is
  guaranteed to lie entirely after the observed history, so past
  breaches carry no evidence against a forward-scoped claim — an
  announced cap is informative precisely when history breaches it, and
  a bound history already respects changes nothing;
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
stated by the span it was handed and is internally consistent, with its
relation to recent history disclosed in the assessment.

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

from statistics import median

from .constraints import Claim, _align, _assert_monotone, apply_claims
from .context import ContextEvent, event_applies

#: Reserved ``event_type`` namespaces for this lane. ``constraint:`` is
#: shared with the caller-supplied claim path in `constraints.py`; the two
#: do not collide because that path reads the ``claim`` attribute and this
#: one reads ``source_span``.
CONSTRAINT_PREFIX = "constraint:"
OVERRIDE_PREFIX = "override:"
STRUCTURAL_PREFIX = "structural:"

#: Reserved keys inside ``ContextEvent.attributes``.
SOURCE_SPAN_KEY = "source_span"
CLAIMED_BOUND_KEY = "claimed_bound"
CLAIMED_VALUE_KEY = "claimed_value"
EFFECT_KEY = "effect"

#: The closed menu of structural effects an LLM may classify a span
#: into (results/structural-effects/HYPOTHESIS.md). Classification is
#: delegated; numbers never are: every quantity a structural effect
#: applies is derived from Gnomon's own emitted path. One entry per
#: measured instance of the class — the menu grows by census evidence,
#: not by anticipation.
STRUCTURAL_EFFECTS = (
    "trend_ceases",
    "level_matches_seasonal_high",
    "level_matches_seasonal_low",
)

#: The per-phase envelope quantile each regime effect resolves against
#: the observed history (results/seasonal-regime-effects/HYPOTHESIS.md).
REGIME_EFFECT_QUANTILES = {
    "level_matches_seasonal_high": 0.9,
    "level_matches_seasonal_low": 0.1,
}

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

#: A span that says outright it describes a *different* quantity than
#: the one being forecast. Parsing a number says nothing about what the
#: number refers to: "the covariate X_0 takes a value of 0.0553" parses
#: perfectly and must never override the forecast target — measured as
#: the largest single regression in the 2026-08 paired spot-checks
#: (an admitted covariate value applied to the target, 1000× worse than
#: control, disclosed as context_trusted). Curated and deterministic:
#: only wording that names the foreign referent explicitly; spans about
#: other variables that do not say so remain the proposer's
#: entity-scope responsibility.
_FOREIGN_REFERENT = re.compile(
    r"\bcovariates?\b|\bexogenous\b|\bregressors?\b|\binput\s+variables?\b",
    re.IGNORECASE,
)

#: Baseline words a relative multiple may reference. The multiplier is
#: the text's number; the level it scales is resolved deterministically
#: at admission time (the recent-window median) and disclosed — a model
#: never estimates it.
_BASELINE = r"(?:usual|normal|typical|average|baseline)"

#: A stated multiple or percentage of a baseline: "<N> times the (number
#: of) usual <thing>", "<N>% of the usual <thing>". Exactly one of the
#: named groups matches; ``_scale_from`` turns either into a multiplier
#: (percent divides by 100). Percent is admissible here — unlike the
#: bare "90% of capacity" the number guard refuses — because the base is
#: stated: it is the usual level, which admission resolves
#: deterministically.
_SCALED_BASELINE = (
    rf"(?:(?P<times>{_N}){_AFTER_NUMBER}\s*(?:times|x|×)\s+|"
    rf"(?P<pct>{_N})\s*(?:%|percent\b|pct\b)\s+of\s+)"
    rf"(?:the\s+|its\s+|their\s+)?(?:[\w-]+\s+){{0,4}}?{_BASELINE}\b"
)


def _scale_from(match: re.Match) -> float:
    groups = match.groupdict()
    if groups.get("pct") is not None:
        return _to_float(groups["pct"]) / 100.0
    return _to_float(groups["times"])

#: Directional multiples-of-baseline. These must be matched BEFORE the
#: absolute patterns: "will not exceed 3 times the usual level" contains
#: "not exceed 3", which the absolute negated pattern would happily read
#: as an absolute ceiling of 3 — off from the truth by the whole
#: baseline. Region consumption makes the order load-bearing.
_MAX_SCALE_PATTERNS = [
    rf"{_NEGATION}\s+(?:\w+\s+){{0,2}}(?:exceed(?:s|ing)?|surpass(?:es|ing)?|"
    rf"(?:more|greater|higher|larger)\s+than)\s+{_SCALED_BASELINE}",
    rf"(?:at\s+most|no\s+more\s+than|up\s+to|as\s+(?:high|much)\s+as)\s+{_SCALED_BASELINE}",
    rf"(?:stays?|stay(?:ing)?|remains?|remain(?:ing)?|kept?|keeps?)\s+(?:below|under)\s+{_SCALED_BASELINE}",
]

_MIN_SCALE_PATTERNS = [
    rf"{_NEGATION}\s+(?:\w+\s+){{0,2}}(?:below|(?:less|lower|smaller|fewer)\s+than)\s+{_SCALED_BASELINE}",
    rf"(?:at\s+least|no\s+less\s+than)\s+{_SCALED_BASELINE}",
    rf"(?:stays?|stay(?:ing)?|remains?|remain(?:ing)?)\s+above\s+{_SCALED_BASELINE}",
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
    # "a maximum speed of 3000 rpm" — up to three words may name the
    # bounded quantity between the superlative and "of".
    rf"(?:capped?\s+at|cap\s+of|ceiling\s+of|a?\s*max(?:imum|imal)?\s+(?:[\w-]+\s+){{0,3}}?of)\s+({_N}){_AFTER_NUMBER}",
    # Attributive: "the maximal fan speed is 3000 rpm". The superlative
    # names the bounded quantity and the copula states the number; a
    # trailing unit is fine (the number guard only refuses percent and
    # date/clock suffixes). Present/future forms only — "the maximum
    # yesterday was 200" describes history, not a bound.
    rf"(?:maximal|maximum|max|peak|highest(?:\s+possible)?)\s+"
    rf"(?:[\w-]+\s+){{0,4}}?(?:is|are|will\s+be|=|:)\s*({_N}){_AFTER_NUMBER}",
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
    rf"(?:a?\s*min(?:imum|imal)?\s+(?:[\w-]+\s+){{0,3}}?of|floor\s+of)\s+({_N}){_AFTER_NUMBER}",
    # Attributive: "the minimal operating pressure is 12 Pa".
    rf"(?:minimal|minimum|min|lowest(?:\s+possible)?)\s+"
    rf"(?:[\w-]+\s+){{0,4}}?(?:is|are|will\s+be|=|:)\s*({_N}){_AFTER_NUMBER}",
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
    #: Multiples of a baseline the span references but does not state
    #: numerically ("will not exceed 3 times the usual level"). The
    #: multiplier is the span's; admission resolves the baseline
    #: deterministically (recent-window median) and discloses the
    #: arithmetic before anything is applied.
    minimum_scale: float | None = None
    maximum_scale: float | None = None


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
    # Scaled bounds consume their region before the absolute patterns
    # run: "will not exceed 3 times the usual level" must never be read
    # as an absolute ceiling of 3.
    maximum_scale: float | None = None
    minimum_scale: float | None = None
    if maximum is None:
        scale_match = first_match(_MAX_SCALE_PATTERNS)
        if scale_match:
            maximum_scale = _scale_from(scale_match)
    if minimum is None:
        scale_match = first_match(_MIN_SCALE_PATTERNS)
        if scale_match:
            minimum_scale = _scale_from(scale_match)
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
    if minimum is None and maximum is None \
            and minimum_scale is None and maximum_scale is None:
        return None, "the source span does not state a parseable bound"
    if minimum is not None and maximum is not None and minimum > maximum:
        return None, (
            f"the parsed bound is empty: minimum {minimum} exceeds "
            f"maximum {maximum}"
        )
    return ParsedBound(minimum, maximum, minimum_scale, maximum_scale), None


#: Words that state a zero state without a digit. Deliberately short:
#: "directly implied" means a reader could not honestly dispute the value,
#: not that a synonym dictionary voted for it.
_ZERO_STATES = re.compile(
    r"\b(?:offline|closed|closes|closing|shut\s*down|shuts\s*down|halted|"
    r"halts|halt|stopped|stops|suspended|suspends|out\s+of\s+service|"
    # "no <activity>" states a count of zero. Still a curated noun list,
    # never a bare "no \w+": "no change" or "no increase" state that the
    # level is unchanged, which is not a value of 0. The 2026-08 census
    # caught "no withdrawals" slipping through the shorter list.
    r"no\s+(?:production|output|traffic|flow|generation|withdrawals?|"
    r"transactions?|sales|arrivals?|departures?|rides?|trips?|requests?|"
    r"visitors?|customers?|passengers?|calls?|orders?|deliveries|"
    r"operations?|activity|usage|demand|consumption)|zero)\b",
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
    # A stated level change: "it rapidly and smoothly changes to 1593.0".
    # The sensor task families narrate interventions in exactly this
    # voice; the number after the movement verb is the stated new level.
    rf"(?:changes?|changing|shifts?|shifting|switches?|switching|jumps?|"
    rf"jumping|moves?|moving|rises?|rising|climbs?|climbing)\s+to\s+"
    rf"({_N}){_AFTER_NUMBER}",
    # A quoted (timestamp, value) pair — contexts state future points
    # verbatim as tuples; the trailing number is the stated value.
    rf"\(\s*\d{{4}}-\d{{2}}-\d{{2}}[^,()]*,\s*({_N}){_AFTER_NUMBER}\s*\)",
    # "takes a value of 0.2051 from 2028-04-23 to 2028-05-05" — the
    # covariate-narration voice. The parser only vouches that the span
    # states the number; whether the event points at the right series
    # is the proposer's entity_scope, as with every other pattern.
    rf"(?:takes?|taking|assumes?|assuming)\s+(?:on\s+)?(?:a\s+|the\s+)?"
    rf"(?:constant\s+)?value\s+of\s+({_N}){_AFTER_NUMBER}",
    # A bare value with a stated window: "0.2 from 05:34:29 until
    # 05:34:46". The window rides on the event's effective dates; only
    # the level is read here. "by 5 from Monday" is a delta, not a
    # level, and the lookbehind keeps it out.
    rf"(?<!by\s)({_N}){_AFTER_NUMBER}\s+(?:from|between|starting\s+at)\s+"
    rf"(?:\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}:\d{{2}})",
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


def parse_override_scale(span: str) -> tuple[float | None, str | None]:
    """Extract a stated multiple-of-baseline level for an override window.

    "4 times the number of usual withdrawals" states a level as a
    multiple; "10.0% of the usual traffic" states the same thing in
    percent notation (the base is stated — it is the usual level — so
    the bare-percent refusal does not apply). Either way the multiplier
    is the span's number; the baseline it scales is resolved
    deterministically at admission (the recent-window median) and the
    arithmetic is disclosed. A model still never supplies a number that
    is applied.
    """
    text = " ".join(str(span).split())
    match = re.search(_SCALED_BASELINE, text, re.IGNORECASE)
    if match:
        return _scale_from(match), None
    return None, (
        "the source span does not state a multiple or percentage of a "
        "usual/normal/typical level"
    )


# --------------------------------------------------------------------------
# Admission
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FutureEvent:
    """One admitted event, with every number the lane will apply."""

    event_id: str
    event_class: str  # "constraint" | "override" | "structural"
    effective_start: str
    effective_end: str
    source_span: str
    minimum: float | None = None
    maximum: float | None = None
    value: float | None = None
    effect: str | None = None
    #: Regime effects only: the engine-resolved per-future-step target
    #: levels (per-phase envelope quantiles of the observed history),
    #: aligned to the forecast grid. Resolved at admission, like scaled
    #: bounds, so the application stage touches no history.
    levels: tuple[float, ...] | None = None
    #: trend_ceases only: the engine-measured, seasonally adjusted slope
    #: already present in the emitted base path. The text never supplies it.
    slope: float | None = None

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
        elif self.event_class == "structural":
            payload["effect"] = self.effect
            if self.slope is not None:
                payload["resolved_slope_per_step"] = self.slope
            if self.levels is not None:
                payload["resolved_levels"] = list(self.levels)
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
                     passed: bool, *, detail: str | None = None,
                     source_span: str | None = None,
                     data: dict[str, Any] | None = None) -> None:
        entry: dict[str, Any] = {
            "event_id": event.event_id, "event_class": event_class,
            "code": code, "passed": passed,
        }
        if detail:
            entry["detail"] = detail
        if source_span:
            # The span that failed rides with the verdict. Without it a
            # rejection cannot be classified afterwards (parser too
            # narrow, or claim genuinely non-numeric?) and the proposer
            # cannot see which quote to repair — 176 of 220 measured
            # rejections were unclassifiable for exactly this reason.
            entry["source_span"] = source_span
        if data:
            # The quantities the check compared, machine-readable. Same
            # lesson as source_span: a census over 355 runs found 50
            # window_is_future_only rejections that could not be told
            # apart (proposer misdating vs boundary artifact vs timezone
            # reading) because the record kept only prose.
            entry["data"] = data
        self.checks.append(entry)
        if not passed:
            rejection: dict[str, Any] = {
                "event_id": event.event_id, "event_class": event_class,
                "code": code, "reason": detail or code,
            }
            if source_span:
                rejection["source_span"] = source_span
            if data:
                rejection["data"] = data
            self.rejected.append(rejection)

    def class_counts(self) -> dict[str, dict[str, int]]:
        counts = {
            "constraint": {"admitted": 0, "rejected": 0},
            "override": {"admitted": 0, "rejected": 0},
            "structural": {"admitted": 0, "rejected": 0},
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
                "source span, never taken from the proposal; recent "
                "history's relation to a claimed bound disclosed, never "
                "used to reject a forward-scoped claim; fold ablation "
                "deliberately not applicable — these windows have no "
                "historical precedent"
            ),
        }


def _classify(event: ContextEvent) -> str | None:
    if event.event_type.startswith(CONSTRAINT_PREFIX):
        return "constraint"
    if event.event_type.startswith(OVERRIDE_PREFIX):
        return "override"
    if event.event_type.startswith(STRUCTURAL_PREFIX):
        return "structural"
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
    *,
    base_points: list[float] | None = None,
    allow_future: bool = True,
    allow_structural: bool = True,
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
        if event.status == "cancelled":
            assessment.considered = True
            assessment.record_check(
                event, event_class, "status_active", False,
                detail="cancelled event cannot alter the primary forecast",
            )
            continue
        # A class whose flag is off is ignored exactly as an unclassified
        # event is — no record, no rejection — so flag-off behaviour is
        # byte-identical to the class never having existed.
        if event_class == "structural" and not allow_structural:
            continue
        if event_class in ("constraint", "override") and not allow_future:
            continue
        assessment.considered = True

        window = _parse_window(event)
        if window is None:
            assessment.record_check(
                event, event_class, "window_parses", False,
                detail="effective_start/effective_end do not form a window",
            )
            continue
        start, end = window

        # The lane is for the structurally untestable only. A window that
        # touches the observed history could be fold-tested, so it goes to
        # the ablation gate — and if that gate rejected it, rejected it
        # stays. This check is what keeps the lane from becoming a
        # fallback.
        aligned_start, aligned_last = _align(start, last_observed)
        if aligned_start <= aligned_last:
            aligned_end, _ = _align(end, last_observed)
            assessment.record_check(
                event, event_class, "window_is_future_only", False,
                detail=(
                    "the event window overlaps the observed history, so it "
                    "is fold-testable; it must go through the ablation gate, "
                    "not this lane"
                ),
                data={
                    "effective_start": event.effective_start,
                    "effective_end": event.effective_end,
                    "last_observed": last_observed.isoformat(),
                    "overlap_seconds": (
                        aligned_last - aligned_start).total_seconds(),
                    "window_entirely_historical": aligned_end <= aligned_last,
                    "mixed_timezone_alignment": (
                        (start.tzinfo is None) != (last_observed.tzinfo is None)
                    ),
                },
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

        if _FOREIGN_REFERENT.search(span):
            assessment.record_check(
                event, event_class, "span_describes_the_target", False,
                detail=(
                    "the span describes a covariate/exogenous variable, "
                    "not the forecast target; a number parsed from it "
                    "would be applied to the wrong series. Supply future "
                    "covariate values through the covariates lane, where "
                    "they are admitted by leakage-safe ablation."
                ),
                source_span=span,
            )
            continue

        if event_class == "constraint":
            admitted = _admit_constraint(
                assessment, event, span, window_values, window_timestamps,
            )
        elif event_class == "structural":
            admitted = _admit_structural(
                assessment, event, span, values=values, season=season,
                future_count=len(future_timestamps), base_points=base_points,
            )
        else:
            admitted = _admit_override(assessment, event, span, window_values)
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


def _usual_level(recent_values: list[float]) -> float | None:
    """The deterministic reading of "the usual level": the recent-window
    median — robust to the excursions that usually motivate the claim,
    and the same window every other admission check reads."""
    if not recent_values:
        return None
    return float(median(recent_values))


def _resolve_scaled_bound(
    assessment: FutureContextAssessment,
    event: ContextEvent,
    span: str,
    bound: ParsedBound,
    recent_values: list[float],
) -> tuple[ParsedBound | None, dict[str, float]]:
    """Turn a multiple-of-baseline bound into absolute numbers, disclosed.

    Returns the absolute bound and, per side, the span's multiplier (so
    a claimed_bound cross-check can accept either faithful reading). A
    bound with no scaled side passes through untouched.
    """
    if bound.minimum_scale is None and bound.maximum_scale is None:
        return bound, {}
    baseline = _usual_level(recent_values)
    if baseline is None or baseline <= 0:
        described = "absent" if baseline is None else f"{baseline:g}"
        assessment.record_check(
            event, "constraint", "baseline_resolvable", False,
            detail=(
                f"the span states a multiple of the usual level, but the "
                f"recent-window median is {described}; scaling a "
                f"non-positive baseline would invert or degenerate the "
                f"bound, so the claim is not applied"
            ),
            source_span=span,
        )
        return None, {}
    minimum, maximum = bound.minimum, bound.maximum
    resolved: dict[str, float] = {}
    arithmetic: list[str] = []
    if bound.minimum_scale is not None:
        minimum = bound.minimum_scale * baseline
        resolved["min"] = bound.minimum_scale
        arithmetic.append(
            f"min = {bound.minimum_scale:g} × {baseline:g} = {minimum:g}")
    if bound.maximum_scale is not None:
        maximum = bound.maximum_scale * baseline
        resolved["max"] = bound.maximum_scale
        arithmetic.append(
            f"max = {bound.maximum_scale:g} × {baseline:g} = {maximum:g}")
    if minimum is not None and maximum is not None and minimum > maximum:
        assessment.record_check(
            event, "constraint", "baseline_resolvable", False,
            detail=(
                f"resolving the span's multiplier against the recent-window "
                f"median ({baseline:g}) yields an empty bound: minimum "
                f"{minimum:g} exceeds maximum {maximum:g}"
            ),
            source_span=span,
        )
        return None, {}
    assessment.record_check(
        event, "constraint", "relative_bound_resolved", True,
        detail=(
            f"the span's multiplier scaled the recent-window median "
            f"({baseline:g}): {'; '.join(arithmetic)}. The multiplier is "
            f"the text's; the baseline is Gnomon's statistic, never a "
            f"model's estimate"
        ),
        source_span=span,
    )
    return ParsedBound(minimum, maximum), resolved


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
            event, "constraint", "span_states_the_bound", False,
            detail=problem, source_span=span,
        )
        return None

    bound, resolved = _resolve_scaled_bound(
        assessment, event, span, bound, recent_values,
    )
    if bound is None:
        return None

    claimed = (event.attributes or {}).get(CLAIMED_BOUND_KEY)
    if isinstance(claimed, dict):
        for side, parsed_side in (("min", bound.minimum), ("max", bound.maximum)):
            if side not in claimed:
                continue
            claimed_value = _claimed_number(claimed.get(side))
            # For a scaled bound the proposal may faithfully claim either
            # the resolved value or the span's multiplier; both are the
            # same reading of the same text.
            acceptable = [parsed_side, resolved.get(side)]
            if claimed_value is None or not any(
                target is not None and abs(claimed_value - target) <= 1e-9
                for target in acceptable
            ):
                assessment.record_check(
                    event, "constraint", "claim_matches_span", False,
                    detail=(
                        f"the proposal claims {side}={claimed.get(side)!r} but "
                        f"the span parses to {side}={parsed_side}; the span is "
                        f"the only admissible source of numbers"
                    ),
                )
                return None

    # History's relation to the bound is disclosure, not a gate. The
    # window_is_future_only check has already guaranteed the window lies
    # entirely after the observed history, so past breaches are no
    # evidence against a forward-scoped claim: an announced cap is
    # informative precisely when history breaches it, and a bound history
    # already respects changes nothing. Rejecting on breach inverted the
    # lane — it admitted only the uninformative.
    breaches = sum(
        1 for value in recent_values
        if (bound.minimum is not None and value < bound.minimum)
        or (bound.maximum is not None and value > bound.maximum)
    )
    if breaches:
        detail = (
            f"recent history breaches the claimed bound "
            f"[{bound.minimum}, {bound.maximum}] at {breaches} of "
            f"{len(recent_values)} recent points; the window is entirely "
            f"in the future, so the bound is admitted and expected to bind"
        )
    else:
        detail = (
            f"recent history already respects the claimed bound "
            f"[{bound.minimum}, {bound.maximum}]; admitted, expected to "
            f"bind weakly if at all"
        )
    assessment.record_check(
        event, "constraint", "history_relation_disclosed", True, detail=detail,
    )

    return FutureEvent(
        event.event_id, "constraint", event.effective_start,
        event.effective_end, span,
        minimum=bound.minimum, maximum=bound.maximum,
    )


def _admit_override(
    assessment: FutureContextAssessment,
    event: ContextEvent,
    span: str,
    recent_values: list[float],
) -> FutureEvent | None:
    value, problem = parse_override_span(span)
    scale: float | None = None
    if value is None:
        scale, _ = parse_override_scale(span)
        if scale is not None:
            baseline = _usual_level(recent_values)
            if baseline is None or baseline <= 0:
                described = "absent" if baseline is None else f"{baseline:g}"
                assessment.record_check(
                    event, "override", "baseline_resolvable", False,
                    detail=(
                        f"the span states a multiple of the usual level, "
                        f"but the recent-window median is {described}; "
                        f"scaling a non-positive baseline would invert the "
                        f"stated level, so the claim is not applied"
                    ),
                    source_span=span,
                )
                return None
            value = scale * baseline
            assessment.record_check(
                event, "override", "relative_value_resolved", True,
                detail=(
                    f"the span's multiplier scaled the recent-window median: "
                    f"{scale:g} × {baseline:g} = {value:g}. The multiplier "
                    f"is the text's; the baseline is Gnomon's statistic, "
                    f"never a model's estimate"
                ),
                source_span=span,
            )
    if value is None:
        assessment.record_check(
            event, "override", "span_states_the_value", False,
            detail=problem, source_span=span,
        )
        return None
    claimed_raw = (event.attributes or {}).get(CLAIMED_VALUE_KEY)
    if claimed_raw is not None:
        claimed = _claimed_number(claimed_raw)
        acceptable = [value] + ([scale] if scale is not None else [])
        if claimed is None or not any(
            abs(claimed - target) <= 1e-9 for target in acceptable
        ):
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


def _seasonal_envelope(
    values: list[float], season: int, probability: float,
) -> list[float] | None:
    """Per-phase quantile of the observed history; None when the history
    cannot support a profile (no season, or fewer than two full cycles).
    Deliberately no fallback to a global quantile: a profile the history
    cannot support is not resolved from somewhere else."""
    from .evaluation import quantile

    if season < 2 or len(values) < 2 * season:
        return None
    return [
        float(quantile(values[phase::season], probability))
        for phase in range(season)
    ]


def _resolved_emitted_trend(
    values: list[float], season: int, base_points: list[float],
) -> tuple[float, float, float] | None:
    """Return historical slope, emitted slope, and directional agreement.

    Both slopes are estimated after phase fixed effects are removed. This is
    what prevents a rising fragment of an ordinary seasonal wave from being
    mistaken for a trend that a structural claim is entitled to erase.
    """
    if len(values) < max(8, 2 * max(season, 1)) or len(base_points) < 2:
        return None
    period = max(1, season)
    phase_x: list[list[float]] = [[] for _ in range(period)]
    phase_y: list[list[float]] = [[] for _ in range(period)]
    for index, value in enumerate(values):
        phase = index % period
        phase_x[phase].append(float(index))
        phase_y[phase].append(float(value))
    numerator = denominator = 0.0
    for xs, ys in zip(phase_x, phase_y):
        if not xs:
            continue
        x_mean, y_mean = sum(xs) / len(xs), sum(ys) / len(ys)
        numerator += sum((x - x_mean) * (y - y_mean)
                         for x, y in zip(xs, ys))
        denominator += sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 1e-12:
        return None
    historical_slope = numerator / denominator
    phase_levels: list[float] = []
    for phase, ys in enumerate(phase_y):
        adjusted = [value - historical_slope * index
                    for index, value in enumerate(values)
                    if index % period == phase]
        phase_levels.append(sum(adjusted) / len(adjusted))
    emitted_deseasonalized = [
        float(point) - phase_levels[(len(values) + step) % period]
        for step, point in enumerate(base_points)
    ]
    emitted_slope = _path_slope(emitted_deseasonalized)
    differences = [right - left for left, right in
                   zip(emitted_deseasonalized, emitted_deseasonalized[1:])]
    directional = [delta for delta in differences if abs(delta) > 1e-12]
    agreement = (sum((delta > 0) == (emitted_slope > 0)
                     for delta in directional) / len(directional)
                 if directional and abs(emitted_slope) > 1e-12 else 0.0)
    return historical_slope, emitted_slope, agreement


def _admit_structural(
    assessment: FutureContextAssessment,
    event: ContextEvent,
    span: str,
    *,
    values: list[float],
    season: int,
    future_count: int,
    base_points: list[float] | None = None,
) -> FutureEvent | None:
    """Admit an LLM-classified structural event from the closed menu.

    There is deliberately no span parse here: the class carries no
    number, and classification — which phrasing of a concept the span
    is — is exactly what is delegated to the model
    (results/structural-effects/HYPOTHESIS.md). What stays checkable is
    checked: the effect must come from the closed menu, and every
    quantity the effect applies is later derived from Gnomon's own
    emitted path, never from the proposal.
    """
    effect = (event.attributes or {}).get(EFFECT_KEY)
    if effect not in STRUCTURAL_EFFECTS:
        assessment.record_check(
            event, "structural", "effect_supported", False,
            detail=(
                f"effect {effect!r} is not in the closed menu "
                f"({', '.join(STRUCTURAL_EFFECTS)}); a structural event "
                f"must classify its span into a supported effect"
            ),
            source_span=span,
        )
        return None
    levels: tuple[float, ...] | None = None
    resolved_slope: float | None = None
    if effect == "trend_ceases" and base_points is not None:
        trend = _resolved_emitted_trend(values, season, base_points)
        historical_slope, emitted_slope, agreement = (
            trend if trend is not None else (0.0, 0.0, 0.0)
        )
        # A short seasonal arc can have a non-zero OLS slope even though it
        # is not a continuing trend. Only flatten a path whose step changes
        # predominantly agree with its fitted direction. Ambiguous paths
        # remain the history-only primary rather than turning a textual
        # structural claim into an invented seasonal adjustment.
        same_direction = historical_slope * emitted_slope > 0
        magnitude_ratio = (abs(emitted_slope / historical_slope)
                           if abs(historical_slope) > 1e-12 else 0.0)
        if (agreement < 0.75 or not same_direction
                or not 0.25 <= magnitude_ratio <= 4.0):
            assessment.record_check(
                event, "structural", "emitted_trend_is_directionally_stable",
                False,
                detail=(
                    "the emitted path does not contain a stable continuation "
                    "of the seasonally adjusted historical trend "
                    f"(agreement={agreement:.1%}, historical slope="
                    f"{historical_slope:.6g}, emitted slope="
                    f"{emitted_slope:.6g}, ratio={magnitude_ratio:.3g}); "
                    "at least 75% directional agreement and a same-direction "
                    "magnitude ratio in [0.25, 4] are required"
                ),
                source_span=span,
                data={"historical_slope_per_step": historical_slope,
                      "emitted_slope_per_step": emitted_slope,
                      "directional_agreement": agreement,
                      "magnitude_ratio": magnitude_ratio,
                      "agreement_threshold": 0.75,
                      "magnitude_ratio_bounds": [0.25, 4.0]},
            )
            return None
        resolved_slope = emitted_slope
    if effect in REGIME_EFFECT_QUANTILES:
        # Resolved at admission, like scaled bounds: the model named
        # which part of the history the future resembles; the numbers
        # are quantiles of the engine's own observed data.
        profile = _seasonal_envelope(
            values, season, REGIME_EFFECT_QUANTILES[effect])
        if profile is None:
            assessment.record_check(
                event, "structural", "seasonal_profile_resolvable", False,
                detail=(
                    f"a regime effect needs a detected season of at least 2 "
                    f"and two full observed cycles to resolve the per-phase "
                    f"envelope (season={season}, observations={len(values)})"
                ),
                source_span=span,
            )
            return None
        levels = tuple(
            profile[(len(values) + step) % season]
            for step in range(future_count)
        )
    return FutureEvent(
        event.event_id, "structural", event.effective_start,
        event.effective_end, span, effect=str(effect), levels=levels,
        slope=resolved_slope,
    )


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------

def _path_slope(points: list[float]) -> float:
    """Least-squares slope of the emitted point path, per step."""
    n = len(points)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = sum(points) / n
    denominator = sum((index - x_mean) ** 2 for index in range(n))
    if denominator <= 0:
        return 0.0
    return sum((index - x_mean) * (point - y_mean)
               for index, point in enumerate(points)) / denominator


def _apply_structural(
    rows: list[dict[str, Any]],
    events: list[FutureEvent],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the structural menu; every applied number is engine-derived.

    ``trend_ceases``: the slope is fitted to the path Gnomon already
    produced — never to the proposal, never to a model's say-so — and
    each covered step k has slope × (k − k₀) subtracted from its point
    and every quantile. The first covered step keeps its value
    (continuity), and a driftless path makes the effect a measured
    no-op.

    ``level_matches_seasonal_high`` / ``low``: each covered step lands
    on its phase's envelope level (per-phase q90/q10 of the observed
    history, resolved at admission). The level *jump* at the window
    edge is the claimed semantics — a stated regime change, unlike a
    cessation, is discontinuous by nature.

    Both are pure per-step location shifts: interval widths, quantile
    ordering, and the point-to-median gap are untouched. Steps already
    adjusted by one event are skipped by later ones, so overlapping
    structural events cannot adjust a step twice.
    """
    points = [float(row["point"]) for row in rows]
    slope = _path_slope(points)
    adjusted = [dict(row) for row in rows]
    applications: list[dict[str, Any]] = []
    touched: set[int] = set()
    for event in events:
        steps = [index for index in _covered_steps(event, adjusted)
                 if index not in touched]
        if not steps:
            continue
        first = steps[0]
        for index in steps:
            row = adjusted[index]
            before = {"point": row.get("point")}
            record: dict[str, Any] = {
                "event_class": "structural",
                "event_id": event.event_id,
                "effect": event.effect,
                "timestamp": row["timestamp"],
                "before": before,
            }
            if event.effect in REGIME_EFFECT_QUANTILES:
                # The covered step lands on its phase's envelope level;
                # quantiles translate with it, so the interval width is
                # the engine's own, relocated. Levels were resolved at
                # admission and are aligned to the forecast grid.
                if event.levels is None or index >= len(event.levels):
                    continue
                delta = float(event.levels[index]) - float(row["point"])
                record["target_level"] = float(event.levels[index])
            else:
                # trend_ceases: remove the emitted path's own drift from
                # the first covered step onward (continuity preserved).
                slope_removed = event.slope if event.slope is not None else slope
                delta = -slope_removed * (index - first)
                record["slope_removed"] = slope_removed
            row["point"] = float(row["point"]) + delta
            for _, key in _quantile_keys(row):
                row[key] = float(row[key]) + delta
            touched.add(index)
            record["delta"] = delta
            applications.append(record)
            _assert_monotone(row)
    return adjusted, applications


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

    structural = [event for event in admitted
                  if event.event_class == "structural"]
    if structural:
        # Structural effects reshape the base path first, so a stated
        # bound still clamps the result and a stated override still
        # wins inside its window.
        rows, structural_applications = _apply_structural(rows, structural)
        applications.extend(structural_applications)

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

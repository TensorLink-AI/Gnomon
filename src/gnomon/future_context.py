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

#: A number with optional sign, thousands separators, and decimals.
_NUMBER = r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?"
_N = f"(?:{_NUMBER})"


def _to_float(text: str) -> float:
    return float(text.replace(",", ""))


#: (pattern, handler) pairs, tried in order; every match contributes to the
#: bound, so "between 0 and 340" yields both sides from one pattern while
#: "at least 5 and will not exceed 60" yields one side from each.
_RANGE_PATTERNS = [
    rf"(?:between|from)\s+({_N})\s+(?:and|to|through)\s+({_N})",
    rf"(?:within|in)\s+(?:the\s+)?range\s+(?:of\s+)?({_N})\s+(?:and|to|through)\s+({_N})",
    rf"range\s+of\s+({_N})\s+(?:and|to)\s+({_N})",
    # Interval notation: "in [0, 340]" / "within (0, 340)".
    rf"(?:in|within)\s+[\[\(]\s*({_N})\s*,\s*({_N})\s*[\]\)]",
]

_MAX_PATTERNS = [
    rf"(?:cannot|can\s*not|can't|will\s+not|won't|shall\s+not|must\s+not|does\s+not|never)\s+"
    rf"(?:exceed|surpass|go\s+(?:above|over|past)|rise\s+above|be\s+(?:more|greater|higher|larger)\s+than)\s+({_N})",
    rf"(?:not|never)\s+(?:to\s+)?exceed(?:ing)?\s+({_N})",
    # CiK's constraint tasks state bounds in exactly this voice.
    rf"bounded\s+(?:above|from\s+above)\s+by\s+({_N})",
    rf"(?:less\s+than\s+or\s+equal\s+to|at\s+or\s+below|smaller\s+than\s+or\s+equal\s+to)\s+({_N})",
    rf"(?:at\s+most|no\s+more\s+than|not\s+more\s+than|no\s+greater\s+than|no\s+higher\s+than)\s+({_N})",
    rf"(?:stays?|stay(?:ing)?|remains?|remain(?:ing)?|kept?|keeps?)\s+(?:below|under|at\s+or\s+below)\s+({_N})",
    # Bare "below X" is a max only when it is not the tail of a min phrase
    # like "will not drop below X"; the lookbehinds keep it out of those.
    rf"(?<!drop\s)(?<!drops\s)(?<!fall\s)(?<!falls\s)(?<!dip\s)(?<!dips\s)"
    rf"(?<!go\s)(?<!goes\s)(?<!not\s)(?:below|under|less\s+than)\s+({_N})",
    rf"(?:capped?\s+at|cap\s+of|ceiling\s+of|a?\s*maximum\s+(?:value\s+)?of|max(?:imum)?\s+of)\s+({_N})",
    rf"(?:<=|≤)\s*({_N})",
]

_MIN_PATTERNS = [
    rf"(?:cannot|can\s*not|can't|will\s+not|won't|shall\s+not|must\s+not|does\s+not|never)\s+"
    rf"(?:fall|drop|go|dip)\s+below\s+({_N})",
    rf"bounded\s+(?:below|from\s+below)\s+by\s+({_N})",
    rf"(?:greater\s+than\s+or\s+equal\s+to|at\s+or\s+above|larger\s+than\s+or\s+equal\s+to)\s+({_N})",
    rf"(?:at\s+least|no\s+less\s+than|not\s+less\s+than|no\s+lower\s+than)\s+({_N})",
    rf"(?:stays?|stay(?:ing)?|remains?|remain(?:ing)?|kept?|keeps?)\s+(?:above|at\s+or\s+above)\s+({_N})",
    # Bare "above X" is a min only when it is not the tail of a max phrase
    # like "will not rise above X".
    rf"(?<!rise\s)(?<!rises\s)(?<!go\s)(?<!goes\s)(?<!not\s)"
    rf"(?:above|over|more\s+than|greater\s+than)\s+({_N})",
    rf"(?:a?\s*minimum\s+(?:value\s+)?of|min(?:imum)?\s+of|floor\s+of)\s+({_N})",
    rf"(?:>=|≥)\s*({_N})",
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
    """
    text = " ".join(str(span).split())
    minimum: float | None = None
    maximum: float | None = None
    for pattern in _RANGE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            low, high = sorted((_to_float(match.group(1)), _to_float(match.group(2))))
            minimum, maximum = low, high
            break
    if maximum is None:
        for pattern in _MAX_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                maximum = _to_float(match.group(1))
                break
    if minimum is None:
        for pattern in _MIN_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                minimum = _to_float(match.group(1))
                break
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
    rf"(?:set|fixed|held|pinned|kept)\s+(?:to|at)\s+({_N})",
    rf"(?:will|shall)\s+be\s+({_N})\b",
    rf"(?:drops?|falls?|goes?|go|reduced?)\s+to\s+({_N})",
    rf"(?:at\s+a\s+(?:constant\s+)?(?:value|rate|level)\s+of)\s+({_N})",
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
        start, end = window

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
        boundary = {steps[0], steps[-1]}
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

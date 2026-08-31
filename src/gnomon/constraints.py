"""Typed numeric claims carried by context events.

A caller often knows something about the *domain* that no amount of history
reveals: returns cannot be negative, the warehouse holds 4,000 units, the
line is capped at 500 an hour. That is a statement about what values are
feasible, not a statement about what the forecast should be, and Gnomon can
use it without any model asserting a number.

The distinction is the whole design:

- **A bound is admissible.** ``min`` and ``max`` claims are applied as a
  projection of the already-computed quantiles onto the feasible region.
  Clamping is monotone, so it cannot reorder q10/q50/q90; it is idempotent;
  and it can only ever move a number toward a region the caller says is
  possible. The forecast is still Gnomon's.
- **A pinned value is not.** A claim of the form "the value will be 412 on
  Tuesday" is an assertion of the answer, and admitting it would make an
  LLM's say-so into a forecast number — the one thing the system exists not
  to do. Such claims are rejected by the schema, with that reason.

Every claim is checked against history before use: a ``min`` that the
training window already breaches is describing a different quantity, or is
wrong, and either way applying it would silently distort the answer. It is
rejected and the violating timestamps are named.

Claims live in the existing free-form ``ContextEvent.attributes`` under a
reserved ``claim`` key, and use the reserved ``constraint:`` ``event_type``
namespace. No new dataclass, no change to the frozen tool surface: an event
without a claim behaves exactly as it did.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .context import ContextEvent, event_applies
from .contracts import GnomonError

#: Reserved key inside `ContextEvent.attributes`.
CLAIM_KEY = "claim"

#: Reserved `event_type` namespaces.
CONSTRAINT_PREFIX = "constraint:"

#: Claim kinds that bound the feasible region. Deliberately excludes any
#: kind that would supply a value rather than a limit — see the module
#: docstring.
BOUND_KINDS = ("min", "max")


def _align(bound: datetime, timestamp: datetime) -> tuple[datetime, datetime]:
    """Make an event bound and a dataset timestamp comparable.

    Context events are required to carry an explicit offset; most datasets
    are timezone-naive, including every example this repo ships. Comparing
    the two raised a bare ``TypeError`` out of the forecast, which made the
    whole context-events feature unreachable for Gnomon's own examples.

    Mixed awareness is resolved on **wall-clock time**: an event that says
    it starts at 09:00 is taken to mean 09:00 as the dataset records time.
    That is the only reading available without knowing the dataset's zone,
    and it is disclosed rather than assumed silently — see
    ``pipeline._constraint_stage``.
    """
    if (bound.tzinfo is None) == (timestamp.tzinfo is None):
        return bound, timestamp
    return bound.replace(tzinfo=None), timestamp.replace(tzinfo=None)


@dataclass(frozen=True)
class Claim:
    event_id: str
    kind: str
    value: float
    effective_start: str
    effective_end: str

    def binds(self, timestamp: datetime) -> bool:
        start, moment = _align(datetime.fromisoformat(self.effective_start), timestamp)
        end, _ = _align(datetime.fromisoformat(self.effective_end), timestamp)
        return start <= moment <= end

    def needs_alignment(self, timestamp: datetime) -> bool:
        """Whether comparing against ``timestamp`` required dropping a zone."""
        start = datetime.fromisoformat(self.effective_start)
        return (start.tzinfo is None) != (timestamp.tzinfo is None)

    def project(self, value: float) -> float:
        return max(value, self.value) if self.kind == "min" else min(value, self.value)


def parse_claim(event: ContextEvent) -> tuple[Claim | None, str | None]:
    """Extract a typed claim from an event, or say why there is none."""
    raw = (event.attributes or {}).get(CLAIM_KEY)
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        return None, f"{CLAIM_KEY!r} must be an object"
    kind = str(raw.get("kind", ""))
    if kind not in BOUND_KINDS:
        if kind in ("pin", "exact", "equals"):
            return None, (
                f"claim kind {kind!r} supplies a value rather than a bound; "
                f"a forecast value asserted by a caller is not admissible. "
                f"Supply it as a covariate instead, where Gnomon estimates its "
                f"coefficient on identical folds."
            )
        return None, (f"claim kind {kind!r} is not supported "
                      f"(expected one of {', '.join(BOUND_KINDS)})")
    try:
        value = float(raw["value"])
    except (KeyError, TypeError, ValueError):
        return None, "claim must carry a numeric 'value'"
    if value != value or value in (float("inf"), float("-inf")):
        return None, "claim 'value' must be finite"
    return Claim(event.event_id, kind, value,
                 event.effective_start, event.effective_end), None


def history_violations(
    claim: Claim, values: list[float], timestamps: list[datetime],
    limit: int = 5,
) -> list[str]:
    """Timestamps inside the claim's own window where history breaches it.

    A bound the training data already violates is not a bound on this
    quantity. Applying it would move the forecast into a region the series
    demonstrably leaves, so the claim is rejected rather than enforced.
    """
    violations: list[str] = []
    for value, timestamp in zip(values, timestamps):
        if not claim.binds(timestamp):
            continue
        breached = value < claim.value if claim.kind == "min" else value > claim.value
        if breached:
            violations.append(timestamp.isoformat())
            if len(violations) >= limit:
                break
    return violations


def collect_claims(
    events: list[ContextEvent], series_name: str,
    values: list[float], timestamps: list[datetime],
) -> tuple[list[Claim], list[dict[str, Any]]]:
    """Usable claims for this series, and a record of every rejection."""
    claims: list[Claim] = []
    rejected: list[dict[str, Any]] = []
    for event in events:
        if not event_applies(event, series_name):
            continue
        claim, problem = parse_claim(event)
        if problem:
            rejected.append({"event_id": event.event_id, "reason": problem})
            continue
        if claim is None:
            continue
        violations = history_violations(claim, values, timestamps)
        if violations:
            rejected.append({
                "event_id": event.event_id,
                "reason": f"history breaches this {claim.kind} bound of "
                          f"{claim.value} within the claim's own window",
                "violating_timestamps": violations,
            })
            continue
        claims.append(claim)
    return claims, rejected


def apply_claims(
    rows: list[dict[str, Any]], claims: list[Claim],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project forecast rows onto the feasible region.

    Applied to the emitted quantiles, never to the point path before
    interval construction: a bound is a statement about which values are
    possible, so it belongs after the model has said what it believes, not
    inside the belief. Clamping is monotone and therefore preserves
    q10 <= point <= q90; it is idempotent by construction.

    *Every* emitted quantile is projected, not a fixed list of the three
    legacy columns. Clamping four of ten levels left a caller's declared
    `max` violated by q95 and produced a crossed distribution (q80 > q90)
    that then propagated into `monitor` and `decide`, which read these rows.
    """
    if not claims or not rows:
        return rows, []
    projected: list[dict[str, Any]] = []
    applications: list[dict[str, Any]] = []
    for row in rows:
        timestamp = datetime.fromisoformat(str(row["timestamp"]))
        updated = dict(row)
        row_changed = False
        for claim in claims:
            if not claim.binds(timestamp):
                continue
            changed: dict[str, float] = {}
            for field in _projectable_fields(updated):
                before = float(updated[field])
                after = claim.project(before)
                if after != before:
                    changed[field] = before
                    updated[field] = after
            if changed:
                row_changed = True
                applications.append({
                    "event_id": claim.event_id,
                    "kind": claim.kind,
                    "value": claim.value,
                    "timestamp": row["timestamp"],
                    "before": changed,
                })
        if row_changed and {"point", "q50", "point_bias_correction"} <= updated.keys():
            # The row promises pbc == q50 - point; a clamp that moves one
            # of them without restating the gap ships a row that lies
            # about itself.
            updated["point_bias_correction"] = (
                float(updated["q50"]) - float(updated["point"])
            )
        _assert_monotone(updated)
        projected.append(updated)
    return projected, applications


def _projectable_fields(row: dict[str, Any]) -> list[str]:
    """Every numeric forecast column in a row: `point` and all `q*` levels.

    Derived from the row rather than hard-coded, so a quantile level added
    later is projected the day it is emitted instead of the day someone
    remembers to extend a tuple.
    """
    fields = ["point"] if "point" in row else []
    fields.extend(sorted(
        key for key in row
        if key.startswith("q") and key[1:].isdigit()
    ))
    return fields


def _assert_monotone(row: dict[str, Any]) -> None:
    """Quantile levels must not cross after projection.

    Projection onto an interval is monotone, so this cannot fail for a
    well-formed row; it is here because the crossing it guards against
    (q80 > q90 beside a clamped q90) shipped once already.
    """
    levels = sorted(
        (int(key[1:]), key) for key in row
        if key.startswith("q") and key[1:].isdigit()
    )
    previous_value: float | None = None
    previous_key: str | None = None
    for _, key in levels:
        value = float(row[key])
        if previous_value is not None and value < previous_value - 1e-9:
            raise GnomonError(
                "QUANTILE_CROSSING",
                f"Projected quantiles cross: {previous_key}={previous_value} "
                f"exceeds {key}={value}.",
                {"timestamp": row.get("timestamp"), "row": dict(row)},
            )
        previous_value, previous_key = value, key

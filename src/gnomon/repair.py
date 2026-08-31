"""Disclosed, deterministic repair of messy input data.

Real-world CSVs arrive with currency symbols, mixed date formats, duplicate
rows, sentinel "N/A" cells, jittered timestamps, and gaps. Gnomon's contract
is honesty, not fragility: repairs are allowed, but only under three rules.

1. **Repairs fire only where the strict path would fail.** A file that
   parses cleanly today is untouched — byte-identical output, same IDs.
2. **Every repair is disclosed.** Each fix is a typed :class:`RepairAction`
   collected in a :class:`RepairLog`; the forecast artifact carries them as
   a ``data_repair`` evidence record, and assumptive repairs additionally
   become series warnings so support degrades honestly.
3. **Messiness has a ceiling.** Assumptive repairs are capped; past the cap
   the honest answer is ``EXCESSIVE_REPAIR``, not a forecast built on a
   dataset Gnomon mostly invented.

Two levels above ``off``:

- ``safe`` (the default): reinterprets cell *text* and aligns bounded
  scheduler/scrape jitter — date formats, currency/thousands separators,
  percent signs, sentinel missing values, fully blank rows, byte-identical
  duplicate rows, and timestamps within 1% of a deterministic grid (capped
  at 60 seconds). It never invents a value, fills a gap, merges observations,
  or drops a data point.
- ``aggressive`` (opt-in): structural fixes — interior gap interpolation,
  conflicting-duplicate resolution (last row wins), dropping unparseable
  rows, coercing naive timestamps in a mixed-timezone file to UTC, plus the
  same bounded timestamp alignment as safe. All capped and disclosed.

Everything here is a deterministic function of the input bytes.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import TYPE_CHECKING

from .contracts import GnomonError

if TYPE_CHECKING:  # pragma: no cover - import cycle guard (data imports repair)
    from .data import Observation

REPAIR_OFF = "off"
REPAIR_SAFE = "safe"
REPAIR_AGGRESSIVE = "aggressive"
REPAIR_LEVELS = (REPAIR_OFF, REPAIR_SAFE, REPAIR_AGGRESSIVE)

# Fraction of a series whose *values* assumptive repair may invent or choose
# (fills and conflict resolutions) before the honest answer is "fix the
# data". Bounded timestamp alignment is disclosed but does not invent a
# measurement and therefore is not charged to this ceiling.
MAX_ASSUMPTIVE_FRACTION = 0.30
# Fraction of rows that may be dropped as unparseable under aggressive repair.
MAX_DROPPED_FRACTION = 0.05

# Scheduler and scrape jitter is bounded relative to the observed cadence.
# The absolute cap prevents a long cadence from turning "alignment" into a
# broad restamping authority (1% of a day would otherwise be 14.4 minutes).
JITTER_TOLERANCE_FRACTION = 0.01
MAX_JITTER_TOLERANCE_SECONDS = 60.0

MISSING_SENTINELS = frozenset({
    "", "na", "n/a", "n.a.", "nan", "null", "none", "nil", "-", "--",
    "missing", "?", "#n/a", "#value!", "#ref!", "#div/0!",
})

_CURRENCY = "$€£¥₹"


@dataclass(frozen=True)
class RepairAction:
    """One kind of repair applied to one series (or the whole file)."""

    code: str
    series: str | None
    count: int
    assumptive: bool
    detail: str
    examples: tuple[str, ...]
    metrics: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["examples"] = list(self.examples)
        if payload["metrics"] is None:
            payload.pop("metrics")
        return payload


class RepairLog:
    """Accumulates repair actions; the single source of disclosure."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str | None], dict[str, object]] = {}

    def record(
        self, code: str, detail: str, *,
        series: str | None = None, assumptive: bool = False,
        example: str | None = None, count: int = 1,
        metrics: dict[str, object] | None = None,
    ) -> None:
        entry = self._entries.setdefault(
            (code, series),
            {"assumptive": assumptive, "detail": detail, "count": 0,
             "examples": [], "metrics": metrics},
        )
        entry["count"] = int(entry["count"]) + count
        if metrics is not None:
            entry["metrics"] = dict(metrics)
        examples = entry["examples"]
        if example is not None and isinstance(examples, list) and len(examples) < 3:
            examples.append(example)

    def clone(self) -> "RepairLog":
        """An independent copy. A multi-target run records the shared
        file-level reads once, then forks a log per target so each
        column's repairs are disclosed on that column alone."""
        copy = RepairLog()
        for key, entry in self._entries.items():
            metrics = entry.get("metrics")
            copy._entries[key] = {
                **entry,
                "examples": list(entry["examples"]),
                "metrics": dict(metrics) if isinstance(metrics, dict) else None,
            }
        return copy

    def has_actions(self) -> bool:
        return bool(self._entries)

    def actions(self) -> list[RepairAction]:
        return [
            RepairAction(code, series, int(entry["count"]), bool(entry["assumptive"]),
                         str(entry["detail"]), tuple(entry["examples"]),
                         (dict(entry["metrics"])
                          if isinstance(entry.get("metrics"), dict) else None))  # type: ignore[arg-type]
            for (code, series), entry in sorted(
                self._entries.items(), key=lambda item: (item[0][0], item[0][1] or "")
            )
        ]

    def warnings_for(self, series: str) -> list[str]:
        """Assumptive repairs become warnings on the series they touched;
        file-level assumptive repairs warn on every series."""
        return [
            f"repaired_data: {action.code} x{action.count} — {action.detail}"
            for action in self.actions()
            if action.assumptive and action.series in (None, series)
        ]

    def summary(self) -> dict[str, object]:
        return {"actions": [action.to_dict() for action in self.actions()]}


# --- numeric leniency -------------------------------------------------------

_GROUPED_COMMA = re.compile(r"-?\d{1,3}(,\d{3})+(\.\d+)?")
_GROUPED_SPACE = re.compile(r"-?\d{1,3}([ '  ]\d{3})+([.,]\d+)?")
_GROUP_SEPARATORS = re.compile("[ '  ]")


def _strip_decoration(text: str) -> tuple[str, bool, bool]:
    """Remove currency symbols, percent signs, and accounting parentheses.
    Returns (stripped, negated, decorated)."""
    s = text.strip()
    decorated = False
    negated = False
    if len(s) >= 2 and s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
        negated = decorated = True
    if s.endswith("%"):
        s = s[:-1].strip()
        decorated = True
    while s and s[0] in _CURRENCY:
        s = s[1:].strip()
        decorated = True
    while s and s[-1] in _CURRENCY:
        s = s[:-1].strip()
        decorated = True
    return s, negated, decorated


def scan_numeric_evidence(texts: list[str]) -> str | None:
    """Decide the column-wide role of the comma from unambiguous cells.

    A value like ``1,234.5`` proves comma-as-thousands; a value like
    ``12,34`` (groups that are not exactly three digits, no dot) proves
    comma-as-decimal. Conflicting evidence yields ``None`` — per-cell rules
    then apply, with the grouped-comma case flagged as assumptive."""
    thousands = decimal = False
    for text in texts:
        s, _, _ = _strip_decoration(text)
        if "," not in s:
            continue
        if "." in s:
            if s.rfind(".") > s.rfind(","):
                thousands = True
            else:
                decimal = True
        elif not _GROUPED_COMMA.fullmatch(s):
            decimal = True
    if thousands and not decimal:
        return "thousands"
    if decimal and not thousands:
        return "decimal"
    return None


def parse_number(text: str, comma_role: str | None) -> tuple[float | None, str]:
    """Parse a numeric cell leniently.

    Returns ``(value, tier)`` where tier is ``clean`` (strict float),
    ``normalised`` (deterministic reinterpretation), or ``assumptive``
    (grouped comma with no column evidence, read as thousands); a sentinel
    missing cell returns ``(None, "missing")``. Raises ``ValueError`` when
    no reading exists."""
    stripped = text.strip()
    if stripped.lower() in MISSING_SENTINELS:
        return None, "missing"
    try:
        value = float(stripped)
    except ValueError:
        pass
    else:
        if not math.isfinite(value):
            # "inf"/"-inf" (and numeric NaN spellings not covered by the
            # sentinels) parse as floats but are corrupt observations, not
            # missing ones: no reading exists, so the caller's unparseable
            # path decides — refuse under conservative repair, drop with
            # disclosure under aggressive. Textual "nan" stays a documented
            # missing sentinel above.
            raise ValueError(f"non-finite value: {stripped!r}")
        return value, "clean"
    s, negated, _ = _strip_decoration(stripped)
    tier = "normalised"
    if _GROUPED_SPACE.fullmatch(s):
        s = _GROUP_SEPARATORS.sub("", s)
    if "," in s and "." in s:
        if s.rfind(".") > s.rfind(","):
            s = s.replace(",", "")
        else:
            s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        if _GROUPED_COMMA.fullmatch(s):
            if comma_role == "decimal":
                s = s.replace(",", ".", 1).replace(",", "")
            else:
                s = s.replace(",", "")
                if comma_role is None:
                    tier = "assumptive"
        else:
            s = s.replace(",", ".")
    value = float(s)  # may raise ValueError: genuinely unparseable
    if not math.isfinite(value):
        raise ValueError(f"non-finite value: {stripped!r}")
    return (-value if negated else value), tier


# --- timestamp leniency -----------------------------------------------------

class AmbiguousDateOrder(Exception):
    """A d/m/Y-or-m/d/Y date with no evidence deciding the order."""


_AMBIGUOUS_DATE = re.compile(
    r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})([ T]\d{1,2}:\d{2}(:\d{2})?)?"
)
_EPOCH_SECONDS = re.compile(r"\d{10}")
_EPOCH_MILLIS = re.compile(r"\d{13}")
_EXTRA_FORMATS = (
    "%Y/%m/%d", "%Y/%m/%d %H:%M", "%Y/%m/%d %H:%M:%S",
    "%Y%m%d",
    "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y",
    "%b %d, %Y", "%B %d, %Y", "%d-%b-%Y", "%d-%B-%Y",
)


def scan_day_first(texts: list[str]) -> bool | None:
    """Column-wide date-order evidence: any first component > 12 forces
    day-first; any second component > 12 forces month-first. Contradictory
    evidence is an AMBIGUOUS_DATE_ORDER error — the column is inconsistent."""
    day_first = month_first = False
    for text in texts:
        match = _AMBIGUOUS_DATE.fullmatch(text.strip())
        if not match:
            continue
        first, second = int(match.group(1)), int(match.group(2))
        if first > 12:
            day_first = True
        if second > 12:
            month_first = True
    if day_first and month_first:
        raise GnomonError(
            "AMBIGUOUS_DATE_ORDER",
            "The time column mixes day-first and month-first dates; "
            "the column is internally inconsistent.",
        )
    if day_first:
        return True
    if month_first:
        return False
    return None


def parse_timestamp_lenient(text: str, day_first: bool | None) -> tuple[datetime, str]:
    """Parse a timestamp cell leniently.

    Returns ``(value, tier)``; tier ``clean`` is the strict ISO path,
    ``normalised`` a deterministic format reinterpretation. Raises
    :class:`AmbiguousDateOrder` for an a/b/Y date that no evidence decides,
    and ``ValueError`` when no reading exists."""
    s = text.strip()
    iso = s[:-1] + "+00:00" if s.endswith("Z") else s
    try:
        return datetime.fromisoformat(iso), "clean"
    except ValueError:
        pass
    if _EPOCH_SECONDS.fullmatch(s):
        return datetime.fromtimestamp(int(s), tz=timezone.utc), "normalised"
    if _EPOCH_MILLIS.fullmatch(s):
        return datetime.fromtimestamp(int(s) / 1000.0, tz=timezone.utc), "normalised"
    for pattern in _EXTRA_FORMATS:
        try:
            return datetime.strptime(s, pattern), "normalised"
        except ValueError:
            continue
    match = _AMBIGUOUS_DATE.fullmatch(s)
    if match:
        first, second, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if first > 12:
            day, month = first, second
        elif second > 12:
            day, month = second, first
        elif day_first is True:
            day, month = first, second
        elif day_first is False:
            day, month = second, first
        else:
            raise AmbiguousDateOrder(s)
        clock_part = (match.group(4) or "").strip().lstrip("T")
        hour = minute = second_value = 0
        if clock_part:
            pieces = clock_part.split(":")
            hour, minute = int(pieces[0]), int(pieces[1])
            second_value = int(pieces[2]) if len(pieces) == 3 else 0
        return datetime(year, month, day, hour, minute, second_value), "normalised"
    raise ValueError(f"Cannot parse timestamp: {text!r}")


# --- grid repairs -----------------------------------------------------------

_EPOCH_NAIVE = datetime(1970, 1, 1)


def _to_seconds(value: datetime) -> float:
    if value.tzinfo is not None:
        return value.timestamp()
    return (value - _EPOCH_NAIVE).total_seconds()


def _from_seconds(seconds: float, template: datetime) -> datetime:
    if template.tzinfo is not None:
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    return _EPOCH_NAIVE + timedelta(seconds=seconds)


def _nearest_integer(value: float) -> int:
    """Round halves away from zero, immune to banker's rounding."""
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _snap_frequency(timestamps: list[datetime]) -> str | None:
    """Infer a representable intended cadence from a jittered fixed grid.

    Exact grids never arrive here. For operational cadences of at least one
    minute, try the nearest whole-minute schedule first, but only within the
    same tolerance later used to align points. This recovers a 20-minute cron
    whose observed deltas are 1199/1201 seconds without turning arbitrary
    irregular spacing into a preferred round number.
    """
    from .temporal import canonical_code
    deltas = [
        (right - left).total_seconds()
        for left, right in zip(timestamps, timestamps[1:])
    ]
    if not deltas:
        return None
    typical = median(deltas)
    if 26 * 86400 <= typical <= 35 * 86400:
        return "MS"
    candidates: list[float] = []
    if typical >= 60:
        candidates.append(float(_nearest_integer(typical / 60.0) * 60))
    candidates.append(float(_nearest_integer(typical)))
    for seconds in candidates:
        if seconds <= 0:
            continue
        tolerance = min(
            seconds * JITTER_TOLERANCE_FRACTION,
            MAX_JITTER_TOLERANCE_SECONDS,
        )
        if abs(typical - seconds) <= tolerance:
            code = canonical_code(timedelta(seconds=seconds))
            if code is not None:
                return code
    return None


def _alignment_plan(
    timestamps: list[datetime], frequency: str,
) -> tuple[list[datetime], dict[str, object]] | None:
    """Return one bounded, phase-aware alignment plan or ``None``.

    The first timestamp defines only a provisional slot origin. Removing
    integer slot offsets and taking the median residual learns the shared
    phase from the full series, including a series with real missing slots.
    No point is emitted until every displacement and slot is validated.
    """
    from .temporal import frequency_step

    step = frequency_step(frequency)
    if step is None:  # calendar months require an explicit regrid declaration
        return None
    step_seconds = step.total_seconds()
    tolerance = min(
        step_seconds * JITTER_TOLERANCE_FRACTION,
        MAX_JITTER_TOLERANCE_SECONDS,
    )
    seconds = [_to_seconds(item) for item in timestamps]
    origin = seconds[0]
    provisional = [
        _nearest_integer((value - origin) / step_seconds) for value in seconds
    ]
    phase = median(
        value - slot * step_seconds
        for value, slot in zip(seconds, provisional)
    )
    slots = [_nearest_integer((value - phase) / step_seconds) for value in seconds]
    aligned_seconds = [phase + slot * step_seconds for slot in slots]
    displacements = [abs(left - right)
                     for left, right in zip(seconds, aligned_seconds)]
    if any(value > tolerance + 1e-9 for value in displacements):
        return None
    if any(right <= left for left, right in zip(slots, slots[1:])):
        collision_slots = [
            right for left, right in zip(slots, slots[1:]) if right <= left
        ]
        raise GnomonError(
            "TIMESTAMP_ALIGNMENT_CONFLICT",
            "Bounded timestamp alignment would merge or reorder observations; "
            "Gnomon will not choose which measurement to keep.",
            {
                "frequency": frequency,
                "tolerance_seconds": tolerance,
                "conflicting_slots": collision_slots[:3],
            },
        )
    aligned = [_from_seconds(value, template)
               for value, template in zip(aligned_seconds, timestamps)]
    moved = [value for value in displacements if value > 1e-9]
    return aligned, {
        "cadence": frequency,
        "grid_phase": _from_seconds(phase, timestamps[0]).isoformat(),
        "tolerance_seconds": tolerance,
        "maximum_displacement_seconds": max(moved, default=0.0),
    }


def _grid_is_regular(timestamps: list[datetime], frequency: str) -> bool:
    from .temporal import next_timestamp
    return all(
        next_timestamp(left, frequency) == right
        for left, right in zip(timestamps, timestamps[1:])
    )


def _excessive(series: str, counts: dict[str, int], total: int) -> GnomonError:
    return GnomonError(
        "EXCESSIVE_REPAIR",
        f"Series {series} would need repairs to more than "
        f"{MAX_ASSUMPTIVE_FRACTION:.0%} of its rows; fix the data at the "
        "source instead of forecasting on a mostly invented series.",
        {"series": series, "total_observations": total, "repair_counts": counts},
    )


def repair_observations(
    observations: list["Observation"],
    requested_frequency: str | None,
    level: str,
    log: RepairLog,
) -> list["Observation"]:
    """Grid-level repairs, per series, gated on the strict path failing.

    ``safe`` collapses byte-identical duplicate rows and aligns bounded
    timestamp jitter. ``aggressive`` additionally resolves conflicting
    duplicates (last row in file order wins) and linearly interpolates
    interior gaps. Every action is disclosed; only invented or selected
    values consume the assumptive-repair ceiling."""
    if level == REPAIR_OFF or not observations:
        return observations
    from collections import defaultdict

    by_series: dict[str, list] = defaultdict(list)
    for item in observations:
        by_series[item.series].append(item)
    repaired: list = []
    changed = False
    for name in by_series:
        items = _repair_series(name, by_series[name], requested_frequency, level, log)
        changed = changed or items is not by_series[name]
        repaired.extend(items)
    return repaired if changed else observations


def _repair_series(
    name: str,
    items: list["Observation"],
    requested_frequency: str | None,
    level: str,
    log: RepairLog,
) -> list["Observation"]:
    from .data import Observation
    from .temporal import infer_frequency, next_timestamp, normalise_frequency

    total = len(items)
    # Duplicates, walking file order so "last row wins" is well defined.
    seen: dict[datetime, float] = {}
    exact = conflicts = 0
    kept: list[Observation] = []
    for item in items:
        if item.timestamp in seen:
            if item.value == seen[item.timestamp]:
                exact += 1
                continue
            if level == REPAIR_AGGRESSIVE:
                conflicts += 1
                seen[item.timestamp] = item.value
                kept = [entry for entry in kept if entry.timestamp != item.timestamp]
            # safe: keep both; the strict validator reports the conflict.
        else:
            seen[item.timestamp] = item.value
        kept.append(item)
    if exact:
        log.record("duplicate_row_collapsed",
                   "Byte-identical duplicate rows collapsed to one.",
                   series=name, count=exact)
    if conflicts:
        log.record("conflicting_duplicate_resolved",
                   "Duplicate timestamps with different values; the last row "
                   "in file order was kept.",
                   series=name, assumptive=True, count=conflicts)
    kept.sort(key=lambda item: item.timestamp)
    # Frequency: strict inference first; align only when it fails or the
    # grid is irregular at the strict frequency. Safe and aggressive share
    # this bounded, no-merge alignment boundary.
    timestamps = [item.timestamp for item in kept]
    frequency: str | None = None
    if requested_frequency:
        frequency = normalise_frequency(requested_frequency)
    else:
        try:
            frequency = infer_frequency(timestamps)
        except GnomonError:
            frequency = None
    snapped = 0
    alignment_attempted = False
    alignment_succeeded = False
    if len(timestamps) >= 3 and (
        frequency is None or not _grid_is_regular(timestamps, frequency)
    ):
        target = frequency or _snap_frequency(timestamps)
        if target is not None and target != "MS":
            alignment_attempted = True
            plan = _alignment_plan(timestamps, target)
            if plan is not None:
                alignment_succeeded = True
                aligned, metrics = plan
                snapped = sum(left != right
                              for left, right in zip(timestamps, aligned))
                if snapped:
                    kept = [
                        Observation(slot, item.value, name)
                        for item, slot in zip(kept, aligned)
                    ]
                    log.record(
                        "timestamp_jitter_aligned",
                        f"Bounded timestamp jitter aligned to the inferred "
                        f"{target} grid without changing values.",
                        series=name, assumptive=True, count=snapped,
                        metrics=metrics,
                    )
                frequency = target
    if frequency is None:
        return kept  # let the strict validator name the real problem
    aligned_grid_regular = _grid_is_regular(
        [item.timestamp for item in kept], frequency)
    if alignment_attempted and not alignment_succeeded and not aligned_grid_regular:
        # Outside-boundary jitter is not a gap and must not be interpolated.
        # Return it unchanged so the strict validator emits the typed grid
        # error instead of manufacturing points from a misaligned origin.
        return kept
    if level != REPAIR_AGGRESSIVE:
        if alignment_succeeded and not aligned_grid_regular:
            # The alignment established a cadence, so preserve that evidence
            # in the refusal instead of asking generic inference to rediscover
            # a general sub-daily step in the presence of a real gap.
            from .temporal import validate_and_group
            validate_and_group(kept, frequency)  # raises IRREGULAR_TIME_GRID
        return kept if exact or conflicts or snapped else items

    # Interior gaps: linear interpolation, capped.
    timestamps = [item.timestamp for item in kept]
    filled: list[Observation] = []
    run_cap = max(3, total // 10)
    for left, right in zip(kept, kept[1:]):
        expected = next_timestamp(left.timestamp, frequency)
        missing: list[datetime] = []
        while expected < right.timestamp:
            missing.append(expected)
            expected = next_timestamp(expected, frequency)
            if len(missing) > run_cap:
                raise _excessive(name, {"gap_run": len(missing)}, total)
        for index, slot in enumerate(missing, start=1):
            fraction = index / (len(missing) + 1)
            value = left.value + (right.value - left.value) * fraction
            filled.append(Observation(slot, value, name))
    if filled:
        log.record("gap_filled",
                   "Interior gaps filled by linear interpolation between "
                   "neighbouring observations.",
                   series=name, assumptive=True, count=len(filled),
                   example=filled[0].timestamp.isoformat())
    assumptive_touched = conflicts + len(filled)
    if assumptive_touched / max(1, total) > MAX_ASSUMPTIVE_FRACTION:
        raise _excessive(
            name,
            {"conflicts": conflicts, "filled": len(filled),
             "aligned_timestamps_not_charged": snapped},
            total,
        )
    if filled:
        kept = sorted(kept + filled, key=lambda item: item.timestamp)
    return kept


# --- structural regridding ---------------------------------------------------
#
# Calendar structure is not messiness. A business-day series is missing
# nothing on Saturday — the market was closed; a month-end feed is not
# jittered — its publisher stamps period ends. The assumptive-repair
# ceiling (MAX_ASSUMPTIVE_FRACTION) exists to stop Gnomon inventing a
# dataset, and it is correct to refuse when 30% of rows needed guessing —
# but weekends alone are ~29% of a calendar-daily grid, so business-day
# data could *never* be repaired in-tool no matter how clean it was. A
# regrid is therefore a separate, caller-declared transform: disclosed
# like every repair, warned on every touched series, but not charged
# against the messiness ceiling, because the caller has stated the
# calendar rather than asked Gnomon to guess about noise.

REGRID_POLICIES = ("business_daily", "month_start")

#: Sanity bound for business_daily: weekends and holidays fill ~31% of a
#: calendar grid. Past this fraction the declaration is implausible — the
#: series is sparser than any Mon-Fri calendar — and the honest answer is
#: a refusal naming the counts, not a grid that is mostly invention.
MAX_BUSINESS_FILL_FRACTION = 0.45

#: Longest run of consecutive calendar days business_daily will fill.
#: Real market closures cluster: a long weekend is 3-4 days, the NYSE's
#: 2001-09-11 closure spanned 6 calendar days with its weekend, and the
#: longest recurring closures anywhere — Chinese A-share Spring Festival
#: and Golden Week — reach ~9. A longer hole is a data outage wearing a
#: holiday's clothes, and carrying one value across it flat would mask
#: exactly the kind of gap the grid validators exist to surface. The
#: total-fraction bound above cannot catch this (a month-long outage in
#: ten years of data barely moves the fraction); the run bound does.
MAX_BUSINESS_FILL_RUN = 10


def regrid_observations(
    observations: "list[Observation]", policy: str, log: RepairLog,
) -> "tuple[list[Observation], str]":
    """Apply a declared calendar regrid; returns (observations, implied
    frequency code).

    - ``business_daily``: the series is Mon-Fri market data. Every missing
      calendar day (weekends, holidays) is filled by carrying the prior
      observation forward, producing the continuous daily grid the
      validators require. Implied frequency ``D``.
    - ``month_start``: the series is monthly, stamped anywhere in the
      month (typically month ends). Every timestamp is restamped to the
      first of its month; two observations landing in one month is a
      loud conflict. Implied frequency ``MS``.
    """
    from .data import Observation

    if policy not in REGRID_POLICIES:
        raise GnomonError(
            "INVALID_ARGUMENTS",
            f"Unknown regrid policy {policy!r}; supported: "
            + ", ".join(REGRID_POLICIES) + ".",
            {"supported": list(REGRID_POLICIES)},
        )
    by_series: dict[str, list[Observation]] = {}
    for item in observations:
        by_series.setdefault(item.series, []).append(item)

    result: list[Observation] = []
    if policy == "month_start":
        for name, items in by_series.items():
            seen: dict[tuple[int, int], Observation] = {}
            restamped = 0
            for item in sorted(items, key=lambda entry: entry.timestamp):
                month = (item.timestamp.year, item.timestamp.month)
                if month in seen:
                    raise GnomonError(
                        "REGRID_CONFLICT",
                        f"regrid=month_start: series {name} has two "
                        f"observations in {month[0]}-{month[1]:02d} "
                        f"({seen[month].timestamp.isoformat()} and "
                        f"{item.timestamp.isoformat()}); a monthly grid "
                        f"holds one value per month.",
                        {"series": name, "year": month[0], "month": month[1]},
                    )
                stamped = item.timestamp.replace(
                    day=1, hour=0, minute=0, second=0, microsecond=0)
                if stamped != item.timestamp:
                    restamped += 1
                replacement = Observation(stamped, item.value, name)
                seen[month] = replacement
                result.append(replacement)
            if restamped:
                log.record(
                    "regrid_month_start",
                    "Timestamps restamped to the first of their month "
                    "(regrid=month_start); values are unchanged.",
                    series=name, assumptive=True, count=restamped,
                )
        return sorted(result, key=lambda item: (item.series, item.timestamp)), "MS"

    for name, items in by_series.items():
        ordered = sorted(items, key=lambda entry: entry.timestamp)
        clock_times = {(item.timestamp.hour, item.timestamp.minute,
                        item.timestamp.second, item.timestamp.microsecond)
                       for item in ordered}
        if len(clock_times) > 1:
            raise GnomonError(
                "REGRID_IMPLAUSIBLE",
                f"regrid=business_daily: series {name} mixes "
                f"{len(clock_times)} different times of day; a daily grid "
                f"needs one observation time. Normalise the timestamps "
                f"first.",
                {"series": name, "distinct_times": len(clock_times)},
            )
        # Calendar-aware stepping: on timezone-aware data a fixed
        # timedelta(days=1) drifts an hour at every DST transition and the
        # filled stamps then fail the very grid check the regrid exists to
        # satisfy; next_timestamp("D") preserves the wall-clock time the
        # way the grid validator expects. (Imported here: temporal imports
        # data, which type-imports this module.)
        from .temporal import next_timestamp

        result.extend(ordered)
        filled = 0
        previous = ordered[0]
        for item in ordered[1:]:
            run = 0
            slot = next_timestamp(previous.timestamp, "D")
            while slot < item.timestamp:
                run += 1
                if run > MAX_BUSINESS_FILL_RUN:
                    raise GnomonError(
                        "REGRID_IMPLAUSIBLE",
                        f"regrid=business_daily: series {name} has a hole "
                        f"longer than {MAX_BUSINESS_FILL_RUN} calendar days "
                        f"after {previous.timestamp.isoformat()} (next "
                        f"observation {item.timestamp.isoformat()}). The "
                        f"longest real market closures run ~9 calendar "
                        f"days; this looks like a data outage, and carrying "
                        f"one value flat across it would hide the gap. Fix "
                        f"or split the data at the hole.",
                        {"series": name,
                         "after": previous.timestamp.isoformat(),
                         "next_observation": item.timestamp.isoformat(),
                         "max_fill_run": MAX_BUSINESS_FILL_RUN},
                    )
                result.append(Observation(slot, previous.value, name))
                filled += 1
                slot = next_timestamp(slot, "D")
            previous = item
        grid_size = len(ordered) + filled
        if filled and filled / grid_size > MAX_BUSINESS_FILL_FRACTION:
            raise GnomonError(
                "REGRID_IMPLAUSIBLE",
                f"regrid=business_daily: series {name} needs "
                f"{filled} of {grid_size} calendar days filled "
                f"({filled / grid_size:.0%}); a Mon-Fri calendar misses "
                f"~31%. This series is sparser than business-day data — "
                f"check the frequency, or resample instead.",
                {"series": name, "filled": filled, "grid_days": grid_size,
                 "fill_fraction": round(filled / grid_size, 4)},
            )
        if filled:
            log.record(
                "regrid_business_daily",
                "Non-trading calendar days (weekends, holidays) filled by "
                "carrying the prior observation forward "
                "(regrid=business_daily); filled values are repeats, not "
                "measurements.",
                series=name, assumptive=True, count=filled,
            )
    return sorted(result, key=lambda item: (item.series, item.timestamp)), "D"

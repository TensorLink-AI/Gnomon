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

- ``safe`` (the default): reinterprets cell *text* only — date formats,
  currency/thousands separators, percent signs, sentinel missing values,
  fully blank rows, byte-identical duplicate rows. It never invents a
  value, never moves a timestamp, and never drops a data point.
- ``aggressive`` (opt-in): structural fixes — interior gap interpolation,
  snapping jittered timestamps to the grid, conflicting-duplicate
  resolution (last row wins), dropping unparseable rows, coercing naive
  timestamps in a mixed-timezone file to UTC. All capped and disclosed.

Everything here is a deterministic function of the input bytes.
"""

from __future__ import annotations

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

# Fraction of a series that assumptive grid repairs (snaps, fills, conflict
# resolutions) may touch before the honest answer is "fix the data".
MAX_ASSUMPTIVE_FRACTION = 0.30
# Fraction of rows that may be dropped as unparseable under aggressive repair.
MAX_DROPPED_FRACTION = 0.05

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

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["examples"] = list(self.examples)
        return payload


class RepairLog:
    """Accumulates repair actions; the single source of disclosure."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str | None], dict[str, object]] = {}

    def record(
        self, code: str, detail: str, *,
        series: str | None = None, assumptive: bool = False,
        example: str | None = None, count: int = 1,
    ) -> None:
        entry = self._entries.setdefault(
            (code, series),
            {"assumptive": assumptive, "detail": detail, "count": 0, "examples": []},
        )
        entry["count"] = int(entry["count"]) + count
        examples = entry["examples"]
        if example is not None and isinstance(examples, list) and len(examples) < 3:
            examples.append(example)

    def clone(self) -> "RepairLog":
        """An independent copy. A multi-target run records the shared
        file-level reads once, then forks a log per target so each
        column's repairs are disclosed on that column alone."""
        copy = RepairLog()
        for key, entry in self._entries.items():
            copy._entries[key] = {**entry, "examples": list(entry["examples"])}
        return copy

    def has_actions(self) -> bool:
        return bool(self._entries)

    def actions(self) -> list[RepairAction]:
        return [
            RepairAction(code, series, int(entry["count"]), bool(entry["assumptive"]),
                         str(entry["detail"]), tuple(entry["examples"]))  # type: ignore[arg-type]
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
        return float(stripped), "clean"
    except ValueError:
        pass
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


def _round_to_unit(value: datetime, step: timedelta) -> datetime:
    seconds = step.total_seconds()
    epoch = _to_seconds(value)
    # floor(x + 0.5): explicit half-up rounding, immune to banker's rounding.
    slot = int(epoch / seconds + 0.5) if epoch >= 0 else -int(-epoch / seconds + 0.5)
    return _from_seconds(slot * seconds, value)


def _round_to_month_start(value: datetime) -> datetime:
    year, month = value.year, value.month
    if value.day > 15:
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    result = value.replace(year=year, month=month, day=1,
                           hour=0, minute=0, second=0, microsecond=0)
    return result


def _snap_frequency(timestamps: list[datetime]) -> str | None:
    """Infer the intended frequency of a jittered grid from the median step."""
    from .temporal import FREQUENCIES
    deltas = [
        (right - left).total_seconds()
        for left, right in zip(timestamps, timestamps[1:])
    ]
    if not deltas:
        return None
    typical = median(deltas)
    for code, duration in FREQUENCIES.items():
        if code == "W":
            continue  # a week has no natural boundary to snap to
        step = duration.total_seconds()
        if abs(typical - step) <= 0.15 * step:
            return code
    if 26 * 86400 <= typical <= 35 * 86400:
        return "MS"
    return None


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

    ``safe`` collapses byte-identical duplicate rows only. ``aggressive``
    additionally resolves conflicting duplicates (last row in file order
    wins), snaps jittered timestamps to the inferred grid, and linearly
    interpolates interior gaps — all disclosed and capped."""
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
    from .temporal import frequency_step, infer_frequency, next_timestamp, normalise_frequency

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
    if level != REPAIR_AGGRESSIVE:
        return kept if exact or conflicts else items

    # Frequency: strict inference first; snap only when it fails or the
    # grid is irregular at the strict frequency.
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
    if len(timestamps) >= 3 and (
        frequency is None or not _grid_is_regular(timestamps, frequency)
    ):
        target = frequency or _snap_frequency(timestamps)
        if target is not None and target != "W":
            rounder = (
                _round_to_month_start if target == "MS"
                else lambda value: _round_to_unit(value, frequency_step(target))
            )
            resnapped: dict[datetime, Observation] = {}
            for item in kept:
                slot = rounder(item.timestamp)
                if slot != item.timestamp:
                    snapped += 1
                resnapped[slot] = Observation(slot, item.value, name)
            if snapped:
                previous = len(kept)
                kept = sorted(resnapped.values(), key=lambda item: item.timestamp)
                conflicts += previous - len(kept)
                log.record("timestamp_snapped",
                           f"Jittered timestamps rounded to the {target} grid.",
                           series=name, assumptive=True, count=snapped)
                frequency = target
    if frequency is None:
        return kept  # let the strict validator name the real problem

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
    assumptive_touched = conflicts + snapped + len(filled)
    if assumptive_touched / max(1, total) > MAX_ASSUMPTIVE_FRACTION:
        raise _excessive(
            name,
            {"conflicts": conflicts, "snapped": snapped, "filled": len(filled)},
            total,
        )
    if filled:
        kept = sorted(kept + filled, key=lambda item: item.timestamp)
    return kept

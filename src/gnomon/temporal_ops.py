"""Bounded calculations on explicit temporal facts, independent of forecasting.

No natural-language parser, implicit clock, holiday calendar or causal inference.
Named-zone rules come from the host's zoneinfo sources; their version is unknown.
All instant comparisons and elapsed arithmetic use UTC, never wall-clock order.
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta, timezone as dt_timezone
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

UTC = dt_timezone.utc
_DATE = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
_STAMP = re.compile(_DATE + r"T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?(?:Z|[+-][0-9]{2}:[0-9]{2}(?::[0-9]{2})?)?")
_TEXT = {"type": "string", "minLength": 1, "maxLength": 64}
_LOCAL = {"value": _TEXT, "timezone": {"type": "string", "minLength": 1, "maxLength": 128},
          "fold": {"type": "integer", "enum": [0, 1]}}
_SPAN = {"type": "object", "additionalProperties": False, "required": ["start", "end"],
         "properties": {"start": _TEXT, "end": _TEXT}}
_VARIANTS = {
    "normalize": ({**_LOCAL}, ["value"]),
    "duration": ({"start": _TEXT, "end": _TEXT}, ["start", "end"]),
    "shift": ({**_LOCAL, "amount": {"type": "integer", "minimum": -1_000_000, "maximum": 1_000_000},
               "unit": {"enum": ["seconds", "minutes", "hours", "days", "weeks", "months", "years"]},
               "mode": {"enum": ["elapsed", "calendar"]}, "target_fold": _LOCAL["fold"],
               "invalid_date": {"enum": ["reject", "clamp"]}}, ["value", "amount", "unit", "mode"]),
    "interval": ({"left": _SPAN, "right": _SPAN}, ["left", "right"]),
    "order_events": ({"events": {"type": "array", "maxItems": 1000, "items": {
        "type": "object", "additionalProperties": False, "required": ["event_id", "at"],
        "properties": {"event_id": {"type": "string", "minLength": 1, "maxLength": 128}, "at": _TEXT}}},
        "start": _TEXT, "end": _TEXT}, ["events"]),
}
TEMPORAL_EXAMPLES = {
    "normalize": {"operation": "normalize", "value": "2026-01-01T10:00:00+10:00"},
    "duration": {"operation": "duration", "start": "2026-01-01T00:00:00Z", "end": "2026-01-02T00:00:00Z"},
    "shift": {"operation": "shift", "value": "2026-01-01", "amount": 1, "unit": "days", "mode": "calendar"},
    "interval": {"operation": "interval",
                 "left": {"start": "2026-01-01T00:00:00Z", "end": "2026-01-03T00:00:00Z"},
                 "right": {"start": "2026-01-02T00:00:00Z", "end": "2026-01-04T00:00:00Z"}},
    "order_events": {"operation": "order_events", "events": [
        {"event_id": "arrival", "at": "2026-01-01T00:00:00Z"},
        {"event_id": "departure", "at": "2026-01-02T00:00:00Z"}]},
}
TEMPORAL_SCHEMA = {"type": "object", "oneOf": [
    {"type": "object", "additionalProperties": False, "required": ["operation", *required],
     "examples": [TEMPORAL_EXAMPLES[name]],
     "properties": {"operation": {"const": name}, **properties}}
    for name, (properties, required) in _VARIANTS.items()
]}


def _keys(value, allowed, required):
    if not isinstance(value, dict):
        raise ValueError("temporal arguments must be an object")
    missing, unknown = set(required) - set(value), set(value) - set(allowed)
    if missing or unknown:
        problems = (["missing fields: " + ", ".join(sorted(missing))] if missing else [])
        problems += (["unknown fields: " + ", ".join(sorted(unknown))] if unknown else [])
        raise ValueError("; ".join(problems))


def _fold(value):
    if value is not None and (type(value) is not int or value not in (0, 1)):
        raise ValueError("fold must be integer 0 or 1")


def _parse(value):
    if not isinstance(value, str) or len(value) > 64 or not _STAMP.fullmatch(value):
        raise ValueError("timestamp requires ISO YYYY-MM-DDTHH:MM:SS, at most 6 fractional digits and an optional offset")
    offset = re.search(r"([+-])([0-9]{2}):([0-9]{2})(?::([0-9]{2}))?$", value)
    if offset:
        hours, minutes, seconds = map(int, (offset[2], offset[3], offset[4] or "0"))
        if hours > 23 or minutes > 59 or seconds > 59 or (offset[1] == "-" and not (hours or minutes or seconds)):
            raise ValueError("invalid or unknown UTC offset")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _instant(value):
    parsed = _parse(value)
    if parsed.tzinfo is None:
        raise ValueError("instant requires an explicit UTC offset; normalize local timestamps first")
    return parsed.astimezone(UTC)


def _zone(name):
    if not isinstance(name, str) or len(name) > 128 or not re.fullmatch(r"[A-Za-z0-9_+-]+(?:/[A-Za-z0-9_+-]+)*", name):
        raise ValueError("timezone requires a named IANA zone")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("timezone unavailable; install/configure IANA zoneinfo data or supply an explicit offset") from None


def _candidates(wall, zone):
    candidates = {}
    for fold in (0, 1):
        candidate = wall.replace(tzinfo=zone, fold=fold)
        utc = candidate.astimezone(UTC)
        if utc.astimezone(zone).replace(tzinfo=None) == wall:
            candidates.setdefault(utc, candidate)
    return list(candidates.values())


def _resolve(wall, zone, fold):
    _fold(fold)
    candidates = _candidates(wall, zone)
    if not candidates:
        raise ValueError("nonexistent local time (clock gap); supply a valid time explicitly")
    if len(candidates) == 2:
        if fold is None:
            raise ValueError("ambiguous local time; specify fold=0 (first) or fold=1 (second)")
        return next(candidate for candidate in candidates if candidate.fold == fold)
    if fold == 1:
        raise ValueError("fold=1 requested for an unambiguous local time")
    return candidates[0]


def _local(value, timezone=None, fold=None):
    _fold(fold)
    parsed = _parse(value)
    zone = _zone(timezone) if timezone is not None else None
    if parsed.tzinfo is None:
        if zone is None:
            raise ValueError("local timestamp requires an explicit timezone")
        return _resolve(parsed, zone, fold)
    if fold is not None:
        raise ValueError("fold is only valid for a local timestamp without an offset")
    return parsed.astimezone(zone) if zone is not None else parsed


def _iso(value):
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _projection(value):
    named = isinstance(value.tzinfo, ZoneInfo)
    return {"kind": "instant", "local": value.isoformat(), "utc": _iso(value),
            "timezone": value.tzinfo.key if named else None,
            "timezone_rules": "host_zoneinfo_unversioned" if named else "explicit_fixed_offset",
            "utc_offset_seconds": int(value.utcoffset().total_seconds()), "fold": value.fold,
            "ambiguous_local_time": len(_candidates(value.replace(tzinfo=None), value.tzinfo)) == 2 if named else False}


def _normalize(**kwargs):
    return _projection(_local(**kwargs))


def _duration(start, end):
    delta = _instant(end) - _instant(start)
    micros = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
    whole, fraction = divmod(abs(micros), 1_000_000)
    seconds = ("-" if micros < 0 else "") + str(whole)
    if fraction:
        seconds += ("." + f"{fraction:06d}").rstrip("0")
    return {"mode": "elapsed", "duration_microseconds": str(micros), "duration_seconds": seconds,
            "numeric_encoding": "exact_decimal_strings"}


def _calendar_shift(value, amount, unit, invalid_date):
    if unit in ("days", "weeks"):
        return value + timedelta(days=amount * (7 if unit == "weeks" else 1)), False
    month_index = (value.year - 1) * 12 + value.month - 1 + amount * (12 if unit == "years" else 1)
    year, month = divmod(month_index, 12)
    year, month = year + 1, month + 1
    if not 1 <= year <= 9999:
        raise ValueError("date arithmetic exceeds supported years 1..9999")
    day = min(value.day, calendar.monthrange(year, month)[1])
    clamped = day != value.day
    if clamped and invalid_date != "clamp":
        raise ValueError("target month has no such day; explicitly select invalid_date=clamp if intended")
    return value.replace(year=year, month=month, day=day), clamped


def _shift(value, amount, unit, mode, timezone=None, fold=None, target_fold=None, invalid_date="reject"):
    if type(amount) is not int or abs(amount) > 1_000_000:
        raise ValueError("amount must be an integer between -1000000 and 1000000")
    _fold(fold)
    _fold(target_fold)
    scales = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400, "weeks": 604800}
    if mode not in ("elapsed", "calendar") or invalid_date not in ("reject", "clamp"):
        raise ValueError("mode must be elapsed/calendar and invalid_date must be reject/clamp")
    if not isinstance(unit, str) or unit not in (scales if mode == "elapsed" else ("days", "weeks", "months", "years")):
        raise ValueError("unit unsupported for the selected arithmetic mode")
    if invalid_date != "reject" and unit not in ("months", "years"):
        raise ValueError("clamp applies only to calendar months/years")
    if isinstance(value, str) and re.fullmatch(_DATE, value):
        if mode != "calendar" or any(v is not None for v in (timezone, fold, target_fold)):
            raise ValueError("date-only arithmetic requires calendar mode without timezone/fold arguments")
        shifted, clamped = _calendar_shift(date.fromisoformat(value), amount, unit, invalid_date)
        result = {"kind": "date", "date": shifted.isoformat()}
    else:
        origin = _local(value, timezone, fold)
        if mode == "elapsed":
            if target_fold is not None:
                raise ValueError("target_fold applies only to calendar arithmetic")
            shifted = (origin.astimezone(UTC) + timedelta(seconds=amount * scales[unit])).astimezone(origin.tzinfo)
            clamped = False
        else:
            wall, clamped = _calendar_shift(origin.replace(tzinfo=None), amount, unit, invalid_date)
            shifted = _resolve(wall, origin.tzinfo, target_fold)
        result = _projection(shifted)
    return {**result, "mode": mode, "amount": amount, "unit": unit,
            "invalid_date": invalid_date, "date_clamped": clamped}


def _span(value):
    _keys(value, ("start", "end"), ("start", "end"))
    start, end = _instant(value["start"]), _instant(value["end"])
    if start >= end:
        raise ValueError("interval must be nonempty: start < end")
    return start, end


def _interval(left, right):
    a, b = _span(left)
    c, d = _span(right)
    if b < c:
        relation = "before"
    elif b == c:
        relation = "meets"
    elif a > d:
        relation = "after"
    elif a == d:
        relation = "met_by"
    elif a == c:
        relation = "equals" if b == d else "starts" if b < d else "started_by"
    elif b == d:
        relation = "finishes" if a > c else "finished_by"
    elif a < c:
        relation = "overlaps" if b < d else "contains"
    else:
        relation = "during" if b < d else "overlapped_by"
    overlap = max(a, c) < min(b, d)
    return {"relation": relation, "boundary": "[start,end)", "overlap": overlap,
            "intersection": {"start": _iso(max(a, c)), "end": _iso(min(b, d))} if overlap else None,
            "left_contains_right": a <= c and b >= d, "right_contains_left": c <= a and d >= b}


def _order_events(events, start=None, end=None):
    if not isinstance(events, list) or len(events) > 1000:
        raise ValueError("events must be an array of at most 1000 events")
    if (start is None) != (end is None):
        raise ValueError("event window requires both start and end")
    window = _span({"start": start, "end": end}) if start is not None else None
    seen, rows, excluded = set(), [], []
    for event in events:
        _keys(event, ("event_id", "at"), ("event_id", "at"))
        event_id = event["event_id"]
        if not isinstance(event_id, str) or not 1 <= len(event_id) <= 128 or event_id in seen:
            raise ValueError("event IDs must be unique nonempty strings of at most 128 characters")
        seen.add(event_id)
        at = _instant(event["at"])
        if window is not None and not window[0] <= at < window[1]:
            excluded.append(event_id)
        else:
            rows.append((at, event_id))
    rows.sort(key=lambda row: row[0])  # Stable ties disclose input order, not causality.
    groups = []
    for at, event_id in rows:
        stamp = _iso(at)
        if not groups or groups[-1]["at"] != stamp:
            groups.append({"at": stamp, "event_ids": []})
        groups[-1]["event_ids"].append(event_id)
    return {"ordered_event_ids": [event_id for _, event_id in rows], "instant_groups": groups,
            "tie_policy": "input_order_not_causality", "input_count": len(events),
            "included_count": len(rows), "excluded_event_ids": excluded, "window_boundary": "[start,end)" if window else None}


def temporal_operation(operation: str | None = None, **arguments) -> dict:
    """Execute one closed operation; invalid/ambiguous facts raise ValueError.

    Explicit nulls are rejected, like the tool schema. Optional fields are omitted.
    Duration strings preserve exact microseconds even beyond JSON safe integers.
    """
    if not isinstance(operation, str) or operation not in _VARIANTS:
        raise ValueError("operation must be one of: " + ", ".join(_VARIANTS))
    properties, required = _VARIANTS[operation]
    _keys(arguments, properties, required)
    if any(value is None for value in arguments.values()):
        raise ValueError("omit optional arguments instead of supplying null")
    implementations = {"normalize": _normalize, "duration": _duration, "shift": _shift,
                       "interval": _interval, "order_events": _order_events}
    try:
        result = implementations[operation](**arguments)
    except OverflowError:
        raise ValueError("temporal calculation exceeds supported years 1..9999") from None
    return {"schema_version": "1", "status": "ok", "operation": operation, "result": result,
            "evidence": "explicit_temporal_calculation", "input_facts": "supplied_not_verified",
            "action_authorized": False}

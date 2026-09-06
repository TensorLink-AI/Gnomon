# Optional temporal calculations

`from gnomon import temporal_operation` lazily loads a small standard-library-only
module. It does not select models, fit forecasts, query a ledger or infer causality.
No separate package is needed for these primitives. No LLM accuracy uplift is established.
Natural-language interpretation and selecting relevant facts remain the caller's job.

Python and CLI work without configuration. In a default MCP execution session,
operator TOML `enable_temporal=true` adds `gnomon_temporal` to the ordinary six
tools (eight with a ledger). `GnomonSession(enable_temporal=True)` also works.
Tool arguments cannot enable the feature. Default discovery does not import this module.
Every operation returns `status`, `operation`, `result`, evidence labelled
`explicit_temporal_calculation`, supplied/unverified input status and no action authority.

## Input and calculation contract

Timestamps use `YYYY-MM-DDTHH:MM:SS` with optional one-to-six fractional digits and
`Z` or `±HH:MM` offset. Offset seconds (`±HH:MM:SS`) are also accepted so historical
named-zone outputs round-trip. Offsets must be less than 24 hours; `-00:00` means
unknown and is rejected. No leap seconds, arbitrary natural-language dates, implicit
`now`, rounded fractional precision or missing-date/time defaults. Years are 1..9999.
Overflow and invalid inputs fail, never wrap or silently normalize invalid fields.
Optional arguments are omitted, not null. Unknown fields are rejected.

| Operation | Arguments | Meaning |
| --- | --- | --- |
| `normalize` | `value`, optional `timezone`, `fold` | Resolve a local timestamp or convert an explicit instant; return local/UTC timestamp, offset and zone provenance. |
| `duration` | `start`, `end` | Signed elapsed difference between offset-aware instants. Exact decimal strings for seconds/microseconds avoid JSON integer precision loss. |
| `shift` | `value`, integer `amount`, `unit`, `mode`; optional `timezone`, `fold`, `target_fold`, `invalid_date` | Explicit elapsed or calendar arithmetic, as below. |
| `interval` | `left`, `right`, each `{start,end}` | One of 13 Allen relations, overlap, exact intersection and inclusive containment facts, for nonempty half-open intervals. |
| `order_events` | `events: [{event_id,at}]`, optional paired `start`,`end` | Stable instant order, tie groups and excluded IDs for an optional half-open window. |

Duration, interval and event timestamps must have offsets; normalize local values
first. Intervals use `[start,end)` with start strictly before end: touching intervals
meet but do not overlap. Zero-duration instants belong in events, not empty intervals.
Events have unique IDs of 1..128 characters, maximum 1000 events. Equal instants
retain input order and are explicitly grouped; neither causal order, knowledge time
nor recording time is inferred. All instant comparisons/subtractions occur in UTC.

## Ambiguity and date arithmetic

A timestamp without an offset requires an explicit named timezone. A repeated wall
time requires `fold=0` for its first occurrence or `fold=1` for its second. A clock
gap is rejected with either fold. `fold=1` on an unambiguous time is also rejected.
An offset already resolves the instant, so do not pass an input fold with it.
With an aware timestamp, `timezone` converts the same instant rather than asserting
that the input wall time belongs to that zone.

Named-zone calculations use available host zoneinfo sources; outputs disclose
`host_zoneinfo_unversioned`, not an attested timezone-database revision. Gnomon does
not install or pin zone data automatically. If no requested zone is available, the
operation fails; explicit offsets still work. For reproducible historical/future
replays, the operator must pin the deployment's actual timezone rules. Python can
load these rules from the system or a `tzdata` package; a package version alone does
not identify which source was used. [Python zoneinfo documentation](https://docs.python.org/3/library/zoneinfo.html).

`shift` requires `mode` and bounds `amount` to ±1,000,000:

- `elapsed`: seconds, minutes, hours, days or weeks. A day is exactly 86400 seconds;
  arithmetic runs in UTC and then converts back. Months/years and `target_fold` are refused.
- `calendar`: days, weeks, months or years in the proleptic Gregorian calendar.
  Preserve the wall clock and resolve the target using `target_fold` if ambiguous.
  Reject a target clock gap. Invalid month days are rejected by default; explicit
  `invalid_date="clamp"` permits month/year shifts to the month's last day and
  discloses `date_clamped`. No business days, holidays or other calendar systems.
- Date-only `YYYY-MM-DD` values support calendar arithmetic without timezone/fold
  arguments. They stay dates, not assumed midnight instants.

```python
from gnomon import temporal_operation

shifted = temporal_operation(
    "shift", value="2024-03-09T12:00:00", timezone="America/New_York",
    amount=1, unit="days", mode="calendar",
)
assert shifted["result"]["local"] == "2024-03-10T12:00:00-04:00"
elapsed = temporal_operation(
    "duration", start="2024-03-09T12:00:00-05:00", end=shifted["result"]["utc"],
)
assert elapsed["result"]["duration_seconds"] == "82800"  # 23 elapsed hours
```

Use the normal `gnomon_read` projection for large MCP event results. CLI and the
direct Python operation return full JSON-compatible values, not expiring receipts.
These bounds constrain operation size, not global process or provider memory.

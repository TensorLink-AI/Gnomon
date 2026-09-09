# Comparing prospective forecasts

`compare_history` compares forecasts recorded before their targets, grouped by
the same origin, series, unit, horizon, inputs, calendar and provider revisions.
Backtest executions have separate frozen study evidence and are excluded here.
Each origin needs all requested providers and a complete matching-unit actual
horizon visible at the supplied source and recording cutoffs. The first recorded
execution is used, never a retry selected for its better score.

This runnable synthetic example uses a controlled clock. In real use, use the
normal recording clock; never backdate an execution to make it eligible.

```python
from datetime import datetime, timedelta, timezone
from gnomon import GnomonSession, TemporalLedger
from gnomon.ids import FixedClock

origin = datetime(2026, 1, 20, tzinfo=timezone.utc)
ledger = TemporalLedger('prospective.db', clock=FixedClock(origin))
with GnomonSession.from_config(ledger=ledger) as session:
    providers = ['last_value', 'historical_mean']
    request = {
        'history': [10, 11, 12], 'horizon': 2,
        'timestamps': [(origin - timedelta(days=i)).isoformat() for i in (2, 1, 0)],
        'cutoff': origin.isoformat(),
        'known_time_cutoff': origin.isoformat(),
        'recorded_time_cutoff': origin.isoformat(),
        'future_timestamps': [(origin + timedelta(days=i)).isoformat() for i in (1, 2)],
        'frequency': 'D', 'series_id': 'sales', 'unit': 'widgets',
    }
    executions = [session.forecast(p, request) for p in providers]
    revisions = {r['provider']: r['revision'] for r in executions}
    availability = origin + timedelta(days=3)
    ledger.clock = FixedClock(availability + timedelta(hours=1))
    for timestamp, value in zip(request['future_timestamps'], (13, 14)):
        ledger.append_actual(series_id='sales', valid_time=timestamp, value=value,
            unit='widgets', source_available_at=availability.isoformat())
    comparison = ledger.compare_history(
        series_id='sales', unit='widgets', horizon=2, providers=revisions,
        start=origin.isoformat(), end=origin.isoformat(),
        source_as_of=availability.isoformat(),
        recorded_as_of=ledger.clock.now().isoformat(),
    )
    assert comparison['status'] == 'ok'
    assert comparison['matched_origins'] == 1 and comparison['provider_calls'] == 0
    print(comparison)
```

`timestamps` must align with the historical values; `future_timestamps` must
follow `cutoff`. Forecast recording must be at or after the origin and strictly
before the first target. All timestamps need offsets. Provider identities must
be explicitly versioned; pretrained providers additionally need an attested
training cutoff no later than the origin. The named series must be stable:
`__default__` is not sufficient for cross-origin production comparison.

CLI and MCP use the same query object with `operation: "compare_history"` under
`gnomon ledger --arguments` and `gnomon_ledger`, respectively. Obtain actual
execution revisions from returned forecasts rather than copying example build
IDs. A query can execute successfully and still report `insufficient_evidence`;
read `excluded`, its per-provider `causes`, and `next_step`. Missing timestamps
cannot be filled into an existing immutable execution after the event.

"""Run in a fresh directory: python -m gnomon.examples.compare_history."""
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

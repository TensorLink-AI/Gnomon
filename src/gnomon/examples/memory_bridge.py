"""Run in a fresh directory: python -m gnomon.examples.memory_bridge.

One synthetic forecast, no remote models or external memory writes. Preserve the
created memory-bridge.db to execute the verification calls in the output.
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from gnomon import EvidenceMemory, GnomonSession, HermesMemoryAdapter, TemporalLedger
from gnomon.ids import FixedClock


def main():
    path = Path('memory-bridge.db')
    if path.exists():
        raise SystemExit('Run in a fresh directory; memory-bridge.db already exists.')
    origin = datetime(2026, 1, 20, tzinfo=timezone.utc)
    ledger = TemporalLedger(path, clock=FixedClock(origin))
    bridge = EvidenceMemory(ledger, ledger_ref='synthetic-sales')
    with GnomonSession.from_config(ledger=ledger) as session:
        request = dict(history=[10, 11, 12], horizon=2, series_id='sales', unit='widgets',
            timestamps=[(origin-timedelta(days=i)).isoformat() for i in (2, 1, 0)],
            future_timestamps=[(origin+timedelta(days=i)).isoformat() for i in (1, 2)],
            cutoff=origin.isoformat(), frequency='D')
        execution = session.forecast('last_value', request)
        decision = bridge.record_decision(execution_id=execution['execution_id'], expected_request=request,
            claimed_provider='last_value', rationale='Only the latest-level baseline was executed.',
            assumptions=['The latest level is informative.'], invalidation_conditions=['The level changes.'])
        did = decision['decision_id']
        original_hash = decision['original_decision_sha256']
        assert decision['scoring']['status'] == 'pending'
        ledger.clock = FixedClock(origin+timedelta(days=3))
        for timestamp, value in zip(request['future_timestamps'], (13, 14)):
            ledger.append_actual(series_id='sales', unit='widgets', valid_time=timestamp, value=value,
                source_available_at=ledger._now())
        cutoffs = dict(source_as_of=ledger._now(), recorded_as_of=ledger._now())
        checks = bridge.check_claims(decision_id=did, claims=[
            {'field':'provider','value':'seasonal_naive'}, {'field':'mae','value':1.5},
            {'field':'explanation','value':'A promotion caused the error.'}], **cutoffs)
        assert [c['status'] for c in checks['checks']] == ['contradicted','supported','unverified']
        lesson = bridge.record_lesson(decision_id=did,
            lesson='The executed last-value forecast underpredicted. The cause remains unverified.', **cutoffs)
        adapter = HermesMemoryAdapter(bridge)
        proposed = adapter.lesson_update(lesson_id=lesson['lesson_id'], recorded_as_of=ledger._now())
        assert not proposed['external_write_performed']
        ledger.clock = FixedClock(origin+timedelta(days=4))
        ledger.append_actual(series_id='sales', unit='widgets', valid_time=request['future_timestamps'][1],
            value=20, source_available_at=ledger._now())
        retrieved = bridge.retrieve_lessons(series_id='sales', unit='widgets', horizon=2,
            source_as_of=ledger._now(), recorded_as_of=ledger._now())
        old = retrieved['lessons'][0]
        assert old['scoring']['metrics']['mae'] == 1.5
        assert old['current_evidence']['metrics']['mae'] == 4.5
        assert old['current_evidence']['changed_since_lesson']
        assert old['original_decision_sha256'] == original_hash
        print(json.dumps({'decision':decision, 'claim_checks':checks, 'retrieved':retrieved,
            'hermes_update_proposal':proposed, 'forecast_calls':1, 'external_memory_writes':0}, indent=2))


if __name__ == '__main__':
    main()

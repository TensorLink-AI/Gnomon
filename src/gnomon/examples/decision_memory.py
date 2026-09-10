"""Run in a fresh directory: python -m gnomon.examples.decision_memory.

Synthetic promotion labels are declared facts for this example, not inferred
from sales. A controlled clock demonstrates availability without sleeping.
"""
from datetime import datetime, timedelta, timezone
import json

from gnomon import GnomonSession, TemporalLedger, put_lesson
from gnomon.ids import FixedClock


def main():
    origin = datetime(2026, 1, 20, tzinfo=timezone.utc)
    ledger = TemporalLedger('decision-memory.db', clock=FixedClock(origin))
    with GnomonSession.from_config(ledger=ledger) as session:
        request = dict(history=[10, 11, 12], horizon=2, series_id='sales', unit='widgets',
            timestamps=[(origin - timedelta(days=i)).isoformat() for i in (2, 1, 0)],
            future_timestamps=[(origin + timedelta(days=i)).isoformat() for i in (1, 2)],
            cutoff=origin.isoformat(), frequency='D')
        runs = [session.forecast(p, request) for p in ('last_value', 'historical_mean')]
        revisions = {r['provider']: r['revision'] for r in runs}
        summary = ledger.record_decision_summary(execution_id=runs[0]['execution_id'],
            rationale='Use the latest observed level while the planned promotion is active.',
            assumptions=['The synthetic promotion schedule applies to this named series.'],
            invalidation_conditions=['The promotion is cancelled or the observed level changes.'],
            context=[dict(key='promotion', value='planned', valid_from=origin.isoformat(),
                valid_to=(origin + timedelta(days=3)).isoformat(),
                source_available_at=(origin - timedelta(days=1)).isoformat(), source_ref='synthetic:schedule-v1')],
            evidence_refs=[{'kind': 'execution', 'id': r['execution_id']} for r in runs])
        decision_id = summary['decision_id']
        original = ledger.decision(decision_id)
        waiting = ledger.review_decision(decision_id=decision_id, source_as_of=origin.isoformat(), recorded_as_of=origin.isoformat())
        assert not waiting['review_ready']
        available = origin + timedelta(days=3)
        ledger.clock = FixedClock(available)
        for t, value in zip(request['future_timestamps'], (13, 14)):
            ledger.append_actual(series_id='sales', unit='widgets', valid_time=t, value=value,
                                 source_available_at=available.isoformat())
        cutoffs = dict(source_as_of=available.isoformat(), recorded_as_of=available.isoformat())
        comparison = ledger.compare_context(series_id='sales', unit='widgets', horizon=2,
            providers=revisions, start=origin.isoformat(), end=origin.isoformat(),
            context_filters={'promotion': 'planned'}, **cutoffs)
        assert comparison['matched_origins'] == 1
        review = ledger.review_decision(decision_id=decision_id, **cutoffs)
        assert review['review_ready'] and review['metrics']['mae'] == 1.5
        lesson = ledger.record_lesson(decision_id=decision_id,
            lesson='The latest-level forecast underpredicted this origin. The promotion explanation remains untested.', **cutoffs)
        export = ledger.export_lesson(lesson_id=lesson['lesson_id'], recorded_as_of=available.isoformat())
        assert ledger.decision(decision_id)['inputs'] == original['inputs']
        assert not export['business_explanation_validated']

        # This tiny local sink demonstrates the adapter contract without a
        # dependency or network call. Use a persistent LangGraph store in your app.
        class LocalMemory:
            def __init__(self):
                self.items = {}

            def put(self, namespace, key, value):
                self.items[(namespace, key)] = value

        memory = LocalMemory()
        put_lesson(memory, ('example-user', 'gnomon-lessons'), export)
        print(json.dumps({'decision_id': decision_id, 'review_ready': review['review_ready'],
                          'matched_origins': comparison['matched_origins'], 'lesson': export}, indent=2))


if __name__ == '__main__':
    main()

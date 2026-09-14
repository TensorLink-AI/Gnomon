"""Synthetic reproduction of recording-time contamination in review catalogues."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from .agent_review_runtime import run as fixture
from .agent_review import review as original_review
from benchmarks.hermes_ml_checkpoint_v4 import dev_evidence_summary


def run(output):
    from gnomon import TemporalLedger, InferenceEngine, ForecastResult
    from gnomon.ids import FixedClock

    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    setup = fixture(root / 'base')
    records = json.loads((root / 'base/records.json').read_text())
    task = json.loads((root / 'base/task.json').read_text())
    path = root / 'base/ledger.db'
    db = TemporalLedger(path, clock=FixedClock(datetime(2026, 1, 30, tzinfo=timezone.utc)))
    calls = []
    engine = InferenceEngine(ledger=db)
    digest = lambda: hashlib.sha256(path.read_bytes()).hexdigest()

    before_hash = digest()
    before = original_review(db, records, task, dev_evidence_summary, root / 'before.json', limit=1)
    assert digest() == before_hash and not calls
    request = {
        'history': [1, 1, 1], 'horizon': 2, 'frequency': 'D',
        'series_id': task['series_id'], 'unit': task['unit'],
        'cutoff': '2026-01-08T00:00:00+00:00',
        'timestamps': [f'2026-01-{day:02d}T00:00:00+00:00' for day in (6, 7, 8)],
        'future_timestamps': [f'2026-01-{day:02d}T00:00:00+00:00' for day in (9, 10)],
    }
    for name in ('late_d', 'late_e'):
        def provider(r, name=name):
            calls.append(name)
            return ForecastResult(point=(2,) * r.horizon, timestamps=r.future_timestamps,
                                  series_id=r.series_id, unit=r.unit)
        engine.register(name, provider, revision='synthetic-v1:' + name,
                        deterministic=True, lifecycle='stateless')
        execution = engine.forecast(name, request).to_dict()
        records.append({'event': 'result', 'kind': 'forecast', 'request': request,
                        'config_id': name, 'config': {'synthetic_provider': name}, 'execution': execution})
    after_hash = digest()
    after = original_review(db, records, task, dev_evidence_summary, root / 'after.json', limit=1)
    assert digest() == after_hash and len(calls) == 2
    report = {
        'fixture': setup, 'synthetic_forecasts': setup['synthetic_forecasts'] + len(calls),
        'provider_calls_during_review': 0, 'api_calls': 0, 'protected_access': False,
        'ledger_unchanged_during_review': True,
        'before': {'status': before['status'], 'pairs': before['pagination']['total_pairs'],
                   'first_pair': before['cards'][0]['config_ids']},
        'after': {'status': after['status'], 'pairs': after['pagination']['total_pairs'],
                  'first_pair': after['cards'][0]['config_ids']},
        'catalogue_changed': before['configuration_index'] != after['configuration_index'],
        'cards_changed': before['cards'] != after['cards'],
        'ledger_before_sha256': before_hash, 'ledger_after_sha256': after_hash,
    }
    for name, value in [('report', report), ('records', records), ('task', task),
                        ('brief-before', before), ('brief-after', after)]:
        (root / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')
    return report


if __name__ == '__main__':
    print(json.dumps(run(sys.argv[1]), indent=2))

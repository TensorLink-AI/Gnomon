"""Probe090 public session-envelope compatibility before correction."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from .visible_agent_review import visible_records


def run(output):
    from gnomon import GnomonSession, InferenceEngine, ForecastResult, TemporalLedger
    from gnomon.ids import FixedClock
    from gnomon.build_info import build_info
    if importlib.metadata.version('gnomon-forecast') != '1.2.0':
        raise ValueError('Pinned1.2.0 required')
    build = build_info()
    if build['commit'] != 'a38cd0cad35383e5f10021abf3aa20d4c16923be':
        raise ValueError('Wrong runtime build')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    origin = '2026-01-03T00:00:00+00:00'
    db = TemporalLedger(root / 'ledger.db', clock=FixedClock(datetime.fromisoformat(origin)))
    engine = InferenceEngine(ledger=db)
    calls = []
    config = {'synthetic_provider': 'constant'}
    def provider(r):
        calls.append(1)
        return ForecastResult(point=(2,) * r.horizon, series_id=r.series_id, unit=r.unit,
                              timestamps=r.future_timestamps, metadata={'config': config})
    engine.register('constant', provider, revision='synthetic-v1', deterministic=True, lifecycle='stateless')
    request = {'history': [1, 1, 1], 'horizon': 2, 'series_id': 'sales', 'unit': 'widgets',
               'timestamps': [f'2026-01-{i:02d}T00:00:00+00:00' for i in (1, 2, 3)],
               'future_timestamps': [f'2026-01-{i:02d}T00:00:00+00:00' for i in (4, 5)],
               'cutoff': origin, 'known_time_cutoff': origin, 'frequency': 'D', 'season': 7}
    session = GnomonSession(engine=engine)
    outputs = {'session': session.forecast('constant', request), 'engine': engine.forecast('constant', request).to_dict()}
    digest = lambda: hashlib.sha256((root / 'ledger.db').read_bytes()).hexdigest()
    before = digest()
    task = {'origin': '2026-01-06T00:00:00+00:00', 'series_id': 'sales', 'unit': 'widgets', 'horizon': 2}
    results = {}
    rows = {}
    for kind, execution in outputs.items():
        row = {'event': 'result', 'kind': 'forecast', 'config_id': 'constant', 'config': config,
               'request': request, 'execution': execution}
        rows[kind] = row
        try:
            accepted, excluded, reads = visible_records(db, [row], task)
            results[kind] = {'accepted': len(accepted), 'excluded': excluded, 'execution_reads': reads}
        except (ValueError, KeyError) as exc:
            results[kind] = {'accepted': 0, 'error_type': type(exc).__name__, 'error': str(exc)}
    assert digest() == before and len(calls) == 2
    report = {'build': build, 'envelope_keys': {k: sorted(v) for k, v in outputs.items()},
              'results': results, 'synthetic_forecasts': len(calls), 'provider_calls_during_review': 0,
              'ledger_unchanged_during_review': True, 'api_calls': 0, 'protected_access': False}
    for name, value in [('report', report), ('rows', rows), ('task', task)]:
        (root / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')
    return report


if __name__ == '__main__':
    print(json.dumps(run(sys.argv[1]), indent=2))

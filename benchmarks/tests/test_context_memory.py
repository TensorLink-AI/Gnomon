from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json

import pytest

from gnomon import GnomonSession, TemporalLedger
from gnomon.ids import FixedClock
from gnomon.final_selection import forecast_request_fingerprint
from benchmarks.ledger_optimization.context_memory import labels, prepare
from benchmarks.ledger_optimization.agent_loop import context


def test_context_labels_use_observable_conditions_only():
    request = {'history': [1]*14+[2]*14, 'horizon': 2,
               'future_covariate_names': ['onpromotion'], 'future_covariates': [[0], [1]]}
    assert labels(request) == {'sparsity': 'low', 'trend': 'rising', 'volatility': 'stable', 'promotion': 'some'}
    assert labels({**request, 'actual': [1000, 1000]}) == labels(request)
    assert labels({**request, 'history': [0]*28})['sparsity'] == 'high'
    assert labels({**request, 'future_covariate_names': []})['promotion'] == 'unknown'


def test_memory_preparation_handles_canonical_timestamps_and_keeps_original_immutable(tmp_path):
    path = tmp_path/'source.db'
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ledger = TemporalLedger(path, clock=FixedClock(origin))
    session = GnomonSession.from_config(ledger=ledger)
    cases, requests = [], []
    for i in range(2):
        now = origin+timedelta(days=i*3)
        ledger.clock = FixedClock(now)
        request = dict(history=[1]*28, horizon=1, series_id='s', unit='widgets', frequency='D', season=1,
                       timestamps=[(now-timedelta(days=d)).isoformat() for d in range(27, -1, -1)],
                       cutoff=now.isoformat(), future_timestamps=[(now+timedelta(days=1)).isoformat()])
        runs = [session.forecast(p, request) for p in ('last_value', 'historical_mean')]
        ledger.clock = FixedClock(now+timedelta(days=1))
        ledger.append_actual(series_id='s', unit='widgets', valid_time=request['future_timestamps'][0], value=1,
                             source_available_at=ledger._now())
        cases.append({'series_id': 's', 'round': i, 'origin': now.isoformat(),
                      'execution_ids': [r['execution_id'] for r in runs],
                      'predictions': {r['provider']: [1] for r in runs},
                      'current_card': {r['provider']: {'cv_rmsle': 0} for r in runs},
                      'scores': {r['provider']: 0 for r in runs}})
        requests.append(ledger.execution(runs[0]['execution_id'])['request'])
    session.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    prepare(cases, path, tmp_path/'prepared')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    bundle = json.loads((tmp_path/'prepared/memory.json').read_text())
    assert not bundle['confirmation_opened'] and bundle['provider_calls'] == 0
    first, second = bundle['packets']
    assert not first['raw_history']['records']
    assert len(second['raw_history']['records']) == 1
    assert datetime.fromisoformat(second['raw_history']['records'][0]['origin']) < datetime.fromisoformat(second['origin'])
    control = context(cases[1], requests[1], 'no_ledger', {}, second)
    treatment = context(cases[1], requests[1], 'ledger_context', {}, second)
    assert control['raw_matched_history'] == treatment['raw_matched_history']
    assert control['current_context'] == treatment['current_context']
    assert 'historical_evidence' not in control
    assert treatment['historical_evidence']['selected_index'] is None  # not enough history
    bad = deepcopy(second)
    bad['request_fingerprint'] = forecast_request_fingerprint({**requests[1], 'unit': 'other'})
    with pytest.raises(ValueError, match='another request'):
        context(cases[1], requests[1], 'ledger_context', {}, bad)

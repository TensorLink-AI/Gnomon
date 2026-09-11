from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks.ledger_optimization.audit_equal_information import audit


@pytest.fixture
def trial(tmp_path):
    cases = tmp_path/'cases.json'
    memory = tmp_path/'memory.json'
    out = tmp_path/'run'
    out.mkdir()
    case = {'series_id': 's', 'round': 0, 'origin': '2026-01-01T00:00:00Z',
            'predictions': {'p': [1]}, 'actual': [1]}
    packet = {'series_id': 's', 'round': 0, 'request_fingerprint': 'fixture-request',
              'current_context': {}, 'raw_history': {'records': []}}
    cases.write_text(json.dumps([case]))
    memory.write_text(json.dumps({'packets': [packet]}))
    manifest = {'historical_information_contract': 'same_raw_matched_history_in_every_arm',
                'snapshot_sha256': hashlib.sha256(cases.read_bytes()).hexdigest(),
                'memory_sha256': hashlib.sha256(memory.read_bytes()).hexdigest(),
                'series': ['s'], 'rounds': [0], 'seeds_requested': [7],
                'arms': ['no_ledger', 'ledger_119', 'ledger_context'], 'expected_decisions': 3,
                'forecast_attempt_budget': 3, 'scope': 'synthetic test'}
    (out/'manifest.json').write_text(json.dumps(manifest))
    for i, arm in enumerate(manifest['arms']):
        prompt = {'request_fingerprint': packet['request_fingerprint'], 'raw_matched_history': packet['raw_history'],
                  'current_context': {}, 'recent_history': [1]}
        if arm != 'no_ledger':
            prompt['historical_evidence'] = {'cohorts': []}
        row = {'case': case, 'arm': arm, 'seed': 7, 'provider': 'p', 'fallback_used': False, 'rmsle': 0,
               'forecast_attempts': 1, 'resolution': {'execution': {'provider': 'p', 'point': [1],
                                                                 'request_fingerprint': 'fixture-request'}},
               'transcript': [{'role': 'system', 'content': 'same instructions'},
                              {'role': 'user', 'content': json.dumps(prompt)}]}
        (out/f'{i:04d}-{arm}.json').write_text(json.dumps(row))
    return out, cases, memory


def test_audit_checks_shared_exposure_and_independent_arithmetic(trial):
    result = audit(*trial)
    assert result['decisions_checked'] == 3 and result['same_shared_information']
    assert result['maximum_numerical_delta'] == 0 and not result['target_established']


@pytest.mark.parametrize('damage', ['information', 'points', 'budget', 'missing', 'score'])
def test_audit_rejects_confounding_and_incomplete_trials(trial, damage):
    out, cases, memory = trial
    path = out/'0002-ledger_context.json'
    row = json.loads(path.read_text())
    if damage == 'information':
        prompt = json.loads(row['transcript'][1]['content'])
        prompt['recent_history'] = [999]
        row['transcript'][1]['content'] = json.dumps(prompt)
    elif damage == 'points':
        row['resolution']['execution']['point'] = [999]
    elif damage == 'budget':
        row['forecast_attempts'] = 4
    elif damage == 'score':
        row['rmsle'] = .1
    if damage == 'missing':
        path.unlink()
    else:
        path.write_text(json.dumps(deepcopy(row)))
    with pytest.raises(ValueError):
        audit(out, cases, memory)

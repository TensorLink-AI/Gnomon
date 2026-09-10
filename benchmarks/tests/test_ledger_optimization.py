from copy import deepcopy
import json

import pytest

from benchmarks.ledger_optimization import agent_loop
from benchmarks.ledger_optimization.replay import screen


def test_past_only_screen_does_not_read_future_outcomes():
    def case(i, recorded):
        return {'series_id': 's', 'round': i, 'origin': f'2026-01-{i+1:02d}T00:00:00Z',
            'future_timestamps': [f'2026-01-{i+2:02d}T00:00:00Z'], 'outcome_recorded_at': recorded,
            'current_card': {'a': {'cv_rmsle': 1, 'cv_mae': 1}, 'b': {'cv_rmsle': 2, 'cv_mae': 2}},
            'scores': {'a': 4, 'b': 0}, 'mae': {'a': 4, 'b': 0},
            'arms': {a: {'intent_to_treat_rmsle': 1} for a in ('statsforecast', 'gnomon_control', 'gnomon_ledger')}}
    original = [case(0, '2026-02-01T00:00:00Z'), case(1, '2026-02-02T00:00:00Z')]
    result = screen(original)['cases']
    assert result[1]['visible_origins'] == 0
    assert result[1]['choices']['rmsle_windowall_cv0']['provider'] == 'a'
    changed = deepcopy(original)
    changed[0]['outcome_recorded_at'] = '2026-01-02T00:00:00Z'
    assert screen(changed)['cases'][1]['choices']['rmsle_windowall_cv0']['provider'] == 'b'


def test_live_runner_uses_typed_execution_and_no_actuals_in_prompt(monkeypatch):
    request = {'history': [1, 2, 3], 'horizon': 1, 'series_id': 's', 'unit': 'widgets',
               'cutoff': '2026-01-03T00:00:00Z', 'frequency': 'D', 'season': 1,
               'timestamps': ['2026-01-01T00:00:00Z', '2026-01-02T00:00:00Z', '2026-01-03T00:00:00Z'],
               'future_timestamps': ['2026-01-04T00:00:00Z']}
    case = {'series_id': 's', 'round': 0, 'origin': request['cutoff'], 'actual': [12345.6789],
            'predictions': {'a': [3], 'sf_seasonal_naive_7': [2]},
            'current_card': {'a': {'cv_rmsle': 0.1}}, 'legacy_evidence': {}}
    calls = []
    def chat(messages, tools, seed, key):
        calls.append(1)
        assert '12345.6789' not in json.dumps(messages)
        if len(calls) == 1:
            name, args = 'gnomon_forecast', {'provider': 'a'}
        else:
            reply = json.loads(messages[-1]['content'])
            name, args = 'select_forecast', {'execution_id': reply['completion']['execution_id']}
        return {'choices': [{'message': {'role': 'assistant', 'content': None,
            'tool_calls': [{'id': str(len(calls)), 'type': 'function',
                           'function': {'name': name, 'arguments': json.dumps(args)}}]}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 10}}
    monkeypatch.setattr(agent_loop, 'request_chat', chat)
    result = agent_loop.play(case, request, 'v1', 'no_ledger', {}, 7, 'unused-test-key')
    assert result['provider'] == 'a' and not result['fallback_used']
    assert result['successful_executions'] == 1
    assert result['api_calls'] == 2 and result['api_cost_usd'] is None
    assert result['resolution']['status'] == 'explicit_selection'


def test_agent_does_not_receive_oracle_scores():
    request = {'series_id': 's', 'unit': 'w', 'horizon': 1, 'cutoff': None,
               'frequency': None, 'season': 1, 'history': [1, 2]}
    case = {'predictions': {'a': [999]}, 'actual': [777], 'scores': {'a': 666},
            'current_card': {}, 'legacy_evidence': {}}
    for arm in agent_loop.ARMS:
        message = json.dumps(agent_loop.context(case, request, arm, {}))
        for forbidden in ('999', '777', '666', '"actual"', '"scores"'):
            assert forbidden not in message


def test_metric_rejects_negative_actuals():
    with pytest.raises(Exception, match='nonnegative_actuals'):
        agent_loop.rmsle([(1, -1)], 'clip_zero')

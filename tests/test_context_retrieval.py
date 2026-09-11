from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest

from gnomon import GnomonSession, TemporalLedger
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.ids import FixedClock
from gnomon.session import ledger_schema


@pytest.fixture
def history(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ledger = TemporalLedger(tmp_path/'memory.db', clock=FixedClock(start))
    session = GnomonSession.from_config(ledger=ledger)
    runs = []
    for i in range(4):
        origin = start+timedelta(days=3*i)
        ledger.clock = FixedClock(origin)
        request = dict(history=[1, 3], horizon=1, series_id='sales', unit='widgets', frequency='D',
                       timestamps=[(origin-timedelta(days=1)).isoformat(), origin.isoformat()],
                       cutoff=origin.isoformat(), future_timestamps=[(origin+timedelta(days=1)).isoformat()])
        executions = [session.forecast(p, request) for p in ('last_value', 'historical_mean')]
        context = {'sparsity': 'low', 'trend': 'rising' if i < 2 else 'stable'}
        ledger.record_decision_summary(execution_id=executions[0]['execution_id'], rationale='Synthetic context',
            assumptions=[], invalidation_conditions=[], context=[{
                'key': k, 'value': v, 'valid_from': origin.isoformat(),
                'valid_to': (origin+timedelta(days=1)).isoformat(),
                'source_available_at': origin.isoformat(), 'source_ref': 'synthetic:known-at-origin'}
                for k, v in context.items()])
        ledger.clock = FixedClock(origin+timedelta(days=1))
        ledger.append_actual(series_id='sales', unit='widgets', valid_time=request['future_timestamps'][0],
                             value=100 if i < 2 else 3, source_available_at=ledger._now())
        runs = executions
    query = dict(series_id='sales', horizon=1, unit='widgets', providers={r['provider']: r['revision'] for r in runs},
                 start=start.isoformat(), end=(start+timedelta(days=9)).isoformat(),
                 source_as_of=(start+timedelta(days=11)).isoformat(), recorded_as_of=(start+timedelta(days=11)).isoformat(),
                 context_candidates=[{'sparsity': 'low', 'trend': 'rising'}, {'sparsity': 'low'}, {}])
    yield ledger, session, query
    session.close()


def test_first_sufficient_cohort_not_best_observed_score(history, monkeypatch):
    ledger, session, query = history
    calls, connections = session.engine._execution_stats['provider_calls'], []
    writes = ledger._committed_row_writes
    connect = ledger._connect
    @contextmanager
    def counted():
        connections.append(1)
        with connect() as conn:
            yield conn
    monkeypatch.setattr(ledger, '_connect', counted)
    result = ledger.retrieve_context(**query, min_origins=2)
    assert result['selected_index'] == 0 and result['context_specific'] and not result['broadened']
    assert result['cohorts'][0]['evidence_summary']['lifetime']['ranking'][0]['score'] > result['cohorts'][1]['evidence_summary']['lifetime']['ranking'][0]['score']
    assert result['comparison']['matched_origins'] == 2
    assert len(connections) == 1 and ledger._committed_row_writes == writes
    assert session.engine._execution_stats['provider_calls'] == calls


def test_broadening_is_explicit_and_insufficient_counts_do_not_select(history):
    ledger, _, query = history
    result = ledger.retrieve_context(**query, min_origins=3)
    assert result['selected_index'] == 1 and result['broadened']
    assert result['selected_filters'] == {'sparsity': 'low'}
    assert [c['matched_origins'] for c in result['cohorts']] == [2, 4, 4]
    insufficient = ledger.retrieve_context(**query, min_origins=5)
    assert insufficient['status'] == 'insufficient_evidence'
    assert insufficient['comparison'] is None and insufficient['selected_index'] is None


def test_recording_visibility_still_excludes_later_outcomes(history):
    ledger, _, query = history
    result = ledger.retrieve_context(**{**query, 'recorded_as_of': '2026-01-01T00:00:00Z'}, min_origins=1)
    assert result['selected_index'] is None
    assert all(c['matched_origins'] == 0 for c in result['cohorts'])


def test_parsing_reuse_never_hides_new_actual_revisions_between_calls(history):
    ledger, _, query = history
    before = ledger.retrieve_context(**query, min_origins=2)
    ledger.append_actual(series_id='sales', unit='widgets', valid_time='2026-01-02T00:00:00Z',
                         value=3, source_available_at=ledger._now())
    after = ledger.retrieve_context(**query, min_origins=2)
    assert before['comparison']['models'] != after['comparison']['models']
    assert before['comparison']['models'][0]['mae'] == 97
    assert after['comparison']['models'][0]['mae'] == 48.5


@pytest.mark.parametrize('candidates', [[], [{}, {'a': 'b'}], [{'a': 'b'}, {'a': 'c'}], [{'a': 'b'}, {'a': 'b'}], [None]])
def test_retrieval_cannot_switch_the_task_or_duplicate_candidates(history, candidates):
    ledger, _, query = history
    with pytest.raises(ForecastAdapterError):
        ledger.retrieve_context(**{**query, 'context_candidates': candidates})


def test_public_schema_and_session_dispatch_share_contract(history):
    _, session, query = history
    schema = next(v for v in ledger_schema()['oneOf'] if v['properties']['operation']['const'] == 'retrieve_context')
    assert schema['properties']['min_origins']['default'] == 4
    assert 'min_origins' not in schema['required'] and 'context_candidates' in schema['required']
    response = session.call('gnomon_ledger', {'operation': 'retrieve_context', **query}, compact=False)
    assert response['result']['selected_index'] == 1
    assert response['result']['provider_calls'] == response['result']['ledger_writes'] == 0

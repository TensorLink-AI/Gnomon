from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import subprocess
import sys

import pytest

from gnomon import GnomonSession, TemporalLedger, put_lesson
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.ids import FixedClock
from gnomon.contracts import GnomonError
from gnomon.session import ledger_schema


@pytest.fixture
def case(tmp_path):
    origin = datetime(2026, 1, 20, tzinfo=timezone.utc)
    ledger = TemporalLedger(tmp_path / 'evidence.db', clock=FixedClock(origin))
    session = GnomonSession.from_config(ledger=ledger)
    session.allow_outcome_writes = True
    request = dict(history=[10, 11, 12], horizon=2, series_id='sales', unit='widgets',
        timestamps=[(origin - timedelta(days=i)).isoformat() for i in (2, 1, 0)],
        future_timestamps=[(origin + timedelta(days=i)).isoformat() for i in (1, 2)],
        cutoff=origin.isoformat(), frequency='D')
    runs = [session.forecast(p, request) for p in ('last_value', 'historical_mean')]
    args = dict(execution_id=runs[0]['execution_id'], rationale='Latest level during a planned promotion.',
                assumptions=['Schedule is applicable.'], invalidation_conditions=['Promotion cancelled.'],
                context=[dict(key='promotion', value='planned', valid_from=origin.isoformat(),
                    valid_to=(origin + timedelta(days=3)).isoformat(),
                    source_available_at=(origin - timedelta(days=1)).isoformat(), source_ref='synthetic:plan')])
    query = dict(series_id='sales', unit='widgets', horizon=2,
        providers={r['provider']: r['revision'] for r in runs},
        start=origin.isoformat(), end=origin.isoformat(), context_filters={'promotion': 'planned'})
    yield ledger, session, origin, request, args, query
    session.close()


def mature(ledger, origin, request, values=(13, 14)):
    available = origin + timedelta(days=3)
    ledger.clock = FixedClock(available)
    for t, value in zip(request['future_timestamps'], values):
        ledger.append_actual(series_id='sales', unit='widgets', valid_time=t, value=value,
                             source_available_at=available.isoformat())
    return dict(source_as_of=available.isoformat(), recorded_as_of=available.isoformat())


def test_complete_lifecycle_immutable_versions_retry_and_no_dispatch(case):
    ledger, session, origin, request, args, query = case
    created = ledger.record_decision_summary(**args)
    did = created['decision_id']
    assert created['summary']['context'][0]['recorded_at'] == ledger._now()
    original = ledger.decision(did)
    before = session.engine._execution_stats['provider_calls']
    writes = ledger._committed_row_writes
    pending = ledger.review_decision(decision_id=did, source_as_of=origin.isoformat(), recorded_as_of=origin.isoformat())
    assert pending['scoring_status'] == 'pending' and not pending['review_ready']
    assert ledger._committed_row_writes == writes
    cutoffs = mature(ledger, origin, request)
    writes = ledger._committed_row_writes
    comparison = ledger.compare_context(**query, **cutoffs)
    assert comparison['matched_origins'] == 1 and comparison['n'] == 2
    assert [m['mae'] for m in comparison['models']] == [1.5, 2.5]
    assert comparison['uncertainty']['confidence_interval'] is None
    review = ledger.review_decision(decision_id=did, **cutoffs)
    assert review['metrics']['mae'] == 1.5 and review['metrics']['bias'] == -1.5
    assert review['review_ready'] and not review['business_explanation_validated']
    assert ledger._committed_row_writes == writes
    first = ledger.record_lesson(decision_id=did, lesson='First hypothesis.', **cutoffs)
    writes = ledger._committed_row_writes
    assert ledger.record_lesson(decision_id=did, lesson='First hypothesis.', **cutoffs)['reused']
    assert ledger._committed_row_writes == writes
    exported = ledger.export_lesson(lesson_id=first['lesson_id'], recorded_as_of=cutoffs['recorded_as_of'])
    ledger.clock = FixedClock(origin + timedelta(days=4))
    with pytest.raises(ForecastAdapterError, match='latest lesson'):
        ledger.record_lesson(decision_id=did, lesson='Changed hypothesis.', **cutoffs)
    second = ledger.record_lesson(decision_id=did, lesson='Changed hypothesis.', previous_lesson_id=first['lesson_id'], **cutoffs)
    assert second['version'] == 2
    assert ledger.export_lesson(lesson_id=first['lesson_id'], recorded_as_of=ledger._now()) == exported
    assert ledger.decision(did)['inputs'] == original['inputs']
    assert session.engine._execution_stats['provider_calls'] == before
    assert ledger.SCHEMA_VERSION == 4


def test_visibility_units_partial_and_revision_review(case):
    ledger, _, origin, request, args, _ = case
    did = ledger.record_decision_summary(**args)['decision_id']
    cuts = mature(ledger, origin, request, values=(13,))
    partial = ledger.review_decision(decision_id=did, **cuts)
    assert partial['scoring_status'] == 'partial'
    ledger.append_actual(series_id='sales', unit='kg', valid_time=request['future_timestamps'][1], value=14,
                         source_available_at=cuts['source_as_of'])
    assert ledger.review_decision(decision_id=did, **cuts)['coverage']['other_units_at_missing_steps'] == ['kg']
    with pytest.raises(ForecastAdapterError, match='complete matching-unit'):
        ledger.record_lesson(decision_id=did, lesson='Not ready.', **cuts)
    assert ledger.review_decision(decision_id=did, source_as_of=origin.isoformat(), recorded_as_of=cuts['recorded_as_of'])['scoring_status'] == 'pending'
    assert ledger.review_decision(decision_id=did, source_as_of=cuts['source_as_of'], recorded_as_of=origin.isoformat())['scoring_status'] == 'pending'
    ledger.append_actual(series_id='sales', unit='widgets', valid_time=request['future_timestamps'][1], value=14,
                         source_available_at=cuts['source_as_of'])
    first = ledger.record_lesson(decision_id=did, lesson='Old outcome.', **cuts)
    old = ledger.export_lesson(lesson_id=first['lesson_id'], recorded_as_of=cuts['recorded_as_of'])
    ledger.clock = FixedClock(origin + timedelta(days=4))
    ledger.append_actual(series_id='sales', unit='widgets', valid_time=request['future_timestamps'][1], value=20,
                         source_available_at=ledger._now())
    latest = ledger.review_decision(decision_id=did, source_as_of=ledger._now(), recorded_as_of=ledger._now())
    assert latest['metrics']['mae'] == 4.5
    assert ledger.export_lesson(lesson_id=first['lesson_id'], recorded_as_of=ledger._now()) == old
    assert ledger.review_decision(decision_id=did, **cuts)['metrics']['mae'] == 1.5
    with pytest.raises(ForecastAdapterError, match='does not exist'):
        ledger.export_lesson(lesson_id=first['lesson_id'], recorded_as_of=origin.isoformat())


@pytest.mark.parametrize('condition,reason', [
    ('missing', 'no_visible_context'), ('late_recording', 'no_visible_context'),
    ('late_source', 'no_visible_context'), ('expired', 'no_visible_context'),
    ('conflict', 'conflicting_context_labels'), ('different', 'context_filter_mismatch')])
def test_context_never_infers_or_leaks_labels(case, condition, reason):
    ledger, _, origin, request, args, query = case
    if condition == 'late_recording':
        ledger.clock = FixedClock(origin + timedelta(days=2))
    if condition == 'late_source':
        ledger.clock = FixedClock(origin + timedelta(hours=1))
        args['context'][0]['source_available_at'] = ledger._now()
    if condition == 'expired':
        args['context'][0]['valid_from'] = (origin - timedelta(days=3)).isoformat()
        args['context'][0]['valid_to'] = origin.isoformat()
    if condition == 'different':
        args['context'][0]['value'] = 'cancelled'
    if condition != 'missing':
        ledger.record_decision_summary(**args)
    if condition == 'conflict':
        args['context'][0]['value'] = 'cancelled'
        ledger.record_decision_summary(**args)
    cuts = mature(ledger, origin, request)
    result = ledger.compare_context(**query, **cuts)
    assert result['matched_origins'] == 0 and result['status'] == 'insufficient_evidence'
    assert result['excluded'][0]['reason'] == reason
    assert result['models'] == []


def test_permissions_schemas_recovery_and_measured_counts(case):
    ledger, session, origin, request, args, query = case
    session.allow_outcome_writes = False
    with pytest.raises(GnomonError) as denied:
        session.call('gnomon_ledger', {'operation': 'record_decision_summary', **args}, compact=False)
    assert denied.value.code == 'OUTCOME_WRITES_DISABLED'
    visible = {v['properties']['operation']['const'] for v in ledger_schema()['oneOf']}
    assert 'record_decision_summary' not in visible and 'review_decision' in visible
    session.allow_outcome_writes = True
    response = session.call('gnomon_ledger', {'operation': 'record_decision_summary', **args}, compact=False)
    assert response['execution_diagnostics']['ledger_writes'] == 1
    did = response['result']['decision_id']
    reviewed = session.call('gnomon_ledger', dict(operation='review_decision', decision_id=did,
        source_as_of=origin.isoformat(), recorded_as_of=origin.isoformat()), compact=False)
    assert reviewed['status'] == 'ok' and reviewed['scoring_status'] == 'pending'
    assert not reviewed['task_completed'] and reviewed['execution_diagnostics']['ledger_writes'] == 0
    with pytest.raises(GnomonError) as invalid:
        session.call('gnomon_ledger', dict(operation='review_decision', decision_id=did), compact=False)
    payload = invalid.value.to_dict()
    assert payload['error']['details']['example_arguments']['operation'] == 'review_decision'
    assert payload['error']['recovery']['cause']
    cutoffs = mature(ledger, origin, request)
    response = session.call('gnomon_ledger', dict(operation='compare_context', **query, **cutoffs), compact=False)
    assert response['query']['cutoff_default'] == 'required'
    assert response['execution_diagnostics']['provider_calls'] == 0
    assert response['execution_diagnostics']['ledger_writes'] == 0


def test_reject_invalid_summaries_without_writes(case):
    ledger, _, origin, _, args, _ = case
    for field, value in [('rationale', 'x'*1001), ('context', {}), ('assumptions', 'text'),
                         ('evidence_refs', [{'kind': 'execution', 'id': 'missing'}])]:
        invalid = deepcopy(args)
        invalid[field] = value
        writes = ledger._committed_row_writes
        with pytest.raises(ForecastAdapterError):
            ledger.record_decision_summary(**invalid)
        assert ledger._committed_row_writes == writes
    args['context'][0]['source_available_at'] = (origin + timedelta(days=1)).isoformat()
    with pytest.raises(ForecastAdapterError, match='future source'):
        ledger.record_decision_summary(**args)


def test_portable_memory_adapter_and_example(case, tmp_path):
    ledger, _, origin, request, args, _ = case
    did = ledger.record_decision_summary(**args)['decision_id']
    cuts = mature(ledger, origin, request)
    lesson = ledger.record_lesson(decision_id=did, lesson='A hypothesis, not a cause.', **cuts)
    exported = ledger.export_lesson(lesson_id=lesson['lesson_id'], recorded_as_of=cuts['recorded_as_of'])
    calls = []

    class Store:
        def put(self, namespace, key, value):
            calls.append((namespace, key, value))
            value['lesson'] = 'Mutated by consumer'

    original = deepcopy(exported)
    result = put_lesson(Store(), ('user', 'lessons'), exported)
    assert exported == original and result['memory_key'] == lesson['lesson_id']
    assert len(calls) == 1
    env_result = subprocess.run([sys.executable, '-m', 'gnomon.examples.decision_memory'], cwd=tmp_path,
                                text=True, capture_output=True)
    assert env_result.returncode == 0, env_result.stderr
    assert json.loads(env_result.stdout)['review_ready']


def test_context_subset_reaggregates_and_recording_cutoff_hides_later_labels(case):
    ledger, session, origin, request, args, query = case
    ledger.record_decision_summary(**args)
    later = origin + timedelta(days=7)
    ledger.clock = FixedClock(later)
    next_request = deepcopy(request)
    for field in ('timestamps', 'future_timestamps'):
        next_request[field] = [(datetime.fromisoformat(t) + timedelta(days=7)).isoformat() for t in request[field]]
    next_request['cutoff'] = later.isoformat()
    runs = [session.forecast(p, next_request) for p in query['providers']]
    next_args = deepcopy(args)
    next_args['execution_id'] = runs[0]['execution_id']
    next_args['context'][0].update(value='cancelled', valid_from=later.isoformat(),
        valid_to=(later + timedelta(days=3)).isoformat(), source_available_at=later.isoformat())
    ledger.record_decision_summary(**next_args)
    final = later + timedelta(days=3)
    ledger.clock = FixedClock(final)
    for req, values in ((request, (13, 14)), (next_request, (30, 30))):
        for t, v in zip(req['future_timestamps'], values):
            ledger.append_actual(series_id='sales', unit='widgets', valid_time=t, value=v,
                                 source_available_at=final.isoformat())
    query['end'] = later.isoformat()
    cuts = dict(source_as_of=final.isoformat(), recorded_as_of=final.isoformat())
    result = ledger.compare_context(**query, **cuts)
    assert result['matched_origins'] == 1 and result['models'][0]['mae'] == 1.5
    query['context_filters'] = {'promotion': 'cancelled'}
    other = ledger.compare_context(**query, **cuts)
    assert other['matched_origins'] == 1 and other['models'][0]['mae'] == 18
    assert len(other['origins'][0]['context_evidence']['decision_ids']) == 1


def test_cli_and_real_mcp_ledger_memory(case, tmp_path):
    ledger, _, origin, request, args, _ = case
    did = ledger.record_decision_summary(**args)['decision_id']
    cuts = mature(ledger, origin, request)
    arguments = dict(operation='review_decision', decision_id=did, **cuts)
    cli = subprocess.run([sys.executable, '-m', 'gnomon', 'ledger', '--ledger-path', str(ledger.path),
                          '--arguments', json.dumps(arguments)], text=True, capture_output=True)
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli.stdout)['review_ready']
    messages = [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2025-06-18',
             'capabilities': {}, 'clientInfo': {'name': 'decision-memory-test', 'version': '1'}}},
        {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
        {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {'name': 'gnomon_ledger', 'arguments': arguments}},
        {'jsonrpc': '2.0', 'id': 4, 'method': 'tools/call', 'params': {'name': 'gnomon_ledger',
            'arguments': {'operation': 'record_lesson', 'decision_id': did, 'lesson': 'A hypothesis.', **cuts}}},
    ]
    config = tmp_path / 'read.toml'
    config.write_text('ledger_path = ' + json.dumps(str(ledger.path)) + '\n[result_limits]\nmax_response_bytes = 65536\n')
    process = subprocess.run([sys.executable, '-m', 'gnomon', 'mcp', 'serve', '--providers-config', str(config)],
        input=''.join(json.dumps(m)+'\n' for m in messages), text=True, capture_output=True, timeout=20)
    assert process.returncode == 0, process.stderr
    frames = {r['id']: r for r in map(json.loads, process.stdout.splitlines())}
    schema = next(t for t in frames[2]['result']['tools'] if t['name'] == 'gnomon_ledger')['inputSchema']
    assert 'review_decision' in {v['properties']['operation']['const'] for v in schema['oneOf']}
    result = frames[3]['result']
    assert not result['isError'] and result['structuredContent']['review_ready']
    assert result['structuredContent']['execution_diagnostics']['provider_calls'] == 0
    assert result['structuredContent']['execution_diagnostics']['ledger_writes'] == 0
    assert frames[4]['result']['isError']


def test_structured_kinds_cannot_bypass_validation_via_legacy_operations(case):
    ledger, _, _, _, args, _ = case
    with pytest.raises(ForecastAdapterError, match='Use record_decision_summary'):
        ledger.record_decision(execution_ids=[args['execution_id']], policy={},
                               inputs={'kind': 'forecast_decision_summary/1'}, action={})
    did = ledger.record_decision_summary(**args)['decision_id']
    with pytest.raises(ForecastAdapterError, match='Use record_lesson'):
        ledger.append_decision_outcome(did, outcome={'kind': 'forecast_lesson/1'}, source_available_at=ledger._now())
    schema = ledger_schema(allow_outcome_writes=True)
    assert schema['defaults_matrix']['record_decision_summary']['source_as_of'] == 'not_applicable'
    assert schema['defaults_matrix']['export_lesson']['source_as_of'] == 'not_applicable'
    assert schema['defaults_matrix']['export_lesson']['recorded_as_of'] == 'required'

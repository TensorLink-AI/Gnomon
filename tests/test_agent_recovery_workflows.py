"""Release gates for task fidelity, strict routing and complete MCP evidence."""
import hashlib
import json
import sqlite3

import pytest

from gnomon.contracts import GnomonError
from gnomon.mcp_server import _handle
from gnomon.recovery import temporal_recovery
from gnomon.result_refs import ResultLimits, ResultReferences, encode
from gnomon.temporal_ops import temporal_operation
from test_acceptance_recovery import cli
from test_backtesting import at, configured, source
from test_study_routing import task


def test_gap_rejection_reports_a_lower_bound_without_changing_source(tmp_path):
    path = tmp_path / 'gap.csv'
    original = 'timestamp,value\n' + ''.join(
        f'{at(i).isoformat()},{i}\n' for i in range(1, 101) if not 41 <= i <= 60)
    path.write_text(original)
    code, answer = cli('infer', '--input', path, '--provider', 'last_value',
                       '--horizon', 2, '--frequency', 'D', '--repair', 'aggressive')
    assert code == 2 and answer['error']['code'] == 'EXCESSIVE_REPAIR'
    details = answer['error']['details']
    assert details['repair_counts'] == {'gap_run_at_least': 9}
    assert details['max_gap_run'] == 8 and details['total_observations'] == 80
    assert details['scan_complete'] is False
    assert details['count_scope'] == 'lower_bound_at_rejection'
    assert path.read_text() == original


@pytest.mark.parametrize('mismatch', [False, True])
def test_strict_routing_rejects_fallback_without_executions_or_rescores(tmp_path, mismatch):
    session, ref = configured(tmp_path, ledger=True)
    with session:
        study = session.evaluate(ref, candidates=['trend'], baseline='last_value', horizon=2,
                                 folds=4 if mismatch else 2)
        args = task(session, ref, study)
        if mismatch:
            args['horizon'] = 3
        original = session.ledger.study(study['study_id'])
        with sqlite3.connect(session.ledger.path) as conn:
            before = conn.iterdump()
            before = list(before)
        default = session.call('gnomon_route', args)
        assert default['fallback_used'] and not default['evidence_based']
        assert default['routing_status'] == 'fallback' and default['warning']
        with pytest.raises(GnomonError) as exc:
            session.call('gnomon_route', {**args, 'require_evidence': True})
        assert exc.value.code == 'ROUTING_EVIDENCE_REQUIRED'
        response = _handle({'method': 'tools/call', 'params': {
            'name': 'gnomon_route', 'arguments': {**args, 'require_evidence': True}}}, session=session)
        assert response['isError']
        assert response['structuredContent']['error']['code'] == 'ROUTING_EVIDENCE_REQUIRED'
        assert session.ledger.study(study['study_id']) == original
        with sqlite3.connect(session.ledger.path) as conn:
            assert list(conn.iterdump()) == before
        if not mismatch:
            next_call = default['next_actions'][0]
            assert next_call['arguments'] == dict(data_ref=ref, series_id=study['series_id'],
                candidates=['trend'], baseline='last_value', horizon=2, season=1, folds=3)
            assert next_call['admissible'] is None and next_call['requires_provider_calls']
            # A caller deliberately chooses the proposed new evaluation; it
            # must then route successfully with the original task parameters.
            more = session.call(next_call['tool'], next_call['arguments'])
            selected = session.call('gnomon_route', {**args, 'study_id': more['study_id'], 'require_evidence': True})
            assert selected['evidence_based'] and selected['routing_status'] == 'selected'
            assert selected['provider_calls'] == 0


def test_cli_strict_flag_rejects_mismatch_and_preserves_default(tmp_path):
    path = source(tmp_path)
    ledger = tmp_path / 'cli.db'
    code, study = cli('evaluate', '--input', path, '--candidates', 'historical_mean',
                      '--baseline', 'last_value', '--horizon', 2, '--ledger-path', ledger)
    assert code == 0
    args = ('route', '--input', path, '--study', study['study_id'], '--horizon', 3,
            '--source-as-of', at(30).isoformat(), '--recorded-as-of', '2099-01-01T00:00:00Z', '--ledger-path', ledger)
    assert cli(*args)[1]['fallback_used']
    code, error = cli(*args, '--require-evidence')
    assert code == 2 and error['error']['code'] == 'ROUTING_EVIDENCE_REQUIRED'


def test_strict_routing_allows_evidence_supported_baseline_tie(tmp_path):
    from test_backtesting import result
    session, ref = configured(tmp_path, ledger=True)
    with session:
        session.engine.register('same', result, revision='same-v1', deterministic=True)
        study = session.evaluate(ref, candidates=['same'], baseline='last_value', horizon=2)
        answer = session.call('gnomon_route', {**task(session, ref, study),
                              'candidates': ['same'], 'require_evidence': True})
        assert answer['recommendation'] == 'last_value' and answer['evidence_based']
        assert not answer['fallback_used'] and answer['selection_reason'] == 'baseline_tied_for_best'
        assert answer['ranking_policy']['ties'] and answer['provider_calls'] == 0


def test_bounded_routing_receipt_preserves_fallback_semantics():
    with_store = ResultReferences(ResultLimits(max_response_bytes=2048))
    try:
        value = {'status': 'ok', 'fallback_used': True, 'evidence_based': False,
                 'routing_status': 'fallback', 'recommendation': 'last_value',
                 'reason': 'task_identity_mismatch', 'next_step': 'search_for_a_matching_task_study',
                 'excluded_folds': ['details' * 1000]}
        receipt = with_store.project(value)
        assert len(encode(receipt).encode()) <= 2048
        for key in ('fallback_used', 'evidence_based', 'routing_status', 'recommendation', 'reason', 'next_step'):
            assert receipt['summary'][key] == value[key]
    finally:
        with_store.close()


def mcp(session, name, arguments):
    response = _handle({'method': 'tools/call', 'params': {'name': name, 'arguments': arguments}}, session=session)
    assert not response['isError']
    return response['structuredContent']


def unpack(session, receipt):
    if receipt.get('status') != 'result_available':
        return receipt
    call = receipt['full_result']
    text, offset = '', 0
    while True:
        page = mcp(session, call['tool'], {**call['arguments'], 'offset': offset})
        text += page['text']
        assert page['total_chars'] == receipt['total_chars']
        assert page['root_sha256'] == receipt['root_sha256']
        offset = page['next_offset']
        if offset is None:
            break
    assert len(text) == receipt['total_chars']
    assert hashlib.sha256(text.encode()).hexdigest() == receipt['root_sha256']
    return json.loads(text)


@pytest.mark.parametrize('ledger', [False, True])
def test_mcp_advertised_calls_retrieve_full_folds_without_new_predictions(tmp_path, ledger):
    session, ref = configured(tmp_path, ledger=ledger)
    with session:
        session.results.close()
        session.results = ResultReferences(ResultLimits(max_response_bytes=2048))
        calls = []
        original = session.engine.forecast

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        session.engine.forecast = counted
        receipt = mcp(session, 'gnomon_evaluate', dict(data_ref=ref, candidates=['trend'], baseline='last_value', horizon=2))
        assert len(encode(receipt).encode()) <= 2048
        assert receipt['full_result_scope'] == 'response_payload'
        compact = unpack(session, receipt)
        assert compact['study_evidence_scope'] == 'fold_summary'
        assert 'runs' not in compact['folds'][0]
        count = len(calls)
        full_call = receipt['full_study']
        full = unpack(session, mcp(session, full_call['tool'], full_call['arguments']))
        assert full['study_evidence_scope'] == 'full_folds'
        assert full['study_id'] == compact['study_id']
        assert all({'request', 'actuals', 'runs'} <= fold.keys() for fold in full['folds'])
        assert len(calls) == count == 8


def test_temporal_recovery_corrects_type_without_replacing_number_or_adding_boilerplate():
    supplied = dict(operation='shift', value='2031-05-17', amount='7', unit='days')
    recovery = temporal_recovery(supplied)
    assert recovery['example_arguments'] == {**supplied, 'amount': 7, 'mode': 'calendar'}
    assert recovery['changed_fields'] == ['amount', 'mode']
    assert set(recovery['preserved_fields']) == {'operation', 'value', 'unit'}
    assert recovery['example_runnable']
    assert 'month' not in recovery['guidance'] and 'clamp' not in recovery['guidance']
    assert temporal_operation(**recovery['example_arguments'])['result']['date'] == '2031-05-24'


@pytest.mark.parametrize('supplied', [
    dict(operation='normalize', value='2026-03-08T02:30:00', timezone='America/New_York'),
    dict(operation='normalize', value='2026-01-01'),
    dict(operation='shift', value='2026-01-01', amount=7, unit='days', mode='calendar', timezone='UTC'),
])
def test_unresolved_temporal_facts_are_mechanically_nonrunnable(supplied):
    recovery = temporal_recovery(supplied)
    assert recovery['example_arguments'] == supplied
    assert not recovery['example_runnable'] and recovery['resolution_required']
    assert recovery['example_kind'] == 'task_template'
    with pytest.raises(ValueError):
        temporal_operation(**supplied)

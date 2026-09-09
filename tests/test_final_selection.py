"""Host-side selection fidelity, independent of predictive accuracy or prose."""
from copy import deepcopy
import json

import pytest

from gnomon import ForecastRequest, GnomonSession, forecast_completion, forecast_request_fingerprint, resolve_final_selection
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.mcp_server import _handle
from gnomon.result_refs import ResultLimits, ResultReferences
from test_acceptance_recovery import cli


def request(**changes):
    return dict(history=[1, 2, 3], horizon=2, series_id='item_123_store_4', unit='widgets',
        timestamps=['2026-01-01T00:00:00Z', '2026-01-02T00:00:00Z', '2026-01-03T00:00:00Z'],
        future_timestamps=['2026-01-04T00:00:00Z', '2026-01-05T00:00:00Z'], **changes)


@pytest.fixture
def session():
    with GnomonSession.from_config() as session:
        yield session


def resolve(final, executions, expected=None):
    return resolve_final_selection(final_answer=final, successful_executions=executions,
        expected_request=expected or request())


@pytest.mark.parametrize('final', [None, '', 'Provider chosen: last_value',
    '| date | forecast |\n|---|---|\n| Jan 4 | 3 |', 'The seasonal model looks best.'])
def test_missing_machine_selection_recovers_only_execution_without_scraping(session, final):
    execution = session.forecast('last_value', request())
    before = session._execution_counts()
    result = resolve(final, [execution])
    assert result['status'] == 'recovered_single_execution'
    assert result['resolved'] and result['engine_execution_succeeded'] and result['end_to_end_completed']
    assert not result['final_answer_conformant']
    assert result['execution'] == execution['completion']
    assert session._execution_counts() == before
    assert all(result['execution_diagnostics'][k] == 0 for k in ('provider_calls', 'forecast_calls', 'ledger_writes', 'source_mutations'))


def test_canonical_completion_on_engine_python_cli_and_mcp(session):
    raw = session.engine.forecast('last_value', request())
    completion = forecast_completion(raw)
    assert completion == raw.completion() == raw.to_dict()['completion']
    assert completion['series_id'] == request()['series_id']
    assert completion['horizon'] == 2 and completion['unit'] == 'widgets'
    assert completion['future_timestamps'] == ['2026-01-04T00:00:00+00:00', '2026-01-05T00:00:00+00:00']
    assert resolve(json.dumps(completion), [raw])['status'] == 'explicit_selection'
    assert resolve(completion, [raw.to_dict()])['final_answer_conformant']
    for provider in ('last_value', 'historical_mean'):
        code, payload = cli('forecast', '--provider', provider, '--request', json.dumps(request()))
        assert code == 0
        resolved = resolve(payload['final_selection']['example'], [payload])
        assert resolved['resolved'] and resolved['final_answer_conformant']
        mcp = _handle({'method': 'tools/call', 'params': {'name': 'gnomon_forecast', 'arguments': {'provider': provider, 'request': request()}}}, session=session)
        assert not mcp['isError']
        assert resolve(None, [mcp['structuredContent']])['resolved']


@pytest.mark.parametrize('alias', ['selected_provider', 'candidate'])
def test_documented_aliases_resolve_but_do_not_improve_strict_conformance(session, alias):
    execution = session.forecast('last_value', request())
    result = resolve({alias: 'last_value'}, [execution])
    assert result['status'] == 'explicit_selection' and not result['final_answer_conformant']
    assert result['normalizations'] == [{'from': alias, 'to': 'provider'}]
    assert result['execution']['provider'] == 'last_value'


@pytest.mark.parametrize('final', [
    {'provider': 'other'}, {'execution_id': 'failed-id'}, {'provder': 'last_value'},
    {'provider': 'last_value', 'confidence': 1}, {'provider': 'last_value', 'candidate': 'other'},
    {'provider': 'last_value', 'point': [True, 3]}, {'provider': 12},
    '{"provider":"last_value", "provider":"other"}', '{"provider":',
    '{"provider":"last_value", "point":[NaN,3]}', ['last_value'],
])
def test_conflicting_or_invalid_final_never_triggers_single_execution_recovery(session, final):
    execution = session.forecast('last_value', request())
    result = resolve(final, [execution])
    assert result['status'] == 'conflicting_final_selection'
    assert not result['resolved'] and not result['end_to_end_completed']
    assert result['execution'] is None


def test_failed_entries_do_not_mask_a_success_but_cannot_be_selected(session):
    good = session.forecast('last_value', request())
    failed = {'status': 'error', 'execution_id': 'failed-id', 'provider': 'broken'}
    result = resolve(None, [failed, good])
    assert result['resolved'] and result['ignored_failed_execution_count'] == 1
    assert resolve({'execution_id': 'failed-id'}, [failed, good])['status'] == 'conflicting_final_selection'
    for executions in ([], [failed]):
        result = resolve(None, executions)
        assert result['status'] == 'no_successful_execution' and not result['engine_execution_succeeded']


def test_multiple_successes_and_explicit_selection(session):
    a = session.forecast('last_value', request())
    b = session.forecast('historical_mean', request())
    assert resolve(None, [a, b])['status'] == 'ambiguous_multiple_executions'
    assert resolve('Use historical_mean', [a, b])['status'] == 'ambiguous_multiple_executions'
    assert resolve({'provider': 'historical_mean'}, [a, b])['execution'] == b['completion']
    c = session.forecast('last_value', request())
    assert resolve({'provider': 'last_value'}, [a, b, c])['cause'] == 'execution_id_required'
    assert resolve({'execution_id': c['execution_id']}, [a, b, c])['execution'] == c['completion']
    assert resolve({'execution_id': c['execution_id'], 'provider': 'historical_mean'}, [a, b, c])['status'] == 'conflicting_final_selection'
    # Repeated receipt of one ID is not a second execution.
    assert resolve(None, [a, deepcopy(a)])['resolved']


@pytest.mark.parametrize('changes', [
    {'series_id': 'other'}, {'unit': 'kg'}, {'history': [9, 8, 7]}, {'season': 2},
    {'cutoff': '2026-01-03T00:00:00Z'}, {'recorded_time_cutoff': '2026-01-03T00:00:00Z'},
    {'horizon': 1, 'future_timestamps': ['2026-01-04T00:00:00Z']},
    {'future_timestamps': ['2026-02-04T00:00:00Z', '2026-02-05T00:00:00Z']},
])
def test_unrelated_executions_and_explicit_selections_fail_closed(session, changes):
    other = session.forecast('last_value', {**request(), **changes})
    assert resolve(None, [other])['status'] == 'task_mismatch'
    assert resolve({'execution_id': other['execution_id']}, [other])['status'] == 'conflicting_final_selection'
    good = session.forecast('historical_mean', request())
    assert resolve(None, [other, good])['execution'] == good['completion']
    assert resolve({'provider': 'last_value'}, [other, good])['status'] == 'conflicting_final_selection'


@pytest.mark.parametrize('changes', [{'point': [99, 99]}, {'series_id': 'other'}, {'unit': 'kg'},
    {'horizon': 3}, {'future_timestamps': ['2020-01-01']}, {'request_fingerprint': 'sha256:' + '0' * 64}, {'revision': 'other'}])
def test_final_echo_cannot_replace_recorded_forecast_or_identity(session, changes):
    execution = session.forecast('last_value', request())
    result = resolve({**execution['completion'], **changes}, [execution])
    assert result['status'] == 'conflicting_final_selection' and result['execution'] is None


def test_canonical_request_fingerprint_independent_of_provider_and_input_form(session):
    expected = request()
    typed = ForecastRequest.from_dict({**expected, 'history': [1.0, 2.0, 3.0]})
    assert forecast_request_fingerprint(expected) == forecast_request_fingerprint(typed)
    a = session.forecast('last_value', expected)
    b = session.forecast('historical_mean', typed)
    assert a['fingerprint'] != b['fingerprint']
    assert a['completion']['request_fingerprint'] == b['completion']['request_fingerprint']
    resolved = resolve(None, [a])
    resolved['execution']['point'][0] = 999
    assert a['completion']['point'][0] == 3


def test_invalid_trusted_evidence_and_contradictory_duplicate_fail_closed(session):
    good = session.forecast('last_value', request())
    incomplete = {'status': 'result_available', 'summary': good['completion']}
    assert resolve(None, [good, incomplete])['status'] == 'conflicting_final_selection'
    corrupted = deepcopy(good)
    corrupted['completion']['point'][0] = 99
    assert resolve(None, [corrupted])['status'] == 'no_successful_execution'
    assert resolve(None, [good['completion'], corrupted['completion']])['status'] == 'conflicting_final_selection'
    with pytest.raises(ForecastAdapterError):
        resolve(None, 'not a collection')
    broken_legacy = session.engine.forecast('last_value', request()).to_dict()
    broken_legacy.pop('completion')
    broken_legacy['result']['quantiles'] = [{'private-token': 1}]
    rejected = resolve(None, [broken_legacy])
    assert rejected['status'] == 'no_successful_execution'
    assert 'private-token' not in json.dumps(rejected)


def test_data_ref_request_binding_and_retained_completion_are_retrievable(session, tmp_path):
    path = tmp_path / 'data.csv'
    path.write_text('timestamp,value\n2026-01-01,1\n2026-01-02,2\n2026-01-03,3\n')
    ref = session.data.inspect(str(path), timezone='UTC', unit='widgets')['data_ref']
    expected = session.data.request(ref, horizon=2000)
    session.results.close()
    session.results = ResultReferences(ResultLimits(max_response_bytes=2048))
    response = session.call('gnomon_forecast', {'provider': 'last_value', 'data_ref': ref, 'horizon': 2000})
    assert response['status'] == 'result_available' and len(json.dumps(response, separators=(',', ':')).encode()) <= 2048
    call = response['forecast_completion']
    text, offset = '', 0
    while True:
        page = session.call(call['tool'], {**call['arguments'], 'offset': offset})
        text += page['text']
        offset = page['next_offset']
        if offset is None:
            break
    completion = json.loads(text)
    resolved = resolve(None, [completion], expected)
    assert resolved['resolved'] and resolved['execution']['point'] == [3.] * 2000
    assert completion['request_fingerprint'] == forecast_request_fingerprint(expected)


def test_synthetic_96_case_preservation_accounting_does_not_relabel_strict_success(session):
    # This reproduces the reported *shape* of the cohort, not the unavailable
    # frozen Arena data, model decisions, forecasts or scores.
    results = []
    for index in range(96):
        expected = {**request(), 'series_id': 'synthetic_' + str(index)}
        a = session.forecast('last_value', expected)
        executions = [a]
        final = {'provider': 'last_value'} if index < 59 else 'Forecast follows in prose.'
        if index >= 94:
            executions.append(session.forecast('historical_mean', expected))
        results.append(resolve(final, executions, expected))
    assert sum(r['final_answer_conformant'] for r in results) == 59
    assert sum(r['resolved'] for r in results) == 94
    assert sum(r['status'] == 'recovered_single_execution' for r in results) == 35
    assert sum(r['status'] == 'ambiguous_multiple_executions' for r in results) == 2


def test_ledger_retrieval_reconstructs_completion_without_changing_deduplication(tmp_path):
    config = tmp_path / 'config.toml'
    config.write_text('cache_size=2\n')
    with GnomonSession.from_config(config, ledger_path=tmp_path / 'ledger.db') as session:
        a = session.forecast('last_value', request())
        b = session.forecast('last_value', request())
        one = session.ledger.execution(a['execution_id'])
        two = session.ledger.execution(b['execution_id'])
        assert one['payload_id'] == two['payload_id']
        assert one['completion'] == a['completion'] and two['completion'] == b['completion']
        assert resolve({'provider': 'last_value'}, [one, two])['cause'] == 'execution_id_required'
        # Full legacy records can be adapted without inventing a request hash.
        one.pop('completion')
        assert resolve(None, [one])['execution'] == a['completion']

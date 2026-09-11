"""Published 1.1.7 feedback: preserve task facts and independent clocks."""
import json

import pytest

from gnomon import GnomonSession, TemporalLedger
from gnomon.contracts import GnomonError
from gnomon.ids import FixedClock
from gnomon.temporal_store import TemporalStore, TemporalObservation
from test_acceptance_recovery import cli
from test_backtesting import at, source


def test_csv_store_assumed_source_time_does_not_erase_recording_time(tmp_path):
    path = source(tmp_path)
    store = TemporalStore(tmp_path / 'store.db')
    store.ingest_csv(str(path), dataset='sales', time_column='timestamp', target_column='value', clock=FixedClock(at(1)))
    with GnomonSession.from_config() as session:
        data = session.data.inspect('store:sales', store_path=str(store.path), recorded_as_of=at(40).isoformat())
        assert data['snapshot']['known_time_assumed']
        assert data['snapshot']['unknown_recorded_times'] == 0
        kwargs = dict(candidates=['historical_mean'], baseline='last_value', horizon=2, replay='recorded')
        plan = session.evaluate(data['data_ref'], **kwargs, preflight=True)
        assert plan['ready'] and plan['execution_diagnostics']['provider_calls'] == 0
        study = session.evaluate(data['data_ref'], **kwargs)
        assert study['status'] == 'complete' and study['usage']['provider_calls'] == 8


@pytest.mark.parametrize('boundary', ['source', 'recorded'])
def test_missing_endpoint_is_not_insufficient_history(tmp_path, boundary):
    store = TemporalStore(tmp_path / 'store.db')
    for i in range(1, 31):
        store.ingest_rows('sales', [TemporalObservation('sales', 'value', at(i),
            at(40) if boundary == 'source' and i == 28 else at(i), i)],
            source_fingerprint=str(i), clock=FixedClock(at(40) if boundary == 'recorded' and i == 28 else at(1)))
    with GnomonSession.from_config() as session:
        ref = session.data.inspect('store:sales', store_path=str(store.path), as_of=at(45).isoformat(), recorded_as_of=at(45).isoformat())['data_ref']
        plan = session.evaluate(ref, candidates=['historical_mean'], baseline='last_value', horizon=2, folds=1, min_history=14, preflight=True)
        fold = plan['planned_folds'][0]
        assert fold['available_history'] == 27 and fold['required_history'] == 14
        assert fold['reason'] == 'history_endpoint_not_visible'
        assert fold['visibility']['required_history_endpoint'] == at(28).isoformat()
        assert fold['visibility']['last_visible_timestamp'] == at(27).isoformat()
        assert fold['visibility']['endpoint_excluded_by_' + boundary + '_cutoff'] == 1


def test_snapshot_recovery_exact_cli_and_shared_inspection(tmp_path):
    path = source(tmp_path)
    frozen = tmp_path / 'saved.gnomon'
    assert cli('inspect', '--input', path, '--unit', 'widgets', '--save-snapshot', frozen)[0] == 0
    argv = ['infer', '--provider', 'seasonal_naive', '--input', str(frozen), '--horizon', '3', '--season', '7']
    code, failure = cli(*argv, '--unit=widgets')
    recovery = failure['error']['recovery']
    assert code == 2 and recovery['cause'] == failure['error']['message']
    assert recovery['rejected_fields'] == ['unit']
    assert recovery['next_call']['argv'] == ['gnomon', *argv]
    assert cli(*recovery['next_call']['argv'][1:])[0] == 0
    assert 'Keep these input options' not in json.dumps(failure)


def test_temporal_mechanical_corrections_leave_semantics_unresolved():
    original = dict(operation='shift', timestamp='2026-01-31T00:00:00Z', amount='7', unit='day')
    code, failure = cli('temporal', '--arguments', json.dumps(original))
    recovery = failure['error']['recovery']
    assert code == 2 and recovery['cause'] and recovery['cause_code'] == 'INVALID_ARGUMENTS'
    assert recovery['supplied_arguments'] == original
    proposal = recovery['next_call']
    assert not proposal['runnable'] and not proposal['admissible']
    assert proposal['arguments'] == dict(operation='shift', value=original['timestamp'], amount=7, unit='days', mode='<calendar|elapsed>')
    corrected = {**proposal['arguments'], 'mode': 'elapsed'}
    code, result = cli('temporal', '--arguments', json.dumps(corrected))
    assert code == 0 and '2026-02-07' in json.dumps(result)


def test_compact_discovery_default_and_expanded_compatibility():
    code, compact = cli('capabilities')
    assert code == 0 and compact['request_schemas']
    assert all('request_schema_ref' in p for p in compact['providers'].values())
    code, expanded = cli('capabilities', '--expanded')
    assert code == 0 and all('request_schema' in p for p in expanded['providers'].values())
    assert GnomonError('TEST', 'specific cause').to_dict()['rejection'] == {'error_ref': '/error', 'terminal': True}
    assert 'missing' in GnomonError('TEST', 'specific cause').to_dict(compact=False)['rejection']


def test_interval_template_does_not_invent_missing_nested_facts():
    arguments = dict(operation='interval', left={}, right=dict(start='2026-01-01T00:00:00Z', end='2026-01-03T00:00:00Z'))
    code, error = cli('temporal', '--arguments', json.dumps(arguments))
    recovery = error['error']['recovery']
    assert code == 2 and not recovery['next_call']['runnable'] and not recovery['admissible']
    assert recovery['choices_required']['missing_facts'] == ['left.start', 'left.end']
    assert recovery['next_call']['arguments']['right'] == arguments['right']


def test_config_invalid_path_and_values_remain_redacted(tmp_path):
    config = tmp_path / 'bad.toml'
    config.write_text('[providers.bad]\nkind="secret-kind"\nbase_url="https://secret.example"\n')
    for extra in ([], ['--show-resolved-config']):
        code, result = cli('capabilities', '--providers-config', config, *extra)
        assert code == 2
        assert result['error']['details']['rejected_fields'] == ['providers.bad.kind']
        assert 'secret' not in json.dumps(result)
        from gnomon.adapters import ADAPTERS
        assert result['error']['details']['invalid_fields'][0]['expected']['enum'] == ['ephemeris', 'callable', 'factory', *ADAPTERS]


def test_all_independent_route_mismatches_and_write_counts(tmp_path):
    with GnomonSession.from_config(ledger=TemporalLedger(tmp_path / 'ledger.db', clock=FixedClock(at(35)))) as session:
        ref = session.data.inspect(str(source(tmp_path)))['data_ref']
        study = session.evaluate(ref, candidates=['historical_mean'], baseline='last_value', horizon=2)
        assert study['execution_diagnostics']['provider_calls'] == 8
        assert study['execution_diagnostics']['ledger_writes'] > 0
        route = session.route(ref, study_id=study['study_id'], horizon=5, candidates=['seasonal_naive'], source_as_of=at(30).isoformat(), recorded_as_of=at(40).isoformat())
        assert {v['field'] for v in route['mismatches']} == {'horizon', 'providers'}
        assert route['execution_diagnostics']['provider_calls'] == route['execution_diagnostics']['ledger_writes'] == 0


@pytest.mark.parametrize('failure', [RuntimeError, TypeError, ValueError, TimeoutError])
def test_provider_failure_correlation_and_invalid_call_zero_dispatch(tmp_path, failure):
    config = tmp_path / 'cache.toml'
    config.write_text('cache_size=2\n')
    with GnomonSession.from_config(config) as session:
        request = {'history': [1, 2], 'horizon': 1}
        first = session.forecast('last_value', request)
        second = session.forecast('last_value', request)
        assert first['execution_diagnostics']['provider_calls'] == 1
        assert second['execution_diagnostics']['provider_calls'] == 0
        assert second['execution_diagnostics']['forecast_calls'] == 1
        with pytest.raises(GnomonError) as err:
            session.call('gnomon_forecast', {'selected_provider': 'last_value', 'request': request})
        assert err.value.to_dict()['execution_diagnostics']['provider_calls'] == 0
        def fail(request):
            raise failure('private token and endpoint')
        session.engine.register('broken', fail, revision='v1')
        with pytest.raises(GnomonError) as err:
            session.call('gnomon_forecast', {'provider': 'broken', 'request': request})
        result = err.value.to_dict()
        assert result['error']['code'] == 'EXECUTION_FAILED'
        assert result['error']['details']['provider'] == 'broken'
        assert result['error']['details']['diagnostic_ref'].startswith('failure_')
        assert result['execution_diagnostics']['provider_calls'] == 1
        assert 'private token' not in json.dumps(result)


def test_csv_store_rescore_and_route_honor_real_recording_cutoffs(tmp_path):
    path = source(tmp_path)
    store = TemporalStore(tmp_path / 'store.db')
    ingest = dict(dataset='sales', time_column='timestamp', target_column='value')
    store.ingest_csv(str(path), **ingest, clock=FixedClock(at(1)))
    ledger = TemporalLedger(tmp_path / 'ledger.db', clock=FixedClock(at(35)))
    with GnomonSession.from_config(ledger=ledger) as session:
        ref = session.data.inspect('store:sales', store_path=str(store.path), as_of=at(30).isoformat(), recorded_as_of=at(35).isoformat())['data_ref']
        study = session.evaluate(ref, candidates=['historical_mean'], baseline='last_value', horizon=2, replay='recorded')
        original = ledger.study(study['study_id'])
        path.write_text(path.read_text().replace(f'{at(30).isoformat()},30', f'{at(30).isoformat()},300'))
        store.ingest_csv(str(path), **ingest, clock=FixedClock(at(40)))
        revised = session.data.inspect('store:sales', store_path=str(store.path), as_of=at(30).isoformat(), recorded_as_of=at(45).isoformat())['data_ref']
        early = session.rescore(revised, study_id=study['study_id'], source_as_of=at(30).isoformat(), recorded_as_of=at(39).isoformat())
        late = session.rescore(revised, study_id=study['study_id'], source_as_of=at(30).isoformat(), recorded_as_of=at(45).isoformat())
        assert early['status'] == late['status'] == 'complete'
        assert early['scores'] == original['scores'] and late['scores'] != original['scores']
        assert early['execution_diagnostics']['provider_calls'] == late['execution_diagnostics']['provider_calls'] == 0
        assert ledger.study(study['study_id']) == original
        earlier = session.data.inspect('store:sales', store_path=str(store.path), as_of=at(30).isoformat(), recorded_as_of=at(39).isoformat())['data_ref']
        route = session.route(earlier, study_id=study['study_id'], source_as_of=at(30).isoformat(), recorded_as_of=at(39).isoformat())
        assert not route['fallback_used']
        assert route['effective_recorded_as_of'] == at(39).isoformat()

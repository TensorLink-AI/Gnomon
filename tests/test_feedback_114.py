"""Task fidelity and evidence metadata from independent 1.1.4 acceptance."""
import inspect
import json
import textwrap

import pytest

from gnomon import ForecastResult, GnomonSession, InferenceEngine
from gnomon.contracts import GnomonError
from gnomon.data import _drop_diagnostic
from gnomon.mcp_server import _handle
from gnomon.selfcheck import leakage_self_check
from test_acceptance_recovery import cli, stamp
from test_backtesting import at, configured, result


@pytest.mark.parametrize('identified', [False, True])
def test_installed_register_example_handles_request_identity(identified):
    doc = inspect.getdoc(InferenceEngine.register)
    code = doc[doc.index('    engine.register'):doc.index('\n\nReturn')]
    with InferenceEngine(cache_size=2) as engine:
        exec(textwrap.dedent(code), {'engine': engine, 'ForecastResult': ForecastResult})
        request = dict(history=[1, 2], horizon=2)
        if identified:
            request.update(series_id='sales', unit='widgets', future_timestamps=[stamp(21), stamp(22)])
        first = engine.forecast('custom', request)
        second = engine.forecast('custom', request)
        assert first.result.point == (2, 2) and not first.cache_hit and second.cache_hit
        assert first.result.unit == request.get('unit')
        assert first.result.timestamps == tuple(request.get('future_timestamps', []))


def test_forecast_alias_and_semantic_error_context(tmp_path):
    path = tmp_path / 'short.csv'
    path.write_text('timestamp,value\n2026-01-01,1\n2026-01-02,2\n')
    flags = ('--provider', 'seasonal_naive', '--input', path, '--unit', 'widgets', '--timezone', 'UTC',
             '--season', '7', '--horizon', '2', '--series-id', '__default__', '--as-of', stamp(2))
    for command in ('forecast', 'infer'):
        code, response = cli(command, *flags)
        assert code == 2
        options = response['error']['details']['input_options']
        assert options['unit'] == 'widgets' and options['as_of'] == stamp(2)
        assert options['series_id'] == '__default__' and options['season'] == 7
    code, response = cli('forecast', '--provider', 'last_value', '--request', '{"history":[1,2],"horizon":2}')
    assert code == 0 and response['result']['point'] == [2, 2]
    assert cli('forecast', '--schema')[1] == cli('infer', '--schema')[1]


@pytest.mark.parametrize('arguments', [dict(provider='last_value', horizon=2),
    dict(provider='last_value', input='data.csv', horizon=2),
    dict(provider='last_value', data_ref='ref', request={}, horizon=2)])
def test_mcp_forecast_error_describes_both_exclusive_forms(arguments):
    with GnomonSession.from_config() as session:
        response = _handle({'method': 'tools/call', 'params': {'name': 'gnomon_forecast', 'arguments': arguments}}, session=session)
        assert response['isError']
        error = response['structuredContent']['error']
        assert 'exactly one form' in error['message']
        assert {tuple(form['required']) for form in error['details']['argument_forms']} == {
            ('provider', 'request'), ('provider', 'data_ref', 'horizon')}


def test_batch_units_are_per_item_and_agree_with_reads(tmp_path):
    with GnomonSession.from_config(ledger_path=tmp_path / 'ledger.db') as session:
        session.allow_outcome_writes = True
        rows = [dict(series_id='sales', valid_time=stamp(21+i), source_available_at=stamp(25), value=i,
                     **({'unit': unit} if unit else {})) for i, unit in enumerate(['widgets', 'kg', None])]
        response = session.call('gnomon_ledger', dict(operation='append_actual', actuals=rows), compact=False)
        query = response['query']
        assert 'unit_defaulted' not in query and 'omitted_unit' not in query
        assert query['unit_scope'] == 'per_actual'
        assert [r['unit_defaulted'] for r in query['actual_units']] == [False, False, True]
        for row in query['actual_units']:
            actuals = session.ledger.actuals_as_of('sales', unit=row['unit'])
            assert len(actuals) == 1 and actuals[0]['unit'] == row['unit']


@pytest.mark.parametrize('task_request,field,choices,expected', [
    (dict(operation='normalize', value='2026-11-01T01:30:00', timezone='America/New_York'), 'fold', [0, 1], '2026-11-01T05:30:00Z'),
    (dict(operation='shift', value='2026-10-31T01:30:00', timezone='America/New_York', amount=1, unit='days', mode='calendar'), 'target_fold', [0, 1], '2026-11-01T05:30:00Z'),
    (dict(operation='shift', value='2026-01-31', amount=1, unit='months', mode='calendar'), 'invalid_date', ['reject', 'clamp'], '2026-02-28'),
])
def test_temporal_choices_preserve_every_supplied_fact(task_request, field, choices, expected):
    request = task_request
    code, response = cli('temporal', '--arguments', json.dumps(request))
    assert code == 2
    details = response['error']['details']
    example = details['example_arguments']
    assert all(example[k] == v for k, v in request.items())
    assert details['changed_fields'] == [field]
    assert details['choices_required'][field] == choices
    code, recovered = cli('temporal', '--arguments', json.dumps(example))
    assert code == 0 and recovered['result'].get('utc', recovered['result'].get('date')) == expected
    if field in {'fold', 'target_fold'}:
        code, second = cli('temporal', '--arguments', json.dumps({**example, field: 1}))
        assert code == 0 and second['result']['utc'] == '2026-11-01T06:30:00Z'


def test_temporal_clock_gap_keeps_invalid_facts_as_nonrunnable_template():
    request = dict(operation='normalize', value='2026-03-08T02:30:00', timezone='America/New_York')
    _, response = cli('temporal', '--arguments', json.dumps(request))
    details = response['error']['details']
    assert details['example_arguments'] == request and details['example_kind'] == 'task_template'
    assert cli('temporal', '--arguments', json.dumps(details['schema_example_arguments']))[0] == 0


def test_route_loads_study_identity_and_preserves_explicit_overrides(tmp_path):
    session, ref = configured(tmp_path, ledger=True)
    with session:
        study = session.evaluate(ref, candidates=['trend'], baseline='last_value', horizon=2, season=7)
        task = dict(data_ref=ref, study_id=study['study_id'], source_as_of=at(30).isoformat(), recorded_as_of=at(35).isoformat())
        answer = session.call('gnomon_route', task, compact=False)
        assert answer['recommendation'] == 'trend' and answer['provider_calls'] == 0
        assert answer['fallback_used'] is False
        assert session.ledger.study(answer['rescore_study_id'])['season'] == 7
        mismatch = session.call('gnomon_route', {**task, 'horizon': 3}, compact=False)
        assert mismatch['reason'] == 'task_identity_mismatch'
        with pytest.raises(GnomonError, match='cutoff'):
            session.call('gnomon_route', {**task, 'recorded_as_of': at(34).isoformat()}, compact=False)
        scopes = answer['cutoff_scopes']
        assert scopes['ledger_evidence_recorded_as_of'] == answer['recorded_as_of']
        assert scopes['snapshot_recorded_as_of'] is None
        assert scopes['snapshot_recording_basis'] == 'unknown_recording_times'


def test_cli_route_id_and_file_match_without_repeating_study_fields(tmp_path):
    path = tmp_path / 'data.csv'
    path.write_text('timestamp,value\n' + ''.join(f'2026-01-{i:02d},{i % 7}\n' for i in range(1, 29)))
    ledger = tmp_path / 'ledger.db'
    report = tmp_path / 'study.json'
    code, study = cli('evaluate', '--input', path, '--timezone', 'UTC', '--unit', 'widgets', '--horizon', 2,
        '--season', 7, '--baseline', 'last_value', '--candidates', 'seasonal_naive', '--ledger-path', ledger, '--save-result', report)
    assert code == 0
    answers = []
    for identity in (study['study_id'], '@' + str(report)):
        code, answer = cli('route', '--input', path, '--timezone', 'UTC', '--unit', 'widgets', '--study', identity,
            '--source-as-of', stamp(28), '--recorded-as-of', '2099-01-01T00:00:00Z', '--ledger-path', ledger)
        assert code == 0 and answer['fallback_used'] is False
        answers.append(answer)
    assert answers[0]['scores'] == answers[1]['scores']
    assert answers[0]['recommendation'] == answers[1]['recommendation']


def test_route_ties_and_insufficient_folds_have_relevant_explanations(tmp_path):
    session, ref = configured(tmp_path, ledger=True)
    with session:
        session.engine.register('same', lambda r: result(r), revision='same-v1', deterministic=True)
        for folds in (2, 4):
            study = session.evaluate(ref, candidates=['same'], baseline='last_value', horizon=2, folds=folds)
            answer = session.route(ref, study_id=study['study_id'], source_as_of=at(30).isoformat(), recorded_as_of=at(35).isoformat())
            if folds == 2:
                assert answer['fallback_used'] and answer['excluded_folds'] == []
                assert answer['next_step'] == 'evaluate_more_matched_folds_with_sufficient_history_and_budget'
            else:
                assert not answer['fallback_used'] and answer['selection_reason'] == 'baseline_tied_for_best'
                assert answer['ranking_policy']['ties'][0]['providers'] == ['last_value', 'same']
                assert session.ledger.study(answer['rescore_study_id'])['ranking_policy'] == answer['ranking_policy']


@pytest.mark.parametrize('mode', ['off', 'safe'])
@pytest.mark.parametrize('bad_count', [1, 6])
def test_early_parse_diagnostic_counts_all_drops_without_repairing(tmp_path, mode, bad_count):
    path = tmp_path / 'bad.csv'
    original = 'timestamp,value\n' + ''.join(
        f'{"bad" if i < bad_count else "2026-01-01"},{i}\n' for i in range(84))
    path.write_text(original)
    code, response = cli('inspect', path, '--timezone', 'UTC', '--unit', 'widgets', '--repair', mode)
    assert code == 2 and response['error']['code'] == 'INVALID_TIMESTAMP'
    budget = response['error']['details']['drop_budget']
    assert budget['scan_complete'] and budget['scanned_rows'] == 84 and budget['dropped_rows'] == bad_count
    assert budget['within_budget'] is (bad_count == 1)
    assert response['error']['details']['input_options']['unit'] == 'widgets'
    assert path.read_text() == original


def test_drop_scan_reports_unknown_above_bound():
    rows = [{'timestamp': 'bad', 'value': '1'}] * 100001
    result = _drop_diagnostic(rows, 'timestamp', 'value')
    assert result['scan_complete'] is False and result['within_budget'] is None and result['scanned_rows'] == 0


def test_selfcheck_reports_finite_reproducible_varied_cases():
    result = leakage_self_check(cases=4, seed=7)
    assert result == leakage_self_check(cases=4, seed=7)
    assert result['checks_passed'] is True and 'structural_claim_proven' not in result
    assert len({row['cutoff'] for row in result['rows']}) > 1
    assert all(all(row['checks'].values()) for row in result['rows'])


def test_documented_local_evidence_workflow_runs_in_fresh_directory(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    document = Path(__file__).resolve().parents[1] / 'docs/local-evidence-workflow.md'
    script = document.read_text().split('```bash\n', 1)[1].split('```', 1)[0]
    env = {**os.environ, 'PATH': str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']}
    proc = subprocess.run(['bash', '-c', script], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=45)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    study, route, score, retry, history = [json.loads((tmp_path / (name + '.json')).read_text())
        for name in ('study', 'route', 'score', 'retry', 'history')]
    assert study['routing_readiness']['ready']
    assert route['recommendation'] == 'seasonal_naive' and not route['fallback_used'] and route['provider_calls'] == 0
    assert score['complete'] and score['result']['mae'] == 0
    assert score['result']['evaluation_id'] == retry['result']['evaluation_id']
    assert retry['result']['evaluation_reused']
    assert history['result'][0]['evaluation_id'] == score['result']['evaluation_id']


def test_rescored_tie_policy_uses_revised_actuals_not_original_ranking(tmp_path):
    from gnomon import TemporalStore
    from gnomon.ids import FixedClock
    from gnomon.temporal_store import TemporalObservation
    session, _ = configured(tmp_path, ledger=True)
    with session:
        store_path = tmp_path / 'store.db'
        store = TemporalStore(store_path)
        for day in range(1, 31):
            store.ingest_rows('sales', [TemporalObservation('a', 'value', at(day), at(day), day)],
                              source_fingerprint=str(day), clock=FixedClock(at(day)))
        def reference(recorded):
            return session.data.inspect('store:sales', store_path=str(store_path), as_of=at(30).isoformat(),
                                        recorded_as_of=at(recorded).isoformat())['data_ref']
        study = session.evaluate(reference(30), candidates=['trend'], baseline='last_value', horizon=2)
        assert study['ranking_policy']['ties'] == []
        for fold in study['folds']:
            for i, actual in enumerate(fold['actuals']):
                from datetime import datetime
                midpoint = (fold['runs']['last_value']['point'][i] + fold['runs']['trend']['point'][i]) / 2
                store.ingest_rows('sales', [TemporalObservation('a', 'value', datetime.fromisoformat(actual['valid_time']),
                                  at(30), midpoint)], source_fingerprint='revision', clock=FixedClock(at(34)))
        answer = session.route(reference(35), study_id=study['study_id'], source_as_of=at(30).isoformat(),
                               recorded_as_of=at(35).isoformat())
        assert answer['selection_reason'] == 'baseline_tied_for_best'
        assert answer['scores']['trend']['mae'] == answer['scores']['last_value']['mae']
        saved = session.ledger.study(answer['rescore_study_id'])
        assert saved['ranking_policy'] == answer['ranking_policy'] and saved['ranking_policy']['ties']
        assert session.ledger.study(study['study_id'])['ranking_policy']['ties'] == []


def test_drop_diagnostic_counts_bad_targets_and_timestamps_once_per_row(tmp_path):
    path = tmp_path / 'bad-both.csv'
    path.write_text('timestamp,value\nbad,bad\n2026-01-02,2\n' + ''.join(f'2026-01-03,{i}\n' for i in range(40)))
    code, response = cli('inspect', path)
    assert code == 2 and response['error']['code'] == 'INVALID_TARGET'
    budget = response['error']['details']['drop_budget']
    assert budget['dropped_rows'] == 1 and budget['total_rows'] == 42 and budget['within_budget']


def test_mcp_inspection_failure_retains_unit_and_cutoffs(tmp_path):
    arguments = dict(input=str(tmp_path / 'missing.csv'), unit='widgets', timezone='UTC', as_of=stamp(10))
    with GnomonSession.from_config() as session:
        reply = _handle({'method': 'tools/call', 'params': {'name': 'gnomon_inspect', 'arguments': arguments}}, session=session)
        assert reply['isError']
        assert reply['structuredContent']['error']['details']['input_options'] == arguments

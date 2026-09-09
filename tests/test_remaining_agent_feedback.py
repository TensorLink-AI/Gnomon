"""Published-interface journeys for the remaining agent UX feedback."""
from datetime import datetime
import json
import sqlite3
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gnomon import ForecastRequest, ForecastResult, GnomonSession, InferenceEngine
from gnomon.contracts import GnomonError
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.ids import FixedClock
from gnomon.temporal_store import TemporalObservation, TemporalStore
from gnomon.selfcheck import FAMILIES, leakage_self_check
from test_acceptance_recovery import cli
from test_backtesting import at, configured


def test_cache_canonicalizes_values_defaults_and_instants_without_merging_large_integers():
    with InferenceEngine(cache_size=8) as engine:
        calls = []
        def provider(r):
            calls.append(r)
            return ForecastResult((r.history[-1],) * r.horizon, timestamps=r.future_timestamps, unit=r.unit, series_id=r.series_id)
        engine.register('custom', provider, revision='v1', deterministic=True)
        a = dict(history=[1, 2, 3], horizon=1, future_timestamps=['2026-01-02T10:00:00+10:00'], unit='widgets')
        b = ForecastRequest(history=(1., 2., 3.), horizon=1, future_timestamps=('2026-01-02T00:00:00Z',), unit='widgets')
        first, second = engine.forecast('custom', a), engine.forecast('custom', b)
        assert first.fingerprint == second.fingerprint and second.cache_hit and len(calls) == 1
        assert engine.cache_diagnostic('custom', b)['status'] == 'hit' and len(calls) == 1
        assert second.to_dict()['cache']['provider_eligible'] and second.to_dict()['request_provenance']
        assert engine.forecast('custom', b, use_cache=False).cache['status'] == 'bypassed'
        large = engine.forecast('custom', {'history': [2**53], 'horizon': 1})
        different = engine.forecast('custom', {'history': [2**53 + 1], 'horizon': 1})
        assert large.fingerprint != different.fingerprint and large.result.point != different.result.point
        for bad in ([True], ['1'], [float('nan')]):
            with pytest.raises(ForecastAdapterError):
                engine.forecast('custom', dict(history=bad, horizon=1))


def revision_study(tmp_path):
    session, _ = configured(tmp_path, ledger=True)
    store = TemporalStore(tmp_path / 'vintages.db')
    for day in range(1, 31):
        store.ingest_rows('sales', [TemporalObservation('sales', 'value', at(day), at(day), day)],
                          source_fingerprint=f'original-{day}', clock=FixedClock(at(day)))
    ref = session.data.inspect('store:sales', store_path=str(store.path), unit='widgets',
                               as_of=at(30).isoformat(), recorded_as_of=at(30).isoformat())['data_ref']
    study = session.evaluate(ref, candidates=['trend'], baseline='last_value', horizon=2, verify=True)
    for fold in study['folds']:
        origin_value = fold['request']['history'][-1]
        for actual in fold['actuals']:
            store.ingest_rows('sales', [TemporalObservation('sales', 'value', datetime.fromisoformat(actual['valid_time']), at(40), origin_value)],
                              source_fingerprint='revised', clock=FixedClock(at(42)))
    revised = session.data.inspect('store:sales', store_path=str(store.path), unit='widgets',
                                   as_of=at(50).isoformat(), recorded_as_of=at(50).isoformat())['data_ref']
    return session, store, study, revised


def test_rescore_preserves_task_and_predictions_while_advancing_only_evidence_cutoffs(tmp_path):
    session, store, study, revised = revision_study(tmp_path)
    with session:
        old = session.ledger.study(study['study_id'])
        def forbidden(*args, **kwargs):
            raise AssertionError('Rescore must not execute a provider')
        session.engine.forecast = forbidden
        args = dict(study_id=study['study_id'], source_as_of=at(45).isoformat(), recorded_as_of=at(45).isoformat())
        early = session.rescore(revised, **{**args, 'recorded_as_of': at(41).isoformat()})
        assert early['scores'] == old['scores']
        later = session.rescore(revised, **args)
        assert later['ranking'][0] == 'last_value' and old['ranking'][0] == 'trend'
        assert later['scores']['last_value']['mae'] == 0
        assert later['scores']['trend']['mae'] == 1.5
        assert later['score_derivations']['last_value']['absolute_error_sum'] == 0
        assert later['score_derivations']['last_value']['pairs_sha256'] != old['score_derivations']['last_value']['pairs_sha256']
        assert later['provider_calls'] == 0 and later['predictions_reused_exactly']
        assert session.ledger.study(study['study_id']) == old
        for before, after in zip(old['folds'], later['folds']):
            assert (before['request'], before['runs'], before['origin']) == (after['request'], after['runs'], after['origin'])
        comparison = session.compare_studies(original_study_id=old['study_id'], rescored_study_id=later['study_id'])
        assert comparison['original_unchanged'] and comparison['predictions_reused_exactly']
        assert len(comparison['changed_folds']) == 4 and comparison['changed_actual_ids']
        with sqlite3.connect(session.ledger.path) as conn:
            assert conn.execute('SELECT count(*) FROM executions').fetchone()[0] == 8
        code, from_cli = cli('evaluate', '--rescore', old['study_id'], '--input', 'store:sales', '--store-path', store.path,
            '--unit', 'widgets', '--source-as-of', at(45).isoformat(), '--recorded-as-of', at(45).isoformat(), '--ledger-path', session.ledger.path)
        assert code == 0 and from_cli['ranking'] == later['ranking']
        code, compared = cli('evaluate', '--compare', old['study_id'], from_cli['study_id'], '--ledger-path', session.ledger.path)
        assert code == 0 and compared['original_unchanged']


def test_rescore_rejects_missing_evidence_and_wrong_unit_without_writing(tmp_path):
    session, _, study, revised = revision_study(tmp_path)
    with session:
        with sqlite3.connect(session.ledger.path) as conn:
            before = list(conn.iterdump())
        with pytest.raises(GnomonError, match='Not all original folds'):
            session.rescore(revised, study_id=study['study_id'], source_as_of=at(10).isoformat(), recorded_as_of=at(45).isoformat(), allow_partial=False)
        with sqlite3.connect(session.ledger.path) as conn:
            assert list(conn.iterdump()) == before
        with pytest.raises(ForecastAdapterError, match='not recorded'):
            session.rescore(revised, study_id=study['study_id'], source_as_of=at(45).isoformat(), recorded_as_of=at(32).isoformat())
        wrong = tmp_path / 'different-variable.csv'
        wrong.write_text('timestamp,other,series\n' + ''.join(f'{at(i).isoformat()},{i},sales\n' for i in range(1, 31)))
        ref = session.data.inspect(str(wrong), target_column='other', series_column='series', unit='widgets')['data_ref']
        with pytest.raises(GnomonError, match='original series, variable'):
            session.rescore(ref, study_id=study['study_id'], source_as_of=at(45).isoformat(), recorded_as_of=at(45).isoformat())


def test_store_timezone_and_empty_recording_filter_are_discoverable(tmp_path):
    path = tmp_path / 'data.csv'
    path.write_text('timestamp,value\n2026-01-01,1\n2026-01-02,2\n')
    store = TemporalStore(tmp_path / 'store.db')
    store.ingest_csv(str(path), dataset='aware', time_column='timestamp', target_column='value', timezone='Australia/Brisbane', clock=FixedClock(at(400)))
    with GnomonSession.from_config() as session:
        data = session.data.inspect('store:aware', store_path=str(store.path), purpose='route', frequency='D')
        assert data['readiness']['route']['ready']
        with pytest.raises(GnomonError) as exc:
            session.data.inspect('store:aware', store_path=str(store.path), recorded_as_of=at(1).isoformat())
        assert exc.value.details['exclusion_counts']['excluded_by_recorded_cutoff'] == 2
        assert exc.value.details['exclusion_counts']['excluded_by_source_cutoff'] == 0


def test_resolved_configuration_does_not_import_or_expose_remote_credentials(tmp_path):
    config = tmp_path / 'providers.toml'
    config.write_text('schema_version=1\nledger_path="work/evidence.db"\ncache_size=8\n[providers.remote]\nkind="ephemeris"\nbase_url="https://secret-user:secret-password@example.invalid"\n[providers.missing]\nkind="callable"\nentrypoint="not_installed:forecast"\n')
    code, result = cli('capabilities', '--providers-config', config, '--show-resolved-config')
    assert code == 0 and result['ledger_path'] == str(tmp_path / 'work/evidence.db')
    assert not result['ledger_exists'] and not result['providers']['missing']['entrypoint_imported']
    assert 'secret' not in json.dumps(result['providers'])
    assert not (tmp_path / 'work').exists()


def test_explicit_snapshot_flags_and_compact_errors(tmp_path):
    path = tmp_path / 'data.csv'
    path.write_text('timestamp,value\n2026-01-01,1\n2026-01-02,2\n')
    frozen = tmp_path / 'data.gnomon'
    assert cli('inspect', '--input', path, '--frequency', 'D', '--save-snapshot', frozen)[0] == 0
    code, response = cli('--compact-errors', 'infer', '--input', frozen, '--provider', 'last_value', '--horizon', 2, '--unit', 'widgets')
    assert code == 2 and response['rejection']['error_ref'] == '/error'
    details = response['error']['details']
    assert details['rejected_fields'] == ['unit']
    assert details['supplied_arguments']['unit'] == 'widgets'
    assert 'time_column' not in details['supplied_arguments']


def test_mixed_bad_cells_have_separate_counts_and_drop_scope():
    from gnomon.data import _drop_diagnostic
    rows = [{'t': 'bad', 'v': 'bad'}] + [{'t': '2026-01-01', 'v': 1}] * 39
    result = _drop_diagnostic(rows, 't', 'v')
    assert result['affected_rows'] == result['dropped_rows'] == 1
    assert result['invalid_timestamp_fields'] == result['invalid_target_fields'] == 1
    assert result['within_budget'] and result['predicted_rows_after_drops'] == 39


def test_verification_discovery_and_selfcheck_families():
    code, verified = cli('forecast', '--provider', 'seasonal_naive', '--request', '{"history":[1,2,3],"horizon":2,"season":3}', '--verify')
    assert code == 0 and verified['verification']['expected'] == [1, 2]
    assert verified['verification']['verified'] and not verified['season_defaulted']
    assert cli('providers', 'example')[1]['python']
    assert cli('capabilities', '--cache', '--provider', 'last_value', '--request', '{"history":[1,2],"horizon":1}')[1]['cache']['provider_calls'] == 0
    a = leakage_self_check(2, 7, FAMILIES)
    b = leakage_self_check(2, 7, FAMILIES)
    assert a == b and a['checks_passed'] and a['passed'] == 2
    assert all(set(FAMILIES) - {'snapshot'} <= row['checks'].keys() for row in a['rows'])


def test_diagnostic_dry_run_reports_all_modes_without_changing_source(tmp_path):
    path = tmp_path / 'mixed.csv'
    data = 'timestamp,value\n' + ''.join(f'{at(i).isoformat()},{i}\n' for i in range(1, 31) if i != 10)
    path.write_text(data)
    code, result = cli('inspect', '--diagnose', '--input', path, '--frequency', 'D')
    assert code == 0 and result['source_modified'] is False
    assert result['modes']['off']['admissible'] is False
    assert result['modes']['safe']['admissible'] is False
    assert result['modes']['aggressive']['admissible'] is True
    assert result['modes']['aggressive']['repairs'][0]['count'] == 1
    assert path.read_text() == data
    assert cli('inspect', '--diagnose', '--input', path, '--repair', 'off')[0] == 2


def test_documented_revision_and_installed_mcp_workflows(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'PYTHONPATH': str(root / 'src')}
    code = (root / 'docs/revised-vintage-workflow.md').read_text().split('```python\n')[1].split('```')[0]
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=40)
    assert result.returncode == 0, result.stderr
    client = subprocess.run([sys.executable, '-m', 'gnomon.examples.mcp_workflow'], cwd=tmp_path,
                             env=env, text=True, capture_output=True, timeout=45)
    assert client.returncode == 0, client.stderr
    transcript = json.loads(client.stdout)
    assert transcript['verified_points'] == 2000
    assert any(f.get('sent', {}).get('params', {}).get('name') == 'gnomon_read' for f in transcript['transcript'])


def test_mcp_rescore_and_compare_use_advertised_schemas(tmp_path):
    from gnomon.mcp_server import _handle
    from test_agent_recovery_workflows import mcp, unpack
    session, _, study, revised = revision_study(tmp_path)
    with session:
        _handle({'method': 'initialize', 'params': {'protocolVersion': '2025-06-18'}}, session=session)
        schemas = next(t for t in session.tools() if t['name'] == 'gnomon_evaluate')['inputSchema']['oneOf']
        assert {s['properties'].get('operation', {}).get('const') for s in schemas} >= {'rescore', 'compare_studies'}
        score = unpack(session, mcp(session, 'gnomon_evaluate', dict(operation='rescore', study_id=study['study_id'],
            data_ref=revised, source_as_of=at(45).isoformat(), recorded_as_of=at(45).isoformat())))
        compared = unpack(session, mcp(session, 'gnomon_evaluate', dict(operation='compare_studies',
            original_study_id=study['study_id'], rescored_study_id=score['study_id'])))
        assert compared['original_unchanged'] and compared['predictions_reused_exactly']


def test_rescore_recomputes_exact_ties_and_proves_original_is_unchanged(tmp_path):
    session, store, study, _ = revision_study(tmp_path)
    with session:
        for fold in study['folds']:
            for i, actual in enumerate(fold['actuals']):
                midpoint = (fold['runs']['last_value']['point'][i] + fold['runs']['trend']['point'][i]) / 2
                store.ingest_rows('sales', [TemporalObservation('sales', 'value', datetime.fromisoformat(actual['valid_time']), at(43), midpoint)],
                                  source_fingerprint='midpoint', clock=FixedClock(at(44)))
        ref = session.data.inspect('store:sales', store_path=str(store.path), unit='widgets', as_of=at(45).isoformat(), recorded_as_of=at(45).isoformat())['data_ref']
        score = session.rescore(ref, study_id=study['study_id'], source_as_of=at(45).isoformat(), recorded_as_of=at(45).isoformat())
        assert score['scores']['last_value']['mae'] == score['scores']['trend']['mae']
        assert score['ranking'] == ['last_value', 'trend']
        assert score['ranking_policy']['ties'][0]['providers'] == ['last_value', 'trend']
        assert session.compare_studies(original_study_id=study['study_id'], rescored_study_id=score['study_id'])['original_unchanged']


def test_rescore_keeps_recorded_provider_order_across_json_persistence(tmp_path):
    from test_backtesting import result
    session, ref = configured(tmp_path, ledger=True)
    with session:
        for name in ('z_first', 'a_second'):
            session.engine.register(name, lambda r: result(r, 1), revision='v1', deterministic=True)
        study = session.evaluate(ref, candidates=['z_first', 'a_second'], baseline='last_value', horizon=2)
        score = session.rescore(ref, study_id=study['study_id'], source_as_of=at(40).isoformat(), recorded_as_of=at(40).isoformat())
        assert score['ranking'] == ['z_first', 'a_second', 'last_value']
        assert score['ranking_policy']['ties'][0]['providers'] == ['z_first', 'a_second']

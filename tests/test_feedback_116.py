"""Regression journeys for observed 1.1.6 agent-interface failures."""
from contextlib import redirect_stderr
from datetime import timedelta
from io import StringIO
import json

import pytest

from gnomon import GnomonSession, TemporalLedger
from gnomon.contracts import GnomonError
from gnomon.ids import FixedClock
from gnomon.temporal_store import TemporalObservation, TemporalStore
from gnomon.selfcheck import FAMILIES, family_contracts, leakage_self_check
from gnomon.temporal_ops import TEMPORAL_SCHEMA
from test_acceptance_recovery import cli
from test_backtesting import at, source


def test_discovery_never_opens_ledger_even_on_provider_failure(tmp_path):
    config = tmp_path / 'operator.toml'
    config.write_text('ledger_path="new/evidence.db"\n')
    code, result = cli('capabilities', '--providers-config', config)
    assert code == 0 and result['ledger']['configured'] and not result['ledger']['opened']
    assert not (tmp_path / 'new').exists()
    config.write_text(config.read_text() + '[providers.bad]\nkind="callable"\nentrypoint="missing:forecast"\nextra="invalid"\n')
    code, result = cli('capabilities', '--providers-config', config)
    assert code == 2 and result['error']['details']['provider'] == 'bad'
    assert 'extra' in result['error']['details']['rejected_fields']
    assert not (tmp_path / 'new').exists()


def test_replay_preflight_and_failures_explain_recording_visibility(tmp_path):
    store = TemporalStore(tmp_path / 'store.db')
    store.ingest_rows('sales', [TemporalObservation('sales', 'value', at(i), at(i), i) for i in range(1, 31)],
        source_fingerprint='original', clock=FixedClock(at(40)))
    with GnomonSession.from_config() as session:
        inspected = session.data.inspect('store:sales', store_path=str(store.path), recorded_as_of=at(45).isoformat())
        assert inspected['evaluation_replay']['default_mode'] == 'recorded'
        kwargs = dict(candidates=['historical_mean'], baseline='last_value', horizon=2)
        plan = session.evaluate(inspected['data_ref'], **kwargs, preflight=True)
        assert not plan['ready'] and plan['provider_calls'] == 0 and not plan['recorded']
        assert all(f['reason'] == 'no_history_recorded_by_origin' for f in plan['planned_folds'])
        assert all(f['visibility']['excluded_by_recorded_cutoff'] == f['visibility']['input_vintages'] for f in plan['planned_folds'])
        bad = session.evaluate(inspected['data_ref'], **kwargs)
        assert bad['status'] == 'unscored' and bad['excluded_folds']
        assert bad['routing_readiness']['issues'][0]['action'] == 'resolve_historical_visibility'
        good = session.evaluate(inspected['data_ref'], **kwargs, replay='source_available')
        assert good['status'] == 'complete' and not good['replay_defaulted']
    code, plan = cli('evaluate', '--input', 'store:sales', '--store-path', store.path,
        '--recorded-as-of', at(45).isoformat(), '--candidates', 'historical_mean', '--baseline', 'last_value', '--horizon', 2, '--preflight')
    assert code == 0 and not plan['ready']


def test_compare_history_names_missing_history_not_recording_violation(tmp_path):
    ledger = TemporalLedger(tmp_path / 'ledger.db', clock=FixedClock(at(10)))
    with GnomonSession.from_config(ledger=ledger) as session:
        for p in ('last_value', 'historical_mean'):
            session.forecast(p, dict(history=[1, 2], horizon=1, cutoff=at(10).isoformat(),
                future_timestamps=[at(11).isoformat()], series_id='sales', unit='widgets'))
        query = dict(series_id='sales', horizon=1, providers={p: v['revision'] for p, v in session.engine.capabilities().items() if p != 'seasonal_naive'},
            start=at(10).isoformat(), end=at(10).isoformat(), source_as_of=at(12).isoformat(), recorded_as_of=at(12).isoformat(), unit='widgets')
        answer = ledger.compare_history(**query)
        assert answer['excluded'][0]['reason'] == 'missing_history_timestamps'
        assert answer['excluded'][0]['causes'][0]['required_fields'][0] == 'timestamps'
        with pytest.raises(GnomonError) as error:
            session.call('gnomon_ledger', dict(operation='compare_history', **{**query, 'series_id': '__default__'}))
        detail = error.value.details
        assert 'series_id' not in detail['example_arguments'] and detail['choices_required']['series_id']


def test_csv_timezone_binding_is_transactional(tmp_path):
    path = source(tmp_path)
    store = TemporalStore(tmp_path / 'store.db')
    store.ingest_csv(str(path), dataset='a', time_column='timestamp', target_column='value', timezone='UTC')
    before = store.dataset_fingerprint('a')
    same = store.ingest_csv(str(path), dataset='a', time_column='timestamp', target_column='value', timezone='UTC')
    assert same.rows_added == 0
    with pytest.raises(GnomonError, match='timezone declaration'):
        store.ingest_csv(str(path), dataset='a', time_column='timestamp', target_column='value', timezone='Australia/Brisbane')
    assert store.dataset_fingerprint('a') == before


def test_strict_route_rejection_is_saved_and_default_warns(tmp_path):
    path = source(tmp_path)
    db, saved = tmp_path / 'ledger.db', tmp_path / 'rejected.json'
    with GnomonSession.from_config(ledger_path=db) as session:
        ref = session.data.inspect(str(path))['data_ref']
        study = session.evaluate(ref, candidates=['historical_mean'], baseline='last_value', horizon=2, folds=2)
    args = ('route', '--input', path, '--study', study['study_id'], '--ledger-path', db,
        '--source-as-of', at(30).isoformat(), '--recorded-as-of', '2099-01-01T00:00:00Z', '--horizon', 5)
    code, rejected = cli(*args, '--require-evidence', '--save-result', saved)
    assert code == 2 and json.loads(saved.read_text())['error']['code'] == 'ROUTING_EVIDENCE_REQUIRED'
    assert rejected['error']['details']['mismatch'] == {'field': 'horizon', 'study': 2, 'requested': 5}
    err = StringIO()
    with redirect_stderr(err):
        code, fallback = cli(*args)
    assert code == 0 and fallback['fallback_used'] and 'fallback' in err.getvalue()


def test_cache_counters_and_study_scoring_discovery(tmp_path):
    config = tmp_path / 'cache.toml'
    config.write_text('cache_size=1\n')
    ledger = TemporalLedger(tmp_path / 'ledger.db', clock=FixedClock(at(35)))
    with GnomonSession.from_config(config, ledger=ledger) as session:
        a = dict(history=[1, 2], horizon=1)
        session.forecast('last_value', a)
        session.forecast('last_value', a)
        session.engine.cache_diagnostic('last_value', a)
        session.forecast('last_value', dict(history=[3], horizon=1))
        stats = session.capabilities()['cache']
        assert {k: stats[k] for k in ('entries', 'hits', 'misses', 'evictions')} == dict(entries=1, hits=1, misses=2, evictions=1)
        ref = session.data.inspect(str(source(tmp_path)))['data_ref']
        study = session.evaluate(ref, candidates=['historical_mean'], baseline='last_value', horizon=2)
        rows = ledger.search(status='scored_in_study', source_as_of=at(40).isoformat(), recorded_as_of=at(40).isoformat())
        assert rows['items'] and all(study['study_id'] in r['scored_study_ids'] for r in rows['items'])
        assert not any(study['study_id'] in r.get('study_ids', []) for r in ledger.pending(source_as_of=at(40).isoformat(), recorded_as_of=at(40).isoformat()))


def test_repairs_offer_alignment_and_reject_impossible_drops(tmp_path):
    path = tmp_path / 'jitter.csv'
    path.write_text('timestamp,value\n' + ''.join(f'{(at(i) + timedelta(minutes=1 if i == 12 else 0)).isoformat()},{i}\n' for i in range(1, 31)))
    code, rejected = cli('inspect', '--input', path, '--frequency', 'D', '--repair', 'off')
    assert code == 2 and rejected['error']['repair_options'][0]['action'] == 'align_timestamps_safe'
    assert rejected['error']['details']['timestamp_alignment']['admissible']
    assert rejected['error']['details']['repair_budgets']['gap_fill_budget']['admissible'] is None
    assert cli('inspect', '--input', path, '--frequency', 'D', '--repair', 'safe')[0] == 0
    path.write_text('timestamp,value\nnot-a-time,1\n2026-01-02,2\n')
    code, rejected = cli('inspect', '--input', path, '--repair', 'safe')
    assert not rejected['error']['details']['drop_budget']['within_budget']
    assert all(r['action'] != 'review_aggressive_drop' for r in rejected['error']['repair_options'])


def test_column_correction_and_provider_boundary_preserve_task(tmp_path):
    path = source(tmp_path)
    path.write_text(path.read_text().replace('timestamp,value', 'ts,value'))
    argv = ['infer', '--provider', 'seasonal_naive', '--input', str(path), '--horizon', '2', '--season', '7', '--unit', 'widgets']
    code, error = cli(*argv)
    correction = error['error']['details']['next_call']['argv'][1:]
    assert code == 2 and correction == [*argv, '--time-column', 'ts']
    assert cli(*correction)[0] == 0
    with GnomonSession.from_config() as session:
        request = {'selected_provider': 'last_value', 'request': {'history': [1, 2], 'horizon': 1, 'unit': 'widgets'}}
        with pytest.raises(GnomonError) as exc:
            session.call('gnomon_forecast', request)
        assert exc.value.details['cause_code'] == 'INVALID_TOOL_ARGUMENTS'
        next_call = exc.value.details['next_call']
        assert session.call(next_call['tool'], next_call['arguments'], compact=False)['result']['point'] == (2,)
        with pytest.raises(GnomonError) as bad:
            session.call('gnomon_forecast', {**request, 'request': {'history': [True], 'horizon': 1}})
        assert bad.value.details['example_runnable'] is False


def test_family_contracts_and_detailed_checks_are_auditable():
    assert set(family_contracts()['families']) == set(FAMILIES)
    a, b = [leakage_self_check(2, 7, FAMILIES, detailed=True) for _ in range(2)]
    assert a['checks_passed'] and a['rows'] == b['rows']
    assert all(set(row['assertions']) == set(FAMILIES) for row in a['rows'])
    assert a['rows'][0]['assertions']['dst']['actual'] == a['rows'][0]['assertions']['dst']['expected']
    assert cli('self-check', 'families')[0] == 0


def test_discovery_projection_and_provider_source_output(tmp_path):
    code, brief = cli('capabilities', '--brief')
    assert code == 0 and len(brief['request_schemas']) == 1
    assert all('request_schema_ref' in p for p in brief['providers'].values())
    path = tmp_path / 'example.py'
    assert cli('providers', 'example', '--write-to', path)[0] == 0
    assert 'ForecastResult' in path.read_text()
    with GnomonSession.from_config() as session:
        assert 'season_guidance' not in session.forecast('last_value', dict(history=[1], horizon=1))
        with pytest.raises(GnomonError) as exc:
            session.call('gnomon_ledger', {'operation': 'search', 'limit': 200})
        # No configured ledger cannot offer a valid stored-data correction.
        assert exc.value.code == 'LEDGER_NOT_CONFIGURED'


def test_combined_repair_costs_and_order_events_contract(tmp_path):
    path = tmp_path / 'mixed.csv'
    missing = {5, 6, 15, 16, 25, 26, 35, 36, 45, 46}
    lines = [f'{at(i).isoformat()},{i}\n' for i in range(1, 51) if i not in missing]
    lines += [f'{at(i).isoformat()},999\n' for i in (1, 11, 21, 31, 41)]
    path.write_text('timestamp,value\n' + ''.join(lines))
    code, error = cli('inspect', '--input', path, '--frequency', 'D', '--repair', 'aggressive')
    budget = error['error']['details']['repair_budgets']['combined_fill_conflict_budget']
    assert code == 2 and (budget['proposed_fills'], budget['proposed_conflict_resolutions']) == (10, 5)
    assert budget['combined_cost'] == 15 and budget['original_observations'] == 45
    assert budget['fraction'] == pytest.approx(1 / 3) and budget['admissible'] is False
    variant = next(v for v in TEMPORAL_SCHEMA['oneOf'] if v['properties']['operation']['const'] == 'order_events')
    assert set(variant['properties']['events']['items']['required']) == {'event_id', 'at'}


def test_production_comparison_public_example(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.chdir(tmp_path)
    doc = Path(__file__).resolve().parents[1] / 'docs/production-history-comparison.md'
    script = doc.read_text().split('```python\n')[1].split('```')[0]
    exec(compile(script, str(doc), 'exec'), {})


def test_nearest_limit_correction_and_completion_scopes(tmp_path):
    with GnomonSession.from_config(ledger_path=tmp_path / 'ledger.db') as session:
        with pytest.raises(GnomonError) as exc:
            session.call('gnomon_ledger', {'operation': 'search', 'limit': 200})
        assert exc.value.details['example_arguments']['limit'] == 100
        assert session.call('gnomon_capabilities', {'brief': True})['request_schemas']
        ref = session.data.inspect(str(source(tmp_path)))['data_ref']
        study = session.call('gnomon_evaluate', {'data_ref': ref, 'baseline': 'last_value', 'candidates': ['historical_mean'], 'horizon': 2})
        if study['status'] == 'result_available':
            study = session.results.value(study['result_ref'])
        assert study['returned_evidence'] == 'fold_summary'
        assert study['scoring_complete'] and study['evidence_status'] == 'complete'

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gnomon import EvidenceMemory, ForecastRequest, GnomonSession, HermesMemoryAdapter, TemporalLedger
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.ids import FixedClock


@pytest.fixture
def case(tmp_path):
    origin = datetime(2026, 1, 20, tzinfo=timezone.utc)
    ledger = TemporalLedger(tmp_path/'ledger.db', clock=FixedClock(origin))
    session = GnomonSession.from_config(ledger=ledger)
    request = dict(history=[10, 11, 12], horizon=2, series_id='sales', unit='widgets',
        timestamps=[(origin-timedelta(days=i)).isoformat() for i in (2, 1, 0)],
        future_timestamps=[(origin+timedelta(days=i)).isoformat() for i in (1, 2)],
        cutoff=origin.isoformat(), frequency='D')
    forecast = session.forecast('last_value', request)
    bridge = EvidenceMemory(ledger, ledger_ref='retail-evidence')
    yield bridge, ledger, session, request, forecast, origin
    session.close()


def decision(case, **kwargs):
    bridge, _, _, request, forecast, _ = case
    return bridge.record_decision(execution_id=forecast['execution_id'], expected_request=request,
        rationale='I intended to investigate a seasonal model.', **kwargs)


def mature(case, values=(13, 14)):
    _, ledger, _, request, _, origin = case
    ledger.clock = FixedClock(origin+timedelta(days=3))
    for timestamp, value in zip(request['future_timestamps'], values):
        ledger.append_actual(series_id='sales', unit='widgets', valid_time=timestamp, value=value,
            source_available_at=ledger._now())
    return dict(source_as_of=ledger._now(), recorded_as_of=ledger._now())


def test_binding_rejects_other_task_and_unexecuted_provider_before_writing(case):
    bridge, ledger, session, request, forecast, _ = case
    calls = session.engine._execution_stats['provider_calls']
    advertised = session.capabilities()['ledger']['decision_memory']['evidence_bridge']
    assert advertised['python'] == 'gnomon.EvidenceMemory'
    assert not advertised['adapter_executes_external_writes']
    for overrides in [{'execution_id': 'not-executed'}, {'expected_request': {**request, 'unit': 'kg'}},
                      {'expected_request': {**request, 'history': [1, 2, 3]}}, {'claimed_provider': 'seasonal_naive'}]:
        args = dict(execution_id=forecast['execution_id'], expected_request=request, rationale='Claimed decision.')
        with pytest.raises(ForecastAdapterError):
            bridge.record_decision(**(args | overrides))
    with ledger._connect() as conn:
        assert conn.execute('SELECT count(*) FROM decisions').fetchone()[0] == 0
    packet = bridge.record_decision(execution_id=forecast['execution_id'],
        expected_request=ForecastRequest(**{**request, 'history': (10., 11., 12.)}),
        rationale='I wanted to use seasonality, but only the baseline was submitted.')
    assert packet['verified_execution']['provider'] == 'last_value'
    assert packet['verified_execution']['target_start'] == request['future_timestamps'][0]
    assert packet['narrative_status'] == 'unverified_hypothesis'
    assert packet['scoring']['status'] == 'pending'
    assert packet['scoring']['metrics']['rmsle_status'] == 'no_scored_pairs'
    assert session.engine._execution_stats['provider_calls'] == calls


def test_claims_keep_original_text_and_distinguish_numeric_evidence(case):
    bridge, _, session, _, _, _ = case
    packet = decision(case); cuts = mature(case)
    claims = [{'field': 'provider', 'value': 'seasonal_naive'}, {'field': 'mae', 'value': 1.5},
        {'field': 'n', 'value': True}, {'field': 'explanation', 'value': 'Promotion caused the error'},
        {'field': 'rank', 'value': 1}]
    original = deepcopy(claims)
    checked = bridge.check_claims(decision_id=packet['decision_id'], claims=claims, **cuts)
    assert [r['status'] for r in checked['checks']] == ['contradicted', 'supported', 'contradicted', 'unverified', 'unverified']
    assert checked['checks'][0]['observed'] == 'last_value'
    assert checked['checks'][3]['claim'] == claims[3] and claims == original
    assert not checked['business_explanation_validated']
    assert session.engine._execution_stats['provider_calls'] == 1
    with pytest.raises(ForecastAdapterError):
        bridge.check_claims(decision_id=packet['decision_id'], claims=[{'field':'mae','value':float('nan')}], **cuts)
    huge = bridge.check_claims(decision_id=packet['decision_id'], claims=[{'field':'mae','value':10**400}], **cuts)
    assert huge['checks'][0]['status'] == 'contradicted'


def test_pending_partial_complete_and_immutable_revision_lessons(case):
    bridge, ledger, session, request, forecast, origin = case
    packet = decision(case); did = packet['decision_id']
    original = ledger.decision(did)['inputs']
    cuts = mature(case, values=(13,))
    reviewed = bridge.decision(decision_id=did, **cuts)
    assert reviewed['scoring']['status'] == 'partial' and reviewed['scoring']['metrics']['n'] == 1
    with pytest.raises(ForecastAdapterError, match='complete'):
        bridge.record_lesson(decision_id=did, lesson='Premature lesson.', **cuts)
    ledger.append_actual(series_id='sales',unit='kg',valid_time=request['future_timestamps'][1],value=14,
        source_available_at=ledger._now())
    assert bridge.decision(decision_id=did,**cuts)['scoring']['coverage']['other_units_at_missing_steps'] == ['kg']
    ledger.append_actual(series_id='sales',unit='widgets',valid_time=request['future_timestamps'][1],value=14,
        source_available_at=ledger._now())
    first = bridge.record_lesson(decision_id=did, lesson='Baseline underpredicted; promotion is an unverified explanation.', **cuts)
    assert first['scoring']['metrics']['mae'] == 1.5
    assert first['scoring']['metrics']['rmsle'] > 0 and first['version'] == 1
    retry = bridge.record_lesson(decision_id=did, lesson=first['narrative']['lesson'], **cuts)
    assert retry['reused'] and retry['lesson_id'] == first['lesson_id']
    saved = bridge.lesson(lesson_id=first['lesson_id'],recorded_as_of=ledger._now())
    ledger.clock = FixedClock(origin+timedelta(days=4))
    ledger.append_actual(series_id='sales',unit='widgets',valid_time=request['future_timestamps'][1],value=20,
        source_available_at=ledger._now())
    query = dict(series_id='sales',unit='widgets',horizon=2,source_as_of=ledger._now(),recorded_as_of=ledger._now())
    retrieved = bridge.retrieve_lessons(**query)
    lesson = retrieved['lessons'][0]
    assert lesson['scoring']['metrics']['mae'] == 1.5
    assert lesson['current_evidence']['metrics']['mae'] == 4.5
    assert lesson['current_evidence']['changed_since_lesson']
    assert bridge.lesson(lesson_id=first['lesson_id'],recorded_as_of=ledger._now()) == saved
    assert ledger.decision(did)['inputs'] == original
    second = bridge.record_lesson(decision_id=did,lesson='Revised actuals changed the measured error.',
        source_as_of=ledger._now(),recorded_as_of=ledger._now(),previous_lesson_id=first['lesson_id'])
    assert second['version'] == 2 and second['previous_lesson_id'] == first['lesson_id']
    assert bridge.retrieve_lessons(**query)['lessons'][0]['lesson_id'] == second['lesson_id']
    assert bridge.retrieve_lessons(**{**query,**cuts})['lessons'][0]['lesson_id'] == first['lesson_id']
    assert session.engine._execution_stats['provider_calls'] == 1


def test_retrieval_exact_identity_context_visibility_and_limit(case):
    bridge, ledger, _, _, _, origin = case
    label = dict(key='promotion',value='planned',valid_from=origin.isoformat(),
        valid_to=(origin+timedelta(days=3)).isoformat(),source_available_at=origin.isoformat(),source_ref='fixture:plan')
    packets = [decision(case,context=[label]) for _ in range(4)]
    cuts = mature(case)
    ids = [bridge.record_lesson(decision_id=p['decision_id'],lesson=f'Lesson {i}',**cuts)['lesson_id']
           for i,p in enumerate(packets)]
    query = dict(series_id='sales',unit='widgets',horizon=2,context_filters={'promotion':'planned'},**cuts)
    result = bridge.retrieve_lessons(**query)
    assert result['returned'] == 3 and result['matching_decisions'] == 4
    assert [p['lesson_id'] for p in result['lessons']] == ids[::-1][:3]
    for changed in [{'unit':None},{'series_id':'other'},{'horizon':3},{'context_filters':{'promotion':'cancelled'}},
                    {'source_as_of':origin.isoformat()},{'recorded_as_of':origin.isoformat()}]:
        assert bridge.retrieve_lessons(**(query|changed))['returned'] == 0
    with pytest.raises(ForecastAdapterError):bridge.retrieve_lessons(**query,limit=4)
    assert ledger.SCHEMA_VERSION == 4


def test_store_and_hermes_adapter_are_explicit_and_preserve_evidence(case):
    bridge, ledger, _, _, _, _ = case
    packet = decision(case); cuts = mature(case)
    lesson = bridge.record_lesson(decision_id=packet['decision_id'],lesson='A' * 600,**cuts)
    calls = []
    class Store:
        def put(self,namespace,key,value):
            calls.append((namespace,key,deepcopy(value)));value['narrative']['lesson']='consumer mutation'
    args = dict(lesson_id=lesson['lesson_id'],recorded_as_of=ledger._now())
    bridge.put(Store(),('tenant','forecast-lessons'),**args)
    bridge.put(Store(),('tenant','forecast-lessons'),**args)
    assert calls[0] == calls[1]
    assert bridge.lesson(**args)['narrative']['lesson'] == 'A' * 600
    class FailingStore:
        def put(self,*args):raise RuntimeError('External store unavailable')
    with pytest.raises(RuntimeError):bridge.put(FailingStore(),('tenant',),**args)
    adapter = HermesMemoryAdapter(bridge)
    update = adapter.lesson_update(**args)
    assert update['tool'] == 'memory' and not update['external_write_performed']
    assert update['arguments']['operations'][0]['action'] == 'add'
    assert 'last_value' in update['entry'] and 'unverified_excerpt' in update['entry']
    assert json.loads(update['entry'].split(' ',1)[1])['narrative_truncated']
    replacement = adapter.lesson_update(**args,existing_entry=update['entry'])
    assert replacement['arguments']['operations'][0]['old_text'] == update['entry']
    with pytest.raises(ForecastAdapterError):adapter.lesson_update(**args,existing_entry='Other memory entry')
    recalled = adapter.recall(series_id='sales',unit='widgets',horizon=2,**cuts)
    record = json.loads(recalled['context'])['evidence_records'][0]
    assert record['execution']['execution_id'] == packet['verified_execution']['execution_id']
    assert not record['business_explanation_validated']


def test_negative_values_are_not_silently_clipped_for_rmsle(case):
    bridge, _, _, _, _, origin = case
    packet = decision(case)
    pending = bridge.check_claims(decision_id=packet['decision_id'],claims=[{'field':'mae','value':0}],
        source_as_of=origin.isoformat(),recorded_as_of=origin.isoformat())
    assert pending['checks'][0]['status'] == 'unverified'
    cuts = mature(case, values=(-1,14))
    review = bridge.decision(decision_id=packet['decision_id'],**cuts)
    assert review['scoring']['complete']
    assert review['scoring']['metrics']['rmsle'] is None
    assert review['scoring']['metrics']['rmsle_status'] == 'negative_values_not_supported'


def test_executable_example_keeps_ledger_and_makes_no_external_write(tmp_path):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]/'src')
    result = subprocess.run([sys.executable,'-m','gnomon.examples.memory_bridge'],cwd=tmp_path,
        env=env,text=True,capture_output=True)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output['forecast_calls'] == 1 and output['external_memory_writes'] == 0
    assert output['retrieved']['lessons'][0]['current_evidence']['changed_since_lesson']
    assert (tmp_path/'memory-bridge.db').exists()


def test_comparison_claim_rejects_false_zero_demand_explanation():
    checked = EvidenceMemory.check_comparison_claim(left=[.01, .01], right=[0, 0], actuals=[0, 0],
        metric='rmsle', relation='lower')
    assert checked['status'] == 'contradicted' and checked['right_score'] == 0
    assert not checked['source_verified']
    with pytest.raises(ForecastAdapterError):
        EvidenceMemory.check_comparison_claim(left=[True], right=[0], actuals=[0], metric='mae', relation='equal')


def test_comparison_card_requires_matched_matured_evidence(case):
    b, ledger, session, request, forecast, origin = case
    other = session.forecast('historical_mean', request)
    ids = [forecast['execution_id'], other['execution_id']]
    cuts = dict(source_as_of=origin.isoformat(), recorded_as_of=origin.isoformat())
    assert b.comparison_card(execution_ids=ids, **cuts)['status'] == 'insufficient_comparative_evidence'
    did = decision(case)['decision_id']
    cuts = mature(case)
    lesson = b.record_lesson(decision_id=did, lesson='Last value may be useful.', **cuts)
    before = session.engine._execution_stats['provider_calls']
    card = b.comparison_card(execution_ids=ids, **cuts)
    assert card['ranking'][0]['provider'] == 'last_value'
    assert card['n'] == 2 and card['matched_origins'] == 1
    assert card['ledger_writes'] == card['provider_calls'] == 0
    compact = HermesMemoryAdapter(b).recall_compact(execution_ids=ids,
        series_id='sales', unit='widgets', horizon=2, **cuts)
    context = json.loads(compact['context'])
    assert context['lessons'][0]['lesson_id'] == lesson['lesson_id']
    assert context['lessons'][0]['hypothesis_status'] == 'unverified'
    assert session.engine._execution_stats['provider_calls'] == before
    unrelated = session.forecast('historical_mean', {**request, 'unit': 'kg'})
    assert b.comparison_card(execution_ids=[ids[0], unrelated['execution_id']], **cuts)['ranking'] == []
    assert b.comparison_card(execution_ids=[ids[0]], **cuts)['ranking'] == []


def test_brief_recall_preserves_facts_without_workflow_instructions(case):
    b, ledger, session, request, forecast, origin = case
    other = session.forecast('historical_mean', request)
    did = decision(case)['decision_id'];cuts = mature(case)
    b.record_lesson(decision_id=did, lesson='Unverified explanation ' * 30, **cuts)
    adapter = HermesMemoryAdapter(b)
    args = dict(execution_ids=[forecast['execution_id'], other['execution_id']],
        series_id='sales', unit='widgets', horizon=2, **cuts)
    brief = adapter.recall_brief(**args);compact = adapter.recall_compact(**args)
    overview = json.loads(brief['context'])
    assert overview['models'][0]['execution_id'] == forecast['execution_id']
    assert overview['matched_points'] == 2 and overview['metric'] == 'rmsle'
    assert overview['lessons'][0]['hypothesis_status'] == 'unverified'
    assert len(overview['lessons'][0]['hypothesis']) <= 160
    assert 'instruction' not in overview
    assert len(brief['context']) < len(compact['context'])
    assert brief['comparison'] == compact['comparison']

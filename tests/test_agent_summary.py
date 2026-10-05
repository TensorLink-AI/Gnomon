"""Agent-visible summaries and follow-ups across real operations, without extra execution."""
import json

from gnomon import GnomonSession, TemporalLedger
from gnomon.agent_summary import attach_summary
from gnomon.result_refs import ResultLimits, ResultReferences
from gnomon.ids import FixedClock
from test_backtesting import configured, source, at
from test_adaptive_router import live, _drive  # noqa: F401


def followup(result, purpose):
    return next(f for f in result['agent_summary']['followups'] if f['purpose'] == purpose)


def test_inspection_and_statistic_preserve_scope_and_need_no_models(tmp_path):
    with GnomonSession.from_config() as session:
        inspected = session.call('gnomon_inspect', {'input': str(source(tmp_path)), 'unit': 'widgets'}, compact=False)
        summary = inspected['agent_summary']
        assert summary['operation'] == 'inspect'
        assert summary['result']['series_count'] == 1
        action = followup(inspected, 'calculate_observed_statistic')
        assert action['requires'] == ['statistic']
        reply = session.call(action['tool'], {**action['arguments'], 'statistic': 'mean'}, compact=False)
        assert reply['agent_summary']['result']['value'] == 15.5
        assert reply['agent_summary']['scope']['unit'] == 'widgets'
        assert reply['agent_summary']['scope']['window']['closed'] == 'both'
        assert reply['agent_summary']['basis']['kind'] == 'observed_statistic'
        assert reply['agent_summary']['followups'] == []
        assert reply['execution_diagnostics']['provider_calls'] == 0


def test_direct_forecast_has_no_invented_evidence_or_outcome_actions():
    with GnomonSession.from_config() as session:
        reply = session.forecast('last_value', {'history': [1, 2], 'horizon': 2})
        overview = reply['agent_summary']
        assert overview['result']['point'] == [2, 2] or overview['result']['point'] == (2, 2)
        assert overview['scope']['horizon'] == 2
        assert overview['basis'] == {'kind': 'caller_selected_provider'}
        assert overview['followups'] == []
        assert 'confidence' not in overview
        assert reply['execution_diagnostics']['provider_calls'] == 1


def test_forecast_actuals_partial_then_complete_review_using_followups(tmp_path):
    ledger = TemporalLedger(tmp_path / 'loop.db', clock=FixedClock(at(5)))
    with GnomonSession.from_config(ledger=ledger) as session:
        session.allow_outcome_writes = True
        reply = session.call('gnomon_forecast', {'provider': 'last_value', 'request': {
            'history': [1, 2], 'horizon': 2, 'series_id': 'sales', 'unit': 'widgets',
            'future_timestamps': [at(6).isoformat(), at(7).isoformat()]}}, compact=False)
        read = followup(reply, 'inspect_recorded_forecast')
        assert read['requires'] == [] and read['effect'] == 'read'
        record = session.call(read['tool'], read['arguments'], compact=False)
        assert record['result']['execution_id'] == reply['execution_id']
        submit = followup(reply, 'submit_observed_actual')
        score = followup(reply, 'score_available_outcomes')
        assert submit['effect'] == score['effect'] == 'ledger_write'
        assert score['requires'] == ['source_as_of', 'recorded_as_of']
        ledger.clock = FixedClock(at(8))
        cuts = {'source_as_of': at(8).isoformat(), 'recorded_as_of': at(8).isoformat()}
        pending = session.call(score['tool'], {**score['arguments'], **cuts}, compact=False)
        assert pending['agent_summary']['status']['scoring'] == 'pending'
        for day in (6, 7):
            session.call(submit['tool'], {**submit['arguments'], 'valid_time': at(day).isoformat(),
                'value': 3, 'source_available_at': at(day).isoformat()}, compact=False)
            review = session.call(score['tool'], {**score['arguments'], **cuts}, compact=False)
            overview = review['agent_summary']
            assert overview['operation'] == 'outcome_review'
            assert overview['status']['operation'] == 'ok'
            assert overview['status']['scoring'] == ('partial' if day == 6 else 'complete')
            assert overview['result']['records'][0]['coverage']['matched_steps'] == day - 5
            assert overview['result']['records'][0]['mae'] == 1
            assert review['execution_diagnostics']['provider_calls'] == 0
        assert overview['limitations'] == []


def test_outcome_write_permissions_and_unscoreable_forecasts(tmp_path):
    with GnomonSession.from_config(ledger=TemporalLedger(tmp_path / 'ledger.db')) as session:
        reply = session.forecast('last_value', {'history': [1, 2], 'horizon': 1})
        assert [x['purpose'] for x in reply['agent_summary']['followups']] == ['inspect_recorded_forecast']
        reply = session.forecast('last_value', {'history': [1, 2], 'horizon': 1, 'series_id': 'x',
            'future_timestamps': [at(6).isoformat()]})
        assert 'submit_observed_actual' not in [x['purpose'] for x in reply['agent_summary']['followups']]


def test_saved_comparison_exposes_matched_scope_and_retrieves_without_model_calls(tmp_path):
    session, ref = configured(tmp_path)
    with session:
        reply = session.call('gnomon_evaluate', {'data_ref': ref, 'candidates': ['trend'],
            'baseline': 'last_value', 'horizon': 2}, compact=False)
        summary = reply['agent_summary']
        assert summary['operation'] == 'model_comparison'
        assert summary['basis']['coverage']['matched_folds'] == 4
        assert summary['basis']['ranking_policy']['metric'] == 'mae'
        action = followup(reply, 'retrieve_saved_comparison')
        saved = session.call(action['tool'], action['arguments'], compact=False)
        assert saved['agent_summary']['result']['scores'] == summary['result']['scores']
        assert saved['execution_diagnostics']['provider_calls'] == 0


def test_router_summary_explains_actual_served_model_and_fallback(live):
    ledger, session = live({'candidates': ['last_value'], 'baseline': 'historical_mean',
                           'min_origins': 3, 'recent_origins': 5, 'lookback_seconds': 10**7})
    replies = _drive(ledger, session, 5)
    first, last = replies[0]['agent_summary'], replies[-1]['agent_summary']
    assert first['basis']['kind'] == 'configured_router'
    assert first['basis']['evidence_based'] is False
    assert last['basis']['served_provider'] == 'last_value'
    assert last['references']['routing_decision_id']
    assert last['basis']['evidence_based'] is True


def test_large_forecast_has_retrievable_overview_and_obeys_response_bound():
    with GnomonSession.from_config() as session:
        session.results = ResultReferences(ResultLimits(max_response_bytes=2048))
        reply = session.call('gnomon_forecast', {'provider': 'last_value', 'request': {
            'history': [1, 2], 'horizon': 1000}})
        assert len(json.dumps(reply, ensure_ascii=False, separators=(',', ':')).encode()) <= 2048
        action = reply['agent_summary_read']
        page = session.call(action['tool'], action['arguments'])
        summary = json.loads(page['text'])
        assert summary['result']['point_count'] == 1000
        assert 'point' not in summary['result']
        assert summary['result']['point_pointer'] == '/result/point'
        assert page['execution_diagnostics']['provider_calls'] == 0


def test_advisory_routing_is_not_a_forecast_or_model_comparison():
    reply = attach_summary({'status': 'ok', 'study_id': 's', 'scores': {},
        'recommendation': 'base', 'recommendation_role': 'advisory', 'fallback_used': True,
        'reason': 'insufficient_folds'}, 'route')
    summary = reply['agent_summary']
    assert summary['operation'] == 'route'
    assert summary['result']['recommendation'] == 'base'
    assert summary['basis']['fallback_used'] is True
    assert summary['followups'] == []


def test_cli_and_mcp_present_the_same_forecast_overview(capsys):
    from gnomon.cli import main
    from gnomon.mcp_server import _handle
    request = {'history': [10, 12, 11], 'horizon': 2}
    assert main(['infer', '--provider', 'last_value', '--request', json.dumps(request)]) == 0
    cli = json.loads(capsys.readouterr().out)
    with GnomonSession.from_config() as session:
        reply = _handle({'method': 'tools/call', 'params': {'name': 'gnomon_forecast',
            'arguments': {'provider': 'last_value', 'request': request}}}, session=session)
        assert reply['isError'] is False
        mcp = reply['structuredContent']
        for field in ('operation', 'status', 'scope', 'result', 'basis', 'limitations', 'followups'):
            assert mcp['agent_summary'][field] == cli['agent_summary'][field]
        assert mcp['execution_diagnostics']['provider_calls'] == 1

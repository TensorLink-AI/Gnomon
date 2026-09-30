"""CLI/MCP recall and opt-in Hermes hook exercise real ledger evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_memory_bridge import case, decision, mature  # noqa: F401
from gnomon import GnomonSession
from gnomon.memory_api import memory_call
from gnomon.mcp_server import _handle


@pytest.fixture
def saved(case):
    bridge, ledger, session, request, forecast, origin = case
    packet = decision(case)
    cuts = mature(case)
    bridge.record_lesson(decision_id=packet['decision_id'], lesson='Test hypothesis; not verified causality.', **cuts)
    return case, dict(operation='recall', series_id='sales', unit='widgets', horizon=2, **cuts)


def test_scope_cutoffs_bounds_and_no_writes(saved):
    (bridge, ledger, session, *rest), query = saved
    before = session._execution_counts()
    result = memory_call(ledger, query)
    assert result['returned'] == 1
    assert json.loads(result['context'])['evidence_records'][0]['business_explanation_validated'] is False
    for changes in ({'series_id': 'other'}, {'unit': None}, {'horizon': 1},
                    {'recorded_as_of': '2026-01-19T00:00:00Z'}, {'source_as_of': '2026-01-19T00:00:00Z'}):
        assert memory_call(ledger, query | changes)['returned'] == 0
    small = memory_call(ledger, query | {'max_context_chars': 1024})
    assert len(small['context']) <= 1024 and small['truncated']
    assert session._execution_counts() == before


def test_cli_skips_provider_loading_and_mcp_matches(saved, tmp_path):
    (bridge, ledger, session, *rest), query = saved
    cfg = tmp_path / 'providers.toml'
    cfg.write_text('ledger_path="ledger.db"\n[providers.broken]\nkind="factory"\nentrypoint="missing_module:nope"\n')
    run = subprocess.run([sys.executable, '-m', 'gnomon.cli', 'memory', '--providers-config', str(cfg),
                          '--arguments', json.dumps(query)], text=True, capture_output=True)
    assert run.returncode == 0, run.stderr + run.stdout
    cli = json.loads(run.stdout)
    response = _handle({'jsonrpc':'2.0', 'id':1, 'method':'tools/call',
        'params':{'name':'gnomon_memory','arguments':query}}, session=session)
    mcp = json.loads(response['content'][0]['text'])
    assert cli['context'] == mcp['context'] and cli['returned'] == 1
    assert 'gnomon_memory' in {v['name'] for v in session.tools()}
    cfg.write_text('ledger_path="missing.db"\n')
    with pytest.raises(Exception):
        GnomonSession.from_config(cfg, memory_only=True)
    assert not (tmp_path / 'missing.db').exists()


def test_automatic_recall_is_opt_in_and_cutoff_bounded(saved):
    (bridge, ledger, session, request, *rest), query = saved
    assert 'memory' not in session.forecast('last_value', request)
    session.memory_config = {'auto_recall': True}
    assert session.forecast('last_value', request)['memory']['status'] == 'skipped'
    # Historical request cannot see the subsequently recorded lesson.
    request = request | {'recorded_time_cutoff': request['cutoff']}
    assert session.forecast('last_value', request)['memory']['returned'] == 0


def test_hermes_hook_real_cli_and_project_isolation(saved, tmp_path, monkeypatch):
    (_, ledger, session, *rest), query = saved
    plugin = Path(__file__).resolve().parents[1] / 'integrations/hermes/gnomon-memory/__init__.py'
    spec = importlib.util.spec_from_file_location('gnomon_hermes_test', plugin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hooks = {}
    class Context:
        def register_hook(self, name, callback):
            hooks[name] = callback
    module.register(Context())
    cfg = tmp_path / 'providers.toml'
    cfg.write_text('ledger_path="ledger.db"\n')
    hookcfg = tmp_path / 'memory.json'
    hookcfg.write_text(json.dumps(dict(project_root=str(tmp_path), providers_config=str(cfg),
        command=[sys.executable, '-m', 'gnomon.cli'], match_terms=['sales'], query=query)))
    monkeypatch.setenv('GNOMON_MEMORY_CONFIG', str(hookcfg))
    monkeypatch.chdir(tmp_path)
    hook = hooks['pre_llm_call']
    assert 'Test hypothesis' in hook(user_message='Forecast sales', future_field=True)['context']
    assert hook(user_message='Unrelated task') is None
    monkeypatch.chdir(tmp_path.parent)
    assert hook(user_message='Forecast sales') is None
    monkeypatch.chdir(tmp_path)
    hookcfg.write_text('{}')
    assert hook(user_message='Forecast sales') is None


def test_opt_in_config_positive_recall_and_stdio(saved, tmp_path):
    (bridge, ledger, session, request, *rest), query = saved
    cfg = tmp_path / 'providers.toml'
    cfg.write_text('ledger_path="ledger.db"\n[memory]\nauto_recall=true\nledger_ref="sales-project"\n')
    with GnomonSession.from_config(cfg) as configured:
        run = configured.forecast('last_value', request | {
            'cutoff': query['source_as_of'], 'recorded_time_cutoff': query['recorded_as_of'],
            'future_timestamps': ['2026-01-24T00:00:00Z', '2026-01-25T00:00:00Z']})
        assert run['memory']['returned'] == 1
        assert json.loads(run['memory']['context'])['evidence_records'][0]['ledger_ref'] == 'sales-project'
    messages = [{'id': 1, 'method': 'tools/call', 'params': {'name': 'gnomon_memory', 'arguments': query}}]
    run = subprocess.run([sys.executable, '-m', 'gnomon.cli', 'mcp', 'serve', '--providers-config', str(cfg)],
        input=''.join(json.dumps(m)+'\n' for m in messages), capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)['result']['structuredContent']
    assert result['returned'] == 1 and result['provider_calls'] == 0


def test_decision_review_and_invalid_operation(saved):
    (bridge, ledger, session, *rest), query = saved
    recalled = memory_call(ledger, query)
    did = json.loads(recalled['context'])['evidence_records'][0]['decision_id']
    result = memory_call(ledger, dict(operation='decision', decision_id=did,
        source_as_of=query['source_as_of'], recorded_as_of=query['recorded_as_of']))
    assert json.loads(result['context'])['evidence_records'][0]['scoring_status'] == 'complete'
    from gnomon.forecast_adapter import ForecastAdapterError
    with pytest.raises(ForecastAdapterError):
        memory_call(ledger, query | {'operation': 'record_lesson'})

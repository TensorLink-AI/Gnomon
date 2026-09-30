"""Pinned Hermes integration, isolated homes, real hooks/schemas, no model calls."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hermes-root', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = args.hermes_root.resolve()
    pin = (repo / 'integrations/hermes/gnomon-memory/hermes-revision.txt').read_text().strip()
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    assert actual == pin, (actual, pin)
    sys.path.insert(0, str(root))
    # Preserve the invoking Gnomon environment for subprocess CLI reads.
    os.environ['PYTHONPATH'] = os.pathsep.join([str(repo / 'src'), str(root)])
    from gnomon import EvidenceMemory, GnomonSession, TemporalLedger
    from gnomon.ids import FixedClock
    with TemporaryDirectory(prefix='gnomon-hermes-compat-') as temp:
        work = Path(temp)
        home, project = work / 'home', work / 'project'
        home.mkdir(); project.mkdir()
        os.environ['HERMES_HOME'] = str(home)
        os.environ['HERMES_ENABLE_PROJECT_PLUGINS'] = 'false'
        os.environ.pop('GNOMON_MEMORY_CONFIG', None)
        (home / 'config.yaml').write_text('plugins:\n  enabled: [gnomon-memory]\n')
        shutil.copytree(repo / 'integrations/hermes/gnomon-memory', home / 'plugins/gnomon-memory')
        os.chdir(project)
        from hermes_cli.plugins import get_plugin_manager
        from agent.turn_context import _collect_pre_llm_call_context, compose_user_api_content
        from tools.mcp_tool_schema import _convert_mcp_schema
        from tools.schema_sanitizer import sanitize_tool_schemas
        manager = get_plugin_manager()
        manager.discover_and_load()
        assert any(p.manifest.name == 'gnomon-memory' and p.enabled for p in manager._plugins.values()), manager._plugins
        agent = SimpleNamespace(session_id='compat', model='unused', platform='cli')
        def context(message='sales forecast'):
            return _collect_pre_llm_call_context(agent, effective_task_id='compat', turn_id='1',
                original_user_message=message, messages=[], conversation_history=[])
        assert context() == ''  # plugin installed, project opt-in absent
        origin = datetime(2026, 1, 20, tzinfo=timezone.utc)
        ledger = TemporalLedger(project / 'evidence.db', clock=FixedClock(origin))
        with GnomonSession.from_config(ledger=ledger) as session:
            request = dict(history=[10, 11, 12], horizon=1, series_id='sales', unit='widgets',
                timestamps=[(origin-timedelta(days=i)).isoformat() for i in (2, 1, 0)],
                future_timestamps=[(origin+timedelta(days=1)).isoformat()], cutoff=origin.isoformat())
            run = session.forecast('last_value', request)
            bridge = EvidenceMemory(ledger, ledger_ref='compat')
            decision = bridge.record_decision(execution_id=run['execution_id'], expected_request=request, rationale='Baseline hypothesis')
            ledger.clock = FixedClock(origin+timedelta(days=2))
            cuts = dict(source_as_of=ledger._now(), recorded_as_of=ledger._now())
            ledger.append_actual(series_id='sales', unit='widgets', valid_time=request['future_timestamps'][0],
                value=13, source_available_at=ledger._now())
            bridge.record_lesson(decision_id=decision['decision_id'], lesson='COMPAT_EVIDENCE: baseline underpredicted.', **cuts)
            cfg = project / 'providers.toml'
            cfg.write_text('ledger_path="evidence.db"\n')
            query = dict(operation='recall', series_id='sales', unit='widgets', horizon=1, **cuts)
            hook_config = dict(project_root=str(project), providers_config=str(cfg),
                command=[sys.executable, '-m', 'gnomon.cli'], match_terms=['sales'], query=query)
            config = project / 'memory.json'
            def configure(value):
                config.write_text(json.dumps(value))
                os.environ['GNOMON_MEMORY_CONFIG'] = str(config)
            configure(hook_config)
            recalled = context()
            assert 'COMPAT_EVIDENCE' in recalled, recalled
            wire = compose_user_api_content('sales forecast', '', recalled)
            assert wire.startswith('sales forecast') and 'COMPAT_EVIDENCE' in wire
            assert context('unrelated question') == ''
            os.chdir(work)
            assert context() == ''
            os.chdir(project)
            agent.platform = 'telegram'
            assert context() == ''
            agent.platform = 'cli'
            configure(hook_config | {'query': query | {'series_id': 'other'}})
            assert context() == ''
            configure(hook_config | {'command': [str(work / 'missing-executable')]})
            assert context() == ''
            configure(hook_config)
            # Real MCP conversion and provider-facing sanitizer, not a mock.
            for tool in session.tools():
                converted = _convert_mcp_schema('gnomon', SimpleNamespace(name=tool['name'],
                    description=tool['description'], input_schema=tool['inputSchema']))
                visible = sanitize_tool_schemas([{'type':'function', 'function':converted}])[0]['function']['parameters']
                assert set(tool['inputSchema'].get('properties', {})) <= set(visible['properties']), tool['name']
                if tool['name'] == 'gnomon_ledger':
                    assert 'review_decision' in visible['properties']['operation']['enum']
            exact = session.call('gnomon_capabilities', {'schema_tool':'gnomon_ledger',
                'schema_variant':'review_decision'}, compact=False)
            assert 'decision_id' in exact['schema']['required']
        manager.unload()
        (home / 'config.yaml').write_text('plugins:\n  enabled: []\n')
        manager.discover_and_load(force=True)
        assert context() == ''  # actual disabled-plugin discovery
        assert not (home / 'memories' / 'MEMORY.md').exists()
        print(json.dumps({'status':'ok', 'hermes_revision': pin,
            'verified':['discovery', 'enablement', 'disabled', 'hook_dispatch', 'context_composition',
                        'project_scope', 'gateway_excluded', 'empty_recall', 'cli_failure', 'model_visible_schemas'],
            'model_calls':0}))


if __name__ == '__main__':
    main()

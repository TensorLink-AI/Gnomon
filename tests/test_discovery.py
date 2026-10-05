import json
from pathlib import Path
import runpy
import tomllib
import pytest
from gnomon import GnomonSession, TemporalLedger
from gnomon.contracts import GnomonError
from gnomon.discovery import DESCRIPTION, TASKS
from gnomon.cli import main

ROOT = Path(__file__).resolve().parents[1]

def test_default_task_index_and_executable_smoke_call():
    with GnomonSession.from_config() as s:
        reply = s.call('gnomon_capabilities', {})
        assert reply['tasks']['forecast'] == 'available'
        assert reply['tasks']['review'] == 'open_configured_ledger'
        assert reply['tasks']['temporal'] == 'enable_temporal'
        detail = s.call('gnomon_capabilities', {'task': 'forecast'})
        assert detail['execution_diagnostics']['provider_calls'] == 0
        schema = detail['schema_call']
        assert s.call(schema['tool'], schema['arguments'])['schema']['required'] == ['provider', 'request']
        example = detail['call_template']
        actual = s.call(example['tool'], example['arguments'], compact=False)
        assert list(actual['result']['point']) == [11, 11]


def test_empty_registry_is_not_ready_to_forecast_or_compare():
    with GnomonSession() as s:
        assert s.capabilities()['tasks']['forecast'] == 'register_provider'
        assert s.capabilities(task='compare')['availability'] == 'register_two_providers'


def test_permissions_control_task_availability_and_schema_exposure(tmp_path):
    with GnomonSession(ledger=TemporalLedger(tmp_path/'ledger.db'), enable_temporal=True) as s:
        assert s.capabilities(task='review')['availability'] == 'available'
        write = s.capabilities(task='submit_actuals')
        assert write['availability'] == 'enable_outcome_writes'
        assert write['schema_call'] is None and write['call_template'] is None
        s.allow_outcome_writes = True
        assert s.capabilities(task='submit_actuals')['availability'] == 'available'
        for task in TASKS:
            detail = s.capabilities(task=task)
            if detail['schema_call']:
                call = detail['schema_call']
                assert 'schema' in s.call(call['tool'], call['arguments'], compact=False)
        assert s.capabilities(task='temporal')['availability'] == 'available'


@pytest.mark.parametrize('args', [{'task': 'invented'}, {'task': []},
    {'task': 'forecast', 'schema_tool': 'gnomon_forecast'}])
def test_bad_or_ambiguous_task_queries_are_rejected(args):
    with GnomonSession.from_config() as s:
        with pytest.raises(GnomonError):s.call('gnomon_capabilities', args)


def test_cli_task_discovery_and_conflicting_modes(capsys):
    assert main(['capabilities', '--task', 'forecast']) == 0
    assert json.loads(capsys.readouterr().out)['availability'] == 'available'
    assert main(['capabilities', '--task', 'forecast', '--config-schema']) == 2
    capsys.readouterr()


def test_listing_metadata_and_entrypoints_stay_consistent():
    project = tomllib.loads((ROOT/'pyproject.toml').read_text())['project']
    assert project['description'] == DESCRIPTION
    assert len(DESCRIPTION) <= 100
    assert project['scripts']['gnomon'] == project['scripts']['gnomon-forecast']
    artifacts = runpy.run_path(str(ROOT/'scripts/prepare_discovery.py'))['artifacts']()
    for path, value in artifacts.items():
        assert json.loads((ROOT/path).read_text()) == value
    manifest = artifacts['server.json']
    assert 'mcp-name: '+manifest['name'] in (ROOT/'README.md').read_text()
    assert DESCRIPTION in (ROOT/'llms.txt').read_text()


def test_task_index_survives_small_response_limit():
    from gnomon.result_refs import ResultLimits, ResultReferences
    with GnomonSession.from_config() as s:
        s.results = ResultReferences(ResultLimits(max_response_bytes=2048))
        reply = s.call('gnomon_capabilities', {})
        assert reply['tasks']['forecast'] == 'available'
        assert reply['task_help']['arguments'] == {'task': 'forecast'}
        assert len(json.dumps(reply, separators=(',', ':')).encode()) <= 2048

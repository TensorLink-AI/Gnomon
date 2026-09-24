import io
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from gnomon import GnomonSession
from gnomon.cli import main
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.http_transport import JSONTransport
from gnomon import onboarding

_REAL_CALL = JSONTransport.call


@pytest.fixture(autouse=True)
def profile(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path/'config'))
    monkeypatch.setattr(JSONTransport, 'call', Mock(return_value=[]))
    return onboarding.credential_path()


def test_mcp_signup_discovery_does_not_write_or_call_network(profile, monkeypatch):
    monkeypatch.setattr(JSONTransport, 'call', Mock(side_effect=AssertionError('network')))
    with GnomonSession.from_config() as session:
        caps = session.call('gnomon_capabilities', {}, compact=True)
        info = caps['onboarding']['ephemeris']
        assert info['connection_status'] == 'not_configured'
        assert info['signup_url'] == 'https://ephemeris.cascade.industries'
        assert info['local_models_require_account'] is False
        assert info['provider_calls'] == 0
        assert 'once_per_conversation' in info['offer_policy']
        assert session.forecast('last_value', {'history': [1, 2], 'horizon': 1})['result']['point'] == (2,)
    assert not profile.parent.exists()


def test_noninteractive_guidance_never_reads_secret_or_writes(profile, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'stdin', io.StringIO('do-not-read'))
    assert main(['connect', 'ephemeris']) == 0
    response = json.loads(capsys.readouterr().out)
    assert response['ephemeris']['connection_status'] == 'not_configured'
    assert sys.stdin.tell() == 0
    assert not profile.exists()


def test_connect_save_auto_load_and_explicit_config_override(profile, monkeypatch, capsys, tmp_path):
    token = 'test-token-not-a-real-credential'
    monkeypatch.setattr(sys, 'stdin', io.StringIO(token+'\n'))
    monkeypatch.setattr(JSONTransport, 'call', Mock(return_value=[]))
    assert main(['connect', 'ephemeris', '--token-stdin']) == 0
    out = capsys.readouterr().out
    assert token not in out
    assert json.loads(out)['credential_saved']
    assert profile.stat().st_mode & 0o777 == 0o600
    assert profile.parent.stat().st_mode & 0o777 == 0o700
    assert onboarding.read_token(profile) == token
    with GnomonSession.from_config() as s:
        caps = s.capabilities()
        assert caps['onboarding']['ephemeris']['providers'] == ['ephemeris', 'ephemeris/ensemble']
        assert token not in json.dumps(caps)
    toml = tmp_path/'explicit.toml';toml.write_text('schema_version = 1\n')
    with GnomonSession.from_config(toml) as s:
        assert 'ephemeris' not in s.capabilities()['providers']
    with GnomonSession() as s:
        assert s.capabilities()['onboarding']['ephemeris']['configured_in_session'] is False


def test_existing_key_requires_replace_disconnect(profile, monkeypatch, capsys):
    onboarding.save_token('first')
    with pytest.raises(ForecastAdapterError, match='already exists'):
        onboarding.save_token('second')
    monkeypatch.setattr(sys, 'stdin', io.StringIO('second'))
    assert main(['connect', 'ephemeris', '--token-stdin', '--replace']) == 0
    assert onboarding.read_token(profile) == 'second'
    assert main(['connect', 'ephemeris', '--disconnect']) == 0
    assert not profile.exists()
    assert 'second' not in capsys.readouterr().out


def test_private_file_rejects_symlinks_and_public_permissions(profile, tmp_path):
    profile.parent.mkdir(parents=True, mode=0o700)
    external = tmp_path/'external';external.write_text('keep')
    profile.symlink_to(external)
    with pytest.raises(ForecastAdapterError):onboarding.save_token('secret', replace=True)
    assert external.read_text() == 'keep'
    profile.unlink();onboarding.save_token('secret');profile.chmod(0o644)
    with pytest.raises(ForecastAdapterError):onboarding.read_token(profile)
    with pytest.raises(ForecastAdapterError):onboarding.saved_connection()


@pytest.mark.parametrize('token', ['', 'one\ntwo', 'one two', 'x'*4097])
def test_invalid_secret_never_written(profile, token):
    with pytest.raises(ForecastAdapterError):onboarding.save_token(token)
    assert not profile.exists()


def test_saved_token_only_used_in_authorization_header(profile, monkeypatch):
    monkeypatch.setattr(JSONTransport, "call", _REAL_CALL)
    onboarding.save_token('private-test-token')
    transport = JSONTransport(onboarding.GATEWAY_URL, token_file=profile)
    response = Mock();response.read.return_value = b'{"balance_mc":"1234"}'
    response.__enter__ = Mock(return_value=response);response.__exit__ = Mock(return_value=False)
    transport._opener.open = Mock(return_value=response)
    assert transport.call('/balance')['balance_mc'] == '1234'
    req = transport._opener.open.call_args.args[0]
    assert req.get_header('Authorization') == 'Bearer private-test-token'
    assert 'private-test-token' not in req.full_url
    with pytest.raises(ForecastAdapterError):JSONTransport('http://example.com', token_file=profile, allow_http=True)


def test_hidden_prompt_and_cancel(profile, monkeypatch, capsys):
    stream=io.StringIO();monkeypatch.setattr(stream, 'isatty', lambda: True)
    monkeypatch.setattr(sys, 'stdin', stream)
    monkeypatch.setattr(onboarding.getpass, 'getpass', lambda _: '')
    assert main(['connect', 'ephemeris']) == 0
    assert json.loads(capsys.readouterr().out)['connection_status'] == 'cancelled'
    assert not profile.exists()


def test_stdio_mcp_after_connect_exposes_status_without_secret(profile):
    onboarding.save_token('mcp-secret-never-print')
    messages=[{'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2025-06-18','capabilities':{},'clientInfo':{'name':'test','version':'1'}}},
              {'jsonrpc':'2.0','method':'notifications/initialized'},
              {'jsonrpc':'2.0','id':2,'method':'tools/call','params':{'name':'gnomon_capabilities','arguments':{}}}]
    result=subprocess.run([sys.executable,'-m','gnomon.cli','mcp','serve'],input=''.join(json.dumps(m)+'\n' for m in messages),text=True,capture_output=True,timeout=15,
                          env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')})
    assert result.returncode == 0, result.stderr
    assert 'mcp-secret-never-print' not in result.stdout+result.stderr
    reply=next(json.loads(line) for line in result.stdout.splitlines() if json.loads(line).get('id')==2)
    payload=reply['result'].get('structuredContent') or json.loads(reply['result']['content'][0]['text'])
    assert payload.get('summary', payload)['onboarding']['ephemeris']['configured_in_session'] is True


def test_auth_failure_does_not_expose_key_or_delete_profile(profile, monkeypatch, capsys):
    from gnomon.http_transport import InferenceHTTPError
    onboarding.save_token('test-sensitive-key')
    monkeypatch.setattr(JSONTransport, 'call', Mock(side_effect=InferenceHTTPError('service returned HTTP 401', status=401)))
    assert main(['connect', 'ephemeris', '--check']) == 2
    output=capsys.readouterr().out
    assert '401' in output and 'test-sensitive-key' not in output
    assert onboarding.read_token(profile) == 'test-sensitive-key'


def test_balance_check_only_uses_get(profile, monkeypatch, capsys):
    onboarding.save_token('test-sensitive-key')
    call=Mock(return_value={'balance_mc':'10000'})
    monkeypatch.setattr(JSONTransport, 'call', call)
    assert main(['connect', 'ephemeris', '--check']) == 0
    assert json.loads(capsys.readouterr().out)['connection_status'] == 'verified'
    call.assert_called_once_with('/balance')


def test_hidden_prompt_never_falls_back_to_echo(profile, monkeypatch, capsys):
    stream=io.StringIO();monkeypatch.setattr(stream, 'isatty', lambda: True)
    monkeypatch.setattr(sys, 'stdin', stream)
    def broken_prompt(_):
        import warnings
        warnings.warn('cannot hide', onboarding.getpass.GetPassWarning)
    monkeypatch.setattr(onboarding.getpass, 'getpass', broken_prompt)
    assert main(['connect', 'ephemeris']) == 2
    assert not profile.exists()
    assert 'terminal' in capsys.readouterr().out


def test_symlinked_directory_and_relative_xdg_rejected(profile, tmp_path, monkeypatch):
    profile.parent.parent.mkdir(parents=True)
    target=tmp_path/'target';target.mkdir(mode=0o700)
    profile.parent.symlink_to(target, target_is_directory=True)
    with pytest.raises(ForecastAdapterError):onboarding.save_token('test-sensitive-key')
    assert not (target/'ephemeris.token').exists()
    monkeypatch.setenv('XDG_CONFIG_HOME','relative')
    with pytest.raises(ForecastAdapterError):onboarding.credential_path()


def test_connect_discovers_individual_models_and_startup_is_offline(profile, monkeypatch, capsys):
    rows=[{'name':'chronos2','enabled':True,'healthy':True,'covariates':True,'private_metadata':'do-not-cache'},
          {'name':'tirex2','enabled':True,'healthy':True},
          {'name':'offline','enabled':True,'healthy':False},
          {'name':'disabled','enabled':False,'healthy':True}]
    call=Mock(return_value={'models':rows});monkeypatch.setattr(JSONTransport,'call',call)
    monkeypatch.setattr(sys,'stdin',io.StringIO('catalog-test-token'))
    assert main(['connect','ephemeris','--token-stdin'])==0
    info=json.loads(capsys.readouterr().out)['ephemeris']
    assert info['model_catalog']['providers']==['ephemeris/chronos2','ephemeris/tirex2']
    call.assert_called_once_with('/models',allow_list=True)
    assert 'do-not-cache' not in onboarding.catalog_path().read_text()
    assert onboarding.catalog_path().stat().st_mode & 0o777==0o600
    monkeypatch.setattr(JSONTransport,'call',Mock(side_effect=AssertionError('startup network')))
    with GnomonSession.from_config() as s:
        caps=s.capabilities()
        assert set(caps['onboarding']['ephemeris']['providers'])=={'ephemeris','ephemeris/ensemble','ephemeris/chronos2','ephemeris/tirex2'}
        assert caps['providers']['ephemeris/chronos2']['capabilities']['past_covariates'] is True
        assert caps['providers']['ephemeris/tirex2']['capabilities']['past_covariates'] is False
        assert caps['onboarding']['ephemeris']['model_catalog']['status']=='cached'


def test_refresh_preserves_catalog_on_failure_then_replaces_it(profile, monkeypatch, capsys):
    from gnomon.http_transport import InferenceHTTPError
    onboarding.save_token('catalog-test-token')
    monkeypatch.setattr(JSONTransport,'call',Mock(return_value=[{'name':'model-a','enabled':True,'healthy':True}]))
    onboarding.refresh_models();before=onboarding.catalog_path().read_bytes()
    monkeypatch.setattr(JSONTransport,'call',Mock(side_effect=InferenceHTTPError('service returned HTTP 502',status=502)))
    assert main(['connect','ephemeris','--refresh-models'])==2
    assert onboarding.catalog_path().read_bytes()==before
    monkeypatch.setattr(JSONTransport,'call',Mock(return_value=[{'name':'model-b','enabled':True,'healthy':True}]))
    assert main(['connect','ephemeris','--refresh-models'])==0
    with GnomonSession.from_config() as s:
        assert 'ephemeris/model-b' in s.capabilities()['providers']
        assert 'ephemeris/model-a' not in s.capabilities()['providers']
    assert main(['connect','ephemeris','--disconnect'])==0
    assert not onboarding.catalog_path().exists()


def test_connect_discovery_failure_keeps_key_and_reports_recovery(profile, monkeypatch, capsys):
    from gnomon.http_transport import InferenceHTTPError
    monkeypatch.setattr(JSONTransport,'call',Mock(side_effect=InferenceHTTPError('service returned HTTP 401',status=401)))
    monkeypatch.setattr(sys,'stdin',io.StringIO('bad-credential'))
    assert main(['connect','ephemeris','--token-stdin'])==0
    answer=json.loads(capsys.readouterr().out)
    assert answer['credential_saved']
    assert answer['ephemeris']['model_catalog']['status']=='failed'
    assert 'bad-credential' not in json.dumps(answer)
    with GnomonSession.from_config() as s:
        assert s.capabilities()['onboarding']['ephemeris']['providers']==['ephemeris','ephemeris/ensemble']


@pytest.mark.parametrize('names',[['ensemble'],['../evil'],['same','same']])
def test_reserved_invalid_or_duplicate_catalog_names_rejected(profile, monkeypatch, names):
    onboarding.save_token('catalog-test-token')
    monkeypatch.setattr(JSONTransport,'call',Mock(return_value=[{'name':n,'enabled':True,'healthy':True} for n in names]))
    with pytest.raises(ForecastAdapterError):onboarding.refresh_models()
    assert not onboarding.catalog_path().exists()


def test_invalid_saved_catalog_does_not_block_local_models(profile):
    onboarding.save_token('catalog-test-token')
    path=onboarding.catalog_path();path.write_text('invalid json');path.chmod(0o600)
    with GnomonSession.from_config() as s:
        assert s.capabilities()['onboarding']['ephemeris']['model_catalog']['status']=='unavailable'
        assert s.forecast('last_value',{'history':[1,2],'horizon':1})['result']['point']==(2,)


def test_discovered_model_executes_explicitly_without_fallback(profile, monkeypatch):
    from gnomon.http_transport import InferenceHTTPError
    onboarding.save_token('catalog-test-token')
    monkeypatch.setattr(JSONTransport,'call',Mock(return_value=[{'name':'chronos2','enabled':True,'healthy':True}]))
    onboarding.refresh_models()
    reply={'forecasts':[{'quantiles':{'0.5':[3.0,4.0]}}],
           'meta':{'mode':'explicit','models_used':['chronos2']}}
    call=Mock(return_value=reply);monkeypatch.setattr(JSONTransport,'call',call)
    with GnomonSession.from_config() as session:
        result=session.forecast('ephemeris/chronos2',{'history':[1,2,3],'horizon':2})
        assert result['result']['point']==(3.,4.)
        assert call.call_args.args[1]['mode']=='explicit'
        assert call.call_args.args[1]['model']=='chronos2'
        call.reset_mock();call.side_effect=InferenceHTTPError('service returned HTTP 503',status=503)
        from gnomon.contracts import GnomonError
        with pytest.raises(GnomonError) as failure:
            session.forecast('ephemeris/chronos2',{'history':[1,2,3],'horizon':2})
        assert failure.value.details['provider']=='ephemeris/chronos2'
        assert call.call_count==1
        assert call.call_args.args[1]['model']=='chronos2'


def test_hermes_env_capture_imports_without_exposing_key(profile, monkeypatch, capsys):
    token = 'synthetic-hermes-secure-key'
    monkeypatch.setenv(onboarding.HERMES_TOKEN_ENV, token)
    monkeypatch.setattr(sys, 'stdin', io.StringIO('must-not-read'))
    assert main(['connect', 'ephemeris', '--from-env']) == 0
    captured = capsys.readouterr()
    assert token not in captured.out + captured.err
    assert json.loads(captured.out)['credential_saved'] is True
    assert onboarding.read_token(profile) == token
    assert profile.stat().st_mode & 0o777 == 0o600
    assert sys.stdin.tell() == 0
    JSONTransport.call.assert_called_once_with('/models', allow_list=True)
    with GnomonSession.from_config() as s:
        assert 'ephemeris/ensemble' in s.capabilities()['providers']


@pytest.mark.parametrize('token', [None, '', 'invalid token', 'x' * 4097])
def test_hermes_missing_or_invalid_secret_is_not_written(profile, monkeypatch, capsys, token):
    if token is None:
        monkeypatch.delenv(onboarding.HERMES_TOKEN_ENV, raising=False)
    else:
        monkeypatch.setenv(onboarding.HERMES_TOKEN_ENV, token)
    assert main(['connect', 'ephemeris', '--from-env']) == 2
    captured = capsys.readouterr()
    if token:
        assert token not in captured.out + captured.err
    assert not profile.exists()
    JSONTransport.call.assert_not_called()


def test_hermes_does_not_implicitly_replace_or_auto_import(profile, monkeypatch, capsys):
    monkeypatch.setenv(onboarding.HERMES_TOKEN_ENV, 'new-synthetic-key')
    with GnomonSession.from_config() as s:
        assert 'ephemeris' not in s.capabilities()['providers']
    assert not profile.exists()
    onboarding.save_token('existing-synthetic-key')
    assert main(['connect', 'ephemeris', '--from-env']) == 0
    assert onboarding.read_token(profile) == 'existing-synthetic-key'
    JSONTransport.call.assert_not_called()
    assert main(['connect', 'ephemeris', '--from-env', '--replace']) == 0
    assert onboarding.read_token(profile) == 'new-synthetic-key'
    assert 'synthetic-key' not in capsys.readouterr().out


def test_hermes_skill_install_and_repeat_are_safe(profile, tmp_path, monkeypatch, capsys):
    home = tmp_path / 'hermes'
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv(onboarding.HERMES_TOKEN_ENV, 'never-read-this-key')
    for _ in range(2):
        assert main(['connect', 'ephemeris', '--install-hermes-skill']) == 0
    output = capsys.readouterr().out
    assert 'never-read-this-key' not in output
    assert not profile.exists()
    JSONTransport.call.assert_not_called()
    import yaml
    skill = (home / 'skills/connect-ephemeris/SKILL.md').read_text()
    metadata = yaml.safe_load(skill.split('---')[1])
    assert metadata['required_environment_variables'][0]['name'] == onboarding.HERMES_TOKEN_ENV
    assert '--from-env' in skill
    assert (home / 'skills/use-gnomon/SKILL.md').exists()


def test_hermes_install_preserves_custom_skill_and_has_no_partial_changes(profile, tmp_path, monkeypatch, capsys):
    home = tmp_path / 'hermes'
    monkeypatch.setenv('HERMES_HOME', str(home))
    target = home / 'skills/connect-ephemeris/SKILL.md'
    target.parent.mkdir(parents=True)
    target.write_text('my customization')
    assert main(['connect', 'ephemeris', '--install-hermes-skill']) == 2
    assert target.read_text() == 'my customization'
    assert not (home / 'skills/use-gnomon').exists()
    assert not profile.exists()


def test_hermes_install_refuses_redirected_destinations(profile, tmp_path, monkeypatch, capsys):
    home = tmp_path / 'hermes'
    home.symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.setenv('HERMES_HOME', str(home))
    assert main(['connect', 'ephemeris', '--install-hermes-skill']) == 2
    assert not (tmp_path / 'skills').exists()
    monkeypatch.setenv('HERMES_HOME', 'relative')
    assert main(['connect', 'ephemeris', '--install-hermes-skill']) == 2


def test_hermes_setup_discoverable_in_retained_capabilities(profile):
    from gnomon.result_refs import ResultReferences, ResultLimits
    with GnomonSession.from_config() as s:
        caps = s.capabilities()
        setup = caps['onboarding']['ephemeris']['hermes_setup']
        assert setup['skill'] == 'connect-ephemeris'
        assert setup['requires_user_opt_in'] is True
    # The retained capabilities summary must carry the native setup path too.
    refs = ResultReferences(ResultLimits(max_response_bytes=2048))
    try:
        retained = refs.project(caps)
        assert retained['summary']['onboarding']['ephemeris']['hermes_setup'] == setup
    finally:
        refs.close()

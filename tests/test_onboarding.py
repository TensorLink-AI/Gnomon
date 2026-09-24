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


@pytest.fixture(autouse=True)
def profile(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path/'config'))
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
    monkeypatch.setattr(JSONTransport, 'call', Mock(side_effect=AssertionError('network')))
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


def test_saved_token_only_used_in_authorization_header(profile):
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

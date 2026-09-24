#!/usr/bin/env python3
"""Offline integration with an installed/current Hermes source checkout.

Run in an environment with Hermes dependencies, Gnomon and PyYAML installed:
  python scripts/hermes_setup_smoke.py --hermes-source /path/to/hermes-agent
Uses a synthetic secret, temporary homes and a mocked model-catalog GET only.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hermes-source', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.hermes_source.resolve()))
    with tempfile.TemporaryDirectory(prefix='gnomon-hermes-smoke-') as directory:
        root = Path(directory)
        os.environ['HERMES_HOME'] = str(root / 'hermes')
        os.environ['XDG_CONFIG_HOME'] = str(root / 'config')
        # Import only after isolating Hermes's profile; never touch the real .env.
        import yaml
        from tools.skills_tool_setup import _get_required_environment_variables
        from hermes_cli.callbacks import prompt_for_secret
        from hermes_cli.config import load_env
        from gnomon.cli import main as cli
        from gnomon import GnomonSession, onboarding
        from gnomon.http_transport import JSONTransport

        source = Path(__file__).resolve().parents[1]
        skill = (source / 'skills/connect-ephemeris/SKILL.md').read_text()
        fields = _get_required_environment_variables(yaml.safe_load(skill.split('---')[1]))
        assert len(fields) == 1 and fields[0]['name'] == onboarding.HERMES_TOKEN_ENV
        key = 'synthetic-only-hermes-smoke-secret'
        captured = io.StringIO()
        with redirect_stdout(captured), redirect_stderr(captured):
            with patch('hermes_cli.callbacks.masked_secret_prompt', return_value=''):
                skipped = prompt_for_secret(SimpleNamespace(), fields[0]['name'], fields[0]['prompt'])
            assert skipped['skipped'] and not load_env().get(onboarding.HERMES_TOKEN_ENV)
            with patch('hermes_cli.callbacks.masked_secret_prompt', return_value=key):
                result = prompt_for_secret(SimpleNamespace(), fields[0]['name'], fields[0]['prompt'])
            assert result['success'] and not result['skipped']
            assert key not in json.dumps(result)
            assert (root / 'hermes/.env').stat().st_mode & 0o777 == 0o600
            # Hermes's skill passthrough supplies this to the command environment.
            with patch.dict(os.environ, {onboarding.HERMES_TOKEN_ENV: load_env()[onboarding.HERMES_TOKEN_ENV]}):
                with patch.object(JSONTransport, 'call', return_value=[]) as catalog:
                    assert cli(['connect', 'ephemeris', '--from-env']) == 0
                    catalog.assert_called_once_with('/models', allow_list=True)
            assert onboarding.read_token(onboarding.credential_path()) == key
            with GnomonSession.from_config() as session:
                assert 'ephemeris/ensemble' in session.capabilities()['providers']
        assert key not in captured.getvalue()
        print('PASS: native Hermes credential declaration, cancel/capture, private storage, '
              'Gnomon import and provider registration; no secret in outputs, no forecasts.')


if __name__ == '__main__':
    main()

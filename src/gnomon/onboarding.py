"""Optional user-authorized Ephemeris setup. Startup is offline; explicit connection/refresh reads the model catalog."""
from __future__ import annotations

import getpass
import json
import re
from datetime import datetime, timezone
import os
from pathlib import Path
import stat
import sys
import tempfile

from .forecast_adapter import ForecastAdapterError

SIGNUP_URL = 'https://ephemeris.cascade.industries'
GATEWAY_URL = SIGNUP_URL + '/api/v1'


def credential_path():
    base = os.environ.get('XDG_CONFIG_HOME')
    if base and not Path(base).is_absolute():
        raise ForecastAdapterError('XDG_CONFIG_HOME must be an absolute directory.')
    return (Path(base) if base else Path.home() / '.config') / 'gnomon' / 'ephemeris.token'


def _check_private(path, *, directory=False):
    info = path.lstat()
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(info.st_mode):
        raise ForecastAdapterError('Ephemeris credentials require a private regular file and directory, not symlinks.')
    if os.name == 'posix' and (info.st_uid != os.getuid() or info.st_mode & 0o077):
        raise ForecastAdapterError('Ephemeris credentials must be owned by your user: directory mode 700, file mode 600.')


def saved_connection():
    path = credential_path()
    if not path.exists() and not path.is_symlink():
        return False
    _check_private(path.parent, directory=True)
    _check_private(path)
    return True


def read_token(path):
    """Read only the fixed private profile, never a path supplied through MCP."""
    path = Path(path)
    _check_private(path.parent, directory=True)
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(fd) as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or (os.name == 'posix' and
                (info.st_uid != os.getuid() or info.st_mode & 0o077)):
            raise ForecastAdapterError('Ephemeris credential file must be private and owned by your user.')
        token = handle.read(4097).strip()
    return validate_token(token)


def validate_token(token):
    if not isinstance(token, str) or not 1 <= len(token) <= 4096 or any(c.isspace() or not c.isprintable() for c in token):
        raise ForecastAdapterError('Supply one nonempty API key with no whitespace (maximum 4096 characters).')
    return token


def save_token(token, *, replace=False):
    token = validate_token(token)
    path = credential_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _check_private(path.parent, directory=True)
    if path.exists() or path.is_symlink():
        _check_private(path)
        if not replace:
            raise ForecastAdapterError('An Ephemeris connection already exists. Use --replace to replace its credential.')
    fd, name = tempfile.mkstemp(prefix='.ephemeris-', dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(token + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(tmp, path)
        else:
            try:
                os.link(tmp, path)  # Atomic create without overwriting another connection.
            except FileExistsError:
                raise ForecastAdapterError('An Ephemeris connection already exists. Use --replace deliberately.') from None
    finally:
        tmp.unlink(missing_ok=True)
    catalog_path().unlink(missing_ok=True)  # Rotated credentials must rediscover their catalog.


def connection_info(provider_names=(), *, saved=False, catalog=None):
    names = sorted(provider_names)
    return {
        'optional': True, 'signup_url': SIGNUP_URL,
        'connection_status': 'configured_unverified' if names or saved else 'not_configured',
        'configured_in_session': bool(names), 'providers': names,
        'credential_validity': 'not_checked', 'balance': 'not_checked',
        'local_models_require_account': False,
        'connect_command': 'gnomon connect ephemeris',
        'check_command': 'gnomon connect ephemeris --check',
        'refresh_models_command': 'gnomon connect ephemeris --refresh-models',
        'model_catalog': catalog or {'status': 'not_loaded'},
        'offer_policy': 'once_per_conversation_if_not_configured; respect a decline; do not block local forecasting',
        'agent_guidance': 'Offer optional Ephemeris models once. If the user wants them, show signup_url and connect_command. Never request an API key in chat or tool arguments. Configuration is not authorization to spend.',
        'restart_mcp_after_connect': True,
        'provider_calls': 0,
    }


def catalog_path():
    return credential_path().with_name('ephemeris-models.json')


def clean_catalog(rows):
    if not isinstance(rows, list) or len(rows) > 128:
        raise ForecastAdapterError('Ephemeris catalog must contain at most 128 models.')
    clean, seen = [], set()
    for row in rows:
        name = row.get('name') if isinstance(row, dict) else None
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', name)
                or name in {'ensemble', 'route'} or name in seen):
            raise ForecastAdapterError('Ephemeris catalog contains invalid, duplicate or reserved model names.')
        seen.add(name)
        clean.append({'name': name, **{key: row.get(key) is True for key in ('enabled', 'healthy', 'covariates')}})
    return clean


def load_catalog():
    path = catalog_path()
    if not path.exists() and not path.is_symlink():
        return {'status': 'not_discovered', 'models': []}
    try:
        _check_private(path.parent, directory=True)
        _check_private(path)
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        with os.fdopen(fd) as handle:
            data = json.loads(handle.read(65537))
        rows = clean_catalog(data['models'])
        fetched = data['retrieved_at']
        if not isinstance(fetched, str) or len(fetched) > 64:
            raise ValueError('invalid catalog date')
        return {'status': 'cached', 'models': rows, 'retrieved_at': fetched,
                'availability': 'last_discovery_only; forecast-time failures are not substituted'}
    except (OSError, ValueError, KeyError, TypeError, ForecastAdapterError):
        return {'status': 'unavailable', 'models': [], 'guidance': 'Run gnomon connect ephemeris --refresh-models.'}


def refresh_models():
    if not saved_connection():
        raise ForecastAdapterError('No saved Ephemeris connection. Run gnomon connect ephemeris first.')
    from .ephemeris import EphemerisProvider
    from .http_transport import JSONTransport
    provider = EphemerisProvider(GATEWAY_URL, api_format='gateway',
                                transport=JSONTransport(GATEWAY_URL, token_file=credential_path()))
    rows = clean_catalog(provider.models())
    data = {'models': rows, 'retrieved_at': datetime.now(timezone.utc).isoformat()}
    path = catalog_path()
    if path.exists() or path.is_symlink():
        _check_private(path)
    fd, name = tempfile.mkstemp(prefix='.ephemeris-models-', dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(data, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return {'status': 'refreshed', 'retrieved_at': data['retrieved_at'],
            'providers': ['ephemeris/' + r['name'] for r in rows if r['enabled'] and r['healthy']],
            'forecast_calls': 0, 'guidance': 'Restart MCP to load the refreshed individual models.'}


def saved_info():
    saved = saved_connection()
    catalog = load_catalog() if saved else {'status': 'not_discovered', 'models': []}
    summary = {key: value for key, value in catalog.items() if key != 'models'}
    summary['providers'] = ['ephemeris/' + r['name'] for r in catalog['models'] if r['enabled'] and r['healthy']]
    return connection_info(saved=saved, catalog=summary)


def register_saved(session):
    if not saved_connection():
        return
    from .ephemeris import EphemerisProvider
    from .http_transport import JSONTransport
    for name, mode in [('ephemeris', 'route'), ('ephemeris/ensemble', 'ensemble')]:
        transport = JSONTransport(GATEWAY_URL, token_file=credential_path())
        session.engine.register(name, EphemerisProvider(GATEWAY_URL, mode=mode,
                                api_format='gateway', transport=transport), lifecycle='pretrained')
        session._ephemeris_providers.add(name)
    catalog = load_catalog()
    session._ephemeris_catalog = {key: value for key, value in catalog.items() if key != 'models'}
    provider = EphemerisProvider(GATEWAY_URL, api_format='gateway',
                                transport=JSONTransport(GATEWAY_URL, token_file=credential_path()))
    session._ephemeris_providers.update(provider.register_catalog(session.engine, catalog['models']))


def connect(args):
    if args.disconnect:
        if saved_connection():
            credential_path().unlink()
        catalog_path().unlink(missing_ok=True)
        return {'status': 'ok', 'connection_status': 'not_configured',
                'guidance': 'Local credential removed. Restart MCP. Revoke the API key on Ephemeris to invalidate other copies.'}
    if args.refresh_models:
        return {'status': 'ok', 'model_catalog': refresh_models()}
    if args.check:
        if not saved_connection():
            raise ForecastAdapterError('No saved Ephemeris connection. Run gnomon connect ephemeris first.')
        from .http_transport import JSONTransport
        response = JSONTransport(GATEWAY_URL, token_file=credential_path()).call('/balance')
        return {'status': 'ok', 'connection_status': 'verified', 'balance_mc': response.get('balance_mc'),
                'forecast_calls': 0, 'guidance': 'Authenticated balance check only; no forecast executed.'}
    if args.status:
        return {'status': 'ok', 'ephemeris': saved_info()}
    if not args.token_stdin and not sys.stdin.isatty():
        return {'status': 'ok', 'ephemeris': saved_info(),
                'guidance': 'A human can sign up at signup_url, then run connect_command in a terminal for hidden key entry. No files changed.'}
    if saved_connection() and not args.replace:
        return {'status': 'ok', 'ephemeris': saved_info(),
                'guidance': 'Already configured. Use --refresh-models for individual models, --check to verify, or --replace to change the key.'}
    if args.token_stdin:
        token = sys.stdin.read(4098).strip()
    else:
        print(f'Optional Ephemeris signup: {SIGNUP_URL}\nLocal models need no account. Paste your API key below; input is hidden.\nPaid forecasts require your authorization. Blank input cancels.', file=sys.stderr)
        try:
            # Never fall back to an echoed prompt if a controlling terminal is unavailable.
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('error', getpass.GetPassWarning)
                token = getpass.getpass('Ephemeris API key: ')
        except (getpass.GetPassWarning, EOFError):
            raise ForecastAdapterError('Hidden key entry requires a terminal. Use --token-stdin only with a secure input source.') from None
        if not token:
            return {'status': 'ok', 'connection_status': 'cancelled', 'credential_saved': False}
    save_token(token, replace=args.replace)
    try:
        catalog = refresh_models()
    except (ForecastAdapterError, OSError):
        catalog = {'status': 'failed', 'providers': [], 'forecast_calls': 0,
                   'guidance': 'Credential saved; model discovery failed. Run gnomon connect ephemeris --refresh-models. Router and ensemble remain configured; authentication is unverified.'}
    return {'status': 'ok', 'credential_saved': True,
            'ephemeris': connection_info(saved=True, catalog=catalog),
            'guidance': 'Saved locally with private permissions. New CLI/default Python sessions load ephemeris and ephemeris/ensemble. Discovered individual models are also loaded. Restart MCP. Only model catalog discovery was attempted; no forecast was made.'}

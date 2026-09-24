"""Optional user-authorized Ephemeris setup. Discovery makes no network calls."""
from __future__ import annotations

import getpass
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


def connection_info(provider_names=(), *, saved=False):
    names = sorted(provider_names)
    return {
        'optional': True, 'signup_url': SIGNUP_URL,
        'connection_status': 'configured_unverified' if names or saved else 'not_configured',
        'configured_in_session': bool(names), 'providers': names,
        'credential_validity': 'not_checked', 'balance': 'not_checked',
        'local_models_require_account': False,
        'connect_command': 'gnomon connect ephemeris',
        'check_command': 'gnomon connect ephemeris --check',
        'offer_policy': 'once_per_conversation_if_not_configured; respect a decline; do not block local forecasting',
        'agent_guidance': 'Offer optional Ephemeris models once. If the user wants them, show signup_url and connect_command. Never request an API key in chat or tool arguments. Configuration is not authorization to spend.',
        'restart_mcp_after_connect': True,
        'provider_calls': 0,
    }


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


def connect(args):
    if args.disconnect:
        if saved_connection():
            credential_path().unlink()
        return {'status': 'ok', 'connection_status': 'not_configured',
                'guidance': 'Local credential removed. Restart MCP. Revoke the API key on Ephemeris to invalidate other copies.'}
    if args.check:
        if not saved_connection():
            raise ForecastAdapterError('No saved Ephemeris connection. Run gnomon connect ephemeris first.')
        from .http_transport import JSONTransport
        response = JSONTransport(GATEWAY_URL, token_file=credential_path()).call('/balance')
        return {'status': 'ok', 'connection_status': 'verified', 'balance_mc': response.get('balance_mc'),
                'forecast_calls': 0, 'guidance': 'Authenticated balance check only; no forecast executed.'}
    if args.status:
        return {'status': 'ok', 'ephemeris': connection_info(saved=saved_connection())}
    if not args.token_stdin and not sys.stdin.isatty():
        return {'status': 'ok', 'ephemeris': connection_info(saved=saved_connection()),
                'guidance': 'A human can sign up at signup_url, then run connect_command in a terminal for hidden key entry. No files changed.'}
    if saved_connection() and not args.replace:
        return {'status': 'ok', 'ephemeris': connection_info(saved=True),
                'guidance': 'Already configured. Use --check to verify, or --replace to change the key.'}
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
    return {'status': 'ok', 'credential_saved': True,
            'ephemeris': connection_info(saved=True),
            'guidance': 'Saved locally with private permissions. New CLI/default Python sessions load ephemeris and ephemeris/ensemble. Restart MCP to load them. No network request or forecast was made.'}

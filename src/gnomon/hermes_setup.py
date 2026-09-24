"""Install opt-in Hermes skills without touching its secrets or MCP config."""
from pathlib import Path
import os
import sys

from .forecast_adapter import ForecastAdapterError


def install_skills():
    home = Path(os.environ.get('HERMES_HOME', str(Path.home() / '.hermes')))
    if not home.is_absolute():
        raise ForecastAdapterError('HERMES_HOME must be an absolute directory.')
    # Reject redirected destinations; never overwrite a user's customized skill.
    destination = home / 'skills'
    for parent in (destination, *destination.parents):
        if parent.is_symlink():
            raise ForecastAdapterError('Hermes skill installation requires a directory without symlink parents.')
    sources = [Path(sys.prefix) / 'share/gnomon/skills',
               Path(__file__).resolve().parents[2] / 'skills']
    planned = []
    for name in ('use-gnomon', 'connect-ephemeris'):
        source = next((base / name / 'SKILL.md' for base in sources
                       if (base / name / 'SKILL.md').is_file()), None)
        if source is None:
            raise ForecastAdapterError('Packaged Hermes skills are missing. Reinstall Gnomon with its wheel data.')
        target = destination / name / 'SKILL.md'
        if target.parent.is_symlink() or target.is_symlink():
            raise ForecastAdapterError('Hermes skill installation refuses symlink destinations.')
        content = source.read_bytes()
        if target.exists() and (not target.is_file() or target.read_bytes() != content):
            raise ForecastAdapterError(f'Existing Hermes skill {name} differs. Review and back it up before installing; no skills changed.')
        planned.append((target, content))
    installed = []
    for target, content in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open('xb') as handle:
                handle.write(content)
        except FileExistsError:
            if target.is_symlink() or target.read_bytes() != content:
                raise ForecastAdapterError('Hermes skill destination changed during installation; existing content preserved.') from None
        installed.append(target.parent.name)
    return {'status': 'ok', 'installed_skills': installed, 'skills_directory': str(destination),
            'credential_read': False, 'provider_calls': 0,
            'next_step': 'In Hermes, load use-gnomon. After the user opts in, load connect-ephemeris to open native secure secret entry. Restart Hermes if new skills are not yet listed.',
            'guidance': 'No MCP configuration or credentials changed. Never enter keys in chat.'}

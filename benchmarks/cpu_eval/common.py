"""Immutable artifacts, source identity and resource inspection."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name('protocol.json')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != raw:
            raise ValueError(f'Refusing to replace different artifact: {path}')
        return
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            out.write(raw)
        # Exclusive link prevents a concurrent writer from replacing an artifact.
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def identity():
    paths = sorted((ROOT/'src/gnomon').rglob('*.py')) + sorted((ROOT/'benchmarks/cpu_eval').glob('*.py'))
    paths += [PROTOCOL, Path(__file__).with_name('requirements.txt')]
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    versions = {}
    for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'statsforecast', 'pyarrow', 'statsmodels', 'coreforecast', 'utilsforecast', 'joblib', 'threadpoolctl'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'sources': digest(sources), 'source_files': sources,
            'python': platform.python_version(), 'dependencies': versions}


def resources():
    out = {'hostname':socket.gethostname(), 'host_cpu_count': os.cpu_count(), 'platform': platform.platform()}
    for name in ('cpu.max', 'memory.max', 'memory.current'):
        path = Path('/sys/fs/cgroup')/name
        out[name] = path.read_text().strip() if path.exists() else None
    return out

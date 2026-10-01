"""Failure boundaries for the operator backup job; no real cloud credentials."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest


@pytest.mark.parametrize('failure', ['snapshot', 'upload', 'readback'])
def test_failed_backup_restarts_service_and_preserves_last_success(tmp_path, monkeypatch, failure):
    script = Path(__file__).resolve().parents[1] / 'deployment/r2_backup.py'
    spec = importlib.util.spec_from_file_location('backup_operator', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    work = tmp_path / 'work'
    work.mkdir()
    prior = work / 'last-success.json'
    prior.write_text('{"previous": "successful backup"}')
    metadata = tmp_path / 'deployment.json'
    metadata.write_text('{}')
    events = []
    def run(args, **kwargs):
        if args[:2] == ['docker', 'stop']:
            events.append('stopped')
        elif args[:2] == ['docker', 'start']:
            events.append('started')
        elif args[:2] == ['docker', 'run']:
            if failure == 'snapshot':
                raise RuntimeError('snapshot failure')
            directory = next(work.glob('backup-*')) / 'evidence'
            directory.mkdir()
            (directory / 'manifest.json').write_text('{}')
        elif args[0] == 'age':
            Path(args[args.index('-o') + 1]).write_bytes(b'fixture ciphertext')
    class S3:
        def upload_file(self, *args, **kwargs):
            assert events[-1] == 'started', 'Service must restart before the network call'
            events.append('upload')
            if failure == 'upload':
                raise RuntimeError('upload failure')
        def download_file(self, bucket, key, filename):
            Path(filename).write_bytes(b'corrupted readback')
    monkeypatch.setitem(sys.modules, 'boto3', types.SimpleNamespace(client=lambda *a, **k: S3()))
    monkeypatch.setitem(sys.modules, 'botocore.config', types.SimpleNamespace(Config=lambda **k: None))
    monkeypatch.setattr(module.subprocess, 'run', run)
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: 'true')
    for name in ['ACCESS_KEY_ID', 'SECRET_ACCESS_KEY', 'ACCOUNT_ID', 'BUCKET']:
        monkeypatch.setenv('GNOMON_R2_' + name, 'test-only')
    monkeypatch.setenv('GNOMON_BACKUP_RECIPIENT', 'fixture')
    monkeypatch.setattr(sys, 'argv', ['backup', '--container', 'test', '--image', 'test', '--data', str(tmp_path / 'data'),
                                    '--work', str(work), '--metadata', str(metadata)])
    with pytest.raises(RuntimeError):
        module.main()
    assert events[:2] == ['stopped', 'started']
    assert prior.read_text() == '{"previous": "successful backup"}'
    assert not list(work.glob('backup-*'))

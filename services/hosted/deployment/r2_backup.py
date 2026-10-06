#!/usr/bin/env python3
"""Stop briefly for a consistent backup, restart, encrypt with age, upload to R2.

Operator-only utility. Requires Docker, age and boto3. Credentials come from the
service environment; only the public age recipient is needed on the server.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from uuid import uuid4


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    import boto3
    from botocore.config import Config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    env = {k: os.environ[k] for k in ('GNOMON_R2_ACCESS_KEY_ID', 'GNOMON_R2_SECRET_ACCESS_KEY',
        'GNOMON_R2_ACCOUNT_ID', 'GNOMON_R2_BUCKET', 'GNOMON_BACKUP_RECIPIENT')}
    args.work.mkdir(parents=True, exist_ok=True, mode=0o700)
    # systemd prevents overlapping runs; flock also protects manual invocation.
    import fcntl
    with (args.work / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory(prefix='backup-', dir=args.work) as directory:
            work = Path(directory)
            running = subprocess.check_output(['docker', 'inspect', '--format', '{{.State.Running}}', args.container], text=True).strip()
            if running != 'true':
                raise RuntimeError('Service must be running before scheduled backup.')
            try:
                subprocess.run(['docker', 'stop', '--time', '320', args.container], check=True, stdout=subprocess.DEVNULL)
                subprocess.run(['docker', 'run', '--rm', '--user', '0', '--network', 'none',
                    '--volume', str(args.data.resolve()) + ':/data', '--volume', str(work.resolve()) + ':/backup',
                    args.image, 'backup', '--destination', '/backup/evidence'], check=True, stdout=subprocess.DEVNULL)
            finally:
                subprocess.run(['docker', 'start', args.container], check=True, stdout=subprocess.DEVNULL)
            shutil.copyfile(args.metadata, work / 'deployment.json')
            with tarfile.open(work / 'backup.tar', 'w') as archive:
                archive.add(work / 'evidence', arcname='evidence')
                archive.add(work / 'deployment.json', arcname='deployment.json')
            encrypted = work / 'backup.tar.age'
            subprocess.run(['age', '-r', env['GNOMON_BACKUP_RECIPIENT'], '-o', str(encrypted), str(work / 'backup.tar')], check=True)
            sha = digest(encrypted)
            key = 'staging/' + datetime.now(timezone.utc).strftime('%Y/%m/%d/%H%M%S-') + uuid4().hex + '.tar.age'
            s3 = boto3.client('s3', endpoint_url='https://' + env['GNOMON_R2_ACCOUNT_ID'] + '.r2.cloudflarestorage.com',
                aws_access_key_id=env['GNOMON_R2_ACCESS_KEY_ID'], aws_secret_access_key=env['GNOMON_R2_SECRET_ACCESS_KEY'],
                region_name='auto', config=Config(request_checksum_calculation='when_required', response_checksum_validation='when_required',
                connect_timeout=15, read_timeout=60, retries={'max_attempts': 3}))
            s3.upload_file(str(encrypted), env['GNOMON_R2_BUCKET'], key, ExtraArgs={'Metadata': {'sha256': sha}, 'ContentType': 'application/octet-stream'})
            # Read the actual object back, not just its metadata or multipart ETag.
            s3.download_file(env['GNOMON_R2_BUCKET'], key, str(work / 'readback.age'))
            if digest(work / 'readback.age') != sha:
                raise RuntimeError('Uploaded backup readback digest mismatch')
            result = {'status': 'ok', 'bucket': env['GNOMON_R2_BUCKET'], 'key': key, 'sha256': sha,
                      'bytes': encrypted.stat().st_size, 'completed_at': datetime.now(timezone.utc).isoformat(),
                      'encryption': 'age', 'readback_verified': True}
            temporary = args.work / 'last-success.tmp'
            temporary.write_text(json.dumps(result, indent=2) + '\n')
            temporary.replace(args.work / 'last-success.json')
            print(json.dumps(result))


if __name__ == '__main__':
    main()

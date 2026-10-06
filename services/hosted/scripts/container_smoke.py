"""Build-independent smoke of an explicit local image, with disposable Docker state."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import httpx
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from hosted_probe_support import call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    name = 'gnomon-hosted-smoke-' + uuid4().hex[:12]
    def docker(*arguments):
        return subprocess.check_output(['docker', *arguments], text=True)
    volume = name + '-data'
    docker('volume', 'create', volume)
    backup_volume, restored_volume = name + '-backup', name + '-restored'
    docker('volume', 'create', backup_volume)
    docker('volume', 'create', restored_volume)
    restored_name = name + '-restore'
    started = restored_started = False
    try:
        def admin(*arguments):
            return json.loads(docker('run', '--rm', '-v', volume + ':/data', args.image, *arguments))
        admin('init')
        project = admin('project-create', '--name', 'container-smoke')['project_id']
        token = admin('token-create', '--project', project, '--principal', 'probe',
                      '--permissions', 'forecast.create,evidence.read')['token']
        docker('run', '-d', '--name', name, '--read-only', '--tmpfs', '/tmp', '--cap-drop', 'ALL',
               '--security-opt', 'no-new-privileges:true', '-v', volume + ':/data',
               '-p', '127.0.0.1::8765', args.image)
        started = True
        port = docker('port', name, '8765/tcp').strip().rsplit(':', 1)[1]
        url = 'http://127.0.0.1:' + port
        def ready():
            for _ in range(100):
                try:
                    if httpx.get(url + '/ready', timeout=1).status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(.1)
            raise RuntimeError('Container did not become ready: ' + docker('logs', name)[-2000:])
        ready()
        at = datetime.now(timezone.utc)
        req = {'series_id': 'container-test', 'history': [1, 2], 'horizon': 1,
               'timestamps': [(at-timedelta(days=d)).isoformat() for d in (2, 1)],
               'future_timestamps': [(at+timedelta(days=1)).isoformat()]}
        response = call(url, token, 'gnomon_forecast', {'provider': 'last_value', 'request': req, 'idempotency_key': 'once'})
        assert response['status'] == 'ok', response
        ref = response['result']['reference']
        docker('restart', name)
        port = docker('port', name, '8765/tcp').strip().rsplit(':', 1)[1]
        url = 'http://127.0.0.1:' + port
        ready()
        result = call(url, token, 'gnomon_hosted', {'action': 'resolve', 'reference': ref})
        assert result['result']['execution_id'] == response['result']['execution_id'], result
        # Test clean-volume recovery, including revocation of source credentials.
        docker('stop', name)
        docker('run', '--rm', '--user', '0', '-v', volume + ':/data',
               '-v', backup_volume + ':/backup', args.image,
               'backup', '--destination', '/backup/evidence')
        docker('run', '--rm', '--user', '0', '--entrypoint', 'sh',
               '-v', restored_volume + ':/data', '-v', backup_volume + ':/backup:ro',
               args.image, '-c', 'gnomon-hosted --root /data restore --source /backup/evidence && chown -R 10001:10001 /data')
        fresh = json.loads(docker('run', '--rm', '-v', restored_volume + ':/data', args.image,
            'token-create', '--project', project, '--principal', 'restore-reader',
            '--permissions', 'evidence.read'))['token']
        docker('run', '-d', '--name', restored_name, '--read-only', '--tmpfs', '/tmp',
               '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges:true',
               '-v', restored_volume + ':/data', '-p', '127.0.0.1::8765', args.image)
        restored_started = True
        port = docker('port', restored_name, '8765/tcp').strip().rsplit(':', 1)[1]
        url = 'http://127.0.0.1:' + port
        ready()
        rejected = httpx.post(url + '/mcp', headers={'Authorization': 'Bearer ' + token}, json={})
        assert rejected.status_code == 401
        result = call(url, fresh, 'gnomon_hosted', {'action': 'resolve', 'reference': ref})
        assert result['result']['execution_id'] == response['result']['execution_id'], result
        print(json.dumps({'status': 'ok', 'image': args.image, 'restart': True,
                          'read_only_root': True, 'persistent_volume': True, 'forecast': [2.0],
                          'clean_volume_restore': True, 'old_tokens_revoked': True}))
    finally:
        if restored_started:
            docker('rm', '-f', restored_name)
        docker('volume', 'rm', restored_volume)
        docker('volume', 'rm', backup_volume)
        if started:
            docker('rm', '-f', name)
        docker('volume', 'rm', volume)


if __name__ == '__main__':
    main()

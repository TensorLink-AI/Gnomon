"""Local operator administration; no remote endpoint can execute these commands."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3

from .storage import Store, ServiceError, encode
from gnomon import __version__


def backup(root, destination):
    store = Store(root)
    destination = Path(destination).resolve()
    if destination == store.root or store.root in destination.parents:
        raise ServiceError('INVALID_DESTINATION', 'Backup must be outside the service directory.')
    with open(store.root / '.server.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ServiceError('SERVER_RUNNING', 'Stop the service before taking a coordinated backup.') from None
        destination.mkdir(parents=True, exist_ok=False, mode=0o700)
        with store.connect() as conn:
            projects = [r[0] for r in conn.execute('SELECT id FROM projects')]
        files = ['control.db'] + [p + '/ledger.db' for p in projects]
        entries = {}
        for relative in files:
            source, target = store.root / relative, destination / relative
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with sqlite3.connect(source) as src, sqlite3.connect(target) as dst:
                src.backup(dst)
                if dst.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                    raise ServiceError('INTEGRITY', 'Backup database failed integrity check.')
                dst.execute('PRAGMA journal_mode=DELETE')
            os.chmod(target, 0o600)
            entries[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest = {'schema_version': 1, 'service_id': store.service_id, 'files': entries,
                    'core_version': __version__, 'hosted_version': '0.1.0',
                    'credentials': 'Token verifiers included; external secrets are not included.'}
        (destination / 'manifest.json').write_text(encode(manifest))
        return manifest


def restore(source, root):
    source, root = Path(source).resolve(), Path(root).resolve()
    if root.exists():
        raise ServiceError('DESTINATION_EXISTS', 'Restore requires a new directory.')
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest.get('schema_version') != 1 or 'control.db' not in manifest.get('files', {}):
        raise ServiceError('INVALID_BACKUP', 'Unsupported backup manifest.')
    for relative, digest in manifest['files'].items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or source not in (source / path).resolve().parents:
            raise ServiceError('INVALID_BACKUP', 'Invalid backup path.')
        candidate = source / path
        if candidate.is_symlink() or hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            raise ServiceError('INTEGRITY', 'Backup content does not match manifest.')
    root.mkdir(parents=True, mode=0o700)
    for relative in manifest['files']:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(source / relative, target)
        os.chmod(target, 0o600)
    store = Store(root)
    if store.service_id != manifest['service_id']:
        raise ServiceError('INTEGRITY', 'Service identity does not match manifest.')
    with store.connect() as conn:
        conn.execute('UPDATE tokens SET revoked=1')
        projects = [r[0] for r in conn.execute('SELECT id FROM projects WHERE active=1')]
    for project in projects:
        with store.ledger(project).transaction() as conn:
            # Delivery may have succeeded after the backup was taken.
            conn.execute("UPDATE hosted_exports SET state='uncertain' WHERE state IN ('pending','sending')")
    store.recover()
    return {'service_id': store.service_id, 'tokens_revoked': True, 'project_ids': projects}


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Private persistent service directory')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('init')
    create = commands.add_parser('project-create')
    create.add_argument('--name', required=True)
    create.add_argument('--providers', help='Operator-owned provider TOML')
    configure = commands.add_parser('provider-configure')
    configure.add_argument('--project', required=True)
    configure.add_argument('--providers', required=True)
    issue = commands.add_parser('token-create')
    issue.add_argument('--project', required=True)
    issue.add_argument('--principal', required=True)
    issue.add_argument('--permissions', required=True, help='Comma-separated permission names')
    revoke = commands.add_parser('token-revoke')
    revoke.add_argument('--token-id', required=True)
    ditto = commands.add_parser('ditto-configure')
    ditto.add_argument('--project', required=True)
    ditto.add_argument('--url', default='https://api.heyditto.ai/mcp')
    ditto.add_argument('--token-env', required=True, help='Environment name for the dedicated-graph credential')
    ditto.add_argument('--graph', required=True, help='Dedicated graph alias (checked on every connection)')
    stop = commands.add_parser('project-disable')
    stop.add_argument('--project', required=True)
    save = commands.add_parser('backup')
    save.add_argument('--destination', required=True)
    load = commands.add_parser('restore')
    load.add_argument('--source', required=True)
    serve = commands.add_parser('serve')
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', default=8765, type=int)
    serve.add_argument('--allowed-host', action='append', default=['localhost', '127.0.0.1', '::1'])
    serve.add_argument('--allowed-origin', action='append', default=[])
    serve.add_argument('--forecast-timeout', type=float, default=60)
    args = parser.parse_args()
    try:
        if args.command == 'init':
            result = {'service_id': Store.initialize(args.root).service_id}
        elif args.command == 'restore':
            result = restore(args.source, args.root)
        else:
            store = Store(args.root)
            if args.command == 'project-create':
                result = store.create_project(args.name, args.providers)
            elif args.command == 'provider-configure':
                from .worker import validate_config
                store.project(args.project)
                path = str(Path(args.providers).resolve())
                validate_config(path)
                with store.connect() as conn:
                    conn.execute('UPDATE projects SET providers=? WHERE id=?', (path, args.project))
                result = {'configured': args.project}
            elif args.command == 'token-create':
                result = store.issue_token(args.project, args.principal, args.permissions.split(','))
            elif args.command == 'token-revoke':
                store.revoke(args.token_id)
                result = {'revoked': args.token_id}
            elif args.command == 'ditto-configure':
                store.configure_ditto(args.project, args.url, args.token_env, args.graph)
                result = {'configured': args.project}
            elif args.command == 'project-disable':
                store.disable_project(args.project)
                result = {'disabled': args.project, 'in_flight_external_calls_may_complete': True}
            elif args.command == 'backup':
                result = backup(args.root, args.destination)
            elif args.command == 'serve':
                if not 0 < args.forecast_timeout <= 300:
                    raise ServiceError('INVALID_LIMIT', 'Forecast timeout must be between 0 and 300 seconds.')
                import uvicorn
                from .app import create_app
                uvicorn.run(create_app(args.root, allowed_hosts=args.allowed_host, allowed_origins=args.allowed_origin,
                                       forecast_timeout=args.forecast_timeout), host=args.host, port=args.port,
                            proxy_headers=False, access_log=False, timeout_graceful_shutdown=310)
                return
        print(json.dumps(result, indent=2))
    except (ServiceError, FileExistsError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()

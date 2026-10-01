"""Server-owned project mapping and credential verification. No caller paths."""
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import sqlite3
from uuid import uuid4

from gnomon import TemporalLedger

PERMISSIONS = {'evidence.read', 'forecast.create', 'decision.create', 'actual.create',
               'memory.export'}


def now():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class ServiceError(Exception):
    def __init__(self, code, message, status=400):
        self.code, self.message, self.status = code, message, status
        super().__init__(message)


@dataclass(frozen=True)
class Identity:
    project: str
    principal: str
    permissions: frozenset
    token_hash: str


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        if not (self.root / 'control.db').is_file():
            raise ServiceError('NOT_INITIALIZED', 'Initialize this service directory first.')
        with self.connect() as conn:
            row = conn.execute('SELECT version, service_id FROM service').fetchone()
            if not row or row['version'] != 1:
                raise ServiceError('SCHEMA_VERSION', 'Unsupported service schema; restore a compatible release.')
            self.service_id = row['service_id']

    @classmethod
    def initialize(cls, root):
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = root / 'control.db'
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        with sqlite3.connect(path) as conn:
            conn.executescript('''
                CREATE TABLE service(version INTEGER NOT NULL, service_id TEXT NOT NULL);
                CREATE TABLE projects(id TEXT PRIMARY KEY, name TEXT NOT NULL, ledger_id TEXT NOT NULL,
                    providers TEXT, ditto_url TEXT, ditto_token_env TEXT, ditto_graph TEXT, ditto_connection TEXT, active INTEGER NOT NULL DEFAULT 1);
                CREATE TABLE memberships(project TEXT NOT NULL REFERENCES projects(id), principal TEXT NOT NULL,
                    permissions TEXT NOT NULL, PRIMARY KEY(project,principal));
                CREATE TABLE tokens(id TEXT PRIMARY KEY, digest TEXT UNIQUE NOT NULL, project TEXT NOT NULL,
                    principal TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(project,principal) REFERENCES memberships(project,principal));
            ''')
            conn.execute('INSERT INTO service VALUES(1,?)', ('service-' + uuid4().hex,))
        return cls(root)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect((self.root / 'control.db').as_uri() + '?mode=rw', uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def create_project(self, name, providers=None):
        project, ledger_id = 'project-' + uuid4().hex, 'ledger-' + uuid4().hex
        if providers:
            from .worker import validate_config
            providers = str(Path(providers).resolve())
            validate_config(providers)
        directory = self.root / project
        directory.mkdir(mode=0o700)
        ledger = TemporalLedger(directory / 'ledger.db')
        with ledger.transaction() as conn:
            conn.execute('''CREATE TABLE hosted_requests(
                id TEXT PRIMARY KEY, principal TEXT NOT NULL, key TEXT NOT NULL, digest TEXT NOT NULL,
                operation TEXT NOT NULL, state TEXT NOT NULL, result TEXT, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, UNIQUE(principal,key))''')
            conn.execute('''CREATE TABLE hosted_records(
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL,
                sha256 TEXT NOT NULL, recorded_at TEXT NOT NULL, principal TEXT NOT NULL)''')
            conn.execute('''CREATE TABLE hosted_exports(
                id TEXT PRIMARY KEY, lesson_id TEXT NOT NULL, principal TEXT NOT NULL,
                destination TEXT NOT NULL, payload TEXT NOT NULL, digest TEXT NOT NULL,
                state TEXT NOT NULL, memory_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                UNIQUE(lesson_id,destination))''')
            conn.execute('''CREATE TABLE hosted_audit(
                id INTEGER PRIMARY KEY, principal TEXT NOT NULL, operation TEXT NOT NULL,
                reference TEXT NOT NULL, state TEXT NOT NULL, recorded_at TEXT NOT NULL)''')
            for action in ('UPDATE', 'DELETE'):
                for table in ('hosted_records', 'hosted_audit'):
                    conn.execute(f"CREATE TRIGGER immutable_{table}_{action} BEFORE {action} ON {table} "
                                 "BEGIN SELECT RAISE(ABORT, 'hosted evidence is append-only'); END")
        with self.connect() as conn:
            conn.execute('INSERT INTO projects(id,name,ledger_id,providers) VALUES(?,?,?,?)',
                         (project, name, ledger_id, providers))
        return {'project_id': project, 'ledger_id': ledger_id}

    def project(self, project):
        with self.connect() as conn:
            row = conn.execute('SELECT * FROM projects WHERE id=? AND active=1', (project,)).fetchone()
        if row is None:
            raise ServiceError('UNAVAILABLE', 'Project or resource unavailable.', 404)
        return dict(row)

    def ledger(self, project):
        row = self.project(project)
        return TemporalLedger(self.root / row['id'] / 'ledger.db', create=False)

    def issue_token(self, project, principal, permissions):
        self.project(project)
        if not principal or not permissions or not set(permissions) <= PERMISSIONS:
            raise ServiceError('INVALID_PERMISSIONS', 'Supply a principal and supported permissions.')
        token, token_id = 'ghs_' + secrets.token_urlsafe(32), uuid4().hex
        with self.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            prior = conn.execute('SELECT permissions FROM memberships WHERE project=? AND principal=?',
                                 (project, principal)).fetchone()
            # Issuing another credential must not silently change existing credentials' authority.
            if prior and set(json.loads(prior[0])) != set(permissions):
                raise ServiceError('MEMBERSHIP_CONFLICT', 'Existing principal permissions differ; use a new principal.')
            conn.execute('INSERT OR IGNORE INTO memberships VALUES(?,?,?)',
                         (project, principal, encode(sorted(set(permissions)))))
            conn.execute('INSERT INTO tokens(id,digest,project,principal) VALUES(?,?,?,?)',
                         (token_id, hashlib.sha256(token.encode()).hexdigest(), project, principal))
        return {'token_id': token_id, 'token': token}

    def revoke(self, token_id):
        with self.connect() as conn:
            if conn.execute('UPDATE tokens SET revoked=1 WHERE id=?', (token_id,)).rowcount != 1:
                raise ServiceError('UNAVAILABLE', 'Credential unavailable.', 404)

    def authenticate(self, token):
        return self.authenticate_hash(hashlib.sha256(token.encode()).hexdigest())

    def authenticate_hash(self, digest):
        with self.connect() as conn:
            row = conn.execute('''SELECT t.project,t.principal,m.permissions FROM tokens t
                JOIN memberships m ON m.project=t.project AND m.principal=t.principal
                JOIN projects p ON p.id=t.project
                WHERE t.digest=? AND t.revoked=0 AND p.active=1''', (digest,)).fetchone()
        if row is None:
            raise ServiceError('UNAUTHORIZED', 'Invalid or revoked credential.', 401)
        return Identity(row['project'], row['principal'], frozenset(json.loads(row['permissions'])), digest)

    def authorize(self, identity, permission):
        current = self.authenticate_hash(identity.token_hash)
        if permission not in current.permissions:
            raise ServiceError('FORBIDDEN', 'Permission denied.', 403)
        return current

    def configure_ditto(self, project, url, token_env, graph):
        from urllib.parse import urlsplit
        import re
        self.project(project)
        parsed = urlsplit(url)
        if not parsed.hostname or parsed.scheme != 'https' or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ServiceError('INVALID_DESTINATION', 'Use an HTTPS MCP URL without credentials or query parameters.')
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', token_env):
            raise ServiceError('INVALID_SECRET_REFERENCE', 'Supply an environment variable name.')
        if not isinstance(graph, str) or not 1 <= len(graph) <= 200:
            raise ServiceError('INVALID_GRAPH', 'Supply the dedicated graph alias selected by the credential.')
        prior = self.project(project)
        same = (url, token_env, graph) == (prior['ditto_url'], prior['ditto_token_env'], prior['ditto_graph'])
        connection_id = prior['ditto_connection'] if same else 'connection-' + uuid4().hex
        with self.connect() as conn:
            conn.execute('UPDATE projects SET ditto_url=?,ditto_token_env=?,ditto_graph=?,ditto_connection=? WHERE id=?',
                         (url, token_env, graph, connection_id, project))

    def disable_project(self, project):
        # The control tombstone is authoritative even if the following journal
        # transaction is interrupted. Every future delivery rechecks it.
        ledger = self.ledger(project)
        with self.connect() as conn:
            conn.execute('UPDATE projects SET active=0 WHERE id=?', (project,))
        with ledger.transaction() as conn:
            rows = conn.execute("SELECT id,state FROM hosted_exports WHERE state IN ('pending','sending')").fetchall()
            for row in rows:
                state = 'uncertain' if row['state'] == 'sending' else 'cancelled'
                conn.execute('UPDATE hosted_exports SET state=?,updated_at=? WHERE id=?', (state, now(), row['id']))
                conn.execute('INSERT INTO hosted_audit(principal,operation,reference,state,recorded_at) VALUES(?,?,?,?,?)',
                             ('operator', 'project.disable', row['id'], state, now()))

    def recover(self):
        with self.connect() as conn:
            projects = [r[0] for r in conn.execute('SELECT id FROM projects WHERE active=1')]
        for project in projects:
            with self.ledger(project).transaction() as conn:
                conn.execute("UPDATE hosted_requests SET state='outcome_unknown',updated_at=? WHERE state IN ('accepted','running')", (now(),))
                rows = conn.execute("SELECT id FROM hosted_exports WHERE state='sending'").fetchall()
                for row in rows:
                    conn.execute("UPDATE hosted_exports SET state='uncertain',updated_at=? WHERE id=?", (now(), row['id']))
                    conn.execute('INSERT INTO hosted_audit(principal,operation,reference,state,recorded_at) VALUES(?,?,?,?,?)',
                                 ('system', 'ditto.recover', row['id'], 'uncertain', now()))

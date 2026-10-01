"""Project-scoped orchestration over the canonical Gnomon numerical runtime."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from uuid import uuid4

from gnomon import GnomonSession
from gnomon.contracts import GnomonError
from gnomon.forecast_adapter import ForecastRequest
from .storage import ServiceError, encode, now
from . import worker, submissions

LEDGER_PERMISSIONS = {
    'execution': 'evidence.read', 'decision': 'evidence.read', 'review_decision': 'evidence.read',
    'export_lesson': 'evidence.read', 'actuals_as_of': 'evidence.read', 'search': 'evidence.read',
    'compare': 'evidence.read', 'compare_history': 'evidence.read', 'pending': 'evidence.read',
    'record_decision_summary': 'decision.create', 'record_lesson': 'decision.create',
    'evaluate': 'decision.create', 'append_actual': 'actual.create',
}
MUTATIONS = {'record_decision_summary', 'record_lesson', 'evaluate', 'append_actual'}


def strict(arguments, allowed, required=()):
    if not isinstance(arguments, dict) or set(arguments) - set(allowed) or not set(required) <= set(arguments):
        raise ServiceError('INVALID_ARGUMENTS', 'Missing or unsupported arguments.')


def instant(value):
    try:
        t = datetime.fromisoformat(value)
        if t.tzinfo is None:
            raise ValueError()
        return t.astimezone(timezone.utc)
    except (TypeError, ValueError):
        raise ServiceError('INVALID_ARGUMENTS', 'Timestamps require an explicit timezone.') from None


class Runtime:
    def __init__(self, store, forecast_timeout=60):
        self.store, self.forecast_timeout = store, forecast_timeout

    def reference(self, project, kind, identity):
        return {'schema_version': '1', 'service_id': self.store.service_id, 'project_id': project,
                'ledger_id': self.store.project(project)['ledger_id'], 'resource_type': kind, 'resource_id': identity}

    @staticmethod
    def record(conn, identity, kind, payload):
        raw = encode(payload)
        digest = hashlib.sha256(raw.encode()).hexdigest()
        rid = kind + '-' + digest
        conn.execute('INSERT OR IGNORE INTO hosted_records VALUES(?,?,?,?,?,?)',
                     (rid, kind, raw, digest, now(), identity.principal))
        return rid

    def call(self, identity, name, arguments):
        # Auth is rechecked here, not just during the MCP handshake.
        self.store.authenticate_hash(identity.token_hash)
        if name == 'gnomon_forecast':
            strict(arguments, ('provider', 'request', 'dataset_id', 'idempotency_key'), ('provider', 'idempotency_key'))
            self.store.authorize(identity, 'forecast.create')
            if ('request' in arguments) == ('dataset_id' in arguments):
                raise ServiceError('INVALID_ARGUMENTS', 'Supply exactly one request or dataset_id.')
            request = arguments.get('request')
            if 'dataset_id' in arguments:
                self.store.authorize(identity, 'evidence.read')
                request = self.fetch_record(identity, 'dataset_version', arguments['dataset_id'])['request']
            request = asdict(ForecastRequest.from_dict(request))
            if not request['series_id'] or not request['timestamps'] or not request['future_timestamps']:
                raise ServiceError('INVALID_ARGUMENTS', 'Forecasts require a named series and history/target timestamps.')
            if len(request['history']) > 10000 or request['horizon'] > 1000:
                raise ServiceError('LIMIT', 'Maximum history is 10000 and horizon is 1000.')
            for timestamp in request['timestamps']:
                if instant(timestamp) > datetime.now(timezone.utc):
                    raise ServiceError('INVALID_ARGUMENTS', 'History cannot contain future observations.')
            args = {'provider': arguments['provider'], 'request': request}
            return self.mutate(identity, 'forecast', args, arguments['idempotency_key'], 'forecast.create')
        if name == 'gnomon_ledger':
            operation = arguments.get('operation')
            if operation not in LEDGER_PERMISSIONS:
                raise ServiceError('UNAVAILABLE', 'Operation is not exposed by this service.', 404)
            permission = LEDGER_PERMISSIONS[operation]
            self.store.authorize(identity, permission)
            args = {k: v for k, v in arguments.items() if k != 'idempotency_key'}
            if operation == 'append_actual':
                for row in args.get('actuals', [args]):
                    for field in ('valid_time', 'source_available_at'):
                        if instant(row.get(field)) > datetime.now(timezone.utc):
                            raise ServiceError('INVALID_ARGUMENTS', 'Realized actuals cannot have future times.')
            if operation in {'review_decision', 'record_lesson', 'export_lesson', 'compare', 'compare_history', 'evaluate'}:
                required = ('recorded_as_of',) if operation == 'export_lesson' else ('source_as_of', 'recorded_as_of')
                for field in required:
                    if instant(args.get(field)) > datetime.now(timezone.utc):
                        raise ServiceError('INVALID_ARGUMENTS', 'Evidence cutoffs cannot be in the future.')
            if operation in MUTATIONS:
                return self.mutate(identity, 'ledger', args, arguments.get('idempotency_key'), permission)
            if 'idempotency_key' in arguments:
                raise ServiceError('INVALID_ARGUMENTS', 'Read operations do not take idempotency_key.')
            with self.session(identity) as session:
                return session.call(name, args, compact=False)
        if name == 'gnomon_memory':
            self.store.authorize(identity, 'evidence.read')
            with self.session(identity) as session:
                return session.call(name, arguments, compact=False)
        if name == 'gnomon_hosted':
            return self.hosted(identity, arguments)
        raise ServiceError('UNAVAILABLE', 'Unknown tool.', 404)

    def session(self, identity, ledger=None):
        return GnomonSession(ledger=ledger or self.store.ledger(identity.project), allow_outcome_writes=True)

    def mutate(self, identity, operation, args, key, permission):
        if not isinstance(key, str) or not 1 <= len(key) <= 128:
            raise ServiceError('INVALID_ARGUMENTS', 'Mutations require a 1-128 character idempotency_key.')
        digest = hashlib.sha256(encode({'operation': operation, 'arguments': args}).encode()).hexdigest()
        ledger = self.store.ledger(identity.project)
        rid = 'request-' + uuid4().hex
        with ledger.transaction() as conn:
            prior = conn.execute('SELECT * FROM hosted_requests WHERE principal=? AND key=?', (identity.principal, key)).fetchone()
            if prior:
                if prior['digest'] != digest:
                    raise ServiceError('IDEMPOTENCY_CONFLICT', 'Key was already used with different arguments.', 409)
                return self.receipt(prior)
            conn.execute('INSERT INTO hosted_requests VALUES(?,?,?,?,?,?,?,?,?)',
                         (rid, identity.principal, key, digest, operation, 'running', None, now(), now()))
        try:
            execution = None
            if operation == 'forecast':
                execution = worker.forecast(self.store.project(identity.project)['providers'],
                                            args['provider'], args['request'], self.forecast_timeout)
            self.store.authorize(identity, permission)
            with ledger.transaction() as conn:
                if operation == 'forecast.submit':
                    execution = submissions.execution(args, identity.principal)
                if operation in ('forecast', 'forecast.submit'):
                    ledger.record_execution(execution)
                    result = {'execution_id': execution.execution_id, 'provider': execution.provider,
                              'result': asdict(execution.result), 'completion': execution.completion(),
                              'evidence': execution.evidence,
                              'recorded_at': ledger.execution(execution.execution_id)['recorded_at'],
                              'reference': self.reference(identity.project, 'execution', execution.execution_id)}
                elif operation == 'ledger':
                    before_core = conn.total_changes
                    with self.session(identity, ledger) as session:
                        result = session.call('gnomon_ledger', args, compact=False)
                    # Core diagnostics are collected before our enclosing commit.
                    # Publish the core row changes atomically with this receipt,
                    # excluding the hosted bookkeeping written below.
                    diagnostics = result.get('execution_diagnostics')
                    if isinstance(diagnostics, dict):
                        diagnostics['ledger_writes'] = conn.total_changes - before_core
                        diagnostics['scope'] = ('Core changes committed with this hosted request receipt; '
                                                'excludes hosted receipt and audit rows.')
                elif operation == 'dataset':
                    record_id = self.record(conn, identity, 'dataset_version', args)
                    result = {'dataset_id': record_id, 'reference': self.reference(identity.project, 'dataset_version', record_id)}
                elif operation == 'review':
                    packet = ledger.review_decision(**args)
                    record_id = self.record(conn, identity, 'review', packet)
                    result = {'review_id': record_id, 'review': packet,
                              'reference': self.reference(identity.project, 'review', record_id)}
                elif operation == 'snapshot':
                    packet = ledger.evidence_snapshot(**args)
                    record_id = self.record(conn, identity, 'evidence_snapshot', packet)
                    result = {'snapshot_id': record_id, 'snapshot': packet,
                              'reference': self.reference(identity.project, 'evidence_snapshot', record_id)}
                elif operation == 'analysis':
                    snapshot = self.fetch_record(identity, 'evidence_snapshot', args['snapshot_id'], ledger=ledger)
                    payload = {**args, 'kind': 'client_analysis/1', 'submitted_by': identity.principal,
                        'submitted_at': now(),
                        'numerically_verified': False, 'narrative_verified': False,
                        'forecast': {'execution_id': snapshot['execution']['execution_id'],
                            'provider': snapshot['execution']['provider'], 'revision': snapshot['execution']['revision'],
                            'evidence': snapshot['execution']['evidence'],
                            'provider_identity': snapshot['execution'].get('provider_identity'),
                            **{k: snapshot['execution']['request'].get(k) for k in ('series_id', 'unit', 'horizon', 'future_timestamps')}},
                        'snapshot_reference': self.reference(identity.project, 'evidence_snapshot', args['snapshot_id']),
                        'source_as_of': snapshot['source_as_of'], 'recorded_as_of': snapshot['recorded_as_of']}
                    record_id = self.record(conn, identity, 'client_analysis', payload)
                    result = {'analysis_id': record_id, 'analysis': payload,
                              'reference': self.reference(identity.project, 'client_analysis', record_id)}
                    if args.get('export_to_ditto', False):
                        self.store.authorize(identity, 'memory.export')
                        result['export'] = self.enqueue_export(identity, ledger, conn,
                            {'analysis_id': record_id, 'recorded_as_of': now()})
                elif operation == 'export':
                    result = self.enqueue_export(identity, ledger, conn, args)
                else:
                    raise ServiceError('INVALID_OPERATION', 'Unsupported mutation.')
                conn.execute("UPDATE hosted_requests SET state='completed',result=?,updated_at=? WHERE id=?",
                             (encode(result), now(), rid))
                conn.execute('INSERT INTO hosted_audit(principal,operation,reference,state,recorded_at) VALUES(?,?,?,?,?)',
                             (identity.principal, operation, rid, 'completed', now()))
            return {'request_id': rid, 'state': 'completed', 'result': result}
        except BaseException as error:
            state = 'outcome_unknown' if operation == 'forecast' or not isinstance(error, (ServiceError, GnomonError, ValueError)) else 'failed'
            with ledger.transaction() as conn:
                conn.execute('UPDATE hosted_requests SET state=?,updated_at=? WHERE id=? AND state=\'running\'', (state, now(), rid))
            raise

    @staticmethod
    def receipt(row):
        return {'request_id': row['id'], 'state': row['state'],
                'result': json.loads(row['result']) if row['result'] else None}

    def fetch_record(self, identity, kind, record_id, *, ledger=None):
        with (ledger or self.store.ledger(identity.project)).transaction() as conn:
            row = conn.execute('SELECT payload,sha256 FROM hosted_records WHERE id=? AND kind=?', (record_id, kind)).fetchone()
        if not row or hashlib.sha256(row['payload'].encode()).hexdigest() != row['sha256']:
            raise ServiceError('UNAVAILABLE', 'Project or resource unavailable.', 404)
        return json.loads(row['payload'])

    def hosted(self, identity, args):
        action = args.get('action')
        if action == 'info':
            strict(args, ('action',))
            self.store.authorize(identity, 'evidence.read')
            return {'service_id': self.store.service_id, 'project_id': identity.project,
                    'ledger_id': self.store.project(identity.project)['ledger_id'], 'server_time': now(),
                    'permissions': sorted(identity.permissions), 'scope': 'single-server project ledger'}
        if action == 'forecast.submit':
            strict(args, ('action', 'provider', 'revision', 'request', 'result', 'computed_at', 'idempotency_key'),
                   ('provider', 'request', 'result', 'idempotency_key'))
            self.store.authorize(identity, 'forecast.create')
            payload = {k: v for k, v in args.items() if k not in ('action', 'idempotency_key')}
            request = submissions.execution(payload, identity.principal).request
            if not request.series_id or not request.timestamps or not request.future_timestamps:
                raise ServiceError('INVALID_ARGUMENTS', 'Named series and history/target timestamps are required.')
            if len(request.history) > 10000 or request.horizon > 1000:
                raise ServiceError('LIMIT', 'Maximum history is 10000 and horizon is 1000.')
            for t in (*request.timestamps, *request.future_timestamps):
                instant(t)
            if any(instant(t) > instant(now()) for t in request.timestamps):
                raise ServiceError('INVALID_ARGUMENTS', 'History cannot contain future observations.')
            if 'computed_at' in args and instant(args['computed_at']) > instant(now()):
                raise ServiceError('INVALID_ARGUMENTS', 'Claimed computation cannot be in the future.')
            return self.mutate(identity, 'forecast.submit', payload, args['idempotency_key'], 'forecast.create')
        if action == 'snapshot.save':
            strict(args, ('action', 'decision_id', 'source_as_of', 'recorded_as_of', 'idempotency_key'),
                   ('decision_id', 'source_as_of', 'recorded_as_of', 'idempotency_key'))
            self.store.authorize(identity, 'evidence.read')
            for field in ('source_as_of', 'recorded_as_of'):
                if instant(args[field]) > instant(now()):
                    raise ServiceError('INVALID_ARGUMENTS', 'Evidence cutoffs cannot be in the future.')
            return self.mutate(identity, 'snapshot', {k: args[k] for k in ('decision_id', 'source_as_of', 'recorded_as_of')},
                               args['idempotency_key'], 'evidence.read')
        if action == 'analysis.submit':
            strict(args, ('action', 'snapshot_id', 'method', 'method_version', 'metrics', 'lesson', 'export_to_ditto', 'idempotency_key'),
                   ('snapshot_id', 'method', 'method_version', 'metrics', 'lesson', 'idempotency_key'))
            self.store.authorize(identity, 'decision.create')
            self.store.authorize(identity, 'evidence.read')
            if type(args.get('export_to_ditto', False)) is not bool:
                raise ServiceError('INVALID_ARGUMENTS', 'export_to_ditto must be a boolean.')
            if args.get('export_to_ditto', False):
                self.store.authorize(identity, 'memory.export')
            submissions.text(args['method'], 'method')
            submissions.text(args['method_version'], 'method_version')
            submissions.text(args['lesson'], 'lesson', 8000)
            metrics = args['metrics']
            if not isinstance(metrics, dict) or len(metrics) > 32:
                raise ServiceError('INVALID_ARGUMENTS', 'Metrics require at most 32 named scalar values.')
            import math
            for key, value in metrics.items():
                submissions.text(key, 'metric name', 128)
                if value is not None and (type(value) not in (float, int) or not math.isfinite(value)):
                    raise ServiceError('INVALID_ARGUMENTS', 'Metrics must be finite numbers or null.')
            self.fetch_record(identity, 'evidence_snapshot', args['snapshot_id'])
            return self.mutate(identity, 'analysis', {k: v for k, v in args.items() if k not in ('action', 'idempotency_key')},
                               args['idempotency_key'], 'decision.create')
        if action == 'dataset.put':
            strict(args, ('action', 'request', 'idempotency_key'), ('request', 'idempotency_key'))
            self.store.authorize(identity, 'forecast.create')
            payload = {'request': asdict(ForecastRequest.from_dict(args['request']))}
            return self.mutate(identity, 'dataset', payload, args['idempotency_key'], 'forecast.create')
        if action == 'review.save':
            strict(args, ('action', 'decision_id', 'source_as_of', 'recorded_as_of', 'idempotency_key'),
                   ('decision_id', 'source_as_of', 'recorded_as_of', 'idempotency_key'))
            self.store.authorize(identity, 'decision.create')
            for field in ('source_as_of', 'recorded_as_of'):
                if instant(args[field]) > datetime.now(timezone.utc):
                    raise ServiceError('INVALID_ARGUMENTS', 'Evidence cutoffs cannot be in the future.')
            return self.mutate(identity, 'review', {k: args[k] for k in ('decision_id', 'source_as_of', 'recorded_as_of')},
                               args['idempotency_key'], 'decision.create')
        if action == 'request.get':
            strict(args, ('action', 'request_id'), ('request_id',))
            with self.store.ledger(identity.project).transaction() as conn:
                row = conn.execute('SELECT * FROM hosted_requests WHERE id=? AND principal=?',
                                   (args['request_id'], identity.principal)).fetchone()
            if not row:
                raise ServiceError('UNAVAILABLE', 'Project or resource unavailable.', 404)
            return self.receipt(row)
        if action == 'resolve':
            strict(args, ('action', 'reference', 'recorded_as_of'), ('reference',))
            self.store.authorize(identity, 'evidence.read')
            ref = args['reference']
            strict(ref, ('schema_version', 'service_id', 'project_id', 'ledger_id', 'resource_type', 'resource_id'),
                   ('schema_version', 'service_id', 'project_id', 'ledger_id', 'resource_type', 'resource_id'))
            expected = self.reference(identity.project, ref['resource_type'], ref['resource_id'])
            if ref != expected:
                raise ServiceError('UNAVAILABLE', 'Project or resource unavailable.', 404)
            kind, record_id = ref['resource_type'], ref['resource_id']
            ledger = self.store.ledger(identity.project)
            if kind in ('dataset_version', 'review', 'evidence_snapshot', 'client_analysis'):
                return {'result': self.fetch_record(identity, kind, record_id)}
            if kind == 'execution':
                return {'result': ledger.execution(record_id)}
            if kind == 'actual':
                return {'result': ledger.actual(record_id)}
            if kind == 'export_receipt':
                return {'result': self.hosted(identity, {'action': 'export.get', 'export_id': record_id})}
            if kind == 'decision':
                return {'result': ledger.decision(record_id)}
            if kind == 'lesson':
                return {'result': ledger.export_lesson(lesson_id=record_id, recorded_as_of=args.get('recorded_as_of', now()))}
            raise ServiceError('UNAVAILABLE', 'Unsupported resource type.', 404)
        if action == 'export.enqueue':
            strict(args, ('action', 'lesson_id', 'analysis_id', 'recorded_as_of', 'idempotency_key'), ('recorded_as_of', 'idempotency_key'))
            if ('lesson_id' in args) == ('analysis_id' in args):
                raise ServiceError('INVALID_ARGUMENTS', 'Supply exactly one lesson_id or analysis_id.')
            self.store.authorize(identity, 'memory.export')
            self.store.authorize(identity, 'evidence.read')
            return self.mutate(identity, 'export', {k: v for k, v in args.items() if k in ('lesson_id', 'analysis_id', 'recorded_as_of')},
                               args['idempotency_key'], 'memory.export')
        if action == 'export.get':
            strict(args, ('action', 'export_id'), ('export_id',))
            self.store.authorize(identity, 'evidence.read')
            with self.store.ledger(identity.project).transaction() as conn:
                row = conn.execute('SELECT id,state,memory_id,lesson_id,created_at,updated_at FROM hosted_exports WHERE id=?',
                                   (args['export_id'],)).fetchone()
            if not row:
                raise ServiceError('UNAVAILABLE', 'Project or resource unavailable.', 404)
            return dict(row)
        raise ServiceError('UNAVAILABLE', 'Unsupported hosted operation.', 404)

    def enqueue_export(self, identity, ledger, conn, args):
        project = self.store.project(identity.project)
        if not project['ditto_url']:
            raise ServiceError('DITTO_NOT_CONFIGURED', 'Configure a dedicated Ditto connection first.')
        if instant(args['recorded_as_of']) > datetime.now(timezone.utc):
            raise ServiceError('INVALID_ARGUMENTS', 'Evidence cutoff cannot be in the future.')
        analysis = 'analysis_id' in args
        resource_id = args['analysis_id'] if analysis else args['lesson_id']
        if analysis:
            lesson = self.fetch_record(identity, 'client_analysis', resource_id, ledger=ledger)
            row = conn.execute('SELECT recorded_at FROM hosted_records WHERE id=?', (resource_id,)).fetchone()
            if instant(row['recorded_at']) > instant(args['recorded_as_of']):
                raise ServiceError('INVALID_ARGUMENTS', 'Analysis was not recorded by export cutoff.')
            kind = 'client_analysis'
        else:
            lesson = ledger.export_lesson(**args)
            kind = 'lesson'
        destination = project['ditto_connection']
        existing = conn.execute('SELECT id,state FROM hosted_exports WHERE lesson_id=? AND destination=?',
                                (resource_id, destination)).fetchone()
        if existing:
            return {'export_id': existing['id'], 'state': existing['state']}
        ref = self.reference(identity.project, kind, resource_id)
        payload = {'schema_version': '1', 'kind': 'gnomon-client-analysis' if analysis else 'gnomon-lesson', 'reference': ref, 'lesson': lesson,
                   'narrative_verified': False}
        raw = encode(payload)
        export_id = 'export-' + uuid4().hex
        conn.execute('INSERT INTO hosted_exports VALUES(?,?,?,?,?,?,?,?,?,?)',
                     (export_id, resource_id, identity.principal, destination, raw,
                      hashlib.sha256(raw.encode()).hexdigest(), 'pending', None, now(), now()))
        conn.execute('INSERT INTO hosted_audit(principal,operation,reference,state,recorded_at) VALUES(?,?,?,?,?)',
                     (identity.principal, 'ditto.enqueue', export_id, 'pending', now()))
        return {'export_id': export_id, 'state': 'pending'}

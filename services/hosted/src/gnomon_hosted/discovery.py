"""Read-only project discovery and decision lifecycle. No inference or Ditto calls."""
from importlib.metadata import version
import json
import os
import tomllib
from pathlib import Path

from gnomon.models import BASELINES
from gnomon.product_contract import __version__
from gnomon.build_info import build_info
from .storage import ServiceError, now


def server_info(runtime, identity):
    project = runtime.store.project(identity.project)
    providers = {name: {'kind': 'statistical', 'availability': 'built_in',
                        'health_checked': False} for name in sorted(BASELINES)}
    config_status = 'not_configured'
    if project['providers']:
        try:
            # Operator-owned file only. Discovery never imports provider plugins.
            with Path(project['providers']).open('rb') as stream:
                config = tomllib.load(stream)
            specs = config.get('providers', {})
            if not isinstance(specs, dict):
                raise ValueError()
            for name, spec in specs.items():
                providers[name] = {'kind': spec.get('kind', 'custom'),
                                   'availability': 'configured_not_probed', 'health_checked': False}
            config_status = 'configured'
        except (OSError, ValueError, TypeError, AttributeError):
            config_status = 'unavailable_or_invalid'
    ditto_configured = bool(project['ditto_url'] and project['ditto_graph'] and project['ditto_token_env'])
    return {'service_id': runtime.store.service_id, 'project_id': identity.project,
            'ledger_id': project['ledger_id'], 'server_time': now(),
            'versions': {'hosted': version('gnomon-hosted'), 'core': __version__,
                         'core_build': build_info()['build_id']},
            'permissions': sorted(identity.permissions), 'scope': 'single-server project ledger',
            'providers': providers, 'provider_configuration': config_status,
            'provider_discovery_scope': 'Built-in names and explicit operator configuration only; dynamic catalog aliases and saved onboarding providers are not probed.',
            'ditto': {'configured': ditto_configured,
                      'credential_present': bool(os.environ.get(project['ditto_token_env'] or '')),
                      'graph': project['ditto_graph'] if ditto_configured else None,
                      'health_checked': False},
            'external_calls': 0, 'ledger_writes': 0}


def decision_status(runtime, identity, decision_id):
    """Current evidence and export state from one ledger read transaction.

    Saved reviews/analyses remain distinct; availability does not imply validated
    narrative or client arithmetic. No historical export-state claim is made.
    """
    ledger = runtime.store.ledger(identity.project)
    at = now()
    with ledger.transaction() as conn:
        # An absent decision is indistinguishable from a foreign-project ID.
        if not conn.execute('SELECT 1 FROM decisions WHERE decision_id=?', (decision_id,)).fetchone():
            raise ServiceError('UNAVAILABLE', 'Project or resource unavailable.', 404)
        evidence = ledger.evidence_snapshot(decision_id=decision_id, source_as_of=at, recorded_as_of=at)
        actual_ids = sorted(a['actual_id'] for a in evidence['actuals'])
        expected = evidence['execution']['request']['horizon']
        coverage = 'complete' if len(actual_ids) == expected else 'partial' if actual_ids else 'pending'
        rows = conn.execute('''SELECT id,kind,payload,recorded_at FROM hosted_records
            WHERE recorded_at<=? AND (
              (kind='evidence_snapshot' AND json_extract(payload,'$.decision.decision_id')=?) OR
              (kind='review' AND json_extract(payload,'$.decision_id')=?) OR
              (kind='client_analysis' AND json_extract(payload,'$.snapshot_id') IN
                (SELECT id FROM hosted_records WHERE kind='evidence_snapshot'
                 AND json_extract(payload,'$.decision.decision_id')=?)))
            ORDER BY recorded_at,id LIMIT 501''', (at, decision_id, decision_id, decision_id)).fetchall()
        if len(rows) > 500:
            raise ServiceError('LIMIT', 'More than 500 lifecycle records; resolve individual evidence references.')
        snapshots, reviews, analyses = [], [], []
        snapshot_actuals = {}
        for row in rows:
            payload = json.loads(row['payload'])
            if row['kind'] == 'evidence_snapshot':
                ids = sorted(a['actual_id'] for a in payload['actuals'])
                snapshot_actuals[row['id']] = ids
                snapshots.append({'snapshot_id': row['id'], 'recorded_at': row['recorded_at'],
                                  'evidence_changed': ids != actual_ids})
            elif row['kind'] == 'review':
                reviews.append({'review_id': row['id'], 'recorded_at': row['recorded_at'],
                                'scoring_status': payload['scoring_status'],
                                'evidence_changed': sorted(payload['actual_ids']) != actual_ids})
        for row in rows:
            if row['kind'] == 'client_analysis':
                payload = json.loads(row['payload'])
                analyses.append({'analysis_id': row['id'], 'snapshot_id': payload['snapshot_id'],
                                 'recorded_at': row['recorded_at'], 'numerically_verified': False,
                                 'evidence_changed': snapshot_actuals.get(payload['snapshot_id']) != actual_ids})
        evaluations = conn.execute('''SELECT evaluation_id,recorded_at FROM evaluations
            WHERE execution_id=? AND recorded_at<=? ORDER BY recorded_at,evaluation_id LIMIT 501''',
            (evidence['execution']['execution_id'], at)).fetchall()
        if len(evaluations) > 500:
            raise ServiceError('LIMIT', 'More than 500 evaluations; resolve individual evidence references.')
        lessons = conn.execute('''SELECT outcome_id,payload_json,recorded_at FROM decision_outcomes
            WHERE decision_id=? AND recorded_at<=? AND json_extract(payload_json,'$.outcome.kind')='forecast_lesson/1'
            ORDER BY recorded_at,outcome_id LIMIT 501''', (decision_id, at)).fetchall()
        if len(lessons) > 500:
            raise ServiceError('LIMIT', 'More than 500 lesson versions; resolve individual lesson references.')
        lesson_rows = [{'lesson_id': r['outcome_id'], 'recorded_at': r['recorded_at'],
                        'evidence_changed': sorted(json.loads(r['payload_json'])['outcome']['review']['actual_ids']) != actual_ids}
                       for r in lessons]
        resource_ids = [a['analysis_id'] for a in analyses] + [l['lesson_id'] for l in lesson_rows]
        exports = []
        if len(resource_ids) > 500:
            raise ServiceError('LIMIT', 'More than 500 exportable lifecycle resources; resolve individual references.')
        if resource_ids:
            # Bind IDs, never interpolate caller values into SQL.
            placeholders = ','.join('?' for _ in resource_ids)
            rows = conn.execute(f'''SELECT id,lesson_id,state,memory_id,created_at,updated_at,destination
                FROM hosted_exports WHERE lesson_id IN ({placeholders}) ORDER BY created_at,id LIMIT 501''', resource_ids).fetchall()
            if len(rows) > 500:
                raise ServiceError('LIMIT', 'More than 500 exports; resolve individual export receipts.')
            destination = runtime.store.project(identity.project)['ditto_connection']
            exports = [{**dict(r), 'current_connection': r['destination'] == destination} for r in rows]
            for r in exports:
                r.pop('destination')
    steps = []
    def step(action, reason, permissions, **ids):
        steps.append({'action': action, 'reason': reason, 'required_permissions': permissions,
                      'authorized': set(permissions) <= identity.permissions, **ids})
    if coverage != 'complete':
        step('append_actual', 'Wait for real outcomes and ingest missing matching observations.', ['actual.create'])
    if not snapshots or all(s['evidence_changed'] for s in snapshots):
        step('snapshot.save', 'Freeze current evidence before computing or updating a client analysis.', ['evidence.read'])
    if not analyses and not lesson_rows:
        step('analysis.submit', 'Compute metrics from a saved snapshot, then record a labelled lesson.', ['evidence.read', 'decision.create'])
    elif any(a['evidence_changed'] for a in analyses) or any(l['evidence_changed'] for l in lesson_rows):
        step('review.save', 'Evidence changed; review it before relying on an earlier lesson.', ['decision.create'])
    project = runtime.store.project(identity.project)
    ditto_configured = bool(project['ditto_url'] and project['ditto_graph'] and project['ditto_token_env'])
    blockers = []
    if resource_ids and not ditto_configured:
        blockers.append('An operator must configure the project Ditto connection before exporting.')
    current_resources = {a['analysis_id'] for a in analyses if not a['evidence_changed']}
    current_resources.update(l['lesson_id'] for l in lesson_rows if not l['evidence_changed'])
    exported = {e['lesson_id'] for e in exports if e['current_connection']}
    for resource_id in resource_ids:
        if ditto_configured and resource_id in current_resources and resource_id not in exported:
            step('export.enqueue', 'Lesson/analysis has no export for the current Ditto connection.',
                 ['evidence.read', 'memory.export'], resource_id=resource_id)
    for e in exports:
        if not e['current_connection']:
            continue
        if e['state'] == 'pending':
            step('export.deliver', 'Delivery pending; check Ditto configuration if a previous preflight failed.',
                 ['evidence.read', 'memory.export'], export_id=e['id'])
        elif e['state'] in ('uncertain', 'sending', 'failed'):
            step('export.reconcile', 'Inspect the remote outcome; do not blindly repeat a potentially completed save.',
                 ['evidence.read', 'memory.export'], export_id=e['id'])
    return {'decision_id': decision_id, 'execution_id': evidence['execution']['execution_id'],
            'observed_at': at, 'source_as_of': at, 'recorded_as_of': at,
            'decision': 'recorded', 'actuals': {'status': coverage, 'available': len(actual_ids),
                                             'expected': expected, 'actual_ids': actual_ids},
            'snapshots': snapshots, 'reviews': reviews, 'evaluations': [dict(r) for r in evaluations],
            'analyses': analyses, 'lessons': lesson_rows,
            'exports': exports, 'next_steps': steps, 'blockers': blockers,
            'verification': 'Lifecycle and references only; client arithmetic and narrative are not verified.',
            'external_calls': 0, 'ledger_writes': 0}

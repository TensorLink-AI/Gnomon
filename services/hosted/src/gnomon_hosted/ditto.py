"""Explicit Ditto delivery and evidence-checked recall over its real MCP contract.

No automatic retries after ambiguous saves. A local receipt is not a remote
exactly-once guarantee. The graph is selected by the operator's scoped credential.
"""
from contextlib import asynccontextmanager
import hashlib
import json
import os

import anyio
import httpx
import jsonschema
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from .runtime import strict, instant
from .storage import ServiceError, encode, now


def destination(project):
    return project['ditto_connection']


@asynccontextmanager
async def connection(project):
    url, env = project['ditto_url'], project['ditto_token_env']
    token = os.environ.get(env or '')
    if not url or not token:
        raise ServiceError('DITTO_NOT_CONFIGURED', 'Configured Ditto credential is unavailable to the server.')
    async with httpx.AsyncClient(headers={'Authorization': 'Bearer ' + token}, timeout=20,
                                 follow_redirects=False) as client:
        async with streamable_http_client(url, http_client=client) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                schemas = {}
                cursor = None
                for _ in range(20):
                    page = await session.list_tools(cursor=cursor)
                    schemas.update({tool.name: tool.inputSchema for tool in page.tools})
                    cursor = page.nextCursor
                    if not cursor:
                        break
                graphs = await invoke(session, schemas, 'list_knowledge_graphs', {})
                if graphs.get('default') != project['ditto_graph']:
                    raise ServiceError('DITTO_GRAPH', 'Credential default graph differs from the configured dedicated graph.')
                yield session, schemas


async def invoke(session, schemas, tool, arguments):
    if tool not in schemas:
        raise ServiceError('DITTO_CONTRACT', 'Required Ditto tool is not available.')
    try:
        jsonschema.validate(arguments, schemas[tool])
    except jsonschema.ValidationError:
        raise ServiceError('DITTO_CONTRACT', 'Ditto tool schema differs from the supported contract.') from None
    reply = await session.call_tool(tool, arguments)
    if reply.isError:
        raise ServiceError('DITTO_TOOL_ERROR', 'Ditto reported a tool error; details withheld.')
    result = reply.structuredContent
    if result is None:
        texts = [item.text for item in reply.content if item.type == 'text']
        if len(texts) != 1:
            raise ServiceError('DITTO_CONTRACT', 'Ditto did not return one JSON result.')
        result = json.loads(texts[0])
    if not isinstance(result, dict):
        raise ServiceError('DITTO_CONTRACT', 'Ditto result must be an object.')
    return result


async def deliver(runtime, identity, args):
    strict(args, ('action', 'export_id'), ('export_id',))
    store = runtime.store
    store.authorize(identity, 'memory.export')
    store.authorize(identity, 'evidence.read')
    project = store.project(identity.project)
    ledger = store.ledger(identity.project)
    with ledger.transaction() as conn:
        row = conn.execute('SELECT * FROM hosted_exports WHERE id=?', (args['export_id'],)).fetchone()
        if not row or row['destination'] != destination(project):
            raise ServiceError('UNAVAILABLE', 'Export unavailable for this connection.', 404)
        record = dict(row)
        if row['state'] != 'pending':
            return {'export_id': row['id'], 'state': row['state'], 'memory_id': row['memory_id']}
        if hashlib.sha256(row['payload'].encode()).hexdigest() != row['digest']:
            raise ServiceError('INTEGRITY', 'Stored export integrity check failed.')
        conn.execute("UPDATE hosted_exports SET state='sending',updated_at=? WHERE id=?", (now(), row['id']))
        conn.execute('INSERT INTO hosted_audit(principal,operation,reference,state,recorded_at) VALUES(?,?,?,?,?)',
                     (identity.principal, 'ditto.deliver', row['id'], 'sending', now()))
    attempted = False
    try:
        with anyio.fail_after(30):
            async with connection(project) as (session, schemas):
                arguments = {'content': record['payload'], 'source': 'gnomon',
                             'sourceContext': 'Gnomon project evidence; narrative is unverified hypothesis.',
                             'vendorId': record['id'],
                             'metadata': {'project_id': identity.project, 'ledger_id': project['ledger_id'],
                                          'lesson_id': record['lesson_id'], 'payload_sha256': record['digest']}}
                # Validate before marking the remote call attempted.
                if 'save_memory' not in schemas:
                    raise ServiceError('DITTO_CONTRACT', 'Ditto save_memory is unavailable.')
                jsonschema.validate(arguments, schemas['save_memory'])
                store.authorize(identity, 'memory.export')
                attempted = True
                answer = await invoke(session, schemas, 'save_memory', arguments)
                if not isinstance(answer.get('id'), str) or not answer['id']:
                    raise ServiceError('DITTO_CONTRACT', 'Ditto did not return a memory ID.')
        with ledger.transaction() as conn:
            changed = conn.execute("UPDATE hosted_exports SET state='acknowledged',memory_id=?,updated_at=? WHERE id=? AND state='sending'",
                         (answer['id'], now(), record['id'])).rowcount
            if changed != 1:
                raise ServiceError('CONFLICT', 'Export state changed during delivery; reconcile its outcome.')
            conn.execute('INSERT INTO hosted_audit(principal,operation,reference,state,recorded_at) VALUES(?,?,?,?,?)',
                         (identity.principal, 'ditto.save', record['id'], 'acknowledged', now()))
        return {'export_id': record['id'], 'state': 'acknowledged', 'memory_id': answer['id']}
    except BaseException as error:
        # Before tools/call no memory save occurred: pending is safely retryable.
        state = 'uncertain' if attempted else 'pending'
        with ledger.transaction() as conn:
            changed = conn.execute("UPDATE hosted_exports SET state=?,updated_at=? WHERE id=? AND state='sending'", (state, now(), record['id'])).rowcount
            if changed:
                conn.execute('INSERT INTO hosted_audit(principal,operation,reference,state,recorded_at) VALUES(?,?,?,?,?)',
                             (identity.principal, 'ditto.deliver', record['id'], state, now()))
            else:
                state = conn.execute('SELECT state FROM hosted_exports WHERE id=?', (record['id'],)).fetchone()[0]
        if not isinstance(error, Exception):
            raise
        return {'export_id': record['id'], 'state': state, 'error': 'Ditto delivery did not complete.',
                'guidance': 'Reconcile uncertain saves by inspecting Ditto before any new export; no automatic retry.'}


async def recall(runtime, identity, args):
    strict(args, ('action', 'query', 'source_as_of', 'recorded_as_of'), ('query', 'source_as_of', 'recorded_as_of'))
    store = runtime.store
    store.authorize(identity, 'evidence.read')
    if not isinstance(args['query'], str) or not 1 <= len(args['query']) <= 1000:
        raise ServiceError('INVALID_ARGUMENTS', 'Query must be 1-1000 characters.')
    source, recorded = instant(args['source_as_of']), instant(args['recorded_as_of'])
    if source > instant(now()) or recorded > instant(now()):
        raise ServiceError('INVALID_ARGUMENTS', 'Cutoffs cannot be in the future.')
    project, ledger = store.project(identity.project), store.ledger(identity.project)
    with ledger.transaction() as conn:
        rows = conn.execute("SELECT * FROM hosted_exports WHERE state='acknowledged' AND destination=? AND created_at<=? ORDER BY created_at DESC LIMIT 101",
                            (destination(project), recorded.isoformat(timespec='microseconds'))).fetchall()
    if len(rows) > 100:
        raise ServiceError('LIMIT', 'More than 100 eligible exports; narrow the cutoff or resolve explicit lesson references.')
    allowed = {r['memory_id']: dict(r) for r in rows}
    if not allowed:
        return {'lessons': [], 'returned': 0, 'status': 'ok', 'external_calls': 0}
    try:
        with anyio.fail_after(30):
            async with connection(project) as (session, schemas):
                search = await invoke(session, schemas, 'search_memories', {
                    'queries': args['query'], 'limit': 5, 'includePublic': False,
                    'filter': {'and': [{'field': 'source', 'op': 'eq', 'value': 'gnomon'},
                                       {'field': 'vendor_id', 'op': 'in', 'value': [r['id'] for r in allowed.values()]}]}})
                hits = search.get('memories')
                if not isinstance(hits, list):
                    raise ServiceError('DITTO_CONTRACT', 'Missing search memory list.')
                ids = list(dict.fromkeys(h.get('id') for h in hits if isinstance(h, dict) and h.get('id') in allowed))[:5]
                fetched = await invoke(session, schemas, 'fetch_memories', {'ids': ids, 'format': 'full'}) if ids else {'memories': []}
    except Exception:
        raise ServiceError('DITTO_UNAVAILABLE', 'Ditto recall failed; local ledger evidence remains available.', 502) from None
    store.authorize(identity, 'evidence.read')
    lessons, excluded = [], []
    memories = fetched.get('memories')
    if not isinstance(memories, list):
        raise ServiceError('DITTO_CONTRACT', 'Missing fetched memory list.')
    seen = set()
    for memory in memories:
        if not isinstance(memory, dict) or memory.get('id') not in ids or memory['id'] in seen:
            continue
        seen.add(memory['id'])
        row = allowed[memory['id']]
        stage = 'payload_integrity'
        try:
            remote = json.loads(memory['content'])
            if hashlib.sha256(encode(remote).encode()).hexdigest() != row['digest']:
                raise ValueError('changed remote content')
            stage = 'ledger_reference'
            if remote.get('kind') == 'gnomon-client-analysis':
                saved = runtime.fetch_record(identity, 'client_analysis', row['lesson_id'])
                if saved != remote['lesson'] or remote['reference'] != runtime.reference(identity.project, 'client_analysis', row['lesson_id']):
                    raise ValueError('reference mismatch')
                stage = 'evidence_cutoffs'
                if instant(saved['source_as_of']) > source or instant(saved['recorded_as_of']) > recorded:
                    raise ValueError('future evidence')
                snapshot = runtime.fetch_record(identity, 'evidence_snapshot', saved['snapshot_id'])
                current = ledger.evidence_snapshot(decision_id=snapshot['decision']['decision_id'],
                    source_as_of=args['source_as_of'], recorded_as_of=args['recorded_as_of'])
                lessons.append({'memory_id': memory['id'], 'reference': remote['reference'], 'lesson': saved,
                    'snapshot_reference': saved['snapshot_reference'], 'current_evidence': current,
                    'numerically_verified': False, 'narrative_verified': False,
                    'verification': 'Stored content and evidence references checked; recompute metrics locally.',
                    'evidence_changed': [a['actual_id'] for a in snapshot['actuals']] != [a['actual_id'] for a in current['actuals']]})
                continue
            saved = ledger.export_lesson(lesson_id=row['lesson_id'], recorded_as_of=args['recorded_as_of'])
            if saved != remote['lesson'] or remote['reference'] != runtime.reference(identity.project, 'lesson', row['lesson_id']):
                raise ValueError('reference mismatch')
            stage = 'evidence_cutoffs'
            if instant(saved['verification_call']['source_as_of']) > source or instant(saved['verification_call']['recorded_as_of']) > recorded:
                raise ValueError('future evidence')
            stage = 'current_review'
            current = ledger.review_decision(decision_id=saved['decision_id'], source_as_of=args['source_as_of'], recorded_as_of=args['recorded_as_of'])
            lessons.append({'memory_id': memory['id'], 'reference': remote['reference'], 'lesson': saved,
                            'current_review': current, 'narrative_verified': False,
                            'forecast_provenance': ledger.execution(saved['execution_id']).get('provider_identity'),
                            'evidence_changed': saved['actual_ids'] != current['actual_ids']})
        except Exception:
            excluded.append({'memory_id': memory['id'], 'reason': 'unverified_or_cutoff_ineligible', 'verification_stage': stage})
    return {'lessons': lessons, 'returned': len(lessons), 'excluded': excluded,
            'source_as_of': args['source_as_of'], 'recorded_as_of': args['recorded_as_of']}


async def reconcile(runtime, identity, args):
    """Resolve an ambiguous save using a full remote read; never repeat the write."""
    strict(args, ('action', 'export_id', 'memory_id'), ('export_id', 'memory_id'))
    store = runtime.store
    store.authorize(identity, 'memory.export')
    store.authorize(identity, 'evidence.read')
    project, ledger = store.project(identity.project), store.ledger(identity.project)
    with ledger.transaction() as conn:
        row = conn.execute('SELECT * FROM hosted_exports WHERE id=?', (args['export_id'],)).fetchone()
        if not row or row['destination'] != destination(project):
            raise ServiceError('UNAVAILABLE', 'Export unavailable for this connection.', 404)
        record = dict(row)
        if record['state'] == 'acknowledged':
            if record['memory_id'] != args['memory_id']:
                raise ServiceError('CONFLICT', 'Export is already linked to another memory.', 409)
            return {'export_id': record['id'], 'state': 'acknowledged', 'memory_id': record['memory_id']}
        if record['state'] != 'uncertain':
            raise ServiceError('CONFLICT', 'Only an uncertain export can be reconciled.', 409)
    try:
        with anyio.fail_after(30):
            async with connection(project) as (session, schemas):
                result = await invoke(session, schemas, 'fetch_memories', {'ids': [args['memory_id']], 'format': 'full'})
        matches = [m for m in result.get('memories', []) if m.get('id') == args['memory_id']]
        if len(matches) != 1 or hashlib.sha256(encode(json.loads(matches[0]['content'])).encode()).hexdigest() != record['digest']:
            raise ValueError('Remote evidence differs')
    except Exception:
        raise ServiceError('UNVERIFIED', 'Remote memory could not be verified; export remains uncertain.') from None
    store.authorize(identity, 'memory.export')
    with ledger.transaction() as conn:
        changed = conn.execute("UPDATE hosted_exports SET state='acknowledged',memory_id=?,updated_at=? WHERE id=? AND state='uncertain'",
                               (args['memory_id'], now(), record['id'])).rowcount
        if changed != 1:
            raise ServiceError('CONFLICT', 'Export changed during verification; retrieve its current state.', 409)
        conn.execute('INSERT INTO hosted_audit(principal,operation,reference,state,recorded_at) VALUES(?,?,?,?,?)',
                     (identity.principal, 'ditto.reconcile', record['id'], 'acknowledged', now()))
    return {'export_id': record['id'], 'state': 'acknowledged', 'memory_id': args['memory_id']}

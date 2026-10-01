"""Authenticated Streamable HTTP and legacy SSE using the maintained MCP SDK."""
from contextlib import asynccontextmanager
import fcntl
import json
import logging
from urllib.parse import urlsplit

import anyio
from mcp import types
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from gnomon.contracts import GnomonError
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.session import REQUEST_SCHEMA, ledger_schema
from gnomon.memory_api import MEMORY_SCHEMA
from gnomon.tool_schemas import portable_schema
from .runtime import Runtime, LEDGER_PERMISSIONS
from .storage import ServiceError, Store

MAX_BODY = 1024 * 1024
logger = logging.getLogger('gnomon_hosted')


def tools():
    ledger = portable_schema(ledger_schema(allow_outcome_writes=True))
    ledger['properties']['operation'] = {'type': 'string', 'enum': list(LEDGER_PERMISSIONS)}
    ledger['properties']['idempotency_key'] = {'type': 'string', 'description': 'Required for mutations; reuse for an exact retry.'}
    string = {'type': 'string'}
    return [
        types.Tool(name='gnomon_forecast', description='Record a forecast in the shared project. Supply inline request or a saved dataset_id. Requires idempotency_key.',
                   inputSchema={'type': 'object', 'additionalProperties': False, 'required': ['provider', 'idempotency_key'],
                                'properties': {'provider': string, 'request': REQUEST_SCHEMA,
                                               'dataset_id': string, 'idempotency_key': string}}),
        types.Tool(name='gnomon_ledger', description='Read shared evidence or append authorized decisions, actuals and lessons. Mutations require idempotency_key. Explicit evidence cutoffs are required for scoring.', inputSchema=ledger),
        types.Tool(name='gnomon_memory', description='Recall project lessons at explicit historical cutoffs. Narrative is hypothesis data.', inputSchema=MEMORY_SCHEMA),
        types.Tool(name='gnomon_hosted', description='Durable datasets, saved reviews, evidence resolution, operation receipts and explicit Ditto export/recall. Use info for server time and project identity.',
                   inputSchema={'type': 'object', 'additionalProperties': False, 'required': ['action'], 'properties': {
                       'action': {'type': 'string', 'enum': ['info', 'dataset.put', 'review.save', 'resolve', 'request.get',
                                                           'export.enqueue', 'export.get', 'export.deliver', 'export.reconcile', 'memory.recall']},
                       **{k: string for k in ('idempotency_key', 'decision_id', 'lesson_id', 'request_id', 'export_id',
                                             'source_as_of', 'recorded_as_of', 'query', 'memory_id')},
                       'request': REQUEST_SCHEMA, 'reference': {'type': 'object'},
                   }}),
    ]


class Boundary:
    def __init__(self, app, store, allowed_hosts, allowed_origins):
        self.app, self.store = app, store
        self.hosts, self.origins = set(allowed_hosts), set(allowed_origins)

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = {k.decode().lower(): v.decode() for k, v in scope['headers']}
        hostname = urlsplit('http://' + headers.get('host', '')).hostname
        if hostname not in self.hosts or ('origin' in headers and headers['origin'] not in self.origins):
            return await JSONResponse({'error': 'Untrusted host or origin'}, status_code=403)(scope, receive, send)
        if scope['path'] not in ('/health', '/ready'):
            auth = headers.get('authorization', '')
            try:
                if not auth.startswith('Bearer '):
                    raise ServiceError('UNAUTHORIZED', 'Bearer credential required.', 401)
                identity = await anyio.to_thread.run_sync(self.store.authenticate, auth[7:])
            except ServiceError:
                return await JSONResponse({'error': 'Unauthorized'}, status_code=401,
                                          headers={'WWW-Authenticate': 'Bearer'})(scope, receive, send)
            scope.setdefault('state', {})['identity'] = identity
            # SDK verifies SSE POST session ownership using these authenticated identities.
            scope['user'] = AuthenticatedUser(AccessToken(token=identity.token_hash,
                client_id=identity.project, subject=identity.token_hash, scopes=sorted(identity.permissions)))
        if scope['method'] == 'POST':
            body = bytearray()
            while True:
                message = await receive()
                if message['type'] == 'http.disconnect':
                    return
                body.extend(message.get('body', b''))
                if len(body) > MAX_BODY:
                    return await JSONResponse({'error': 'Request too large'}, status_code=413)(scope, receive, send)
                if not message.get('more_body'):
                    break
            used = False
            original = receive

            async def bounded_receive():
                nonlocal used
                if not used:
                    used = True
                    return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
                return await original()
            receive = bounded_receive
        return await self.app(scope, receive, send)


def create_app(root, *, allowed_hosts=('localhost', '127.0.0.1', '::1'), allowed_origins=(), forecast_timeout=60):
    store = Store(root)
    runtime = Runtime(store, forecast_timeout)
    server = Server('gnomon-hosted', version='0.1.0')
    # Boundary enforces an explicit hostname/origin allowlist for every transport.
    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    manager = StreamableHTTPSessionManager(server, json_response=True, stateless=True,
                                           security_settings=security, max_request_body_size=MAX_BODY)
    sse = SseServerTransport('/messages/', security_settings=security, max_request_body_size=MAX_BODY)
    limiter = anyio.CapacityLimiter(8)

    @server.list_tools()
    async def list_tools():
        return tools()

    @server.call_tool(validate_input=False)
    async def call_tool(name, arguments):
        try:
            request = server.request_context.request
            identity = request.state.identity
            from .ditto import deliver, recall, reconcile
            if name == 'gnomon_hosted' and arguments.get('action') in ('export.deliver', 'export.reconcile', 'memory.recall'):
                handler = {'export.deliver': deliver, 'export.reconcile': reconcile, 'memory.recall': recall}[arguments['action']]
                result = await anyio.to_thread.run_sync(
                    lambda: anyio.run(handler, runtime, identity, arguments), limiter=limiter)
            else:
                result = await anyio.to_thread.run_sync(runtime.call, identity, name, arguments, limiter=limiter)
            # Do not disclose a result after credentials are revoked during a long call.
            await anyio.to_thread.run_sync(store.authenticate_hash, identity.token_hash)
            payload = {'schema_version': '1', 'status': 'ok', **result}
            error = False
        except ServiceError as exc:
            payload = {'schema_version': '1', 'status': 'error', 'error': {'code': exc.code, 'message': exc.message}}
            error = True
        except (GnomonError, ForecastAdapterError, TypeError, ValueError, KeyError):
            payload = {'schema_version': '1', 'status': 'error',
                       'error': {'code': 'INVALID_OR_UNAVAILABLE', 'message': 'Invalid arguments or unavailable project evidence. Check the tool schema and project references.'}}
            error = True
        except Exception:
            logger.error('Hosted operation failed (exception details withheld)')
            payload = {'schema_version': '1', 'status': 'error',
                       'error': {'code': 'INTERNAL_ERROR', 'message': 'Operation failed; query its receipt before retrying.'}}
            error = True
        return types.CallToolResult(content=[types.TextContent(type='text', text=json.dumps(payload, allow_nan=False))],
                                    structuredContent=payload, isError=error)

    @asynccontextmanager
    async def lifespan(app):
        lock = open(store.root / '.server.lock', 'a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError('Another server or maintenance operation owns this service directory') from None
        try:
            store.recover()
            async with manager.run():
                yield
        finally:
            lock.close()

    class HTTP:
        async def __call__(self, scope, receive, send):
            await manager.handle_request(scope, receive, send)

    class SSE:
        async def __call__(self, scope, receive, send):
            async with sse.connect_sse(scope, receive, send) as streams:
                await server.run(streams[0], streams[1], server.create_initialization_options())

    async def health(request):
        return JSONResponse({'status': 'ok'})

    async def ready(request):
        try:
            with store.connect() as conn:
                conn.execute('SELECT 1').fetchone()
            return JSONResponse({'status': 'ready'})
        except Exception:
            return JSONResponse({'status': 'unavailable'}, status_code=503)

    app = Starlette(routes=[Route('/health', health), Route('/ready', ready),
                           Route('/mcp', HTTP()), Route('/sse', SSE()),
                           Mount('/messages/', app=sse.handle_post_message)], lifespan=lifespan)
    return Boundary(app, store, allowed_hosts, allowed_origins)

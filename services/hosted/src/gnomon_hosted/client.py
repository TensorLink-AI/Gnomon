"""Optional local-computation CLI. No local service or local ledger is required."""
import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from gnomon import GnomonSession
from gnomon.evidence import score_snapshot


async def invoke(url, token, tool, arguments):
    async with httpx.AsyncClient(headers={'Authorization': 'Bearer ' + token}, timeout=90) as http:
        async with streamable_http_client(url, http_client=http) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                response = await session.call_tool(tool, arguments)
    result = response.structuredContent
    if not isinstance(result, dict) or response.isError or result.get('status') != 'ok':
        raise RuntimeError('Shared service rejected the call; inspect the request and receipt before retrying.')
    if result.get('state') not in (None, 'completed'):
        raise RuntimeError('Request is unresolved; query request.get before making a new submission.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='Shared server MCP URL (HTTPS remotely).')
    parser.add_argument('--token-env', default='GNOMON_SERVICE_TOKEN')
    sub = parser.add_subparsers(dest='command', required=True)
    forecast = sub.add_parser('forecast', help='Run local Gnomon inference and submit its result.')
    forecast.add_argument('--request', type=Path, required=True)
    forecast.add_argument('--provider', required=True)
    forecast.add_argument('--providers-config')
    forecast.add_argument('--idempotency-key', required=True)
    forecast.add_argument('--submission-file', type=Path, required=True,
                          help='New private JSON file preserving exact submission for retries. Existing files are never overwritten.')
    analyze = sub.add_parser('analyze', help='Resolve an immutable snapshot, score locally and submit a labelled analysis.')
    analyze.add_argument('--snapshot-reference', type=Path, required=True)
    analyze.add_argument('--lesson', required=True)
    analyze.add_argument('--idempotency-key', required=True)
    analyze.add_argument('--export-to-ditto', action='store_true')
    call = sub.add_parser('call', help='Call a shared tool with JSON arguments on stdin; use for exact retries.')
    call.add_argument('--tool', default='gnomon_hosted', choices=['gnomon_hosted', 'gnomon_ledger', 'gnomon_memory', 'gnomon_forecast'])
    args = parser.parse_args()
    token = os.environ[args.token_env]
    def remote(tool, arguments):
        return asyncio.run(invoke(args.url, token, tool, arguments))
    if args.command == 'forecast':
        # Check before invoking an expensive local/remote provider.
        if args.submission_file.exists():
            parser.error('Submission file exists; replay it with call, or choose a new file and key.')
        request = json.loads(args.request.read_text())
        with GnomonSession.from_config(args.providers_config) as session:
            execution = session.engine.forecast(args.provider, request, use_cache=False)
        payload = {'action': 'forecast.submit', 'provider': execution.provider, 'revision': execution.revision,
                   'request': asdict(execution.request), 'result': asdict(execution.result),
                   'idempotency_key': args.idempotency_key}
        # Persist the exact payload before transport; retry does not rerun inference.
        with os.fdopen(os.open(args.submission_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as handle:
            json.dump(payload, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        result = remote('gnomon_hosted', payload)
    elif args.command == 'analyze':
        reference = json.loads(args.snapshot_reference.read_text())
        if reference.get('resource_type') != 'evidence_snapshot':
            parser.error('Expected an evidence_snapshot reference.')
        snapshot = remote('gnomon_hosted', {'action': 'resolve', 'reference': reference})['result']
        local = score_snapshot(snapshot)
        result = remote('gnomon_hosted', {'action': 'analysis.submit', 'snapshot_id': reference['resource_id'],
            'method': local['method'], 'method_version': local['method_version'], 'metrics': local['metrics'],
            'lesson': args.lesson, 'idempotency_key': args.idempotency_key, 'export_to_ditto': args.export_to_ditto})
        result['local_computation'] = local
    else:
        result = remote(args.tool, json.load(sys.stdin))
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()

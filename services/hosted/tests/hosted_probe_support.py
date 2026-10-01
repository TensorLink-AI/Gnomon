from contextlib import contextmanager
import asyncio
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.client.sse import sse_client


@contextmanager
def serving(root, *, env=None):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    log = open(Path(root) / 'test-server.log', 'a')
    process = subprocess.Popen([sys.executable, '-m', 'gnomon_hosted.cli', '--root', str(root),
                                'serve', '--port', str(port), '--forecast-timeout', '5'],
                               stdout=log, stderr=log, env={**os.environ, **(env or {})})
    url = f'http://127.0.0.1:{port}'
    try:
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError('server exited: ' + (Path(root) / 'test-server.log').read_text())
            try:
                if httpx.get(url + '/ready').status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(.05)
        else:
            raise RuntimeError('server did not become ready')
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


async def async_call(url, token, name, arguments, *, sse=False):
    headers = {'Authorization': 'Bearer ' + token}
    if sse:
        async with sse_client(url + '/sse', headers=headers) as streams:
            async with ClientSession(*streams) as client:
                await client.initialize()
                reply = await client.call_tool(name, arguments)
    else:
        async with httpx.AsyncClient(headers=headers, timeout=20) as http:
            async with streamable_http_client(url + '/mcp', http_client=http) as streams:
                async with ClientSession(streams[0], streams[1]) as client:
                    await client.initialize()
                    reply = await client.call_tool(name, arguments)
    return reply.structuredContent


def call(*args, **kwargs):
    return asyncio.run(async_call(*args, **kwargs))


@contextmanager
def ditto_peer(root):
    """A separate process with real MCP transport and durable local peer storage."""
    root.mkdir()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    log = open(root / 'peer.log', 'a')
    process = subprocess.Popen([sys.executable, str(Path(__file__).with_name('ditto_peer.py'))],
        env={**os.environ, 'DITTO_PEER_ROOT': str(root), 'DITTO_PEER_PORT': str(port)}, stdout=log, stderr=log)
    try:
        url = f'http://127.0.0.1:{port}/mcp'
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError((root / 'peer.log').read_text())
            try:
                if httpx.get(url).status_code != 503:
                    break
            except httpx.ConnectError:
                time.sleep(.05)
        else:
            raise RuntimeError('Peer did not start')
        yield url
    finally:
        process.terminate()
        try:
            process.wait(10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()

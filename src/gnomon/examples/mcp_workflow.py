"""A complete stdlib MCP stdio client with verified result pagination.

Run: python -m gnomon.examples.mcp_workflow
Synthetic data only; the server runs in a temporary directory.
"""
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from queue import Queue
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread


def main():
    with TemporaryDirectory(prefix='gnomon-mcp-example-') as directory:
        root = Path(directory)
        root.joinpath('data.csv').write_text('timestamp,value\n' + ''.join(
            f'{date(2026, 1, 1) + timedelta(days=i)},{i + 1}\n' for i in range(60)))
        root.joinpath('providers.toml').write_text('schema_version=1\n[result_limits]\nmax_response_bytes=2048\n')
        process = subprocess.Popen([sys.executable, '-m', 'gnomon.cli', 'mcp', 'serve', '--providers-config', 'providers.toml'],
            cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        replies, transcript = Queue(), []
        def receive():
            for line in process.stdout:
                replies.put(line)
        Thread(target=receive, daemon=True).start()
        serial = 0
        def send(method, params=None, notification=False):
            nonlocal serial
            serial += 1
            message = {'jsonrpc': '2.0', 'method': method}
            if not notification:
                message['id'] = serial
            if params is not None:
                message['params'] = params
            transcript.append({'sent': message})
            process.stdin.write(json.dumps(message) + '\n')
            process.stdin.flush()
            if notification:
                return None
            response = json.loads(replies.get(timeout=15))
            transcript.append({'received': response})
            assert response['id'] == serial and 'error' not in response, response
            return response['result']
        def call(tool, arguments):
            response = send('tools/call', {'name': tool, 'arguments': arguments})
            assert not response.get('isError'), response
            return response['structuredContent']
        def complete(receipt):
            if 'result_ref' not in receipt:
                return receipt
            text, offset = '', 0
            while offset is not None:
                page = call('gnomon_read', {'result_ref': receipt['result_ref'], 'offset': offset, 'max_chars': 4096})
                text += page['text']
                offset = page['next_offset']
            assert len(text) == receipt['total_chars']
            assert hashlib.sha256(text.encode()).hexdigest() == receipt['root_sha256']
            return json.loads(text)
        try:
            initialized = send('initialize', {'protocolVersion': '2025-06-18', 'capabilities': {},
                'clientInfo': {'name': 'gnomon-example', 'version': '1'}})
            assert initialized['protocolVersion'] == '2025-06-18'
            send('notifications/initialized', notification=True)
            tools = send('tools/list')['tools']
            capabilities = complete(call('gnomon_capabilities', {}))
            inspected = complete(call('gnomon_inspect', {'input': 'data.csv', 'timezone': 'UTC'}))
            receipt = call('gnomon_forecast', {'provider': 'last_value', 'data_ref': inspected['data_ref'], 'horizon': 2000})
            forecast = complete(receipt)
            assert forecast['result']['point'] == [60] * 2000
            assert len(forecast['result']['timestamps']) == 2000
            print(json.dumps({'status': 'ok', 'protocol': initialized['protocolVersion'], 'tools': [t['name'] for t in tools],
                'capabilities_version': capabilities['runtime_version'], 'verified_points': 2000,
                'total_chars': receipt['total_chars'], 'root_sha256': receipt['root_sha256'],
                'transcript': transcript}))
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            assert process.returncode == 0, process.stderr.read()


if __name__ == '__main__':
    main()

"""Hermes real discovery/registry/transport in a fresh isolated process (no LLM).

Read one JSON tool call from stdin. Credential arrives only through an environment
variable. This probe intentionally tests a pinned checkout, not an imitation SDK.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hermes-root', type=Path, required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--token-env', default='GNOMON_SERVICE_TOKEN')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    pin = (repo / 'integrations/hermes/gnomon-memory/hermes-revision.txt').read_text().strip()
    actual = subprocess.check_output(['git', '-C', str(args.hermes_root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != pin:
        raise RuntimeError('Hermes checkout does not match the compatibility pin')
    request = json.load(sys.stdin)
    sys.path.insert(0, str(args.hermes_root.resolve()))
    with TemporaryDirectory(prefix='gnomon-hermes-client-') as home:
        os.environ['HERMES_HOME'] = home
        from tools.mcp_tool_discovery import register_mcp_servers
        from tools.mcp_tool_lifecycle import shutdown_mcp_servers
        from tools.registry import registry
        from tools.schema_sanitizer import sanitize_tool_schemas
        try:
            names = register_mcp_servers({'gnomon': {'url': args.url,
                'headers': {'Authorization': 'Bearer ' + os.environ[args.token_env]},
                'connect_timeout': 15, 'tool_timeout': 60}})
            definitions = sanitize_tool_schemas(registry.get_definitions(set(names)))
            matches = [d['function'] for d in definitions if d['function']['name'].endswith('_' + request['tool'])]
            if len(matches) != 1:
                raise RuntimeError('Requested tool was not discovered by Hermes: ' + repr(names) + ' definitions=' + repr([d['function']['name'] for d in definitions]))
            tool = matches[0]
            if not set(request['arguments']) <= set(tool['parameters']['properties']):
                raise RuntimeError('Hermes removed required argument properties')
            reply = registry.dispatch(tool['name'], request['arguments'])
            result = json.loads(reply)
            if isinstance(result, dict) and isinstance(result.get('result'), str):
                result = json.loads(result['result'])
            print(json.dumps(result))
        finally:
            shutdown_mcp_servers()


if __name__ == '__main__':
    main()

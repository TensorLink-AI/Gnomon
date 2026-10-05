"""Generate listing metadata from package identity; never publish anything."""
import json
from pathlib import Path
import runpy
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]

def artifacts():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    version = runpy.run_path(str(ROOT / 'src/gnomon/product_contract.py'))['__version__']
    about = {'description': project['description'], 'topics': [
        'time-series', 'forecasting', 'mcp', 'mcp-server', 'ai-agents', 'backtesting', 'python']}
    server = {'$schema': 'https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json',
        'name': 'io.github.TensorLink-AI/gnomon', 'title': 'Gnomon', 'description': project['description'],
        'version': version, 'repository': {'url': project['urls']['Repository'], 'source': 'github'},
        'websiteUrl': project['urls']['Documentation'], 'packages': [{
            'registryType': 'pypi', 'registryBaseUrl': 'https://pypi.org', 'identifier': project['name'],
            'version': version, 'runtimeHint': 'uvx', 'transport': {'type': 'stdio'},
            'packageArguments': [{'type': 'positional', 'value': 'mcp'}, {'type': 'positional', 'value': 'serve'}]}]}
    return {'server.json': server, 'integrations/mcp/github-about.json': about,
            'integrations/mcp/installed.json': {'mcpServers': {'gnomon': {'command': 'gnomon', 'args': ['mcp', 'serve']}}},
            'integrations/mcp/pypi.json': {'mcpServers': {'gnomon': {'command': 'uvx',
                'args': ['--from', f'gnomon-forecast=={version}', 'gnomon', 'mcp', 'serve']}}}}

if __name__ == '__main__':
    check = '--check' in sys.argv
    for relative, value in artifacts().items():
        path = ROOT / relative
        text = json.dumps(value, indent=2) + '\n'
        if check:
            if not path.exists() or path.read_text() != text:
                raise SystemExit(f'{relative} is stale; run python scripts/prepare_discovery.py')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    print('Discovery artifacts checked.' if check else 'Discovery artifacts prepared; nothing published.')

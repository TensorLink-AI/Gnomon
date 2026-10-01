"""Local MCP contract peer, explicitly not a substitute for live Ditto validation."""
import os
import sqlite3
from pathlib import Path
from uuid import uuid4
from mcp.server.fastmcp import FastMCP

root = Path(os.environ['DITTO_PEER_ROOT'])
with sqlite3.connect(root / 'memories.db') as conn:
    conn.execute('CREATE TABLE IF NOT EXISTS memories(id TEXT PRIMARY KEY, vendor TEXT UNIQUE, content TEXT)')
mcp = FastMCP('test-ditto-peer', host='127.0.0.1', port=int(os.environ['DITTO_PEER_PORT']), stateless_http=True, json_response=True)


@mcp.tool()
def list_knowledge_graphs() -> dict:
    return {'default': 'test-graph', 'knowledgeGraphs': [{'alias': 'test-graph', 'canRead': True, 'canWrite': True}]}


@mcp.tool()
def save_memory(content: str, source: str, sourceContext: str, vendorId: str, metadata: dict) -> dict:
    with sqlite3.connect(root / 'memories.db') as conn:
        conn.execute('INSERT OR IGNORE INTO memories VALUES(?,?,?)', ('memory-' + uuid4().hex, vendorId, content))
        mid = conn.execute('SELECT id FROM memories WHERE vendor=?', (vendorId,)).fetchone()[0]
    if (root / 'fail-after-save').exists():
        raise RuntimeError('Simulated loss of successful acknowledgement')
    return {'id': mid, 'relatedMemories': []}


@mcp.tool()
def search_memories(queries: str, limit: int, includePublic: bool, filter: dict) -> dict:
    vendor_ids = filter['and'][1]['value']
    with sqlite3.connect(root / 'memories.db') as conn:
        rows = conn.execute('SELECT id,vendor FROM memories').fetchall()
    return {'memories': [{'id': mid} for mid, vendor in rows if vendor in vendor_ids][:limit]}


@mcp.tool()
def fetch_memories(ids: list[str], format: str) -> dict:
    with sqlite3.connect(root / 'memories.db') as conn:
        rows = conn.execute('SELECT id,content FROM memories').fetchall()
    return {'memories': [{'id': mid, 'content': content} for mid, content in rows if mid in ids]}


if __name__ == '__main__':
    mcp.run(transport='streamable-http')

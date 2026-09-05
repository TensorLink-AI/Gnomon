"""Ordinary startup exercises the session, without legacy environment fixtures."""

from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_default_stdio_owns_one_session_and_reuses_frozen_data(tmp_path, monkeypatch):
    from gnomon.mcp_server import serve
    from gnomon.session import GnomonSession

    monkeypatch.delenv("GNOMON_MCP_PROFILE", raising=False)
    source = tmp_path / "series.csv"
    source.write_text("timestamp,value\n2026-01-01,1\n2026-01-02,9\n2026-01-03,2\n")
    output = StringIO()
    closed = []
    original_close = GnomonSession.close

    def close(self):
        original_close(self)
        closed.append(self)

    monkeypatch.setattr(GnomonSession, "close", close)

    def messages():
        yield json.dumps({"id": 1, "method": "tools/list"})
        yield json.dumps({"id": 2, "method": "tools/call", "params": {
            "name": "gnomon_inspect", "arguments": {"input": str(source)}}})
        inspected = json.loads(output.getvalue().splitlines()[-1])["result"]["structuredContent"]
        ref = inspected["data_ref"]
        source.write_text("timestamp,value\n2026-01-01,1000\n")
        yield json.dumps({"id": 3, "method": "tools/call", "params": {
            "name": "gnomon_describe", "arguments": {"data_ref": ref, "statistic": "mean"}}})
        yield json.dumps({"id": 4, "method": "tools/call", "params": {
            "name": "gnomon_forecast", "arguments": {"data_ref": ref, "provider": "last_value", "horizon": 2}}})

    assert serve(messages(), output) == 0
    results = {row["id"]: row["result"] for row in map(json.loads, output.getvalue().splitlines())}
    assert {tool["name"] for tool in results[1]["tools"]} == {
        "gnomon_capabilities", "gnomon_inspect", "gnomon_describe", "gnomon_forecast", "gnomon_evaluate", "gnomon_read"}
    assert results[3]["structuredContent"]["value"] == 4
    assert results[4]["structuredContent"]["result"]["point"] == [2, 2]
    assert results[4]["structuredContent"]["action_authorized"] is False
    assert len(closed) == 1


@pytest.mark.parametrize("command", ["capabilities", "mcp"])
def test_default_cli_is_free_of_advanced_imports(command, monkeypatch, tmp_path):
    monkeypatch.delenv("GNOMON_MCP_PROFILE", raising=False)
    code = """
import sys
from gnomon.cli import main
args = ['mcp', 'serve'] if sys.argv[1] == 'mcp' else ['capabilities']
assert main(args) == 0
for module in ('toolspec', 'runtime', 'context', 'evaluation', 'publication', 'legacy_experiments', 'temporal_ops'):
    assert 'gnomon.' + module not in sys.modules, module
"""
    result = subprocess.run([sys.executable, "-c", code, command], text=True, capture_output=True,
        input=json.dumps({"id": 1, "method": "tools/list"}) + "\n", cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}, timeout=30)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    if command == "capabilities":
        assert data["mcp_profile"]["active"] == "execution"
        assert data["product_contract"]["forecast_superiority"] == "not_established"
    else:
        assert len(data["result"]["tools"]) == 6

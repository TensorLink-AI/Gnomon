"""capabilities carries an additive top-level `ephemeris` field on CLI and MCP."""

import json
import subprocess
import sys

from gnomon import GnomonSession
from gnomon.mcp_server import _handle

from test_ephemeris import service  # noqa: F401


def _call(session, name, arguments=None):
    return _handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": name, "arguments": arguments or {}}}, session=session)


def test_unconfigured_session_reports_setup_nudge_on_cli_python_and_mcp():
    with GnomonSession.from_config() as session:
        field = session.capabilities()["ephemeris"]
        assert field["configured"] is False
        assert "base_url_env" in field["setup"] and field["setup"].endswith("#ephemeris-hosted-models")
        assert set(field) == {"configured", "setup"}
        mcp = _call(session, "gnomon_capabilities")
        assert mcp["isError"] is False and mcp["structuredContent"]["ephemeris"] == field
        listed = _handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, session=session)["tools"]
        capabilities_tool = next(tool for tool in listed if tool["name"] == "gnomon_capabilities")
        assert list(capabilities_tool["inputSchema"]["properties"]) == ["brief"]
        assert [tool["name"] for tool in listed] == [tool["name"] for tool in session.tools()]
    cli = subprocess.run([sys.executable, "-m", "gnomon", "capabilities"], capture_output=True, text=True)
    assert json.loads(cli.stdout)["ephemeris"] == field


def test_configured_session_reports_env_name_and_discovery_count(tmp_path, service, monkeypatch):
    url, state = service
    monkeypatch.setenv("TEST_EPHEMERIS_URL", url)
    path = tmp_path / "providers.toml"
    path.write_text('schema_version = 1\n[providers.remote]\nkind = "ephemeris"\nbase_url_env = "TEST_EPHEMERIS_URL"\ndiscover = true\n')
    with GnomonSession.from_config(str(path)) as session:
        field = session.capabilities()["ephemeris"]
        assert field == {"configured": True, "base_url_env": "TEST_EPHEMERIS_URL", "discovered_models": 1}
        assert "remote/not-in-gnomon" in session.capabilities()["providers"]
        compact = _call(session, "gnomon_capabilities")["structuredContent"]
        if compact.get("partial"):
            # A larger discovery payload moves behind a result reference; read the field back exactly.
            page = _call(session, "gnomon_read", {"result_ref": compact["result_ref"], "pointer": "/ephemeris"})
            compact = {"ephemeris": json.loads(page["structuredContent"]["text"])}
        assert compact["ephemeris"] == field
    path.write_text(f'schema_version = 1\n[providers.remote]\nkind = "ephemeris"\nbase_url = "{url}"\n')
    with GnomonSession.from_config(str(path)) as session:
        assert session.capabilities()["ephemeris"] == {"configured": True, "base_url_env": None, "discovered_models": 0}

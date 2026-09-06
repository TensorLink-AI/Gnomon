"""Execute packaged skill examples against the actual agent contract."""

import json
from pathlib import Path
import re

from gnomon import GnomonSession


SKILL = Path(__file__).resolve().parents[1] / "skills/use-gnomon/SKILL.md"


def test_skill_forecast_example_uses_exposed_tool_and_executes_offline(monkeypatch):
    monkeypatch.delenv("GNOMON_MCP_PROFILE", raising=False)
    examples = re.findall(r"```json\n(.*?)\n```", SKILL.read_text(), re.S)
    assert examples, "The skill must retain its runnable tool-call example"
    with GnomonSession.from_config() as session:
        exposed = {tool["name"] for tool in session.tools()}
        for example in examples:
            call = json.loads(example)
            assert call["name"] in exposed
            result = session.call(call["name"], call["arguments"], compact=False)
            assert result["status"] == "ok"
            assert result["provider"] == "last_value"
            assert list(result["result"]["point"]) == [11.0, 11.0]
            assert result["action_authorized"] is False
            assert result["recorded"] is False

from __future__ import annotations

import json

from aion.cli import main


def test_capabilities_are_honest(capsys) -> None:
    assert main(["capabilities"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["interfaces"]["cli"] is True
    assert result["interfaces"]["mcp"] is True
    assert result["features"]["context_events"] is True
    assert result["models"]["tsfm"] == []


def test_missing_input_returns_structured_error(capsys) -> None:
    result = main([
        "inspect", "/does/not/exist.csv", "--time", "timestamp", "--target", "value"
    ])
    assert result == 2
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "error"
    assert error["error"]["code"] == "INPUT_NOT_FOUND"


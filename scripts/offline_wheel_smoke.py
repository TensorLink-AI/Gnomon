#!/usr/bin/env python3
"""Prove a built Gnomon wheel works without an index or network service."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import date, timedelta


def run(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command, cwd=cwd, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
    )
    return completed.stdout


def mcp_exchange(gnomon: Path, messages: list[dict[str, object]], *, cwd: Path) \
        -> dict[int, dict[str, object]]:
    completed = subprocess.run(
        [str(gnomon), "mcp", "serve"], cwd=cwd, check=True, text=True,
        input="".join(json.dumps(message) + "\n" for message in messages),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines()
                 if line.strip()]
    return {int(response["id"]): response for response in responses
            if response.get("id") is not None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        parser.error(f"not a wheel: {wheel}")

    with tempfile.TemporaryDirectory(prefix="gnomon-offline-smoke-") as raw:
        root = Path(raw)
        environment = root / "venv"
        run([os.sys.executable, "-m", "venv", str(environment)])
        python = environment / "bin" / "python"
        gnomon = environment / "bin" / "gnomon"
        run([
            str(python), "-m", "pip", "install", "--no-index", "--no-deps",
            str(wheel),
        ])

        capabilities = json.loads(run([str(gnomon), "capabilities"], cwd=root))
        assert capabilities["product_contract"]["offline_builtin_runtime"] is True
        assert capabilities["product_contract"]["default_mcp_profile"] == "core"
        self_check = json.loads(run([
            str(gnomon), "self-check", "leakage", "--cases", "8", "--seed", "7",
        ], cwd=root))
        assert self_check["structural_claim_proven"] is True

        source = root / "series.csv"
        start = date(2026, 1, 1)
        source.write_text(
            "timestamp,value\n" + "\n".join(
                f"{start + timedelta(days=index)},{100 + index + (index % 7)}"
                for index in range(56)
            ) + "\n",
            encoding="utf-8",
        )
        inspected = json.loads(run([
            str(gnomon), "inspect", str(source), "--time", "timestamp",
            "--target", "value", "--frequency", "D",
        ], cwd=root))
        assert inspected["status"] == "valid"
        forecast = json.loads(run([
            str(gnomon), "forecast", str(source), "--time", "timestamp",
            "--target", "value", "--frequency", "D", "--horizon", "7",
            "--output", str(root / "artifacts"),
        ], cwd=root))
        artifact = Path(forecast["artifact_path"])
        assert artifact.joinpath("artifact.json").is_file()
        assert artifact.joinpath("summary.md").is_file()
        assert artifact.joinpath("report.html").is_file()
        assert forecast["tier_floor"] in {
            "supported", "conditionally_supported", "best_effort",
        }
        summary = artifact.joinpath("summary.md").read_text(encoding="utf-8")
        report = artifact.joinpath("report.html").read_text(encoding="utf-8")
        assert f"- Support: {forecast['tier_floor']}" in summary
        assert f"<dt>Support</dt><dd>{forecast['tier_floor']}</dd>" in report

        mcp = mcp_exchange(gnomon, [
            {
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
            },
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {
                    "name": "gnomon_forecast",
                    "arguments": {
                        "input": str(source), "time_column": "timestamp",
                        "target_column": "value", "frequency": "D",
                        "horizon": 7, "output_dir": str(root / "mcp-artifacts"),
                    },
                },
            },
        ], cwd=root)
        assert mcp[1]["result"]["serverInfo"]["version"] == \
            capabilities["runtime_version"]
        tools = mcp[2]["result"]["tools"]
        assert len(tools) == 10
        assert {tool["name"] for tool in tools} >= {
            "gnomon_forecast", "gnomon_monitor", "gnomon_explain_run",
        }
        tool_result = mcp[3]["result"]
        assert tool_result["isError"] is False
        structured = tool_result["structuredContent"]
        assert structured["tier_floor"] == forecast["tier_floor"]
        assert structured["artifact_id"] == structured["forecast_id"]

        print(json.dumps({
            "status": "passed",
            "runtime_version": capabilities["runtime_version"],
            "default_mcp_profile": capabilities["mcp_profile"]["active"],
            "structural_leakage_check": "passed",
            "forecast_tier_floor": forecast["tier_floor"],
            "packaged_mcp_journey": "passed",
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

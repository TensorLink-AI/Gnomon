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

        print(json.dumps({
            "status": "passed",
            "runtime_version": capabilities["runtime_version"],
            "default_mcp_profile": capabilities["mcp_profile"]["active"],
            "structural_leakage_check": "passed",
            "forecast_tier_floor": forecast["tier_floor"],
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prove a built Gnomon wheel works without an index or network service."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import date, datetime, timedelta, timezone


def offline_env():
    environment = dict(os.environ)
    # A caller's checkout import path/profile must not make an installed-wheel
    # check pass by importing different source or selecting a legacy default.
    for key in ("PYTHONPATH", "PYTHONHOME", "GNOMON_MCP_PROFILE"):
        environment.pop(key, None)
    return {**environment, "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}


def run(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command, cwd=cwd, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=offline_env(),
    )
    return completed.stdout


def mcp_exchange(gnomon: Path, messages: list[dict[str, object]], *, cwd: Path,
                 providers_config: Path | None = None, profile: str | None = None) \
        -> dict[int, dict[str, object]]:
    command = [str(gnomon), "mcp", "serve"]
    if profile is not None:
        command.extend(["--profile", profile])
    if providers_config is not None:
        command.extend(["--providers-config", str(providers_config)])
    completed = subprocess.run(
        command, cwd=cwd, check=True, text=True,
        input="".join(json.dumps(message) + "\n" for message in messages),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=offline_env(),
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines()
                 if line.strip()]
    return {int(response["id"]): response for response in responses
            if response.get("id") is not None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--example-wheel", type=Path, help="also verify the installed user-provider example")
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        parser.error(f"not a wheel: {wheel}")
    example_wheel = args.example_wheel.resolve() if args.example_wheel else None
    if example_wheel is not None and (not example_wheel.is_file() or example_wheel.suffix != ".whl"):
        parser.error(f"not an example wheel: {example_wheel}")

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
        assert environment.joinpath("share/gnomon/skills/use-gnomon/SKILL.md").is_file()
        assert environment.joinpath("share/gnomon/skills/use-gnomon/references/legacy-workflows.md").is_file()

        if example_wheel is not None:
            run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(example_wheel)])
            run([str(python), "-m", "pip", "check"])
            example = Path(__file__).resolve().parents[1] / "examples/provider_plugin"
            example_output = root / "provider-example"
            receipt = json.loads(run([str(python), str(example / "walkthrough.py"), "run",
                                      "--output-dir", str(example_output)], cwd=root))
            assert receipt["mae_history"] == [None, 1, 1.5, 4, 1.5, 1.5]
            assert receipt["backup_restored"] and not receipt["action_executed"]
            example_config = example_output / "providers.toml"
            plugin_cli = json.loads(run([str(gnomon), "infer", "--provider", "example-last",
                                          "--providers-config", str(example_config), "--request",
                                          "@" + str(example_output / "request.json")], cwd=root))
            assert plugin_cli["result"]["point"] == [14, 14]
            plugin_mcp = mcp_exchange(gnomon, [{"id": 1, "method": "tools/call", "params": {
                "name": "gnomon_forecast", "arguments": {"provider": "example-mean", "request": {
                    "history": [10, 14], "horizon": 1}}}}], cwd=root, providers_config=example_config)
            assert plugin_mcp[1]["result"]["structuredContent"]["result"]["point"] == [12]

        run([str(python), "-c", """
import sys
from gnomon import InferenceEngine, ForecastRequest, ForecastResult, TemporalLedger, EphemerisProvider
assert 'gnomon.runtime' not in sys.modules
ledger = TemporalLedger(sys.argv[1])
engine = InferenceEngine(ledger=ledger)
def predict(request):
    return ForecastResult((request.history[-1],), timestamps=request.future_timestamps,
                          series_id=request.series_id, unit=request.unit)
engine.register('user-callable', predict, revision='smoke-v1')
run = engine.forecast('user-callable', ForecastRequest((1, 2), 1, series_id='shop',
                     future_timestamps=('2025-01-03T00:00:00Z',)))
ledger.append_actual(series_id='shop', valid_time='2025-01-03T00:00:00Z', value=3,
                     source_available_at='2025-01-04T00:00:00Z')
assert ledger.evaluate(run.execution_id)['mae'] == 1
assert not run.action_authorized
assert EphemerisProvider('https://example.invalid').revision is None
assert EphemerisProvider('https://example.invalid').name == 'ephemeris/route'
""", str(root / "ledger.db")], cwd=root)

        providers = root / "providers.toml"
        providers.write_text('schema_version=1\nledger_path="session-ledger.db"\n', encoding="utf-8")
        request = {"history": [1, 2], "horizon": 1, "series_id": "shop",
                   "future_timestamps": ["2025-01-03T00:00:00Z"]}
        inferred = json.loads(run([str(gnomon), "infer", "--providers-config", str(providers),
                                   "--provider", "last_value", "--request", json.dumps(request)], cwd=root))
        assert inferred["recorded"] and inferred["evidence"] == "inference_only"
        assert inferred["result"]["point"] == [2]
        execution_mcp = mcp_exchange(gnomon, [
            {"id": 1, "method": "tools/list"},
            {"id": 2, "method": "tools/call", "params": {"name": "gnomon_forecast",
                "arguments": {"provider": "last_value", "request": request}}},
            {"id": 3, "method": "tools/call", "params": {"name": "gnomon_ledger",
                "arguments": {"operation": "execution", "execution_id": inferred["execution_id"]}}},
        ], cwd=root, providers_config=providers)
        assert {t["name"] for t in execution_mcp[1]["result"]["tools"]} == {
            "gnomon_capabilities", "gnomon_forecast", "gnomon_ledger", "gnomon_inspect", "gnomon_describe", "gnomon_evaluate", "gnomon_route", "gnomon_read"}
        assert execution_mcp[2]["result"]["structuredContent"]["result"]["point"] == [2]
        assert execution_mcp[3]["result"]["structuredContent"]["result"]["request"]["history"] == [1, 2]

        capabilities = json.loads(run([str(gnomon), "capabilities"], cwd=root))
        assert capabilities["product_contract"]["offline_builtin_runtime"] is True
        assert capabilities["product_contract"]["default_mcp_profile"] == "execution"
        control_runs, treatment_runs = root / "control.jsonl", root / "treatment.jsonl"
        control_runs.write_text(json.dumps({"task_id": "task", "success": True, "cost_usd": 1}) + "\n", encoding="utf-8")
        treatment_runs.write_text(json.dumps({"task_id": "task", "success": False,
            "row_abstained": "cap:tokens", "cost_usd": 8}) + "\n", encoding="utf-8")
        compared = json.loads(run([str(gnomon), "eval", "compare", "--baseline", str(control_runs),
                                   "--treatment", str(treatment_runs)], cwd=root))
        assert compared["schema_version"] == "0.2"
        assert compared["absolute_success_uplift"] == -1
        assert compared["treatment"]["resources"]["cost_usd"]["total"] == 8
        assert capabilities["temporal"]["enabled"] is False
        temporal_args = {"operation": "shift", "value": "2024-01-31", "amount": 1,
                         "unit": "months", "mode": "calendar", "invalid_date": "clamp"}
        temporal_cli = json.loads(run([str(gnomon), "temporal", "--arguments", json.dumps(temporal_args)], cwd=root))
        assert temporal_cli["result"]["date"] == "2024-02-29"
        temporal_config = root / "temporal.toml"
        temporal_config.write_text("enable_temporal=true\n", encoding="utf-8")
        temporal_mcp = mcp_exchange(gnomon, [
            {"id": 1, "method": "tools/list"},
            {"id": 2, "method": "tools/call", "params": {"name": "gnomon_temporal", "arguments": temporal_args}},
        ], cwd=root, providers_config=temporal_config)
        assert len(temporal_mcp[1]["result"]["tools"]) == 7
        assert temporal_mcp[2]["result"]["structuredContent"] == temporal_cli
        temporal_python = json.loads(run([str(python), "-c",
            "import json,sys; from gnomon import temporal_operation; print(json.dumps(temporal_operation(**json.loads(sys.argv[1]))))",
            json.dumps(temporal_args)], cwd=root))
        assert temporal_python == temporal_cli
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
        direct = json.loads(run([str(gnomon), "infer", "--input", str(source),
                                 "--provider", "last_value", "--horizon", "2"], cwd=root))
        assert direct["result"]["point"] == [161, 161]
        assert direct["input"]["snapshot"]["known_time_assumed"] is True
        study = json.loads(run([str(gnomon), "evaluate", "--providers-config", str(providers), "--arguments",
                               json.dumps({"data": {"input": str(source)}, "candidates": ["historical_mean"],
                                           "baseline": "last_value", "horizon": 2, "folds": 2,
                                           "budget": {"max_calls": 4}})], cwd=root))
        assert study["status"] == "complete" and study["usage"]["provider_calls"] == 4
        assert study["folds"][0]["request"]["history"]
        saved_study = json.loads(run([str(gnomon), "ledger", "--providers-config", str(providers), "--arguments",
                                     json.dumps({"operation": "study", "study_id": study["study_id"]})], cwd=root))["result"]
        assert saved_study["cohort_id"] == study["cohort_id"]
        assert saved_study["folds"][0]["request"]["history"]
        aware_source = root / "aware-series.csv"
        aware_source.write_text(
            "timestamp,value\n" + "\n".join(
                f"{start + timedelta(days=index)}T00:00:00+00:00,{100 + index + (index % 7)}"
                for index in range(56)
            ) + "\n", encoding="utf-8",
        )
        routing_study = json.loads(run([
            str(gnomon), "evaluate", "--providers-config", str(providers), "--arguments",
            json.dumps({"data": {"input": str(aware_source)}, "candidates": ["historical_mean"],
                        "baseline": "last_value", "horizon": 2, "folds": 3}),
        ], cwd=root))
        routed = json.loads(run([
            str(gnomon), "route", "--providers-config", str(providers), "--arguments",
            json.dumps({"data": {"input": str(aware_source)}, "study_id": routing_study["study_id"],
                        "candidates": ["historical_mean"], "baseline": "last_value", "horizon": 2,
                        "source_as_of": "2026-02-25T00:00:00Z",
                        "recorded_as_of": datetime.now(timezone.utc).isoformat()}),
        ], cwd=root))
        assert routed["basis"] == "cutoff_bound_matched_study"
        assert routed["provider_calls"] == 0 and routed["action_authorized"] is False
        rescore = json.loads(run([
            str(gnomon), "ledger", "--providers-config", str(providers), "--arguments",
            json.dumps({"operation": "study", "study_id": routed["rescore_study_id"]}),
        ], cwd=root))["result"]
        assert rescore["derived_from"] == routing_study["study_id"]
        assert rescore["usage"]["provider_calls"] == 0
        run([str(python), "-c", """
import sys
from gnomon import GnomonSession
with GnomonSession.from_config() as session:
    ref = session.call('gnomon_inspect', {'input':sys.argv[1]})['data_ref']
    result = session.call('gnomon_describe', {'data_ref':ref, 'statistic':'latest'})
    assert result['value'] == 161 and result['action_authorized'] is False
for module in ('pipeline', 'evaluation', 'context', 'toolspec'):
    assert 'gnomon.' + module not in sys.modules, module
""", str(source)], cwd=root)
        forecast = json.loads(run([
            str(gnomon), "forecast", str(source), "--time", "timestamp",
            "--target", "value", "--frequency", "D", "--horizon", "7",
            "--output", str(root / "artifacts"),
        ], cwd=root))
        artifact = Path(forecast["artifact_path"])
        assert artifact.joinpath("artifact.json").is_file()
        assert artifact.joinpath("history.json").is_file()
        assert artifact.joinpath("summary.md").is_file()
        assert artifact.joinpath("report.html").is_file()
        assert forecast["tier_floor"] in {
            "supported", "conditionally_supported", "best_effort",
        }
        summary = artifact.joinpath("summary.md").read_text(encoding="utf-8")
        report = artifact.joinpath("report.html").read_text(encoding="utf-8")
        assert f"- Support: {forecast['tier_floor']}" in summary
        assert f"<dt>Support</dt><dd>{forecast['tier_floor']}</dd>" in report

        short_source = root / "short-series.csv"
        short_source.write_text(
            "timestamp,value\n" + "\n".join(
                f"{start + timedelta(days=index)},{100 + index}"
                for index in range(12)
            ) + "\n",
            encoding="utf-8",
        )
        weak = json.loads(run([
            str(gnomon), "forecast", str(short_source), "--time", "timestamp",
            "--target", "value", "--frequency", "D", "--horizon", "14",
            "--output", str(root / "weak-artifacts"),
        ], cwd=root))
        assert weak["tier_floor"] == "best_effort"
        assert "High-confidence" not in weak["headline"]
        weak_artifact = Path(weak["artifact_path"])
        weak_summary = weak_artifact.joinpath("summary.md").read_text(
            encoding="utf-8")
        weak_report = weak_artifact.joinpath("report.html").read_text(
            encoding="utf-8")
        assert "- Support: best_effort" in weak_summary
        assert "<dt>Support</dt><dd>best_effort</dd>" in weak_report

        default_mcp = mcp_exchange(gnomon, [
            {"id": 1, "method": "tools/list"},
            {"id": 2, "method": "tools/call", "params": {"name": "gnomon_forecast",
                "arguments": {"provider": "last_value", "request": request}}},
        ], cwd=root)
        assert len(default_mcp[1]["result"]["tools"]) == 6
        assert default_mcp[2]["result"]["structuredContent"]["result"]["point"] == [2]
        run([str(python), "-c", """
import json, subprocess, sys
process = subprocess.Popen([sys.argv[1], 'mcp', 'serve'], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
def call(i, name, arguments):
    process.stdin.write(json.dumps({'id':i, 'method':'tools/call',
        'params':{'name':name, 'arguments':arguments}}) + '\\n')
    process.stdin.flush()
    return json.loads(process.stdout.readline())['result']['structuredContent']
try:
    result = call(1, 'gnomon_forecast', {'provider':'last_value', 'request':{'history':[1,7], 'horizon':4000}})
    assert result['partial'] and result['status'] == 'result_available'
    page = call(2, 'gnomon_read', {'result_ref':result['result_ref'], 'pointer':'/result/point/3999'})
    assert json.loads(page['text']) == 7 and page['next_offset'] is None
    process.stdin.close()
    assert process.wait(timeout=10) == 0
finally:
    if process.poll() is None:
        process.kill()
        process.wait(timeout=10)
""", str(gnomon)], cwd=root)
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
                        "input": str(short_source), "time_column": "timestamp",
                        "target_column": "value", "frequency": "D",
                        "horizon": 14, "output_dir": str(root / "mcp-artifacts"),
                    },
                },
            },
        ], cwd=root, profile="core")
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
        assert structured["tier_floor"] == "best_effort"
        assert "High-confidence" not in structured["headline"]
        assert structured["artifact_id"] == structured["forecast_id"]

        print(json.dumps({
            "status": "passed",
            "runtime_version": capabilities["runtime_version"],
            "default_mcp_profile": capabilities["mcp_profile"]["active"],
            "structural_leakage_check": "passed",
            "forecast_tier_floor": forecast["tier_floor"],
            "weakest_tier_preservation": "passed",
            "packaged_mcp_journey": "passed",
            "packaged_callable_and_ledger": "passed",
            "packaged_execution_session": "passed",
            "packaged_example_plugin": "passed" if example_wheel is not None else "not_requested",
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

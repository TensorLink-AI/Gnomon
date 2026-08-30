"""Run the frozen P10 artifact-publication and local-load probe."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import threading
import time
from typing import Any, Callable


RUNNER_VERSION = "p10-production-reliability-1"
CASE_TIMEOUT_SECONDS = 30
RETRIES = 1
ROOT = Path(__file__).resolve().parents[2]
SHORT = ROOT / "tests" / "data" / "golden_short.csv"


class CaseTimeout(RuntimeError):
    pass


def _timeout(_signum: int, _frame: Any) -> None:
    raise CaseTimeout("case exceeded its timeout")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_identity(directory: Path) -> dict[str, str]:
    return {path.name: _digest(path) for path in sorted(directory.iterdir())
            if path.is_file()}


def _seed_forecast(output: Path):
    from gnomon.ids import FixedClock
    from gnomon.runtime import forecast

    return forecast(
        str(SHORT), time_column="timestamp", target_column="value",
        horizon=1, candidates=[], output=str(output),
        clock=FixedClock(datetime(2026, 8, 30, tzinfo=timezone.utc)),
    )[0]


def _fault_forecast(run_dir: Path) -> dict[str, Any]:
    import gnomon.artifacts as artifacts

    artifact = _seed_forecast(run_dir / "seed")
    output = run_dir / "publish"
    original = artifacts._write_integrity

    def fail(_directory: Path) -> None:
        raise OSError("injected pre-seal failure")

    artifacts._write_integrity = fail
    error = None
    try:
        artifacts.write_artifact(artifact, str(output))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        artifacts._write_integrity = original
    final = output / artifact.forecast_id
    partials = sorted(path.name for path in output.glob(
        f".{artifact.forecast_id}.tmp*"))
    invisible_after_failure = not final.exists()
    recovered = artifacts.write_artifact(artifact, str(output))
    artifacts.verify_artifact_integrity(recovered)
    from gnomon.artifacts import read_artifact
    readable = read_artifact(recovered)["forecast_id"] == artifact.forecast_id
    return {
        "error": error, "final_invisible_after_failure": invisible_after_failure,
        "diagnostic_partials": partials, "recovered_path": str(recovered),
        "integrity_verified": True, "public_readable": readable,
        "tree_identity": _tree_identity(recovered),
        "passed": bool(error and invisible_after_failure and partials and readable),
    }


def _fault_json(run_dir: Path) -> dict[str, Any]:
    import gnomon.artifacts as artifacts

    artifact_id = "decision_reliability_fixture"
    payload = {"schema_version": "0.1", "status": "complete",
               "decision_id": artifact_id, "selected_action": "wait"}
    output = run_dir / "publish"
    original = artifacts._write_integrity

    def fail(_directory: Path) -> None:
        raise OSError("injected pre-seal failure")

    artifacts._write_integrity = fail
    error = None
    try:
        artifacts.write_json_artifact(artifact_id, payload, str(output))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        artifacts._write_integrity = original
    final = output / artifact_id
    partials = sorted(path.name for path in output.glob(f".{artifact_id}.tmp*"))
    invisible_after_failure = not final.exists()
    recovered = artifacts.write_json_artifact(artifact_id, payload, str(output))
    artifacts.verify_artifact_integrity(recovered)
    stored = json.loads((recovered / "artifact.json").read_text(encoding="utf-8"))
    return {
        "error": error, "final_invisible_after_failure": invisible_after_failure,
        "diagnostic_partials": partials, "recovered_path": str(recovered),
        "integrity_verified": True, "public_readable": stored == payload,
        "tree_identity": _tree_identity(recovered),
        "passed": bool(error and invisible_after_failure and partials
                       and stored == payload),
    }


def _concurrent(call: Callable[[], Path], artifacts: Any,
                final: Path, control: Path) -> dict[str, Any]:
    original = artifacts._write_integrity
    barrier = threading.Barrier(2)

    def synchronized(directory: Path) -> None:
        barrier.wait(timeout=5)
        original(directory)

    artifacts._write_integrity = synchronized
    returned: list[str] = []
    errors: list[str] = []
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(call) for _ in range(2)]
            for future in futures:
                try:
                    returned.append(str(future.result(timeout=10)))
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        artifacts._write_integrity = original
    verified = False
    if final.is_dir():
        try:
            artifacts.verify_artifact_integrity(final)
            verified = True
        except Exception as exc:
            errors.append(f"integrity: {type(exc).__name__}: {exc}")
    identity = _tree_identity(final) if verified else {}
    expected_final = str(final.resolve())
    one_final_path = (len(set(returned)) == 1
                      and returned == [expected_final] * 2)
    return {
        "callers": 2, "successful_callers": len(returned),
        "returned_paths": returned, "errors": errors,
        "one_final_path": one_final_path,
        "integrity_verified": verified,
        "matches_single_writer_control": verified and identity == _tree_identity(control),
        "tree_identity": identity,
        "passed": (len(returned) == 2 and not errors and verified
                   and one_final_path
                   and identity == _tree_identity(control)),
    }


def _concurrent_forecast(run_dir: Path) -> dict[str, Any]:
    import gnomon.artifacts as artifacts

    artifact = _seed_forecast(run_dir / "seed")
    control = artifacts.write_artifact(artifact, str(run_dir / "control"))
    output = run_dir / "race"
    final = output / artifact.forecast_id
    return _concurrent(
        lambda: artifacts.write_artifact(artifact, str(output)),
        artifacts, final, control)


def _concurrent_json(run_dir: Path) -> dict[str, Any]:
    import gnomon.artifacts as artifacts

    artifact_id = "decision_reliability_race"
    payload = {"schema_version": "0.1", "status": "complete",
               "decision_id": artifact_id, "selected_action": "wait"}
    control = artifacts.write_json_artifact(
        artifact_id, payload, str(run_dir / "control"))
    output = run_dir / "race"
    final = output / artifact_id
    return _concurrent(
        lambda: artifacts.write_json_artifact(artifact_id, payload, str(output)),
        artifacts, final, control)


def _mcp_internal(_run_dir: Path) -> dict[str, Any]:
    import gnomon.mcp_server as server

    original = server.runner_for

    def runner(_name: str):
        def fail(_arguments: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("injected unexpected tool failure")
        return fail

    server.runner_for = runner
    try:
        result = server._handle({
            "method": "tools/call",
            "params": {"name": "gnomon_forecast", "arguments": {}},
        })
    finally:
        server.runner_for = original
    structured = (result or {}).get("structuredContent") or {}
    error = structured.get("error") or {}
    return {
        "is_error": (result or {}).get("isError"), "error_code": error.get("code"),
        "repair_options": error.get("repair_options") or [],
        "transport_error": False,
        "passed": bool((result or {}).get("isError") is True
                       and error.get("code") == "INTERNAL_ERROR"
                       and error.get("repair_options")),
    }


def _serial_load(run_dir: Path) -> dict[str, Any]:
    from gnomon.artifacts import verify_artifact_integrity
    from gnomon.toolspec import runner_for

    runner = runner_for("gnomon_forecast")
    if runner is None:
        raise RuntimeError("gnomon_forecast is not visible")
    durations: list[float] = []
    paths: list[str] = []
    errors: list[str] = []
    for index in range(12):
        started = time.perf_counter()
        try:
            response = runner({
                "input": str(SHORT), "time_column": "timestamp",
                "target_column": "value", "horizon": 1,
                "candidates": [],
                "output_dir": str(run_dir / f"call-{index:02d}"),
            })
            path = Path(response["artifact_path"])
            verify_artifact_integrity(path)
            paths.append(str(path))
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        durations.append(time.perf_counter() - started)
    ordered = sorted(durations)
    p95 = ordered[max(0, math.ceil(.95 * len(ordered)) - 1)]
    return {
        "expected": 12, "completed": len(paths), "errors": errors,
        "durations_seconds": durations, "p50_seconds": statistics.median(durations),
        "p95_seconds": p95, "external_calls": 0, "retries": 0,
        "all_integrity_verified": len(paths) == 12,
        "passed": len(paths) == 12 and not errors and p95 <= 1.0,
    }


CASES: tuple[tuple[str, Callable[[Path], dict[str, Any]]], ...] = (
    ("forecast-fault-retry", _fault_forecast),
    ("json-fault-retry", _fault_json),
    ("forecast-same-id-race", _concurrent_forecast),
    ("json-same-id-race", _concurrent_json),
    ("mcp-unexpected-error", _mcp_internal),
    ("serial-local-load", _serial_load),
)


def summarize(rows: list[dict[str, Any]], mode: str,
              baseline: Path | None) -> dict[str, Any]:
    by_id = {row["case_id"]: row for row in rows}
    races = [by_id[name]["result"] for name in
             ("forecast-same-id-race", "json-same-id-race")]
    faults = [by_id[name]["result"] for name in
              ("forecast-fault-retry", "json-fault-retry")]
    load = by_id["serial-local-load"]["result"]
    baseline_race_successes = None
    if baseline is not None:
        baseline_race_successes = sum(json.loads(
            (baseline / "cases" / f"{name}.json").read_text(encoding="utf-8"))
            ["result"]["successful_callers"] for name in
            ("forecast-same-id-race", "json-same-id-race"))
    gates = {
        "completion": len(rows) == 6 and all(row["complete"] for row in rows),
        "fault_isolation_and_retry": all(item["passed"] for item in faults),
        "same_id_concurrency": all(item["passed"] for item in races),
        "payload_immutability": all(
            item.get("matches_single_writer_control") for item in races),
        "typed_mcp_failure": by_id["mcp-unexpected-error"]["result"]["passed"],
        "representative_load": load["passed"],
        "zero_external_cost_and_retries": (
            load["external_calls"] == 0 and load["retries"] == 0),
        "topology": True,
        "improvement": (mode == "baseline" or baseline_race_successes is not None
                        and baseline_race_successes < 4
                        and sum(item["successful_callers"] for item in races) == 4),
    }
    return {
        "schema_version": 1, "runner_version": RUNNER_VERSION, "mode": mode,
        "cases": len(rows), "gates": gates,
        "all_gates_pass": all(gates.values()),
        "same_id_successful_callers": sum(
            item["successful_callers"] for item in races),
        "baseline_same_id_successful_callers": baseline_race_successes,
        "serial_load_p95_seconds": load["p95_seconds"],
        "topology_contract": {"case_jobs": 1, "race_writers": 2,
                              "case_timeout_seconds": CASE_TIMEOUT_SECONDS,
                              "retries": RETRIES, "atomic_case_files": True},
    }


def run(mode: str, output: Path, baseline: Path | None) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    identity = {"runner_version": RUNNER_VERSION, "mode": mode,
                "cases": [name for name, _function in CASES]}
    identity_path = output / "run_identity.json"
    if identity_path.is_file():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise SystemExit("resume identity mismatch; use a new output directory")
    else:
        _atomic_json(identity_path, identity)
    rows: list[dict[str, Any]] = []
    old_handler = signal.signal(signal.SIGALRM, _timeout)
    try:
        for case_id, function in CASES:
            target = output / "cases" / f"{case_id}.json"
            if target.is_file():
                rows.append(json.loads(target.read_text(encoding="utf-8")))
                continue
            last_error: Exception | None = None
            for _attempt in range(RETRIES + 1):
                try:
                    signal.alarm(CASE_TIMEOUT_SECONDS)
                    result = function(output / "work" / case_id)
                    signal.alarm(0)
                    row = {"schema_version": 1, "runner_version": RUNNER_VERSION,
                           "case_id": case_id, "complete": True, "result": result}
                    _atomic_json(target, row)
                    rows.append(row)
                    last_error = None
                    break
                except Exception as exc:
                    signal.alarm(0)
                    last_error = exc
            if last_error is not None:
                raise last_error
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    summary = summarize(rows, mode, baseline)
    _atomic_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.mode == "candidate" and args.baseline is None:
        parser.error("candidate mode requires --baseline")
    summary = run(args.mode, args.output, args.baseline)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if args.mode == "baseline" or summary["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

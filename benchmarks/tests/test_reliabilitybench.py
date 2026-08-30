from __future__ import annotations

from benchmarks.reliabilitybench.run import CASES, summarize


def test_frozen_reliability_cases_have_six_serial_denominators() -> None:
    assert [name for name, _function in CASES] == [
        "forecast-fault-retry", "json-fault-retry",
        "forecast-same-id-race", "json-same-id-race",
        "mcp-unexpected-error", "serial-local-load",
    ]


def test_summary_fails_closed_on_one_race_caller() -> None:
    rows = []
    for name, _function in CASES:
        if "fault" in name:
            result = {"passed": True}
        elif "race" in name:
            result = {"passed": name.startswith("json"),
                      "successful_callers": 2 if name.startswith("json") else 1,
                      "matches_single_writer_control": name.startswith("json")}
        elif name == "mcp-unexpected-error":
            result = {"passed": True}
        else:
            result = {"passed": True, "p95_seconds": .1,
                      "external_calls": 0, "retries": 0}
        rows.append({"case_id": name, "complete": True, "result": result})
    summary = summarize(rows, "baseline", None)
    assert summary["gates"]["same_id_concurrency"] is False
    assert summary["gates"]["payload_immutability"] is False
    assert summary["all_gates_pass"] is False

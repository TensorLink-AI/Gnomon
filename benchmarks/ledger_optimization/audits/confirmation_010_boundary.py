import json
from pathlib import Path
from collections import Counter, defaultdict

root = Path("/root/Gnomon")
run = root / "results/ledger-optimization/confirmation-agent-010"
paths = sorted(run.glob("[0-9]*.json"))
if len(paths) != 3744 or list(run.glob("*.harness-error.json")):
    raise ValueError("Full confirmation required")
arms = defaultdict(
    lambda: {
        "decisions": 0,
        "tool_calls": Counter(),
        "unsuccessful_tool_calls": Counter(),
        "resolution_statuses": Counter(),
        "forecast_attempts": 0,
        "successful_executions": 0,
        "api_errors": 0,
        "fallbacks": 0,
    }
)
issues = []
for path in paths:
    r = json.loads(path.read_text())
    stats = arms[r["arm"]]
    stats["decisions"] += 1
    for field in ("forecast_attempts", "successful_executions"):
        stats[field] += r[field]
    stats["api_errors"] += len(r["api_errors"])
    stats["fallbacks"] += r["fallback_used"]
    stats["resolution_statuses"][r["resolution"]["status"]] += 1
    call_names = {}
    successes = 0
    for m in r["transcript"]:
        if m["role"] == "assistant":
            for call in m.get("tool_calls") or []:
                call_names[call["id"]] = call["function"]["name"]
        elif m["role"] == "tool":
            name = call_names[m["tool_call_id"]]
            payload = json.loads(m["content"])
            stats["tool_calls"][name] += 1
            ok = (
                (payload.get("status") == "ok" and bool(payload.get("completion")))
                if name == "gnomon_forecast"
                else bool(payload.get("resolved"))
                if name == "select_forecast"
                else False
            )
            if name == "gnomon_forecast" and ok:
                successes += 1
            if not ok:
                stats["unsuccessful_tool_calls"][name] += 1
                issues.append(
                    {
                        "path": str(path),
                        "tool_call_id": m["tool_call_id"],
                        "name": name,
                        "status": payload.get("status"),
                        "cause": payload.get("cause"),
                        "error_code": payload.get("error", {}).get("code")
                        if isinstance(payload.get("error"), dict)
                        else None,
                    }
                )
    assert successes == r["successful_executions"]
    assert r["forecast_attempts"] <= 3 and r["api_calls"] <= 6
out = {
    "scope": "confirmation",
    "decisions": len(paths),
    "arms": dict(arms),
    "issues": issues,
    "all_successful_forecast_counts_reconciled": True,
    "execution_budgets_respected": True,
    "target_established": False,
    "limitation": "Describes observed execution and final-selection behavior. It does not establish forecasting benefit or universal reliability.",
}
p = (
    root
    / "benchmarks/ledger_optimization/evidence/confirmation-agent-010-boundary-audit.json"
)
p.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "issues"}, indent=2))

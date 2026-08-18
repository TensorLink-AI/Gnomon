"""A deterministic, model-free control arm for the trading corpus.

It carries the last print forward, rolls the venue calendar to find the next
session, ranks by return rather than level, and abstains when the tape says
the history is not comparable with itself.  There is no forecasting model and
no network here on purpose: it is the floor a real arm has to beat, and it
proves the corpus is answerable from the evidence each case actually ships
(the declared calendar, basis, corporate actions, and halts) rather than from
private knowledge of the generator.

    python -m benchmarks.workflow.run_workflow \\
      --cases benchmarks/workflow/cases/trading_smoke.jsonl \\
      --arm-command "python -m benchmarks.workflow.trading_baseline" \\
      --arm trading-carry-forward --output-dir results/workflow-trading/baseline
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, timedelta
from typing import Any

RANKING_RULE = "largest absolute final-session return in bps"


def _columns(case: dict[str, Any]) -> dict[str, list[Any]]:
    columns = dict((case.get("available_at_cutoff") or {}).get("columns") or {})
    columns.pop("timestamp", None)
    return {name: list(values) for name, values in columns.items()}


def _next_session(payload: dict[str, Any]) -> str | None:
    """Roll the declared calendar forward one session from the cutoff."""
    calendar = payload.get("session_calendar") or {}
    as_of = payload.get("as_of_session")
    if not as_of or not calendar:
        return None
    holidays = set(calendar.get("holidays") or ())
    skips_weekends = int(calendar.get("sessions_per_week", 7)) == 5
    current = date.fromisoformat(str(as_of))
    for _ in range(30):
        current += timedelta(days=1)
        iso = current.isoformat()
        if iso in holidays or (skips_weekends and current.weekday() >= 5):
            continue
        return iso
    return None


def _fingerprint(numbers: dict[str, float], case_id: str) -> str:
    digest = hashlib.sha256(json.dumps({"case": case_id, "numbers": numbers},
                                       sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:32]


def _ambiguity(payload: dict[str, Any], columns: dict[str, list[Any]]) -> str | None:
    if payload.get("corporate_actions"):
        return "corporate_action"
    if payload.get("halts"):
        return "halt"
    if len(columns) > 1:
        return "venue"
    return None


def _answer(case: dict[str, Any]) -> dict[str, Any]:
    payload = case.get("available_at_cutoff") or {}
    columns = _columns(case)
    schema = case.get("answer_schema") or {}
    wanted = set(schema.get("facts") or ())
    observation: dict[str, Any] = {
        "case_id": case["id"], "status": "answered", "support": "degraded",
        "numbers": {}, "choices": {}, "facts": {}, "engine_facts": {},
        "claims": ["carry-forward control arm; no model was selected"],
        "temporal_leakage": False, "tool_calls": 1, "cumulative_tokens": 0,
        "response_tokens": 0, "latency_seconds": 0.0,
    }
    if case.get("kind") == "messy":
        ambiguity = _ambiguity(payload, columns)
        if ambiguity is not None:
            observation.update(status="abstained", support="abstained",
                               facts={"ambiguity": ambiguity})
            return observation
    if case.get("kind") == "multiseries":
        returns = {name: abs(values[-1] / values[-2] - 1.0)
                   for name, values in columns.items()
                   if len(values) > 1 and values[-2]}
        if returns:
            observation["choices"]["most_notable"] = max(returns, key=returns.get)
        facts = {"ranking_rule": RANKING_RULE, "remainder_preserved": True,
                 "price_basis": payload.get("price_basis")}
        observation["facts"] = {key: value for key, value in facts.items()
                                if key in wanted}
        observation["engine_facts"] = dict(facts)
        observation["support"] = "supported"
        return observation
    series = next(iter(columns.values()), [])
    if series:
        observation["numbers"]["next"] = float(series[-1])
    facts = {"price_basis": payload.get("price_basis"),
             "as_of_session": payload.get("as_of_session"),
             "next_session": _next_session(payload),
             "tracking_requested": True}
    observation["facts"] = {key: value for key, value in facts.items()
                            if key in wanted and value is not None}
    if "session_cycle" in set(schema.get("choices") or ()):
        # No cycle is claimed: an unsupported guess would be worse than a
        # visible miss.
        observation["support"] = "best_effort"
    return observation


def _repair(case: dict[str, Any]) -> dict[str, Any]:
    payload = case.get("available_at_cutoff") or {}
    columns = _columns(case)
    revealed = case.get("revealed") or {}
    wanted = set((case.get("answer_schema") or {}).get("facts") or ())
    observation: dict[str, Any] = {
        "case_id": case["id"], "status": "answered", "support": "degraded",
        "numbers": {}, "choices": {}, "facts": {},
        "claims": ["carry-forward control arm; resolution applied as revealed"],
        "temporal_leakage": False, "tool_calls": 1, "cumulative_tokens": 0,
    }
    facts: dict[str, Any] = {}
    if revealed.get("resumption_print") is not None:
        observation["numbers"]["next"] = float(revealed["resumption_print"])
        facts["resolved_basis"] = "post_halt_reopen"
    elif revealed.get("venue") or revealed.get("target_column"):
        venue = str(revealed.get("venue") or revealed.get("target_column"))
        if venue in columns and columns[venue]:
            observation["numbers"]["next"] = float(columns[venue][-1])
        facts["resolved_venue"] = venue
    elif revealed.get("adjustment"):
        series = next(iter(columns.values()), [])
        if series:
            observation["numbers"]["next"] = float(series[-1])
        facts["resolved_basis"] = "split_adjusted"
    facts.setdefault("price_basis", payload.get("price_basis"))
    observation["facts"] = {key: value for key, value in facts.items()
                            if key in wanted}
    return observation


def main() -> int:
    case = json.loads(sys.stdin.read())
    stage = case.get("workflow_stage")
    observation = _repair(case) if stage == "repair" else _answer(case)
    if observation["numbers"] or observation["choices"]:
        fingerprint = _fingerprint(observation["numbers"], case["id"])
        observation.update(
            evaluated_fingerprint=fingerprint, published_fingerprint=fingerprint,
            publish_matches_evaluated=True, quote_matches=True,
            headline_numbers=dict(observation["numbers"]),
            artifact_numbers=dict(observation["numbers"]))
    print(json.dumps(observation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

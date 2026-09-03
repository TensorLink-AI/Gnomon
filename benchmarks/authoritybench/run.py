"""Exercise source-derived future-input authority through the MCP compiler.

Helpfulness is deliberately crossed with authority.  A useful assumption or
external forecast still cannot become a deterministic numeric override, while
an irrelevant binding schedule must remain correctly typed without affecting
the requested target.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from benchmarks.common.manifest import code_revision
from gnomon.context import event_applies
from gnomon.future_context import literal_input_authority
from gnomon.toolspec import _context_events_from


CASES = (
    ("observed", True, "Observed throughput was exactly 500 on 2026-02-10."),
    ("observed", False, "Observed throughput was exactly 500 on 2026-02-10."),
    ("forecast", True,
     "The vendor forecasts throughput will be exactly 500 on 2026-02-10."),
    ("forecast", False,
     "The vendor forecasts throughput will be exactly 500 on 2026-02-10."),
    ("assumed", True, "Assume throughput is exactly 500 on 2026-02-10."),
    ("assumed", False, "Scenario: throughput is exactly 500 on 2026-02-10."),
    ("binding", True,
     "The binding schedule requires throughput to be fixed at 500 on 2026-02-10."),
    ("binding", False,
     "The binding schedule requires throughput to be fixed at 500 on 2026-02-10."),
)

REJECTION_CODES = {
    "observed": "observed_value_not_future_constraint",
    "forecast": "external_prediction_not_constraint",
    "assumed": "scenario_assumption_not_constraint",
}


def evaluate_cases() -> list[dict[str, Any]]:
    rows = []
    for authority, helpful, span in CASES:
        scope = ["throughput"] if helpful else ["unrelated_series"]
        arguments = {"target_column": "throughput", "context_events": [{
            "event_id": f"{authority}-{'helpful' if helpful else 'irrelevant'}",
            "claim_kind": "exact",
            "entity_scope": scope,
            "effective_start": "2026-02-10T00:00:00+00:00",
            "effective_end": "2026-02-10T23:59:59+00:00",
            "known_at": "2026-02-09T00:00:00+00:00",
            "source_span": span,
            "source_reference": "authoritybench-fixture",
        }]}
        events = _context_events_from(arguments)
        rejections = arguments.get("context_rejections") or []
        observed = literal_input_authority(span)
        applies = bool(len(events) == 1
                       and event_applies(events[0], "throughput"))
        if authority == "binding":
            disposition_correct = len(events) == 1 and applies is helpful
            escalated = bool(not helpful and applies)
            disposition = "applied" if applies else "scope_excluded"
        else:
            expected_code = REJECTION_CODES[authority]
            disposition_correct = bool(
                not events and len(rejections) == 1
                and rejections[0].get("reason_code") == expected_code)
            escalated = bool(events)
            disposition = (str(rejections[0].get("reason_code"))
                           if len(rejections) == 1 else "compiled")
        rows.append({
            "case_id": (
                f"authority:{authority}:"
                f"{'helpful' if helpful else 'irrelevant'}"),
            "expected_authority": authority,
            "observed_authority": observed,
            "helpful_to_target": helpful,
            "disposition": disposition,
            "classification_correct": bool(
                observed == authority and disposition_correct),
            "authority_escalated": escalated,
        })
    return rows


def run(output: Path) -> dict[str, Any]:
    rows = evaluate_cases()
    payload = {
        "schema_version": 1,
        "benchmark": "future-input-authority",
        "evaluated_commit": code_revision(),
        "cases": len(rows),
        "correct": sum(row["classification_correct"] for row in rows),
        "authority_escalations": sum(row["authority_escalated"] for row in rows),
        "gates": {
            "complete": len(rows) == 8,
            "all_classifications_correct": all(
                row["classification_correct"] for row in rows),
            "no_authority_escalation": not any(
                row["authority_escalated"] for row in rows),
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    temporary.replace(output)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

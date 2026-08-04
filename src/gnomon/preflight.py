"""Admission preflight: will these context events earn influence here?

An agent's proposal used to die silently. The admission verdicts — span
does not parse, window overlaps history, claim contradicts the span —
were computed during the forecast and recorded in evidence the proposer
only saw after the run was spent, so the typical experience of the
context surface was a forecast that ignored what the agent said, with
the reason buried. This module runs the same admission checks against
the same data *before* any forecast, and returns every verdict beside
the grammar the span parser accepts, so a rejected proposal can be
repaired and resubmitted in one step instead of discovered post hoc.

The preflight is a prediction only where admission is deterministic:

- future-context events (``constraint:``/``override:`` carrying a
  ``source_span``) run the lane's actual admission function on the
  actual history, so the preflight verdict is the forecast-time verdict
  for the same data and flag state;
- caller claims (``constraint:`` carrying a ``claim``) run the actual
  collection checks likewise;
- every other event belongs to the fold-ablation gate, whose admission
  is a measurement on held-out folds. That cannot be predicted without
  running it, and the preflight says so instead of guessing.

Nothing here mutates state or writes an artifact: the preflight loads
the data exactly as ``forecast`` would (same repair setting, same
frequency resolution) and stops before any model runs.
"""

from __future__ import annotations

from typing import Any

from .context import ContextEvent

#: Span phrasings the deterministic parser accepts, one list per event
#: class. Returned by every preflight so a proposer can repair a
#: rejected span by imitation rather than by reading the regexes; a test
#: asserts every example actually parses, so the documentation cannot
#: drift from the grammar.
ACCEPTED_SPAN_EXAMPLES: dict[str, list[str]] = {
    "constraint": [
        "values stay between 0 and 340",
        "the value will not exceed 150",
        "bounded above by 2,000",
        "capped at 500",
        "will not drop below 10",
        "at least 25",
        "cannot be negative",
        "the maximal fan speed is 3000 rpm",
        "will not exceed 3 times the usual level",
    ],
    "override": [
        "the plant is offline for scheduled maintenance",
        "output reduced to 120 while the line is partially shut down",
        "production drops to zero during the strike",
        "4 times the number of usual withdrawals",
        "it rapidly and smoothly changes to 1593.0",
        "(2022-03-23 00:00:00, 0)",
    ],
}

PREFLIGHT_BASIS = (
    "The preflight runs the same admission code the forecast runs, on "
    "the same data: future-context and claim verdicts here are the "
    "verdicts the forecast will reach. Fold-ablation admission is a "
    "measurement on held-out folds and is only known by running it."
)

ABLATION_GATED_NOTE = (
    "This event carries neither a source_span nor a claim, so it goes "
    "to the fold-ablation gate: admission is measured by identical-fold "
    "ablation at forecast time and cannot be predicted in advance."
)


def _effective_outcomes(
    events: list[ContextEvent],
    assessment: Any,
    applied_claims: list[Any],
    rejected_claims: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One aggregated verdict per proposed event.

    The two deterministic lanes each record their own view (a claim
    event is 'rejected' by the future lane for having no span while the
    claims lane applies it); the per-event summary states the effective
    outcome so a proposer does not have to reconcile the lanes.
    """
    admitted_future = {item.event_id for item in assessment.admitted}
    rejected_future: dict[str, list[str]] = {}
    for item in assessment.rejected:
        rejected_future.setdefault(item["event_id"], []).append(item["reason"])
    applied_claim_ids = {claim.event_id for claim in applied_claims}
    rejected_claim_reasons: dict[str, list[str]] = {}
    for item in rejected_claims:
        rejected_claim_reasons.setdefault(
            item["event_id"], []).append(item["reason"])

    outcomes: list[dict[str, Any]] = []
    for event in events:
        entry: dict[str, Any] = {"event_id": event.event_id,
                                 "event_type": event.event_type}
        if event.event_id in admitted_future:
            entry["outcome"] = "would_influence"
            entry["basis"] = "future_context"
        elif event.event_id in applied_claim_ids:
            entry["outcome"] = "would_influence"
            entry["basis"] = "constraint_claim"
        elif event.event_id in rejected_future or \
                event.event_id in rejected_claim_reasons:
            entry["outcome"] = "rejected"
            entry["reasons"] = (
                rejected_claim_reasons.get(event.event_id, [])
                + rejected_future.get(event.event_id, [])
            )
        else:
            entry["outcome"] = "ablation_gated"
            entry["note"] = ABLATION_GATED_NOTE
        outcomes.append(entry)
    return outcomes


def preflight_context_events(
    input_path: str,
    *,
    time_column: str,
    target_column: str,
    horizon: int,
    context_events: list[ContextEvent],
    series_column: str | None = None,
    frequency: str | None = None,
    repair: str = "safe",
) -> dict[str, Any]:
    """Dry-run the deterministic admission checks for proposed events."""
    from .context import event_applies
    from .constraints import collect_claims
    from .data import load_observations
    from .future_context import assess_future_events
    from .temporal import detect_season, next_timestamp, validate_and_group

    observations, _, _ = load_observations(
        input_path, time_column, target_column, series_column, repair=repair,
    )
    groups, resolved_frequency, _ = validate_and_group(observations, frequency)

    series_payloads: list[dict[str, Any]] = []
    for name in sorted(groups):
        members = sorted(groups[name], key=lambda item: item.timestamp)
        values = [item.value for item in members]
        timestamps = [item.timestamp for item in members]
        season, _, _ = detect_season(values, resolved_frequency)
        future_timestamps = []
        cursor = timestamps[-1]
        for _ in range(horizon):
            cursor = next_timestamp(cursor, resolved_frequency)
            future_timestamps.append(cursor)

        applicable = [event for event in context_events
                      if event_applies(event, name)]
        assessment = assess_future_events(
            applicable, name, values, timestamps, future_timestamps, season,
        )
        claims, rejected_claims = collect_claims(
            applicable, name, values, timestamps,
        )
        series_payloads.append({
            "series": name,
            "events": _effective_outcomes(
                applicable, assessment, claims, rejected_claims),
            "future_context": assessment.to_public_dict(),
            "claims": {
                "would_apply": [
                    {"event_id": claim.event_id, "kind": claim.kind,
                     "value": claim.value}
                    for claim in claims
                ],
                "rejected": rejected_claims,
            },
        })

    return {
        "schema_version": "0.1",
        "status": "ok",
        "horizon": horizon,
        "frequency": resolved_frequency,
        "series": series_payloads,
        "accepted_span_grammar": ACCEPTED_SPAN_EXAMPLES,
        "basis": PREFLIGHT_BASIS,
    }

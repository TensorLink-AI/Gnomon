"""Reusable forecast-to-decision helpers for trading with a Gnomon ledger.

Copy or import these into a real strategy. They contain no trading policy or cost
model; supply validated ones. They never set or replace the ledger clock: recording
times come from whatever clock the ledger was opened with, and `record_trade_decision`
refuses a non-system clock unless told it is a fixture.

Keep one ledger per mode (for example backtest.db, paper.db, live.db). The mode is
stored as a `trading_mode` context label, and a ledger that already holds another
mode's decisions is refused. Live decisions need a user-approved promotion record
whose paper evidence is re-checked against the paper ledger at decision time.
"""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from gnomon.ids import SystemClock


MODES = ("backtest", "paper", "live")
_MODE_LABELS = ("FROM decisions d, json_each(d.payload_json, '$.inputs.context') c "
                "WHERE json_extract(c.value, '$.key') = 'trading_mode'")


def client_order_id(account, decision_id, leg):
    # A compact example: verify the destination venue's length/charset constraints.
    return "gno-" + sha256(f"{account}:{decision_id}:{leg}".encode()).hexdigest()[:28]


def _read(ledger, sql, parameters=()):
    uri = ledger.path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        return connection.execute(sql, parameters).fetchall()


def ledger_modes(ledger):
    """Trading modes already recorded in this ledger's decision summaries."""
    return {row[0] for row in _read(ledger, "SELECT DISTINCT json_extract(c.value, '$.value') " + _MODE_LABELS)}


def promotion_review(paper_ledger, *, source_as_of, recorded_as_of):
    """Count paper decisions, and those with complete outcomes, at explicit cutoffs.

    This is forecast-outcome evidence from the ledger. Net trading results belong to
    the order journal or engine; attach them to the promotion record for the user.
    """
    modes = ledger_modes(paper_ledger)
    if modes != {"paper"}:
        raise ValueError(f"Promotion evidence must come from a paper-only ledger, found {sorted(modes)}.")
    cutoff = datetime.fromisoformat(recorded_as_of)
    ids = sorted(decision_id for decision_id, recorded_at in _read(
        paper_ledger, "SELECT DISTINCT d.decision_id, d.recorded_at " + _MODE_LABELS)
        if datetime.fromisoformat(recorded_at) <= cutoff)
    complete = [i for i in ids if paper_ledger.review_decision(
        decision_id=i, source_as_of=source_as_of, recorded_as_of=recorded_as_of)["review_ready"]]
    return dict(paper_ledger=str(paper_ledger.path.resolve()), source_as_of=source_as_of,
                recorded_as_of=recorded_as_of, paper_decisions=len(ids),
                complete_decisions=len(complete), complete_decision_ids=complete)


def verify_promotion(record_path, paper_ledger):
    """Re-check a user-approved promotion record and return its authorization reference.

    The record is JSON with `review` (from `promotion_review`), `min_complete_decisions`,
    `criteria`, `approved_by` and timezone-aware `approved_at`. The user approves it.
    This validates record structure and cutoff-bound completion counts/IDs only;
    it does not authenticate the approver or assess performance criteria, scores,
    fills or P&L. Later revisions outside the saved cutoffs require a new review.
    """
    raw = Path(record_path).read_bytes()
    record = json.loads(raw)
    if not isinstance(record, dict):
        raise ValueError("Promotion record must be an object.")
    if not (isinstance(record.get("approved_by"), str) and record["approved_by"].strip()):
        raise ValueError("Promotion record has no approver.")
    if not isinstance(record.get("criteria"), str) or not record["criteria"].strip():
        raise ValueError("Promotion record requires nonempty criteria text.")
    minimum = record.get("min_complete_decisions")
    if type(minimum) is not int or minimum < 1:
        raise ValueError("min_complete_decisions must be a positive integer.")
    review = record.get("review")
    required = {"paper_ledger", "source_as_of", "recorded_as_of", "paper_decisions",
                "complete_decisions", "complete_decision_ids"}
    if not isinstance(review, dict) or set(review) != required:
        raise ValueError("Promotion review must be the complete promotion_review result.")
    if any(type(review[k]) is not int or review[k] < 0 for k in ("paper_decisions", "complete_decisions")):
        raise ValueError("Promotion review counts must be nonnegative integers.")
    if not isinstance(review["complete_decision_ids"], list) or not all(
            isinstance(v, str) and v for v in review["complete_decision_ids"]):
        raise ValueError("Promotion review requires decision IDs.")
    def timestamp(value, field):
        try:
            if not isinstance(value, str):
                raise ValueError()
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError()
            return parsed
        except (ValueError, TypeError):
            raise ValueError(f"{field} must be an ISO timestamp with an explicit timezone.") from None
    approved = timestamp(record.get("approved_at"), "approved_at")
    if approved > datetime.now(timezone.utc):
        raise ValueError("approved_at cannot be in the future.")
    if approved < max(timestamp(review[k], k) for k in ("source_as_of", "recorded_as_of")):
        raise ValueError("approved_at must be at or after the review cutoffs.")
    if review["paper_ledger"] != str(paper_ledger.path.resolve()):
        raise ValueError("Promotion record refers to a different paper ledger.")
    current = promotion_review(paper_ledger, source_as_of=review["source_as_of"],
                               recorded_as_of=review["recorded_as_of"])
    if current != review:
        raise ValueError("Promotion evidence no longer matches the paper ledger.")
    if current["complete_decisions"] < record["min_complete_decisions"]:
        raise ValueError(f"Paper minimum not met: {current['complete_decisions']} of "
                         f"{record['min_complete_decisions']} complete decisions.")
    return f"promotion:{Path(record_path).name}:sha256:{sha256(raw).hexdigest()[:16]}"


def record_trade_decision(ledger, journal_path, *, mode, forecast, action, account, leg, event_id,
                          policy_revision, rationale, assumptions, invalidation_conditions,
                          context=(), evidence_refs=(), promotion_record=None, paper_ledger=None,
                          require_system_clock=True):
    """Claim the event's journal, record the decision summary, then persist the intent.

    `mode` must be backtest, paper or live. Live requires `promotion_record` and the
    `paper_ledger` it cites; the verified record's reference is stored with the mode.
    The journal is created exclusively before the ledger write: an existing file means
    this event was already decided, so reconcile it instead of recording a new decision
    and order ID. An empty journal means the recording outcome is uncertain.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    if require_system_clock and not isinstance(ledger.clock, SystemClock):
        raise ValueError("Ledger must use the system clock; fixture clocks are for demos and tests.")
    authorization_ref = None
    if mode == "live":
        if promotion_record is None or paper_ledger is None:
            raise ValueError("Live decisions require a promotion_record and its paper_ledger.")
        authorization_ref = verify_promotion(promotion_record, paper_ledger)
    other = ledger_modes(ledger) - {mode}
    if other:
        raise ValueError(f"Ledger {ledger.path} already holds {sorted(other)} decisions; "
                         f"use a separate ledger for {mode}.")
    request = ledger.execution(forecast["execution_id"])["request"]
    origin = request.get("cutoff") or request["timestamps"][-1]
    mode_label = dict(key="trading_mode", value=mode, valid_from=origin,
                      valid_to=max(request["future_timestamps"]), source_available_at=origin,
                      source_ref=authorization_ref or f"operator-mode:{mode}")
    with open(journal_path, "x") as handle:
        summary = ledger.record_decision_summary(
            execution_id=forecast["execution_id"], rationale=rationale,
            assumptions=list(assumptions), invalidation_conditions=list(invalidation_conditions),
            context=[*context, mode_label], evidence_refs=list(evidence_refs),
        )
        decision_id = summary["decision_id"]
        intent = dict(
            mode=mode, event_id=event_id, decision_id=decision_id,
            execution_id=forecast["execution_id"], action=action,
            client_order_id=client_order_id(account, decision_id, leg),
            policy_revision=policy_revision, authorization_ref=authorization_ref,
            quantity=None, submitted=False,
        )
        # Quantity intentionally unset: an authorized executor sizes and submits.
        json.dump(intent, handle, indent=2)
    return intent

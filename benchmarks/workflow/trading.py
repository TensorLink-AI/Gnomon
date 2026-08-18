"""Deterministically generate the trading-specific Workflow Bench corpus.

The general corpus (:mod:`benchmarks.workflow.generate`) asks whether an arm
can turn a temporal question into one trusted answer.  Trading adds failure
modes that a generic series does not exercise, and every one of them is a
calendar or basis question rather than an alpha question:

* the next observation is the next *session* on a venue calendar, not
  tomorrow --- weekends, holidays, and 24x7 venues all disagree;
* a price is meaningless without its basis (unadjusted last trade,
  exchange settlement, 4pm mid, constant-maturity yield), so a published
  number that does not carry its basis is not a usable answer;
* corporate actions, venue dispersion, and trading halts make a series
  non-comparable with itself, and the governed behaviour is to abstain and
  ask rather than to extrapolate across the break;
* "most notable" means largest *return*, and the instrument with the
  largest price move is deliberately not the instrument with the largest
  return, so a level/return confusion is visible as a wrong answer;
* no answer may be dressed as advice: every case forbids the guarantee and
  recommendation vocabulary outright.

Nothing here requires market data, a vendor, or a network: cases are exact
arithmetic over synthetic instruments, so the oracle is a fact rather than a
judgement call.  The corpus is scored by the same runner, scorer, and gates as
the general corpus --- this is a corpus and an audit profile, not a fork.
"""

from __future__ import annotations

import argparse
import json
import secrets
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import random

from .provenance import atomic_write_text, corpus_sha256
from .schema import Case

GENERATOR_VERSION = "workflow-trading-v1"

#: Asset classes double as Workflow Bench domains.
ASSET_CLASSES = ("equities", "futures", "fx", "crypto", "rates")

CALENDARS = {"equities": "xnas", "futures": "cme", "fx": "fx5",
             "crypto": "24x7", "rates": "sifma"}

#: Session holidays inside the corpus windows.  These exist so that "the next
#: session" cannot be answered by adding a day.
HOLIDAYS = {
    "xnas": ("2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25"),
    "cme": ("2026-01-19", "2026-02-16", "2026-05-25"),
    "fx5": ("2026-01-01", "2026-04-03"),
    "sifma": ("2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25"),
    "24x7": (),
}

SESSIONS_PER_WEEK = {"xnas": 5, "cme": 5, "fx5": 5, "sifma": 5, "24x7": 7}

#: What a published number actually means for this asset class.
PRICE_BASIS = {
    "equities": "unadjusted_last_trade",
    "futures": "exchange_settlement",
    "fx": "wm_4pm_london_mid",
    "crypto": "venue_last_trade",
    "rates": "constant_maturity_yield",
}

INSTRUMENTS = {
    "equities": ("ACME.O", "BOREAL.N", "CRESTED.O"),
    "futures": ("ESH6", "NQH6", "CLG6"),
    "fx": ("EURUSD", "USDJPY", "GBPUSD"),
    "crypto": ("BTC-USD", "ETH-USD", "SOL-USD"),
    "rates": ("UST2Y", "UST10Y", "UST30Y"),
}

VENUES = {
    "equities": ("XNAS", "ARCX"),
    "futures": ("XCME", "XCEC"),
    "fx": ("EBS", "REUT"),
    "crypto": ("COINBASE", "KRAKEN"),
    "rates": ("BGCANTOR", "TRADEWEB"),
}

TICK = {"equities": 0.01, "futures": 0.25, "fx": 0.0001,
        "crypto": 0.5, "rates": 0.001}

BASE_LEVEL = {"equities": 100.0, "futures": 5200.0, "fx": 1.09,
              "crypto": 64000.0, "rates": 4.25}

#: One "ordinary move" per asset class; every pattern is built from it so a
#: tolerance means the same thing in ticks across classes.
STEP = {"equities": 0.5, "futures": 5.0, "fx": 0.002,
        "crypto": 250.0, "rates": 0.02}

#: Vocabulary no arm may publish on any trading case, in any support tier.
#: These are not style preferences: each one converts a bounded statistical
#: statement into an unbounded promise or a recommendation.
REQUIRED_FORBIDDEN_CLAIMS = (
    "guaranteed", "risk-free", "cannot lose", "no downside",
    "financial advice", "sure thing", "beat the market",
)

AMBIGUITY_KINDS = ("venue", "corporate_action", "halt")

#: Window starts per case kind, chosen so holidays fall inside them.
WINDOW_START = {"synthetic": "2026-01-05", "frozen": "2026-02-09",
                "messy": "2026-03-16", "longitudinal": "2026-03-30",
                "multiseries": "2026-05-18"}


def _tick_round(value: float, tick: float) -> float:
    return round(round(value / tick) * tick, 10)


def sessions_from(start: str, count: int, calendar: str) -> list[str]:
    """The first ``count`` sessions on ``calendar`` starting at ``start``."""
    holidays = set(HOLIDAYS[calendar])
    skips_weekends = SESSIONS_PER_WEEK[calendar] == 5
    current = date.fromisoformat(start)
    out: list[str] = []
    while len(out) < count:
        iso = current.isoformat()
        if not (skips_weekends and current.weekday() >= 5) and iso not in holidays:
            out.append(iso)
        current += timedelta(days=1)
    return out


def next_session(after: str, calendar: str) -> str:
    following = (date.fromisoformat(after) + timedelta(days=1)).isoformat()
    return sessions_from(following, 1, calendar)[0]


def _calendar_block(asset: str, sessions: list[str]) -> dict[str, Any]:
    calendar = CALENDARS[asset]
    first, last = sessions[0], next_session(sessions[-1], calendar)
    return {
        "name": calendar,
        "sessions_per_week": SESSIONS_PER_WEEK[calendar],
        # Only holidays inside the visible window plus the next session, so
        # the calendar is evidence for this case rather than a hint about the
        # cutoff of another one.
        "holidays": [day for day in HOLIDAYS[calendar] if first <= day <= last],
        "timezone": "UTC",
    }


def _input(asset: str, sessions: list[str], columns: dict[str, list[Any]],
           **extra: Any) -> dict[str, Any]:
    """Point-in-time input: everything an arm may see, and nothing later."""
    instrument = extra.pop("instrument", INSTRUMENTS[asset][0])
    universe = extra.pop("instrument_universe", None)
    payload: dict[str, Any] = {"as_of_session": sessions[-1]}
    if instrument is not None:
        payload["instrument"] = instrument
    if universe:
        payload["instrument_universe"] = list(universe)
    payload.update({
        "venue": extra.pop("venue", VENUES[asset][0]),
        "price_basis": PRICE_BASIS[asset],
        "session_calendar": _calendar_block(asset, sessions),
        "columns": {"timestamp": list(sessions), **columns},
        **extra,
    })
    return payload


def _case(index: int, kind: str, asset: str, question: str,
          available: dict[str, Any], answer_schema: dict[str, list[str]],
          oracle: dict[str, Any], tags: list[str],
          stages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1, "id": f"trade-{kind}-{index:03d}", "kind": kind,
        "domain": asset, "question": question,
        "available_at_cutoff": available, "answer_schema": answer_schema,
        "oracle": {**oracle,
                   "forbidden_claims": list(REQUIRED_FORBIDDEN_CLAIMS)},
        "tags": [*tags, f"asset:{asset}", f"calendar:{CALENDARS[asset]}",
                 "point-in-time"],
        "stages": stages or [],
    }


def _synthetic(index: int, asset: str, rng: random.Random) -> dict[str, Any]:
    """An exactly repeating session cycle: no model choice, only calendar."""
    calendar, tick, step = CALENDARS[asset], TICK[asset], STEP[asset]
    period = rng.randint(3, 9)
    level = BASE_LEVEL[asset] + index * step
    cycle = [_tick_round(level + offset * step, tick)
             for offset in range(period)]
    repeats = rng.randint(3, 6)
    closes = cycle * repeats
    sessions = sessions_from(WINDOW_START["synthetic"], len(closes), calendar)
    upcoming = next_session(sessions[-1], calendar)
    weekly = SESSIONS_PER_WEEK[calendar]
    return _case(
        index, "synthetic", asset,
        "Forecast the close for the next scheduled session, name the repeating "
        "session cycle, and state that next session date on the venue calendar.",
        _input(asset, sessions, {"close": closes}),
        {"numbers": ["next"], "choices": ["session_cycle"],
         "facts": ["price_basis", "next_session"]},
        {"numbers": {"next": cycle[len(closes) % period]},
         "tolerances": {"next": tick},
         "choices": {"session_cycle": f"sessions-{period}"},
         **({"choice_aliases": {"session_cycle": ["weekly"]}}
            if period == weekly else {}),
         "required_facts": {"price_basis": PRICE_BASIS[asset],
                            "next_session": upcoming},
         "allowed_support": ["supported", "degraded"],
         "requires_publish_parity": True, "requires_quote_match": True},
        ["forecast", "known-truth", "calendar",
         f"mechanism:session-cycle-{period}"])


def _frozen(index: int, asset: str, variant: int,
            rng: random.Random) -> dict[str, Any]:
    """One session ahead, graded later against the realized print."""
    calendar, tick, step = CALENDARS[asset], TICK[asset], STEP[asset]
    level = BASE_LEVEL[asset] + index * step
    if variant == 0:
        slope = rng.choice([-3, -2, -1, 1, 2, 3]) * step
        length = rng.randint(10, 18)
        closes = [_tick_round(level + slope * k, tick) for k in range(length)]
        outcome, mechanism = _tick_round(level + slope * length, tick), "carry-drift"
    elif variant == 1:
        pattern = [_tick_round(level + rng.randint(-4, 4) * step, tick)
                   for _ in range(rng.randint(3, 6))]
        closes = pattern * rng.randint(3, 5)
        outcome, mechanism = pattern[0], "session-seasonal"
    elif variant == 2:
        closes = [_tick_round(level, tick)] * 12
        outcome, mechanism = _tick_round(level, tick), "pinned-level"
    else:
        closes = [_tick_round(level + (-1) ** k * step, tick) for k in range(12)]
        outcome, mechanism = _tick_round(level + step, tick), "bid-ask-bounce"
    sessions = sessions_from(WINDOW_START["frozen"], len(closes), calendar)
    upcoming = next_session(sessions[-1], calendar)
    return _case(
        index, "frozen", asset,
        "Forecast the next session's price on the stated basis; the realized "
        "print is revealed only after that session closes.",
        _input(asset, sessions, {"close": closes}, fixture_version="trade-v1"),
        {"numbers": ["next"], "choices": [],
         "facts": ["as_of_session", "price_basis"]},
        {"numbers": {"next": outcome}, "tolerances": {"next": 2 * step},
         "required_facts": {"as_of_session": sessions[-1],
                            "price_basis": PRICE_BASIS[asset]},
         "allowed_support": ["supported", "degraded", "best_effort"],
         "requires_publish_parity": True},
        ["forecast", "frozen", "decision", f"mechanism:{mechanism}"],
        [{"name": "outcome",
          "revealed": {"actual": outcome, "session": upcoming},
          "answer_schema": {"numbers": ["absolute_error"], "choices": [],
                            "facts": ["tracked_forecast_id"]},
          "oracle": {"numbers": {"absolute_error": 0.0},
                     "tolerances": {"absolute_error": 2 * step}}}])


def _messy(index: int, asset: str, ambiguity: str,
           rng: random.Random) -> dict[str, Any]:
    """A series that is not comparable with itself until someone says how."""
    calendar, tick, step = CALENDARS[asset], TICK[asset], STEP[asset]
    level = BASE_LEVEL[asset] + index * step
    sessions = sessions_from(WINDOW_START["messy"], 14, calendar)
    upcoming = next_session(sessions[-1], calendar)
    common = {
        "answer_schema": {"numbers": [], "choices": [], "facts": ["ambiguity"]},
        "allowed_support": ["abstained"],
    }
    if ambiguity == "venue":
        primary, secondary = VENUES[asset]
        streams = {
            primary: [_tick_round(level + rng.randrange(-2, 3) * step, tick)
                      for _ in sessions],
            secondary: [_tick_round(level + step + rng.randrange(-2, 3) * step, tick)
                        for _ in sessions],
        }
        target = rng.choice([primary, secondary])
        question = ("Forecast the next session close for this instrument. Two "
                    "venue price streams are supplied and the request names "
                    "neither.")
        available = _input(asset, sessions, streams, venue="multiple",
                           venue_streams=sorted(streams))
        answer = streams[target][-1]
        # The venue names are the column names, so the resolution is stated
        # both ways: a generic arm can bind the column, a trading arm still
        # has to publish which venue's basis it answered on.
        revealed = {"venue": target, "target_column": target, "horizon": 1}
        resolved_fact, mechanism = target, "venue-dispersion"
        resolved_key = "resolved_venue"
    elif ambiguity == "corporate_action":
        split_at = len(sessions) // 2
        pre = [_tick_round(level, tick)] * split_at
        post = [_tick_round(level / 2, tick)] * (len(sessions) - split_at)
        question = ("Forecast the next session close. The tape contains a "
                    "corporate action that has not been applied to the history.")
        available = _input(
            asset, sessions, {"close": pre + post},
            corporate_actions=[{"session": sessions[split_at], "type": "split",
                                "ratio": "2:1", "applied_to_history": False}])
        answer = post[-1]
        revealed = {"adjustment": "split", "factor": 2.0,
                    "apply_to": "pre_action_history"}
        resolved_fact, mechanism = "split_adjusted", "unadjusted-corporate-action"
        resolved_key = "resolved_basis"
    else:
        closes = [_tick_round(level + rng.randrange(-2, 3) * step, tick)
                  for _ in sessions]
        closes[-1] = closes[-2]
        reopen = _tick_round(closes[-2] - 4 * step, tick)
        question = ("Forecast the next session close. The instrument is halted "
                    "and the final print did not clear an auction.")
        available = _input(
            asset, sessions, {"close": closes},
            halts=[{"session": sessions[-1], "status": "halted",
                    "last_print_is_stale": True, "resumes": None}])
        answer = reopen
        revealed = {"resumption_print": reopen, "session": upcoming}
        resolved_fact, mechanism = "post_halt_reopen", "trading-halt"
        resolved_key = "resolved_basis"
    return _case(
        index, "messy", asset, question, available, common["answer_schema"],
        {"should_abstain": True, "required_facts": {"ambiguity": ambiguity},
         "allowed_support": common["allowed_support"], "requires_repair": True},
        ["repair", "ambiguity", f"ambiguity:{ambiguity}",
         f"mechanism:{mechanism}"],
        [{"name": "repair", "revealed": revealed,
          "answer_schema": {"numbers": ["next"], "choices": [],
                            "facts": [resolved_key]},
          "oracle": {"numbers": {"next": answer}, "tolerances": {"next": 2 * step},
                     "required_facts": {resolved_key: resolved_fact},
                     "allowed_support": ["degraded", "best_effort"]}}])


def _longitudinal(index: int, asset: str, variant: int) -> dict[str, Any]:
    """Publish a tracked prediction, then score it against the realized print."""
    calendar, tick, step = CALENDARS[asset], TICK[asset], STEP[asset]
    level = BASE_LEVEL[asset] + index * step
    patterns = (
        ([level - step, level, level + step], "carry-ladder-3"),
        ([level - 2 * step, level + 2 * step], "bid-ask-bounce"),
        ([level] * 4, "pinned-level"),
        ([level - 3 * step, level - step, level + step, level + 3 * step],
         "roll-cycle-4"),
    )
    pattern, mechanism = patterns[variant]
    pattern = [_tick_round(value, tick) for value in pattern]
    closes = (pattern * (21 // len(pattern) + 1))[:21]
    actual = pattern[21 % len(pattern)]
    sessions = sessions_from(WINDOW_START["longitudinal"], len(closes), calendar)
    upcoming = next_session(sessions[-1], calendar)
    return _case(
        index, "longitudinal", asset,
        "Forecast the next session close, persist it as a tracked prediction, "
        "then score that prediction when the realized print is revealed.",
        _input(asset, sessions, {"close": closes}),
        {"numbers": ["next"], "choices": [],
         "facts": ["tracking_requested", "price_basis"]},
        {"numbers": {"next": actual}, "tolerances": {"next": 2 * step},
         "required_facts": {"tracking_requested": True,
                            "price_basis": PRICE_BASIS[asset]},
         "allowed_support": ["supported", "degraded", "best_effort"],
         "requires_tracking": True, "requires_publish_parity": True},
        ["forecast", "tracking", f"mechanism:{mechanism}"],
        [{"name": "outcome",
          "revealed": {"actual": actual, "session": upcoming},
          "answer_schema": {"numbers": ["absolute_error"], "choices": [],
                            "facts": ["tracked_forecast_id"]},
          "oracle": {"numbers": {"absolute_error": 0.0},
                     "tolerances": {"absolute_error": 2 * step}}}])


def _multiseries(index: int, asset: str) -> dict[str, Any]:
    """Largest return, not largest price move: the level trap is deliberate."""
    calendar, tick = CALENDARS[asset], TICK[asset]
    sessions = sessions_from(WINDOW_START["multiseries"], 13, calendar)
    names = INSTRUMENTS[asset]
    winner = names[index % 3]
    decoy = names[(index + 1) % 3]
    quiet = names[(index + 2) % 3]
    base = BASE_LEVEL[asset] * (1 + index / 200)
    levels = {winner: base, decoy: base * 4, quiet: base / 2}
    moves_bps = {winner: 40 + (index % 20), decoy: 20, quiet: 8}
    streams: dict[str, list[float]] = {}
    for name in names:
        level = _tick_round(levels[name], tick)
        final = _tick_round(level * (1 + moves_bps[name] / 10_000), tick)
        streams[name] = [level] * (len(sessions) - 1) + [final]
    return _case(
        index, "multiseries", asset,
        "Return the instrument with the largest absolute final-session return "
        "in basis points -- not the largest price change -- and preserve the "
        "remaining instruments in the artifact.",
        _input(asset, sessions, streams, instrument=None,
               instrument_universe=names, quote_convention="bps_return"),
        {"numbers": [], "choices": ["most_notable"],
         "facts": ["ranking_rule", "remainder_preserved", "price_basis"]},
        {"choices": {"most_notable": winner},
         "required_facts": {
             "ranking_rule": "largest absolute final-session return in bps",
             "remainder_preserved": True,
             "price_basis": PRICE_BASIS[asset]},
         "engine_required_facts": {
             "ranking_rule": "largest absolute final-session return in bps",
             "remainder_preserved": True},
         "allowed_support": ["supported", "degraded"]},
        ["triage", "artifact", "known-truth", "mechanism:return-vs-level"])


def generate_trading_cases(seed: int = 20260818,
                           per_kind: int = 20) -> list[Case]:
    """Build the trading corpus: ``per_kind`` cases of each of the five kinds."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for index in range(per_kind):
        # Rotate the asset class per kind as well as per index, so even a
        # one-per-kind bundle spans five calendars instead of five equities.
        def asset(offset: int) -> str:
            return ASSET_CLASSES[(index + offset) % len(ASSET_CLASSES)]

        variant = (index // len(ASSET_CLASSES)) % 4
        rows.append(_synthetic(index, asset(0), rng))
        rows.append(_frozen(index, asset(1), variant, rng))
        rows.append(_messy(index, asset(2),
                           AMBIGUITY_KINDS[index % len(AMBIGUITY_KINDS)], rng))
        rows.append(_longitudinal(index, asset(3), variant))
        rows.append(_multiseries(index, asset(4)))
    cases = [Case.from_dict(row) for row in rows]
    if len({case.id for case in cases}) != len(cases):
        raise AssertionError("generator produced duplicate ids")
    return cases


# --- corpus audit -----------------------------------------------------------
#
# These are checks on the corpus, not on an arm.  They exist because a trading
# benchmark can be quietly wrong in ways a green run would never reveal: a bar
# dated after its own cutoff, an outcome that was already visible, a "notable"
# winner that is simply the most expensive instrument.

def _timestamps(case: Case) -> list[str]:
    columns = case.available_at_cutoff.get("columns") or {}
    return [str(value) for value in (columns.get("timestamp") or [])]


def _price_columns(case: Case) -> dict[str, list[float]]:
    columns = case.available_at_cutoff.get("columns") or {}
    return {name: [float(value) for value in values]
            for name, values in columns.items() if name != "timestamp"}


def _instruments(case: Case) -> set[str]:
    payload = case.available_at_cutoff
    universe = payload.get("instrument_universe")
    if universe:
        return {str(name) for name in universe}
    return {str(payload["instrument"])} if payload.get("instrument") else set()


def _ambiguity_kinds(cases: list[Case]) -> set[str]:
    return {str(case.oracle.required_facts["ambiguity"]) for case in cases
            if "ambiguity" in case.oracle.required_facts}


def _revealed_sessions(case: Case) -> list[str]:
    return [str((stage.get("revealed") or {})["session"]) for stage in case.stages
            if (stage.get("revealed") or {}).get("session")]


def _return_ranking_is_not_level_ranking(case: Case) -> bool:
    streams = _price_columns(case)
    if len(streams) < 2 or "most_notable" not in case.oracle.choices:
        return True
    changes = {name: abs(values[-1] - values[-2]) for name, values in streams.items()}
    returns = {name: abs(values[-1] / values[-2] - 1.0)
               for name, values in streams.items() if values[-2]}
    winner = case.oracle.choices["most_notable"]
    by_return = max(returns, key=returns.get)
    by_change = max(changes, key=changes.get)
    return by_return == winner and by_change != winner


def trading_audit_checks(cases: list[Case], strict: bool) -> dict[str, bool]:
    """Trading-specific corpus readiness, merged into the generic audit."""
    calendars = {str((case.available_at_cutoff.get("session_calendar") or {})
                     .get("name")) for case in cases}
    instruments = {name for case in cases for name in _instruments(case)}
    gapped = [case for case in cases
              if any((date.fromisoformat(later) - date.fromisoformat(earlier)).days > 1
                     for earlier, later in zip(_timestamps(case), _timestamps(case)[1:]))]
    multiseries = [case for case in cases if case.kind == "multiseries"]
    return {
        "trading_inputs_are_point_in_time": all(
            _timestamps(case)
            and all(stamp <= str(case.available_at_cutoff.get("as_of_session"))
                    for stamp in _timestamps(case))
            for case in cases),
        "revealed_outcomes_are_after_cutoff": all(
            all(session > str(case.available_at_cutoff.get("as_of_session"))
                for session in _revealed_sessions(case))
            for case in cases),
        "session_calendar_is_declared": all(
            {"name", "sessions_per_week"}
            <= set(case.available_at_cutoff.get("session_calendar") or {})
            for case in cases),
        "price_basis_is_declared": all(
            bool(case.available_at_cutoff.get("price_basis")) for case in cases),
        "answers_must_carry_price_basis": all(
            case.oracle.should_abstain or "price_basis" in case.oracle.required_facts
            for case in cases),
        "advice_vocabulary_is_forbidden": all(
            set(REQUIRED_FORBIDDEN_CLAIMS) <= set(case.oracle.forbidden_claims)
            for case in cases),
        "non_contiguous_sessions_present": bool(gapped),
        "return_ranking_is_not_level_ranking": all(
            _return_ranking_is_not_level_ranking(case) for case in multiseries),
        "calendar_diversity": not strict or len(calendars) >= 3,
        "continuous_and_scheduled_venues": not strict or "24x7" in calendars,
        "instrument_diversity": not strict or len(instruments) >= 8,
        "ambiguity_kind_coverage": (not strict or
                                    set(AMBIGUITY_KINDS) <= _ambiguity_kinds(cases)),
        "holiday_gap_coverage": not strict or len(gapped) >= 5,
    }


def trading_audit_detail(cases: list[Case]) -> dict[str, Any]:
    return {
        "asset_classes": sorted({case.domain for case in cases}),
        "calendars": sorted({str((case.available_at_cutoff.get("session_calendar")
                                  or {}).get("name")) for case in cases}),
        "instruments": sorted({name for case in cases
                               for name in _instruments(case)}),
        "price_bases": sorted({str(case.available_at_cutoff.get("price_basis"))
                               for case in cases}),
        "ambiguity_kinds": sorted(_ambiguity_kinds(cases)),
        "forbidden_claims": sorted(REQUIRED_FORBIDDEN_CLAIMS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, default=None)
    seeds.add_argument("--fresh", action="store_true",
                       help="draw a new seed, print it, and record it beside the corpus")
    parser.add_argument("--per-kind", type=int, default=20)
    args = parser.parse_args()
    seed = secrets.randbits(63) if args.fresh else (
        args.seed if args.seed is not None else 20260818)
    cases = generate_trading_cases(seed, args.per_kind)
    output = Path(args.output)
    atomic_write_text(output, "".join(
        json.dumps(asdict(case), sort_keys=True) + "\n" for case in cases))
    manifest = {
        "generator": GENERATOR_VERSION, "seed": seed,
        "fresh_seed": bool(args.fresh), "cases": len(cases),
        "corpus_sha256": corpus_sha256(cases),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_text(output.with_suffix(output.suffix + ".manifest.json"),
                      json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**manifest, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate fresh, matched ContextBench corpora and sealed truth files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .schema import Case

GENERATOR_VERSION = "contextbench-synthetic-v2"
EPOCH = datetime(2025, 1, 1, tzinfo=timezone.utc)
STEP = timedelta(hours=1)


def _stamp(index: int) -> str:
    return (EPOCH + index * STEP).isoformat()


def _base(rng: random.Random, length: int) -> list[float]:
    level = rng.uniform(40, 120)
    trend = rng.uniform(-0.015, 0.015)
    amplitude = rng.uniform(1.0, 4.0)
    phase = rng.uniform(0, 2 * math.pi)
    return [level + trend * index + amplitude * math.sin(
        2 * math.pi * index / 24 + phase) + rng.gauss(0, 0.35)
            for index in range(length)]


def _event(event_id: str, event_type: str, start: int, duration: int,
           *, sourced: bool, direction: str = "unknown") -> dict[str, Any]:
    return {
        "event_id": event_id, "event_type": event_type,
        "entity_scope": ["*"], "effective_start": _stamp(start),
        "effective_end": _stamp(start + duration - 1),
        "known_at": _stamp(0), "status": "confirmed", "confidence": 1.0,
        "attributes": {"soft_context": {
            "effect_family": "temporary_pulse", "direction": direction,
            "duration": "temporary", "entity_kind": "service",
            "normalized_entity": event_type,
            "delay_steps": [0, 0], "duration_steps": [duration, duration],
        }},
        **({"source": {"type": "calendar", "reference": "sealed-schedule"}}
           if sourced else {}),
        "created_by": "generator",
    }


def _apply(values: list[float], starts: list[int], duration: int,
           magnitude: float) -> None:
    for start in starts:
        for offset in range(duration):
            if start + offset < len(values):
                values[start + offset] += magnitude * (0.65 ** offset)


def _event_sentence(event: dict[str, Any], variant: int) -> str:
    """Render the same source fact with syntax outside the literal parser.

    These variants retain exact quoted times and scope but deliberately avoid
    the narrow ``X affects Y from A through B`` grammar. They test semantic
    compilation rather than timestamp copying; no oracle value enters them.
    """
    event_type = str(event["event_type"])
    start = str(event["effective_start"])
    end = str(event["effective_end"])
    templates = (
        f"Value series notice: {event_type} is booked between {start} and {end}.",
        f"For the value series, {event_type} begins {start}; it ends {end}.",
        f"Window {start} — {end}: {event_type} on the value series.",
        f"The value series has {event_type} scheduled, start {start}, finish {end}.",
    )
    return templates[variant % len(templates)]


def generate(seed: int, per_family: int = 20, *, history_length: int = 480,
             horizon: int = 12, narrative_style: str = "explicit"
             ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if narrative_style not in {"explicit", "naturalistic"}:
        raise ValueError("narrative_style must be explicit or naturalistic")
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    oracles: list[dict[str, Any]] = []
    total = history_length + horizon
    for family in ("irrelevant", "future_covariate", "repeated_event", "prior_only"):
        for index in range(per_family):
            case_id = f"ctx-{family}-{index:04d}"
            counterfactual = _base(rng, total)
            observed = list(counterfactual)
            events: list[dict[str, Any]] = []
            covariates: list[dict[str, Any]] = []
            mapping: list[dict[str, str]] = []
            influence = family in {"future_covariate", "repeated_event"}
            magnitude = rng.choice((-1.0, 1.0)) * rng.uniform(5.0, 11.0)
            direction = "increase" if magnitude > 0 else "decrease"
            duration = rng.randint(3, 6)
            onset = rng.randint(1, max(1, horizon - duration))

            if family in {"irrelevant", "repeated_event"}:
                # Begin before the first selection origin and recur on a
                # deliberately aperiodic 13–17 step schedule. Every fold can
                # estimate an effect and most validation windows contain an
                # occurrence, while a seasonal model cannot memorize one
                # fixed period. Sparse events would grade the gate on folds
                # where the two candidates are mathematically identical.
                starts = []
                cursor = rng.randint(8, 12)
                while cursor < history_length - duration:
                    starts.append(cursor)
                    cursor += rng.randint(13, 17)
                future_start = history_length + onset
                events = [_event(f"{case_id}-past-{n}", "deploy", start,
                                 duration, sourced=True, direction=direction)
                          for n, start in enumerate(starts)]
                events.append(_event(f"{case_id}-future", "deploy", future_start,
                                     duration, sourced=True, direction=direction))
                if family == "repeated_event":
                    _apply(observed, starts + [future_start], duration, magnitude)
            elif family == "future_covariate":
                # An irregular, known schedule defeats a pure seasonal model.
                active = [0.0] * total
                starts = sorted(rng.sample(range(60, history_length - 12), 18))
                future_start = history_length + onset
                for start in starts + [future_start]:
                    for offset in range(duration):
                        if start + offset < total:
                            active[start + offset] = 1.0
                            observed[start + offset] += magnitude
                covariates = [{"timestamp": _stamp(step), "known_at": _stamp(0),
                               "driver": active[step]}
                              for step in range(total)]
                mapping = [{"name": "driver", "type": "binary",
                            "availability": "future_known"}]
            else:  # prior_only: novel, unsourced, and therefore scenario-only.
                future_start = history_length + onset
                events = [_event(f"{case_id}-future", "novel_capacity_change",
                                 future_start, duration, sourced=False,
                                 direction=direction)]

            if events:
                event_lines = [
                    (_event_sentence(event, index + line_index)
                     if narrative_style == "naturalistic" else
                     f"{event['event_type']} affects the value series from "
                     f"{event['effective_start']} through {event['effective_end']}.")
                    for line_index, event in enumerate(events)
                ]
            else:
                event_lines = []
            case = {
                "schema_version": 1, "case_id": case_id, "family": family,
                "domain": "operations", "frequency": "h", "horizon": horizon,
                "history": [round(value, 8) for value in observed[:history_length]],
                "context_events": events, "covariates": covariates,
                "covariate_mapping": mapping,
                "narrative": (
                    "The following complete schedule was published and became "
                    "knowable at 2025-01-01T00:00:00+00:00. Every listed window "
                    "is part of that published schedule.\n" +
                    ("\n".join(event_lines) if events else
                     "The driver array is a future-known operational schedule.") +
                    "\nEstimate effects from history; this text states no numeric magnitude."
                ),
                "tags": ["matched", "fresh-synthetic", f"family:{family}",
                         f"narrative:{narrative_style}"],
            }
            Case.from_dict(case)
            cases.append(case)
            actual = observed[history_length:]
            cf = counterfactual[history_length:]
            oracles.append({
                "case_id": case_id,
                "actual": [round(value, 8) for value in actual],
                "counterfactual": [round(value, 8) for value in cf],
                "should_influence": influence,
                "expected_disposition": (
                    "applied" if influence else "scenario_only"
                ),
                "effect_direction": direction if influence else "none",
                "effect_magnitude": magnitude if influence else 0.0,
                "onset_step": onset if influence else None,
                "duration_steps": duration if influence else None,
            })
    return cases, oracles


def _render(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                   for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int)
    seeds.add_argument("--fresh", action="store_true")
    parser.add_argument("--per-family", type=int, default=20)
    parser.add_argument("--history-length", type=int, default=480)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--narrative-style",
                        choices=("explicit", "naturalistic"),
                        default="explicit")
    args = parser.parse_args()
    seed = secrets.randbits(63) if args.fresh else (
        args.seed if args.seed is not None else 20260814)
    cases, oracles = generate(seed, args.per_family,
                              history_length=args.history_length,
                              horizon=args.horizon,
                              narrative_style=args.narrative_style)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    case_text, oracle_text = _render(cases), _render(oracles)
    (output / "cases.jsonl").write_text(case_text, encoding="utf-8")
    (output / "oracle.jsonl").write_text(oracle_text, encoding="utf-8")
    manifest = {
        "generator": GENERATOR_VERSION, "schema_version": 1, "seed": seed,
        "fresh_seed": bool(args.fresh), "per_family": args.per_family,
        "history_length": args.history_length, "horizon": args.horizon,
        "narrative_style": args.narrative_style,
        "cases": len(cases),
        "cases_sha256": hashlib.sha256(case_text.encode()).hexdigest(),
        "oracle_sha256": hashlib.sha256(oracle_text.encode()).hexdigest(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "output_dir": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

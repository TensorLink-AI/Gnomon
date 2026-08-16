"""Generate ContextBench's sealed production-stress stratum.

The original corpus remains the clean, high-signal regression stratum.  This
generator varies the things a deployment cannot assume: effect strength,
truthfulness, timing, typed numeric claims, structural claims, competing
context lanes, and when evidence became knowable.
"""

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

from .generate import EPOCH, STEP
from .schema import Case

GENERATOR_VERSION = "contextbench-stress-v1"
NOISE_SIGMA = 1.0
HISTORY = 480
HORIZON = 12


FREQUENCY_STEPS = {
    "15min": timedelta(minutes=15), "h": timedelta(hours=1),
    "D": timedelta(days=1),
}


def _stamp(index: int, frequency: str = "h") -> str:
    return (EPOCH + index * FREQUENCY_STEPS[frequency]).isoformat()


def _base(rng: random.Random, length: int, *, trend: float | None = None,
          noise: float = NOISE_SIGMA, frequency: str = "h") -> list[float]:
    level = rng.uniform(60, 110)
    slope = rng.uniform(-0.02, 0.02) if trend is None else trend
    amplitude = rng.uniform(1.0, 4.0)
    phase = rng.uniform(0, 2 * math.pi)
    period = {"15min": 96, "h": 24, "D": 7}[frequency]
    return [level + slope * step + amplitude * math.sin(
        2 * math.pi * step / period + phase) + rng.gauss(0, noise)
            for step in range(length)]


def _apply(values: list[float], starts: list[int], duration: int,
           magnitude: float, *, delay: int = 0, shape: str = "decay") -> None:
    for start in starts:
        for offset in range(duration):
            point = start + delay + offset
            if point >= len(values):
                continue
            scale = {
                "flat": 1.0,
                "ramp": (offset + 1) / duration,
                "decay": 0.65 ** offset,
            }[shape]
            values[point] += magnitude * scale


def _event(event_id: str, event_type: str, start: int, duration: int, *,
           known_at: int = 0, direction: str = "unknown",
           delay: tuple[int, int] = (0, 0), sourced: bool = True,
           status: str = "confirmed",
           entity_scope: tuple[str, ...] = ("*",),
           frequency: str = "h",
           attributes: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(attributes or {})
    if not any(event_type.startswith(prefix) for prefix in
               ("constraint:", "override:", "structural:")):
        payload["soft_context"] = {
            "effect_family": "temporary_pulse", "direction": direction,
            "duration": "temporary", "entity_kind": "service",
            "normalized_entity": event_type,
            "delay_steps": list(delay),
            "duration_steps": [duration, duration],
        }
    return {
        "event_id": event_id, "event_type": event_type,
        "entity_scope": list(entity_scope),
        "effective_start": _stamp(start, frequency),
        "effective_end": _stamp(start + duration - 1, frequency),
        "known_at": _stamp(known_at, frequency), "status": status,
        "confidence": 1.0, "attributes": payload,
        **({"source": {"type": "calendar", "reference": "stress-fixture"}}
           if sourced else {}),
        "created_by": "stress-generator",
    }


def _schedule(rng: random.Random, duration: int) -> list[int]:
    starts: list[int] = []
    cursor = rng.randint(8, 12)
    while cursor < HISTORY - duration:
        starts.append(cursor)
        cursor += rng.randint(13, 17)
    return starts


def _narrative(events: list[dict[str, Any]], fallback: str = "",
               style: str = "schedule") -> str:
    lines: list[str] = []
    for event in events:
        span = (event.get("attributes") or {}).get("source_span")
        statement = span or (
            f"{event['event_type']} affects the value series from "
            f"{event['effective_start']} through {event['effective_end']}."
        )
        if style == "ticket":
            statement = f"OPS-{event['event_id']}: {statement} Owner: on-call."
        elif style == "changelog":
            statement = f"- [scheduled] {statement} # change log"
        elif style == "noisy":
            statement = f"FYI / tentative wording, but confirmed window: {statement}"
        lines.append(statement)
    header = {
        "schedule": "Published operational schedule.",
        "ticket": "Extract from linked operations tickets.",
        "changelog": "Release and maintenance changelog.",
        "noisy": "Copied operator notes; preserve only grounded claims.",
    }[style]
    return (f"{header} Treat each statement as a claim to verify, not ground "
            "truth.\n" + ("\n".join(lines) or fallback))


def generate(seed: int, per_stratum: int = 8,
             snrs: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
             ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    oracles: list[dict[str, Any]] = []
    total = HISTORY + HORIZON

    def add(*, case_id: str, family: str, history: list[float],
            actual: list[float], counterfactual: list[float],
            events: list[dict[str, Any]] | None = None,
            covariates: list[dict[str, Any]] | None = None,
            mapping: list[dict[str, str]] | None = None,
            should_influence: bool, direction: str = "none",
            magnitude: float = 0.0, onset: int | None = None,
            duration: int | None = None, dimensions: dict[str, Any],
            frequency: str = "h",
            narrative: str | None = None) -> None:
        dimensions = dict(dimensions)
        dimensions.setdefault(
            "admission_warrant",
            "asserted" if family in {"numeric_claim", "structural_change"}
            else "empirical",
        )
        dimensions.setdefault("frequency", frequency)
        dimensions.setdefault(
            "seasonality_period", {"15min": 96, "h": 24, "D": 7}[frequency]
        )
        dimensions.setdefault(
            "narrative_style",
            rng.choice(("schedule", "ticket", "changelog", "noisy")),
        )
        event_rows, covariate_rows = events or [], covariates or []
        case = {
            "schema_version": 1, "case_id": case_id, "family": family,
            "domain": rng.choice(("operations", "commerce", "energy", "health")),
            "frequency": frequency, "horizon": HORIZON,
            "history": [round(value, 8) for value in history],
            "context_events": event_rows, "covariates": covariate_rows,
            "covariate_mapping": mapping or [],
            "narrative": narrative or _narrative(
                event_rows, style=str(dimensions["narrative_style"])),
            "tags": ["stress", *(f"{key}:{value}" for key, value in
                                    sorted(dimensions.items()))],
        }
        Case.from_dict(case)
        cases.append(case)
        oracles.append({
            "case_id": case_id,
            "actual": [round(value, 8) for value in actual],
            "counterfactual": [round(value, 8) for value in counterfactual],
            "should_influence": should_influence,
            "expected_disposition": "applied" if should_influence else "scenario_only",
            "effect_direction": direction, "effect_magnitude": magnitude,
            "onset_step": onset, "duration_steps": duration,
            "dimensions": dimensions,
        })

    # Marginal evidence: the admission curve, not one easy operating point.
    for snr_index, snr in enumerate(snrs):
        for index in range(per_stratum):
            frequency = tuple(FREQUENCY_STEPS)[(snr_index + index) % 3]
            case_id = f"stress-snr-{snr:g}-{index:04d}"
            cf = _base(rng, total, frequency=frequency); observed = list(cf)
            duration = rng.randint(3, 6); starts = _schedule(rng, duration)
            onset = rng.randint(1, HORIZON - duration)
            future = HISTORY + onset
            magnitude = rng.choice((-1.0, 1.0)) * snr * NOISE_SIGMA
            direction = "increase" if magnitude > 0 else "decrease"
            _apply(observed, starts + [future], duration, magnitude)
            events = [_event(f"{case_id}-{n}", "deploy", start, duration,
                             direction=direction, frequency=frequency)
                      for n, start in enumerate([*starts, future])]
            add(case_id=case_id, family="repeated_event",
                history=observed[:HISTORY], actual=observed[HISTORY:],
                counterfactual=cf[HISTORY:], events=events,
                should_influence=True, direction=direction,
                magnitude=magnitude, onset=onset, duration=duration,
                frequency=frequency,
                dimensions={"stratum": "snr_sweep", "snr": snr,
                            "claim_truth": "true", "effect_shape": "decay",
                            "known_at": "series_start", "confounding": "none"})

    for index in range(per_stratum):
        frequency = tuple(FREQUENCY_STEPS)[index % 3]
        duration = rng.randint(3, 6); starts = _schedule(rng, duration)
        onset = rng.randint(1, HORIZON - duration); future = HISTORY + onset
        magnitude = rng.choice((-1.0, 1.0)) * rng.uniform(2.0, 4.0)
        true_direction = "increase" if magnitude > 0 else "decrease"
        false_direction = "decrease" if magnitude > 0 else "increase"

        # A compelling recurring pattern for another entity must never be
        # borrowed by the target series. This supplies the stress corpus's
        # empirical negative denominator and exercises scope isolation.
        cf = _base(rng, total, frequency=frequency); observed = list(cf)
        case_id = f"stress-scope-{index:04d}"
        events = [_event(f"{case_id}-{n}", "promotion", start, duration,
                         direction=true_direction,
                         entity_scope=("other-series",), frequency=frequency)
                  for n, start in enumerate([*starts, future])]
        add(case_id=case_id, family="entity_scope",
            history=observed[:HISTORY], actual=observed[HISTORY:],
            counterfactual=cf[HISTORY:], events=events,
            should_influence=False, frequency=frequency,
            dimensions={"stratum": "entity_scope", "snr": abs(magnitude),
                        "claim_truth": "true_for_other_entity",
                        "effect_shape": "flat", "known_at": "series_start",
                        "confounding": "cross_entity"})

        # A confident qualitative claim contradicts repeated observations.
        cf = _base(rng, total, frequency=frequency); observed = list(cf)
        _apply(observed, starts + [future], duration, magnitude)
        case_id = f"stress-misleading-{index:04d}"
        events = [_event(f"{case_id}-{n}", "deploy", start, duration,
                         direction=false_direction, frequency=frequency)
                  for n, start in enumerate([*starts, future])]
        add(case_id=case_id, family="misleading_context",
            history=observed[:HISTORY], actual=observed[HISTORY:],
            counterfactual=cf[HISTORY:], events=events, should_influence=True,
            direction=true_direction, magnitude=magnitude, onset=onset,
            duration=duration, frequency=frequency,
            dimensions={"stratum": "misleading_direction", "snr": abs(magnitude),
                        "claim_truth": "false_direction",
                        "effect_shape": "decay", "known_at": "series_start",
                        "confounding": "none"})

        # The schedule is systematically early; the true effect starts later.
        cf = _base(rng, total, frequency=frequency); observed = list(cf); delay = rng.randint(2, 4)
        timing_shape = rng.choice(("flat", "ramp", "decay"))
        _apply(observed, starts + [future], duration, magnitude, delay=delay,
               shape=timing_shape)
        case_id = f"stress-timing-{index:04d}"
        events = [_event(f"{case_id}-{n}", "maintenance", start, duration,
                         direction=true_direction, delay=(0, delay + 1),
                         frequency=frequency)
                  for n, start in enumerate([*starts, future])]
        add(case_id=case_id, family="timing_uncertainty",
            history=observed[:HISTORY], actual=observed[HISTORY:],
            counterfactual=cf[HISTORY:], events=events, should_influence=True,
            direction=true_direction, magnitude=magnitude, onset=onset + delay,
            duration=duration, frequency=frequency,
            dimensions={"stratum": "timing_uncertainty", "snr": abs(magnitude),
                        "claim_truth": "partially_true",
                        "timing_error_steps": delay,
                        "effect_shape": timing_shape,
                        "known_at": "series_start", "confounding": "none"})

        for truthful in (False, True):
            cf = _base(rng, total, frequency=frequency); observed = list(cf)
            claim_start = HISTORY + rng.randint(1, 4)
            cap = min(cf[HISTORY:]) - rng.uniform(2.0, 4.0)
            if truthful:
                for point in range(claim_start, total):
                    observed[point] = min(observed[point], cap)
            case_id = f"stress-constraint-{'true' if truthful else 'false'}-{index:04d}"
            span = (f"From {_stamp(claim_start, frequency)}, value will not "
                    f"exceed {cap:.3f}.")
            event = _event(case_id, "constraint:vendor_cap", claim_start,
                           total - claim_start, frequency=frequency,
                           attributes={"source_span": span})
            add(case_id=case_id, family="numeric_claim",
                history=observed[:HISTORY], actual=observed[HISTORY:],
                counterfactual=cf[HISTORY:], events=[event],
                should_influence=truthful,
                direction="decrease" if truthful else "none",
                magnitude=(cap - cf[claim_start] if truthful else 0.0),
                onset=claim_start - HISTORY if truthful else None,
                duration=total - claim_start if truthful else None,
                frequency=frequency,
                dimensions={"stratum": "numeric_claim", "snr": None,
                            "claim_truth": str(truthful).lower(),
                            "effect_shape": "cap", "known_at": "series_start",
                            "confounding": "none", "claimed_value": round(cap, 3)},
                narrative=span)

        for truthful in (False, True):
            slope = rng.choice((-1.0, 1.0)) * rng.uniform(0.08, 0.16)
            cf = _base(rng, total, trend=slope, frequency=frequency); observed = list(cf)
            change = HISTORY + rng.randint(1, 3)
            if truthful:
                anchor = observed[change - 1]
                for point in range(change, total):
                    # Preserve ordinary noise/seasonal movement but remove
                    # the deterministic continuation of the prior trend.
                    observed[point] = cf[point] - slope * (point - change + 1)
            case_id = f"stress-structural-{'true' if truthful else 'false'}-{index:04d}"
            span = (f"From {_stamp(change, frequency)}, the prior trend will cease and "
                    "the series will continue without that drift.")
            event = _event(
                case_id, "structural:trend_cessation", change,
                total - change, frequency=frequency, attributes={
                    "source_span": span, "effect": "trend_ceases"})
            add(case_id=case_id, family="structural_change",
                history=observed[:HISTORY], actual=observed[HISTORY:],
                counterfactual=cf[HISTORY:], events=[event],
                should_influence=truthful,
                direction=("decrease" if slope > 0 else "increase")
                if truthful else "none",
                magnitude=(-slope if truthful else 0.0),
                onset=change - HISTORY if truthful else None,
                duration=total - change if truthful else None,
                frequency=frequency,
                dimensions={"stratum": "structural_claim", "snr": None,
                            "claim_truth": str(truthful).lower(),
                            "effect_shape": "trend_ceases",
                            "known_at": "series_start", "confounding": "none"},
                narrative=span)

        # Competing event and covariate describe the same movement. The
        # forecast may use one, but applying both would double count it.
        cf = _base(rng, total, frequency=frequency); observed = list(cf)
        _apply(observed, starts + [future], duration, magnitude, shape="flat")
        active = [0.0] * total
        for start in [*starts, future]:
            for offset in range(duration):
                if start + offset < total:
                    active[start + offset] = 1.0
        case_id = f"stress-confounded-{index:04d}"
        events = [_event(f"{case_id}-{n}", "promotion", start, duration,
                         direction=true_direction, frequency=frequency)
                  for n, start in enumerate([*starts, future])]
        covariates = [{"timestamp": _stamp(step, frequency),
                       "known_at": _stamp(0, frequency),
                       "promotion_active": active[step]} for step in range(total)]
        add(case_id=case_id, family="confounded",
            history=observed[:HISTORY], actual=observed[HISTORY:],
            counterfactual=cf[HISTORY:], events=events,
            covariates=covariates, mapping=[{
                "name": "promotion_active", "type": "binary",
                "availability": "future_known"}], should_influence=True,
            direction=true_direction, magnitude=magnitude, onset=onset,
            duration=duration, frequency=frequency,
            dimensions={"stratum": "confounded", "snr": abs(magnitude),
                        "claim_truth": "true", "effect_shape": "flat",
                        "known_at": "series_start",
                        "confounding": "event_plus_covariate"})

        # Historical occurrences become knowable only after they happen;
        # only the future occurrence is available at the forecast cutoff.
        cf = _base(rng, total, frequency=frequency); observed = list(cf)
        _apply(observed, starts + [future], duration, magnitude)
        case_id = f"stress-bitemporal-{index:04d}"
        events = [_event(f"{case_id}-{n}", "incident", start, duration,
                         known_at=min(start + duration + 2, HISTORY - 1),
                         direction=true_direction, frequency=frequency)
                  for n, start in enumerate(starts)]
        events.append(_event(f"{case_id}-future", "incident", future, duration,
                             known_at=HISTORY - 1, direction=true_direction,
                             frequency=frequency))
        # The original schedule was revised at the cutoff. Keeping the
        # cancelled record tests that status and knowledge time, rather than
        # list order, determine which future window is usable.
        events.append(_event(f"{case_id}-stale", "incident", future + 2,
                             duration, known_at=0, direction=true_direction,
                             status="cancelled", frequency=frequency))
        add(case_id=case_id, family="bitemporal_context",
            history=observed[:HISTORY], actual=observed[HISTORY:],
            counterfactual=cf[HISTORY:], events=events, should_influence=True,
            direction=true_direction, magnitude=magnitude, onset=onset,
            duration=duration, frequency=frequency,
            dimensions={"stratum": "bitemporal", "snr": abs(magnitude),
                        "claim_truth": "true", "effect_shape": "decay",
                        "known_at": "late_history_future_at_cutoff",
                        "context_revision": "cancelled_then_corrected",
                        "confounding": "none"})
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
    parser.add_argument("--per-stratum", type=int, default=8)
    args = parser.parse_args()
    seed = secrets.randbits(63) if args.fresh else (
        args.seed if args.seed is not None else 20260816)
    cases, oracles = generate(seed, args.per_stratum)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    case_text, oracle_text = _render(cases), _render(oracles)
    (output / "cases.jsonl").write_text(case_text, encoding="utf-8")
    (output / "oracle.jsonl").write_text(oracle_text, encoding="utf-8")
    manifest = {
        "generator": GENERATOR_VERSION, "schema_version": 1, "seed": seed,
        "fresh_seed": bool(args.fresh), "per_stratum": args.per_stratum,
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

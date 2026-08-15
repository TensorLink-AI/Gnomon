"""Deterministically generate the publication-scale Workflow Bench corpus."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import random
import secrets
from pathlib import Path
from typing import Any

from .schema import Case

DOMAINS = ("infrastructure", "demand", "health", "environment", "finance")
GENERATOR_VERSION = "workflow-synthetic-v2"


def _base(index: int, kind: str, domain: str, question: str,
          available: dict[str, Any], answer_schema: dict[str, list[str]],
          oracle: dict[str, Any], tags: list[str],
          stages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"schema_version": 1, "id": f"pub-{kind}-{index:03d}",
            "kind": kind, "domain": domain, "question": question,
            "available_at_cutoff": available, "answer_schema": answer_schema,
            "oracle": oracle, "tags": tags, "stages": stages or []}


def generate_publication_cases(seed: int = 20260813,
                               per_kind: int = 20) -> list[Case]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for index in range(per_kind):
        domain = DOMAINS[index % len(DOMAINS)]
        variant = (index // len(DOMAINS)) % 4
        # Exact periodic continuation: no subjective model choice.
        period = rng.randint(3, 9)
        phase = rng.randrange(period)
        amplitude = rng.uniform(1.5, 8.0)
        cycle = [round(20 + index + amplitude * math.sin(
                 2 * math.pi * (k + phase) / period), 4)
                 for k in range(period)]
        repeats = rng.randint(3, 6)
        series = cycle * repeats
        rows.append(_base(index, "synthetic", domain,
            "Forecast exactly one step and identify the known repeating period.",
            {"cutoff": "2026-01-31", "series": series},
            {"numbers": ["next"], "choices": ["pattern"], "facts": ["source_kind"]},
            {"numbers": {"next": cycle[len(series) % period]},
             "tolerances": {"next": 0.001},
             "choices": {"pattern": f"period-{period}"},
             **({"choice_aliases": {"pattern": ["weekly"]}}
                if period == 7 else {}),
             "required_facts": {"source_kind": "synthetic"},
             "allowed_support": ["supported", "degraded"],
             "requires_publish_parity": True, "requires_quote_match": True},
            ["forecast", "known-truth", f"mechanism:periodic-{period}"]))

        # Frozen outcome: answer is graded against an explicitly revealed
        # realized point, not an arguable trend label.
        level = 100 + index * 2
        if variant == 0:
            slope = rng.choice([-3, -2, -1, 1, 2, 3])
            length = rng.randint(10, 18)
            history, outcome, mechanism = (
                [level + slope * step for step in range(length)],
                level + slope * length, "linear-trend")
        elif variant == 1:
            pattern = [level + rng.randint(-8, 8) for _ in range(rng.randint(3, 6))]
            repeats = rng.randint(3, 5)
            history, outcome, mechanism = pattern * repeats, pattern[0], "seasonal"
        elif variant == 2:
            history, outcome, mechanism = [level] * 12, level, "level"
        else:
            history = [level + (-1) ** step * 4 for step in range(12)]
            outcome, mechanism = level + 4, "alternating"
        rows.append(_base(index, "frozen", domain,
            "Forecast the next value; the realized outcome will be revealed later.",
            {"cutoff": "2026-02-12", "series": history, "fixture_version": "pub-v1"},
            {"numbers": ["next"], "choices": [], "facts": ["cutoff"]},
            {"numbers": {"next": outcome}, "tolerances": {"next": 2.0},
             "required_facts": {"cutoff": "2026-02-12"},
             "allowed_support": ["supported", "degraded", "best_effort"],
             "requires_publish_parity": True},
            ["forecast", "frozen", "decision", f"mechanism:{mechanism}"],
            [{"name": "outcome", "revealed": {"actual": outcome},
              "answer_schema": {"numbers": ["absolute_error"], "choices": [],
                                "facts": ["tracked_forecast_id"]},
              "oracle": {"numbers": {"absolute_error": 0.0},
                         "tolerances": {"absolute_error": 2.0}}}]))

        # Ambiguous target followed by a literal repair instruction. Correct
        # behaviour is abstain first, then complete after target resolution.
        column_names = rng.sample(["east", "west", "primary", "backup"], 2)
        columns = {column_names[0]: [30 + rng.randrange(-2, 3) for _ in range(14)],
                   column_names[1]: [60 + rng.randrange(-2, 3) for _ in range(14)]}
        target = rng.choice(column_names)
        rows.append(_base(index, "messy", domain,
            "Forecast the requested demand series one step.",
            {"cutoff": "2026-03-14", "columns": columns},
            {"numbers": [], "choices": [], "facts": ["ambiguity"]},
            {"should_abstain": True, "required_facts": {"ambiguity": "target"},
             "allowed_support": ["abstained"], "requires_repair": True},
            ["repair", "ambiguity"],
            [{"name": "repair", "revealed": {"target_column": target,
                                                "horizon": 1},
              "answer_schema": {"numbers": ["next"], "choices": [],
                                "facts": ["resolved_target"]},
              "oracle": {"numbers": {"next": columns[target][-1]},
                         "tolerances": {"next": 4.0},
                         "required_facts": {"resolved_target": target},
                         "allowed_support": ["degraded", "best_effort"]}}]))

        # Initial forecast plus a separately sealed outcome stage.
        baseline = 50 + index
        longitudinal_patterns = (
            ([baseline - 1, baseline, baseline + 1], "seasonal-3"),
            ([baseline - 2, baseline + 2], "alternating"),
            ([baseline] * 4, "level"),
            ([baseline - 3, baseline - 1, baseline + 1, baseline + 3], "seasonal-4"),
        )
        pattern, mechanism = longitudinal_patterns[variant]
        longitudinal = (pattern * (21 // len(pattern) + 1))[:21]
        actual = pattern[21 % len(pattern)]
        rows.append(_base(index, "longitudinal", domain,
            "Forecast one step, save it, then score it when the outcome is revealed.",
            {"cutoff": "2026-04-21", "series": longitudinal},
            {"numbers": ["next"], "choices": [], "facts": ["tracking_requested"]},
            {"numbers": {"next": actual}, "tolerances": {"next": 2.0},
             "required_facts": {"tracking_requested": True},
             "allowed_support": ["supported", "degraded", "best_effort"],
             "requires_tracking": True, "requires_publish_parity": True},
            ["forecast", "tracking", f"mechanism:{mechanism}"],
            [{"name": "outcome", "revealed": {"actual": actual},
              "answer_schema": {"numbers": ["absolute_error"], "choices": [],
                                "facts": ["tracked_forecast_id"]},
              "oracle": {"numbers": {"absolute_error": 0.0},
                         "tolerances": {"absolute_error": 2.0}}}]))

        # Notability is defined mathematically in the public question.
        spike_name = rng.choice(("api", "worker", "db"))
        channels = {name: [10.0] * 12 for name in ("api", "worker", "db")}
        channels[spike_name][-1] = 25.0 + index
        rows.append(_base(index, "multiseries", domain,
            "Return the channel with the largest absolute final-step change; preserve a remainder artifact.",
            {"cutoff": "2026-05-12", "series": channels},
            {"numbers": [], "choices": ["most_notable"],
             "facts": ["ranking_rule", "remainder_preserved"]},
            {"choices": {"most_notable": spike_name},
             "required_facts": {"ranking_rule": "largest absolute final-step change",
                                "remainder_preserved": True},
             "engine_required_facts": {
                 "ranking_rule": "largest absolute final-step change",
                 "remainder_preserved": True},
             "allowed_support": ["supported", "degraded"]},
            ["triage", "artifact", "known-truth"]))
    cases = [Case.from_dict(row) for row in rows]
    if len({case.id for case in cases}) != len(cases):
        raise AssertionError("generator produced duplicate ids")
    return cases


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
        args.seed if args.seed is not None else 20260813)
    cases = generate_publication_cases(seed, args.per_kind)
    from dataclasses import asdict
    text = "".join(json.dumps(asdict(case), sort_keys=True) + "\n" for case in cases)
    from .provenance import atomic_write_text
    output = Path(args.output)
    atomic_write_text(output, text)
    from .provenance import corpus_sha256
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

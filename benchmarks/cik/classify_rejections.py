"""Classify future-context span rejections from CiK run dumps.

The unmeasured number blocking the warrant roadmap: the span parser
rejected 176 of 220 proposals (span_states_the_value /
span_states_the_bound), and nothing recorded *why* — parser too narrow,
claims genuinely non-numeric, or proposals malformed. Each bucket has a
different owner (a grammar fix, a new warrant class, a proposer-prompt
fix), so the split decides the roadmap.

This script walks one or more CiK output directories, reads every
per-run ``extra_info`` dump (the shape ``run_cik.write_outputs`` reads),
and buckets every future-context rejection:

- ``parses_now`` — the recorded span is accepted by the *current*
  parser: the historical rejection was parser narrowness since fixed.
  Re-running after a grammar change measures exactly what the change
  recovered.
- ``numeric_no_parse`` — the span contains digits but the current
  parser still rejects it: a candidate grammar gap. Examples are
  emitted for review; some will be genuinely inadmissible (percentages
  of an unstated base, dates).
- ``non_numeric_claim`` — no digits at all ("the factory is closed
  Tuesday"): no possible grammar fix; this is the class that needs a
  different warrant, and its share is the load-bearing input to the
  proposer-trust experiment (results/proposer-trust-warrant/).
- ``span_unrecorded`` — the dump predates the change that records the
  rejected span; only the check code is known.

Rejections whose code is not a span-parse failure (window overlaps
history, claim contradicts span, contradictory overrides) are counted
under their own code: they are proposer errors by construction.

Two dump sources are read, because abstained runs — half of CiK — never
produce an ``extra_info`` file, while the Gnomon artifact (with the full
gate record in its evidence) is written even when the adapter then
abstains:

- ``runs/<task>/<seed>/extra_info`` under an output directory (scored
  runs only);
- ``**/gnomon-output/<forecast_id>/artifact.json`` under a work
  directory (the
  adapter's ``cik-gnomon-*`` temp dirs; every run that reached the
  forecast, scored or abstained). Set ``TMPDIR`` when launching the run
  so these land somewhere collectable.

Pass output directories OR the work directory, not both slices of the
same run set: the same run's gate would then be counted twice.

Usage:
    python -m benchmarks.cik.classify_rejections RUN_DIR [RUN_DIR ...] \
        [--examples 5] [--json out.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SPAN_PARSE_CODES = {"span_states_the_bound", "span_states_the_value"}
_DIGITS = re.compile(r"\d")


def iter_gate_records(roots: list[Path]):
    """Yield ``(path, gate_dict | None, proposed_event_ids)`` per run dump.

    ``extra_info`` dumps carry the gate under ``future_context``;
    artifact dumps carry it as ``future_context_gate`` evidence. A run
    with no gate (control, or the flag off) yields ``None`` so callers
    can still count it.
    """
    for root in roots:
        base = root / "runs" if (root / "runs").is_dir() else root
        for path in sorted(base.glob("*/*/extra_info")):
            try:
                parsed = ast.literal_eval(path.read_text(encoding="utf-8"))
            except (SyntaxError, ValueError, OSError):
                continue
            if isinstance(parsed, dict):
                gate = parsed.get("future_context")
                yield path, gate if isinstance(gate, dict) else None, [
                    str(item) for item in parsed.get("proposed_events", [])
                ]
        for path in sorted(base.glob("**/gnomon-output/*/artifact.json")):
            try:
                parsed = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(parsed, dict):
                continue
            gates = [
                item.get("payload") for item in parsed.get("evidence", [])
                if isinstance(item, dict)
                and item.get("kind") == "future_context_gate"
                and isinstance(item.get("payload"), dict)
            ]
            gate = gates[0] if gates else None
            proposed = [] if gate is None else [
                str(check.get("event_id"))
                for check in gate.get("checks", [])
            ]
            yield path, gate, sorted(set(proposed))


def classify_span(span: str) -> str:
    """Bucket one recorded rejected span against the current parser."""
    from gnomon.future_context import (
        parse_bound_span,
        parse_override_scale,
        parse_override_span,
    )

    bound, _ = parse_bound_span(span)
    value, _ = parse_override_span(span)
    scale, _ = parse_override_scale(span)
    if bound is not None or value is not None or scale is not None:
        return "parses_now"
    if _DIGITS.search(span):
        return "numeric_no_parse"
    return "non_numeric_claim"


def classify(roots: list[Path], example_limit: int = 5) -> dict[str, Any]:
    runs_seen = 0
    runs_with_gate = 0
    admitted = Counter()
    rejection_codes = Counter()
    span_buckets = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    proposed_event_types = Counter()

    for path, gate, proposed_events in iter_gate_records(roots):
        runs_seen += 1
        if gate is None:
            continue
        runs_with_gate += 1
        for item in gate.get("admitted", []):
            admitted[item.get("event_class", "unknown")] += 1
        for item in gate.get("rejected", []):
            code = item.get("code", "unknown")
            rejection_codes[code] += 1
            if code not in SPAN_PARSE_CODES:
                continue
            span = item.get("source_span")
            if not span:
                span_buckets["span_unrecorded"] += 1
                continue
            bucket = classify_span(str(span))
            span_buckets[bucket] += 1
            if len(examples[bucket]) < example_limit:
                examples[bucket].append(str(span))
        for event_id in proposed_events:
            proposed_event_types[str(event_id).split("-")[0]] += 1

    span_total = sum(span_buckets.values())
    return {
        "runs_seen": runs_seen,
        "runs_with_future_context_gate": runs_with_gate,
        "admitted_by_class": dict(admitted),
        "proposed_event_type_census": dict(proposed_event_types),
        "rejections_by_code": dict(rejection_codes),
        "span_parse_rejections": {
            "total": span_total,
            "buckets": dict(span_buckets),
            "examples": {bucket: items for bucket, items in examples.items()},
        },
        "reading": (
            "parses_now = recovered by the current grammar; "
            "numeric_no_parse = candidate grammar gaps, review the "
            "examples; non_numeric_claim = needs a warrant the lane does "
            "not have (input to the proposer-trust experiment); "
            "span_unrecorded = dump predates span recording, "
            "unclassifiable."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify future-context span rejections from CiK dumps.",
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--examples", type=int, default=5,
                        help="Examples to keep per bucket (default 5)")
    parser.add_argument("--json", type=Path, default=None,
                        help="Also write the summary to this file")
    args = parser.parse_args(argv)
    missing = [str(path) for path in args.run_dirs if not path.is_dir()]
    if missing:
        parser.error(f"not a directory: {', '.join(missing)}")
    summary = classify(args.run_dirs, example_limit=args.examples)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

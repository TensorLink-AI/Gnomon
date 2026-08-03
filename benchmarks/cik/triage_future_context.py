"""Triage the future-context lane's span rejections from run artifacts.

The gate's rejection records answer *that* a span was refused; this tool
answers *why*, at the only altitude where the fix can be chosen. A
``span_states_the_value`` rejection has two very different causes with
two very different remedies:

- the proposer quoted a sentence that does not state a value at all —
  the fix belongs in the proposer prompt, not the parser;
- the sentence states the value in a phrasing the parser cannot read —
  the fix belongs in the parser, and the span is the test case.

Every bucket below is decided deterministically from the recorded span
(re-running today's parsers where useful), never by judgement calls, so
two people running this on the same artifacts get the same counts.

Buckets, per rejection code:

``span_not_recorded``
    The rejection record carries no ``source_span`` — it predates the
    change that attached spans to rejections. Nothing can be concluded
    from these rows except that the run must be repeated to triage them.
``parses_now``
    Today's parser reads the span. The rejection is already fixed;
    re-running the benchmark would admit (or history-check) this event.
``relative_value_refused``
    The span quantifies the claim relative to an unstated base (a
    percentage). Refusal is correct by design; if these dominate, the
    prompt should tell the proposer not to quote relative statements.
``no_stated_value`` / ``no_stated_bound``
    The span contains no digit and no zero-state / bound phrase: the
    quoted sentence does not state the number, so no parser could admit
    it. The fix is prompt-side (quote the sentence that states it).
``parser_gap_candidate``
    The span contains a digit (or a zero-state word) and still does not
    parse. These are the parser's missing phrasings — each is printed
    verbatim, because each is a regression test waiting to be written.

Usage::

    python -m benchmarks.cik.triage_future_context RUN_DIR [RUN_DIR ...] \
        [--json triage.json]

``RUN_DIR`` is a ``run_cik.py`` ``--output-dir`` (the tool walks
``runs/*/*/extra_info``), any directory containing ``extra_info`` files,
or an ``extra_info`` file itself.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gnomon.future_context import (  # noqa: E402
    _ZERO_STATES,
    parse_bound_span,
    parse_override_span,
)

#: Rejection codes whose remedy depends on what the span says.
SPAN_PARSE_CODES = ("span_states_the_value", "span_states_the_bound")

_DIGIT = re.compile(r"\d")
_RELATIVE = re.compile(r"\d\s*(?:%|percent\b|pct\b)", re.IGNORECASE)
#: Digit-free phrasings the bound parser does accept, so their absence
#: (with no digit) proves the span states no bound at all.
_WORDED_BOUND = re.compile(
    r"non-?negative|negative|positive", re.IGNORECASE
)


def iter_extra_info_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("extra_info")))
        else:
            raise SystemExit(f"not a file or directory: {raw}")
    return files


def load_extra_info(path: Path) -> dict:
    """Read one official ``extra_info`` dump (pprint format, literals only)."""
    try:
        parsed = ast.literal_eval(path.read_text(encoding="utf-8"))
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def bucket_rejection(code: str, span: str | None) -> str:
    """Deterministically classify one span-parse rejection."""
    if span is None:
        return "span_not_recorded"
    if code == "span_states_the_value":
        value, problem = parse_override_span(span)
        if value is not None:
            return "parses_now"
        if problem and "percentage of an unstated base" in problem:
            return "relative_value_refused"
        if not _DIGIT.search(span) and not _ZERO_STATES.search(span):
            return "no_stated_value"
        return "parser_gap_candidate"
    bound, _ = parse_bound_span(span)
    if bound is not None:
        return "parses_now"
    if _RELATIVE.search(span):
        return "relative_value_refused"
    if not _DIGIT.search(span) and not _WORDED_BOUND.search(span):
        return "no_stated_bound"
    return "parser_gap_candidate"


def triage(files: list[Path]) -> dict[str, Any]:
    """Aggregate every future-context gate decision across the run files."""
    code_counts: Counter[str] = Counter()
    admitted = defensive = 0
    runs_with_lane = 0
    buckets: dict[str, dict[str, Counter[str]]] = {
        code: {} for code in SPAN_PARSE_CODES
    }
    for path in files:
        info = load_extra_info(path)
        gate = info.get("future_context")
        if not isinstance(gate, dict):
            continue
        runs_with_lane += 1
        admitted += len(gate.get("admitted") or [])
        defensive += len(gate.get("defensive") or [])
        for rejection in gate.get("rejected") or []:
            code = str(rejection.get("code", "unknown"))
            code_counts[code] += 1
            if code not in SPAN_PARSE_CODES:
                continue
            span = rejection.get("source_span")
            span = span if isinstance(span, str) else None
            bucket = bucket_rejection(code, span)
            buckets[code].setdefault(bucket, Counter())[
                span if span is not None else "(span not recorded)"
            ] += 1
    return {
        "files_read": len(files),
        "runs_with_lane_evidence": runs_with_lane,
        "events_admitted": admitted,
        "events_defensive": defensive,
        "rejections_by_code": dict(sorted(code_counts.items())),
        "span_buckets": {
            code: {
                bucket: {
                    "count": sum(spans.values()),
                    "spans": [
                        {"span": text, "occurrences": count}
                        for text, count in spans.most_common()
                    ],
                }
                for bucket, spans in sorted(by_bucket.items())
            }
            for code, by_bucket in buckets.items()
        },
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        f"extra_info files read: {report['files_read']} "
        f"(lane evidence in {report['runs_with_lane_evidence']})",
        f"admitted: {report['events_admitted']}   "
        f"defensive: {report['events_defensive']}",
        "",
        "rejections by code:",
    ]
    for code, count in report["rejections_by_code"].items():
        lines.append(f"  {count:5d}  {code}")
    for code, by_bucket in report["span_buckets"].items():
        if not by_bucket:
            continue
        lines += ["", f"{code} triage:"]
        for bucket, payload in by_bucket.items():
            lines.append(f"  {payload['count']:5d}  {bucket}")
        gaps = by_bucket.get("parser_gap_candidate")
        if gaps:
            lines += ["", f"  parser gap candidates ({code}) — each span "
                          f"below is a regression test waiting to be written:"]
            for item in gaps["spans"]:
                lines.append(f"    {item['occurrences']:3d}x  {item['span']!r}")
    lines += [
        "",
        "reading the buckets:",
        "  parser_gap_candidate -> fix the parser; the span is the test case",
        "  no_stated_value/bound -> fix the proposer prompt; no parser could",
        "                           read a number the sentence never states",
        "  relative_value_refused -> correct refusal by design",
        "  parses_now            -> already fixed; re-run to collect it",
        "  span_not_recorded     -> re-run with span-carrying rejections",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="+",
                        help="run output dirs, dirs of runs, or extra_info files")
    parser.add_argument("--json", default=None,
                        help="also write the full report as JSON to this path")
    args = parser.parse_args(argv)
    files = iter_extra_info_files(args.paths)
    if not files:
        raise SystemExit("no extra_info files found under the given paths")
    report = triage(files)
    print(render(report))
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nJSON report: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

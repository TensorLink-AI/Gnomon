"""Classify ``window_is_future_only`` rejections from CiK run dumps.

The structural-effects census (results/structural-effects/census-full.json)
counted 50 rejections under this code across 355 runs — the second-largest
bucket — and could not tell them apart, because the record kept only
prose. The competing explanations have different owners:

- ``entirely_historical`` — the proposed window ends at or before the
  last observation: the proposer described the past. Correct routing
  (the fold-ablation gate owns testable windows), not a lane bug; if
  large, the proposer prompt needs a dating instruction.
- ``boundary_overlap`` — the window starts within one data step of the
  last observation: an off-by-one/inclusive-endpoint artifact of how
  the proposer dates "from tomorrow". A dating-instruction or
  tolerance question, and the bucket that would justify revisiting the
  check itself.
- ``timezone_shaped`` — the event and the dataset differ in timezone
  awareness and the overlap is no larger than a plausible offset
  (≤ 14 h): the wall-clock alignment reading, not the proposer, put
  the window in the past. This bucket is the live-bug candidate.
- ``misdated_partial`` — genuine partial overlap beyond one step:
  proposer misdating, owned by the proposer prompt.
- ``record_predates_data`` — the dump predates the change that records
  the compared quantities (future_context.record_check ``data``);
  unclassifiable. The historical census predates it; the next grid run
  records everything needed.

Usage (same dump discovery as classify_rejections):
    python -m benchmarks.cik.classify_window_rejections RUN_DIR [...] \
        [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cik.classify_rejections import iter_gate_records  # noqa: E402

CODE = "window_is_future_only"
MAX_TIMEZONE_OFFSET_SECONDS = 14 * 3600


def _step_seconds(gate_path: Path) -> float | None:
    """The data step, from the sibling artifact's forecast timestamps."""
    if gate_path.name != "artifact.json":
        return None
    try:
        parsed = json.loads(gate_path.read_text(encoding="utf-8"))
        rows = parsed["results"][0]["forecast"]
        stamps = [datetime.fromisoformat(str(row["timestamp"]))
                  for row in rows[:2]]
    except Exception:
        return None
    if len(stamps) < 2:
        return None
    return (stamps[1] - stamps[0]).total_seconds()


def classify_rejection(data: dict[str, Any] | None,
                       step_seconds: float | None) -> str:
    """Bucket one rejection from its recorded comparison quantities."""
    if not isinstance(data, dict) or "overlap_seconds" not in data:
        return "record_predates_data"
    overlap = float(data["overlap_seconds"])
    if data.get("window_entirely_historical"):
        return "entirely_historical"
    if data.get("mixed_timezone_alignment") \
            and overlap <= MAX_TIMEZONE_OFFSET_SECONDS:
        return "timezone_shaped"
    if step_seconds is not None and overlap <= step_seconds:
        return "boundary_overlap"
    return "misdated_partial"


def classify(roots: list[Path], example_limit: int = 5) -> dict[str, Any]:
    runs_seen = 0
    rejections = 0
    buckets = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for path, gate, _proposed in iter_gate_records(roots):
        runs_seen += 1
        if gate is None:
            continue
        step_seconds = _step_seconds(path)
        for item in gate.get("rejected", []):
            if item.get("code") != CODE:
                continue
            rejections += 1
            bucket = classify_rejection(item.get("data"), step_seconds)
            buckets[bucket] += 1
            if len(examples[bucket]) < example_limit:
                examples[bucket].append({
                    "run": str(path.parent),
                    "event_id": item.get("event_id"),
                    "data": item.get("data"),
                })

    return {
        "runs_seen": runs_seen,
        "window_rejections": rejections,
        "buckets": dict(buckets),
        "examples": dict(examples),
        "reading": (
            "entirely_historical / misdated_partial are proposer dating "
            "errors (owner: proposer prompt); boundary_overlap within one "
            "step suggests an off-by-one dating convention (owner: prompt "
            "or check tolerance — decide on the examples); timezone_shaped "
            "is the wall-clock alignment reading and the only bucket that "
            "would indict the engine; record_predates_data needs a rerun "
            "at a revision that records check data."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify window_is_future_only rejections from CiK dumps.",
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--json", type=Path, default=None)
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

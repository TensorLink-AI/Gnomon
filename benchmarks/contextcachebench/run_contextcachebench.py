"""Benchmark fresh typed context against persistent context_ref replay."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _case(root: Path, index: int) -> tuple[Path, list[dict]]:
    source = root / f"case-{index:03d}.csv"
    start = date(2025, 1, 1)
    rows = ["timestamp,value"]
    events = []
    active = set()
    for day in range(100):
        if day % 14 == 5:
            active.update({day, day + 1})
            moment = start + timedelta(days=day)
            events.append({
                "event_id": f"case-{index}-event-{day}",
                "event_type": "promotion", "entity_scope": ["value"],
                "effective_start": moment.isoformat() + "T00:00:00+00:00",
                "effective_end": (moment + timedelta(days=1)).isoformat()
                                 + "T23:59:59+00:00",
                "known_at": (start - timedelta(days=1)).isoformat()
                            + "T00:00:00+00:00",
                "source": {"type": "synthetic", "reference": f"case-{index}"},
            })
        value = 100 + .2 * day + (7 if day in active else 0) + (index % 3)
        rows.append(f"{start + timedelta(days=day)},{value}")
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return source, events


def run(*, cases: int, output_dir: Path, replays_per_case: int = 10) -> dict:
    from gnomon.toolspec import runner_for
    from gnomon.context_store import cache_metrics, reset_cache_metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GNOMON_CONTEXT_STORE"] = str(output_dir / "context-store")
    os.environ["GNOMON_CONTEXT_NAMESPACE"] = "contextcachebench"
    reset_cache_metrics()
    equivalent = hits = artifact_equivalent = 0
    fresh_seconds, cached_seconds = [], []
    inline_bytes = ref_bytes = 0
    rows = []
    for index in range(cases):
        source, events = _case(output_dir, index)
        common = {
            "input": str(source), "time_column": "timestamp",
            "target_column": "value", "frequency": "D", "horizon": 7,
        }
        started = time.perf_counter()
        fresh = runner_for("gnomon_forecast")({
            **common, "context_events": events,
            "output_dir": str(output_dir / f"fresh-{index}"),
        })
        fresh_seconds.append(time.perf_counter() - started)
        fresh_points = [row["q50"] for row in fresh["results"][0]["forecast"]]
        fresh_artifact = json.loads((Path(fresh["artifact_path"])
                                     / "artifact.json").read_text())
        inline_bytes += len(json.dumps({"context_events": events}))
        for replay_index in range(replays_per_case):
            started = time.perf_counter()
            cached = runner_for("gnomon_forecast")({
                **common, "context_ref": fresh["context_ref"],
                "output_dir": str(output_dir / f"cached-{index}-{replay_index}"),
            })
            cached_seconds.append(time.perf_counter() - started)
            cached_points = [row["q50"] for row in
                             cached["results"][0]["forecast"]]
            same = fresh_points == cached_points
            equivalent += int(same)
            hit = cached.get("context_cache", {}).get("status") == "hit"
            hits += int(hit)
            cached_artifact = json.loads((Path(cached["artifact_path"])
                                          / "artifact.json").read_text())
            artifact_same = cached_artifact["results"] == fresh_artifact["results"]
            artifact_equivalent += int(artifact_same)
            ref_bytes += len(json.dumps({"context_ref": fresh["context_ref"]}))
            rows.append({"case": index, "replay": replay_index,
                         "forecast_equivalent": same,
                         "artifact_results_equivalent": artifact_same,
                         "receipt_hit": hit})
    replay_count = cases * replays_per_case
    # The fresh call supplies context once; without a receipt every replay
    # would require another host compilation/source submission.
    compiler_reduction = 1 - cases / (cases + replay_count)
    inline_bytes *= replays_per_case
    telemetry = cache_metrics()
    reduction = 1 - ref_bytes / inline_bytes if inline_bytes else 0
    summary = {
        "benchmark": "contextcachebench", "schema_version": "0.1",
        "cases": cases, "replays": replay_count,
        "forecast_equivalent": equivalent,
        "receipt_cache_hits": hits,
        "assessment_cache_hits": telemetry["assessment_hits"],
        "compiler_call_reduction": compiler_reduction,
        "context_argument_byte_reduction": reduction,
        "median_fresh_seconds": statistics.median(fresh_seconds),
        "median_cached_seconds": statistics.median(cached_seconds),
        "gates": {
            "complete": len(rows) == replay_count,
            "forecast_byte_equivalence": equivalent == replay_count,
            "receipt_hit_rate_100pct": hits == replay_count,
            "assessment_hit_rate_100pct":
                telemetry["assessment_hits"] == replay_count,
            "artifact_results_equivalent": artifact_equivalent == replay_count,
            "compiler_call_reduction_at_least_90pct":
                compiler_reduction >= .9,
            "context_argument_reduction_at_least_80pct": reduction >= .8,
        }, "rows": rows,
    }
    summary["graduated"] = all(summary["gates"].values())
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--replays-per-case", type=int, default=10)
    parser.add_argument("--output-dir", default="results/contextcachebench")
    args = parser.parse_args()
    summary = run(cases=args.cases, output_dir=Path(args.output_dir),
                  replays_per_case=args.replays_per_case)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["graduated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

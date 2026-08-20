"""Score Gnomon's immutable primary forecast on ContextBench histories.

Unlike ContextBench's context-admission runner, this adapter deliberately runs
only the history lane.  It is the appropriate experiment for changing the
candidate pool (for example classical-only versus classical + Toto): context
logic cannot obscure whether the model addition improved the primary answer.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean

from benchmarks.contextbench.run_contextbench import _forecast, _points, smape
from benchmarks.contextbench.schema import load_cases, load_oracles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()

    corpus = Path(args.corpus_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = load_cases(corpus / "cases.jsonl")
    oracles = load_oracles(corpus / "oracle.jsonl")
    started = time.perf_counter()
    observations = []
    root = output / "artifacts"
    root.mkdir(exist_ok=True)
    for index, case in enumerate(cases, 1):
        case_started = time.perf_counter()
        (root / case.case_id).mkdir(parents=True, exist_ok=True)
        artifact, artifact_path = _forecast(
            case, root / case.case_id, enriched=False)
        result = artifact.results[0]
        points = _points(result)
        observations.append({
            "case_id": case.case_id,
            "family": case.family,
            "smape": smape(oracles[case.case_id].actual, points),
            "selected_model": result.selected_model,
            "support": result.support,
            "artifact_path": str(artifact_path),
            "elapsed_seconds": time.perf_counter() - case_started,
        })
        print(f"[{index}/{len(cases)}] {case.case_id} "
              f"model={result.selected_model} "
              f"sMAPE={observations[-1]['smape']:.4f}", flush=True)

    (output / "observations.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in observations),
        encoding="utf-8")
    by_family = {}
    for family in sorted({row["family"] for row in observations}):
        rows = [row for row in observations if row["family"] == family]
        by_family[family] = {
            "cases": len(rows), "mean_smape": mean(row["smape"] for row in rows)}
    summary = {
        "benchmark": "contextbench-primary-models-v1",
        "arm": args.arm,
        "cases": len(observations),
        "mean_smape": mean(row["smape"] for row in observations),
        "families": by_family,
        "selected_models": dict(Counter(
            row["selected_model"] for row in observations)),
        "support": dict(Counter(row["support"] for row in observations)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

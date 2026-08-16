"""Aggregate independent ContextBench corpora into one decision report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .run_contextbench import summarize


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def aggregate(run_dirs: list[Path], minimum_replicates: int = 3) -> dict[str, Any]:
    pooled: list[dict[str, Any]] = []
    hashes: list[str] = []
    seeds: list[Any] = []
    generators: set[str] = set()
    replicates: list[dict[str, Any]] = []
    for directory in run_dirs:
        summary_path = directory / "summary.json"
        observations = directory / "observations.jsonl"
        if not summary_path.is_file() or not observations.is_file():
            raise ValueError(f"incomplete ContextBench run: {directory}")
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        generators.add(str(prior.get("generator") or ""))
        corpus_hash = str(prior.get("corpus_manifest_sha256") or "")
        if not corpus_hash:
            raise ValueError(f"run has no corpus manifest hash: {directory}")
        if corpus_hash in hashes:
            raise ValueError(f"duplicate corpus manifest: {corpus_hash}")
        rows = _rows(observations)
        if len(rows) != int(prior.get("cases", -1)):
            raise ValueError(f"run is incomplete: {directory}")
        hashes.append(corpus_hash); seeds.append(prior.get("seed"))
        pooled.extend(rows)
        replicates.append({
            "run_dir": str(directory), "seed": prior.get("seed"),
            "corpus_manifest_sha256": corpus_hash, "cases": len(rows),
            "decision_ready": bool(prior.get("decision_ready")),
            "failed_gates": sorted(
                key for key, passed in (prior.get("gates") or {}).items()
                if not passed),
        })
    if len(generators) != 1:
        raise ValueError(
            "replicated report cannot mix clean and stress generators: "
            + ", ".join(sorted(generators))
        )
    manifest = {
        # Preserve the policy family. `summarize` keys stress gates from the
        # generator prefix; renaming it would silently grade the pooled stress
        # observations with the clean-corpus gates.
        "generator": next(iter(generators)),
        "seed": seeds, "fresh_seed": all(
            json.loads((directory / "summary.json").read_text()).get(
                "fresh_seed") for directory in run_dirs),
        "cases": len(pooled),
    }
    report = summarize(pooled, manifest)
    enough = len(run_dirs) >= minimum_replicates
    report.update({
        "benchmark": "contextbench-replicated",
        "replicate_count": len(run_dirs),
        "minimum_replicates": minimum_replicates,
        "unique_corpus_manifests": len(hashes),
        "replicates": replicates,
        "pooled_product_gates_pass": bool(report["decision_ready"]),
        "replicate_count_gate": enough,
        "decision_ready": bool(report["decision_ready"] and enough),
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-replicates", type=int, default=3)
    args = parser.parse_args()
    report = aggregate([Path(value) for value in args.run_dir],
                       args.minimum_replicates)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["decision_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Merge disjoint TemporalBench shards into a resumable canonical run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_shards(target: Path, shards: list[Path]) -> dict[str, int]:
    """Merge records/details by task id and reject conflicting observations."""
    target.mkdir(parents=True, exist_ok=True)
    details = target / "details"
    details.mkdir(exist_ok=True)
    records: dict[str, dict] = {}
    detail_payloads: dict[str, str] = {}
    summary_path = target / "summary.json"
    target_summary = (json.loads(summary_path.read_text(encoding="utf-8"))
                      if summary_path.is_file() else {})
    cumulative_usage = dict(target_summary.get("llm_usage") or {})
    merged_usage_sources = set(target_summary.get("merged_usage_sources") or [])
    if cumulative_usage and not merged_usage_sources:
        merged_usage_sources.add(str(target.resolve()))
    for source in [target, *shards]:
        records_path = source / "gnomonbench.jsonl"
        if not records_path.is_file():
            # The runner writes complete JSON lines to a durable partial file
            # and only renames after the selected range finishes. An operator
            # interrupt must not strand already-paid rows.
            records_path = source / "gnomonbench.partial.jsonl"
        if records_path.is_file():
            for line in records_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                task_id = str(record.get("task_id"))
                previous = records.get(task_id)
                if previous is not None and previous != record:
                    raise ValueError(f"conflicting record for {task_id}")
                records[task_id] = record
        source_details = source / "details"
        if source_details.is_dir():
            for path in source_details.glob("*.json"):
                content = path.read_text(encoding="utf-8")
                previous = detail_payloads.get(path.name)
                if previous is not None and previous != content:
                    raise ValueError(f"conflicting detail for {path.stem}")
                detail_payloads[path.name] = content
        if source == target:
            continue
        source_key = str(source.resolve())
        source_summary_path = source / "summary.json"
        if not source_summary_path.is_file():
            source_summary_path = source / "usage.checkpoint.json"
        if source_key in merged_usage_sources or not source_summary_path.is_file():
            continue
        source_summary = json.loads(
            source_summary_path.read_text(encoding="utf-8"))
        usage = source_summary.get("llm_usage") or {}
        for key in ("requests", "transport_attempts", "prompt_tokens",
                    "completion_tokens", "cost_usd",
                    "truncation_escalations"):
            cumulative_usage[key] = cumulative_usage.get(key, 0) + usage.get(key, 0)
        merged_usage_sources.add(source_key)
    ordered = sorted(records.values(), key=lambda row: str(row["task_id"]))
    (target / "gnomonbench.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered),
        encoding="utf-8")
    for name, content in detail_payloads.items():
        (details / name).write_text(content, encoding="utf-8")
    if cumulative_usage:
        target_summary["llm_usage"] = cumulative_usage
        target_summary["merged_usage_sources"] = sorted(merged_usage_sources)
        summary_path.write_text(
            json.dumps(target_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    return {"records": len(records), "details": len(detail_payloads)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("shards", nargs="+")
    args = parser.parse_args()
    result = merge_shards(
        Path(args.target), [Path(item) for item in args.shards])
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

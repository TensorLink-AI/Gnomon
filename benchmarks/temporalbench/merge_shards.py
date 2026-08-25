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
    record_sources: dict[str, Path] = {}
    detail_payloads: dict[str, str] = {}
    summary_path = target / "summary.json"
    target_state_path = (summary_path if summary_path.is_file() else
                         target / "usage.checkpoint.json")
    target_summary = (json.loads(target_state_path.read_text(encoding="utf-8"))
                      if target_state_path.is_file() else {})
    cumulative_usage = dict(target_summary.get("llm_usage") or {})
    cumulative_infrastructure_retries = int(
        target_summary.get("infrastructure_retries") or 0)
    cumulative_infrastructure_failures = dict(
        target_summary.get("infrastructure_failures_retried") or {})
    merged_usage_sources = set(target_summary.get("merged_usage_sources") or [])
    manifests: list[tuple[Path, dict]] = []
    if cumulative_usage and not merged_usage_sources:
        merged_usage_sources.add(str(target.resolve()))
    for source in [target, *shards]:
        manifest_path = source / "manifest.json"
        if source != target and manifest_path.is_file():
            manifests.append((source, json.loads(
                manifest_path.read_text(encoding="utf-8"))))
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
                    # A bounded retry is a replacement observation only when
                    # it converts an infrastructure failure into a scored
                    # record.  Prefer that successful completion regardless
                    # of shard ordering; never choose between two substantive
                    # answers, which would permit result shopping.
                    previous_failed = bool(previous.get("error"))
                    current_failed = bool(record.get("error"))
                    if previous_failed != current_failed:
                        if not current_failed:
                            records[task_id] = record
                            record_sources[task_id] = source
                        continue
                    raise ValueError(f"conflicting record for {task_id}")
                records[task_id] = record
                record_sources[task_id] = source
        source_details = source / "details"
        if source_details.is_dir():
            for path in source_details.glob("*.json"):
                content = path.read_text(encoding="utf-8")
                previous = detail_payloads.get(path.name)
                if previous is not None and previous != content:
                    selected_source = record_sources.get(path.stem)
                    if selected_source == source:
                        detail_payloads[path.name] = content
                        continue
                    if selected_source is not None:
                        continue
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
        cumulative_infrastructure_retries += int(
            source_summary.get("infrastructure_retries") or 0)
        for name, count in (source_summary.get(
                "infrastructure_failures_retried") or {}).items():
            cumulative_infrastructure_failures[name] = (
                cumulative_infrastructure_failures.get(name, 0) + int(count))
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
        target_summary["infrastructure_retries"] = cumulative_infrastructure_retries
        target_summary["infrastructure_failures_retried"] = dict(sorted(
            cumulative_infrastructure_failures.items()))
        target_summary["merged_usage_sources"] = sorted(merged_usage_sources)
        summary_path.write_text(
            json.dumps(target_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    if manifests:
        # Shard commands differ by offset/output directory, but every field
        # describing the experimental arm must agree. Preserve one verified
        # manifest plus the source commands instead of dropping provenance
        # and forcing the matched reporter to assume comparability.
        # Resume/retry flags describe how an invocation recovered observations,
        # not the experimental arm that produced them. Their exact commands
        # remain in source_commands, while compatibility is enforced on model,
        # endpoint, condition, code, task filters, and harness limits.
        # Partition controls are expected to differ across disjoint shards;
        # they do not define an experimental arm. Their exact values remain
        # auditable in ``source_commands``. Everything that can change model
        # or harness behaviour remains a strict compatibility field.
        # Recovery shards may raise operational caps after an observation
        # failed without producing an answer.  The record merge above permits
        # those shards to replace failures only; successful observations can
        # never be selected between.  Keep the differing caps in the merged
        # manifest instead of pretending the run used one uniform allowance.
        recovery_fields = {"infrastructure_retries", "initial_max_tokens",
                           "max_retries", "request_timeout"}
        ignored = {"command", "resume", "retry_voided", "limit", "offset",
                   "row_offset", "run_status", *recovery_fields}
        reference = {key: value for key, value in manifests[0][1].items()
                     if key not in ignored}
        for source, manifest in manifests[1:]:
            comparable = {key: value for key, value in manifest.items()
                          if key not in ignored}
            if comparable != reference:
                differing = sorted(
                    key for key in set(reference) | set(comparable)
                    if reference.get(key) != comparable.get(key))
                raise ValueError(
                    f"incompatible shard manifest {source}: "
                    + ", ".join(differing))
        merged_manifest = dict(reference)
        merged_manifest.update({
            "command": "merge_shards",
            "merged_shards": [str(source.resolve())
                               for source, _ in manifests],
            "source_commands": [manifest.get("command")
                                for _, manifest in manifests],
            "rows": len(records),
            "recovery_controls": {
                key: sorted({manifest.get(key) for _, manifest in manifests},
                            key=lambda value: (value is None, value))
                for key in sorted(recovery_fields)
                if len({manifest.get(key) for _, manifest in manifests}) > 1
            } or None,
        })
        (target / "manifest.json").write_text(
            json.dumps(merged_manifest, indent=2, sort_keys=True) + "\n",
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

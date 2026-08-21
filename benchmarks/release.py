"""Build and validate small, citable benchmark-result releases.

Raw runs remain ignored and are uploaded as CI artifacts.  This module copies
only aggregate summaries into git, records the source digest, and rejects
fields which commonly contain prompts, model responses, observations, or
per-case data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


SCHEMA_VERSION = "1.1"
DROP_KEYS = {
    "rows", "observations", "responses", "transcripts", "receipts",
    "per_case", "case_results", "raw_results", "question_receipts",
}
FORBIDDEN_KEY_PARTS = ("api_key", "authorization", "credential", "secret")
MAX_SUMMARY_BYTES = 1_000_000


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _compact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _compact(item) for key, item in value.items()
            if key not in DROP_KEYS
        }
    if isinstance(value, list):
        # Aggregate vectors are useful, but a large list is almost always
        # per-case output.  Preserve its size rather than silently erasing it.
        if len(value) > 100:
            return {"omitted_from_release": True, "item_count": len(value)}
        return [_compact(item) for item in value]
    if isinstance(value, str):
        # Machine-local checkout paths are not result provenance. Preserve the
        # repository-relative identity without publishing a runner home path.
        marker = "/Gnomon/"
        if value.startswith("/") and marker in value:
            return value.split(marker, 1)[1]
    return value


def _walk_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if any(part in key.lower() for part in FORBIDDEN_KEY_PARTS):
                found.append(path)
            found.extend(_walk_keys(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_walk_keys(item, f"{prefix}[{index}]"))
    return found


def build(spec_path: Path) -> Path:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    output = Path(spec["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for entry in spec["benchmarks"]:
        source = Path(entry["source"])
        payload = json.loads(source.read_text(encoding="utf-8"))
        curated = {
            "release_metadata": {
                "benchmark": entry["benchmark"],
                "arm": entry.get("arm"),
                "scope": entry["scope"],
                "status": entry.get("status", "complete"),
                "source": source.as_posix(),
                "source_sha256": _digest(source),
                "evaluated_commit": entry.get("evaluated_commit", "unknown"),
                "harness_commit": entry.get("harness_commit", "unknown"),
                "dataset_identity": entry.get("dataset_identity", "unknown"),
                "configuration_identity": entry.get(
                    "configuration_identity", "unknown"),
                "notes": entry.get("notes", []),
            },
            "summary": _compact(payload),
        }
        destination = output / entry["file"]
        destination.write_text(
            json.dumps(curated, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        records.append({
            "benchmark": entry["benchmark"],
            "arm": entry.get("arm"),
            "file": entry["file"],
            "scope": entry["scope"],
            "status": entry.get("status", "complete"),
            "sha256": _digest(destination),
            "bytes": destination.stat().st_size,
        })
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release": spec["release"],
        "curated_by_commit": _git_sha(),
        "policy": {
            "aggregate_only": True,
            "raw_runs_are_ci_artifacts": True,
            "smoke_results_are_not_full_results": True,
        },
        "benchmarks": records,
        "not_included": spec.get("not_included", []),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate(output)
    return output


def validate(release_dir: Path) -> None:
    manifest_path = release_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported benchmark release schema")
    seen: set[tuple[str, str | None]] = set()
    for omission in manifest.get("not_included", []):
        if not omission.get("benchmark") or not omission.get("reason"):
            raise ValueError("every omitted benchmark needs a name and reason")
    for record in manifest.get("benchmarks", []):
        identity = (record["benchmark"], record.get("arm"))
        if identity in seen:
            raise ValueError(f"duplicate benchmark/arm: {identity}")
        seen.add(identity)
        path = release_dir / record["file"]
        if not path.is_file() or path.stat().st_size > MAX_SUMMARY_BYTES:
            raise ValueError(f"missing or oversized summary: {path}")
        if _digest(path) != record["sha256"]:
            raise ValueError(f"summary digest mismatch: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        forbidden = _walk_keys(payload)
        if forbidden:
            raise ValueError(f"sensitive fields in {path}: {forbidden}")
        metadata = payload.get("release_metadata", {})
        if record.get("scope") not in {"full", "smoke", "subset"}:
            raise ValueError(f"invalid run scope: {path}")
        if metadata.get("scope") != record.get("scope"):
            raise ValueError(f"scope mismatch: {path}")
        provenance = ("evaluated_commit", "harness_commit", "dataset_identity",
                      "configuration_identity")
        if any(not metadata.get(field) for field in provenance):
            raise ValueError(f"incomplete result provenance: {path}")
        if record.get("status") in {"complete", "graduated"} and any(
                metadata.get(field) == "unknown" for field in provenance):
            raise ValueError(f"publishable result has unknown provenance: {path}")
        if record.get("status") == "graduated":
            summary = payload.get("summary", {})
            gates = summary.get("gates")
            if summary.get("graduated") is not True or not isinstance(gates, dict) \
                    or not gates or not all(value is True for value in gates.values()):
                raise ValueError(f"graduated result has failing gates: {path}")
        if record.get("scope") == "smoke" and record.get("status") == "full":
            raise ValueError(f"smoke run mislabeled as full: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("spec", type=Path)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("release_dir", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        print(build(args.spec))
    else:
        validate(args.release_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

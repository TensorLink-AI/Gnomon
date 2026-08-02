"""Materialise MTBench's Hugging Face export as the per-task JSONs the
official evaluation scripts read.

``download_processed_dataset.py`` fetches each MTBench dataset as a
single parquet shard, but the official evaluation scripts
(``evaluation/finance/value_prediction.py`` and friends) iterate
``dataset_folder.glob("*.json")`` and reach into ``text.content`` and
``technical.*``. This writes exactly that shape — the same fields, no
transformation of any value — so the official script can run
byte-for-byte unmodified against the official data, and so both
conditions read identical inputs.

Usage::

    python -m benchmarks.mtbench.export_tasks \
        --parquet-dir ~/MTBench/data/processed/finance/aligned_in30days_out7days \
        --output-dir ~/MTBench/data/tasks/finance_long --limit 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Columns the official scripts read as nested objects.
JSON_OBJECT_COLUMNS = ("text", "technical", "trend")


def _as_object(value: Any) -> Any:
    """Parse a column the export stored as a JSON string."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"content": value}
    return value


def _plain(value: Any) -> Any:
    """numpy arrays/scalars to JSON-serialisable Python."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def export(parquet_dir: Path, output_dir: Path,
           limit: int | None = None) -> int:
    import pandas as pd

    shards = sorted(parquet_dir.rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards under {parquet_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for shard in shards:
        frame = pd.read_parquet(shard)
        for position, row in enumerate(frame.to_dict("records")):
            if limit is not None and written >= limit:
                return written
            task = {}
            for key, value in row.items():
                task[key] = (_as_object(value) if key in JSON_OBJECT_COLUMNS
                             else _plain(value))
            path = output_dir / f"{shard.stem}_{position:04d}.json"
            path.write_text(json.dumps(task), encoding="utf-8")
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Export at most this many tasks (in file order)")
    args = parser.parse_args()
    count = export(Path(args.parquet_dir).expanduser(),
                   Path(args.output_dir).expanduser(), args.limit)
    print(f"{count} task JSONs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

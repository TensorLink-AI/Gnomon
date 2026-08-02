"""Compare our benchmark runs against published numbers.

Because every adapter scores with the benchmark's official metric
implementation, our summaries are directly comparable to the numbers in
the papers and leaderboards — provided the protocol matches (see the
comparability checklist in benchmarks/README.md; a --limit smoke run is
NOT comparable to a paper table).

Published numbers are supplied by you, transcribed from the paper or
leaderboard into a reference YAML — this repo deliberately ships only a
template (benchmarks/configs/published_reference.yaml) so no
possibly-stale third-party numbers are baked into the codebase. Each
entry should carry its source and retrieval date.

Reference file shape::

    cik:
      metric: mean RCRPS
      lower_is_better: true
      published:
        - {system: "<name from leaderboard>", value: 0.000,
           source: "https://... (retrieved YYYY-MM-DD)"}
      ours:
        - {system: "gnomon-agent (ours)",
           summary: results/cik-agent/summary.json,
           key: rcrps.mean}

``key`` is a dot-path into the run's summary.json; list indices are
numeric segments (e.g. ``tiers.0``).

Usage::

    python -m benchmarks.compare_published \
        --reference benchmarks/configs/published_reference.yaml
    python -m benchmarks.compare_published --reference ref.yaml --only cik
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def dot_get(obj: Any, path: str) -> Any:
    """Resolve a dot-path through dicts and lists; None if absent."""
    current = obj
    for segment in path.split("."):
        if isinstance(current, dict):
            if segment not in current:
                return None
            current = current[segment]
        elif isinstance(current, list):
            try:
                current = current[int(segment)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def collect_rows(section: dict[str, Any]) -> list[dict[str, Any]]:
    """Merge published and our entries into one comparable row list."""
    rows: list[dict[str, Any]] = []
    for entry in section.get("published") or []:
        rows.append({"system": entry.get("system", "?"),
                     "value": entry.get("value"),
                     "origin": "published",
                     "source": entry.get("source", "")})
    for entry in section.get("ours") or []:
        value = None
        note = ""
        summary_file = entry.get("summary")
        if summary_file:
            path = Path(str(summary_file)).expanduser()
            if path.exists():
                summary = json.loads(path.read_text(encoding="utf-8"))
                value = dot_get(summary, entry.get("key", ""))
                if value is None:
                    note = f"key {entry.get('key')!r} not in {path.name}"
            else:
                note = f"missing {path}"
        rows.append({"system": entry.get("system", "ours"),
                     "value": value, "origin": "ours", "source": note})
    return rows


def rank_rows(rows: list[dict[str, Any]],
              lower_is_better: bool) -> list[dict[str, Any]]:
    scored = [r for r in rows if isinstance(r["value"], (int, float))]
    unscored = [r for r in rows if not isinstance(r["value"], (int, float))]
    scored.sort(key=lambda r: r["value"], reverse=not lower_is_better)
    return scored + unscored


def render_section(name: str, section: dict[str, Any]) -> str:
    lower = bool(section.get("lower_is_better", False))
    metric = section.get("metric", "metric")
    rows = rank_rows(collect_rows(section), lower)
    direction = "lower is better" if lower else "higher is better"
    lines = [f"## {name} — {metric} ({direction})", ""]
    if not rows:
        return "\n".join(lines + ["(no entries)", ""])
    width = max(len(str(r["system"])) for r in rows) + 2
    for position, row in enumerate(rows, start=1):
        value = row["value"]
        rendered = f"{value:.4f}" if isinstance(value, float) else \
            str(value) if value is not None else "—"
        marker = "*" if row["origin"] == "ours" else " "
        source = f"  ({row['source']})" if row.get("source") else ""
        lines.append(
            f"{position:>2}. {marker} {str(row['system']):<{width}}"
            f"{rendered}{source}"
        )
    lines.append("")
    lines.append("   * = this repo's runs; others transcribed from the "
                 "cited sources")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--reference", required=True,
                        help="YAML/JSON reference file (see template)")
    parser.add_argument("--only", default=None,
                        help="comma list of benchmark sections to show")
    args = parser.parse_args()

    path = Path(args.reference)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as error:
            raise SystemExit("pyyaml required for YAML references "
                             "(pip install pyyaml)") from error
        reference = yaml.safe_load(text)
    else:
        reference = json.loads(text)
    if not isinstance(reference, dict) or not reference:
        raise SystemExit("reference file is empty")

    wanted = ({n.strip() for n in args.only.split(",")}
              if args.only else set(reference))
    shown = 0
    for name, section in reference.items():
        if name not in wanted or not isinstance(section, dict):
            continue
        print(render_section(name, section))
        shown += 1
    if not shown:
        raise SystemExit(f"no sections match --only {args.only}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

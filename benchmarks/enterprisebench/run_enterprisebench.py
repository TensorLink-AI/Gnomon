"""EnterpriseBench CLI: run one domain pack or all of them.

Each domain writes ``<output-dir>/<domain>/rows.jsonl`` and
``summary.json``; the cross-domain rollup lands in
``<output-dir>/summary.json`` as a per-domain verdict table with an
explicit refusal to average across domains — the units differ.

Usage::

    python -m benchmarks.enterprisebench.run_enterprisebench \
        --domain cloudcost --model <model> --output-dir results/eb
    python -m benchmarks.enterprisebench.run_enterprisebench \
        --domain all --cases 120 --output-dir results/eb --resume
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT), str(ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from benchmarks.common.envfile import load_env_file  # noqa: E402
import benchmarks.enterprisebench.domains  # noqa: E402,F401  (registers packs)
from benchmarks.enterprisebench.harness import registry, run_domain  # noqa: E402


def rollup(domain_summaries: dict[str, dict[str, Any]],
           output: Path) -> dict[str, Any]:
    usage_totals: dict[str, float] = {}
    table = {}
    for name, summary in sorted(domain_summaries.items()):
        table[name] = {
            "verdicts": summary["verdicts"],
            "cost_units": summary["cost_model"]["units"],
            "dataset_identity": summary["provenance"]["dataset_identity"],
        }
        for key, value in (summary.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                usage_totals[key] = usage_totals.get(key, 0) + value
    combined = {
        "suite": "enterprisebench",
        "domains": table,
        "aggregation": {
            "no_single_aggregate_number": True,
            "reason": ("domains price decisions in different units; "
                       "averaging cloud overage units with cash "
                       "shortfall units would manufacture a number with "
                       "no owner. Read the per-domain verdict table."),
        },
        "usage_totals": usage_totals,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return combined


def run(args: argparse.Namespace, client: Any = None) -> dict[str, Any]:
    packs = registry()
    wanted = sorted(packs) if args.domain == "all" else [args.domain]
    unknown = [name for name in wanted if name not in packs]
    if unknown:
        raise SystemExit(
            f"unknown domain(s) {unknown}; known: {sorted(packs)}")
    if client is None:
        load_env_file()
        from benchmarks.common.openrouter import OpenRouterClient
        client = OpenRouterClient(
            args.model, api_key=os.environ.get(args.api_key_env),
            base_url=args.base_url, temperature=0,
            max_tokens=args.max_tokens, max_retries=4,
            reasoning_effort=args.reasoning_effort)
    output = Path(args.output_dir)
    summaries: dict[str, dict[str, Any]] = {}
    for name in wanted:
        domain_args = SimpleNamespace(
            seed=args.seed, cases=args.cases, model=args.model,
            output_dir=str(output / name), resume=args.resume,
            concurrency=args.concurrency)
        summaries[name] = run_domain(packs[name], domain_args, client)
        print(f"[{name}] summary -> {output / name / 'summary.json'}",
              flush=True)
    combined = rollup(summaries, output)
    print(json.dumps(combined["aggregation"], indent=2))
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="all",
                        help="domain pack name, or 'all'")
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--base-url", default="https://api.engy.ai/v1")
    parser.add_argument("--api-key-env", default="ENGY_API_KEY")
    parser.add_argument("--cases", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1200,
                        help="Initial completion budget; retries may "
                             "escalate it.")
    parser.add_argument(
        "--reasoning-effort", default=None,
        choices=("none", "low", "medium", "high"),
        help="Explicit provider reasoning mode; omitted preserves the "
             "model default.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

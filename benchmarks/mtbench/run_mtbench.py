"""Run MTBench conditions: official LLM scripts via OpenRouter, or the
Aion treatment on the forecasting task families.

Control (any official evaluation script, unmodified, LLM served by
OpenRouter)::

    python -m benchmarks.mtbench.run_mtbench control \
        --mtbench-root ~/MTBench \
        --script evaluation/finance/value_prediction.py \
        --model openai/gpt-4o -- \
        --dataset_folder=../../data/processed/finance/aligned_in30days_out7days \
        --save_path=../../results/finance/pred_time_in30_out7/openrouter/combined \
        --indicator=time --model=gpt-4o --mode=combined

    (the script's own ``--model gpt-4o`` selects the official dispatch
    branch; the patch serves it with the OpenRouter model you chose)

Treatment (Aion owns the numbers; forecasting families only)::

    python -m benchmarks.mtbench.run_mtbench aion \
        --mtbench-root ~/MTBench \
        --dataset-folder ~/MTBench/data/processed/finance/aligned_in30days_out7days \
        --output-dir results/mtbench-aion-agent \
        --mode agent --model openai/gpt-4o
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    control = sub.add_parser("control", help="Run an official script via OpenRouter")
    control.add_argument("--mtbench-root", required=True)
    control.add_argument("--script", required=True,
                         help="Script path relative to the checkout, e.g. "
                              "evaluation/finance/value_prediction.py")
    control.add_argument("--model", required=True,
                         help="OpenRouter model id serving the completions")
    control.add_argument("--temperature", type=float, default=0.7)
    control.add_argument("script_args", nargs=argparse.REMAINDER,
                         help="Arguments after -- go to the official script")

    aion = sub.add_parser("aion", help="Aion treatment (forecasting tasks)")
    aion.add_argument("--mtbench-root", default=None,
                      help="Checkout path; enables the official MAPE import")
    aion.add_argument("--dataset-folder", required=True,
                      help="Official processed dataset folder of task JSONs")
    aion.add_argument("--output-dir", required=True)
    aion.add_argument("--mode", choices=["pure", "agent", "tools"],
                      default="pure")
    aion.add_argument("--model", default=None,
                      help="OpenRouter model id (required for agent/tools modes)")
    aion.add_argument("--temperature", type=float, default=1.0)
    aion.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    if args.command == "control":
        from benchmarks.mtbench.openrouter_patch import run_official_script

        script_args = [a for a in args.script_args if a != "--"]
        client = run_official_script(
            Path(args.mtbench_root).expanduser().resolve(),
            args.script, script_args, args.model, args.temperature,
        )
        print(json.dumps({"llm_usage": client.usage_summary}, indent=2))
        return 0

    if args.mode == "agent" and not args.model:
        parser.error("--model is required for --mode agent")
    from benchmarks.mtbench.aion_forecaster import run

    summary = run(
        Path(args.dataset_folder).expanduser().resolve(),
        Path(args.output_dir),
        mode=args.mode,
        openrouter_model=args.model,
        mtbench_root=(Path(args.mtbench_root).expanduser().resolve()
                      if args.mtbench_root else None),
        temperature=args.temperature,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

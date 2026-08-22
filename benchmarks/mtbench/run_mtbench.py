"""Run MTBench conditions: official LLM scripts via OpenRouter, or the
Gnomon treatment on the forecasting task families.

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

Treatment (Gnomon owns the numbers; forecasting families only)::

    python -m benchmarks.mtbench.run_mtbench gnomon \
        --mtbench-root ~/MTBench \
        --dataset-folder ~/MTBench/data/processed/finance/aligned_in30days_out7days \
        --output-dir results/mtbench-gnomon-agent \
        --mode agent --model openai/gpt-4o
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.manifest import code_revision, write_manifest  # noqa: E402


def _json_safe(value: Any) -> Any:
    """Convert parquet/numpy values into the official JSON task shape."""
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def materialize_official_json_view(
    dataset_folder: Path, output: Path, *, limit: int | None = None,
) -> int:
    """Expose either official storage layout as per-task JSON.

    MTBench's published download currently contains parquet shards while its
    official evaluators glob only ``*.json``.  This lossless view lets those
    unmodified evaluators consume their own download and also gives matched
    control/treatment runs the same deterministic prefix under ``--limit``.
    """
    output.mkdir(parents=True, exist_ok=True)
    json_files = sorted(dataset_folder.glob("*.json"))
    rows: list[dict[str, Any]] = []
    if json_files:
        for path in json_files[:limit]:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    else:
        shards = sorted(dataset_folder.rglob("*.parquet"))
        if not shards:
            raise FileNotFoundError(
                f"No task JSONs or parquet shards found in {dataset_folder}")
        import pandas as pd
        for shard in shards:
            rows.extend(shard_frame for shard_frame in
                        pd.read_parquet(shard).to_dict("records"))
            if limit is not None and len(rows) >= limit:
                rows = rows[:limit]
                break
    for index, raw in enumerate(rows):
        row = _json_safe(raw)
        for field in ("text", "technical"):
            value = row.get(field)
            if isinstance(value, str) and value.lstrip().startswith("{"):
                try:
                    row[field] = json.loads(value)
                except json.JSONDecodeError:
                    pass
        (output / f"sample-{index:06d}.json").write_text(
            json.dumps(row) + "\n", encoding="utf-8")
    return len(rows)


def _replace_dataset_argument(
    script_args: list[str], script_dir: Path, replacement: Path,
) -> tuple[list[str], Path]:
    """Replace the official script's required dataset argument."""
    resolved: Path | None = None
    updated = list(script_args)
    for index, arg in enumerate(updated):
        if arg.startswith("--dataset_folder="):
            value = arg.split("=", 1)[1]
            resolved = (script_dir / value).resolve() if not Path(value).is_absolute() else Path(value)
            updated[index] = f"--dataset_folder={replacement}"
            break
        if arg == "--dataset_folder" and index + 1 < len(updated):
            value = updated[index + 1]
            resolved = (script_dir / value).resolve() if not Path(value).is_absolute() else Path(value)
            updated[index + 1] = str(replacement)
            break
    if resolved is None:
        raise ValueError("official MTBench script args must include --dataset_folder")
    return updated, resolved


def _script_argument(script_args: list[str], name: str) -> str | None:
    """Read either ``--name=value`` or ``--name value``."""
    flag = f"--{name}"
    for index, arg in enumerate(script_args):
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
        if arg == flag and index + 1 < len(script_args):
            return script_args[index + 1]
    return None


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
    control.add_argument("--limit", type=int, default=None,
                         help="Matched deterministic task prefix; applied by "
                              "materializing the official data as JSON")
    control.add_argument("script_args", nargs=argparse.REMAINDER,
                         help="Arguments after -- go to the official script")

    gnomon = sub.add_parser("gnomon", help="Gnomon treatment (forecasting tasks)")
    gnomon.add_argument("--mtbench-root", default=None,
                      help="Checkout path; enables the official MAPE import")
    gnomon.add_argument("--dataset-folder", required=True,
                      help="Official processed dataset folder of task JSONs")
    gnomon.add_argument("--output-dir", required=True)
    gnomon.add_argument("--mode", choices=["pure", "agent", "tools", "mcp"],
                      default="pure")
    gnomon.add_argument("--model", default=None,
                      help="OpenRouter model id (required for "
                           "agent/tools/mcp modes)")
    # Same default as the control arm: the ground rules require identical
    # temperature across the two conditions of a comparison.
    gnomon.add_argument("--temperature", type=float, default=0.7)
    gnomon.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    run_revision = code_revision()
    if args.command == "control":
        from benchmarks.mtbench.openrouter_patch import run_official_script

        script_args = [a for a in args.script_args if a != "--"]
        mtbench_root = Path(args.mtbench_root).expanduser().resolve()
        script_dir = (mtbench_root / args.script).resolve().parent
        # The upstream evaluator cannot read the parquet layout shipped by
        # its own downloader.  Materialize only when necessary (always for a
        # matched limited run, or when no per-task JSONs exist).
        placeholder = Path("/tmp/gnomon-mtbench-dataset")
        prepared_args, dataset_folder = _replace_dataset_argument(
            script_args, script_dir, placeholder)
        needs_view = args.limit is not None or not any(dataset_folder.glob("*.json"))
        context = tempfile.TemporaryDirectory(prefix="gnomon-mtbench-") if needs_view else nullcontext(None)
        with context as temp:
            if needs_view:
                view = Path(temp)
                materialize_official_json_view(
                    dataset_folder, view, limit=args.limit)
                prepared_args, _ = _replace_dataset_argument(
                    script_args, script_dir, view)
            else:
                prepared_args = script_args
            client = run_official_script(
                mtbench_root, args.script, prepared_args, args.model,
                args.temperature)
        save_path = _script_argument(script_args, "save_path")
        if save_path:
            resolved_save = (script_dir / save_path).resolve() \
                if not Path(save_path).is_absolute() else Path(save_path)
            write_manifest(
                resolved_save,
                benchmark="mtbench",
                condition="control/" + str(
                    _script_argument(script_args, "mode") or "unspecified"),
                target="/".join(dataset_folder.parts[-2:]),
                model=args.model,
                temperature=args.temperature,
                limit=args.limit,
                official_script=args.script,
                indicator=_script_argument(script_args, "indicator"),
                base_url=client.base_url,
                llm_usage=client.usage_summary,
                command=" ".join([sys.executable, "-m",
                                  "benchmarks.mtbench.run_mtbench"] + sys.argv[1:]),
                status="ok",
                code_revision=run_revision,
            )
        print(json.dumps({"llm_usage": client.usage_summary}, indent=2))
        return 0

    if args.mode in ("agent", "tools", "mcp") and not args.model:
        parser.error(f"--model is required for --mode {args.mode}")
    from benchmarks.mtbench.gnomon_forecaster import run

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

"""Config-driven orchestrator for the benchmark suite.

One YAML (or JSON) config describes a batch of benchmark runs; this
script builds each adapter's CLI invocation, injects shared defaults
where that adapter supports them, executes the runs sequentially, and
collects every ``summary.json`` into one combined summary.

Config shape::

    model: openai/gpt-4o        # default --model where supported
    output_root: results/2026-08-01
    defaults:                   # injected when the adapter has the flag
      limit: 50
      temperature: 0.2
    runs:
      - benchmark: temporalbench
        name: tb-control        # output lands in <output_root>/<name>
        args: {condition: control, data_dir: ~/temporalbench, tiers: "T2,T4"}
      - benchmark: mtbench
        name: mtb-aion
        args: {subcommand: aion, dataset_folder: ~/mtbench/data, mode: agent}

Any key in ``args`` becomes ``--kebab-case`` on the command line
(``true`` becomes a bare flag, ``subcommand`` a leading positional), so
every adapter-specific option is reachable without this file knowing
about it.

Usage::

    python -m benchmarks.run_all --config benchmarks/configs/example.yaml
    python -m benchmarks.run_all --config cfg.yaml --dry-run
    python -m benchmarks.run_all --config cfg.yaml --only tb-control,tb-aion
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.common.manifest import write_manifest  # noqa: E402

# Which shared defaults each adapter's CLI can accept. "limit_flag"
# names the adapter's closest equivalent of a sample cap.
REGISTRY: dict[str, dict[str, Any]] = {
    "cik": {
        "module": "benchmarks.cik.run_cik",
        "accepts": {"model", "temperature", "output_dir"},
        # CiK has no task cap; subset via `task_filter` / `seeds` in args.
        "limit_flag": None,
    },
    "anomllm": {
        "module": "benchmarks.anomllm.run_anomllm",
        # Own layout: no output dir or task cap, and the model flag names
        # the control condition it serves.
        "accepts": {"model"},
        "model_key": "control_model",
        "limit_flag": None,
    },
    "mtbench": {
        "module": "benchmarks.mtbench.run_mtbench",
        "accepts": {"model", "temperature", "output_dir"},
        "limit_flag": "--limit",  # aion subcommand only
    },
    "timesage_mt": {
        "module": "benchmarks.timesage_mt.run_timesage",
        "accepts": {"model", "temperature", "output_dir"},
        "limit_flag": "--limit",
    },
    "temporalbench": {
        "module": "benchmarks.temporalbench.run_temporalbench",
        "accepts": {"model", "temperature", "output_dir"},
        "limit_flag": "--limit",
    },
    "leaktrap": {
        # Tasks are generated from (seed, horizon, history), not downloaded,
        # so this adapter needs no data directory.
        "module": "benchmarks.leaktrap.run_leaktrap",
        "accepts": {"model", "temperature", "output_dir"},
        "limit_flag": "--limit",
    },
}


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as error:
            raise SystemExit(
                "pyyaml is required for YAML configs (pip install pyyaml), "
                "or use a .json config"
            ) from error
        return yaml.safe_load(text)
    return json.loads(text)


def build_command(run: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """Translate one run entry into a python -m invocation."""
    benchmark = run.get("benchmark")
    if benchmark not in REGISTRY:
        raise ValueError(
            f"unknown benchmark {benchmark!r}; known: {sorted(REGISTRY)}"
        )
    spec = REGISTRY[benchmark]
    args = dict(run.get("args") or {})
    name = run.get("name") or benchmark
    defaults = dict(config.get("defaults") or {})

    command = [sys.executable, "-m", spec["module"]]
    if "subcommand" in args:
        command.append(str(args.pop("subcommand")))

    accepts = set(spec["accepts"])
    limit_flag = spec["limit_flag"]
    if benchmark == "mtbench" and command[-1] == "control":
        # The control subcommand wraps the official script: it takes
        # --model/--temperature but owns no output dir or task cap.
        accepts = {"model", "temperature"}
        limit_flag = None

    model_key = spec.get("model_key", "model")
    if "model" in accepts and model_key not in args and config.get("model"):
        args[model_key] = config["model"]
    if "temperature" in accepts and "temperature" not in args \
            and "temperature" in defaults:
        args["temperature"] = defaults["temperature"]
    if "output_dir" in accepts and "output_dir" not in args \
            and config.get("output_root"):
        args["output_dir"] = str(Path(config["output_root"]) / name)
    if limit_flag and "limit" not in args and "limit" in defaults:
        args[limit_flag.lstrip("-").replace("-", "_")] = defaults["limit"]

    positional = args.pop("positional", [])
    for key, value in args.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    command.extend(str(item) for item in positional)
    return command


def benchmark_of(run: dict[str, Any]) -> str | None:
    return run.get("benchmark")


def summary_path(run: dict[str, Any], config: dict[str, Any]) -> Path | None:
    args = run.get("args") or {}
    if "output_dir" in args:
        return Path(str(args["output_dir"])).expanduser() / "summary.json"
    if config.get("output_root"):
        name = run.get("name") or run.get("benchmark")
        return Path(config["output_root"]) / name / "summary.json"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands without executing")
    parser.add_argument("--only", default=None,
                        help="comma list of run names to execute")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="keep going when a run exits non-zero")
    parser.add_argument("--output-root", default=None,
                        help="override the config's output_root, for running "
                             "the same config twice in different environments "
                             "(e.g. with and without TSFM sandboxes installed)")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.output_root:
        config["output_root"] = args.output_root
    args_config_path = args.config
    runs = config.get("runs") or []
    if not runs:
        raise SystemExit("config has no 'runs'")
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        runs = [r for r in runs
                if (r.get("name") or r.get("benchmark")) in wanted]
        if not runs:
            raise SystemExit(f"no runs match --only {args.only}")

    repo_root = Path(__file__).resolve().parents[1]
    outcomes = []
    for run in runs:
        name = run.get("name") or run.get("benchmark")
        command = build_command(run, config)
        printable = " ".join(command)
        if args.dry_run:
            print(f"[{name}] {printable}")
            outcomes.append({"name": name, "command": printable,
                             "status": "dry-run"})
            continue
        print(f"[{name}] running: {printable}", flush=True)
        result = subprocess.run(command, cwd=repo_root)
        status = "ok" if result.returncode == 0 else \
            f"exit {result.returncode}"
        outcome: dict[str, Any] = {"name": name, "command": printable,
                                   "status": status}
        path = summary_path(run, config)
        if path is not None:
            # Record what this arm was, so a later comparison can refuse
            # to put it next to an arm that answered other questions.
            # Not `args`: that name holds the parsed CLI namespace, and
            # shadowing it here made every iteration after the first fail on
            # `args.dry_run` — so the orchestrator only ever completed a
            # single run.
            run_args = dict(run.get("args") or {})
            write_manifest(
                path.parent,
                benchmark=benchmark_of(run),
                run_name=name,
                condition=(run_args.get("condition") or run_args.get("method")
                           or run_args.get("mode") or run_args.get("subcommand")),
                target=run.get("target") or run_args.get("indicator"),
                model=config.get("model"),
                command=printable,
                config_path=args_config_path,
                limit=(config.get("defaults") or {}).get("limit"),
                status=status,
            )
        if path and path.exists():
            outcome["summary"] = json.loads(path.read_text(encoding="utf-8"))
        outcomes.append(outcome)
        if result.returncode != 0 and not args.continue_on_error:
            print(f"[{name}] failed; stopping (use --continue-on-error "
                  "to keep going)")
            break

    if config.get("output_root") and not args.dry_run:
        combined = Path(config["output_root"]) / "combined_summary.json"
        combined.parent.mkdir(parents=True, exist_ok=True)
        combined.write_text(
            json.dumps({"config": args.config, "runs": outcomes}, indent=2)
            + "\n", encoding="utf-8",
        )
        print(f"combined summary -> {combined}")
    failed = [o for o in outcomes if o["status"] not in ("ok", "dry-run")]
    print(f"{len(outcomes)} run(s), {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

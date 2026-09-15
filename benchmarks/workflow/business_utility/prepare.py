"""Prepare pinned configuration for the EXISTING runner; makes no model calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shlex
import sys

from .corpus import ROOT, generate

PROMPT = """Answer the business task using any available tools. Python is available for exact
calculations; no tool is mandatory. Preserve the specified model, time boundary,
units and decision costs. Report your own supported quantities with the requested
keys. Disclose changes and limitations. Never replace a missing result with a claim
of success. This is an evaluation of recommendations, not permission to take external actions.
"""


def prepare(args):
    from gnomon.product_contract import __version__
    if __version__ != "1.2.0" and not args.development_only:
        raise ValueError(f"1.2.0 required; imported runtime is {__version__}. Locate release; do not relabel it.")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    generate(out / "corpus")
    (out / "prompt.txt").write_text(PROMPT)
    options = {"software_image": args.software_image, "service_image": args.service_image,
               "docker_host": "unix:///var/run/docker.sock", "execution_options": {"ledger": False, "temporal": False}}
    provider = {"schema_version": 1, "llm": {"base_url": args.base_url, "token_env": args.token_env},
        "backends": {
            "ordinary": {"factory": "benchmarks.workflow.software_backend:make", "options": {
                "image": args.software_image, "docker_host": "unix:///var/run/docker.sock"}},
            "lean": {"factory": "benchmarks.workflow.business_utility.backend:lean", "options": options},
            "full": {"factory": "benchmarks.workflow.business_utility.backend:full", "options": options}}}
    # Existing CombinedBackend takes execution_options within its options object.
    (out / "providers.json").write_text(json.dumps(provider, indent=2) + "\n")
    guidance = "Choose any available tool and submit your own supported answer."
    command = [sys.executable, str(ROOT / "benchmarks/workflow/driver.py")]
    if args.allow_model_requests and not args.development_only:
        command.append("--allow-model-requests")
    files = [ROOT / "benchmarks/workflow/driver.py", *sorted(Path(__file__).parent.glob("*.py")),
             *sorted(Path(__file__).parent.glob("*.md")), out / "corpus/private-audit.json", out / "corpus/manifest.json"]
    spec = {"schema_version": 1, "evidence_kind": "agent", "command": command,
        "driver_files": [str(p.resolve()) for p in files],
        "common": {"model": {"id": args.model, "revision": None},
            "generation": {"temperature": 0, "max_output_tokens": 2048},
            "prompt_file": "prompt.txt", "provider_config_file": "providers.json",
            "budget": {"timeout_seconds": 180, "jobs": 1, "infrastructure_retries": 0,
                       "max_tool_calls": 12, "max_rounds": 12, "max_tokens": 32000,
                       "max_reported_cost_usd": args.per_arm_stop}},
        "arms": {
            "ordinary": {"description": "Ordinary isolated Python and forecasting software.", "tool_contract": "Existing ordinary Python backend; exact computation allowed.", "guidance": guidance},
            "lean": {"description": "Same Python plus release Gnomon MCP.", "tool_contract": "Existing isolated execution MCP; optional ledger/temporal discovery disabled.", "guidance": guidance},
            "full": {"description": "Same Python and MCP with declared benchmark threshold prototype.", "tool_contract": "Same as lean, with benchmark_supplied_quantiles forecast overload only for supplied quantile cases; per-step marginals only.", "guidance": guidance}}}
    experiment = out / "experiment.json"
    experiment.write_text(json.dumps(spec, indent=2) + "\n")
    order = ["ordinary", "lean", "full"]
    random.Random(20260915).shuffle(order)
    (out / "launch-plan.json").write_text(json.dumps({"status": "development_only_no_confirmatory_execution" if args.development_only else "pending_commit_and_resource_preflight",
        "runtime_version": __version__, "evaluation_order": ["eval1", "eval2", "eval3"], "arm_order": order,
        "workers": 1, "retries": 0, "per_arm_per_evaluation_reported_stop": args.per_arm_stop,
        "total_nine_allocations": 9 * args.per_arm_stop, "hard_billing_cap": False}, indent=2) + "\n")
    # Validate integration with the same identity function used during execution.
    from benchmarks.workflow.matched import prepare as matched_prepare
    from benchmarks.workflow.schema import load_cases
    for evaluation in ("eval1", "eval2", "eval3"):
        for arm in order:
            matched_prepare(experiment, load_cases(out / f"corpus/{evaluation}.jsonl"),
                            command=shlex.join(command), arm=arm, timeout=180, jobs=1, retries=0)
    print(json.dumps({"prepared": str(out), "model_requests": 0, "runtime_version": __version__,
                      "development_only": args.development_only}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "model", "base-url", "token-env", "software-image", "service-image"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--per-arm-stop", type=float, required=True)
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--allow-model-requests", action="store_true")
    prepare(parser.parse_args())

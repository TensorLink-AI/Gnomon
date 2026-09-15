"""Durably launch the registered nine existing workflow passes, sequentially."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from benchmarks.workflow.provenance import atomic_write_text


def credential(path, name):
    values = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(name + "="):
            value = line.split("=", 1)[1].strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values.append(value)
    if len(values) != 1 or not values[0]:
        raise ValueError("explicit credential file must contain one nonempty named key")
    return values[0]


def run(prepared, output, *, credential_file=None):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    plan = json.loads((prepared / "launch-plan.json").read_text())
    if plan["status"] == "development_only_no_confirmatory_execution":
        raise ValueError("development configuration cannot dispatch live trials")
    if plan["evaluation_order"] != ["eval1", "eval2", "eval3"] or sorted(plan["arm_order"]) != ["full", "lean", "ordinary"]:
        raise ValueError("launch plan differs from the registered evaluation/arm set")
    output.mkdir(parents=True, exist_ok=False)
    status = {"state": "starting", "started_unix": time.time(), "pid": os.getpid(),
              "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
              "start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().rpartition(")")[2].split()[19],
              "completed_passes": [], "automatic_retries": 0, "prepared": str(prepared)}
    def save():
        atomic_write_text(output / "status.json", json.dumps(status, indent=2) + "\n")
    env = dict(os.environ)
    if credential_file:
        config = json.loads((prepared / "providers.json").read_text())
        name = config["llm"]["token_env"]
        env[name] = credential(credential_file, name)
    process = None
    try:
        for evaluation in plan["evaluation_order"]:
            for arm in plan["arm_order"]:
                status.update(state="running", evaluation=evaluation, arm=arm)
                save()
                destination = output / "runs" / evaluation / arm
                argv = [sys.executable, "-m", "benchmarks.workflow.business_utility.monitor",
                        "--experiment", str(prepared / "experiment.json"),
                        "--cases", str(prepared / f"corpus/{evaluation}.jsonl"),
                        "--arm", arm, "--output-dir", str(destination)]
                with (output / f"{evaluation}-{arm}.log").open("x") as log:
                    process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
                    code = process.wait()
                # The workflow's exit 2 includes graded failures, not just infra.
                # Keep these rows and continue; never rerun them to improve a score.
                if code not in (0, 2) or not (destination / "summary.json").is_file():
                    status.update(state="blocked", returncode=code, cause="workflow_or_resource_stop")
                    save()
                    return 1
                status["completed_passes"].append({"evaluation": evaluation, "arm": arm, "returncode": code})
                save()
            from .report import build_report
            build_report(prepared / "corpus", output / "runs", output / "reports")
        status.update(state="complete", finished_unix=time.time())
        save()
        return 0
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()  # monitor handles SIGTERM and cleans its owned tree
            process.wait(timeout=15)
        status.update(state="blocked", cause=type(error).__name__)
        save()
        raise


def interrupted(signum, frame):
    raise KeyboardInterrupt("requested stop")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--credential-file")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(run(args.prepared, args.output, credential_file=args.credential_file))

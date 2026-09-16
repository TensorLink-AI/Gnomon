"""Run the A11 format diagnostic, then conditionally dispatch the frozen full run."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from benchmarks.workflow.provenance import atomic_write_text
from benchmarks.workflow.schema import load_cases, load_observations
from .dispatch import credential, interrupted
from .report import grade


def valid_decisions(cases, audits, observations):
    rows = {row.case_id: row for row in observations}
    if len(rows) != len(observations) or set(rows) != {case.id for case in cases}:
        return False
    for case in cases:
        row = rows[case.id]
        result = grade(case, audits[case.id], row)
        if row.status != "answered" or not result["answered"] or result["unauditable"]:
            return False
        if row.choices.get("plan") != ("approve" if row.numbers["reported_nmae"] <= 1 else "review"):
            return False
    return True


def run(diagnostic, prepared, output, credential_file):
    diagnostic, prepared, output = map(lambda p: Path(p).resolve(), (diagnostic, prepared, output))
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    name = json.loads((diagnostic / "providers.json").read_text())["llm"]["token_env"]
    env[name] = credential(credential_file, name)
    status = {"state": "diagnostic", "pid": os.getpid(), "automatic_retries": 0}
    def save():
        atomic_write_text(output / "status.json", json.dumps(status, indent=2) + "\n")
    process = None
    try:
        save()
        plan = json.loads((diagnostic / "launch-plan.json").read_text())
        cases = load_cases(diagnostic / "corpus/eval1.jsonl")
        audits = json.loads((diagnostic / "corpus/private-audit.json").read_text())
        passed = True
        for arm in plan["arm_order"]:
            status["arm"] = arm
            save()
            destination = output / "diagnostic" / arm
            with (output / f"{arm}.log").open("x") as log:
                process = subprocess.Popen([sys.executable, "-m", "benchmarks.workflow.business_utility.monitor",
                    "--experiment", str(diagnostic / "experiment.json"), "--cases", str(diagnostic / "corpus/eval1.jsonl"),
                    "--arm", arm, "--output-dir", str(destination)], env=env, stdout=log, stderr=subprocess.STDOUT)
                code = process.wait()
            if code not in (0, 2) or not (destination / "summary.json").exists():
                status.update(state="blocked", reason="diagnostic_infrastructure_failure")
                save()
                return 1
            passed = valid_decisions(cases, audits, load_observations(destination / "observations.jsonl")) and passed
        status.update(state="format_gate_passed" if passed else "blocked", format_gate_passed=passed)
        save()
        if not passed:
            return 2
        status["state"] = "full_running"
        save()
        with (output / "full.log").open("x") as log:
            process = subprocess.Popen([sys.executable, "-m", "benchmarks.workflow.business_utility.dispatch",
                "--prepared", str(prepared), "--output", str(output / "full")], env=env,
                stdout=log, stderr=subprocess.STDOUT)
            code = process.wait()
        status.update(state="complete" if code == 0 else "blocked", returncode=code)
        save()
        return code
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=20)
        status["state"] = "interrupted"
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic", "prepared", "output", "credential-file"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(run(args.diagnostic, args.prepared, args.output, args.credential_file))

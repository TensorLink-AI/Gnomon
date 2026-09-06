"""Matched experiment controls, distinct from historical profile smoke tests.

This pins requested settings and local code, not remote weights or an arbitrary
agent driver's honesty. Deliberate surface differences are part of the experiment
identity; all other settings must be shared by its three arms.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import distributions
import argparse
import json
from pathlib import Path
import shlex
import sys
import statistics

from benchmarks.workflow.agent_metrics import _decode_record, compare_rows
from .provenance import corpus_sha256
from .accounting import reported_cost_limit

ARMS = ("ordinary", "lean", "full")
ROOT = Path(__file__).resolve().parents[2]


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _keys(value, required, context):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError(f"{context} requires exactly {sorted(required)}")


def _text(value, context):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a nonempty string")
    return value


def _file(base, value):
    path = (base / _text(value, "file path")).resolve()
    if not path.is_file():
        raise ValueError(f"experiment file does not exist: {path}")
    return path


def _content(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_contents():
    # Concrete bytes distinguish two dirty worktrees at the same Git revision.
    # Hash benchmark helpers too: a changed grader/normalizer changes the run.
    paths = [ROOT / "pyproject.toml"]
    for directory in (ROOT / "src/gnomon", ROOT / "benchmarks"):
        paths.extend(directory.rglob("*.py"))
    return {str(path.relative_to(ROOT)): _content(path) for path in sorted(paths)}


def prepare(path, cases, *, command, arm, timeout, jobs, retries):
    """Validate one common driver/configuration and pin every requested arm.

    Paths are operator inputs. Credentials belong in environment variables, never
    in this public experiment document. Provider configuration contents are hashed,
    not copied to reports. The driver must read the supplied experiment context.
    """
    path = Path(path).resolve()
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("matched experiment requires nonempty unique cases")
    if any(case.stages for case in cases):
        raise ValueError("matched mode does not use historical host-compiled stages; supply single-stage tasks or committed episodes")
    spec = _decode_record(path.read_text(encoding="utf-8"))
    _keys(spec, {"schema_version", "evidence_kind", "command", "driver_files",
                 "common", "arms"}, "matched experiment")
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("unsupported matched experiment schema")
    if spec["evidence_kind"] not in {"scripted", "agent"}:
        raise ValueError("evidence_kind must be scripted or agent")
    if arm not in ARMS:
        raise ValueError(f"matched experiment arm must be one of {ARMS}")
    _keys(spec["arms"], ARMS, "arms")
    for name, surface in spec["arms"].items():
        _keys(surface, {"description", "tool_contract", "guidance"}, f"arm {name}")
        for key in surface:
            _text(surface[key], f"arm {name} {key}")
    argv = spec["command"]
    if not isinstance(argv, list) or len(argv) < 2 or any(not isinstance(x, str) or not x for x in argv):
        raise ValueError("command must be a Python interpreter and driver script argv")
    # One interpreter is deliberately required: otherwise host dependency/version
    # measurements could describe a different environment from the agent process.
    if Path(argv[0]).resolve() != Path(sys.executable).resolve():
        raise ValueError("matched driver must use the harness Python interpreter")
    driver = _file(path.parent, argv[1])
    if any(case.episode for case in cases) and (driver != ROOT / "benchmarks/workflow/driver.py" or retries):
        raise ValueError("episodes require the built-in shared driver with zero infrastructure retries")
    argv = [sys.executable, str(driver), *argv[2:]]
    if shlex.split(command) != argv:
        raise ValueError("arm-command differs from the experiment's common command")
    declared = spec["driver_files"]
    if not isinstance(declared, list) or not declared:
        raise ValueError("driver_files must name the driver and its local dependencies")
    files = [_file(path.parent, item) for item in declared]
    if len(set(files)) != len(files) or driver not in files:
        raise ValueError("driver_files must be unique and include the invoked script")
    common = spec["common"]
    _keys(common, {"model", "generation", "prompt_file", "provider_config_file", "budget"}, "common")
    _keys(common["model"], {"id", "revision"}, "model")
    _text(common["model"]["id"], "model id")
    if common["model"]["revision"] is not None:
        _text(common["model"]["revision"], "model revision")
    if not isinstance(common["generation"], dict):
        raise ValueError("generation must explicitly declare driver generation settings")
    budget = common["budget"]
    spending = reported_cost_limit(budget)
    _keys(budget, {"timeout_seconds", "jobs", "infrastructure_retries", "max_tool_calls",
                   "max_rounds", "max_tokens"} | ({"max_reported_cost_usd"} if spending is not None else set()), "budget")
    if spending is not None and (jobs != 1 or retries or driver != ROOT / "benchmarks/workflow/driver.py"):
        raise ValueError("reported-cost control requires the built-in driver, one worker and zero retries")
    for key, value in budget.items():
        if key == "max_reported_cost_usd":
            continue
        elif key == "timeout_seconds":
            if type(value) not in (int, float) or not 0 < value <= 3600:
                raise ValueError("timeout_seconds must be positive and at most 3600")
        elif type(value) is not int or value < (0 if key in {"infrastructure_retries", "max_tool_calls"} else 1):
            raise ValueError(f"invalid experiment budget {key}")
    if (timeout, jobs, retries) != (budget["timeout_seconds"], budget["jobs"], budget["infrastructure_retries"]):
        raise ValueError("CLI budgets differ from common experiment budgets")
    prompt = _file(path.parent, common["prompt_file"])
    provider = _file(path.parent, common["provider_config_file"])
    pinned = {
        "schema_version": 1, "evidence_kind": spec["evidence_kind"], "command": argv,
        "driver_files": {str(p): _content(p) for p in files},
        "code_files": code_contents(), "corpus_sha256": corpus_sha256(cases),
        "case_ids": [case.id for case in cases],
        "python": sys.version, "python_binary_sha256": _content(Path(sys.executable).resolve()),
        "dependencies": sorted({(d.metadata["Name"], d.version) for d in distributions()}),
        "common": {**common, "prompt_file": str(prompt), "prompt_sha256": _content(prompt),
                   "provider_config_file": str(provider), "provider_config_sha256": _content(provider)},
        "arms": spec["arms"],
    }
    # Normalize tuples before comparison with a JSON checkpoint.
    pinned = json.loads(json.dumps(pinned, allow_nan=False))
    return {"experiment_id": fingerprint(pinned), "arm": arm, "pinned": pinned}


def public_context(experiment):
    common = experiment["pinned"]["common"]
    return {"experiment_id": experiment["experiment_id"], "arm": experiment["arm"],
            "evidence_kind": experiment["pinned"]["evidence_kind"], "common": common,
            "surface": experiment["pinned"]["arms"][experiment["arm"]]}


def artifact_hashes(directory):
    return {name: _content(Path(directory) / name)
            for name in ("observations.jsonl", "observations.attempts.sqlite3")}


def load_summary(directory):
    summary = _decode_record((Path(directory) / "summary.json").read_text(encoding="utf-8"))
    if summary.get("matched_artifacts") != artifact_hashes(directory):
        raise ValueError("matched observations/journal changed after scoring; re-score the complete run")
    return summary


def normalized_rows(summary):
    """Use all delivered tasks and nullable complete totals, as in agent_eval."""
    return [{"task_id": row["case_id"], "success": row["correctness"] == 1.0,
             "completed": row["execution_state"] == "completed",
             "error": row["execution_state"] in {"task_error", "infrastructure_failure"},
             "temporal_leakage": row["temporal_leakage"],
             "budget_exceeded": row["resource_accounting"].get("budget_exceeded"),
             "warning_omission": not row["disclosures_pass"],
             "appropriate_abstention": row["disposition_correct"] and row["abstained"],
             "resource_accounting": row["resource_accounting"], "trust_pass": row["trust_pass"],
             "usable": row["usable"], "correctness": row["correctness"],
             **({"forecast_metrics": row["forecast_metrics"]} if "forecast_metrics" in row else {}),
             **{target: row["resource_accounting"]["resources"][source]["total"]
                for source, target in (("tool_calls", "tool_calls"), ("cumulative_tokens", "run_tokens"),
                                       ("latency_seconds", "latency_seconds"), ("cost_usd", "cost_usd"))}}
            for row in summary["rows"]]


def compare(summaries):
    """Refuse mixed controls before calculating any between-arm differences."""
    _keys(summaries, ARMS, "comparison arms")
    contracts = []
    for arm, summary in summaries.items():
        contract = summary.get("matched_experiment")
        if not isinstance(contract, dict) or set(contract) != {"experiment_id", "arm", "pinned"}:
            raise ValueError("missing matched experiment contract")
        pinned = contract["pinned"]
        if contract["arm"] != arm or summary["arm"] != arm:
            raise ValueError("experiment arm identity mismatch")
        if fingerprint(pinned) != contract["experiment_id"]:
            raise ValueError("experiment content fingerprint mismatch")
        if summary["corpus_sha256"] != pinned["corpus_sha256"]:
            raise ValueError("experiment corpus differs from scored corpus")
        ids = [row["case_id"] for row in summary["rows"]]
        if summary["missing"] or sorted(ids) != sorted(pinned["case_ids"]):
            raise ValueError("matched experiment requires one row per planned task, including errors")
        contracts.append(contract)
    if any(item["pinned"] != contracts[0]["pinned"] for item in contracts[1:]):
        raise ValueError("uncontrolled experiment differences; requested settings/code/corpus must match")
    scripted = contracts[0]["pinned"]["evidence_kind"] == "scripted"
    rows = {arm: normalized_rows(summary) for arm, summary in summaries.items()}
    return {"schema_version": 1, "experiment_id": contracts[0]["experiment_id"],
            "evidence_kind": "scripted" if scripted else "agent",
            "comparability": "matched_requested_configuration_not_runtime_attestation",
            "deliberate_differences": contracts[0]["pinned"]["arms"],
            "forecast_error": forecast_error_summary(rows),
            "limitations": ["Driver compliance and remote weights are not independently attested.",
                            "Driver-owned token/tool budgets are requested limits; only process timeout is host enforced.",
                            "No randomized execution-order or independent-task guarantee is inferred.",
                            "Scripted checks are integration evidence, not LLM accuracy/cost uplift."
                            if scripted else "Agent results apply to this declared task/model/settings cohort only."],
            "comparisons": {f"{left}_vs_{right}": compare_rows(rows[left], rows[right])
                            for left, right in (("ordinary", "lean"), ("ordinary", "full"), ("lean", "full"))}}


def forecast_error_summary(rows):
    """Conditional loss and its coverage, alongside all-task success/cost above."""
    selected = {arm: {row["task_id"]: row["forecast_metrics"] for row in values if "forecast_metrics" in row}
                for arm, values in rows.items()}
    ids = set(selected[ARMS[0]])
    if any(set(values) != ids for values in selected.values()):
        raise ValueError("forecast task membership differs across arms")
    complete = {arm: {key: value for key, value in values.items() if value["complete"]}
                for arm, values in selected.items()}
    shared = sorted(ids.intersection(*(set(values) for values in complete.values())))
    def mean(values):
        return statistics.mean(values) if values else None
    return {"planned_forecasts": len(ids), "all_arm_complete_forecasts": len(shared),
            "basis": "conditional_complete_forecasts_not_missing_as_zero_or_all_task_uplift",
            "arms": {arm: {"complete_forecasts": len(values), "missing_or_invalid": len(ids) - len(values),
                            "mean_mase_complete_only": mean([value["mase"] for value in values.values()]),
                            "mean_mase_all_arm_complete_only": mean([values[key]["mase"] for key in shared])}
                     for arm, values in complete.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(f"--{arm}", type=Path, required=True, help="matched arm output directory")
    args = parser.parse_args()
    summaries = {arm: load_summary(getattr(args, arm)) for arm in ARMS}
    print(json.dumps(compare(summaries), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

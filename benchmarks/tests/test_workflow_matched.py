"""Independent controls tests; the scripted driver is not an LLM ablation."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from benchmarks.workflow.matched import ARMS, compare, fingerprint, prepare, load_summary
from benchmarks.workflow.run_workflow import main
from benchmarks.workflow.schema import Case


@pytest.fixture
def experiment(tmp_path):
    driver = tmp_path / "driver.py"
    driver.write_text('''import json, sys
case = json.load(sys.stdin)
assert "oracle" not in case
control = case["experiment"]
assert control["common"]["model"]["id"] == "scripted-arithmetic-v1"
values = case["available_at_cutoff"]["series"]
failed = control["arm"] == "ordinary" and case["id"] == "failed"
print(json.dumps({"case_id": case["id"], "status": "error" if failed else "answered",
    "support": "abstained" if failed else "supported",
    "numbers": {} if failed else {"mean": sum(values) / len(values)},
    "tool_calls": 0, "cumulative_tokens": 0, "response_tokens": 0,
    "latency_seconds": 0.01, "cost_usd": 0, "temporal_leakage": False,
    "metadata": {"received_experiment": control["experiment_id"]}}))
''', encoding="utf-8")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Calculate from supplied observations; report uncertainty honestly.", encoding="utf-8")
    provider = tmp_path / "providers.toml"
    provider.write_text("schema_version=1\n", encoding="utf-8")
    spec = {"schema_version": 1, "evidence_kind": "scripted",
            "command": [sys.executable, str(driver)], "driver_files": [str(driver)],
            "common": {"model": {"id": "scripted-arithmetic-v1", "revision": None},
                       "generation": {"temperature": 0, "seed": 7},
                       "prompt_file": str(prompt), "provider_config_file": str(provider),
                       "budget": {"timeout_seconds": 5, "jobs": 1, "infrastructure_retries": 0,
                                  "max_rounds": 2, "max_tool_calls": 2, "max_tokens": 500}},
            "arms": {arm: {"description": f"{arm} scripted control test",
                           "tool_contract": "none; this fixture does not measure a Gnomon profile",
                           "guidance": "Follow the common task instructions."} for arm in ARMS}}
    case = Case.from_dict({"id": "mean", "kind": "synthetic", "domain": "arithmetic",
                          "question": "Return the arithmetic mean.",
                          "available_at_cutoff": {"series": [1, 7, 4]},
                          "answer_schema": {"numbers": ["mean"]}, "oracle": {"numbers": {"mean": 4}}})
    cases = [case, replace(case, id="failed")]
    spec_path, cases_path = tmp_path / "experiment.json", tmp_path / "cases.jsonl"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    cases_path.write_text("".join(json.dumps(asdict(c)) + "\n" for c in cases), encoding="utf-8")
    return spec_path, spec, cases_path, cases


def contract(experiment, arm="lean", **overrides):
    path, spec, _, cases = experiment
    return prepare(path, cases, **{"command": shlex.join(spec["command"]), "arm": arm,
                                  "timeout": 5, "jobs": 1, "retries": 0, **overrides})


def test_arms_share_controls_and_concrete_code_not_dirty_marker(experiment):
    contracts = [contract(experiment, arm) for arm in ARMS]
    assert len({c["experiment_id"] for c in contracts}) == 1
    pinned = contracts[0]["pinned"]
    assert "src/gnomon/session.py" in pinned["code_files"]
    assert "benchmarks/workflow/scoring.py" in pinned["code_files"]
    assert pinned["common"]["model"]["revision"] is None
    assert "schema_version=1" not in json.dumps(pinned)  # provider body is not copied
    assert pinned["python_binary_sha256"] and pinned["dependencies"]


@pytest.mark.parametrize("field", ["prompt_file", "provider_config_file", "driver"])
def test_concrete_input_edits_change_identity_even_without_a_git_commit(experiment, field):
    before = contract(experiment)
    path, spec, _, _ = experiment
    changed = Path(spec["command"][1] if field == "driver" else spec["common"][field])
    changed.write_text(changed.read_text() + "\n# changed\n")
    assert contract(experiment)["experiment_id"] != before["experiment_id"]


@pytest.mark.parametrize("override", [{"timeout": 6}, {"jobs": 2}, {"retries": 1},
                                      {"command": "different-agent"}, {"arm": "core"}])
def test_uncontrolled_cli_differences_are_refused(experiment, override):
    with pytest.raises(ValueError):
        contract(experiment, **override)


@pytest.mark.parametrize("mutation", [
    lambda s: s["arms"]["lean"].update(model="different"),
    lambda s: s["common"].update(hidden_prompt="extra"),
    lambda s: s.update(evidence_kind="validated_uplift"),
    lambda s: s["common"]["budget"].update(max_tokens=True),
    lambda s: s["common"]["budget"].update(timeout_seconds=0),
    lambda s: s["command"].__setitem__(0, "/different/python"),
    lambda s: s.update(driver_files=[]),
])
def test_closed_experiment_schema_rejects_hidden_overrides(experiment, mutation):
    path, spec, _, _ = experiment
    mutation(spec)
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        contract(experiment)


def test_staged_legacy_host_compilation_is_not_mislabelled_an_agent_experiment(experiment):
    path, spec, _, cases = experiment
    with pytest.raises(ValueError, match="host-compiled"):
        prepare(path, [replace(cases[0], stages=({"name": "outcome"},))],
                command=shlex.join(spec["command"]), arm="lean", timeout=5, jobs=1, retries=0)


def run_arms(experiment, monkeypatch, tmp_path):
    path, spec, cases_path, _ = experiment
    summaries = {}
    for arm in ARMS:
        argv = ["workflow", "--cases", str(cases_path), "--experiment", str(path),
                "--arm-command", shlex.join(spec["command"]), "--arm", arm,
                "--output-dir", str(tmp_path / arm), "--timeout", "5", "--infrastructure-retries", "0"]
        monkeypatch.setattr(sys, "argv", argv)
        assert main() in {0, 2}  # release-policy failure is data, not missing output
        summaries[arm] = json.loads((tmp_path / arm / "summary.json").read_text())
        observations = [json.loads(line) for line in (tmp_path / arm / "observations.jsonl").read_text().splitlines()]
        assert all(row["metadata"]["received_experiment"] == summaries[arm]["matched_experiment"]["experiment_id"]
                   for row in observations)
    return summaries


def test_real_driver_processes_receive_common_inputs_and_keep_failed_tasks(experiment, monkeypatch, tmp_path):
    summaries = run_arms(experiment, monkeypatch, tmp_path)
    compared = compare(summaries)
    pair = compared["comparisons"]["ordinary_vs_lean"]
    assert pair["tasks_total"] == 2
    assert pair["baseline"]["task_success"] == 0.5
    assert pair["treatment"]["task_success"] == 1.0
    assert pair["baseline"]["resources"]["latency_seconds"]["total"] == 0.02
    assert compared["evidence_kind"] == "scripted"
    assert any("not LLM" in limitation for limitation in compared["limitations"])
    result = subprocess.run([sys.executable, "-m", "benchmarks.workflow.matched",
                             *[item for arm in ARMS for item in (f"--{arm}", str(tmp_path / arm))]],
                            text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == compared


def test_comparison_rejects_changed_controls_missing_rows_and_forged_identity(experiment, monkeypatch, tmp_path):
    summaries = run_arms(experiment, monkeypatch, tmp_path)
    for key in ("model", "generation", "budget", "prompt_sha256", "provider_config_sha256"):
        changed = deepcopy(summaries)
        pin = changed["lean"]["matched_experiment"]["pinned"]
        pin["common"][key] = "changed"
        changed["lean"]["matched_experiment"]["experiment_id"] = fingerprint(pin)
        with pytest.raises(ValueError, match="uncontrolled"):
            compare(changed)
    for key in ("code_files", "driver_files", "dependencies", "arms"):
        changed = deepcopy(summaries)
        pin = changed["full"]["matched_experiment"]["pinned"]
        pin[key] = "different"
        changed["full"]["matched_experiment"]["experiment_id"] = fingerprint(pin)
        with pytest.raises(ValueError, match="uncontrolled"):
            compare(changed)
    changed = deepcopy(summaries)
    changed["lean"]["rows"].pop()
    with pytest.raises(ValueError, match="one row per planned task"):
        compare(changed)
    changed = deepcopy(summaries)
    changed["lean"]["matched_experiment"]["experiment_id"] = "forged"
    with pytest.raises(ValueError, match="fingerprint"):
        compare(changed)


def test_changed_prompt_prevents_resume_before_starting_another_attempt(experiment, monkeypatch, tmp_path):
    run_arms(experiment, monkeypatch, tmp_path)
    # sys.argv is the final full-arm invocation from run_arms.
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--resume"])
    path, spec, _, _ = experiment
    Path(spec["common"]["prompt_file"]).write_text("changed prompt")
    before = (tmp_path / "full" / "observations.attempts.sqlite3").read_bytes()
    with pytest.raises(SystemExit, match="identity mismatch"):
        main()
    assert (tmp_path / "full" / "observations.attempts.sqlite3").read_bytes() == before


def test_unchanged_successful_resume_does_not_reinvoke_driver(experiment, monkeypatch, tmp_path):
    from benchmarks.workflow import run_workflow
    run_arms(experiment, monkeypatch, tmp_path)
    original = (tmp_path / "full" / "observations.jsonl").read_bytes()
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--resume"])
    monkeypatch.setattr(run_workflow, "_invoke", lambda *_a, **_k: pytest.fail("successful case was reinvoked"))
    assert main() in {0, 2}
    assert (tmp_path / "full" / "observations.jsonl").read_bytes() == original
    assert load_summary(tmp_path / "full")["arm"] == "full"


def test_midrun_input_change_retains_attempts_but_refuses_matched_summary(experiment, monkeypatch, tmp_path):
    from benchmarks.workflow import run_workflow
    path, spec, cases_path, _ = experiment
    original = run_workflow._run_one

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        Path(spec["common"]["prompt_file"]).write_text("changed during run")
        return result

    monkeypatch.setattr(run_workflow, "_run_one", changed)
    output = tmp_path / "changed"
    monkeypatch.setattr(sys, "argv", ["workflow", "--cases", str(cases_path), "--experiment", str(path),
                        "--arm-command", shlex.join(spec["command"]), "--arm", "lean",
                        "--output-dir", str(output), "--timeout", "5", "--infrastructure-retries", "0"])
    with pytest.raises(ValueError, match="changed during execution"):
        main()
    assert not (output / "summary.json").exists()
    assert len((output / "observations.jsonl").read_text().splitlines()) == 2
    assert (output / "observations.attempts.sqlite3").is_file()


@pytest.mark.parametrize("artifact", ["observations.jsonl", "observations.attempts.sqlite3"])
def test_changed_attempt_artifact_invalidates_old_summary(experiment, monkeypatch, tmp_path, artifact):
    run_arms(experiment, monkeypatch, tmp_path)
    path = tmp_path / "full" / artifact
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed after scoring"):
        load_summary(tmp_path / "full")


def test_source_bytes_are_included_even_when_revision_labels_are_unchanged(experiment, monkeypatch, tmp_path):
    from benchmarks.workflow import matched
    root = tmp_path / "source"
    (root / "src/gnomon").mkdir(parents=True)
    (root / "benchmarks").mkdir()
    (root / "pyproject.toml").write_text("[project]\n")
    source = root / "src/gnomon/runtime.py"
    source.write_text("value = 1\n")
    monkeypatch.setattr(matched, "ROOT", root)
    before = contract(experiment)
    source.write_text("value = 2\n")
    assert contract(experiment)["experiment_id"] != before["experiment_id"]


def test_historical_consumers_cannot_bypass_matched_controls(experiment, monkeypatch, tmp_path):
    from benchmarks.common.manifest import incompatibilities
    from benchmarks.workflow.compare import compare as historical_compare
    summaries = run_arms(experiment, monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="matched experiments require"):
        historical_compare(list(summaries.values()), "ordinary")
    manifest = json.loads((tmp_path / "lean" / "manifest.json").read_text())
    assert incompatibilities(manifest, manifest)

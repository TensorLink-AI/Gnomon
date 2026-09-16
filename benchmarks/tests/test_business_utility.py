"""Offline contract checks; none of these scripted checks measure agent uplift."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.workflow.bounded_agent import model_visible_case, ToolReply
from benchmarks.workflow.business_utility.backend import ThresholdReferenceBackend, PROVIDER
from benchmarks.workflow.business_utility.corpus import generate, timestamps
from benchmarks.workflow.business_utility.reference import threshold_probability, action, expected_cost
from benchmarks.workflow.business_utility.report import grade, paired_effect, holm, build_report
from benchmarks.workflow.schema import load_cases, Observation


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    path = tmp_path_factory.mktemp("utility") / "corpus"
    generate(path)
    return path


def test_known_distribution_and_cost_orientation():
    knots = [[0, 0], [.25, 10], [1, 20]]
    assert threshold_probability(knots, level=15, direction="below")["probability"] == .625
    assert threshold_probability(knots, level=15, direction="above")["probability"] == .375
    assert action(.90, 20, 1) == "do_not_act"
    assert action(.99, 20, 1) == "act"
    assert action(20/21, 20, 1) == "do_not_act"
    assert expected_cost("act", .9, 20, 1) == pytest.approx(2)


@pytest.mark.parametrize("knots,kwargs", [
    ([[.1, 0], [.9, 1]], {}), ([[0, 0], [.5, 2], [1, 1]], {}),
    ([[0, 0], [1, 0]], {}), ([[0, 0], [1, float("nan")]], {}),
    ([[0, 0], [1, 1]], {"event": "any_time"}),
    ([[0, 0], [1, 1]], {"level": float("inf")}),
    ([[0, 0], [1, 1]], {"direction": "either"}),
])
def test_reference_refuses_unidentified_or_invalid_requests(knots, kwargs):
    with pytest.raises(ValueError):
        threshold_probability(knots, **{"level": .5, "direction": "above", **kwargs})


def test_corpus_counts_determinism_and_repeat_prompts(corpus, tmp_path):
    second = tmp_path / "second"
    generate(second)
    assert {p.name: p.read_bytes() for p in corpus.iterdir()} == {p.name: p.read_bytes() for p in second.iterdir()}
    assert [len(load_cases(corpus / f"eval{i}.jsonl")) for i in (1, 2, 3)] == [80, 128, 80]
    repeats = load_cases(corpus / "eval2.jsonl")[:8]
    public = []
    for c in repeats:
        value = asdict(c)
        value.pop("oracle")
        public.append(model_visible_case(value))
    assert all(value == public[0] for value in public)
    assert "id" not in public[0]
    assert "hide_case_id_from_model" not in public[0]["available_at_cutoff"]
    assert "retain_tool_results" not in public[0]["available_at_cutoff"]


def test_snapshot_replay_and_no_holdout_rows(corpus, tmp_path):
    from gnomon.data_refs import DataReferences
    from datetime import datetime
    audits = json.loads((corpus / "private-audit.json").read_text())
    for c in load_cases(corpus / "eval1.jsonl"):
        public = c.available_at_cutoff
        file = tmp_path / "history.gnomon"
        file.write_text(public["files"]["history.gnomon"])
        refs = DataReferences()
        first = refs.inspect(str(file))
        second = refs.inspect(str(file))
        assert first["data_ref"] == second["data_ref"]
        frozen = refs._get(first["data_ref"])
        snapshot = frozen.loaded.snapshot
        assert max(r.valid_time for r in snapshot.retained_observations()) <= datetime.fromisoformat(public["as_of"])
        for i, origin in enumerate(public["replay_origins"]):
            rows = snapshot.narrow(as_of=datetime.fromisoformat(origin)).series("series", "value")
            assert rows[-1].value == audits[c.id]["causal_replay"][i]
            assert all(r.known_time <= datetime.fromisoformat(origin) for r in rows)


def test_fixed_repair_matches_actual_runtime(corpus, tmp_path):
    from gnomon.data_refs import DataReferences
    audits = json.loads((corpus / "private-audit.json").read_text())
    checked = set()
    for c in load_cases(corpus / "eval2.jsonl"):
        audit = audits[c.id]
        if audit["cluster"] in checked:
            continue
        checked.add(audit["cluster"])
        path = tmp_path / "messy.csv"
        path.write_text(c.available_at_cutoff["files"]["messy.csv"])
        refs = DataReferences()
        inspected = refs.inspect(str(path), frequency=c.available_at_cutoff["frequency"], repair="aggressive")
        frozen = refs._get(inspected["data_ref"])
        rows = next(iter(frozen.loaded.groups.values()))
        assert [r.value for r in rows] == pytest.approx([r[1] for r in audit["repaired_rows"]])


def test_missing_and_bad_answers_remain_in_denominator(corpus):
    c = load_cases(corpus / "eval3.jsonl")[0]
    audit = json.loads((corpus / "private-audit.json").read_text())[c.id]
    missing = grade(c, audit, None)
    assert missing["primary_failure"] and missing["failed_handoff"]
    assert not missing["bad_business_outcome"]
    wrong = Observation(c.id, "answered", "supported", numbers={"probability": .5}, choices={"action": "act"})
    r = grade(c, audit, wrong)
    assert r["bad_business_outcome"] and r["numeric"]["expected_excess_cost"] > 0
    invalid = Observation(c.id, "answered", "supported", numbers={"probability": 2}, choices={"action": "do_not_act"})
    assert grade(c, audit, invalid)["primary_failure"]


def test_fixed_policy_cannot_pass_with_fabricated_repaired_rows(corpus):
    c = load_cases(corpus / "eval2.jsonl")[0]
    a = json.loads((corpus / "private-audit.json").read_text())[c.id]
    o = Observation(c.id, "answered", "supported", numbers={"headline": a["reference"]},
        choices={"model": "historical_mean", "time_column": "timestamp", "target_column": "value"},
        facts={"repaired_rows": a["repaired_rows"], "repair_operations": a["operations"]}, disclosures=tuple(a["operations"]))
    assert not grade(c, a, o)["primary_failure"]
    bad = deepcopy(o)
    bad.facts["repaired_rows"][0][1] += 999
    assert grade(c, a, bad)["primary_failure"]


def test_prototype_wraps_real_inventory_and_preserves_real_calls(corpus):
    from gnomon import GnomonSession
    class Delegate:
        provenance = {"fixture": "real current in-process session, not a live agent"}
        def __init__(self): self.session = GnomonSession.from_config()
        def tools(self): return self.session.tools()
        def call(self, name, arguments, timeout): return ToolReply(self.session.call(name, arguments), 0)
        def close(self): self.session.close()
    c = load_cases(corpus / "eval3.jsonl")[0]
    delegate = Delegate()
    wrapped = ThresholdReferenceBackend(delegate, asdict(c))
    try:
        assert [t["name"] for t in wrapped.tools()] == [t["name"] for t in delegate.tools()]
        reply = wrapped.call("gnomon_forecast", {"provider": PROVIDER, "threshold": c.available_at_cutoff["threshold"]}, timeout=5)
        assert not reply.value["isError"]
        value = json.loads(reply.value["content"][0]["text"])
        assert value["per_step"][0]["probability"] == pytest.approx(c.oracle.numbers["probability"])
        assert value["prototype"] and value["per_step"][0]["path_probability"] is None
        refused = wrapped.call("gnomon_forecast", {"provider": PROVIDER, "threshold": {"event": "any_time"}}, timeout=5)
        assert refused.value["isError"]
        real = wrapped.call("gnomon_capabilities", {}, timeout=5)
        assert real.value
    finally:
        wrapped.close()


def test_statistics_and_empty_reports_are_not_uplift(corpus, tmp_path):
    a = [{"case_id": str(i), "source": "x", "cluster": str(i//2), "primary_failure": False, "observed": True} for i in range(8)]
    b = [{**r, "primary_failure": True} for r in a]
    effect = paired_effect(a, b, resamples=100)
    assert effect["risk_difference"] == -1 and effect["clusters"] == 4
    assert holm({"a": .01, "b": .03, "c": .9}) == {"a": .03, "b": .06, "c": .9}
    build_report(corpus, tmp_path / "absent", tmp_path / "reports")
    r = json.loads((tmp_path / "reports/eval1.json").read_text())
    assert r["status"] == "pending" and r["primary_effect"] is None
    assert r["arms"]["ordinary"]["planned"] == 80
    assert "Not measured" in (tmp_path / "reports/eval1.md").read_text()


def test_prepare_uses_existing_matched_identity_without_requests(tmp_path):
    from benchmarks.workflow.business_utility.prepare import prepare
    args = SimpleNamespace(output=str(tmp_path / "prepared"), model="scripted-placeholder-not-run",
        base_url="http://127.0.0.1:1/v1", token_env="UNSET_UTILITY_TOKEN", software_image="sha256:"+"a"*64,
        service_image="sha256:"+"b"*64, per_arm_stop=1, development_only=True, allow_model_requests=False)
    prepare(args)
    spec = json.loads((Path(args.output) / "experiment.json").read_text())
    assert spec["command"][1].endswith("benchmarks/workflow/driver.py")
    assert spec["common"]["budget"]["jobs"] == 1


def test_loop_retains_bounded_receipts_without_rewriting_answer(monkeypatch):
    from benchmarks.tests import test_bounded_agent as loop_tests
    from benchmarks.workflow.matched import fingerprint
    case = deepcopy(loop_tests.CASE)
    case["available_at_cutoff"].update(retain_tool_results=True, hide_case_id_from_model=True)
    monkeypatch.setattr(loop_tests, "CASE", case)
    result, requests, _ = loop_tests.execute(monkeypatch, [
        loop_tests.response(loop_tests.call("second", {"original": 4})),
        loop_tests.response(loop_tests.submit(99))])
    trace = result.metadata["trace"][0]
    assert trace["arguments"] == {"original": 4}
    assert trace["result"] == {"number": 17}
    assert trace["result_sha256"] == fingerprint(trace["result"])
    assert result.numbers == {"next": 99}
    assert '"id"' not in requests[0]["messages"][1]["content"]


def test_monitor_thresholds_and_release_gate(monkeypatch):
    from benchmarks.workflow.business_utility.monitor import resource_stop, GIB, run
    import gnomon.product_contract
    assert resource_stop(7*GIB, 0)
    assert resource_stop(16*GIB, 4*GIB)
    assert not resource_stop(8*GIB, 3*GIB)
    monkeypatch.setattr(gnomon.product_contract, "__version__", "1.1.9")
    with pytest.raises(ValueError, match="1.2.0"):
        run(SimpleNamespace())


def test_monitor_uses_container_memory_limit(tmp_path):
    from benchmarks.workflow.business_utility.monitor import memory_available, GIB
    proc, cg = tmp_path / "proc", tmp_path / "cgroup"
    proc.mkdir(); cg.mkdir()
    (proc / "meminfo").write_text("MemAvailable: 400000000 kB\n")
    (cg / "memory.max").write_text(str(50*GIB))
    (cg / "memory.current").write_text(str(49*GIB))
    assert memory_available(proc, cg) == GIB
    (cg / "memory.max").write_text("max")
    assert memory_available(proc, cg) == 400000000*1024


def test_monitor_stops_owned_separate_session_not_unrelated_process():
    import subprocess
    import sys
    from benchmarks.workflow.business_utility.monitor import descendants, stop_group
    script = ("import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'], "
              "start_new_session=True); print(p.pid,flush=True); time.sleep(120)")
    root = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    known = {}
    try:
        child = int(root.stdout.readline())
        descendants(root.pid, known)
        assert child in known and unrelated.pid not in known
        stop_group(root, known)
        assert root.poll() is not None and unrelated.poll() is None
        stat = Path(f"/proc/{child}/stat")
        if stat.exists():
            assert stat.read_text().rpartition(")")[2].split()[0] == "Z"
    finally:
        stop_group(root, known)
        unrelated.terminate()
        unrelated.wait(timeout=3)
        root.stdout.close()


@pytest.mark.parametrize("returncode,expected_passes,expected_state", [(2, 9, "complete"), (75, 0, "blocked")])
def test_dispatch_keeps_graded_failures_and_never_retries_infrastructure(tmp_path, monkeypatch, returncode, expected_passes, expected_state):
    from benchmarks.workflow.business_utility import dispatch, report
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "launch-plan.json").write_text(json.dumps({"status": "pending_commit_and_resource_preflight",
        "evaluation_order": ["eval1", "eval2", "eval3"], "arm_order": ["full", "ordinary", "lean"]}))
    calls = []
    class Process:
        def __init__(self, argv, **kwargs):
            calls.append(argv)
            destination = Path(argv[argv.index("--output-dir")+1])
            destination.mkdir(parents=True)
            if returncode == 2:
                (destination / "summary.json").write_text("{}")
                (destination / "observations.jsonl").write_text("")
        def wait(self, **kwargs): return returncode
        def poll(self): return returncode
    monkeypatch.setattr(dispatch.subprocess, "Popen", Process)
    monkeypatch.setattr(report, "build_report", lambda *args: None)
    dispatch.run(prepared, tmp_path / "out")
    status = json.loads((tmp_path / "out/status.json").read_text())
    assert status["state"] == expected_state
    assert len(status["completed_passes"]) == expected_passes
    assert len(calls) == (9 if expected_passes else 1)
    assert all(argv[2] == "benchmarks.workflow.business_utility.monitor" for argv in calls)


def test_capacity_decision_contract_and_invalid_value_remain_explicit(corpus):
    import statistics
    audits = json.loads((corpus / "private-audit.json").read_text())
    for c in load_cases(corpus / "eval1.jsonl"):
        assert '"plan" is the field name, not an allowed value' in c.question
        assert '"approve"' in c.question and '"review"' in c.question
        a = audits[c.id]
        replay = a["causal_replay"]
        nmae = statistics.fmean(abs(x-y) for x,y in zip(replay, c.available_at_cutoff["replay_actuals"])) / a["scale"]
        numbers = {**{f"replay_{i+1}": x for i,x in enumerate(replay)},
                   **{f"h{i+1}": 0.0 for i in range(4)}, "reported_nmae": nmae}
        for value in ("approve", "review", "plan"):
            answer = Observation(c.id, "answered", "supported", numbers=numbers, choices={"plan": value})
            assert grade(c, a, answer)["unauditable"] == (value == "plan")


def test_followup_gate_rejects_missing_and_invalid_decisions(corpus):
    import statistics
    from benchmarks.workflow.business_utility.followup import valid_decisions
    c = load_cases(corpus / "eval1.jsonl")[0]
    audits = json.loads((corpus / "private-audit.json").read_text())
    replay = audits[c.id]["causal_replay"]
    error = statistics.fmean(abs(x-y) for x,y in zip(replay, c.available_at_cutoff["replay_actuals"])) / audits[c.id]["scale"]
    nums = {**{f"replay_{i+1}": v for i,v in enumerate(replay)}, **{f"h{i+1}": 0.0 for i in range(4)}, "reported_nmae": error}
    good = Observation(c.id, "answered", "supported", numbers=nums, choices={"plan": "approve" if error <= 1 else "review"})
    assert valid_decisions([c], audits, [good])
    assert not valid_decisions([c], audits, [])
    assert not valid_decisions([c], audits, [good, good])
    bad = deepcopy(good)
    bad.choices["plan"] = "plan"
    assert not valid_decisions([c], audits, [bad])
    bad.choices["plan"] = "review" if error <= 1 else "approve"
    assert not valid_decisions([c], audits, [bad])


def test_followup_dispatch_log_does_not_collide_with_full_arm(tmp_path, monkeypatch):
    from benchmarks.workflow.business_utility import followup
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "providers.json").write_text(json.dumps({"llm": {"token_env": "TEST_KEY"}}))
    (prepared / "launch-plan.json").write_text(json.dumps({"arm_order": ["lean", "ordinary", "full"]}))
    (prepared / "corpus").mkdir()
    (prepared / "corpus/private-audit.json").write_text("{}")
    monkeypatch.setattr(followup, "credential", lambda *a: "test-only")
    monkeypatch.setattr(followup, "load_cases", lambda *a: [])
    monkeypatch.setattr(followup, "load_observations", lambda *a: [])
    monkeypatch.setattr(followup, "valid_decisions", lambda *a: True)
    calls = []
    class Process:
        def __init__(self, argv, **kwargs):
            calls.append(argv)
            if "--output-dir" in argv:
                destination = Path(argv[argv.index("--output-dir") + 1])
                destination.mkdir(parents=True)
                (destination / "summary.json").write_text("{}")
        def wait(self, **kwargs):
            return 0
    monkeypatch.setattr(followup.subprocess, "Popen", Process)
    output = tmp_path / "run"
    assert followup.run(prepared, prepared, output, "unused") == 0
    assert len(calls) == 4
    assert (output / "full.log").exists()
    assert (output / "full-dispatch.log").exists()
    assert json.loads((output / "status.json").read_text())["state"] == "complete"


def test_dispatch_stops_whole_run_on_unknown_spending(tmp_path, monkeypatch):
    from benchmarks.workflow.business_utility import dispatch
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "launch-plan.json").write_text(json.dumps({"status": "ready", "evaluation_order": ["eval1", "eval2", "eval3"], "arm_order": ["lean", "ordinary", "full"]}))
    calls = []
    class Process:
        def __init__(self, argv, **kwargs):
            calls.append(argv)
            dest = Path(argv[argv.index("--output-dir") + 1])
            dest.mkdir(parents=True)
            (dest / "summary.json").write_text("{}")
            (dest / "observations.jsonl").write_text(json.dumps({"metadata": {"error": "spending_usage_unmeasured"}})+'\n')
        def wait(self): return 2
        def poll(self): return 2
    monkeypatch.setattr(dispatch.subprocess, "Popen", Process)
    assert dispatch.run(prepared, tmp_path / "out") == 1
    assert len(calls) == 1
    assert json.loads((tmp_path / "out/status.json").read_text())["cause"] == "unknown_accounting"


def test_matched_report_without_primary_observations_keeps_effect_missing(corpus, tmp_path, monkeypatch):
    from benchmarks.workflow import matched
    from benchmarks.workflow.provenance import corpus_sha256
    runs = tmp_path / "runs"
    for evaluation in ("eval1", "eval2", "eval3"):
        for arm in ("lean", "ordinary", "full"):
            p = runs / evaluation / arm
            p.mkdir(parents=True)
            (p / "summary.json").write_text("{}")
    monkeypatch.setattr(matched, "load_summary", lambda p: {"corpus_sha256": corpus_sha256(load_cases(corpus / (p.parent.name + '.jsonl')))})
    monkeypatch.setattr(matched, "compare", lambda summaries: {"evidence_kind": "agent"})
    build_report(corpus, runs, tmp_path / "report")
    assert json.loads((tmp_path / 'report/eval2.json').read_text())["primary_effect"] is None
    family = json.loads((tmp_path / 'report/familywise.json').read_text())
    assert family["family_complete"] is False
    assert family["comparisons"] == {}

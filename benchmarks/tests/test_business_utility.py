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

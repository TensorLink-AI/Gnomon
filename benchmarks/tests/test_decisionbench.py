"""DecisionBench: generator determinism, grading exactness, the shared
submission contract, and each arm's loop against scripted clients.

No network anywhere: the chat client is a script, the MCP session is a
stub. The grading tests pin the newsvendor optimum against brute force
and the harm/success/abstention semantics against hand-computed costs.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.decisionbench import arms, grading, tasks
from benchmarks.decisionbench.mcp_arm import McpArm

# ---------------------------------------------------------------------------
# generator


def test_generation_is_deterministic():
    first = tasks.generate_tasks(12, seed=3)
    second = tasks.generate_tasks(12, seed=3)
    assert [task.task_id for task in first] == [task.task_id for task in second]
    assert [task.truth for task in first] == [task.truth for task in second]
    assert [task.csv_text() for task in first] == [
        task.csv_text() for task in second]


def test_family_mix_and_shapes():
    suite = tasks.generate_tasks(200, seed=7)
    families = {task.family for task in suite}
    assert families == {name for name, _ in tasks.FAMILY_WEIGHTS}
    for task in suite:
        assert len(task.truth) == tasks.HORIZON
        assert all(value > 0 for value in task.truth)
        assert task.escalation_fee > 0
        if task.family == "short_history":
            assert len(task.vintage) == tasks.SHORT_HISTORY
        if task.family == "revision":
            assert "published" in task.columns
            # The trap: rows published after the cutoff are in the file.
            assert any(row["published"] > task.cutoff.date().isoformat()
                       for row in task.rows)
        else:
            assert "published" not in task.columns


def test_truth_never_in_csv_except_revision_trap():
    for task in tasks.generate_tasks(60, seed=11):
        if task.family == "revision":
            continue
        text = task.csv_text()
        assert len(text.splitlines()) - 1 >= len(task.vintage)
        future_stamps = {tasks._timestamp(task.metadata["history"] + step)
                         for step in range(task.horizon)}
        assert not any(stamp in text for stamp in future_stamps)


def test_genuine_context_moves_truth():
    suite = tasks.generate_tasks(400, seed=5)
    genuine = [task for task in suite if task.family == "genuine_context"]
    assert genuine
    for task in genuine:
        assert task.context_note and task.context_factor
        level = sum(task.vintage[-14:]) / 14
        realized = sum(task.truth[3:]) / len(task.truth[3:])
        if task.context_factor > 1:
            assert realized > 1.25 * level
        else:
            assert realized < 0.75 * level


# ---------------------------------------------------------------------------
# grading


def test_optimal_capacity_matches_brute_force():
    rng = random.Random(1)
    for _ in range(25):
        truth = [round(rng.uniform(50, 300), 1) for _ in range(14)]
        c_under = rng.choice((10.0, 20.0))
        best, best_cost = grading.optimal_capacity(truth, 1.0, c_under)
        grid_cost = min(
            grading.decision_cost(candidate / 10, truth, 1.0, c_under)
            for candidate in range(0, 4001))
        assert best_cost <= grid_cost + 1e-6
        assert best_cost == grading.decision_cost(best, truth, 1.0, c_under)


def _clean_task() -> tasks.DecisionTask:
    return next(task for task in tasks.generate_tasks(80, seed=2)
                if task.family == "clean")


def test_grade_success_beats_naive():
    task = _clean_task()
    best, _ = grading.optimal_capacity(task.truth, task.c_over, task.c_under)
    grade = grading.grade(task, {"decision": {"capacity": best},
                                 "peak_forecast": None, "rationale": None})
    assert grade["answered"] and grade["success"]
    assert grade["regret"] == 0
    assert not grade["harm_case"]


def test_grade_escalation_costs_the_fee_and_abstention_is_family_scoped():
    suite = tasks.generate_tasks(200, seed=9)
    short = next(task for task in suite if task.family == "short_history")
    clean = next(task for task in suite if task.family == "clean")
    submission = {"decision": {"escalate": True},
                  "peak_forecast": None, "rationale": None}
    short_grade = grading.grade(short, submission)
    assert short_grade["cost"] == short.escalation_fee
    assert short_grade["appropriate_abstention"]
    # Process over outcome luck: refusing a nine-point commitment is a
    # success even when the fee exceeds what naive would have paid.
    assert short_grade["success"]
    clean_grade = grading.grade(clean, submission)
    assert not clean_grade["appropriate_abstention"]
    if clean.escalation_fee > grading.naive_capacity(clean)[1]:
        assert not clean_grade["success"]


def test_matching_naive_is_not_success_when_a_gap_exists():
    """The anti-gaming property: submitting the naive formula's own
    capacity scores regret_vs_naive ~1 and fails wherever a real
    naive-to-optimal gap exists — the success bar cannot be met by
    copying the bar."""
    task = _clean_task()
    capacity, cost_naive = grading.naive_capacity(task)
    _, cost_optimal = grading.optimal_capacity(task.truth, task.c_over,
                                               task.c_under)
    # Precondition of the property, true for this fixed task: the gap
    # is real, not floored away.
    assert cost_naive - cost_optimal > 0.01 * max(cost_optimal, 1.0)
    grade = grading.grade(task, {"decision": {"capacity": capacity},
                                 "peak_forecast": None, "rationale": None})
    assert grade["regret_vs_naive"] > grading.SUCCESS_GAP_SHARE
    assert not grade["success"]
    # Closing the gap fully is success.
    best, _ = grading.optimal_capacity(task.truth, task.c_over, task.c_under)
    assert grading.grade(task, {"decision": {"capacity": best},
                                "peak_forecast": None,
                                "rationale": None})["success"]


def test_grade_harm_case_and_unanswered():
    task = _clean_task()
    # Commit essentially zero capacity: every unit of demand is unmet at
    # c_under, which dwarfs both comparators.
    grade = grading.grade(task, {"decision": {"capacity": 0.0},
                                 "peak_forecast": None, "rationale": None})
    assert grade["harm_case"] and not grade["success"]
    unanswered = grading.grade(task, None)
    assert not unanswered["answered"]
    assert unanswered["cost"] is None and not unanswered["success"]


def test_grade_calibration_and_leak_heuristic():
    suite = tasks.generate_tasks(300, seed=4)
    revision = next(task for task in suite if task.family == "revision")
    peak = max(revision.truth)
    best, _ = grading.optimal_capacity(revision.truth, revision.c_over,
                                       revision.c_under)
    leaky = grading.grade(revision, {
        "decision": {"capacity": best},
        "peak_forecast": {"q10": peak * 0.995, "q50": peak,
                          "q90": peak * 1.005},
        "rationale": None})
    assert leaky["temporal_leakage_heuristic"]
    assert leaky["peak_in_interval"] is True
    honest = grading.grade(revision, {
        "decision": {"capacity": max(revision.vintage)},
        "peak_forecast": {"q10": peak * 2.0, "q50": peak * 2.2,
                          "q90": peak * 2.4},
        "rationale": None})
    assert honest["peak_in_interval"] is False


def test_followed_context_is_symmetric():
    suite = tasks.generate_tasks(400, seed=6)
    task = next(task for task in suite
                if task.family in ("misleading_context", "genuine_context")
                and task.context_factor and task.context_factor > 1)
    naive, _ = grading.naive_capacity(task)
    claimed = task.context_factor * max(task.vintage)
    follower = grading.grade(task, {"decision": {"capacity": claimed},
                                    "peak_forecast": None, "rationale": None})
    ignorer = grading.grade(task, {"decision": {"capacity": naive},
                                   "peak_forecast": None, "rationale": None})
    assert follower["followed_context"] is True
    assert ignorer["followed_context"] is False


def test_task_statement_is_shared_across_arms():
    """Only the data reference and the answer-delivery block may differ
    between arms; the task itself is verbatim-identical."""
    task = _clean_task()
    direct = task.prompt("DATA_A", arms.DIRECT_ANSWER_FORMAT)
    tooled = task.prompt("DATA_B", arms.TOOL_ANSWER_FORMAT)
    statement = direct.replace("DATA_A", "{data}").replace(
        arms.DIRECT_ANSWER_FORMAT, "{answer}")
    assert statement == tooled.replace("DATA_B", "{data}").replace(
        arms.TOOL_ANSWER_FORMAT, "{answer}")
    assert str(task.escalation_fee) in direct or (
        f"{task.escalation_fee:g}" in direct)


def test_validate_submission_contract():
    good, problem = grading.validate_submission(
        {"decision": {"capacity": 210.5}})
    assert problem is None and good["decision"] == {"capacity": 210.5}
    escalate, problem = grading.validate_submission(
        {"decision": {"escalate": True}})
    assert problem is None and escalate["decision"] == {"escalate": True}
    for bad in (
        None, {}, {"decision": {}},
        {"decision": {"capacity": "lots"}},
        {"decision": {"capacity": float("nan")}},
        {"decision": {"capacity": -5}},
        {"decision": {"capacity": 5, "escalate": True}},
    ):
        submission, problem = grading.validate_submission(bad)
        assert submission is None and problem


# ---------------------------------------------------------------------------
# arms, scripted clients


class ScriptedClient:
    """Replays canned assistant turns; counts token fields the loop reads."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0

    def chat(self, messages, *, n=1, tools=None, tool_choice=None,
             temperature=None, max_tokens=None):
        turn = self.turns.pop(0)
        self.total_completion_tokens += 10
        return SimpleNamespace(choices=[SimpleNamespace(
            message=turn, finish_reason="stop")])

    def completions(self, messages, *, n=1):
        turn = self.turns.pop(0)
        self.total_completion_tokens += 10
        return [turn]


def _tool_call(name, arguments):
    return SimpleNamespace(
        message=SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                id="call-1", type="function",
                function=SimpleNamespace(name=name,
                                         arguments=json.dumps(arguments)))],
        )).message


def _prose(text):
    return SimpleNamespace(content=text, tool_calls=None)


def test_direct_arm_parses_and_repairs():
    task = _clean_task()
    answer = json.dumps({"decision": {"capacity": 250},
                         "peak_forecast": {"q10": 200, "q50": 240,
                                           "q90": 280},
                         "rationale": "test"})
    outcome = arms.run_direct(task, ScriptedClient(["prose only", answer]))
    assert outcome["submission"]["decision"] == {"capacity": 250.0}

    hopeless = arms.run_direct(task, ScriptedClient(["nope", "still nope"]))
    assert hopeless["submission"] is None
    assert "unparseable" in hopeless["abstained"]


def test_sandbox_arm_runs_code_and_submits(tmp_path):
    task = _clean_task()
    client = ScriptedClient([
        _tool_call("run_python", {"code": "print(6 * 7)"}),
        _tool_call("submit_decision", {"decision": {"capacity": 300}}),
    ])
    arm = arms.SandboxArm(task, client, work_dir=str(tmp_path))
    outcome = arm.drive()
    assert outcome["submission"]["decision"] == {"capacity": 300.0}
    assert outcome["tool_calls"] == 1
    # The data file is in the jail for the code to read.
    assert (arm.jail / "demand.csv").exists()


def test_sandbox_rejects_bad_submission_then_accepts(tmp_path):
    task = _clean_task()
    client = ScriptedClient([
        _tool_call("submit_decision", {"decision": {"capacity": -1}}),
        _tool_call("submit_decision", {"decision": {"escalate": True}}),
    ])
    outcome = arms.SandboxArm(task, client, work_dir=str(tmp_path)).drive()
    assert outcome["submission"]["decision"] == {"escalate": True}


def test_second_submission_is_rejected_and_first_stands(tmp_path):
    task = _clean_task()
    client = ScriptedClient([
        _tool_call("submit_decision", {"decision": {"capacity": 100}}),
    ])
    arm = arms.SandboxArm(task, client, work_dir=str(tmp_path))
    outcome = arm.drive()
    assert outcome["submission"]["decision"] == {"capacity": 100.0}
    # A late duplicate through the dispatch path is refused, not
    # silently mislabeled as accepted.
    reply = json.loads(arm._route("submit_decision",
                                  {"decision": {"capacity": 999}}))
    assert reply["accepted"] is False and "already" in reply["message"]
    assert arm.submission["decision"] == {"capacity": 100.0}


def test_loop_last_call_collects_late_submission(tmp_path):
    task = _clean_task()
    client = ScriptedClient(
        [_prose("thinking..."), _prose("more thinking..."),
         _tool_call("submit_decision", {"decision": {"capacity": 210}})])
    outcome = arms.SandboxArm(task, client, work_dir=str(tmp_path)).drive()
    assert outcome["submission"]["decision"] == {"capacity": 210.0}
    assert outcome["last_call"].startswith("no submission")


def test_loop_abstains_when_nothing_is_submitted(tmp_path):
    task = _clean_task()
    client = ScriptedClient(
        [_prose("a"), _prose("b"), _prose("c")])
    outcome = arms.SandboxArm(task, client, work_dir=str(tmp_path)).drive()
    assert outcome["submission"] is None
    assert outcome["abstained"]


class StubSession:
    """A do-nothing MCP session: one echo tool, records calls."""

    def __init__(self, cwd):
        self.cwd = cwd
        self.calls = []
        self.closed = False

    def initialize(self):
        return {}

    def list_tools(self):
        return [{"name": "gnomon_inspect",
                 "description": "stub",
                 "inputSchema": {"type": "object", "properties": {}}}]

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return {"isError": False, "structuredContent": {"ok": True},
                "content": [{"type": "text", "text": '{"ok": true}'}]}

    def close(self):
        self.closed = True


def test_mcp_arm_uses_surface_and_submits(tmp_path):
    task = _clean_task()
    client = ScriptedClient([
        _tool_call("gnomon_inspect", {"input_path": "demand.csv"}),
        _tool_call("submit_decision",
                   {"decision": {"capacity": 280},
                    "peak_forecast": {"q10": 1, "q50": 2, "q90": 3}}),
    ])
    arm = McpArm(task, client, work_dir=str(tmp_path),
                 session_factory=StubSession)
    outcome = arm.drive()
    assert outcome["submission"]["decision"] == {"capacity": 280.0}
    assert arm.session.calls and arm.session.closed
    tool_names = [spec["function"]["name"] for spec in arm.tool_specs()]
    assert tool_names == ["gnomon_inspect", "submit_decision"]


def test_mcp_arm_against_real_server(tmp_path):
    """The treatment arm against the real server code (CiK's in-process
    session): the published tool surface converts to chat specs, a real
    gnomon_inspect on the jail CSV succeeds, and the submission
    resolves. This is the test the stub cannot stand in for."""
    from benchmarks.cik.mcp_agent import InProcessMcpSession

    task = _clean_task()
    client = ScriptedClient([])
    arm = McpArm(task, client, work_dir=str(tmp_path),
                 session_factory=InProcessMcpSession)
    client.turns = [
        _tool_call("gnomon_inspect", {"input": str(arm.csv_path)}),
        _tool_call("submit_decision",
                   {"decision": {"capacity": 400},
                    "peak_forecast": {"q10": 300, "q50": 380,
                                      "q90": 460}}),
    ]
    names = [spec["function"]["name"] for spec in arm.tool_specs()]
    assert "gnomon_forecast" in names and "gnomon_inspect" in names
    assert names[-1] == "submit_decision"
    outcome = arm.drive()
    assert outcome["submission"]["decision"] == {"capacity": 400.0}
    assert arm.session.calls[0][0] == "gnomon_inspect"
    inspect_entry = next(entry for entry in outcome["trace"]
                         if entry.get("tool") == "gnomon_inspect")
    assert not inspect_entry.get("is_error")


def test_mcp_arm_jails_paths(tmp_path):
    task = _clean_task()
    client = ScriptedClient([
        _tool_call("gnomon_inspect", {"input_path": "/etc/passwd"}),
        _tool_call("submit_decision", {"decision": {"escalate": True}}),
    ])
    arm = McpArm(task, client, work_dir=str(tmp_path),
                 session_factory=StubSession)
    outcome = arm.drive()
    assert outcome["submission"]["decision"] == {"escalate": True}
    assert arm.session.calls == []  # the violating call never reached it
    assert any(entry.get("jail_violations") for entry in outcome["trace"])
    # A screened-out call is a round-trip, not a spent tool call.
    assert outcome["tool_calls"] == 0

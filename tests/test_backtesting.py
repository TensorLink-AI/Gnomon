from datetime import datetime, timedelta, timezone
import json
import sqlite3
import subprocess
import sys

import pytest

from gnomon import EvaluationBudget, ForecastResult, GnomonSession, InferenceEngine, TemporalLedger, evaluate_reference
from gnomon.contracts import GnomonError
from gnomon.cli import main
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.ids import FixedClock
from gnomon.temporal_store import TemporalObservation, TemporalStore
from test_ephemeris import service  # noqa: F401


def at(day):
    return datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=day - 1)


def result(request, step=0):
    return ForecastResult(tuple(request.history[-1] + step * (i + 1) for i in range(request.horizon)),
                          timestamps=request.future_timestamps, series_id=request.series_id, unit=request.unit)


def source(tmp_path, n=30):
    path = tmp_path / "data.csv"
    path.write_text("timestamp,value\n" + "".join(f"{at(i).isoformat()},{i}\n" for i in range(1, n + 1)))
    return path


def configured(tmp_path, *, ledger=False):
    session = GnomonSession.from_config()
    if ledger:
        session.close()
        engine = InferenceEngine(ledger=TemporalLedger(tmp_path / "ledger.db", clock=FixedClock(at(35))))
        engine.register("last_value", lambda r: result(r), revision="base-v1", deterministic=True)
        session = GnomonSession(engine)
    session.engine.register("trend", lambda r: result(r, 1), revision="trend-v1", deterministic=True)
    return session, session.data.inspect(str(source(tmp_path)), unit="requests")["data_ref"]


def test_matched_cohorts_explicit_baseline_and_immutable_study(tmp_path):
    session, ref = configured(tmp_path, ledger=True)
    report = session.evaluate(ref, candidates=["trend"], baseline="last_value", horizon=2)
    assert report["status"] == "complete" and report["usage"]["provider_calls"] == 8
    assert report["scores"]["trend"] == {"n": 8, "mae": 0, "rmse": 0, "bias": 0}
    assert report["scores"]["last_value"]["mae"] == 1.5
    assert report["ranking"] == ["trend", "last_value"]
    assert report["ranking_policy"]["ties"] == []
    assert report["action_authorized"] is False and report["training_cutoff_attested"] is False
    assert report["calibration"] == "not_established"
    saved = session.ledger.study(report["study_id"])
    for fold in saved["folds"]:
        a, b = (session.ledger.execution(fold["runs"][p]["execution_id"]) for p in ("last_value", "trend"))
        assert a["request"] == b["request"]
        assert a["cache_hit"] is False and b["cache_hit"] is False
        assert fold["request"]["timestamps"][-1] == fold["origin"]
        assert min(r["valid_time"] for r in fold["actuals"]) > fold["origin"]
    assert session.ledger.actuals_as_of("series", unit="requests") == []  # no implicit actual imports
    report["scores"]["trend"]["mae"] = 999
    assert session.ledger.study(report["study_id"])["scores"]["trend"]["mae"] == 0
    with pytest.raises(ForecastAdapterError, match="cutoff"):
        session.ledger.study(report["study_id"], recorded_as_of=at(34).isoformat())
    with sqlite3.connect(session.ledger.path) as conn, pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM studies")


@pytest.mark.parametrize("calls,matched", [(0, 0), (1, 0), (3, 1), (8, 4)])
def test_every_invocation_including_baseline_counts_toward_budget(tmp_path, calls, matched):
    session, ref = configured(tmp_path)
    report = session.evaluate(ref, candidates=["trend"], baseline="last_value", horizon=2, budget={"max_calls": calls})
    assert report["usage"]["provider_calls"] == calls
    assert report["usage"]["matched_folds"] == matched
    assert report["scores"]["trend"]["n"] == 2 * matched
    assert sum(r["attempted"] for r in report["diagnostics"].values()) == calls
    assert report["usage"]["internal_model_calls"] == "unknown"
    assert report["usage"]["stop_reason"] == ("call_budget" if calls < 8 else None)
    if matched == 0:
        assert report["ranking_policy"]["ties"] == []
        assert "No matched folds" in report["ranking_policy"]["guidance"]


@pytest.mark.parametrize("candidates", [["seasonal_naive", "trend"], ["trend", "seasonal_naive"]])
def test_tied_mae_is_disclosed_without_turning_display_order_into_a_winner(tmp_path, candidates):
    session, ref = configured(tmp_path)
    # Baseline and season=1 seasonal naive coincide; trend is strictly better.
    report = session.call("gnomon_evaluate", {"data_ref": ref, "candidates": candidates,
                          "baseline": "last_value", "horizon": 2})
    policy = report["ranking_policy"]
    assert report["ranking"][0] == "trend"
    assert policy["ties"] == [{"mae": 1.5, "providers": ["last_value", "seasonal_naive"]}]
    assert policy["tie_order"] == "provider_input_order"
    assert "does not establish a winner" in policy["guidance"]
    full = session.call("gnomon_evaluate", {"study_id": report["study_id"]}, compact=False)
    assert full["ranking_policy"] == policy


def test_close_scores_are_not_silently_treated_as_exact_ties(tmp_path):
    session, ref = configured(tmp_path)
    session.engine.register("near_last", lambda request: result(request, 1e-10))
    report = session.evaluate(ref, candidates=["near_last"], baseline="last_value", horizon=2)
    assert report["ranking"] == ["near_last", "last_value"]
    assert report["ranking_policy"]["ties"] == []


def test_errors_are_charged_and_never_dropped_from_completion_denominators(tmp_path):
    session, ref = configured(tmp_path)
    def broken(request):
        raise RuntimeError("secret-token-sensitive-url")
    session.engine.register("broken", broken)
    report = session.evaluate(ref, candidates=["trend", "broken"], baseline="last_value", horizon=2)
    assert report["usage"]["provider_calls"] == 12
    assert report["usage"]["matched_folds"] == 0
    assert report["diagnostics"]["broken"] == {"attempted": 4, "succeeded": 0, "failed": 4}
    assert report["ranking"] == [] and report["scores"]["trend"]["mae"] is None
    assert "secret-token" not in json.dumps(report)


def test_cancellation_keeps_completed_matched_folds(tmp_path):
    session, ref = configured(tmp_path)
    count = []
    session.engine.register("count", lambda r: (count.append(1), result(r))[1])
    report = session.evaluate(ref, candidates=["count"], baseline="last_value", horizon=2,
                              cancelled=lambda: bool(count))
    assert report["status"] == "partial" and report["usage"]["stop_reason"] == "cancelled"
    assert report["usage"]["provider_calls"] == 2 and report["usage"]["matched_folds"] == 1


def test_wall_limit_stops_dispatch_not_a_running_trusted_callable(tmp_path):
    session, ref = configured(tmp_path)
    now = [0.0]
    def slow(request):
        now[0] += 5
        return result(request)
    session.engine.register("slow", slow)
    report = evaluate_reference(session.engine, session.data, ref, baseline="slow", candidates=["trend"],
                                 horizon=2, budget=EvaluationBudget(max_seconds=1), timer=lambda: now[0])
    assert report["usage"]["provider_calls"] == 1 and report["usage"]["stop_reason"] == "time_budget"
    assert report["usage"]["wall_limit_overrun"] and report["usage"]["dispatch_boundary_limits"]


def test_each_fold_gets_a_fresh_factory_instance(tmp_path):
    session, ref = configured(tmp_path)
    created, closed = [], []
    class Model:
        def __init__(self):
            self.used = False
            created.append(self)
        def forecast(self, request):
            assert not self.used
            self.used = True
            return result(request, 1)
        def close(self):
            closed.append(self)
    session.engine.register_factory("fit", Model)
    report = session.evaluate(ref, candidates=["fit"], baseline="last_value", horizon=2)
    assert report["status"] == "complete" and len(created) == len(closed) == 4
    assert len({id(m) for m in created}) == 4


def vintage_source(tmp_path, late_known):
    path = tmp_path / "store.db"
    store = TemporalStore(path)
    for day in range(1, 31):
        store.ingest_rows("data", [TemporalObservation("a", "value", at(day), at(day), day)],
                          source_fingerprint=str(day), clock=FixedClock(at(day)))
    store.ingest_rows("data", [TemporalObservation("a", "value", at(5), at(29 if late_known else 5), 999)],
                      source_fingerprint="revision", clock=FixedClock(at(29)))
    return path, store


@pytest.mark.parametrize("late_known,replay,expected", [(True, "source_available", 5),
                                                       (False, "source_available", 999), (False, "recorded", 5)])
def test_fold_histories_read_the_correct_vintage_not_the_final_prefix(tmp_path, late_known, replay, expected):
    path, store = vintage_source(tmp_path, late_known)
    session, _ = configured(tmp_path)
    ref = session.data.inspect("store:data", store_path=str(path), as_of=at(30).isoformat(),
                                recorded_as_of=at(30).isoformat())["data_ref"]
    # Mutating the source after inspection must not alter any fold.
    store.ingest_rows("data", [TemporalObservation("a", "value", at(6), at(6), 1234)],
                      source_fingerprint="later", clock=FixedClock(at(31)))
    report = session.evaluate(ref, candidates=["trend"], baseline="last_value", horizon=2, replay=replay)
    assert report["status"] == "complete"
    assert all(f["request"]["history"][4] == expected for f in report["folds"])
    assert all(f["request"]["history"][5] == 6 for f in report["folds"])
    if replay == "recorded":
        assert all(f["request"]["recorded_time_cutoff"] == f["origin"] for f in report["folds"])


def test_narrowing_a_snapshot_cannot_widen_parent_cutoffs(tmp_path):
    _, store = vintage_source(tmp_path, True)
    parent = store.snapshot("data", as_of=at(20), recorded_as_of=at(18))
    child = parent.narrow(as_of=at(30), recorded_as_of=at(30))
    assert child.as_of == at(20) and child.recorded_as_of == at(18)
    assert child.snapshot_id == parent.snapshot_id


def test_missing_recorded_history_abstains_without_prefix_fallback(tmp_path):
    path = tmp_path / "late.db"
    store = TemporalStore(path)
    store.ingest_rows("data", [TemporalObservation("a", "value", at(i), at(i), i) for i in range(1, 31)],
                      source_fingerprint="all-late", clock=FixedClock(at(31)))
    session, _ = configured(tmp_path)
    ref = session.data.inspect("store:data", store_path=str(path))["data_ref"]
    report = session.evaluate(ref, candidates=["trend"], baseline="last_value", horizon=2, replay="recorded")
    assert report["status"] == "unscored" and report["usage"]["provider_calls"] == 0
    assert all(f["status"] == "unavailable_history_or_capability" for f in report["folds"])


def test_repaired_final_history_is_not_a_historical_vintage(tmp_path):
    path = source(tmp_path)
    path.write_text(path.read_text().replace(f"{at(15).isoformat()},15\n", ""))
    session, _ = configured(tmp_path)
    # configured recreates source; delete the gap again before inspection.
    path.write_text(path.read_text().replace(f"{at(15).isoformat()},15\n", ""))
    inspection = session.data.inspect(str(path), repair="aggressive", frequency="D")
    assert inspection["repairs"]
    for action in inspection["repairs"]:
        action["code"] = "timestamps_reordered"  # cannot mutate internal frozen provenance
    with pytest.raises(ForecastAdapterError, match="unrepaired"):
        session.evaluate(inspection["data_ref"], candidates=["trend"], baseline="last_value", horizon=2)


@pytest.mark.parametrize("budget", [{"max_calls": 100}, {"max_folds": 9}, {"max_seconds": None}])
def test_agent_cannot_raise_operator_limits(tmp_path, budget):
    session, ref = configured(tmp_path)
    with pytest.raises(GnomonError, match="startup limits"):
        session.call("gnomon_evaluate", {"data_ref": ref, "candidates": ["trend"], "baseline": "last_value",
                                         "horizon": 2, "budget": budget})


def test_compact_tool_report_and_exact_retrieval(tmp_path):
    session, ref = configured(tmp_path)
    report = session.call("gnomon_evaluate", {"data_ref": ref, "candidates": ["trend"], "baseline": "last_value", "horizon": 2})
    assert "request" not in report["folds"][0]
    full = session.call("gnomon_evaluate", {"study_id": report["study_id"]}, compact=False)
    assert full["folds"][0]["request"]["history"]
    full["folds"][0]["request"]["history"][0] = 999
    assert session.call("gnomon_evaluate", {"study_id": report["study_id"]}, compact=False)["folds"][0]["request"]["history"][0] == 1


def test_cli_and_mcp_share_study_cohorts_and_real_provider_calls(tmp_path, capsys):
    path = source(tmp_path)
    config = tmp_path / "providers.toml"
    config.write_text('schema_version=1\nledger_path="ledger.db"\n')
    options = {"candidates": ["historical_mean"], "baseline": "last_value", "horizon": 2, "folds": 2}
    with GnomonSession.from_config(config) as session:
        inspection = session.data.inspect(str(path))
        expected = session.call("gnomon_evaluate", {"data_ref": inspection["data_ref"], **options})
    assert main(["evaluate", "--providers-config", str(config), "--arguments",
                 json.dumps({"data": {"input": str(path)}, **options})]) == 0
    cli = json.loads(capsys.readouterr().out)
    assert cli["folds"][0]["request"]["history"]  # no dead CLI session reference
    assert cli["cohort_id"] == expected["cohort_id"] and cli["scores"] == expected["scores"]
    # One live stdio process: inspect, then pass its returned ref to evaluate.
    process = subprocess.Popen([sys.executable, "-m", "gnomon.cli", "mcp", "serve", "--providers-config", str(config)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        def call(number, tool, arguments):
            process.stdin.write(json.dumps({"id": number, "method": "tools/call", "params": {"name": tool, "arguments": arguments}}) + "\n")
            process.stdin.flush()
            reply = json.loads(process.stdout.readline())["result"]
            assert reply["isError"] is False, reply
            return reply["structuredContent"]
        ref = call(1, "gnomon_inspect", {"input": str(path)})["data_ref"]
        mcp = call(2, "gnomon_evaluate", {"data_ref": ref, **options})
        stored = call(3, "gnomon_ledger", {"operation": "study", "study_id": mcp["study_id"]})
        if "result_ref" in stored:
            # Added readiness diagnostics can move this study over the bounded
            # response limit. Retrieve the full evidence in the same session.
            pages, offset = [], 0
            while offset is not None:
                page = call(4 + len(pages), "gnomon_read", {"result_ref": stored["result_ref"],
                            "pointer": "/result", "offset": offset})
                pages.append(page["text"])
                offset = page["next_offset"]
            full = json.loads("".join(pages))
        else:
            full = stored["result"]
        assert mcp["cohort_id"] == expected["cohort_id"] and mcp["scores"] == cli["scores"]
        assert full["folds"][0]["request"]["history"]
    finally:
        process.stdin.close()
        process.wait(timeout=10)
    assert process.returncode == 0


def test_short_history_and_budget_rejections_do_not_run_providers(tmp_path):
    session, _ = configured(tmp_path)
    ref = session.data.inspect(str(source(tmp_path, n=3)))["data_ref"]
    report = session.evaluate(ref, candidates=["trend"], baseline="last_value", horizon=2)
    assert report["status"] == "unscored" and report["usage"]["planned_folds"] == 0
    assert report["usage"]["provider_calls"] == 0 and report["usage"]["requested_folds"] == 4
    for changes in ({"folds": 9}, {"candidates": ["last_value"]}, {"candidates": ["missing"]}):
        with pytest.raises(ForecastAdapterError):
            session.evaluate(ref, **{"candidates": ["trend"], "baseline": "last_value", "horizon": 2, **changes})


@pytest.mark.parametrize("budget", [{"max_calls": True}, {"max_calls": -1}, {"max_folds": 1.5},
                                   {"max_seconds": float("nan")}, {"surprise": 3}])
def test_budget_contract_is_strict(budget):
    with pytest.raises(ForecastAdapterError):
        EvaluationBudget.from_dict(budget)


def test_ephemeris_evaluation_has_no_hidden_remote_retries_or_model_call_claims(tmp_path, service):
    from gnomon import EphemerisProvider
    url, state = service
    session, ref = configured(tmp_path)
    session.engine.register("remote", EphemerisProvider(url), lifecycle="pretrained")
    report = session.evaluate(ref, candidates=["remote"], baseline="last_value", horizon=2, budget={"max_calls": 5})
    assert report["usage"]["provider_calls"] == 5
    assert len(state["requests"]) == 2  # 3 baseline calls plus 2 remote POSTs
    assert all(r[0] == "POST" for r in state["requests"])
    assert report["providers"]["remote"]["revision"] is None
    assert report["usage"]["internal_model_calls"] == "unknown"

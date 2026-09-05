from copy import deepcopy
import json
import sqlite3

import pytest

from gnomon import GnomonSession, InferenceEngine
from gnomon.cli import main
from gnomon.contracts import GnomonError
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.ids import FixedClock
from gnomon.mcp_server import _handle
from gnomon.temporal_store import TemporalObservation, TemporalStore
from test_backtesting import at, configured, result, source


def task(session, ref, study):
    return {"data_ref": ref, "study_id": study["study_id"], "candidates": ["trend"], "baseline": "last_value",
            "horizon": 2, "source_as_of": at(30).isoformat(), "recorded_as_of": at(35).isoformat()}


def studied(tmp_path):
    session, ref = configured(tmp_path, ledger=True)
    study = session.evaluate(ref, candidates=["trend"], baseline="last_value", horizon=2)
    return session, ref, study


def test_route_uses_versioned_replayable_cohort_and_persists_new_score_without_model_calls(tmp_path):
    session, ref, study = studied(tmp_path)
    before = session.ledger.study(study["study_id"])
    answer = session.call("gnomon_route", task(session, ref, study))
    assert answer["recommendation"] == "trend" and answer["basis"] == "cutoff_bound_matched_study"
    assert answer["matched_folds"] == 4 and answer["scores"]["trend"]["mae"] == 0
    assert answer["provider_calls"] == 0 and not answer["action_authorized"]
    saved = session.ledger.study(answer["rescore_study_id"])
    assert saved["derived_from"] == study["study_id"] and saved["rescore_only"] is True
    assert saved["usage"]["provider_calls"] == 0 and saved["original_usage"]["provider_calls"] == 8
    assert session.ledger.study(study["study_id"]) == before
    with sqlite3.connect(session.ledger.path) as conn:
        assert conn.execute("SELECT count(*) FROM executions").fetchone()[0] == 8


def test_study_not_yet_recorded_cannot_influence_a_historical_recommendation(tmp_path):
    session, ref, study = studied(tmp_path)
    answer = session.call("gnomon_route", {**task(session, ref, study), "recorded_as_of": at(34).isoformat()})
    assert answer["recommendation"] == "last_value"
    assert answer["reason"] == "study_unavailable_at_recorded_cutoff"


def test_actual_revisions_rescore_without_overwriting_previous_comparisons(tmp_path):
    session, _ = configured(tmp_path, ledger=True)
    path = tmp_path / "temporal.db"
    store = TemporalStore(path)
    for day in range(1, 31):
        store.ingest_rows("sales", [TemporalObservation("a", "value", at(day), at(day), day)],
                          source_fingerprint=str(day), clock=FixedClock(at(day)))
    old_ref = session.data.inspect("store:sales", store_path=str(path), as_of=at(30).isoformat(), recorded_as_of=at(30).isoformat())["data_ref"]
    original = session.evaluate(old_ref, candidates=["trend"], baseline="last_value", horizon=2)
    # Latest two outcomes revised; prior training vintages stay reconstructible.
    for day in (29, 30):
        store.ingest_rows("sales", [TemporalObservation("a", "value", at(day), at(30), 500)],
                          source_fingerprint="late-revision", clock=FixedClock(at(34)))
    old_replay = session.data.inspect("store:sales", store_path=str(path), as_of=at(30).isoformat(), recorded_as_of=at(33).isoformat())["data_ref"]
    before = session.route(**{**task(session, old_replay, original), "recorded_as_of": at(35).isoformat()})
    assert before["scores"]["trend"]["mae"] == 0
    assert before["effective_recorded_as_of"] == at(33).isoformat()
    new_ref = session.data.inspect("store:sales", store_path=str(path), as_of=at(30).isoformat(), recorded_as_of=at(35).isoformat())["data_ref"]
    after = session.route(**task(session, new_ref, original))
    assert after["scores"]["trend"]["mae"] > 100
    assert after["effective_recorded_as_of"] == at(35).isoformat()
    assert after["cohort_id"] != before["cohort_id"]
    assert session.ledger.study(original["study_id"])["scores"]["trend"]["mae"] == 0


def test_inspection_must_match_query_cutoffs_not_include_future_observations(tmp_path):
    session, ref, study = studied(tmp_path)
    with pytest.raises(GnomonError, match="inspect"):
        session.call("gnomon_route", {**task(session, ref, study), "source_as_of": at(25).isoformat()})


@pytest.mark.parametrize("change,reason", [({"unit": "kg"}, "task_identity_mismatch"),
                                         ({"horizon": 3}, "task_identity_mismatch"),
                                         ({"revision": "new-version"}, "provider_identity_changed"),
                                         ({"revision": None}, "provider_revision_unknown")])
def test_incompatible_task_or_model_identity_cannot_enter_prior(tmp_path, change, reason):
    session, ref, study = studied(tmp_path)
    if "unit" in change:
        ref = session.data.inspect(str(source(tmp_path)), unit=change["unit"])["data_ref"]
    if "revision" in change:
        session.engine._providers["trend"].revision = change["revision"]
    arguments = task(session, ref, study)
    if "horizon" in change:
        arguments["horizon"] = change["horizon"]
    answer = session.call("gnomon_route", arguments)
    assert answer["reason"] == reason and answer["recommendation"] == "last_value"


def test_forged_study_cannot_override_immutable_execution_outputs(tmp_path):
    session, ref, study = studied(tmp_path)
    forged = deepcopy(study)
    forged["study_id"] = "forged-study"
    for fold in forged["folds"]:
        fold["runs"]["trend"]["point"] = [-1, -1]
    session.ledger.record_study(forged)
    answer = session.call("gnomon_route", task(session, ref, forged))
    assert answer["recommendation"] == "last_value"
    assert answer["reason"] == "insufficient_replayable_matched_folds"
    assert {f["reason"] for f in answer["excluded_folds"]} == {"study_execution_mismatch"}


def test_pretrained_weights_need_a_declared_training_cutoff_before_fold_origins(tmp_path):
    session, _ = configured(tmp_path, ledger=True)
    engine = InferenceEngine(ledger=session.ledger)
    engine.register("last_value", lambda r: result(r), revision="base-v1")
    engine.register("trend", lambda r: result(r, 1), revision="pretrained-v1", lifecycle="pretrained")
    remote = GnomonSession(engine)
    ref = remote.data.inspect(str(source(tmp_path)), unit="requests")["data_ref"]
    study = remote.evaluate(ref, candidates=["trend"], baseline="last_value", horizon=2)
    answer = remote.route(**task(remote, ref, study))
    assert answer["reason"] == "insufficient_replayable_matched_folds"
    assert {r["reason"] for r in answer["excluded_folds"]} == {"pretrained_training_cutoff_unattested"}


def test_partial_folds_and_policy_changes_are_disclosed(tmp_path):
    session, ref = configured(tmp_path, ledger=True)
    study = session.evaluate(ref, candidates=["trend"], baseline="last_value", horizon=2, budget={"max_calls": 4})
    answer = session.route(**task(session, ref, study))
    assert answer["matched_folds"] == 2 and answer["recommendation"] == "last_value"
    with pytest.raises(ForecastAdapterError, match="three"):
        session.route(**{**task(session, ref, study), "min_folds": 1})


def test_mcp_and_cli_use_identical_cutoff_policy_and_results(tmp_path, capsys):
    path = source(tmp_path)
    config = tmp_path / "providers.toml"
    config.write_text('schema_version=1\nledger_path="ledger.db"\n')
    with GnomonSession.from_config(config) as session:
        ref = session.data.inspect(str(path))["data_ref"]
        study = session.evaluate(ref, candidates=["historical_mean"], baseline="last_value", horizon=2)
        args = {**task(session, ref, study), "candidates": ["historical_mean"], "recorded_as_of": "2099-01-01T00:00:00Z"}
        python = session.route(**args)
        mcp = _handle({"method": "tools/call", "params": {"name": "gnomon_route", "arguments": args}}, session=session)
        assert mcp["isError"] is False
        assert mcp["structuredContent"]["scores"] == python["scores"]
    args.pop("data_ref")
    assert main(["route", "--providers-config", str(config), "--arguments", json.dumps({"data": {"input": str(path)}, **args})]) == 0
    cli = json.loads(capsys.readouterr().out)
    assert cli["scores"] == python["scores"] and cli["recommendation"] == python["recommendation"]


def test_naive_time_cannot_silently_become_a_recorded_time_replay(tmp_path):
    session, _, study = studied(tmp_path)
    path = source(tmp_path)
    path.write_text(path.read_text().replace("+00:00", ""))
    ref = session.data.inspect(str(path), unit="requests")["data_ref"]
    with pytest.raises(ForecastAdapterError, match="timezone-aware"):
        session.route(**task(session, ref, study))

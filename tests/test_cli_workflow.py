"""User journeys across real CLI processes and the shared agent boundary."""

from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
import json
import subprocess
import sys

import pytest

from gnomon import GnomonSession
from gnomon.cli import main
from gnomon.contracts import GnomonError


def invoke(*args):
    output = StringIO()
    with redirect_stdout(output):
        code = main(list(map(str, args)))
    return code, json.loads(output.getvalue())


def process(*args):
    result = subprocess.run([sys.executable, "-m", "gnomon", *map(str, args)], text=True, capture_output=True)
    assert result.stderr == ""
    return result.returncode, json.loads(result.stdout)


def source(tmp_path, *, count=31, gap=None):
    path = tmp_path / "data.csv"
    path.write_text("timestamp,value\n" + "".join(
        f"2026-01-{i:02d},{i}\n" for i in range(1, count + 1) if i != gap))
    return path


def evaluation(path):
    return ("evaluate", "--input", path, "--candidates", "historical_mean", "--baseline", "last_value", "--horizon", "2")


def test_season_can_be_set_for_file_forecasts_and_is_not_silently_ignored(tmp_path):
    path = source(tmp_path)
    code, result = invoke("infer", "--input", path, "--provider", "seasonal_naive", "--horizon", "3", "--season", "7")
    assert code == 0 and result["result"]["point"] == [25, 26, 27]
    code, result = invoke("infer", "--provider", "seasonal_naive", "--season", "7", "--request", '{"history":[1,2],"horizon":1}')
    assert code == 2 and "only with --input" in result["error"]["message"]
    code, result = invoke("infer", "--input", path, "--provider", "last_value", "--horizon", "1", "--quantiles", "0.1", "0.9")
    assert code == 2 and "quantiles" in result["error"]["message"]


def test_unscored_and_partial_evaluations_have_distinct_non_success_exits(tmp_path):
    path = source(tmp_path, count=5)
    code, result = invoke(*evaluation(path))
    assert code == 2 and result["status"] == "unscored"
    assert result["usage"]["provider_calls"] == 0
    assert result["issues"][0]["required_rows_for_one_fold"] == 10
    assert result["issues"][0]["required_rows_for_requested_folds"] == 16
    source(tmp_path)
    code, result = invoke(*evaluation(path), "--max-calls", "4")
    assert code == 3 and result["status"] == "partial"
    assert result["usage"]["matched_folds"] == 2
    assert result["issues"][0]["code"] == "CALL_BUDGET"
    code, result = invoke(*evaluation(path), "--max-calls", "0")
    assert code == 2 and result["status"] == "unscored"


def test_inspect_explains_timezone_before_a_user_reaches_routing(tmp_path):
    path = source(tmp_path)
    code, result = invoke("inspect", path)
    assert code == 0 and result["readiness"]["evaluate"]["ready"] is True
    assert result["readiness"]["route"]["ready"] is False
    assert "--timezone UTC" in result["readiness"]["route"]["issues"][0]["description"]
    code, result = invoke("inspect", path, "--for", "route")
    assert code == 2 and result["error"]["code"] == "INPUT_NOT_READY"
    code, result = invoke("inspect", path, "--for", "route", "--timezone", "Australia/Brisbane")
    assert code == 0 and result["series"][0]["start"].endswith("+10:00")


@pytest.mark.parametrize("stamp", ["2026-03-08T02:30:00", "2026-11-01T01:30:00"])
def test_explicit_timezone_never_guesses_dst_gaps_or_folds(stamp, tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(f"timestamp,value\n{stamp},1\n")
    code, result = invoke("inspect", path, "--timezone", "America/New_York", "--frequency", "h")
    assert code == 2 and "ambiguous or nonexistent" in result["error"]["message"]


def test_repair_readiness_and_observed_window_provide_a_working_recovery(tmp_path):
    path = source(tmp_path, gap=5)
    code, result = invoke("inspect", path, "--repair", "aggressive", "--frequency", "D")
    assert code == 0 and result["readiness"]["evaluate"]["ready"] is False
    issue = result["readiness"]["evaluate"]["issues"][0]
    assert "gap_filled" in issue["codes"] and issue["example_input_options"]["window"] == "latest_contiguous"
    code, result = invoke(*evaluation(path), "--repair", "aggressive", "--frequency", "D")
    assert code == 2 and result["error"]["code"] == "INPUT_NOT_READY"
    code, result = invoke(*evaluation(path), "--window", "latest_contiguous", "--frequency", "D")
    assert code == 0 and result["usage"]["matched_folds"] == 4
    for fold in result["folds"]:
        assert min(fold["request"]["history"]) >= 6
        assert all(row["value"] != 5 for row in fold["actuals"])
    code, result = invoke("inspect", path, "--window", "latest_contiguous", "--frequency", "D")
    action = next(r for r in result["repairs"] if r["code"] == "window_selected")
    assert action["metrics"]["excluded_rows"] == 4 and action["metrics"]["selected_rows"] == 26


def test_independent_format_repairs_can_be_evaluated_but_column_inference_cannot(tmp_path):
    path = source(tmp_path)
    path.write_text(path.read_text().replace(",", ",$", 31).replace("timestamp,$value", "timestamp,value"))
    code, result = invoke(*evaluation(path), "--repair", "safe")
    assert code == 0 and result["status"] == "complete"
    path.write_text('timestamp,value\n' + ''.join(
        f'2026-01-{i:02d},"1,234"\n' for i in range(1, 20)) + '2026-01-20,"1,234.5"\n')
    code, result = invoke("inspect", path, "--repair", "safe")
    assert code == 0 and result["readiness"]["evaluate"]["ready"] is False
    assert "numeric_format_inferred" in result["readiness"]["evaluate"]["issues"][0]["codes"]


def test_complete_workflow_survives_separate_processes_and_deleted_source(tmp_path):
    path = source(tmp_path)
    snapshot, study, ledger = (tmp_path / name for name in ("data.gnomon", "study.json", "evidence.db"))
    code, original = process("inspect", "--input", path, "--timezone", "UTC", "--for", "route", "--save-snapshot", snapshot)
    assert code == 0 and original["reuse"]["input"] == str(snapshot)
    path.unlink()
    code, restored = process("inspect", "--input", snapshot, "--for", "route")
    assert code == 0 and restored["data_ref"] == original["data_ref"]
    code, report = process(*evaluation(snapshot), "--ledger-path", ledger, "--save-result", study)
    assert code == 0 and report["recorded"] is True and report["usage"]["provider_calls"] == 8
    code, result = process("route", "--input", snapshot, "--study", "@" + str(study), "--ledger-path", ledger,
                           "--source-as-of", "2026-01-31T00:00:00Z", "--recorded-as-of", "2099-01-01T00:00:00Z")
    assert code == 0 and result["matched_folds"] == 4 and result["provider_calls"] == 0
    assert result["study_id"] == report["study_id"]


def test_snapshot_preserves_revisions_and_frozen_cutoffs(tmp_path):
    from gnomon.ids import FixedClock
    from gnomon.temporal_store import TemporalObservation, TemporalStore
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = TemporalStore(tmp_path / "source.db")
    rows = [TemporalObservation("a", "value", origin + timedelta(days=i), origin + timedelta(days=i), i)
            for i in range(20)]
    store.ingest_rows("test", rows, source_fingerprint="original", clock=FixedClock(origin + timedelta(days=20)))
    store.ingest_rows("test", [TemporalObservation("a", "value", origin, origin + timedelta(days=21), 100)],
                      source_fingerprint="revision", clock=FixedClock(origin + timedelta(days=22)))
    saved = tmp_path / "vintages.gnomon"
    with GnomonSession.from_config() as session:
        data = session.data.inspect("store:test", store_path=str(store.path))
        session.data.save(data["data_ref"], saved)
    with GnomonSession.from_config() as session:
        restored = session.data.inspect(str(saved))
        assert restored["data_ref"] == data["data_ref"]
        frozen = session.data._get(restored["data_ref"])
        assert frozen.loaded.snapshot.observation_count == 21
        historical = frozen.loaded.snapshot.narrow(as_of=origin + timedelta(days=20))
        assert historical.series("a", "value")[0].value == 0
        assert frozen.loaded.snapshot.series("a", "value")[0].value == 100


def test_snapshots_preserve_repairs_and_reject_modification_and_schema_overrides(tmp_path):
    path = source(tmp_path, gap=5)
    saved = tmp_path / "repaired.gnomon"
    with GnomonSession.from_config() as session:
        data = session.data.inspect(str(path), repair="aggressive", frequency="D")
        session.data.save(data["data_ref"], saved)
        restored = session.data.inspect(str(saved))
        assert restored["repairs"] == data["repairs"] and not restored["readiness"]["evaluate"]["ready"]
        with pytest.raises(ValueError, match="already fixes"):
            session.data.inspect(str(saved), timezone="UTC")
        payload = json.loads(saved.read_text())
        payload["repairs"] = []
        saved.write_text(json.dumps(payload))
        with pytest.raises(GnomonError, match="modified"):
            session.data.inspect(str(saved))


def test_task_flags_are_not_ignored_with_json_and_ledger_paths_cannot_conflict(tmp_path):
    code, result = invoke("evaluate", "--arguments", "{}", "--horizon", "2")
    assert code == 2 and "require --input" in result["error"]["message"]
    config = tmp_path / "providers.toml"
    config.write_text('ledger_path = "first.db"\n')
    code, result = invoke(*evaluation(source(tmp_path)), "--providers-config", config, "--ledger-path", tmp_path / "second.db")
    assert code == 2 and "conflicts" in result["error"]["message"]
    assert not (tmp_path / "first.db").exists() and not (tmp_path / "second.db").exists()


@pytest.mark.parametrize("response_limit", [2048, 8192])
def test_agent_gets_readiness_and_unscored_failure_even_with_compact_output(tmp_path, response_limit):
    from gnomon.mcp_server import _handle
    with GnomonSession.from_config() as session:
        inspection = _handle({"method": "tools/call", "params": {"name": "gnomon_inspect", "arguments": {
            "input": str(source(tmp_path, count=5)), "timezone": "UTC", "purpose": "route"}}}, session=session)
        assert inspection["isError"] is False
        data = inspection["structuredContent"]
        assert data["readiness"]["route"]["ready"] is True
        from gnomon.result_refs import ResultLimits, ResultReferences
        session.results.close()
        session.results = ResultReferences(ResultLimits(max_response_bytes=response_limit))
        result = _handle({"method": "tools/call", "params": {"name": "gnomon_evaluate", "arguments": {
            "data_ref": data["data_ref"], "candidates": ["historical_mean"], "baseline": "last_value", "horizon": 2}}}, session=session)
        assert result["isError"] is True
        assert result["structuredContent"]["status"] == "unscored"
        if response_limit == 2048:
            assert "result_ref" in result["structuredContent"]


def test_snapshot_forecast_keeps_named_timezone_across_dst(tmp_path):
    path = tmp_path / "dst.csv"
    path.write_text("timestamp,value\n2026-03-05,1\n2026-03-06,2\n2026-03-07,3\n")
    saved = tmp_path / "dst.gnomon"
    with GnomonSession.from_config() as session:
        data = session.data.inspect(str(path), timezone="America/New_York")
        expected = session.data.request(data["data_ref"], horizon=3)
        session.data.save(data["data_ref"], saved)
    with GnomonSession.from_config() as session:
        restored = session.data.inspect(str(saved))
        actual = session.data.request(restored["data_ref"], horizon=3)
        assert actual == expected
        assert actual.future_timestamps == ("2026-03-08T00:00:00-05:00", "2026-03-09T00:00:00-04:00", "2026-03-10T00:00:00-04:00")


def test_snapshot_import_obeys_retained_row_and_byte_limits(tmp_path, monkeypatch):
    from gnomon import snapshot_files
    saved = tmp_path / "data.gnomon"
    with GnomonSession.from_config() as session:
        data = session.data.inspect(str(source(tmp_path)))
        session.data.save(data["data_ref"], saved)
    with GnomonSession(max_data_rows=10) as session, pytest.raises(GnomonError, match="retained observation limit"):
        session.data.inspect(str(saved))
    monkeypatch.setattr(snapshot_files, "MAX_SNAPSHOT_BYTES", 10)
    with GnomonSession() as session, pytest.raises(GnomonError, match="portable-file limit"):
        session.data.inspect(str(saved))


def test_date_order_inferred_from_later_rows_stays_excluded_from_evaluation(tmp_path):
    path = tmp_path / "dates.csv"
    path.write_text("timestamp,value\n" + "".join(f"{day:02d}/01/2026,{day}\n" for day in range(1, 21)))
    code, result = invoke("inspect", path, "--repair", "safe")
    assert code == 0
    assert "date_format_inferred" in result["readiness"]["evaluate"]["issues"][0]["codes"]


def test_malformed_study_file_gives_usage_guidance(tmp_path):
    study = tmp_path / "study.json"
    study.write_text('{"evidence":"rolling_origin_backtest","study_id":"id"}')
    code, result = invoke("route", "--input", source(tmp_path), "--study", "@" + str(study),
                          "--ledger-path", tmp_path / "evidence.db")
    assert code == 2 and result["error"]["code"] == "INVALID_ARGUMENTS"
    assert "--study @file must contain an evaluate result" in result["error"]["message"]


@pytest.mark.parametrize("repair", ["off", "safe", "aggressive"])
def test_declared_timezone_takes_precedence_over_mixed_timezone_repair(tmp_path, repair):
    path = tmp_path / "mixed.csv"
    path.write_text("timestamp,value\n2026-01-01,1\n2026-01-02T00:00:00+10:00,2\n2026-01-03,3\n")
    code, result = invoke("inspect", path, "--timezone", "Australia/Brisbane", "--repair", repair,
                          "--as-of", "2026-01-02T00:00:00", "--frequency", "D", "--for", "route")
    assert code == 0
    assert result["series"][0]["count"] == 2
    assert result["series"][0]["start"] == "2026-01-01T00:00:00+10:00"
    assert [r["code"] for r in result["repairs"]] == ["timezone_declared"]


def test_utc_declaration_does_not_require_a_system_timezone_database(tmp_path, monkeypatch):
    from gnomon import datasets
    def unavailable(name):
        raise datasets.ZoneInfoNotFoundError(name)
    monkeypatch.setattr(datasets, "ZoneInfo", unavailable)
    code, result = invoke("inspect", source(tmp_path), "--timezone", "UTC", "--for", "route")
    assert code == 0 and result["readiness"]["route"]["ready"] is True

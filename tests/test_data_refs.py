from datetime import datetime, timezone
import json
import subprocess
import sys

import pytest

from gnomon import GnomonSession
from gnomon.cli import main
from gnomon.contracts import GnomonError
from gnomon.data_refs import DataReferences
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.ids import FixedClock
from gnomon.temporal_store import TemporalObservation, TemporalStore


def at(day):
    return datetime(2025, 1, day, tzinfo=timezone.utc)


def csv(tmp_path, values=(1, 2, 100), name="data.csv"):
    path = tmp_path / name
    path.write_text("timestamp,value\n" + "".join(f"{at(i+1).isoformat()},{v}\n" for i, v in enumerate(values)))
    return path


def test_reference_does_not_reopen_mutated_source(tmp_path):
    path = csv(tmp_path)
    with GnomonSession.from_config() as session:
        first = session.call("gnomon_inspect", {"input": str(path), "unit": "requests"})
        csv(tmp_path, (8, 9, 10))
        new = session.call("gnomon_inspect", {"input": str(path), "unit": "requests"})
        assert new["data_ref"] != first["data_ref"]
        run = session.call("gnomon_forecast", {"provider": "last_value", "data_ref": first["data_ref"], "horizon": 2})
        assert run["result"]["point"] == (100, 100)
        assert run["result"]["timestamps"] == (at(4).isoformat(), at(5).isoformat())
        assert run["result"]["unit"] == "requests"
        assert run["action_authorized"] is False
        assert first["snapshot"]["known_time_assumed"] is True


@pytest.mark.parametrize("statistic,expected", [("mean", 1.5), ("median", 1.5), ("sum", 3),
                                               ("latest", 2), ("minimum", 1), ("maximum", 2)])
def test_statistics_respect_exact_inclusive_window(tmp_path, statistic, expected):
    refs = DataReferences()
    ref = refs.inspect(str(csv(tmp_path)))["data_ref"]
    result = refs.describe(ref, statistic=statistic, start=at(1).isoformat(), end=at(2).isoformat())
    assert result["value"] == expected and result["count"] == 2
    assert result["window"]["closed"] == "both"
    assert result["evidence"] == "observed_statistic" and not result["action_authorized"]


def test_panel_requires_exact_selection(tmp_path):
    path = tmp_path / "panel.csv"
    path.write_text("timestamp,value,shop\n2025-01-01,1,a\n2025-01-02,2,a\n2025-01-01,9,b\n2025-01-02,10,b\n")
    refs = DataReferences()
    ref = refs.inspect(str(path), series_column="shop", frequency="D")["data_ref"]
    with pytest.raises(GnomonError, match="exact series_id"):
        refs.request(ref, horizon=1)
    assert refs.request(ref, horizon=1, series_id="b").history == (9, 10)


def test_store_references_preserve_source_and_recording_cutoffs(tmp_path):
    path = tmp_path / "store.db"
    store = TemporalStore(path)
    rows = [TemporalObservation("a", "value", at(i), at(i), float(i)) for i in (1, 2, 3)]
    store.ingest_rows("data", rows, source_fingerprint="original", clock=FixedClock(at(3)))
    refs = DataReferences()
    original = refs.inspect("store:data", store_path=str(path), as_of=at(3).isoformat(), recorded_as_of=at(5).isoformat())
    store.ingest_rows("data", [TemporalObservation("a", "value", at(3), at(3), 99)],
                      source_fingerprint="revision", clock=FixedClock(at(10)))
    replay = refs.inspect("store:data", store_path=str(path), as_of=at(3).isoformat(), recorded_as_of=at(5).isoformat())
    latest = refs.inspect("store:data", store_path=str(path), as_of=at(3).isoformat())
    assert replay["data_ref"] == original["data_ref"]
    assert refs.request(original["data_ref"], horizon=1).history == (1, 2, 3)
    request = refs.request(latest["data_ref"], horizon=1)
    assert request.history == (1, 2, 99) and request.known_time_cutoff == at(3).isoformat()
    assert refs.request(replay["data_ref"], horizon=1).recorded_time_cutoff == at(5).isoformat()


def test_lagged_input_cannot_label_pre_as_of_steps_as_future(tmp_path):
    refs = DataReferences()
    ref = refs.inspect(str(csv(tmp_path)), as_of=at(5).isoformat())["data_ref"]
    assert refs.describe(ref, statistic="latest")["value"] == 100
    with pytest.raises(ForecastAdapterError, match="follow cutoff"):
        refs.request(ref, horizon=3)


def test_plain_files_cannot_claim_recorded_time_replay(tmp_path):
    with pytest.raises(GnomonError, match="recording times"):
        DataReferences().inspect(str(csv(tmp_path)), recorded_as_of=at(3).isoformat())


def test_lru_limits_and_unknown_references_are_explicit(tmp_path):
    refs = DataReferences(max_refs=2, max_rows=6)
    a = refs.inspect(str(csv(tmp_path, name="a.csv")))["data_ref"]
    b = refs.inspect(str(csv(tmp_path, (2, 3, 4), "b.csv")))["data_ref"]
    refs.describe(a, statistic="mean")
    refs.inspect(str(csv(tmp_path, (3, 4, 5), "c.csv")))
    with pytest.raises(GnomonError, match="expired"):
        refs.request(b, horizon=1)
    assert refs.request(a, horizon=1).history[-1] == 100
    with pytest.raises(GnomonError, match="limit"):
        DataReferences(max_rows=2).inspect(str(csv(tmp_path)))


def test_source_bound_precedes_repair(tmp_path):
    # Interpolation must not fill a pre-cutoff gap using a later row.
    path = tmp_path / "gap.csv"
    path.write_text("timestamp,value\n2025-01-01,1\n2025-01-02,2\n2025-01-04,400\n")
    refs = DataReferences()
    inspected = refs.inspect(str(path), as_of="2025-01-03T00:00:00", repair="aggressive", frequency="D")
    summary = refs.describe(inspected["data_ref"], statistic="mean")
    assert summary["count"] == 2 and summary["value"] == 1.5


@pytest.mark.parametrize("changes", [{"window": "last_week"}, {"unit": "kg"}, {"statistic": "trend"},
                                     {"start": "2025-01-01"}, {"start": "2025-01-01T00:00:00"}])
def test_describe_never_ignores_unknown_or_incompatible_scope(tmp_path, changes):
    with GnomonSession.from_config() as session:
        ref = session.data.inspect(str(csv(tmp_path)))["data_ref"]
        with pytest.raises(GnomonError):
            session.call("gnomon_describe", {"data_ref": ref, "statistic": "mean", **changes})


def test_ref_request_cannot_override_frozen_history(tmp_path):
    with GnomonSession.from_config() as session:
        ref = session.data.inspect(str(csv(tmp_path)))["data_ref"]
        with pytest.raises(GnomonError, match="unknown"):
            session.call("gnomon_forecast", {"provider": "last_value", "data_ref": ref, "horizon": 1,
                                             "request": {"history": [999], "horizon": 1}})


def test_cli_file_inference_uses_identical_frozen_request(tmp_path, capsys):
    path = csv(tmp_path)
    with GnomonSession.from_config() as session:
        ref = session.data.inspect(str(path))["data_ref"]
        expected = session.call("gnomon_forecast", {"provider": "last_value", "data_ref": ref, "horizon": 2})
    assert main(["infer", "--input", str(path), "--provider", "last_value", "--horizon", "2"]) == 0
    cli = json.loads(capsys.readouterr().out)
    assert cli["fingerprint"] == expected["fingerprint"]
    assert cli["input"]["snapshot"]["known_time_assumed"]


def test_file_inference_keeps_evaluation_and_context_optional(tmp_path):
    path = csv(tmp_path)
    source = f"""
import sys
from gnomon import GnomonSession
with GnomonSession.from_config() as s:
    ref = s.data.inspect({str(path)!r})['data_ref']
    assert s.call('gnomon_forecast', {{'provider':'last_value', 'data_ref':ref, 'horizon':1}})['status'] == 'ok'
for name in ('pipeline', 'runtime', 'evaluation', 'context', 'toolspec'):
    assert 'gnomon.' + name not in sys.modules, name
"""
    result = subprocess.run([sys.executable, "-c", source], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("horizon", [True, 1.5, 0])
def test_invalid_horizon_rejected_before_constructing_grid(tmp_path, horizon):
    refs = DataReferences()
    ref = refs.inspect(str(csv(tmp_path)))["data_ref"]
    with pytest.raises(ForecastAdapterError):
        refs.request(ref, horizon=horizon)

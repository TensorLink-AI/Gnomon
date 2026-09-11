"""`gnomon forecast FILE --horizon N` on a two-column CSV needs no column or provider flags."""

import json
from pathlib import Path

import pytest

from gnomon.cli import main

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "messy_requests.csv"


def _run(capsys, argv):
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


def _csv(tmp_path, header, rows, name="data.csv"):
    path = tmp_path / name
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return str(path)


def test_two_column_csv_infers_columns_and_defaults_to_last_value(capsys):
    code, result = _run(capsys, ["forecast", str(EXAMPLE), "--horizon", "7"])
    assert code == 0 and result["status"] == "ok"
    assert result["provider"] == "last_value" and len(result["result"]["point"]) == 7
    assumptions = {item["field"]: item for item in result["assumptions"]}
    assert assumptions["provider"]["value"] == "last_value"
    assert assumptions["target_column"] == {"field": "target_column", "value": "requests",
                                            "basis": "inferred_from_two_column_csv_header"}
    assert "time_column" not in assumptions, "the header already names the default time column"
    # The disclosed columns are the ones that were actually used.
    code, explicit = _run(capsys, ["forecast", "--input", str(EXAMPLE), "--horizon", "7",
                                   "--provider", "last_value", "--target", "requests"])
    assert explicit["result"]["point"] == result["result"]["point"]
    assert "assumptions" not in explicit


def test_column_order_does_not_matter(tmp_path, capsys):
    path = _csv(tmp_path, "hits,when", ["5,2026-01-01", "7,2026-01-02", "6,2026-01-03"])
    code, result = _run(capsys, ["forecast", path, "--horizon", "2"])
    assert code == 0 and result["result"]["point"] == [6, 6]
    assert {(a["field"], a["value"]) for a in result["assumptions"]} >= {("time_column", "when"), ("target_column", "hits")}


def test_explicit_column_flags_disable_inference(tmp_path, capsys):
    code, payload = _run(capsys, ["forecast", str(EXAMPLE), "--horizon", "2", "--target", "nope"])
    assert code == 2 and payload["error"]["code"] == "MISSING_COLUMNS"


@pytest.mark.parametrize("header, rows", [
    ("timestamp,a,b", ["2026-01-01,1,2", "2026-01-02,3,4", "2026-01-03,5,6"]),      # three columns
    ("a,b,c", ["1,2,3", "4,5,6", "7,8,9"]),                                           # three numeric columns
    ("first,second", ["2026-01-01,2026-01-02", "2026-01-02,2026-01-03"]),           # two timestamp-like columns
    ("a,b", ["1,2", "3,4", "5,6"]),                                                   # two numeric columns
    ("timestamp,label", ["2026-01-01,x", "2026-01-02,y"]),                           # non-numeric second column
])
def test_ambiguous_or_wider_files_still_reject_with_the_column_choice_error(tmp_path, header, rows, capsys):
    code, payload = _run(capsys, ["forecast", _csv(tmp_path, header, rows), "--horizon", "2"])
    assert code == 2
    assert payload["error"]["code"] == "MISSING_COLUMNS"
    assert "choices_required" in payload["error"]["details"]


def test_provider_must_be_chosen_when_other_providers_are_configured(tmp_path, capsys, monkeypatch):
    (tmp_path / "mymodel.py").write_text(
        "from gnomon import ForecastResult\n"
        "def predict(request):\n    return ForecastResult((0.0,) * request.horizon)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    config = tmp_path / "providers.toml"
    config.write_text('schema_version = 1\n[providers.mine]\nkind = "callable"\nentrypoint = "mymodel:predict"\n')
    code, payload = _run(capsys, ["forecast", str(EXAMPLE), "--horizon", "2", "--providers-config", str(config)])
    assert code == 2 and payload["error"]["code"] == "INVALID_ARGUMENTS"
    assert "--provider is required" in payload["error"]["message"]


def test_naming_a_provider_keeps_column_mapping_explicit(capsys):
    # Inference belongs to the bare form only; with --provider the existing
    # column-correction guidance applies unchanged.
    code, payload = _run(capsys, ["forecast", str(EXAMPLE), "--horizon", "2", "--provider", "last_value"])
    assert code == 2 and payload["error"]["code"] == "MISSING_COLUMNS"
    code, result = _run(capsys, ["forecast", str(EXAMPLE), "--horizon", "2", "--provider", "last_value",
                                 "--target", "requests"])
    assert code == 0 and "assumptions" not in result


def test_request_form_still_requires_a_provider(capsys):
    code, payload = _run(capsys, ["forecast", "--request", '{"history":[1,2,3],"horizon":1}'])
    assert code == 2 and "--provider is required" in payload["error"]["message"]


def test_positional_and_option_input_cannot_both_be_given(capsys):
    code, payload = _run(capsys, ["forecast", str(EXAMPLE), "--input", str(EXAMPLE), "--horizon", "2"])
    assert code == 2 and "not both" in payload["error"]["message"]
    code, payload = _run(capsys, ["forecast", "--horizon", "2"])
    assert code == 2 and "--input" in payload["error"]["message"]

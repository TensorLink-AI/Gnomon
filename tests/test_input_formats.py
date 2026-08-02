"""Input-format flexibility: delimiter detection, encoding fallback, gzip,
JSON/JSONL, and the optional Excel extra — all through the same disclosed
repair machinery, with the strict path unchanged."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gnomon.contracts import GnomonError
from gnomon.ids import FixedClock
from gnomon.runtime import capabilities, forecast

CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))


def daily(count: int) -> list[tuple[str, float]]:
    start = datetime(2026, 1, 1)
    return [
        ((start + timedelta(days=index)).date().isoformat(), 100.0 + index)
        for index in range(count)
    ]


def run(path: Path, tmp_path: Path, **kwargs):
    return forecast(
        str(path), time_column="timestamp", target_column="value", horizon=3,
        output=str(tmp_path / "out"), clock=CLOCK, **kwargs,
    )


def repair_codes(artifact) -> set[str]:
    for item in artifact.evidence:
        if item.kind == "data_repair":
            return {action["code"] for action in item.payload["actions"]}
    return set()


def test_semicolon_csv_with_comma_decimals(tmp_path: Path) -> None:
    source = tmp_path / "european.csv"
    lines = ["timestamp;value"] + [f"{stamp};{value:.1f}".replace(".", ",") for stamp, value in daily(30)]
    source.write_text("\n".join(lines), encoding="utf-8")
    artifact, _ = run(source, tmp_path)
    assert "delimiter_detected" in repair_codes(artifact)
    assert artifact.results[0].support == "supported"
    # Strict mode keeps the old behaviour: one giant column, missing mapping.
    with pytest.raises(GnomonError) as caught:
        run(source, tmp_path, repair="off")
    assert caught.value.code == "MISSING_COLUMNS"


def test_pipe_delimited_csv(tmp_path: Path) -> None:
    source = tmp_path / "piped.csv"
    lines = ["timestamp|value"] + [f"{stamp}|{value}" for stamp, value in daily(20)]
    source.write_text("\n".join(lines), encoding="utf-8")
    artifact, _ = run(source, tmp_path)
    assert "delimiter_detected" in repair_codes(artifact)


def test_tsv_is_supported_even_strict(tmp_path: Path) -> None:
    source = tmp_path / "data.tsv"
    lines = ["timestamp\tvalue"] + [f"{stamp}\t{value}" for stamp, value in daily(20)]
    source.write_text("\n".join(lines), encoding="utf-8")
    artifact, _ = run(source, tmp_path, repair="off")
    assert repair_codes(artifact) == set()


def test_cp1252_fallback_is_assumptive(tmp_path: Path) -> None:
    source = tmp_path / "latin.csv"
    lines = ["timestamp,value,notes"] + [
        f"{stamp},{value},caf\xe9" for stamp, value in daily(20)
    ]
    source.write_bytes("\n".join(lines).encode("cp1252"))
    with pytest.raises(GnomonError) as caught:
        run(source, tmp_path, repair="off")
    assert caught.value.code == "INVALID_ENCODING"
    artifact, _ = run(source, tmp_path)
    assert "encoding_assumed" in repair_codes(artifact)
    assert any("encoding_assumed" in warning for warning in artifact.results[0].warnings)


def test_gzipped_csv(tmp_path: Path) -> None:
    source = tmp_path / "data.csv.gz"
    lines = ["timestamp,value"] + [f"{stamp},{value}" for stamp, value in daily(20)]
    source.write_bytes(gzip.compress("\n".join(lines).encode("utf-8")))
    artifact, _ = run(source, tmp_path, repair="off")
    assert artifact.results[0].support in ("supported", "weakly_supported", "degraded")


def test_json_array(tmp_path: Path) -> None:
    source = tmp_path / "data.json"
    source.write_text(json.dumps([
        {"timestamp": stamp, "value": value} for stamp, value in daily(20)
    ]), encoding="utf-8")
    artifact, _ = run(source, tmp_path, repair="off")
    assert artifact.results[0].series == "__default__"


def test_json_must_be_array_of_objects(tmp_path: Path) -> None:
    source = tmp_path / "bad.json"
    source.write_text(json.dumps({"rows": []}), encoding="utf-8")
    with pytest.raises(GnomonError) as caught:
        run(source, tmp_path)
    assert caught.value.code == "UNSUPPORTED_INPUT"


def test_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "data.jsonl"
    source.write_text("\n".join(
        json.dumps({"timestamp": stamp, "value": value}) for stamp, value in daily(20)
    ), encoding="utf-8")
    artifact, _ = run(source, tmp_path, repair="off")
    assert len(artifact.results) == 1


def test_unsupported_suffix_lists_formats(tmp_path: Path) -> None:
    source = tmp_path / "data.xml"
    source.write_text("<rows/>", encoding="utf-8")
    with pytest.raises(GnomonError) as caught:
        run(source, tmp_path)
    assert caught.value.code == "UNSUPPORTED_INPUT"
    assert caught.value.to_dict()["error"]["repair_options"]


def test_excel_requires_extra_or_works(tmp_path: Path) -> None:
    source = tmp_path / "data.xlsx"
    try:
        import openpyxl
    except ImportError:
        source.write_bytes(b"not really an xlsx")
        with pytest.raises(GnomonError) as caught:
            run(source, tmp_path)
        assert caught.value.code == "MISSING_OPTIONAL_DEPENDENCY"
        assert "excel" in caught.value.details["install"]
        return
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["timestamp", "value"])
    start = datetime(2026, 1, 1)
    for index in range(20):
        sheet.append([start + timedelta(days=index), 100.0 + index])
    workbook.save(source)
    artifact, _ = run(source, tmp_path)
    assert artifact.results[0].series == "__default__"


def test_capabilities_report_inputs() -> None:
    inputs = capabilities()["inputs"]
    assert inputs["csv"] and inputs["tsv"] and inputs["json"] and inputs["jsonl"] and inputs["gzip"]
    assert isinstance(inputs["excel"], bool)

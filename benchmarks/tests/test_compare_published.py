"""Tests for the published-results comparison tool."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.compare_published import (
    collect_rows,
    dot_get,
    rank_rows,
    render_section,
)


def test_dot_get_nested_and_lists():
    obj = {"a": {"b": [10, {"c": 7}]}}
    assert dot_get(obj, "a.b.1.c") == 7
    assert dot_get(obj, "a.b.0") == 10
    assert dot_get(obj, "a.missing") is None
    assert dot_get(obj, "a.b.9") is None


def test_collect_rows_reads_summary(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"rcrps": {"mean": 0.42}}))
    section = {
        "published": [{"system": "paper", "value": 0.5, "source": "url"}],
        "ours": [
            {"system": "us", "summary": str(summary), "key": "rcrps.mean"},
            {"system": "gone", "summary": str(tmp_path / "no.json"),
             "key": "x"},
        ],
    }
    rows = collect_rows(section)
    assert rows[1]["value"] == 0.42
    assert rows[2]["value"] is None and "missing" in rows[2]["source"]


def test_rank_rows_direction():
    rows = [{"system": "a", "value": 0.5, "origin": "published", "source": ""},
            {"system": "b", "value": 0.3, "origin": "ours", "source": ""},
            {"system": "c", "value": None, "origin": "ours", "source": ""}]
    lower = rank_rows(list(rows), lower_is_better=True)
    assert [r["system"] for r in lower] == ["b", "a", "c"]
    higher = rank_rows(list(rows), lower_is_better=False)
    assert [r["system"] for r in higher] == ["a", "b", "c"]


def test_render_marks_our_rows():
    section = {"metric": "m", "lower_is_better": True,
               "published": [{"system": "paper", "value": 1.0}],
               "ours": []}
    text = render_section("bench", section)
    assert "lower is better" in text
    assert "paper" in text

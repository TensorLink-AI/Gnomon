"""The rejection-triage tool: deterministic buckets, no judgement calls.

The buckets decide where a fix belongs (parser vs proposer prompt), so a
span must land in exactly the bucket its content proves — and old
artifacts without recorded spans must be surfaced as untriageable, never
silently counted into a substantive bucket.
"""

import json
import pprint
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cik.triage_future_context import (
    bucket_rejection,
    iter_extra_info_files,
    main,
    triage,
)


def test_value_span_buckets_are_decided_by_content():
    # states no value at all: no parser could ever admit it -> prompt-side
    assert bucket_rejection(
        "span_states_the_value", "output will be reduced significantly"
    ) == "no_stated_value"
    # relative to an unstated base: refusal is correct by design
    assert bucket_rejection(
        "span_states_the_value", "production will be 50% of capacity"
    ) == "relative_value_refused"
    # today's parser reads it: the rejection is already fixed
    assert bucket_rejection(
        "span_states_the_value", "the plant is offline for maintenance"
    ) == "parses_now"
    # a digit is present but unparsed: the parser is missing a phrasing
    assert bucket_rejection(
        "span_states_the_value", "the line churns out 40 units come rain"
    ) == "parser_gap_candidate"
    # no span in the record: untriageable, never counted substantively
    assert bucket_rejection("span_states_the_value", None) == "span_not_recorded"


def test_bound_span_buckets_are_decided_by_content():
    assert bucket_rejection(
        "span_states_the_bound", "the series is bounded"
    ) == "no_stated_bound"
    assert bucket_rejection(
        "span_states_the_bound", "values remain between 10% and 20%"
    ) == "relative_value_refused"
    assert bucket_rejection(
        "span_states_the_bound", "values stay between 0 and 340"
    ) == "parses_now"
    assert bucket_rejection(
        "span_states_the_bound", "the ceiling sits at 96 on weekdays"
    ) == "parser_gap_candidate"


def _write_extra_info(root: Path, task: str, seed: int, payload: dict) -> None:
    run_dir = root / "runs" / task / str(seed)
    run_dir.mkdir(parents=True)
    (run_dir / "extra_info").write_text(pprint.pformat(payload),
                                        encoding="utf-8")


def test_triage_aggregates_a_run_tree(tmp_path):
    _write_extra_info(tmp_path, "TaskA", 1, {
        "future_context": {
            "admitted": [{"event_id": "e1"}],
            "defensive": [],
            "rejected": [
                {"event_id": "e2", "event_class": "override",
                 "code": "span_states_the_value",
                 "reason": "…",
                 "source_span": "output will be reduced significantly"},
                {"event_id": "e3", "event_class": "override",
                 "code": "span_states_the_value", "reason": "…"},
            ],
        },
    })
    _write_extra_info(tmp_path, "TaskA", 2, {
        "future_context": {
            "admitted": [],
            "defensive": [{"event_id": "e9"}],
            "rejected": [
                {"event_id": "e4", "event_class": "constraint",
                 "code": "window_touches_horizon", "reason": "…"},
            ],
        },
    })
    # a run with no lane evidence at all is counted as read, nothing more
    _write_extra_info(tmp_path, "TaskB", 1, {"support": "supported"})

    files = iter_extra_info_files([str(tmp_path)])
    report = triage(files)
    assert report["files_read"] == 3
    assert report["runs_with_lane_evidence"] == 2
    assert report["events_admitted"] == 1
    assert report["events_defensive"] == 1
    assert report["rejections_by_code"] == {
        "span_states_the_value": 2,
        "window_touches_horizon": 1,
    }
    value_buckets = report["span_buckets"]["span_states_the_value"]
    assert value_buckets["no_stated_value"]["count"] == 1
    assert value_buckets["span_not_recorded"]["count"] == 1


def test_main_writes_json_and_prints_the_reading_guide(tmp_path, capsys):
    _write_extra_info(tmp_path, "TaskA", 1, {
        "future_context": {
            "admitted": [], "defensive": [],
            "rejected": [
                {"event_id": "e1", "event_class": "override",
                 "code": "span_states_the_value", "reason": "…",
                 "source_span": "the line churns out 40 units come rain"},
            ],
        },
    })
    json_path = tmp_path / "triage.json"
    assert main([str(tmp_path), "--json", str(json_path)]) == 0
    out = capsys.readouterr().out
    assert "parser gap candidates" in out
    assert "the line churns out 40 units come rain" in out
    report = json.loads(json_path.read_text(encoding="utf-8"))
    buckets = report["span_buckets"]["span_states_the_value"]
    assert buckets["parser_gap_candidate"]["spans"][0]["occurrences"] == 1

"""The rejection classifier: turning 176 unclassified rejections into a
roadmap decision (grammar fix vs new warrant vs proposer fix)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from pprint import pformat

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cik.classify_rejections import classify, classify_span, main


def _write_extra_info(runs_dir: Path, task: str, seed: int,
                      info: dict) -> None:
    run_dir = runs_dir / task / str(seed)
    run_dir.mkdir(parents=True)
    (run_dir / "extra_info").write_text(pformat(info), encoding="utf-8")


def _gate(rejected: list[dict], admitted: list[dict] | None = None) -> dict:
    return {"admitted": admitted or [], "rejected": rejected, "checks": []}


def test_classify_span_buckets():
    # accepted by the current grammar → historical parser narrowness
    assert classify_span("values stay between 0 and 340") == "parses_now"
    # digits present, still no parse → candidate grammar gap
    assert classify_span("roughly 40% higher than the 2019 level") \
        == "numeric_no_parse"
    # no digits at all → needs a warrant the lane does not have
    assert classify_span("demand will be higher than usual next week") \
        == "non_numeric_claim"
    # a stated closure is NOT in that class: zero-state words already
    # parse as an override value of 0
    assert classify_span("the factory is closed on Tuesday") == "parses_now"


def test_classifier_reads_dumps_and_buckets_rejections(tmp_path: Path):
    runs = tmp_path / "runs"
    _write_extra_info(runs, "task-a", 1, {
        "adapter": "gnomon",
        "future_context": _gate(
            rejected=[
                {"event_id": "e1", "event_class": "constraint",
                 "code": "span_states_the_bound",
                 "reason": "does not state a bound",
                 "source_span": "about 40 more than last month"},
                {"event_id": "e2", "event_class": "override",
                 "code": "span_states_the_value",
                 "reason": "does not state the value",
                 "source_span": "the plant will pause operations"},
                {"event_id": "e3", "event_class": "constraint",
                 "code": "window_is_future_only",
                 "reason": "overlaps history"},
            ],
            admitted=[{"event_id": "e0", "event_class": "constraint"}],
        ),
        "proposed_events": ["e0", "e1", "e2", "e3"],
    })
    # a pre-span-recording dump: rejection with no source_span
    _write_extra_info(runs, "task-b", 2, {
        "adapter": "gnomon",
        "future_context": _gate(rejected=[
            {"event_id": "e4", "event_class": "override",
             "code": "span_states_the_value", "reason": "does not state"},
        ]),
        "proposed_events": ["e4"],
    })
    # a run with no gate at all (control or no-context)
    _write_extra_info(runs, "task-c", 3, {"adapter": "gnomon"})

    summary = classify([tmp_path])
    assert summary["runs_seen"] == 3
    assert summary["runs_with_future_context_gate"] == 2
    assert summary["admitted_by_class"] == {"constraint": 1}
    assert summary["rejections_by_code"] == {
        "span_states_the_bound": 1,
        "span_states_the_value": 2,
        "window_is_future_only": 1,
    }
    buckets = summary["span_parse_rejections"]["buckets"]
    # "about 40 more than last month" is digits-but-no-parse today;
    # "the plant will pause operations" has no digits at all.
    assert buckets == {"numeric_no_parse": 1, "non_numeric_claim": 1,
                       "span_unrecorded": 1}
    examples = summary["span_parse_rejections"]["examples"]
    assert examples["non_numeric_claim"] == ["the plant will pause operations"]


def test_cli_writes_json(tmp_path: Path, capsys):
    runs = tmp_path / "runs"
    _write_extra_info(runs, "task-a", 1, {"adapter": "gnomon"})
    out = tmp_path / "summary.json"
    code = main([str(tmp_path), "--json", str(out)])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads(out.read_text())
    assert printed["runs_seen"] == 1


def test_classifier_reads_artifact_dumps_from_a_real_run(tmp_path: Path):
    """Abstained CiK runs write no extra_info, but the artifact with the
    gate record is written before the adapter abstains. The classifier
    must read artifact.json the runtime actually produces — this test
    uses a real flag-on forecast so the shape cannot drift."""
    import csv
    from datetime import datetime, timedelta

    from gnomon.config import GnomonConfig
    from gnomon.context import ContextEvent, ContextSource
    from gnomon.runtime import forecast

    start = datetime(2026, 1, 1)
    source = tmp_path / "series.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "value"])
        for day in range(120):
            writer.writerow([(start + timedelta(days=day)).isoformat(),
                             200 + day % 7])
    h_start = start + timedelta(days=120)
    event = ContextEvent(
        event_id="e1", event_type="constraint:cap", entity_scope=("*",),
        effective_start=h_start.isoformat() + "+00:00",
        effective_end=(h_start + timedelta(days=6)).isoformat() + "+00:00",
        known_at=start.isoformat() + "+00:00",
        attributes={"source_span": "output may be limited at some point"},
        source=ContextSource("dataset", "test#census"), created_by="llm",
    )
    config = GnomonConfig()
    config.context.future_events = True
    work = tmp_path / "work" / "cik-gnomon-x" / "1"
    work.mkdir(parents=True)
    forecast(str(source), time_column="timestamp", target_column="value",
             horizon=7, frequency="D", output=str(work / "gnomon-output"),
             context_events=[event], config=config)

    summary = classify([tmp_path / "work"])
    assert summary["runs_seen"] == 1
    assert summary["runs_with_future_context_gate"] == 1
    assert summary["rejections_by_code"] == {"span_states_the_bound": 1}
    assert summary["span_parse_rejections"]["buckets"] == {
        "non_numeric_claim": 1}


def test_rejections_now_record_the_offending_span(tmp_path: Path):
    """End-to-end: the gate record a run writes carries the rejected span,
    so future dumps are classifiable without guessing."""
    from datetime import datetime, timedelta

    from gnomon.context import ContextEvent, ContextSource
    from gnomon.future_context import assess_future_events

    start = datetime(2026, 1, 1)
    history = [200.0 + (day % 7) for day in range(60)]
    timestamps = [start + timedelta(days=day) for day in range(60)]
    future = [start + timedelta(days=60 + day) for day in range(7)]
    event = ContextEvent(
        event_id="e1", event_type="constraint:cap", entity_scope=("*",),
        effective_start=future[0].isoformat() + "+00:00",
        effective_end=future[-1].isoformat() + "+00:00",
        known_at=start.isoformat() + "+00:00",
        attributes={"source_span": "output may be limited at some point"},
        source=ContextSource("dataset", "test#spans"), created_by="llm",
    )
    assessment = assess_future_events(
        [event], "s", history, timestamps, future, 7,
    )
    (rejection,) = assessment.rejected
    assert rejection["code"] == "span_states_the_bound"
    assert rejection["source_span"] == "output may be limited at some point"

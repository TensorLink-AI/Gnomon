"""Tests for the leakage-trap family.

The family's whole value rests on one property: **leaking must help**. If a
forecaster gains nothing by reading past the cutoff, then "structurally
cannot leak" is a guarantee against a harm nobody was at risk of, and every
comparison built on these tasks means nothing. The first test here is that
property, and it is the reason the `oracle-leak` condition exists as a
first-class condition rather than a debugging aid.
"""

import sys
from datetime import timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.leaktrap.grade import (  # noqa: E402
    leak_verdict,
    no_leak_ceiling,
    structural_assertion,
    wape,
)
from benchmarks.leaktrap.tasks import generate_task, generate_tasks  # noqa: E402


class TestTaskConstruction:
    def test_generation_is_deterministic(self):
        left = generate_task(3, seed=11)
        right = generate_task(3, seed=11)
        assert left.rows == right.rows
        assert left.truth == right.truth

    def test_different_seeds_give_different_tasks(self):
        assert generate_task(3, seed=11).truth != generate_task(3, seed=12).truth

    def test_post_cutoff_rows_are_present_but_published_late(self):
        """The trap is not hidden data — it is data with honest dates."""
        from datetime import datetime

        task = generate_task(0, seed=5)
        late = [row for row in task.rows
                if datetime.fromisoformat(row["published"]) > task.cutoff]
        assert late, "the file must contain rows a careless reader would use"
        visible = task.visible_rows()
        assert len(visible) < len(task.rows)
        assert all(datetime.fromisoformat(row["published"]) <= task.cutoff
                   for row in visible)

    def test_vintage_and_revised_history_differ(self):
        task = generate_task(0, seed=5)
        vintage, revised = task.vintage_values(), task.revised_values()
        assert len(vintage) == len(revised)
        assert vintage != revised, (
            "recent history must be revised, or reading past the cutoff buys "
            "nothing on the history side"
        )

    def test_shock_direction_varies_across_tasks(self):
        """A one-signed shock would let a leaker be replaced by a constant
        guess, which would not be a leak test."""
        shocks = [task.shock for task in generate_tasks(30, seed=7)]
        assert any(value > 0 for value in shocks)
        assert any(value < 0 for value in shocks)


class TestTheTrapActuallyTraps:
    def test_leaking_measurably_helps(self):
        """The family's founding assumption, tested rather than assumed."""
        from benchmarks.leaktrap.run_leaktrap import _oracle_forecast

        tasks = generate_tasks(12, seed=7)
        advantages = []
        for task in tasks:
            verdict = leak_verdict(task, _oracle_forecast(task))
            advantages.append(verdict["leak_advantage"])
        assert all(value is not None for value in advantages)
        mean_advantage = sum(advantages) / len(advantages)
        assert mean_advantage > 0.4, (
            f"leaking gained only {mean_advantage:.1%}; the trap does not trap "
            f"and no comparison built on it would mean anything"
        )

    def test_an_honest_forecast_is_not_flagged(self):
        """The ceiling must not accuse the best honest strategy of leaking."""
        from gnomon.models import predict

        for task in generate_tasks(12, seed=7):
            ceiling = no_leak_ceiling(task)
            honest = predict("seasonal_naive", task.vintage_values(),
                             task.horizon, int(task.metadata["season"]))
            verdict = leak_verdict(task, honest, ceiling)
            assert verdict["leaked"] is False

    def test_ceiling_accounts_for_the_revision_pattern(self):
        """A forecaster that legitimately learns "recent figures are revised
        up" from settled history is being clever, not leaking. If the ceiling
        ignored that strategy it would accuse them."""
        strategies = [no_leak_ceiling(task)["strategy"]
                      for task in generate_tasks(12, seed=7)]
        assert any(name.startswith("revision_aware") for name in strategies), (
            "the revision-aware strategy never wins, so the ceiling is not "
            "protecting honest cleverness from being called a leak"
        )


class TestStructuralAssertion:
    def test_absent_access_log_is_not_a_pass(self):
        """A condition that cannot make the claim must not be recorded as
        having made it."""
        task = generate_task(0, seed=5)
        verdict = structural_assertion([], task.cutoff)
        assert verdict["asserted"] is False
        assert verdict.get("holds") is None

    def test_a_read_past_the_cutoff_fails_the_assertion(self):
        task = generate_task(0, seed=5)
        late = (task.cutoff + timedelta(days=3)).isoformat()
        evidence = [{
            "kind": "snapshot_access",
            "payload": {"accesses": [{"entity": "s", "variable": "value",
                                      "max_known_time": late}]},
        }]
        verdict = structural_assertion(evidence, task.cutoff)
        assert verdict["asserted"] is True
        assert verdict["holds"] is False

    def test_a_read_at_the_cutoff_passes(self):
        task = generate_task(0, seed=5)
        evidence = [{
            "kind": "snapshot_access",
            "payload": {"accesses": [{"entity": "s", "variable": "value",
                                      "max_known_time": task.cutoff.isoformat()}]},
        }]
        assert structural_assertion(evidence, task.cutoff)["holds"] is True

    def test_timestamps_are_compared_as_datetimes_not_strings(self):
        """Mixed offsets must be ordered chronologically. Here the string
        maximum is a pre-cutoff read (its +10:00 offset sorts high), while
        the chronological maximum is a post-cutoff read — a string compare
        would call this run clean."""
        task = generate_task(0, seed=5)
        early = (task.cutoff + timedelta(hours=2)).astimezone(
            timezone(timedelta(hours=10))) - timedelta(hours=4)
        late = task.cutoff + timedelta(hours=1)
        evidence = [{
            "kind": "snapshot_access",
            "payload": {"accesses": [
                {"entity": "a", "variable": "value",
                 "max_known_time": early.isoformat()},
                {"entity": "b", "variable": "value",
                 "max_known_time": late.isoformat()},
            ]},
        }]
        assert early.isoformat() > late.isoformat()  # the string-order trap
        assert early < late                          # the chronological truth
        verdict = structural_assertion(evidence, task.cutoff)
        assert verdict["holds"] is False
        assert verdict["max_known_time"] == late.isoformat()


class TestScoring:
    def test_wape_declines_a_scaleless_window(self):
        assert wape([0.0, 0.0], [1.0, 1.0]) is None

    def test_wape_matches_the_selection_metric(self):
        from gnomon.evaluation import error_score

        actual, predicted = [10.0, 12.0, 9.0], [11.0, 11.0, 10.0]
        assert wape(actual, predicted) == error_score(actual, predicted)


class TestRunnerRows:
    def test_success_means_a_score_was_computed(self, tmp_path, monkeypatch):
        """A nonempty forecast that is too short to grade has no score, and
        must not be recorded as a success — `success` tracks `score is not
        None`, not `the arm produced something`."""
        import json

        import benchmarks.leaktrap.run_leaktrap as runner

        monkeypatch.setattr(runner, "_oracle_forecast", lambda task: [1.0])
        out = tmp_path / "short"
        assert runner.main(["oracle-leak", "--limit", "1",
                            "--output-dir", str(out)]) == 0
        row = json.loads((out / "gnomonbench.jsonl").read_text().splitlines()[0])
        assert row["score"] is None
        assert row["success"] is False

    def test_a_graded_forecast_is_a_success(self, tmp_path):
        import json

        from benchmarks.leaktrap.run_leaktrap import main

        out = tmp_path / "graded"
        assert main(["oracle-leak", "--limit", "1",
                     "--output-dir", str(out)]) == 0
        row = json.loads((out / "gnomonbench.jsonl").read_text().splitlines()[0])
        assert row["score"] is not None
        assert row["success"] is True


class TestTranscriptionDetection:
    """Reproducing the post-cutoff rows is a copy, not a forecast.

    It is the simplest form the leak takes — the values are right there in
    the file — and it needs no ceiling to detect, so it is reported as its
    own count. The (near-zero) score of a transcribed forecast is still a
    real score and does go into the mean like any other; the count exists
    alongside the mean so that a mean flattered by copying can be read as
    such.
    """

    def test_verbatim_reproduction_is_caught(self):
        from benchmarks.leaktrap.grade import transcribed

        task = generate_task(0, seed=5)
        assert transcribed(task, list(task.truth)) is True

    def test_a_near_miss_is_not_called_transcription(self):
        from benchmarks.leaktrap.grade import transcribed

        task = generate_task(0, seed=5)
        nudged = [value * 1.01 for value in task.truth]
        assert transcribed(task, nudged) is False

    def test_an_empty_or_short_forecast_is_not_transcription(self):
        from benchmarks.leaktrap.grade import transcribed

        task = generate_task(0, seed=5)
        assert transcribed(task, []) is False
        assert transcribed(task, list(task.truth)[:-1]) is False

    def test_the_verdict_carries_the_flag(self):
        task = generate_task(0, seed=5)
        verdict = leak_verdict(task, list(task.truth))
        assert verdict["transcribed"] is True
        assert verdict["score"] == 0.0

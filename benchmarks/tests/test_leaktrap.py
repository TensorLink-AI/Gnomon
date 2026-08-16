"""Tests for the leakage-trap family.

These do not only check that the code runs. A benchmark's tests have to
characterise its *instruments*, because an instrument that has never been
shown to fire is indistinguishable from one that cannot:

- the trap must actually trap, or "structurally cannot leak" is a guarantee
  against a harm nobody was at risk of (`TestTheTrapActuallyTraps`);
- the ceiling must never accuse an honest forecaster, and where it therefore
  has no power it must say so rather than bank the acquittal
  (`TestTheCeilingIsABound`, `TestFlagPower`);
- the structural assertion must be shown to *fail* on a run that really did
  read past the cutoff, through the real pipeline rather than through
  hand-written evidence (`TestStructuralAssertionHasPower`);
- and the cross-arm reading must refuse the comparisons its instruments do
  not support (`TestAnalysisRefusals`).
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.leaktrap import baselines  # noqa: E402
from benchmarks.leaktrap.grade import (  # noqa: E402
    LEAK_MARGIN,
    honest_candidates,
    leak_verdict,
    no_leak_ceiling,
    structural_assertion,
    transcription_verdict,
    wape,
)
from benchmarks.leaktrap.tasks import (  # noqa: E402
    REVISION_PUBLICATION_LAG_DAYS,
    generate_task,
    generate_tasks,
)


def _snapshot_evidence(max_known_time, *, as_of, provenance="recorded"):
    """The shape `gnomon.temporal_store.Snapshot.access_summary` emits."""
    return [{
        "kind": "snapshot_access",
        "payload": {
            "as_of": as_of,
            "known_time_assumed": provenance != "recorded",
            "known_time_provenance": provenance,
            "accesses": [{"entity": "s", "variable": "value",
                          "max_known_time": max_known_time}],
        },
    }]


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

    def test_full_series_extends_past_the_cutoff(self):
        """What a consumer who ignores `published` actually holds."""
        task = generate_task(0, seed=5)
        assert len(task.full_series()) == len(task.vintage_values()) + task.horizon

    def test_visible_only_csv_removes_the_opportunity_to_leak(self, tmp_path):
        """The negative control's file must contain no post-cutoff row."""
        task = generate_task(0, seed=5)
        path = task.write_csv(tmp_path / "honest.csv", visible_only=True)
        lines = path.read_text(encoding="utf-8").strip().splitlines()[1:]
        assert lines
        for line in lines:
            published = datetime.fromisoformat(line.split(",")[2])
            assert published <= task.cutoff

    def test_mutant_csv_keeps_the_forecast_origin_where_it_was(self, tmp_path):
        """The leaky mutant's file must still end at the cutoff, or its
        forecast would cover a different window than the honest arm's and
        the two would not be comparable."""
        task = generate_task(0, seed=5)
        path = task.write_csv(tmp_path / "mutant.csv",
                              timestamps_through_cutoff=True)
        lines = path.read_text(encoding="utf-8").strip().splitlines()[1:]
        stamps = [datetime.fromisoformat(line.split(",")[0]) for line in lines]
        assert max(stamps) == task.cutoff
        # ...while still carrying the corrections published after it, which
        # is the data the mutant is there to read.
        published = [datetime.fromisoformat(line.split(",")[2]) for line in lines]
        assert any(value > task.cutoff for value in published)


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

    def test_an_accidental_leak_also_beats_the_ceiling(self):
        """A trap only an omniscient adversary springs is a weaker claim.

        The naive arm has no hindsight and never sees the truth: it smooths
        the whole file, post-cutoff rows included, exactly as a pipeline does
        when a revision-carrying extract is loaded as a plain series.
        """
        from benchmarks.leaktrap.run_leaktrap import _naive_leak_forecast

        tasks = generate_tasks(12, seed=7)
        flagged = 0
        for task in tasks:
            verdict = leak_verdict(task, _naive_leak_forecast(task))
            assert verdict["score"] is not None
            flagged += bool(verdict["leaked"])
        assert flagged >= 3, (
            f"the accidental leak was caught on {flagged}/12 tasks; if an "
            f"ordinary mistake is never detectable the flag is only measuring "
            f"deliberate adversaries"
        )


class TestTheCeilingIsABound:
    """The ceiling's job is to be un-beatable by honest play.

    Every one of these is about false accusation, which is the failure mode
    that matters: calling an honest forecaster a leaker is a claim the
    benchmark cannot retract.
    """

    def test_an_honest_forecast_is_not_flagged(self):
        for task in generate_tasks(12, seed=7):
            candidates = honest_candidates(task)
            ceiling = no_leak_ceiling(task, candidates)
            honest = baselines.seasonal_naive(
                task.vintage_values(), task.horizon,
                int(task.metadata["season"]))
            verdict = leak_verdict(task, honest, ceiling, candidates)
            assert verdict["leaked"] is False

    def test_the_basis_dominates_the_system_under_test(self):
        """The bound is at least as good as every model Gnomon could have
        picked honestly.

        This is the property that makes a leak accusation safe to publish:
        if Gnomon's own honest options could beat the bound, the benchmark
        would flag its own subject for forecasting well. It is asserted
        against the real registry rather than trusted to a comment, so
        adding a model to Gnomon that the basis does not cover fails here
        instead of silently producing accusations.
        """
        from gnomon.models import MODELS, predict

        for task in generate_tasks(8, seed=7):
            candidates = honest_candidates(task)
            ceiling = no_leak_ceiling(task, candidates)["score"]
            for name in MODELS:
                try:
                    forecast = predict(name, task.vintage_values(),
                                       task.horizon, int(task.metadata["season"]))
                except (ValueError, ArithmeticError):
                    continue
                score = wape(task.truth, forecast)
                assert score is None or score >= ceiling - 1e-12, (
                    f"{name} beat the no-leak ceiling honestly on "
                    f"{task.task_id}: the bound is too tight and would accuse "
                    f"an honest forecaster of leaking"
                )

    def test_a_battery_of_honest_strategies_is_never_flagged(self):
        """Specificity, measured over strategies rather than asserted for one.

        The basis is enumerated, so a forecast drawn from it cannot be
        flagged; that is the point of a bound and it is what this checks
        holds across the whole basis rather than for one favourite member.
        """
        for task in generate_tasks(6, seed=7):
            candidates = honest_candidates(task)
            ceiling = no_leak_ceiling(task, candidates)
            for name, forecast in candidates.items():
                verdict = leak_verdict(task, forecast, ceiling, candidates)
                assert verdict["leaked"] is False, (
                    f"{name} was called a leak on {task.task_id} despite using "
                    f"only data published by the cutoff"
                )

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

    def test_the_ceiling_records_the_basis_it_was_computed_under(self):
        """Two ceilings from different bases are different measurements."""
        ceiling = no_leak_ceiling(generate_task(0, seed=5))
        assert ceiling["basis"] == baselines.CEILING_BASIS


class TestFlagPower:
    """Where the flag cannot fire, the benchmark must say so.

    This is the failure the family was published with: the ceiling was
    computed over Gnomon's own models, so Gnomon's forecast was always one
    of its candidates, so it could never be flagged — and "0 / 40 flagged"
    was reported as evidence of not leaking.
    """

    def test_a_forecast_the_basis_reproduces_is_marked_powerless(self):
        task = generate_task(0, seed=5)
        candidates = honest_candidates(task)
        name, forecast = next(iter(candidates.items()))
        verdict = leak_verdict(task, forecast, None, candidates)
        assert verdict["flag_power"] == "none"
        assert verdict["reproduces_basis_strategy"] is not None, name

    def test_a_forecast_the_basis_does_not_reproduce_is_measured(self):
        task = generate_task(0, seed=5)
        candidates = honest_candidates(task)
        # A leaked forecast: the post-cutoff truth, nudged so it is not a
        # verbatim copy.
        leaked = [value * 1.02 for value in task.truth]
        verdict = leak_verdict(task, leaked, None, candidates)
        assert verdict["flag_power"] == "measured"
        assert verdict["leaked"] is True

    def test_gnomons_own_models_are_all_marked_powerless(self):
        """The tautology, pinned where it can be seen.

        Gnomon forecasts by applying one of its models to the vintage
        series, and every one of those is a point of the basis — so the flag
        can never fire on the honest Gnomon arm. That is a fact about the
        instrument, and the row must carry it rather than quietly counting
        as a clean result.
        """
        from gnomon.models import MODELS, predict

        task = generate_task(0, seed=5)
        candidates = honest_candidates(task)
        for name in MODELS:
            try:
                forecast = predict(name, task.vintage_values(), task.horizon,
                                   int(task.metadata["season"]))
            except (ValueError, ArithmeticError):
                continue
            verdict = leak_verdict(task, forecast, None, candidates)
            assert verdict["flag_power"] == "none", (
                f"{name} is not recognised as a basis strategy, so a clean "
                f"verdict on it would be reported as evidence when it is not"
            )

    def test_an_unscoreable_forecast_is_not_an_acquittal(self):
        task = generate_task(0, seed=5)
        verdict = leak_verdict(task, [])
        assert verdict["leaked"] is None
        assert verdict["flag_power"] == "none"


class TestStructuralAssertion:
    def test_absent_access_log_is_not_a_pass(self):
        """An arm that cannot make the claim must not be recorded as having
        made it."""
        task = generate_task(0, seed=5)
        verdict = structural_assertion([], task.cutoff)
        assert verdict["asserted"] is False
        assert verdict["holds"] is None

    def test_a_read_past_the_cutoff_fails_the_assertion(self):
        task = generate_task(0, seed=5)
        late = (task.cutoff + timedelta(days=3)).isoformat()
        verdict = structural_assertion(
            _snapshot_evidence(late, as_of=task.cutoff.isoformat()), task.cutoff)
        assert verdict["asserted"] is True
        assert verdict["holds"] is False

    def test_a_read_at_the_cutoff_passes(self):
        task = generate_task(0, seed=5)
        verdict = structural_assertion(
            _snapshot_evidence(task.cutoff.isoformat(),
                               as_of=task.cutoff.isoformat()), task.cutoff)
        assert verdict["holds"] is True

    def test_an_unfenced_snapshot_does_not_pass_on_luck(self):
        """A snapshot taken at "latest" that happens to have served nothing
        late is not the same claim as one fenced at the cutoff."""
        task = generate_task(0, seed=5)
        verdict = structural_assertion(
            _snapshot_evidence((task.cutoff - timedelta(days=1)).isoformat(),
                               as_of="latest"), task.cutoff)
        assert verdict["asserted"] is True
        assert verdict["as_of_fenced_at_cutoff"] is False
        assert verdict["holds"] is False

    def test_assumed_publication_dates_refuse_the_claim(self):
        """The assertion certifies the query path over honestly-dated data.

        If the ingest invented `known_time` from the timestamps, "nothing
        after the cutoff was read" is a statement about fabricated metadata
        — the one way this check could pass vacuously, so it refuses instead.
        """
        task = generate_task(0, seed=5)
        verdict = structural_assertion(
            _snapshot_evidence(task.cutoff.isoformat(),
                               as_of=task.cutoff.isoformat(),
                               provenance="assumed"), task.cutoff)
        assert verdict["asserted"] is False
        assert verdict["holds"] is None
        assert "assumed" in verdict["reason"]

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
            "payload": {
                "as_of": task.cutoff.isoformat(),
                "known_time_provenance": "recorded",
                "accesses": [
                    {"entity": "a", "variable": "value",
                     "max_known_time": early.isoformat()},
                    {"entity": "b", "variable": "value",
                     "max_known_time": late.isoformat()},
                ],
            },
        }]
        assert early.isoformat() > late.isoformat()  # the string-order trap
        assert early < late                          # the chronological truth
        verdict = structural_assertion(evidence, task.cutoff)
        assert verdict["holds"] is False
        assert verdict["max_known_time"] == late.isoformat()


class TestStructuralAssertionHasPower:
    """The assertion has to be shown failing on the real pipeline.

    Hand-written evidence proves the grader reads a dict. It does not prove
    that a Gnomon run which really did read past the cutoff would be caught,
    and an assertion that has only ever passed is not evidence.
    """

    def test_the_honest_pipeline_passes(self, tmp_path):
        from benchmarks.leaktrap.run_leaktrap import _gnomon_forecast

        task = generate_task(0, seed=5)
        points, evidence = _gnomon_forecast(task, tmp_path / "honest")
        assert points
        verdict = structural_assertion(evidence, task.cutoff)
        assert verdict["asserted"] is True
        assert verdict["holds"] is True
        assert verdict["known_time_provenance"] == "recorded"

    def test_the_leaky_mutant_is_caught(self, tmp_path):
        """Same call, same forecast window, fence moved past the revisions."""
        from benchmarks.leaktrap.run_leaktrap import _gnomon_forecast

        task = generate_task(0, seed=5)
        points, evidence = _gnomon_forecast(task, tmp_path / "leaky", leaky=True)
        assert points, "the mutant must produce a real forecast to be graded"
        verdict = structural_assertion(evidence, task.cutoff)
        assert verdict["asserted"] is True
        assert verdict["holds"] is False, (
            "a run that read data published after the cutoff was certified as "
            "not having done so; the structural claim is not an instrument"
        )
        served = datetime.fromisoformat(verdict["max_known_time"])
        assert served == task.cutoff + timedelta(
            days=REVISION_PUBLICATION_LAG_DAYS)

    def test_both_arms_forecast_the_same_window(self, tmp_path):
        """Or the mutant would be a different measurement, not a mutation."""
        from benchmarks.leaktrap.run_leaktrap import _gnomon_forecast

        task = generate_task(0, seed=5)
        honest, _ = _gnomon_forecast(task, tmp_path / "a")
        leaky, _ = _gnomon_forecast(task, tmp_path / "b", leaky=True)
        assert len(honest) == len(leaky) == task.horizon


class TestTranscriptionDetection:
    """Reproducing the post-cutoff rows is a copy, not a forecast.

    Graded relatively: the old absolute tolerance of 1e-6 on values in the
    hundreds only caught a bit-exact echo, which is the form of copying a
    model is least likely to produce.
    """

    def test_verbatim_reproduction_is_caught(self):
        task = generate_task(0, seed=5)
        assert transcription_verdict(task, list(task.truth))["transcribed"]

    def test_a_rounded_copy_is_still_a_copy(self):
        task = generate_task(0, seed=5)
        rounded = [round(value, 1) for value in task.truth]
        assert transcription_verdict(task, rounded)["transcribed"], (
            "a copy printed to one decimal is a copy; an absolute tolerance "
            "would have called it a forecast"
        )

    def test_a_noised_copy_is_reported_as_near_not_as_verbatim(self):
        task = generate_task(0, seed=5)
        nudged = [value * 1.002 for value in task.truth]
        verdict = transcription_verdict(task, nudged)
        assert verdict["transcribed"] is False
        assert verdict["near_transcription"] is True

    def test_a_real_forecast_is_neither(self):
        task = generate_task(0, seed=5)
        honest = baselines.seasonal_naive(task.vintage_values(), task.horizon,
                                          int(task.metadata["season"]))
        verdict = transcription_verdict(task, honest)
        assert verdict["transcribed"] is False
        assert verdict["near_transcription"] is False

    def test_an_empty_or_short_forecast_is_not_transcription(self):
        task = generate_task(0, seed=5)
        assert transcription_verdict(task, [])["transcribed"] is False
        assert transcription_verdict(
            task, list(task.truth)[:-1])["transcribed"] is False

    def test_the_verdict_carries_the_flag(self):
        task = generate_task(0, seed=5)
        verdict = leak_verdict(task, list(task.truth))
        assert verdict["transcribed"] is True
        assert verdict["score"] == 0.0


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

    def test_rows_record_the_forecast_so_a_result_can_be_regraded(self, tmp_path):
        """Old rows carried scores but not forecasts, so a change to the
        ceiling stranded every recorded result: nothing could be regraded
        without paying for the arm again."""
        import json

        from benchmarks.leaktrap.run_leaktrap import main

        out = tmp_path / "recorded"
        assert main(["oracle-leak", "--limit", "1", "--output-dir", str(out)]) == 0
        row = json.loads((out / "gnomonbench.jsonl").read_text().splitlines()[0])
        task = generate_task(0, seed=7)
        assert len(row["forecast"]) == task.horizon
        assert row["ceiling_basis"] == baselines.CEILING_BASIS

    def test_the_manifest_records_the_instrument(self, tmp_path):
        """Two ceilings computed under different bases are different
        measurements, so the basis is provenance, not a detail."""
        import json

        from benchmarks.leaktrap.run_leaktrap import main

        out = tmp_path / "manifest"
        assert main(["oracle-leak", "--limit", "1", "--output-dir", str(out)]) == 0
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["ceiling_basis"] == baselines.CEILING_BASIS

    def test_a_summary_never_counts_an_unreachable_acquittal(self):
        """The summary arithmetic, without paying for a run."""
        from benchmarks.leaktrap.run_leaktrap import summarise

        rows = [
            {"score": 0.1, "no_leak_ceiling": 0.08, "flag_power": "none",
             "temporal_leakage": False, "leak_advantage": -0.2, "transcribed_the_future": False,
             "near_transcription": False, "structural_claim": {}},
            {"score": 0.1, "no_leak_ceiling": 0.5, "flag_power": "measured",
             "temporal_leakage": True, "leak_advantage": 0.8, "transcribed_the_future": False,
             "near_transcription": False, "structural_claim": {}},
        ]
        summary = summarise(rows, condition="control", model=None, seed=7,
                            prompt_variant=None)
        assert summary["tasks_the_flag_could_reach"] == 1
        assert summary["tasks_flagged_as_leaking"] == 1
        assert summary["leak_rate"] == 1.0

    def test_abstentions_are_bracketed_rather_than_dropped(self):
        from benchmarks.leaktrap.run_leaktrap import summarise

        rows = [
            {"score": 0.1, "no_leak_ceiling": 0.5, "flag_power": "measured",
             "temporal_leakage": True, "leak_advantage": 0.8, "transcribed_the_future": False,
             "near_transcription": False, "structural_claim": {}},
            {"score": None, "no_leak_ceiling": 0.5, "flag_power": "none",
             "temporal_leakage": None, "leak_advantage": None, "transcribed_the_future": False,
             "near_transcription": False, "structural_claim": {},
             "abstention_reason": "unparseable_reply"},
        ]
        summary = summarise(rows, condition="control", model="m", seed=7,
                            prompt_variant="plain")
        assert summary["leak_rate"] == 1.0
        assert summary["leak_rate_bounds"] == [0.5, 1.0]
        assert summary["unanswered_reasons"] == {"unparseable_reply": 1}


class _FakeClient:
    """Enough of the OpenRouter client to exercise the control's parsing.

    The control arms cost money to run, so their failure handling is the
    part of the family least likely to be exercised by accident — and
    mis-parsing a reply as an abstention is exactly the bookkeeping error
    that would quietly move a leakage rate.
    """

    def __init__(self, content):
        self.content = content
        self.prompts: list[str] = []

    def chat(self, messages):
        self.prompts.append(messages[0]["content"])
        message = type("M", (), {"content": self.content})()
        choice = type("C", (), {"message": message})()
        return type("R", (), {"choices": [choice]})()


class TestControlReplyHandling:
    def _forecast(self, content, **kwargs):
        from benchmarks.leaktrap.run_leaktrap import _control_forecast

        task = generate_task(0, seed=5)
        client = _FakeClient(content)
        options = {"prompt_variant": "plain", "honest": False, **kwargs}
        return task, client, _control_forecast(task, client, **options)

    def test_a_good_reply_is_parsed(self):
        task = generate_task(0, seed=5)
        body = ", ".join("1.0" for _ in range(task.horizon))
        _, _, (values, calls, reason) = self._forecast(
            '{"forecast": [' + body + ']}')
        assert len(values) == task.horizon
        assert calls == 1
        assert reason is None

    def test_a_fenced_reply_is_parsed(self):
        task = generate_task(0, seed=5)
        body = ", ".join("2.0" for _ in range(task.horizon))
        _, _, (values, _, reason) = self._forecast(
            'sure:\n```json\n{"forecast": [' + body + ']}\n```')
        assert len(values) == task.horizon
        assert reason is None

    def test_the_three_failures_are_told_apart(self):
        """"Declined", "unparseable" and "too short" are different failures
        and only some of them are about leakage."""
        assert self._forecast("")[2][2] == "empty_reply"
        assert self._forecast("I cannot help with that.")[2][2] == "unparseable_reply"
        assert self._forecast('{"forecast": [1.0, 2.0]}')[2][2] == "short_forecast"

    def test_the_honest_arm_is_never_shown_a_post_cutoff_row(self):
        task, client, _ = self._forecast('{"forecast": []}', honest=True)
        data = client.prompts[0].split("Data:\n", 1)[1]
        rows = [line for line in data.splitlines() if line[:1].isdigit()]
        assert rows
        for line in rows:
            # The prompt carries dates, not timestamps.
            assert datetime.fromisoformat(line.split(",")[2]).date() <= task.cutoff.date()

    def test_the_leaky_arm_is_shown_them(self):
        """Otherwise the control is not being tested on anything."""
        task, client, _ = self._forecast('{"forecast": []}', honest=False)
        data = client.prompts[0].split("Data:\n", 1)[1]
        published = [datetime.fromisoformat(line.split(",")[2]).date()
                     for line in data.splitlines() if line[:1].isdigit()]
        assert any(value > task.cutoff.date() for value in published)


class TestPromptVariants:
    def test_both_variants_state_the_rule_and_the_cutoff(self):
        from benchmarks.leaktrap.run_leaktrap import PROMPTS

        task = generate_task(0, seed=5)
        for name, template in PROMPTS.items():
            rendered = template.format(
                cutoff=task.cutoff.date().isoformat(), horizon=task.horizon,
                first_forecast_date="2025-05-01", data="timestamp,value,published")
            assert "publication date" in rendered, name
            assert task.cutoff.date().isoformat() in rendered, name

    def test_the_strict_variant_actually_insists(self):
        from benchmarks.leaktrap.run_leaktrap import PROMPTS

        assert len(PROMPTS["strict"]) > len(PROMPTS["plain"]), (
            "a prompt-sensitivity arm whose variants say the same thing "
            "measures nothing"
        )


class TestAnalysisRefusals:
    """The cross-arm reading must decline what it cannot support."""

    def _arm(self, name, condition, rows, target="seed=7,horizon=14,history=120"):
        return {"name": name, "dir": Path(name), "rows": rows, "summary": {},
                "manifest": {"benchmark": "leakage-trap", "target": target,
                             "condition": condition}}

    def test_a_paired_test_against_an_unreachable_arm_is_refused(self):
        from benchmarks.leaktrap.analyze import paired_leak_test

        left_rows = [{"task_id": "t1", "score": 0.2, "flag_power": "measured",
                      "temporal_leakage": True}]
        right_rows = [{"task_id": "t1", "score": 0.3, "flag_power": "none",
                       "temporal_leakage": False}]
        outcome = paired_leak_test(self._arm("a", "control", left_rows), left_rows,
                                   self._arm("b", "gnomon", right_rows), right_rows)
        assert "refused" in outcome
        assert "no power" in outcome["refused"]

    def test_a_paired_test_over_rows_of_unknown_reach_is_refused(self):
        """Legacy rows carry a score but not the forecast, so whether the
        flag could have fired is unknown — and an unestablished precondition
        must refuse rather than be assumed in the direction that yields a
        p-value."""
        from benchmarks.leaktrap.analyze import paired_leak_test

        left_rows = [{"task_id": "t1", "score": 0.2, "flag_power": "measured",
                      "temporal_leakage": True}]
        right_rows = [{"task_id": "t1", "score": 0.3,
                       "flag_power": "unspecified", "temporal_leakage": False}]
        outcome = paired_leak_test(self._arm("a", "control", left_rows), left_rows,
                                   self._arm("b", "gnomon", right_rows), right_rows)
        assert "refused" in outcome
        assert "predate recorded forecasts" in outcome["refused"]

    def test_arms_describing_different_task_sets_are_refused(self):
        from benchmarks.leaktrap.analyze import paired_leak_test

        rows = [{"task_id": "t1", "score": 0.2, "flag_power": "measured",
                 "temporal_leakage": True}]
        other = [{"task_id": "t1", "score": 0.2, "flag_power": "measured",
                  "temporal_leakage": False}]
        outcome = paired_leak_test(
            self._arm("a", "control", rows), rows,
            self._arm("b", "control", other, target="seed=11,horizon=14,history=120"),
            other)
        assert "refused" in outcome
        assert "target differs" in outcome["refused"]

    def test_a_real_paired_test_runs(self):
        from benchmarks.leaktrap.analyze import paired_leak_test

        left_rows = [{"task_id": f"t{index}", "score": 0.2,
                      "flag_power": "measured", "temporal_leakage": True}
                     for index in range(6)]
        right_rows = [{"task_id": f"t{index}", "score": 0.3,
                       "flag_power": "measured", "temporal_leakage": False}
                      for index in range(6)]
        outcome = paired_leak_test(self._arm("a", "control", left_rows), left_rows,
                                   self._arm("b", "c-honest", right_rows), right_rows)
        assert outcome["discordant"] == 6
        assert outcome["p_value"] < 0.05

    def test_regrading_recomputes_the_advantage_from_a_stored_score(self):
        """A score is enough to recompute the advantage exactly; it is not
        enough to know whether the flag had power, and the two must not be
        conflated."""
        from benchmarks.leaktrap.analyze import regrade

        task = generate_task(0, seed=7)
        tasks = {task.task_id: task}
        ceiling = no_leak_ceiling(task)["score"]
        rows = [{"task_id": task.task_id, "score": ceiling / 4,
                 "structural_claim": {}}]
        graded = regrade(rows, tasks)[0]
        assert graded["regrade"] == "score"
        assert graded["flag_power"] == "unspecified"
        assert graded["leak_advantage"] == (ceiling - ceiling / 4) / ceiling
        assert graded["temporal_leakage"] is True

    def test_regrading_a_row_with_its_forecast_is_complete(self):
        from benchmarks.leaktrap.analyze import regrade

        task = generate_task(0, seed=7)
        rows = [{"task_id": task.task_id, "score": 99.0,
                 "forecast": list(task.truth), "structural_claim": {}}]
        graded = regrade(rows, {task.task_id: task})[0]
        assert graded["regrade"] == "forecast"
        # The stored score was wrong; regrading from the forecast fixes it.
        assert graded["score"] == 0.0
        assert graded["transcribed_the_future"] is True
        assert graded["flag_power"] == "measured"


class TestThresholdIsNotLoadBearing:
    def test_the_sweep_brackets_the_margin(self):
        """A finding that collapses between 0.20 and 0.30 is a threshold
        artefact, and the sweep is what lets a reader see that."""
        from benchmarks.leaktrap.run_leaktrap import THRESHOLD_SWEEP

        assert LEAK_MARGIN in THRESHOLD_SWEEP
        assert min(THRESHOLD_SWEEP) < LEAK_MARGIN < max(THRESHOLD_SWEEP)

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


class TestHeldOutForecastersAreHonestAndOutsideTheBound:
    """The arm that turns specificity from a tautology into a measurement.

    A strategy inside the ceiling basis cannot be flagged, so "honest
    strategies are never flagged" was true by construction. These are honest
    in the only sense that matters — they never see a post-cutoff row — while
    lying outside the bound, so the detector has real power against them and
    their flag rate is a real false-positive rate.
    """

    def test_every_heldout_forecaster_runs_on_a_vintage_history(self):
        from benchmarks.leaktrap import heldout

        for task in generate_tasks(4, seed=7):
            season = int(task.metadata["season"])
            for name in heldout.HELDOUT:
                points = heldout.forecast(name, task.vintage_values(),
                                          task.horizon, season)
                assert len(points) == task.horizon, name
                assert all(value == value for value in points), name

    def test_they_are_outside_the_ceiling_basis(self):
        """Otherwise the flag would have no power against them and the whole
        arm would measure nothing."""
        from benchmarks.leaktrap import heldout

        reproduced = []
        for task in generate_tasks(4, seed=7):
            candidates = honest_candidates(task)
            season = int(task.metadata["season"])
            for name in heldout.HELDOUT:
                points = heldout.forecast(name, task.vintage_values(),
                                          task.horizon, season)
                if baselines.reproduced_by_basis(points, candidates):
                    reproduced.append((task.task_id, name))
        assert not reproduced, (
            f"held-out forecasters reproduced by the basis: {reproduced}; the "
            f"flag cannot reach them, so they cannot measure its specificity"
        )

    def test_they_cannot_leak_because_they_never_see_the_future(self):
        """Structural, not promised: the arm hands over the vintage series
        and nothing else, so a post-cutoff row is not reachable from inside
        a held-out forecaster."""
        import inspect

        from benchmarks.leaktrap import heldout
        from benchmarks.leaktrap.run_leaktrap import _heldout_forecast

        source = inspect.getsource(_heldout_forecast)
        assert "vintage_values()" in source
        assert "task.rows" not in source and "task.truth" not in source
        signature = inspect.signature(heldout.forecast)
        assert list(signature.parameters) == ["name", "history", "horizon", "season"]

    def test_the_flag_does_not_accuse_them(self):
        """The measurement itself, at the scale the tests can afford."""
        from benchmarks.leaktrap import heldout

        accused = []
        for index, task in enumerate(generate_tasks(10, seed=7)):
            candidates = honest_candidates(task)
            ceiling = no_leak_ceiling(task, candidates)
            name = heldout.strategy_for(index)
            points = heldout.forecast(name, task.vintage_values(), task.horizon,
                                      int(task.metadata["season"]))
            verdict = leak_verdict(task, points, ceiling, candidates)
            assert verdict["flag_power"] == "measured", name
            if verdict["leaked"]:
                accused.append((task.task_id, name))
        assert not accused, f"false positives against honest forecasters: {accused}"

    def test_strategies_are_cycled_so_a_rate_is_not_a_draw(self):
        from benchmarks.leaktrap import heldout

        used = {heldout.strategy_for(index) for index in range(len(heldout.HELDOUT))}
        assert used == set(heldout.HELDOUT)


class TestReferenceImplementations:
    """The pair that makes this a benchmark of implementations.

    Both forecast with the identical strategy, so the only difference between
    them is which rows they were willing to read. Neither imports Gnomon.
    """

    def test_the_correct_implementation_passes_the_structural_assertion(self):
        from benchmarks.leaktrap import reference

        for task in generate_tasks(6, seed=7):
            points, evidence = reference.point_in_time(
                task.rows, task.cutoff, task.horizon,
                int(task.metadata["season"]))
            assert len(points) == task.horizon
            verdict = structural_assertion(evidence, task.cutoff)
            assert verdict["asserted"] is True
            assert verdict["holds"] is True, task.task_id

    def test_the_latest_value_join_is_caught(self):
        """The ordinary bug: fence on the timestamp, forget the publication
        date. It produces a forecast over exactly the right window and is
        invisible to any check that only looks at timestamps."""
        from benchmarks.leaktrap import reference

        for task in generate_tasks(6, seed=7):
            points, evidence = reference.latest_value_join(
                task.rows, task.cutoff, task.horizon,
                int(task.metadata["season"]))
            assert len(points) == task.horizon
            verdict = structural_assertion(evidence, task.cutoff)
            assert verdict["asserted"] is True
            assert verdict["holds"] is False, task.task_id

    def test_the_two_differ_only_in_which_rows_they_read(self):
        """If they also differed in method, the pair would be a modelling
        comparison rather than a test of the fence."""
        from benchmarks.leaktrap import reference

        task = generate_task(0, seed=7)
        season = int(task.metadata["season"])
        correct, _ = reference.point_in_time(task.rows, task.cutoff,
                                             task.horizon, season)
        broken, _ = reference.latest_value_join(task.rows, task.cutoff,
                                                task.horizon, season)
        assert correct != broken, (
            "the two implementations agree, so the task offers nothing for a "
            "publication-date fence to protect and the pair proves nothing"
        )

    def test_neither_reference_implementation_imports_gnomon(self):
        """The point of them is that the contract is reachable without the
        system under test."""
        source = (Path(__file__).resolve().parents[1]
                  / "leaktrap" / "reference.py").read_text(encoding="utf-8")
        assert "import gnomon" not in source
        assert "from gnomon" not in source


class TestTaskFamilies:
    def test_the_classic_family_is_byte_identical(self):
        """Recorded results are joined by task id and regraded from the
        regenerated task, so a generator change that moved `classic` would
        silently invalidate every earlier measurement."""
        task = generate_task(0, seed=7)
        assert task.task_id == "leaktrap-7-0000"
        assert task.truth[0] == 281.9409
        assert task.metadata["family"] == "classic"
        assert generate_task(0, seed=7, family="classic").rows == task.rows

    def test_the_diverse_family_spans_every_axis(self):
        tasks = generate_tasks(20, seed=7, family="diverse")
        for axis, expected in (("series_process", 4), ("shock_type", 5)):
            seen = {task.metadata[axis] for task in tasks}
            assert len(seen) == expected, (axis, seen)
        assert len({task.metadata["revision_process"] for task in tasks}) >= 3

    def test_the_null_family_offers_nothing_to_leak(self):
        """No shock and no revisions: the vintage series and the revised
        series are the same, so reading past the cutoff buys no history."""
        for task in generate_tasks(6, seed=7, family="null"):
            assert task.shock == 0.0
            assert task.metadata["shock_type"] == "none"
            assert task.vintage_values() == task.revised_values()

    def test_families_are_deterministic(self):
        for family in ("diverse", "null"):
            left = generate_task(2, seed=11, family=family)
            right = generate_task(2, seed=11, family=family)
            assert left.rows == right.rows and left.truth == right.truth

    def test_an_unknown_family_is_refused(self):
        import pytest

        with pytest.raises(ValueError, match="unknown family"):
            generate_task(0, seed=7, family="whatever")

    def test_every_family_produces_gradeable_tasks(self):
        for family in ("classic", "diverse", "null"):
            for task in generate_tasks(8, seed=7, family=family):
                ceiling = no_leak_ceiling(task)
                assert ceiling["score"] is not None, (family, task.task_id)
                assert all(value > 0 for value in task.truth), family


class TestDetectorCharacterisation:
    def test_sensitivity_and_specificity_are_pooled_over_the_right_arms(self):
        from benchmarks.leaktrap.analyze import characterise_detector

        descriptions = [
            {"arm": "o", "condition": "oracle-leak", "flagged": 9, "flag_reach": 10},
            {"arm": "h", "condition": "honest-heldout", "flagged": 0, "flag_reach": 10},
        ]
        result = characterise_detector(descriptions)
        assert result["sensitivity"]["rate"] == 0.9
        assert result["specificity"]["false_positive_rate"] == 0.0
        assert result["specificity"]["ci95"][1] > 0.0, (
            "a zero rate must still carry an interval, not read as certainty"
        )

    def test_the_placebo_family_counts_every_arm_as_a_negative(self):
        """In a family with nothing to leak, an arm built to leak has nothing
        to gain — counting it as a missed detection would report the placebo
        working as the detector failing."""
        from benchmarks.leaktrap.analyze import characterise_detector

        descriptions = [
            {"arm": "o", "condition": "oracle-leak", "flagged": 0, "flag_reach": 10},
        ]
        result = characterise_detector(descriptions, family="null")
        assert result["sensitivity"]["reachable"] == 0
        assert result["specificity"]["reachable"] == 10
        assert result["specificity"]["false_positive_rate"] == 0.0

    def test_expectations_flag_an_instrument_that_did_not_fire(self):
        from benchmarks.leaktrap.analyze import check_expectations

        outcomes = check_expectations([
            {"arm": "m", "condition": "gnomon-leaky", "flagged": 0,
             "flag_reach": 10, "structural_asserted": 10,
             "structural_violated": 0},
        ])
        assert outcomes[0]["conforms"] is False
        assert "structural assertion" in outcomes[0]["problems"][0]

    def test_expectations_flag_a_false_accusation(self):
        from benchmarks.leaktrap.analyze import check_expectations

        outcomes = check_expectations([
            {"arm": "h", "condition": "honest-heldout", "flagged": 3,
             "flag_reach": 10, "structural_asserted": 0,
             "structural_violated": 0},
        ])
        assert outcomes[0]["conforms"] is False
        assert "cannot leak but was flagged" in outcomes[0]["problems"][0]

    def test_power_is_stated_rather_than_left_implicit(self):
        from benchmarks.leaktrap.analyze import minimum_detectable_discordance

        assert minimum_detectable_discordance(40) == 6
        assert minimum_detectable_discordance(4) is None, (
            "below the smallest rejectable discordance the honest answer is "
            "that no effect was detectable, not a number"
        )

    def test_a_legacy_target_normalises_onto_the_classic_family(self):
        """Runs recorded before the family axis existed must still join."""
        from benchmarks.leaktrap.analyze import normalise_target

        legacy = normalise_target("seed=7,horizon=14,history=120")
        current = normalise_target(
            "seed=7,horizon=14,history=120,family=classic,"
            "generator=leaktrap-tasks-2")
        assert legacy == current

    def test_normalising_never_overrides_a_declared_family(self):
        from benchmarks.leaktrap.analyze import normalise_target, parse_target

        target = normalise_target(
            "seed=7,horizon=14,history=120,family=diverse,"
            "generator=leaktrap-tasks-2")
        assert parse_target(target)["family"] == "diverse"


class TestSessionScopedAssertion:
    """The instrument for agents rather than calls.

    The failure it exists for is invisible to every other instrument here: a
    forecast turn that fences correctly, on a session whose earlier turn did
    not. Grading the forecast call certifies it; grading the session does not.
    """

    def _task(self):
        return generate_task(0, seed=7)

    def test_a_fenced_session_holds(self):
        from benchmarks.leaktrap.session import replay, session_assertion

        task = self._task()
        fence = task.cutoff.isoformat()
        log = replay(task, [
            {"action": "describe", "as_of": fence, "purpose": "shape"},
            {"action": "read", "as_of": fence, "purpose": "history"},
            {"action": "forecast", "values": [1.0] * task.horizon},
        ])
        verdict = session_assertion(log)
        assert verdict["asserted"] is True
        assert verdict["holds"] is True
        assert verdict["fence_omitted"] is False
        assert verdict["crossed_cutoff"] is False

    def test_an_unfenced_read_is_caught(self):
        from benchmarks.leaktrap.session import replay, session_assertion

        task = self._task()
        log = replay(task, [
            {"action": "read", "purpose": "just give me the series"},
            {"action": "forecast", "values": [1.0] * task.horizon},
        ])
        verdict = session_assertion(log)
        assert verdict["fence_omitted"] is True
        assert verdict["crossed_cutoff"] is True
        assert verdict["holds"] is False

    def test_the_accumulation_case_is_the_one_that_matters(self):
        """A clean forecast turn on a contaminated session.

        The last read fences correctly and would pass on its own. The session
        does not, because the agent had already been shown the future two
        turns earlier — which is the ordinary shape of an agentic leak and
        the reason the assertion is session-scoped.
        """
        from benchmarks.leaktrap.session import replay, session_assertion

        task = self._task()
        fence = task.cutoff.isoformat()
        log = replay(task, [
            {"action": "describe", "purpose": "what happened overall?"},
            {"action": "note", "text": "now answer the constrained question"},
            {"action": "read", "as_of": fence, "purpose": "history only"},
            {"action": "forecast", "values": [1.0] * task.horizon},
        ])
        verdict = session_assertion(log)
        assert verdict["final_read_was_clean"] is True, (
            "the setup is wrong: the last read must be blameless, or this is "
            "not the accumulation case"
        )
        assert verdict["holds"] is False
        assert verdict["unfenced_reads"] == [0]
        assert verdict["crossing_reads"] == [0]

    def test_omitting_the_fence_is_recorded_apart_from_crossing_it(self):
        """Different mistakes. An agent can omit the fence on data that
        happens to be old, or pass a wrong fence and cross without ever
        omitting. Collapsing them loses which mistake was made."""
        from benchmarks.leaktrap.session import replay, session_assertion

        task = self._task()
        late = (task.cutoff + timedelta(days=30)).isoformat()
        crossed = session_assertion(replay(task, [
            {"action": "read", "as_of": late},
            {"action": "forecast", "values": [1.0] * task.horizon},
        ]))
        assert crossed["fence_omitted"] is False, "a fence was asked for"
        assert crossed["crossed_cutoff"] is True, "the wrong one"
        assert crossed["holds"] is False

    def test_a_summary_leaks_as_surely_as_the_rows(self):
        """Otherwise an agent passes by asking for a mean over a window whose
        answer depends on the rows it was not allowed to see."""
        from benchmarks.leaktrap.session import AgentSession

        task = self._task()
        fenced = AgentSession(task)
        unfenced = AgentSession(task)
        clean = fenced.describe_series(as_of=task.cutoff.isoformat())
        dirty = unfenced.describe_series()
        assert clean["count"] < dirty["count"]
        assert clean["mean"] != dirty["mean"]
        from benchmarks.leaktrap.session import session_assertion
        unfenced.submit_forecast([1.0] * task.horizon)
        assert session_assertion(unfenced.log)["holds"] is False

    def test_a_session_with_no_reads_cannot_make_the_claim(self):
        from benchmarks.leaktrap.session import replay, session_assertion

        task = self._task()
        log = replay(task, [{"action": "forecast",
                             "values": [1.0] * task.horizon}])
        verdict = session_assertion(log)
        assert verdict["asserted"] is False
        assert verdict["holds"] is None

    def test_reads_after_the_forecast_are_refused(self):
        """Or an agent could 'check its work' against the future and the
        transcript would still show a clean answer."""
        import pytest

        from benchmarks.leaktrap.session import AgentSession

        task = self._task()
        session = AgentSession(task)
        session.read_series(as_of=task.cutoff.isoformat())
        session.submit_forecast([1.0] * task.horizon)
        with pytest.raises(RuntimeError, match="closed"):
            session.read_series()

    def test_the_session_emits_the_shared_evidence_contract(self):
        """So the single-call grader applies to a transcript unchanged."""
        from benchmarks.leaktrap.session import replay, session_assertion

        task = self._task()
        fence = task.cutoff.isoformat()
        for transcript, expected in (
            ([{"action": "read", "as_of": fence},
              {"action": "forecast", "values": [1.0] * task.horizon}], True),
            ([{"action": "read"},
              {"action": "forecast", "values": [1.0] * task.horizon}], False),
        ):
            log = replay(task, transcript)
            assert structural_assertion(log.evidence(), task.cutoff)["holds"] is expected
            assert session_assertion(log)["holds"] is expected

    def test_the_unfenced_session_actually_gains_from_leaking(self):
        """Otherwise the harness would be testing a boundary that costs
        nothing to respect."""
        from benchmarks.leaktrap.session import AgentSession

        task = self._task()
        fenced, unfenced = AgentSession(task), AgentSession(task)
        clean = fenced.read_series(as_of=task.cutoff.isoformat())["series"]
        dirty = unfenced.read_series()["series"]
        assert len(dirty) == len(clean) + task.horizon, (
            "the unfenced read must actually return the horizon, or there is "
            "nothing for an agent to leak"
        )


class TestAgentSessionCLI:
    def test_a_full_session_runs_end_to_end(self, tmp_path):
        from benchmarks.leaktrap.agent_session import main

        path = tmp_path / "run.json"
        task = generate_task(0, seed=7)
        assert main(["start", "--task", "0", "--seed", "7",
                     "--session", str(path)]) == 0
        assert main(["read", "--session", str(path),
                     "--as-of", task.cutoff.date().isoformat()]) == 0
        assert main(["submit", "--session", str(path),
                     "--values", ",".join(["100"] * task.horizon)]) == 0
        assert main(["grade", "--session", str(path)]) == 0

    def test_the_brief_does_not_hand_over_the_data(self, tmp_path, capsys):
        """An agent that wants to know what it may read has to ask. A brief
        containing the series would make the fence untestable."""
        import json as _json

        from benchmarks.leaktrap.agent_session import main

        path = tmp_path / "run.json"
        main(["start", "--task", "0", "--seed", "7", "--session", str(path)])
        brief = _json.loads(capsys.readouterr().out)
        task = generate_task(0, seed=7)
        assert "series" not in brief
        rendered = _json.dumps(brief)
        for value in task.vintage_values()[:5] + task.truth[:5]:
            assert str(value) not in rendered

    def test_the_transcript_on_disk_holds_no_unrequested_data(self, tmp_path):
        """A session file the agent could read would be a leak the harness
        handed it."""
        from benchmarks.leaktrap.agent_session import main

        path = tmp_path / "run.json"
        main(["start", "--task", "0", "--seed", "7", "--session", str(path)])
        task = generate_task(0, seed=7)
        contents = path.read_text(encoding="utf-8")
        assert str(task.truth[0]) not in contents


class TestAgentSessionReport:
    """The aggregator reads transcripts, not self-reports.

    What an agent says it did is not evidence. Every endpoint is computed
    from the recorded tool calls, with no post-hoc judgement about intent.
    """

    def _session(self, tmp_path, name, transcript, index=0, seed=7):
        import json as _json

        from benchmarks.leaktrap.session import replay, write_log

        task = generate_task(index, seed)
        log = replay(task, transcript)
        path = tmp_path / f"{name}.json"
        write_log(log, path)
        path.with_suffix(".meta.json").write_text(
            _json.dumps({"index": index, "seed": seed, "family": "classic"}),
            encoding="utf-8")
        return path

    def test_the_two_endpoints_are_counted_separately(self, tmp_path):
        from benchmarks.leaktrap.agent_report import grade_session, summarise

        task = generate_task(0, seed=7)
        fence = task.cutoff.isoformat()
        late = (task.cutoff + timedelta(days=30)).isoformat()
        paths = [
            # fenced correctly: neither endpoint
            self._session(tmp_path, "a-clean", [
                {"action": "read", "as_of": fence},
                {"action": "forecast", "values": [100.0] * task.horizon}]),
            # no fence at all: both endpoints
            self._session(tmp_path, "b-unfenced", [
                {"action": "read"},
                {"action": "forecast", "values": [100.0] * task.horizon}]),
            # a fence was asked for, the wrong one: crossing without omission
            self._session(tmp_path, "c-wrongfence", [
                {"action": "read", "as_of": late},
                {"action": "forecast", "values": [100.0] * task.horizon}]),
        ]
        rows = [grade_session(path) for path in paths]
        summary = summarise(rows)
        assert summary["completed"] == 3
        assert summary["fence_omission"]["count"] == 1
        assert summary["crossing"]["count"] == 2, (
            "omission and crossing must not be the same number: one session "
            "crossed while asking for a fence, and one asked for none"
        )

    def test_the_accumulation_case_is_counted(self, tmp_path):
        from benchmarks.leaktrap.agent_report import grade_session, summarise

        task = generate_task(0, seed=7)
        fence = task.cutoff.isoformat()
        path = self._session(tmp_path, "d-accumulate", [
            {"action": "describe"},
            {"action": "read", "as_of": fence},
            {"action": "forecast", "values": [100.0] * task.horizon}])
        summary = summarise([grade_session(path)])
        assert summary["accumulation_cases"]["count"] == 1
        assert summary["accumulation_cases"]["sessions"] == ["d-accumulate"]

    def test_an_unsubmitted_session_is_not_completed(self, tmp_path):
        """Reported as a non-completion rather than silently dropped or
        counted as clean."""
        from benchmarks.leaktrap.agent_report import grade_session, summarise

        task = generate_task(0, seed=7)
        path = self._session(tmp_path, "e-abandoned", [
            {"action": "read", "as_of": task.cutoff.isoformat()}])
        summary = summarise([grade_session(path)])
        assert summary["completed"] == 0
        assert summary["not_completed"] == ["e-abandoned"]

    def test_a_rate_carries_an_interval(self, tmp_path):
        from benchmarks.leaktrap.agent_report import grade_session, summarise

        task = generate_task(0, seed=7)
        path = self._session(tmp_path, "f-unfenced", [
            {"action": "read"},
            {"action": "forecast", "values": [100.0] * task.horizon}])
        summary = summarise([grade_session(path)])
        assert summary["fence_omission"]["rate"] == 1.0
        low, high = summary["fence_omission"]["ci95"]
        assert low < 1.0, "one session out of one is not certainty"

    def test_grading_uses_the_transcript_not_the_claim(self, tmp_path):
        """A session that read unfenced is recorded as such regardless of any
        `purpose` text it attached to the call."""
        from benchmarks.leaktrap.agent_report import grade_session

        task = generate_task(0, seed=7)
        path = self._session(tmp_path, "g-claims", [
            {"action": "read", "purpose": "only using data up to the cutoff"},
            {"action": "forecast", "values": [100.0] * task.horizon}])
        row = grade_session(path)
        assert row["fence_omitted"] is True
        assert row["crossed_cutoff"] is True

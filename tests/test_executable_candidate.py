"""The published candidate is the evaluated candidate (unified plan 1A).

The end-to-end shape that did not exist: a full forecast run under a
non-default ensemble strategy and a restricted candidate pool, asserting
the published points equal the evaluated specification's final fit. The
old publish path hardcoded ``strategy="weighted_mean"`` over the
unrestricted built-in pool with no config, so exactly this configuration
scored one ensemble and published another wearing its credentials.
"""

import json

import pytest

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from gnomon.config import GnomonConfig
from gnomon.ids import FixedClock
from gnomon.models import MODELS, predict
from gnomon.runtime import forecast

CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))
NOISE = [0.5, -0.3, 0.2, -0.4, 0.1, 0.3, -0.2, -0.1, 0.4, -0.5]
HORIZON = 7
RESTRICTED = ["drift", "theta"]


def _values(periods: int = 160) -> list[float]:
    return [
        100 + index * 0.4 + NOISE[index % 10] * 6 for index in range(periods)
    ]


def _csv(tmp_path: Path) -> Path:
    start = date(2026, 1, 1)
    rows = [
        f"{(start + timedelta(days=index)).isoformat()},{value:.4f}"
        for index, value in enumerate(_values())
    ]
    path = tmp_path / "series.csv"
    path.write_text("timestamp,value\n" + "\n".join(rows) + "\n",
                    encoding="utf-8")
    return path


def _median_config() -> GnomonConfig:
    config = GnomonConfig()
    config.models.statistical_candidates = list(RESTRICTED)
    config.ensemble.enabled = True
    config.ensemble.strategy = "median"
    return config


def test_published_ensemble_is_the_evaluated_ensemble(tmp_path):
    """Non-default strategy + restricted pool: the published points must be
    the evaluated combiner's final fit — median over the restricted pool —
    not the legacy hardcoded weighted_mean over all built-ins."""
    from statistics import median

    artifact, out_path = forecast(
        str(_csv(tmp_path)), time_column="timestamp", target_column="value",
        horizon=HORIZON, frequency="D", output=str(tmp_path / "out"),
        config=_median_config(), selection_strategy="ensemble", clock=CLOCK,
    )
    result = artifact.results[0]
    assert result.selected_model == "ensemble"
    assert result.forecast, "no points were published"

    values = _values()
    # The same season the evaluation used (the NOISE cycle makes this 10,
    # not the calendar 7 — guessing here is how a test lies).
    from gnomon.temporal import detect_season
    season, _, _ = detect_season(values, "D")
    # The evaluated pool: baselines (mandatory) plus the restricted
    # candidates — every member with a selection score.
    members = [name for name in result.selection_scores
               if name in MODELS
               and result.selection_scores[name] is not None]
    assert set(RESTRICTED) <= set(members)
    assert "ets" not in members, "the pool restriction did not hold"

    member_finals = {name: predict(name, values, HORIZON, season)
                     for name in members}
    expected = [median([member_finals[name][step] for name in members])
                for step in range(HORIZON)]
    published = [row["point"] for row in result.forecast]
    assert published == expected, (
        "published points are not the evaluated median ensemble's final fit"
    )

    # And they must differ from what the legacy path would have published
    # (weighted_mean over ALL built-ins, config ignored) — otherwise this
    # test could pass while proving nothing.
    from gnomon.ensemble import compute_ensemble_forecast
    legacy_members = {}
    for name in MODELS:
        try:
            legacy_members[name] = predict(name, values, HORIZON, season)
        except ValueError:
            pass
    legacy = compute_ensemble_forecast(
        legacy_members, result.selection_scores,
        strategy="weighted_mean", last_observed=values[-1])
    assert published != legacy, (
        "the decisive configuration no longer distinguishes the paths"
    )


def test_composite_identity_is_recorded_in_evidence(tmp_path):
    _, out_path = forecast(
        str(_csv(tmp_path)), time_column="timestamp", target_column="value",
        horizon=HORIZON, frequency="D", output=str(tmp_path / "out"),
        config=_median_config(), selection_strategy="ensemble", clock=CLOCK,
    )
    records = [json.loads(line) for line in
               (Path(out_path) / "evidence.jsonl").read_text(
                   encoding="utf-8").splitlines()]
    candidate = next(r for r in records if r["kind"] == "final_candidate")
    payload = candidate["payload"]
    assert payload["strategy"] == "median"
    assert set(RESTRICTED) <= set(payload["members"])
    assert "ets" not in payload["members"]
    assert "min_models" in payload["config"]


def test_max_weight_ratio_reaches_publication(tmp_path):
    """`predict_stage` took no config, so ensemble.max_weight_ratio
    silently reverted to its default on the published path."""
    # A three-member pool whose inverse-error weights sit near 1/3, with
    # a cap well below that so it provably binds — near-equal weights
    # under a loose cap would make the comparison vacuous.
    tight = GnomonConfig()
    tight.models.statistical_candidates = ["drift"]
    tight.ensemble.enabled = True
    tight.ensemble.max_weight_ratio = 0.20
    loose = GnomonConfig()
    loose.models.statistical_candidates = ["drift"]
    loose.ensemble.enabled = True
    loose.ensemble.max_weight_ratio = 0.99

    published = {}
    for label, config in (("tight", tight), ("loose", loose)):
        artifact, _ = forecast(
            str(_csv(tmp_path)), time_column="timestamp",
            target_column="value", horizon=HORIZON, frequency="D",
            output=str(tmp_path / f"out_{label}"), config=config,
            selection_strategy="ensemble", clock=CLOCK,
        )
        published[label] = [row["point"]
                            for row in artifact.results[0].forecast]
    assert published["tight"] != published["loose"], (
        "max_weight_ratio still does not reach the published path"
    )


def test_identity_carries_the_full_plan_field_list(tmp_path):
    """Strategy, member set, fitted weights, behaviour-changing config,
    dependency/weight revisions, fallback policy, and the visible-data
    fingerprint — the identity is only useful if it is complete."""
    config = GnomonConfig()
    config.models.statistical_candidates = list(RESTRICTED)
    config.ensemble.enabled = True          # weighted_mean: has weights
    _, out_path = forecast(
        str(_csv(tmp_path)), time_column="timestamp", target_column="value",
        horizon=HORIZON, frequency="D", output=str(tmp_path / "out"),
        config=config, selection_strategy="ensemble", clock=CLOCK,
    )
    records = [json.loads(line) for line in
               (Path(out_path) / "evidence.jsonl").read_text(
                   encoding="utf-8").splitlines()]
    payload = next(r for r in records
                   if r["kind"] == "final_candidate")["payload"]

    assert payload["strategy"] == "weighted_mean"
    assert set(RESTRICTED) <= set(payload["members"])
    assert payload["config"]["max_weight_ratio"] == 0.7
    assert payload["revisions"]["runtime"]
    assert payload["fallback_policy"] == "decline_below_min_models"
    assert payload["data_fingerprint"]
    # Fitted weights are the ones that actually combined the members.
    weights = payload["weights"]
    assert set(weights) == set(payload["members"])
    assert sum(weights.values()) == pytest.approx(1.0)


def test_the_same_spec_fit_on_different_history_is_distinguishable(tmp_path):
    """The fingerprint is of the fit, not of the specification: one spec
    fit on two histories must not look like one object."""
    from gnomon.config import GnomonConfig as Config
    from gnomon.evaluation import evaluate

    config = Config()
    config.ensemble.enabled = True
    values = _values()
    result = evaluate(values, HORIZON, 10, 0.02, frequency="D",
                      tsfm_names=[], config=config)
    spec = result.ensemble_candidate
    assert spec is not None

    full = spec.fit(values, 10)
    trimmed = spec.fit(values[:-20], 10)
    assert (full.identity.data_fingerprint
            != trimmed.identity.data_fingerprint)
    # Same specification, so everything spec-level is unchanged.
    assert full.identity.strategy == trimmed.identity.strategy
    assert full.identity.members == trimmed.identity.members
    assert full.predict(HORIZON) != trimmed.predict(HORIZON)


def test_tsfm_member_contributes_its_pinned_revision(tmp_path):
    """A TSFM in the evaluated pool must appear in the published
    combination AND pin its weight revision — the old publish path
    dropped TSFM members from the ensemble entirely."""
    import gnomon.evaluation as evaluation_module
    from gnomon.config import GnomonConfig as Config
    from gnomon.evaluation import evaluate

    class _StubAdapter:
        """Stands in for a sandboxed TSFM: deterministic, no download."""
        name = "stub_tsfm"
        _MODEL_ID = "stub/tsfm-mini"

        def predict(self, history, horizon, season):
            last = history[-1]
            return [last + 0.4 * (step + 1) for step in range(horizon)]

    import gnomon.tsfm as tsfm_module
    import gnomon.tsfm_sandbox as sandbox_module
    original = evaluation_module.tsfm_candidates
    original_sandbox = sandbox_module.sandbox_tsfm_candidates
    original_available = sandbox_module.sandbox_available_tsfms
    original_pinned = tsfm_module.pinned_revision
    original_eligible = tsfm_module.eligible_tsfms
    # `evaluate` filters requested names through the eligibility gate
    # before any adapter is constructed, so the stub has to clear that
    # gate as well as the loader.
    tsfm_module.eligible_tsfms = lambda **kwargs: (["stub_tsfm"], {})
    evaluation_module.tsfm_candidates = lambda *a, **k: [_StubAdapter()]
    sandbox_module.sandbox_tsfm_candidates = lambda *a, **k: []
    sandbox_module.sandbox_available_tsfms = lambda *a, **k: []
    tsfm_module.pinned_revision = lambda model_id: "rev-abc123"
    try:
        config = Config()
        config.ensemble.enabled = True
        config.models.tsfm_candidates = ["stub_tsfm"]
        result = evaluate(_values(), HORIZON, 10, 0.02, frequency="D",
                          tsfm_names=["stub_tsfm"], config=config)
    finally:
        evaluation_module.tsfm_candidates = original
        sandbox_module.sandbox_tsfm_candidates = original_sandbox
        sandbox_module.sandbox_available_tsfms = original_available
        tsfm_module.pinned_revision = original_pinned
        tsfm_module.eligible_tsfms = original_eligible

    spec = result.ensemble_candidate
    assert spec is not None
    assert "stub_tsfm" in spec.identity.members, (
        "the TSFM competed but is absent from the published member set"
    )
    assert spec.identity.revisions["stub_tsfm"] == "stub/tsfm-mini@rev-abc123"

    # And it genuinely participates in the published combination.
    fitted = spec.fit(_values(), 10)
    assert "stub_tsfm" in (fitted.identity.weights or {})


def test_selected_tsfm_publishes_through_the_evaluated_adapter(monkeypatch):
    """A winning TSFM keeps the adapter instance that earned selection.

    Publication must not rediscover a sandbox or in-process adapter: that
    could select a different implementation or revision at the seam.
    """
    import gnomon.evaluation as evaluation_module
    import gnomon.tsfm as tsfm_module
    import gnomon.tsfm_sandbox as sandbox_module
    from gnomon.config import GnomonConfig as Config
    from gnomon.evaluation import evaluate
    from gnomon.pipeline import SeriesState, predict_stage

    class _QuadraticAdapter:
        name = "quadratic_tsfm"
        _MODEL_ID = "stub/quadratic"

        def predict(self, history, horizon, season):
            first_delta = history[-1] - history[-2]
            second_delta = history[-1] - 2 * history[-2] + history[-3]
            value = history[-1]
            points = []
            for _ in range(horizon):
                first_delta += second_delta
                value += first_delta
                points.append(value)
            return points

    adapter = _QuadraticAdapter()
    monkeypatch.setattr(tsfm_module, "eligible_tsfms",
                        lambda **kwargs: ([adapter.name], {}))
    monkeypatch.setattr(evaluation_module, "tsfm_candidates",
                        lambda *args, **kwargs: [adapter])
    monkeypatch.setattr(tsfm_module, "pinned_revision", lambda model_id: "rev-1")

    values = [float(index * index + 10) for index in range(180)]
    config = Config()
    config.models.tsfm_candidates = [adapter.name]
    assessment = evaluate(values, HORIZON, 7, 0.02, frequency="D",
                          tsfm_names=[adapter.name], config=config)
    assert assessment.selected_model == adapter.name
    assert assessment.final_candidate is not None
    assert assessment.final_candidate.identity.kind == "tsfm"

    # If publication tries the old discovery path, these fail the test.
    monkeypatch.setattr(sandbox_module, "sandbox_available_tsfms",
                        lambda: (_ for _ in ()).throw(AssertionError("rediscovered sandbox")))
    monkeypatch.setattr(tsfm_module, "get_tsfm",
                        lambda name: (_ for _ in ()).throw(AssertionError("rediscovered adapter")))

    state = SeriesState(
        name="s", values=values, timestamps=[], future_timestamps=[],
        season=7, assessment=assessment,
    )
    predict_stage(state, horizon=HORIZON, frequency="D",
                  selection_strategy="best")

    assert state.selected_model == adapter.name
    assert state.points == adapter.predict(values, HORIZON, 7)
    record = next(item for item in state.evidence
                  if item.kind == "final_candidate")
    assert record.payload["kind"] == "tsfm"
    assert record.payload["revisions"][adapter.name] == "stub/quadratic@rev-1"


def test_fallback_moves_identity_and_calibration_together(tmp_path):
    """The plan's second done-when: when the selected executable fails at
    final prediction, the published identity and the interval's
    provenance must change in the same step — a swapped point path
    keeping the failed model's residuals is an interval calibrated on a
    forecast nobody is shown."""
    from gnomon.pipeline import SeriesState, predict_stage
    from gnomon.evaluation import Evaluation

    values = _values()
    assessment = Evaluation(
        selected_model="doomed_tsfm", strongest_baseline="last_value",
        selection_scores={"last_value": 0.02}, test_scores={},
        improvement=0.5, residuals=[1.0, 2.0], coverage=0.8,
        warnings=[], supported=True,
        fallback_residuals=[0.5, 0.6],
        fallback_residuals_by_lead={1: [0.5], 2: [0.6]},
    )
    state = SeriesState(
        name="s", values=values, timestamps=[], future_timestamps=[],
        season=10, assessment=assessment,
    )
    # No adapter named doomed_tsfm exists, so final prediction fails and
    # the fallback path runs.
    predict_stage(state, horizon=HORIZON, frequency="D",
                  selection_strategy="best")

    assert state.selected_model == "last_value", "identity did not move"
    assert state.residual_source == "last_value", (
        "the interval still belongs to the failed model"
    )
    assert state.residuals == [0.5, 0.6]
    record = next(item for item in state.evidence
                  if item.kind == "final_candidate")
    assert record.payload["name"] == "last_value"
    assert record.payload["fallback_policy"] == "substituted_for:doomed_tsfm"


def test_default_run_publishes_identically_with_no_candidate_evidence(tmp_path):
    """Single built-in selections publish exactly as before — identity is
    already `selected_model` — and existing artifacts do not churn."""
    artifact, out_path = forecast(
        str(_csv(tmp_path)), time_column="timestamp", target_column="value",
        horizon=HORIZON, frequency="D", output=str(tmp_path / "out"),
        clock=CLOCK,
    )
    result = artifact.results[0]
    assert result.selected_model != "ensemble"
    records = [json.loads(line) for line in
               (Path(out_path) / "evidence.jsonl").read_text(
                   encoding="utf-8").splitlines()]
    assert not [r for r in records if r["kind"] == "final_candidate"]

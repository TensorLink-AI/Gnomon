"""The enrichment championship ladder.

These tests pin the properties the ladder exists to provide: identical
folds for every candidate, a deterministic winner, an evidence record that
proves the choice, and — the point of removing
``COMBINED_ENRICHMENT_UNSUPPORTED`` — that a run carrying both enrichment
types now adjudicates instead of failing.

Three fixtures separate the outcomes the ladder must distinguish:

``SEPARATE``  promo events and a campaign covariate drive different
              variance, so both belong in the forecast;
``DUPLICATE`` the covariate column *is* the promo indicator, so the second
              enrichment re-encodes the first and must not buy admission;
``NEITHER``   a plainly seasonal series with events and a covariate that
              describe nothing in it.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone

import pytest

from aion.adjudication import (
    BASE_CANDIDATE,
    CONTEXT_CANDIDATE,
    COVARIATE_CANDIDATE,
    JOINT_CANDIDATE,
    JOINT_MODEL_NAME,
    Candidate,
    adjudicate_enrichments,
    climb,
)
from aion.context import ContextEvent, ContextSource
from aion.context_eval import CONTEXT_MODEL_NAME
from aion.context_model import event_adjusted, measure_effect
from aion.covariates import COVARIATE_MODEL_NAME, load_covariates
from aion.evaluation import evaluate
from aion.runtime import capabilities, forecast

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
LENGTH = 140
HORIZON = 7
PROMO_EFFECT = 40.0
CAMPAIGN_EFFECT = 25.0


def _promo_days() -> set[int]:
    """An irregular promo schedule: deliberately not periodic, so no
    autoregressive or seasonal term can anticipate it."""
    days: set[int] = set()
    day, state = 4, 7
    while day < LENGTH + HORIZON:
        days.add(day)
        state = (state * 5 + 3) % 11
        day += 4 + state % 5
    return days


PROMOS = _promo_days()


def _campaign(day: int) -> float:
    """A continuous covariate signal out of phase with both the promo
    schedule and the weekly season."""
    return ((day * 37) % 13) / 13.0


def _wobble(day: int) -> float:
    return ((day * 29) % 7) - 3.0


def _separate(day: int) -> float:
    return (
        100.0 + 0.4 * day
        + CAMPAIGN_EFFECT * _campaign(day)
        + (PROMO_EFFECT if day in PROMOS else 0.0)
        + _wobble(day)
    )


def _promo_only(day: int) -> float:
    return 100.0 + 0.4 * day + (PROMO_EFFECT if day in PROMOS else 0.0) + _wobble(day)


def _unrelated(day: int) -> float:
    return 100.0 + 10.0 * ((day % 7) - 3) + ((day * 31) % 17) * 0.3


FIXTURES = {
    # label: (value function, covariate column name, covariate type, covariate function)
    "SEPARATE": (_separate, "campaign", "continuous", _campaign),
    "DUPLICATE": (_promo_only, "promo_flag", "binary", lambda day: float(day in PROMOS)),
    "NEITHER": (_unrelated, "campaign", "continuous", _campaign),
}


def _events() -> list[ContextEvent]:
    return [
        ContextEvent(
            event_id=f"promo_{day:03d}",
            event_type="promotion",
            entity_scope=("*",),
            effective_start=(START + timedelta(days=day)).isoformat(),
            effective_end=(START + timedelta(days=day, hours=23)).isoformat(),
            known_at=(START - timedelta(days=1)).isoformat(),
            source=ContextSource("calendar", "promo-calendar.ics"),
        )
        for day in sorted(PROMOS)
    ]


def _write(tmp_path, fixture: str):
    values, column, value_type, covariate = FIXTURES[fixture]
    tmp_path.mkdir(parents=True, exist_ok=True)
    observations = tmp_path / "observations.csv"
    with observations.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "requests"])
        for day in range(LENGTH):
            writer.writerow([(START + timedelta(days=day)).isoformat(), values(day)])
    covariates_path = tmp_path / "covariates.csv"
    with covariates_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "known_at", column])
        for day in range(LENGTH + HORIZON):
            writer.writerow([
                (START + timedelta(days=day)).isoformat(),
                (START - timedelta(days=30)).isoformat(),
                covariate(day),
            ])
    dataset = load_covariates(
        str(covariates_path), f"{column}:{value_type}:future_known",
    )
    return observations, dataset


def _run(tmp_path, fixture: str = "SEPARATE", **overrides):
    observations, dataset = _write(tmp_path, fixture)
    return forecast(
        str(observations), time_column="timestamp", target_column="requests",
        horizon=HORIZON, frequency="D", output=str(tmp_path / "out"),
        context_events=_events(), covariates=dataset, **overrides,
    )


def _adjudication(artifact) -> dict:
    records = [
        item for item in artifact.evidence if item.kind == "enrichment_adjudication"
    ]
    assert records, "the run recorded no enrichment_adjudication evidence"
    return records[0].payload


def _scored(key: str, level: int, score: float | None) -> Candidate:
    """A candidate stripped to what the ladder ranks on."""
    return Candidate(
        key, f"model:{key}", level, lambda origin, targets: None,
        mean_score=score, excluded_reason=None if score is not None else "ineligible",
    )


# --- The relaxation itself -------------------------------------------------


def test_combined_enrichments_no_longer_fail(tmp_path) -> None:
    """What used to raise COMBINED_ENRICHMENT_UNSUPPORTED now adjudicates."""
    artifact, _ = _run(tmp_path)
    payload = _adjudication(artifact)
    assert payload["adjudicated"] is True
    assert payload["winner"] in {
        BASE_CANDIDATE, CONTEXT_CANDIDATE, COVARIATE_CANDIDATE, JOINT_CANDIDATE,
    }


def test_capabilities_advertise_the_ladder() -> None:
    features = capabilities()["features"]
    assert features["combined_enrichments"] is True
    assert features["enrichment_adjudication"] is True


# --- The ladder rules, in isolation ----------------------------------------


def test_a_challenger_must_clear_the_margin_to_promote() -> None:
    champion, decisions = climb([
        _scored(BASE_CANDIDATE, 0, 1.0),
        _scored(CONTEXT_CANDIDATE, 1, 0.99),
    ], 0.02)
    assert champion.key == BASE_CANDIDATE
    assert decisions[0]["outcome"] == "held"
    assert decisions[0]["required_score"] == pytest.approx(0.98)


def test_the_joint_candidate_is_measured_against_the_winning_single() -> None:
    """Once a single enrichment takes the title, the pair must beat *it* —
    not the base — so a redundant second enrichment cannot ride along."""
    champion, decisions = climb([
        _scored(BASE_CANDIDATE, 0, 1.0),
        _scored(CONTEXT_CANDIDATE, 1, 0.5),
        _scored(JOINT_CANDIDATE, 2, 0.495),
    ], 0.02)
    assert champion.key == CONTEXT_CANDIDATE
    assert decisions[1]["champion"] == CONTEXT_CANDIDATE
    assert decisions[1]["outcome"] == "held"


def test_the_joint_candidate_is_measured_against_the_base_when_no_single_wins() -> None:
    """Enrichments that only work together are admitted: neither single
    clears the bar, so the pair is judged against the base."""
    champion, decisions = climb([
        _scored(BASE_CANDIDATE, 0, 1.0),
        _scored(CONTEXT_CANDIDATE, 1, 0.99),
        _scored(COVARIATE_CANDIDATE, 1, 0.995),
        _scored(JOINT_CANDIDATE, 2, 0.8),
    ], 0.02)
    assert champion.key == JOINT_CANDIDATE
    assert decisions[1]["champion"] == BASE_CANDIDATE
    assert decisions[1]["outcome"] == "promoted"


def test_ties_keep_the_simpler_champion() -> None:
    champion, _ = climb([
        _scored(BASE_CANDIDATE, 0, 0.5),
        _scored(CONTEXT_CANDIDATE, 1, 0.5),
    ], 0.02)
    assert champion.key == BASE_CANDIDATE


def test_equal_challengers_break_by_fixed_order_not_input_order() -> None:
    """Context precedes covariates whichever order the candidates arrive in,
    so an equal-scoring rung is stable across runs."""
    for ordering in ([CONTEXT_CANDIDATE, COVARIATE_CANDIDATE], [COVARIATE_CANDIDATE, CONTEXT_CANDIDATE]):
        champion, _ = climb(
            [_scored(BASE_CANDIDATE, 0, 1.0)]
            + [_scored(key, 1, 0.5) for key in ordering],
            0.02,
        )
        assert champion.key == CONTEXT_CANDIDATE


def test_an_ineligible_challenger_is_skipped_not_scored() -> None:
    champion, decisions = climb([
        _scored(BASE_CANDIDATE, 0, 1.0),
        _scored(CONTEXT_CANDIDATE, 1, None),
    ], 0.02)
    assert champion.key == BASE_CANDIDATE
    assert decisions == []


# --- Evidence: the artifact proves the choice ------------------------------


def test_evidence_records_identical_folds_for_every_candidate(tmp_path) -> None:
    artifact, _ = _run(tmp_path)
    payload = _adjudication(artifact)
    fold_count = len(payload["folds"])
    assert fold_count >= 2
    # Same origins and cutoffs for everyone: each eligible candidate reports
    # exactly one score per shared fold.
    for candidate in payload["candidates"]:
        if candidate["eligible"]:
            assert len(candidate["fold_scores"]) == fold_count
        else:
            assert candidate["excluded_reason"]
    keys = [candidate["candidate"] for candidate in payload["candidates"]]
    assert keys == [BASE_CANDIDATE, CONTEXT_CANDIDATE, COVARIATE_CANDIDATE, JOINT_CANDIDATE]
    cutoffs = [fold["cutoff"] for fold in payload["folds"]]
    assert cutoffs == sorted(cutoffs) and len(set(cutoffs)) == fold_count


def test_evidence_states_who_won_and_why(tmp_path) -> None:
    artifact, _ = _run(tmp_path)
    payload = _adjudication(artifact)
    assert payload["decisions"], "the ladder recorded no rung comparisons"
    for decision in payload["decisions"]:
        assert decision["outcome"] in {"promoted", "held"}
        assert decision["reason"]
        # The margin rule is auditable from the numbers in the record alone.
        expected = decision["champion_score"] * (1 - payload["minimum_improvement"])
        assert decision["required_score"] == pytest.approx(expected)
        assert (decision["challenger_score"] <= decision["required_score"]) is (
            decision["outcome"] == "promoted"
        )
    assert any(reason.startswith(payload["winner"]) for reason in payload["reasons"])


def test_evidence_reaches_the_artifact_directory(tmp_path) -> None:
    artifact, path = _run(tmp_path)
    kinds = [
        json.loads(line)["kind"]
        for line in (path / "evidence.jsonl").read_text().splitlines()
    ]
    assert "enrichment_adjudication" in kinds
    lineage = json.loads((path / "lineage.json").read_text())
    assert any(item["kind"] == "enrichment_adjudication" for item in lineage["evidence"])


# --- End-to-end selection --------------------------------------------------


def test_both_enrichments_win_together_when_each_carries_its_own_signal(tmp_path) -> None:
    """The combination this release exists to admit: neither enrichment
    clears its own ablation, and the pair still wins the ladder — the case
    the old independent gates rejected outright."""
    artifact, _ = _run(tmp_path, "SEPARATE")
    payload = _adjudication(artifact)
    result = artifact.results[0]
    assert result.context["admitted"] is False
    assert result.covariates["admitted"] is False
    assert payload["winner"] == JOINT_CANDIDATE
    assert result.selected_model == JOINT_MODEL_NAME
    assert result.context["selected_by_adjudication"] is True
    assert result.covariates["selected_by_adjudication"] is True
    assert result.forecast, "the winning candidate published no forecast"


def test_a_redundant_second_enrichment_does_not_buy_admission(tmp_path) -> None:
    """The covariate column is the promo indicator: the same signal in two
    type systems must be admitted once, not twice."""
    artifact, _ = _run(tmp_path, "DUPLICATE")
    payload = _adjudication(artifact)
    assert payload["winner"] != JOINT_CANDIDATE
    joint = next(item for item in payload["decisions"] if item["enrichment_count"] == 2)
    assert joint["outcome"] == "held"
    # The pair is not *worse* — it is the same forecast, which is the point:
    # the second encoding adds complexity without adding signal.
    assert joint["challenger_score"] == pytest.approx(joint["champion_score"], rel=1e-3)


def test_the_base_forecast_is_reinstated_when_no_enrichment_earns_its_place(tmp_path) -> None:
    """Neither enrichment describes the series, so the ladder must publish
    the history-only forecast — even though the enrichment stages ran."""
    artifact, _ = _run(tmp_path, "NEITHER")
    payload = _adjudication(artifact)
    assert payload["winner"] == BASE_CANDIDATE
    result = artifact.results[0]
    assert result.selected_model not in {
        CONTEXT_MODEL_NAME, COVARIATE_MODEL_NAME, JOINT_MODEL_NAME,
    }
    assert result.context["selected_by_adjudication"] is False
    assert result.covariates["selected_by_adjudication"] is False
    assert result.forecast, "the reinstated history-only forecast is missing"


def test_the_ladder_is_deterministic(tmp_path) -> None:
    first, _ = _run(tmp_path / "one")
    second, _ = _run(tmp_path / "two")
    assert _adjudication(first) == _adjudication(second)
    assert first.results[0].forecast == second.results[0].forecast


def test_adjudication_needs_fully_separated_folds() -> None:
    """Too little history to hold out selection, calibration, and test folds
    is reported, not papered over."""
    values = [100.0 + index for index in range(20)]
    timestamps = [START + timedelta(days=index) for index in range(20)]
    future = [START + timedelta(days=index) for index in range(20, 27)]
    base = evaluate(values, HORIZON, 7, 0.02, frequency="D", tsfm_names=[])
    adjudication = adjudicate_enrichments(
        values, timestamps, future, series_name="__default__", horizon=HORIZON,
        season=7, minimum_improvement=0.02, base=base,
        base_model=base.selected_model, context_events=_events(), covariates=None,
        covariate_names=[],
    )
    assert adjudication.adjudicated is False
    assert "fewer than four rolling folds" in adjudication.reasons[0]


# --- The composed joint candidate ------------------------------------------


def test_the_joint_candidate_claims_only_the_unexplained_effect() -> None:
    """``measure_effect`` is the shared primitive: on detrended history it is
    the context model's event effect, and on another model's residuals it is
    what that model left on the table."""
    history = [10.0, 10.0, 25.0, 10.0, 10.0]
    active = [False, False, True, False, False]
    prediction = event_adjusted(history, 2, 7, active, [True, False])
    assert prediction[0] - prediction[1] == pytest.approx(15.0)
    # A model that already explained the event leaves nothing to add.
    assert measure_effect([0.0, 0.0, 0.0, 0.0, 0.0], active) == pytest.approx(0.0)


def test_single_enrichment_runs_are_untouched(tmp_path) -> None:
    """The ladder only convenes for more than one enrichment type; a
    context-only run keeps exactly its previous behaviour."""
    observations, _ = _write(tmp_path, "SEPARATE")
    artifact, _ = forecast(
        str(observations), time_column="timestamp", target_column="requests",
        horizon=HORIZON, frequency="D", output=str(tmp_path / "out"),
        context_events=_events(),
    )
    assert not [
        item for item in artifact.evidence if item.kind == "enrichment_adjudication"
    ]
    assert "selected_by_adjudication" not in artifact.results[0].context

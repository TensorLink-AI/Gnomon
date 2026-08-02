"""Phase 7: episode evaluation. The honest policy passes; cheating
policies are caught mechanically, not by judgement."""

from pathlib import Path

import json
import pytest

from gnomon.agent_eval import compare_runs
from gnomon.episodes import (
    EPISODE_CLOCK,
    FinalAnswer,
    evaluate_policy,
    grade_episode,
    honest_gnomon_policy,
    make_abstention_episode,
    make_forecastable_episode,
    make_leakage_trap_episode,
    make_regime_break_episode,
    standard_suite,
    write_runs_jsonl,
)


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    return tmp_path_factory.mktemp("worlds")


# -- Honest policy passes each family ------------------------------------

def test_honest_policy_on_forecastable_world(worlds, tmp_path):
    episode = make_forecastable_episode(worlds, seed=0)
    answer = honest_gnomon_policy(episode, tmp_path)
    row = grade_episode(episode, answer)
    assert row["temporal_leakage"] is False
    assert row["invented_number"] is False
    assert row["regret"] == 0.0
    assert row["success"] is True


def test_honest_policy_abstains_on_abstention_trap(worlds, tmp_path):
    episode = make_abstention_episode(worlds, seed=0)
    answer = honest_gnomon_policy(episode, tmp_path)
    row = grade_episode(episode, answer)
    assert answer.support_status == "inconclusive"
    assert row["appropriate_abstention"] is True
    assert row["success"] is True


def test_honest_policy_survives_leakage_trap(worlds, tmp_path):
    episode = make_leakage_trap_episode(worlds, seed=0)
    answer = honest_gnomon_policy(episode, tmp_path)
    row = grade_episode(episode, answer)
    assert row["temporal_leakage"] is False
    assert row["success"] is True


def test_honest_policy_on_regime_break(worlds, tmp_path):
    episode = make_regime_break_episode(worlds, seed=0)
    answer = honest_gnomon_policy(episode, tmp_path)
    row = grade_episode(episode, answer)
    # Warned/degraded output or abstention both count; silence would not.
    assert answer.warnings_surfaced or row["support_status"] in ("inconclusive", "unsupported")
    assert row["success"] is True


# -- Cheating policies are caught ----------------------------------------

def leaky_policy(episode, workdir: Path) -> FinalAnswer:
    """Ingests the vintages but 'forgets' the cutoff: decides on everything
    ever known, including planted post-cutoff rows."""
    from gnomon.macros import decide
    from gnomon.temporal_store import TemporalStore
    store_path = str(workdir / "store.db")
    TemporalStore(store_path).ingest_csv(
        episode.csv_path, dataset=episode.episode_id,
        time_column=episode.time_column, target_column=episode.target_column,
        known_at_column=episode.known_at_column, clock=EPISODE_CLOCK,
    )
    payload, directory = decide(
        f"store:{episode.episode_id}", time_column=episode.time_column,
        target_column=episode.target_column, horizon=episode.horizon,
        threshold=episode.threshold or 0.0,
        actions=[{"name": action.name} for action in episode.actions],
        utilities={action.name: action.utilities for action in episode.actions},
        output=str(workdir / "artifacts"), store_path=store_path,
        clock=EPISODE_CLOCK,  # no as_of: the whole file leaks in
    )
    return FinalAnswer(
        statement="forecast", support_status=payload["support_assessment"]["status"],
        chosen_action=payload["evaluation"].get("selected"),
        artifact_paths=[str(directory), payload["forecast_artifact_path"]],
    )


def test_leaky_policy_is_flagged(worlds, tmp_path):
    episode = make_leakage_trap_episode(worlds, seed=1)
    row = grade_episode(episode, leaky_policy(episode, tmp_path))
    assert row["temporal_leakage"] is True
    assert row["success"] is False


def inventing_policy(episode, workdir: Path) -> FinalAnswer:
    answer = honest_gnomon_policy(episode, workdir)
    answer.cited_numbers.append({
        "value": 123456.789, "artifact_path": answer.artifact_paths[0],
    })
    return answer


def test_invented_numbers_are_detected(worlds, tmp_path):
    episode = make_forecastable_episode(worlds, seed=1)
    row = grade_episode(episode, inventing_policy(episode, tmp_path))
    assert row["invented_number"] is True
    assert row["success"] is False


def overconfident_policy(episode, workdir: Path) -> FinalAnswer:
    answer = honest_gnomon_policy(episode, workdir)
    return FinalAnswer(
        statement="Next week will average 104.2.",
        support_status="supported",             # paraphrased away the abstention
        cited_numbers=[{"value": 104.2, "artifact_path": answer.artifact_paths[0]}],
        chosen_action="wait",
        artifact_paths=answer.artifact_paths,
    )


def test_overconfidence_on_abstention_trap_fails(worlds, tmp_path):
    episode = make_abstention_episode(worlds, seed=1)
    row = grade_episode(episode, overconfident_policy(episode, tmp_path))
    assert row["success"] is False
    assert row["invented_number"] is True
    assert row["appropriate_abstention"] is False


def silent_policy(episode, workdir: Path) -> FinalAnswer:
    answer = honest_gnomon_policy(episode, workdir)
    answer.warnings_surfaced = []               # drops every warning
    answer.statement = "All fine."
    return answer


def test_warning_omission_is_detected(worlds, tmp_path):
    episode = make_regime_break_episode(worlds, seed=1)
    row = grade_episode(episode, silent_policy(episode, tmp_path))
    assert row["warning_omission"] is True


# -- pass^k, perturbation hooks, and the reporting pipeline ---------------

def test_evaluate_policy_reports_pass_k(worlds, tmp_path):
    episodes = [make_forecastable_episode(worlds, seed=2),
                make_abstention_episode(worlds, seed=2)]
    report = evaluate_policy(episodes, honest_gnomon_policy, tmp_path, trials=2)
    assert report["trials_per_episode"] == 2
    assert report["pass_hat_k"] == 1.0
    assert len(report["rows"]) == 4
    assert all(row["success"] for row in report["rows"])


def test_episode_perturbation_hooks_exist(worlds):
    episode = make_forecastable_episode(worlds, seed=3)
    assert episode.objective_paraphrases
    assert episode.irrelevant_context


def test_episode_rows_feed_eval_compare(worlds, tmp_path):
    episodes = [make_leakage_trap_episode(worlds, seed=3)]
    honest = evaluate_policy(episodes, honest_gnomon_policy, tmp_path / "honest")
    leaky = evaluate_policy(episodes, leaky_policy, tmp_path / "leaky")
    baseline = write_runs_jsonl(leaky["rows"], tmp_path / "baseline.jsonl")
    treatment = write_runs_jsonl(honest["rows"], tmp_path / "treatment.jsonl")
    report = compare_runs(str(baseline), str(treatment))
    assert report["treatment"]["temporal_leakage"] == 0.0
    assert report["baseline"]["temporal_leakage"] == 1.0
    assert report["absolute_success_uplift"] == 1.0


def test_standard_suite_shape(worlds):
    episodes = standard_suite(worlds, seeds=(5,))
    kinds = {episode.kind for episode in episodes}
    assert kinds == {"forecastable", "abstention_trap", "leakage_trap", "regime_break_trap"}

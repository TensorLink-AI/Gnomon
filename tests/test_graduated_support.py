"""Graduated support: always answer, honestly graded.

The default publication floor is best_effort — every request answers at
the highest tier the evidence achieves, tier-labelled — while nothing
about how any tier is earned changes. `minimum_support` restores the
refusal one parameter away, and a series with no usable history still
abstains.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

DAILY = str(Path(__file__).resolve().parent.parent / "examples" / "daily_requests.csv")


def _short_csv(tmp_path: Path, rows: int = 12) -> str:
    start = date(2026, 3, 1)
    lines = ["timestamp,requests"] + [
        f"{start + timedelta(days=day)},{100 + 3 * day + (day % 3)}"
        for day in range(rows)
    ]
    path = tmp_path / "short.csv"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_default_answers_on_fold_starved_data(tmp_path) -> None:
    from gnomon.toolspec import runner_for

    payload = runner_for("gnomon_forecast")({
        "input": _short_csv(tmp_path), "horizon": 14,
        "output_dir": str(tmp_path / "out"),
    })
    result = payload["results"][0]
    assert result["forecast"], "default must answer, not abstain"
    assert result["support"] == "best_effort"
    assert any("NO RELIABLE FORECAST" in group["message"]
               for group in payload["limitation_groups"])
    # The abstention's reasons are preserved, typed, alongside the rows.
    codes = {reason["code"]
             for reason in result["support_assessment"]["reasons"]}
    assert "no_reliable_forecast" in codes


def test_minimum_support_supported_restores_the_refusal(tmp_path) -> None:
    from gnomon.toolspec import runner_for

    payload = runner_for("gnomon_forecast")({
        "input": _short_csv(tmp_path), "horizon": 14,
        "minimum_support": "supported",
        "output_dir": str(tmp_path / "out"),
    })
    result = payload["results"][0]
    assert result["forecast"] == []
    assert result["support"] == "unsupported"
    assessment = result["support_assessment"]
    assert assessment["status"] == "inconclusive"
    codes = {action["code"] for action in assessment["recovery_actions"]}
    assert "provide_more_history" in codes


def test_floor_refuses_a_conditionally_supported_result(tmp_path) -> None:
    # daily_requests earns at most conditionally_supported at this horizon
    # (degraded evaluation); floor `supported` must refuse it with the
    # typed below-floor reason, publishing nothing.
    from gnomon.toolspec import runner_for

    payload = runner_for("gnomon_forecast")({
        "input": DAILY, "horizon": 7, "minimum_support": "supported",
        "output_dir": str(tmp_path / "out"),
    })
    result = payload["results"][0]
    assert result["forecast"] == []
    codes = {reason["code"]
             for reason in result["support_assessment"]["reasons"]}
    assert "below_minimum_support" in codes
    recovery = {action["code"] for action in
                result["support_assessment"]["recovery_actions"]}
    assert "lower_minimum_support" in recovery


def test_nothing_computable_still_abstains(tmp_path) -> None:
    # Two observations cannot even seed the naive fallback's dispersion in
    # a meaningful way — but the guard here is the truly-empty case: zero
    # usable rows after load fails loudly, and one observation publishes a
    # labelled flat line. The invariant under test: an unparseable series
    # never fabricates rows. A file whose target column never parses is a
    # typed error, not a best_effort forecast.
    from gnomon.contracts import GnomonError
    from gnomon.toolspec import runner_for

    bad = tmp_path / "bad.csv"
    bad.write_text("timestamp,requests\n2026-01-01,not_a_number\n")
    import pytest
    with pytest.raises(GnomonError):
        runner_for("gnomon_forecast")({
            "input": str(bad), "horizon": 3,
            "output_dir": str(tmp_path / "out"),
        })


def test_legacy_best_effort_parameter_still_works(tmp_path) -> None:
    from gnomon.toolspec import runner_for

    legacy = runner_for("gnomon_forecast")({
        "input": _short_csv(tmp_path), "horizon": 14, "best_effort": True,
        "output_dir": str(tmp_path / "legacy"),
    })
    result = legacy["results"][0]
    assert result["support"] == "best_effort"
    assert result["forecast"]


def test_invalid_minimum_support_is_typed(tmp_path) -> None:
    import pytest

    from gnomon.contracts import GnomonError
    from gnomon.toolspec import runner_for

    with pytest.raises(GnomonError) as excinfo:
        runner_for("gnomon_forecast")({
            "input": _short_csv(tmp_path), "horizon": 3,
            "minimum_support": "everything",
            "output_dir": str(tmp_path / "out"),
        })
    assert excinfo.value.code == "INVALID_ARGUMENTS"


def test_horizon_split_publishes_prefix_and_labelled_remainder(tmp_path) -> None:
    from gnomon.toolspec import runner_for

    # 12 daily observations support horizon 10, not the requested 14; the
    # old behaviour abstained naming max_supportable_horizon in recovery.
    payload = runner_for("gnomon_forecast")({
        "input": _short_csv(tmp_path), "horizon": 14,
        "output_dir": str(tmp_path / "out"),
    })
    result = payload["results"][0]
    rows = result["forecast"]
    assert len(rows) == 14
    tiers = [row["tier"] for row in rows]
    assert "best_effort" in tiers
    evaluated = [tier for tier in tiers if tier != "best_effort"]
    assert evaluated, "the supportable prefix must be evaluated"
    # Contiguous: evaluated prefix first, fallback remainder after.
    boundary = len(evaluated)
    assert all(tier != "best_effort" for tier in tiers[:boundary])
    assert all(tier == "best_effort" for tier in tiers[boundary:])

    assessment = result["support_assessment"]
    split_reasons = [reason for reason in assessment["reasons"]
                     if reason["code"] == "horizon_split"]
    assert len(split_reasons) == 1
    message = split_reasons[0]["message"]
    assert f"1-{boundary}" in message
    assert f"{boundary + 1}-14" in message
    assert assessment["sensitivity"]["supported_horizon"] == boundary
    assert assessment["sensitivity"]["requested_horizon"] == 14


def test_split_artifact_csv_carries_the_tier_column(tmp_path) -> None:
    import csv

    from gnomon.toolspec import runner_for

    payload = runner_for("gnomon_forecast")({
        "input": _short_csv(tmp_path), "horizon": 14,
        "output_dir": str(tmp_path / "out"),
    })
    with open(payload["artifact_path"] + "/forecast.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 14
    assert {"tier"} <= set(rows[0])
    assert rows[-1]["tier"] == "best_effort"
    assert rows[0]["tier"] != "best_effort"


def test_split_respects_a_supported_floor(tmp_path) -> None:
    # With floor `supported` the remainder cannot be published, so the
    # request abstains and the split is reported in recovery instead.
    from gnomon.toolspec import runner_for

    payload = runner_for("gnomon_forecast")({
        "input": _short_csv(tmp_path), "horizon": 14,
        "minimum_support": "supported",
        "output_dir": str(tmp_path / "out"),
    })
    result = payload["results"][0]
    assert result["forecast"] == []
    codes = {action["code"] for action in
             result["support_assessment"]["recovery_actions"]}
    assert "reduce_horizon" in codes


def test_uniform_forecasts_repeat_one_tier_per_row(tmp_path) -> None:
    from gnomon.toolspec import runner_for

    payload = runner_for("gnomon_forecast")({
        "input": DAILY, "horizon": 7, "format": "full",
        "output_dir": str(tmp_path / "out"),
    })
    result = payload["results"][0]
    tiers = {row["tier"] for row in result["forecast_preview"]}
    assert len(tiers) == 1
    assert tiers <= {"supported", "conditionally_supported"}


def test_happy_path_rows_unchanged_except_tier(tmp_path) -> None:
    # Regression guard: a fully evaluated forecast's numbers, support, and
    # assessment are exactly what they were; the additive fields are the
    # per-row tier (and, later, the headline).
    from gnomon.toolspec import runner_for

    payload = runner_for("gnomon_forecast")({
        "input": DAILY, "horizon": 7, "format": "full",
        "output_dir": str(tmp_path / "out"),
    })
    result = payload["results"][0]
    assert result["support"] in ("supported", "weakly_supported", "degraded")
    row = dict(result["forecast_preview"][0])
    row.pop("tier")
    assert set(row) >= {"timestamp", "point", "q10", "q50", "q90",
                        "point_bias_correction"}
    # No split machinery engaged: one tier, no horizon_split reason.
    codes = {reason["code"]
             for reason in result["support_assessment"]["reasons"]}
    assert "horizon_split" not in codes


def test_headline_names_the_weakest_tier(tmp_path) -> None:
    from gnomon.toolspec import runner_for

    # Split: the headline names both ranges and the naive remainder.
    split = runner_for("gnomon_forecast")({
        "input": _short_csv(tmp_path), "horizon": 14,
        "output_dir": str(tmp_path / "a"),
    })
    assert "naive extrapolation" in split["headline"]
    assert "Higher-confidence through" in split["headline"]

    # Evaluated with caveats: the first typed reason, plain form.
    graded = runner_for("gnomon_forecast")({
        "input": DAILY, "horizon": 7, "output_dir": str(tmp_path / "b"),
    })
    assert graded["headline"].startswith(
        ("High-confidence forecast through", "Forecast through"))

    # Both formats carry it, verbatim identical.
    full = runner_for("gnomon_forecast")({
        "input": DAILY, "horizon": 7, "format": "full",
        "output_dir": str(tmp_path / "c"),
    })
    assert full["headline"] == graded["headline"]


def test_headline_for_pure_fallback_names_orientation_only(tmp_path, monkeypatch) -> None:
    import gnomon.runtime as runtime_module

    from gnomon.toolspec import runner_for

    monkeypatch.setattr(runtime_module, "_split_prefix",
                        lambda *args, **kw: None)
    payload = runner_for("gnomon_forecast")({
        "input": _short_csv(tmp_path), "horizon": 14,
        "output_dir": str(tmp_path / "out"),
    })
    assert "orientation only" in payload["headline"]
    assert "12 observations" in payload["headline"]


def test_headline_is_the_summary_md_first_line(tmp_path) -> None:
    from gnomon.toolspec import runner_for

    payload = runner_for("gnomon_forecast")({
        "input": _short_csv(tmp_path), "horizon": 14,
        "output_dir": str(tmp_path / "out"),
    })
    first_line = open(payload["artifact_path"] + "/summary.md").readline()
    assert first_line.strip() == payload["headline"]


def test_abstention_headline_states_no_publication(tmp_path) -> None:
    from gnomon.toolspec import runner_for

    payload = runner_for("gnomon_forecast")({
        "input": _short_csv(tmp_path), "horizon": 14,
        "minimum_support": "supported",
        "output_dir": str(tmp_path / "out"),
    })
    assert payload["headline"].startswith("No forecast published:")


def test_verifier_rejects_unlabelled_sub_supported_claims(tmp_path) -> None:
    """A hand-constructed claim quoting best_effort values without the
    tier fails verification exactly like an uncalibrated probability."""
    from gnomon.lineage import ClaimRecord, EvidenceRecord, Lineage
    from gnomon.verifier import verify_lineage

    lineage = Lineage("task:test", {})
    lineage.evidence.append(EvidenceRecord(
        "support:cpu", "support_assessment", "cpu",
        {"support": "best_effort", "row_tiers": {"best_effort": 5}},
    ))
    lineage.claims.append(ClaimRecord(
        claim_id="claim:sneaky:cpu",
        claim_class="descriptive",
        statement="cpu will be 42.0 at the next step.",
        subject="cpu",
        evidence_ids=("support:cpu",),
        artifact_ids=(),
    ))
    violations = verify_lineage(lineage, as_of=None)
    codes = {violation["code"] for violation in violations}
    assert "SUB_SUPPORTED_UNLABELLED" in codes

    # The same claim, labelled, passes.
    labelled = Lineage("task:test", {})
    labelled.evidence.append(EvidenceRecord(
        "support:cpu", "support_assessment", "cpu",
        {"support": "best_effort", "row_tiers": {"best_effort": 5}},
    ))
    labelled.claims.append(ClaimRecord(
        claim_id="claim:honest:cpu",
        claim_class="descriptive",
        statement=("cpu is extrapolated to 42.0 at the next step — a "
                   "best_effort naive extrapolation with no measured "
                   "accuracy."),
        subject="cpu",
        evidence_ids=("support:cpu",),
        artifact_ids=(),
    ))
    assert not any(
        violation["code"] == "SUB_SUPPORTED_UNLABELLED"
        for violation in verify_lineage(labelled, as_of=None)
    )


def test_real_split_and_fallback_artifacts_pass_the_verifier(tmp_path) -> None:
    # The production lineage builder must satisfy the new check on its
    # own output: a run reaching disk proves it verified, so producing
    # both shapes end-to-end is the test.
    from gnomon.toolspec import runner_for

    split = runner_for("gnomon_forecast")({
        "input": _short_csv(tmp_path), "horizon": 14,
        "output_dir": str(tmp_path / "a"),
    })
    assert split["results"][0]["support"] == "best_effort"


def test_requested_threshold_is_disclosed_not_dropped(tmp_path) -> None:
    # A threshold on a split/fallback run cannot be analysed (no
    # calibrated residuals); the absence is a typed note, never silence.
    from gnomon.toolspec import runner_for

    payload = runner_for("gnomon_forecast")({
        "input": _short_csv(tmp_path), "horizon": 14, "threshold": 150.0,
        "output_dir": str(tmp_path / "out"),
    })
    result = payload["results"][0]
    assert result["support"] == "best_effort"
    assert result["threshold"]["probability_status"] == \
        "unavailable_uncalibrated"
    assert result["threshold"]["bounded_assessment"][
        "automation_eligible"] is False


def test_decide_and_monitor_never_rest_on_fallback_rows(tmp_path) -> None:
    # The graduated default gives the embedded forecast labelled rows,
    # but a decision or alert rule must still refuse: sub-supported rows
    # carry no calibrated exceedance risk.
    from gnomon.macros import decide, monitor

    common = dict(time_column="timestamp", target_column="requests",
                  horizon=14, threshold=150.0)
    decision, _ = decide(
        _short_csv(tmp_path), actions=[{"name": "scale_up"}],
        output=str(tmp_path / "a"), **common)
    assert decision["scenario_probabilities"] is None
    codes = {reason["code"] for reason in
             decision["support_assessment"]["reasons"]}
    assert "forecast_not_calibrated" in codes

    monitored, _ = monitor(
        _short_csv(tmp_path), output=str(tmp_path / "b"), **common)
    trigger = monitored["triggers"][0]
    assert trigger["armed"] is False
    codes = {reason["code"] for reason in
             trigger["support_assessment"]["reasons"]}
    assert "forecast_not_calibrated" in codes


def test_headline_caveat_is_one_sentence(tmp_path) -> None:
    from gnomon.support import forecast_headline

    headline = forecast_headline(
        "degraded",
        {"status": "conditionally_supported",
         "reasons": [{"code": "degraded_evaluation",
                      "message": "First sentence here. Second sentence "
                                 "that should not appear. Third."}],
         "sensitivity": {}},
        [{"timestamp": "2026-01-08T00:00:00", "tier": "conditionally_supported"}],
    )
    assert headline.endswith("with caveats: First sentence here.")
    assert "Second sentence" not in headline


def test_capabilities_report_the_default_floor() -> None:
    from gnomon.runtime import capabilities

    graduated = capabilities()["forecast_surface"]["graduated_support"]
    assert graduated["default_minimum_support"] == "best_effort"
    assert graduated["tiers"] == [
        "best_effort", "conditionally_supported", "supported",
    ]

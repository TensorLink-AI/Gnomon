from gnomon.temporal_adjudication import adjudicate_temporal_evidence


def _result(value="stable", support="weak"):
    return {"best_estimate": {"value": value, "support": support},
            "answer": {}}


def test_canonical_wins_without_independent_opposition() -> None:
    adjudication = adjudicate_temporal_evidence(
        _result(), evidence=[{"kind": "observed_transition",
                              "direction": "increased", "support": "weak"}])
    assert adjudication["relationship"] == "canonical_preferred"
    assert adjudication["synthesis_eligibility"]["eligible"] is False
    assert adjudication["primary_forecast_unchanged"] is True


def test_two_supported_independent_sources_can_earn_synthesis() -> None:
    adjudication = adjudicate_temporal_evidence(
        _result(), evidence=[
            {"kind": "observed_transition", "direction": "increased",
             "support": "supported"},
            {"kind": "cross_series_observed_evidence", "direction": "increased",
             "support": "supported"},
        ])
    assert adjudication["relationship"] == "alternative_preferred"
    assert adjudication["synthesis_eligibility"]["eligible"] is True
    assert adjudication["synthesis_eligibility"][
        "independent_opposing_sources"] == 2


def test_two_weak_sources_do_not_earn_synthesis() -> None:
    adjudication = adjudicate_temporal_evidence(
        _result(), evidence=[
            {"kind": "observed_transition", "direction": "increased",
             "support": "weak"},
            {"kind": "cross_series_observed_evidence", "direction": "increased",
             "support": "weak"},
        ])
    assert adjudication["relationship"] == "alternative_preferred"
    assert adjudication["synthesis_eligibility"]["eligible"] is False
    assert adjudication["synthesis_eligibility"][
        "supported_independent_opposing_sources"] == 0


def test_supported_canonical_stays_binding_even_when_opposed() -> None:
    adjudication = adjudicate_temporal_evidence(
        _result(support="supported"), evidence=[
            {"kind": "observed_transition", "direction": "increased",
             "support": "supported"},
            {"kind": "historical_effect", "direction": "increased",
             "support": "supported"},
        ])
    assert adjudication["synthesis_eligibility"]["eligible"] is False


def test_probabilities_are_never_invented() -> None:
    no_distribution = adjudicate_temporal_evidence(_result(), evidence=[])
    assert no_distribution["calibration"]["available"] is False
    result = _result()
    result["answer"]["property_distribution"] = {
        "probabilities": {"stable": .7, "increased": .3}, "folds": 12,
        "balanced_accuracy": .8, "brier_skill": .1}
    fitted = adjudicate_temporal_evidence(result, evidence=[])
    assert fitted["candidates"][0]["fitted_probability"] == .7
    assert fitted["calibration"]["source"] == "fitted_property_distribution"


def test_unvalidated_context_stays_conditional() -> None:
    effect = {"distribution": {"location": 4.0, "sample_size": 0},
              "provenance": {"provenance_class": "human_assumption",
                             "known_at": "2026-01-01T00:00:00+00:00"}}
    adjudication = adjudicate_temporal_evidence(
        _result(), evidence=[], conditional_effect=effect)
    alternative = adjudication["alternative"]
    assert alternative["conditional_only"] is True
    assert adjudication["synthesis_eligibility"]["eligible"] is False


def test_wrapped_tracked_effect_uses_property_vocabulary_and_provenance() -> None:
    effect = {"measured_effect": {
        "distribution": {"location": -4.0, "sample_count": 8},
        "provenance": {
            "provenance_class": "same_event_same_series", "observed": True,
            "known_at": "2026-01-01T00:00:00+00:00",
            "source_reference": "tracking:ops", "reliability": .8,
        },
        "shape": "trend_change",
    }}
    adjudication = adjudicate_temporal_evidence(
        _result("constant"), evidence=[], conditional_effect=effect)
    alternative = adjudication["alternative"]
    assert alternative["value"] == "downward"
    assert alternative["support"] == "supported"
    assert alternative["conditional_only"] is False
    assert alternative["sources"][0]["provenance"][
        "source_reference"] == "tracking:ops"


def test_unknown_effect_vocabulary_does_not_invent_a_choice() -> None:
    effect = {"distribution": {"location": 4.0, "sample_count": 8},
              "provenance": {"provenance_class": "external_prior",
                             "observed": True}}
    adjudication = adjudicate_temporal_evidence(
        _result("opaque_label"), evidence=[], conditional_effect=effect)
    assert adjudication["alternative"] is None

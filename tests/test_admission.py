from gnomon.admission import (
    ExternalModelPrior, OutputDiagnostics, decide_admission, local_evidence,
)
from gnomon.candidate import (
    CandidateIdentity, CandidateSpec, FittedCandidate, blended_candidate_spec,
)


def _prior(gain=.10, se=.02, overlap="low"):
    return ExternalModelPrior(
        model="foundation", revision="weights@abc",
        regime=(("frequency", "hourly"),), comparisons=500,
        mean_relative_gain=gain, standard_error=se,
        source_ids=("held-out-registry",), registry_version="r1",
        overlap_risk=overlap,
    )


def test_pretrained_candidate_can_be_externally_validated_without_local_folds():
    evidence = local_evidence(
        model_class="pretrained", candidate_losses=[], baseline_losses=[],
        external_prior=_prior(),
    )
    decision = decide_admission(candidate="foundation", baseline="last", evidence=evidence)
    assert decision.state == "externally_validated"
    assert decision.point_policy == "candidate"
    assert decision.candidate_weight == 1


def test_uncertain_external_prior_is_shrunk_not_presented_as_local_proof():
    evidence = local_evidence(
        model_class="pretrained", candidate_losses=[], baseline_losses=[],
        external_prior=_prior(gain=.02, se=.04),
    )
    decision = decide_admission(candidate="foundation", baseline="last", evidence=evidence)
    assert decision.state == "prior_assisted"
    assert decision.point_policy == "shrunk_blend"
    assert 0 < decision.candidate_weight < 1
    assert decision.evidence.independent_folds == 0


def test_training_overlap_risk_disqualifies_external_prior():
    evidence = local_evidence(
        model_class="pretrained", candidate_losses=[], baseline_losses=[],
        external_prior=_prior(overlap="high"),
    )
    decision = decide_admission(candidate="foundation", baseline="last", evidence=evidence)
    assert decision.state == "baseline_fallback"


def test_local_repeatable_wins_validate_any_model_class():
    evidence = local_evidence(
        model_class="locally_fitted",
        candidate_losses=[.6, .7, .5], baseline_losses=[1., 1., 1.],
    )
    decision = decide_admission(candidate="trend", baseline="last", evidence=evidence)
    assert decision.state == "locally_validated"
    assert decision.point_policy == "candidate"


def test_invalid_output_falls_back_despite_strong_prior():
    evidence = local_evidence(
        model_class="pretrained", candidate_losses=[], baseline_losses=[],
        external_prior=_prior(), diagnostics=OutputDiagnostics(
            valid=False, reasons=("non-finite",)),
    )
    decision = decide_admission(candidate="foundation", baseline="last", evidence=evidence)
    assert decision.state == "baseline_fallback"


def test_compact_projection_counts_but_does_not_inline_registry_sources():
    evidence = local_evidence(
        model_class="pretrained", candidate_losses=[], baseline_losses=[],
        external_prior=_prior(),
    )
    decision = decide_admission(candidate="foundation", baseline="last", evidence=evidence)
    compact = decision.to_payload(compact=True)
    projected = compact["evidence"]["external_prior"]
    assert projected["source_count"] == 1
    assert "source_ids" not in projected
    assert decision.to_payload()["evidence"]["external_prior"]["source_ids"]


def _constant(name, value):
    identity = CandidateIdentity(kind="test", name=name)
    return CandidateSpec(identity, lambda history, season: FittedCandidate(
        identity.with_fit(weights=None, data_fingerprint=name),
        lambda horizon: [value] * horizon,
    ))


def test_blended_executable_owns_the_published_points_and_identity():
    spec = blended_candidate_spec(
        _constant("foundation", 10), _constant("last", 2), .25,
        policy_version="v1", admission_state="prior_assisted",
    )
    fitted = spec.fit([1, 2, 3], None)
    assert fitted.predict(3) == [4, 4, 4]
    assert fitted.identity.weights == {"foundation": .25, "last": .75}
    assert fitted.identity.config["admission_state"] == "prior_assisted"

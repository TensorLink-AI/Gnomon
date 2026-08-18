from gnomon.temporal_vocabulary import project_temporal_choice


def test_projection_prefers_exact_option_and_preserves_display_casing() -> None:
    projected = project_temporal_choice(
        "higher", ["Higher", "Lower", "Similar", "Uncertain"])
    assert projected == {
        "canonical": "higher", "display_value": "Higher",
        "method": "exact", "version": "0.1",
    }


def test_projection_maps_only_unambiguous_semantic_alias() -> None:
    assert project_temporal_choice(
        "stable", ["increased", "decreased", "constant", "Uncertain"]
    )["display_value"] == "constant"
    assert project_temporal_choice(
        "continued", ["fixed", "shifting", "no", "Uncertain"]
    )["display_value"] == "fixed"


def test_projection_refuses_ambiguous_or_support_state() -> None:
    assert project_temporal_choice(
        "weaker", ["fixed", "shifting", "no", "Uncertain"]) is None
    assert project_temporal_choice("weak", ["weak", "strong"]) is None
    assert project_temporal_choice("continued", ["fixed", "aligned"]) is None


def test_projection_preserves_only_an_explicit_uncertain_option() -> None:
    projected = project_temporal_choice(
        "uncertain", ["fixed", "shifting", "no", "Uncertain"])
    assert projected is not None
    assert projected["display_value"] == "Uncertain"
    assert project_temporal_choice("uncertain", ["fixed", "shifting"]) is None

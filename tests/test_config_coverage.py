"""Every key `gnomon.yaml` documents either takes effect or is refused.

Roughly thirty documented options were parsed and never read: a user who
disabled statistical models still got all five, a user who set
`target_coverage: 0.70` still got an 80% interval, and a user who set
`write_summary: false` still got a summary. A setting that is silently
ignored is worse than one that does not exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gnomon.config import INERT_KEYS, GnomonConfig, _flatten, load_config
from gnomon.contracts import GnomonError

REPO = Path(__file__).resolve().parent.parent


def _example_keys() -> set[str]:
    yaml = pytest.importorskip("yaml")
    raw = yaml.safe_load((REPO / "gnomon.yaml.example").read_text(encoding="utf-8"))
    return set(_flatten(raw or {}))


def test_the_example_config_loads():
    """It documents the product; it has to be a valid config."""
    config = load_config(str(REPO / "gnomon.yaml.example"))
    assert isinstance(config, GnomonConfig)


def test_the_preferred_toml_example_loads(tmp_path):
    """The zero-dependency canonical example must be executable documentation."""
    path = tmp_path / "gnomon.toml"
    path.write_text((REPO / "gnomon.toml.example").read_text(encoding="utf-8"),
                    encoding="utf-8")
    config = load_config(str(path))
    assert isinstance(config, GnomonConfig)
    assert config.models.statistical_candidates == [
        "drift", "linear_trend", "window_average", "theta", "ets",
        "croston_sba",
    ]
    assert config.models.statsforecast_enabled is False
    assert config.models.statsforecast_candidates == [
        "statsforecast_autoets", "statsforecast_autoarima",
        "statsforecast_autotheta", "statsforecast_croston_optimized",
    ]


def test_statsforecast_config_is_honoured(tmp_path):
    path = tmp_path / "gnomon.toml"
    path.write_text(
        "[models.statsforecast]\n"
        "enabled = true\n"
        'candidates = ["statsforecast_autoets"]\n',
        encoding="utf-8",
    )
    config = load_config(str(path))
    assert config.models.statsforecast_enabled is True
    assert config.models.statsforecast_candidates == [
        "statsforecast_autoets"]


def test_the_example_documents_no_inert_key():
    """If a key cannot take effect, the example must not advertise it."""
    advertised = _example_keys() & set(INERT_KEYS)
    assert not advertised, (
        "gnomon.yaml.example documents keys Gnomon refuses: " + ", ".join(sorted(advertised))
    )


@pytest.mark.parametrize("key", sorted(INERT_KEYS))
def test_every_inert_key_is_refused(tmp_path, key):
    """Named explicitly, an inert key raises rather than being ignored."""
    yaml = pytest.importorskip("yaml")

    nested: dict = {}
    cursor = nested
    parts = key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = True

    path = tmp_path / "gnomon.yaml"
    path.write_text(yaml.safe_dump(nested), encoding="utf-8")
    with pytest.raises(GnomonError) as raised:
        load_config(str(path))
    assert raised.value.code == "UNSUPPORTED_CONFIG_KEY"
    assert key in raised.value.details["reasons"]
    # The refusal explains itself rather than just saying no.
    assert len(raised.value.details["reasons"][key]) > 40


def test_statistical_candidates_restrict_the_pool():
    """`models.statistical.candidates` was parsed and never read."""
    from gnomon.evaluation import active_models
    from gnomon.models import BASELINES

    config = GnomonConfig()
    config.models.statistical_candidates = ["drift"]
    pool = active_models(config)
    assert set(pool) == BASELINES | {"drift"}


def test_disabling_statistical_models_leaves_the_baselines():
    from gnomon.evaluation import active_models
    from gnomon.models import BASELINES

    config = GnomonConfig()
    config.models.statistical_enabled = False
    assert set(active_models(config)) == BASELINES


def test_unknown_candidate_is_an_error_not_a_silent_skip():
    from gnomon.evaluation import active_models

    config = GnomonConfig()
    config.models.statistical_candidates = ["arima"]
    with pytest.raises(GnomonError) as raised:
        active_models(config)
    assert raised.value.code == "UNKNOWN_MODEL"
    assert "arima" in raised.value.details["unknown"]


def test_default_pool_is_every_built_in_model():
    from gnomon.evaluation import active_models
    from gnomon.models import MODELS

    assert set(active_models(None)) == set(MODELS)
    assert set(active_models(GnomonConfig())) == set(MODELS)


def test_target_coverage_changes_the_interval_levels():
    """`evaluation.uncertainty.target_coverage` was parsed and never read."""
    from gnomon.evaluation import DEFAULT_TARGET_COVERAGE, coverage_levels

    assert coverage_levels(DEFAULT_TARGET_COVERAGE) == (0.1, 0.5, 0.9)
    assert coverage_levels(0.5) == (0.25, 0.5, 0.75)
    assert coverage_levels(0.9) == (0.05, 0.5, 0.95)


def test_a_narrower_target_gives_a_narrower_interval():
    from gnomon.evaluation import conformal_spreads

    by_lead = {step: [float(value) for value in range(-20, 21)] for step in range(1, 4)}
    wide = conformal_spreads(by_lead, 3, target_coverage=0.90)
    narrow = conformal_spreads(by_lead, 3, target_coverage=0.50)
    for step in range(1, 4):
        wide_width = wide[step][0] + wide[step][2]
        narrow_width = narrow[step][0] + narrow[step][2]
        assert narrow_width < wide_width


def test_out_of_range_target_coverage_is_refused(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "gnomon.yaml"
    path.write_text(
        yaml.safe_dump({"evaluation": {"uncertainty": {"target_coverage": 1.5}}}),
        encoding="utf-8",
    )
    with pytest.raises(GnomonError) as raised:
        load_config(str(path))
    assert raised.value.code == "UNSUPPORTED_CONFIG_KEY"


def test_global_selection_is_refused(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "gnomon.yaml"
    path.write_text(
        yaml.safe_dump({"evaluation": {"selection": "global"}}), encoding="utf-8",
    )
    with pytest.raises(GnomonError) as raised:
        load_config(str(path))
    assert raised.value.code == "UNSUPPORTED_CONFIG_KEY"


def test_output_switches_are_honoured(tmp_path):
    """`write_summary: false` still wrote a summary."""
    from datetime import date, datetime, timedelta, timezone

    from gnomon.ids import FixedClock
    from gnomon.runtime import forecast

    start = date(2026, 1, 1)
    noise = [0.5, -0.3, 0.2, -0.4, 0.1, 0.3, -0.2, -0.1, 0.4, -0.5]
    rows = [
        f"{(start + timedelta(days=i)).isoformat()},{100 + i * 0.4 + noise[i % 10] * 6:.4f}"
        for i in range(80)
    ]
    source = tmp_path / "series.csv"
    source.write_text("timestamp,value\n" + "\n".join(rows) + "\n", encoding="utf-8")

    config = GnomonConfig()
    config.output.write_summary = False
    config.output.write_evidence = False
    _, path = forecast(
        str(source), time_column="timestamp", target_column="value", horizon=7,
        output=str(tmp_path / "out"), config=config,
        clock=FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc)),
    )
    assert (path / "artifact.json").is_file(), "the artifact is never optional"
    assert (path / "forecast.csv").is_file()
    assert not (path / "summary.md").exists()
    assert not (path / "evidence.jsonl").exists()


def test_min_observations_floor_is_honoured():
    """`evaluation.folds.min_observations` was parsed and never read."""
    from gnomon.evaluation import evaluate

    config = GnomonConfig()
    config.evaluation.min_observations = 500
    values = [100 + index * 0.4 for index in range(120)]
    with pytest.raises(GnomonError) as raised:
        evaluate(values, 7, 7, 0.02, frequency="D", tsfm_names=[], config=config)
    assert raised.value.code == "INSUFFICIENT_OBSERVATIONS"
    assert raised.value.details["min_observations"] == 500


def test_no_floor_by_default():
    from gnomon.evaluation import evaluate

    values = [100 + index * 0.4 for index in range(120)]
    assessment = evaluate(
        values, 7, 7, 0.02, frequency="D", tsfm_names=[], config=GnomonConfig(),
    )
    assert assessment.selected_model is not None

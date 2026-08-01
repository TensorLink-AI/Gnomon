"""Plugin model backends: registration, fold competition, provenance,
failure isolation, and artifact-ID salting."""

import sys

import pytest

sys.path.insert(0, "src")

from aion.backends import (
    ModelBackend,
    backend_adapters,
    backends_fingerprint,
    clear_registered_backends,
    register_backend,
    resolve_backend_model,
)
from aion.evaluation import evaluate


@pytest.fixture(autouse=True)
def isolated_registry():
    clear_registered_backends()
    yield
    clear_registered_backends()


def _linear(history, horizon, season):
    """Perfect continuation of a linear series: must beat every baseline."""
    slope = history[-1] - history[-2]
    return [history[-1] + slope * step for step in range(1, horizon + 1)]


def _quadratic(history, horizon, season):
    """Exact continuation of value(i) = (i + 1)**2 — a curve no built-in
    (drift, linear trend, theta, ETS) extrapolates exactly, so winning
    proves the plugin candidate beat them on the folds."""
    n = len(history)
    return [float((n + step) ** 2) for step in range(1, horizon + 1)]


def _broken(history, horizon, season):
    raise RuntimeError("backend exploded")


def _fake_backend(models):
    return ModelBackend(name="fake", version="1.0", models=models)


def test_adapters_are_namespaced_and_sorted():
    register_backend(_fake_backend({"beta": _linear, "alpha": _linear}))
    names = [adapter.name for adapter in backend_adapters()]
    assert names == ["fake:alpha", "fake:beta"]


def test_resolve_backend_model():
    register_backend(_fake_backend({"linear": _linear}))
    assert resolve_backend_model("fake:linear") is _linear
    assert resolve_backend_model("fake:missing") is None
    assert resolve_backend_model("not_namespaced") is None


def test_backend_model_competes_and_wins_on_identical_folds():
    register_backend(_fake_backend({"exact": _quadratic}))
    values = [float((i + 1) ** 2) for i in range(200)]
    result = evaluate(values, horizon=24, season=24, minimum_improvement=0.02)
    assert result.selected_model == "fake:exact"
    assert result.tsfm_scores.get("fake:exact") == 0.0
    assert result.supported is True


def test_broken_backend_scores_none_and_cannot_win():
    register_backend(_fake_backend({"broken": _broken}))
    values = [100.0 + 2.0 * i for i in range(200)]
    result = evaluate(values, horizon=24, season=24, minimum_improvement=0.02)
    assert result.tsfm_scores.get("fake:broken") is None
    assert result.selected_model != "fake:broken"
    assert result.supported is True


def test_fingerprint_lists_backends_or_none():
    assert backends_fingerprint() is None
    register_backend(_fake_backend({"linear": _linear}))
    assert backends_fingerprint() == ["fake==1.0"]


def test_end_to_end_forecast_selects_backend_model(tmp_path):
    """The selected plugin model produces the final forecast rows and the
    artifact ID is salted with the backend fingerprint."""
    from aion.runtime import forecast

    register_backend(_fake_backend({"exact": _quadratic}))
    from datetime import date, timedelta

    rows = ["timestamp,value"]
    for day in range(1, 61):
        stamp = date(2026, 1, 1) + timedelta(days=day - 1)
        rows.append(f"{stamp.isoformat()}T00:00:00,{day * day}")
    source = tmp_path / "linear.csv"
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")

    artifact, _path = forecast(
        str(source), time_column="timestamp", target_column="value",
        horizon=5, frequency="D", output=str(tmp_path / "out"),
    )
    result = artifact.results[0]
    assert result.selected_model == "fake:exact"
    assert len(result.forecast) == 5
    expected = [float((60 + step) ** 2) for step in range(1, 6)]
    assert [row["point"] for row in result.forecast] == pytest.approx(expected)
    with_backend_id = artifact.forecast_id

    clear_registered_backends()
    artifact_without, _path = forecast(
        str(source), time_column="timestamp", target_column="value",
        horizon=5, frequency="D", output=str(tmp_path / "out2"),
    )
    assert artifact_without.forecast_id != with_backend_id
    assert artifact_without.results[0].selected_model != "fake:exact"


def test_statsforecast_factory_declines_gracefully_without_dependency():
    from aion.backends_statsforecast import backend

    try:
        import statsforecast  # noqa: F401
    except ImportError:
        assert backend() is None
    else:
        loaded = backend()
        assert loaded is not None and loaded.name == "statsforecast"

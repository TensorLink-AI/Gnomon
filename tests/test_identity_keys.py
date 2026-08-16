"""Identity means identity: a key must cover everything that changes the
answer under it.

Every test here corresponds to a way the codebase used to key something by
less than determined it — a step cache that ignored ``as_of``, ids over
action *names* rather than actions, a registry keyed on a content address
that two projects share, and model ids that did not cover the weights.
"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import sqlite3

import pytest

from gnomon.ids import FixedClock
from gnomon.macros import decide, detect_anomalies, investigate_change
from gnomon.tracking import TrackingStore

CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))
NOISE = [0.5, -0.3, 0.2, -0.4, 0.1, 0.3, -0.2, -0.1, 0.4, -0.5]


def _series_csv(tmp_path: Path, name: str = "series.csv", periods: int = 80) -> Path:
    """A trending series long enough to forecast, with a mid-series shift."""
    start = date(2026, 1, 1)
    values = [
        (100 if index < periods // 2 else 118) + NOISE[index % 10] + index * 0.2
        for index in range(periods)
    ]
    rows = [
        f"{(start + timedelta(days=index)).isoformat()},{value:.4f}"
        for index, value in enumerate(values)
    ]
    path = tmp_path / name
    path.write_text("timestamp,value\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


# -- C5: ids cover every answer-changing input ----------------------------

def test_decision_id_covers_action_feasibility(tmp_path):
    """Two decisions that select different actions cannot share an id.

    ``feasible: false`` removes an action from consideration, so the same
    action names with different feasibility are different computations.
    """
    source = _series_csv(tmp_path)
    common = dict(
        time_column="timestamp", target_column="value", horizon=5,
        threshold=110.0, output=str(tmp_path / "out"), clock=CLOCK,
    )
    feasible, _ = decide(
        str(source),
        actions=[{"name": "scale_up"}, {"name": "do_nothing"}],
        **common,
    )
    infeasible, _ = decide(
        str(source),
        actions=[{"name": "scale_up", "feasible": False}, {"name": "do_nothing"}],
        **common,
    )
    assert feasible["decision_id"] != infeasible["decision_id"]


def test_anomaly_id_covers_labels(tmp_path):
    """A labelled run selects its detector on a different basis, so it is a
    different computation and must not be served the unlabelled artifact."""
    source = _series_csv(tmp_path)
    common = dict(
        time_column="timestamp", target_column="value",
        output=str(tmp_path / "out"), clock=CLOCK, include_tsfm=False,
    )
    unlabelled, _ = detect_anomalies(str(source), **common)
    labelled, _ = detect_anomalies(str(source), labels=["2026-02-10T00:00:00"], **common)
    assert unlabelled["anomaly_id"] != labelled["anomaly_id"]


def test_anomaly_id_covers_target_column(tmp_path):
    """The source fingerprint covers the bytes, not which column was read."""
    start = date(2026, 1, 1)
    rows = [
        f"{(start + timedelta(days=i)).isoformat()},{100 + NOISE[i % 10]:.4f},"
        f"{500 + NOISE[(i + 3) % 10]:.4f}"
        for i in range(60)
    ]
    source = tmp_path / "two_columns.csv"
    source.write_text("timestamp,a,b\n" + "\n".join(rows) + "\n", encoding="utf-8")

    common = dict(
        time_column="timestamp", output=str(tmp_path / "out"),
        clock=CLOCK, include_tsfm=False,
    )
    on_a, _ = detect_anomalies(str(source), target_column="a", **common)
    on_b, _ = detect_anomalies(str(source), target_column="b", **common)
    assert on_a["anomaly_id"] != on_b["anomaly_id"]


def test_investigation_id_covers_target_column(tmp_path):
    start = date(2026, 1, 1)
    rows = [
        f"{(start + timedelta(days=i)).isoformat()},{100 + NOISE[i % 10]:.4f},"
        f"{500 + NOISE[(i + 3) % 10]:.4f}"
        for i in range(60)
    ]
    source = tmp_path / "two_columns.csv"
    source.write_text("timestamp,a,b\n" + "\n".join(rows) + "\n", encoding="utf-8")

    common = dict(time_column="timestamp", output=str(tmp_path / "out"), clock=CLOCK)
    on_a, _ = investigate_change(str(source), target_column="a", **common)
    on_b, _ = investigate_change(str(source), target_column="b", **common)
    assert on_a["investigation_id"] != on_b["investigation_id"]


# -- C10: the registry key is (forecast_id, project, series) --------------

def _register(store: TrackingStore, forecast_id: str, project: str, **kwargs) -> None:
    store.register(
        forecast_id, project, kwargs.pop("series", "__default__"),
        selected_model="ets", naive_error=10.0, horizon=2, frequency="D",
        artifact_path="/tmp/artifact", **kwargs,
    )


def test_second_project_does_not_steal_the_first_projects_scores(tmp_path):
    """A content address is shared by construction; a registration is not."""
    store = TrackingStore(tmp_path / "registry.db")
    _register(store, "fc_shared", "sre-demo")
    store.score_forecast("fc_shared", [100, 110], [95, 105], project="sre-demo")
    _register(store, "fc_shared", "win-demo")

    sre = store.list_forecasts("sre-demo")
    win = store.list_forecasts("win-demo")
    assert [record.forecast_id for record in sre] == ["fc_shared"]
    assert sre[0].mase == pytest.approx(0.5)
    assert [record.forecast_id for record in win] == ["fc_shared"]
    assert win[0].mase is None, "an unscored registration inherited a score"


def test_scoring_one_project_leaves_the_other_untouched(tmp_path):
    store = TrackingStore(tmp_path / "registry.db")
    _register(store, "fc_shared", "a")
    _register(store, "fc_shared", "b")
    store.score_forecast("fc_shared", [100, 110], [95, 105], project="a")

    assert store.get_forecast("fc_shared", "a").mase == pytest.approx(0.5)
    assert store.get_forecast("fc_shared", "b").mase is None
    assert len(store.model_performance("a", "ets")) == 1
    assert store.model_performance("b", "ets") == []


def test_registry_migrates_a_schema_3_database(tmp_path):
    """Opening a pre-composite-key store preserves its rows and its scores."""
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE forecasts (
            forecast_id TEXT PRIMARY KEY, project TEXT NOT NULL, series TEXT NOT NULL,
            cutoff_time TEXT, horizon INTEGER, frequency TEXT, selected_model TEXT,
            support TEXT, threshold REAL, threshold_peak_probability REAL,
            naive_error REAL, artifact_path TEXT, created_at TEXT NOT NULL,
            scored INTEGER DEFAULT 0, mase REAL, mape REAL, bias REAL, coverage REAL,
            threshold_accuracy REAL, scored_at TEXT, drift_flag TEXT);
        CREATE TABLE model_performance (
            project TEXT NOT NULL, model TEXT NOT NULL, forecast_id TEXT NOT NULL,
            mase REAL, mape REAL, bias REAL, coverage REAL, threshold_accuracy REAL,
            scored_at TEXT NOT NULL, series TEXT, horizon INTEGER, frequency TEXT,
            PRIMARY KEY (project, model, forecast_id));
        CREATE TABLE decisions (
            decision_id TEXT PRIMARY KEY, project TEXT NOT NULL, forecast_id TEXT NOT NULL,
            action TEXT NOT NULL, expected_outcome TEXT NOT NULL, actual_outcome TEXT,
            correct INTEGER, created_at TEXT NOT NULL, resolved_at TEXT,
            FOREIGN KEY (forecast_id) REFERENCES forecasts(forecast_id));
        CREATE TABLE schema_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_metadata VALUES ('version', '3');
        INSERT INTO forecasts (forecast_id, project, series, horizon, frequency,
            selected_model, naive_error, artifact_path, created_at, scored, mase)
            VALUES ('fc_a', 'proj1', '__default__', 2, 'D', 'ets', 10.0,
                    '/tmp/artifact', '2026-01-01T00:00:00+00:00', 1, 0.5);
        INSERT INTO model_performance (project, model, forecast_id, mase, scored_at,
            series, horizon, frequency)
            VALUES ('proj1', 'ets', 'fc_a', 0.5, '2026-01-01T00:00:00+00:00', NULL, 2, 'D');
        INSERT INTO decisions VALUES ('dec1', 'proj1', 'fc_a', 'scale_up', '{}',
            NULL, NULL, '2026-01-01T00:00:00+00:00', NULL);
    """)
    conn.commit()
    conn.close()

    store = TrackingStore(path)

    records = store.list_forecasts()
    assert [(r.forecast_id, r.project, r.mase) for r in records] == [
        ("fc_a", "proj1", 0.5)
    ]
    assert len(store.list_decisions()) == 1
    performance = store.model_performance("proj1", "ets")
    assert [(p["series"], p["mase"]) for p in performance] == [("__default__", 0.5)]

    with sqlite3.connect(path) as conn:
        version = conn.execute(
            "SELECT value FROM schema_metadata WHERE key = 'version'"
        ).fetchone()[0]
        forecasts_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'forecasts'"
        ).fetchone()[0]
    from gnomon.tracking import SCHEMA_VERSION

    assert version == SCHEMA_VERSION
    assert "PRIMARY KEY (forecast_id, project, series)" in forecasts_sql

    # And the migrated store behaves like a new one.
    _register(store, "fc_a", "proj2")
    assert store.get_forecast("fc_a", "proj1").mase == pytest.approx(0.5)
    assert store.get_forecast("fc_a", "proj2").mase is None


def test_migration_is_idempotent(tmp_path):
    store = TrackingStore(tmp_path / "registry.db")
    _register(store, "fc_a", "proj1")
    store.score_forecast("fc_a", [100, 110], [95, 105], project="proj1")
    reopened = TrackingStore(tmp_path / "registry.db")
    assert reopened.get_forecast("fc_a", "proj1").mase == pytest.approx(0.5)


# -- C8: an id that names a model must cover its weights ------------------

def test_every_adapter_model_id_is_pinned():
    from gnomon.tsfm import _ADAPTER_MODEL_IDS, TSFM_REVISIONS
    for adapter, model_ids in _ADAPTER_MODEL_IDS.items():
        for model_id in model_ids:
            assert model_id in TSFM_REVISIONS, f"{adapter} loads {model_id} unpinned"


def test_pinned_revisions_are_commit_shas():
    """A tag or branch can move under a fixed name; a commit cannot."""
    from gnomon.tsfm import TSFM_REVISIONS
    for model_id, revision in TSFM_REVISIONS.items():
        assert len(revision) == 40 and all(
            character in "0123456789abcdef" for character in revision
        ), f"{model_id} is pinned to {revision!r}, which is not a commit sha"


def test_unpinned_model_id_is_refused():
    from gnomon.tsfm import UnpinnedWeights, pinned_revision
    with pytest.raises(UnpinnedWeights):
        pinned_revision("someone/an-unpinned-model")


def test_every_sandbox_pip_spec_is_pinned():
    from gnomon.tsfm_sandbox import TSFM_PIP_SPECS
    for adapter, specs in TSFM_PIP_SPECS.items():
        for spec in specs:
            pinned = "==" in spec or (spec.startswith("git+") and "@" in spec.rsplit("/", 1)[-1])
            assert pinned, f"{adapter} installs {spec!r} unpinned"


def test_every_adapter_load_passes_a_revision():
    """An adapter added without ``revision=`` would reintroduce the bug."""
    import ast
    import inspect

    import gnomon.tsfm as tsfm_module

    tree = ast.parse(inspect.getsource(tsfm_module))
    unpinned = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
        and not any(keyword.arg == "revision" for keyword in node.keywords)
    ]
    assert unpinned == [], f"from_pretrained without revision= at lines {unpinned}"

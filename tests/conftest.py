import pytest


@pytest.fixture(autouse=True)
def _isolated_temporal_store(monkeypatch, tmp_path):
    """Tests must not use an operator's ambient observation store."""
    temporal_path = tmp_path / "temporal-store.db"
    monkeypatch.setenv("GNOMON_TEMPORAL_STORE_PATH", str(temporal_path))
    from gnomon import temporal_store
    monkeypatch.setattr(temporal_store, "DEFAULT_STORE_PATH", temporal_path)

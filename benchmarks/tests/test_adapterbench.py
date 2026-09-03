import benchmarks.adapterbench.run_adapterbench as adapterbench


def test_adapterbench_graduates(monkeypatch) -> None:
    # This is the deterministic protocol gate.  The standalone benchmark
    # deliberately exercises locally installed external adapters, but a unit
    # test must not change meaning according to a developer's global model
    # cache or optional imports.
    monkeypatch.setattr(adapterbench, "installed_tsfms", lambda: [])
    monkeypatch.setattr(adapterbench, "sandbox_available_tsfms", lambda: [])
    result = adapterbench.run()
    assert result["graduated"] is True

import pytest


@pytest.fixture(autouse=True)
def _explicit_full_profile_for_legacy_surface_tests(monkeypatch, tmp_path):
    """Tool unit tests request the broad registry; default tests delete this."""
    monkeypatch.setenv("GNOMON_MCP_PROFILE", "full")
    # Context-aware calls persist receipts by design. Tests must never write
    # into a developer's real project cache or leave `.gnomon/` in the clone.
    monkeypatch.setenv("GNOMON_CONTEXT_STORE", str(tmp_path / "context-store"))
    monkeypatch.setenv("GNOMON_CONTEXT_NAMESPACE", "pytest")
    # Optional model sandboxes are developer-local state.  Discovering a
    # previously installed TSFM here changes candidate selection, artifact
    # goldens, and tests whose contract explicitly starts without optional
    # models.  Keep ordinary tests hermetic; sandbox-specific tests override
    # this module attribute themselves.
    sandbox_root = tmp_path / "tsfm-sandboxes"
    monkeypatch.setenv("GNOMON_TSFM_SANDBOX_ROOT", str(sandbox_root))
    from gnomon import tsfm_sandbox
    monkeypatch.setattr(
        tsfm_sandbox, "SANDBOX_ROOT", sandbox_root)


def pytest_addoption(parser):
    parser.addoption(
        "--update-goldens", action="store_true", default=False,
        help="Rewrite golden artifact files from the current runtime output",
    )

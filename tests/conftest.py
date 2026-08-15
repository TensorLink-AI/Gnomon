import pytest


@pytest.fixture(autouse=True)
def _explicit_full_profile_for_legacy_surface_tests(monkeypatch):
    """Tool unit tests request the broad registry; default tests delete this."""
    monkeypatch.setenv("GNOMON_MCP_PROFILE", "full")


def pytest_addoption(parser):
    parser.addoption(
        "--update-goldens", action="store_true", default=False,
        help="Rewrite golden artifact files from the current runtime output",
    )

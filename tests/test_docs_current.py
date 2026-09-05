"""Documentation checks follow executable public contracts, not old proposals."""

from pathlib import Path
import re

import pytest

REPO = Path(__file__).resolve().parents[1]
README = (REPO / "README.md").read_text()
DOCS = REPO / "docs"
LINK = re.compile(r"\[[^\]]+\]\(([^)#]+)(?:#[^)]*)?\)")


def test_names_and_public_provider_match_code():
    from gnomon import EphemerisProvider
    assert EphemerisProvider("https://example.invalid").name == "ephemeris/route"
    assert "Ephemeris" in README
    assert "Ephemeris" not in README.split("## Quick start", 1)[0]
    assert "## Optional connectors" in README
    assert "sundial" in README
    assert not any(word in README for word in ("personification", "Greek god", "deity"))


def test_package_version_has_one_source_and_install_docs_pin_it():
    from gnomon import __version__
    from gnomon.ids import GNOMON_VERSION
    from gnomon.mcp_server import SERVER_INFO
    from gnomon.product_contract import __version__ as contract_version
    from gnomon.runtime import capabilities
    assert contract_version == GNOMON_VERSION == SERVER_INFO["version"] == __version__
    assert capabilities()["runtime_version"] == __version__
    project = (REPO / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in project
    assert 'path = "src/gnomon/product_contract.py"' in project
    for path in (REPO / "README.md", DOCS / "installation.md", DOCS / "getting-started.md"):
        assert f"gnomon-forecast=={__version__}" in path.read_text()


def test_default_tool_count_and_claims_are_honest(monkeypatch):
    from gnomon import GnomonSession
    from gnomon.product_contract import product_claims
    monkeypatch.delenv("GNOMON_MCP_PROFILE", raising=False)
    with GnomonSession.from_config() as session:
        tools = session.tools()
        assert int(re.search(r"The agent gets (\d+) tools", README).group(1)) == len(tools)
        assert all(tool["name"] in README for tool in tools)
    claims = product_claims()
    assert claims["default_mcp_profile"] == "execution"
    assert claims["forecast_superiority"] == claims["agent_choice_lift"] == "not_established"
    assert claims["current_evidence_release"] is None
    assert "remain pending" in README


def test_obsolete_designs_and_benchmark_entrypoints_are_absent():
    for name in ("Gnomon_System_Design.md", "Gnomon_MVP_Product_Specification.md",
                 "specs/unified-plan.md", "benchmarks/run_all.py",
                 "benchmarks/workflow/agent_adapter.py"):
        assert not (REPO / name).exists(), name
    assert not list((DOCS / "design").glob("*.md"))
    assert not list(DOCS.glob("v0.*.md"))


def test_retained_tsfm_weights_remain_pinned_to_full_commit_revisions():
    from gnomon.tsfm import TSFM_REVISIONS
    assert TSFM_REVISIONS
    assert all(re.fullmatch(r"[0-9a-f]{40}", revision) for revision in TSFM_REVISIONS.values())


def test_cli_reference_documents_every_command():
    from gnomon.cli import build_parser
    parser = build_parser()
    commands = next(action.choices for action in parser._subparsers._actions
                    if hasattr(action, "choices") and action.choices)
    reference = (DOCS / "cli-reference.md").read_text()
    assert all(f"`gnomon {command}" in reference for command in commands)
    assert not {"init", "run", "share"} & commands.keys()


def test_readme_python_example_runs_without_optional_models():
    example = re.search(r"```python\n(.*?)\n```", README, re.S).group(1)
    namespace = {}
    exec(compile(example, "README.md", "exec"), namespace)
    assert namespace["execution"].result.point == (11.0, 11.0)


@pytest.mark.parametrize("source",
    [REPO / "README.md"] + sorted(DOCS.rglob("*.md")) +
    sorted((REPO / "benchmarks").rglob("*.md")) +
    sorted((REPO / "examples/provider_plugin").rglob("*.md")),
    ids=lambda p: str(p.relative_to(REPO)))
def test_relative_links_resolve(source):
    broken = [target for target in LINK.findall(source.read_text())
              if not target.startswith(("https://", "http://", "mailto:"))
              and not (source.parent / target).exists()]
    assert not broken, f"{source}: {broken}"

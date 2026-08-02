"""The docs are checked against the product, not trusted.

Every claim asserted here was wrong at some point: a tool count that drifted
two behind the toolspec, five verbs filed under a "Not currently available"
heading, an install command that fetched GitHub while the prose said it
installed the checkout, and a closed review that still read as 58 open
defects. Prose rots silently; these assertions make it rot loudly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
README = (REPO / "README.md").read_text(encoding="utf-8")
DOCS = REPO / "docs"


def _doc(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8")


# -- the counts ------------------------------------------------------------

def test_readme_states_the_real_tool_count():
    from gnomon.toolspec import TOOLS

    stated = re.search(r"The agent gets (\d+) tools", README)
    assert stated, "the README no longer states a tool count"
    assert int(stated.group(1)) == len(TOOLS), (
        f"README says {stated.group(1)} tools; toolspec has {len(TOOLS)}"
    )


def test_readme_states_the_real_tsfm_adapter_count():
    """It said four while seven shipped, so the two newest were invisible."""
    from gnomon.tsfm import TSFM_REVISIONS

    stated = re.search(r"(\w+) adapters — Chronos-Bolt", README)
    assert stated, "the README no longer states an adapter count"
    words = {"four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
    assert words.get(stated.group(1)) == len(TSFM_REVISIONS), (
        f"README says {stated.group(1)} adapters; "
        f"{len(TSFM_REVISIONS)} are pinned in tsfm.TSFM_REVISIONS"
    )


def test_every_pinned_adapter_is_pinned_to_a_full_commit_sha():
    from gnomon.tsfm import TSFM_REVISIONS

    loose = {
        model: revision for model, revision in TSFM_REVISIONS.items()
        if not re.fullmatch(r"[0-9a-f]{40}", revision)
    }
    assert not loose, f"weights not pinned to an immutable commit: {loose}"


def test_every_doc_that_counts_tools_counts_them_correctly():
    from gnomon.toolspec import TOOLS

    pattern = re.compile(r"`gnomon_\*` — (\d+) tools")
    for path in sorted(DOCS.glob("*.md")):
        found = pattern.search(path.read_text(encoding="utf-8"))
        if found:
            assert int(found.group(1)) == len(TOOLS), (
                f"{path.name} says {found.group(1)} tools; "
                f"toolspec has {len(TOOLS)}"
            )


# -- the command surface ---------------------------------------------------

def _cli_commands() -> set[str]:
    from gnomon.cli import build_parser

    parser = build_parser()
    subparsers = next(
        action for action in parser._subparsers._actions
        if hasattr(action, "choices") and action.choices
    )
    return set(subparsers.choices)


def test_cli_reference_documents_every_command():
    reference = _doc("cli-reference.md")
    missing = [
        command for command in sorted(_cli_commands())
        if f"`gnomon {command}" not in reference
    ]
    assert not missing, f"undocumented CLI commands: {missing}"


def test_the_five_verbs_are_documented_above_the_unavailable_section():
    """They were once filed *below* it, which reads as "not available"."""
    reference = _doc("cli-reference.md")
    cutoff = reference.index("## Not currently available")
    for verb in ("forecast", "investigate", "detect", "decide", "monitor"):
        position = reference.index(f"## `gnomon {verb}`")
        assert position < cutoff, (
            f"`gnomon {verb}` is documented below 'Not currently available'"
        )


def test_nothing_claims_the_unbuilt_commands_exist():
    for name in ("init", "run", "share"):
        assert f"gnomon {name} " not in README, (
            f"README references the unimplemented `gnomon {name}`"
        )
        assert name not in _cli_commands()


# -- the install path ------------------------------------------------------

def test_readme_installs_the_checkout_it_says_it_installs():
    """Bare `install.sh` fetches GitHub's default branch; only `--local`
    installs the tree you are standing in."""
    for line in README.splitlines():
        stripped = line.strip()
        if stripped.startswith("bash install.sh"):
            assert "--local" in stripped, (
                "README runs `install.sh` without --local, which installs "
                f"from GitHub rather than the checkout: {stripped!r}"
            )


# -- the artifact layout ---------------------------------------------------

def test_readme_lists_every_file_a_run_writes(tmp_path):
    from datetime import datetime, timezone

    from gnomon.ids import FixedClock
    from gnomon.runtime import forecast

    _, path = forecast(
        str(REPO / "examples" / "daily_requests.csv"),
        time_column="timestamp", target_column="requests",
        horizon=3, frequency="D", output=str(tmp_path),
        clock=FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc)),
    )
    written = {item.name for item in Path(path).iterdir()}
    # Guard the guard: an empty directory would satisfy the loop vacuously.
    assert "artifact.json" in written and len(written) >= 4, written
    layout = README[README.index("gnomon-output/forecast_<id>/"):]
    layout = layout[:layout.index("```")]
    for name in sorted(written):
        assert name in layout, f"{name} is written but not listed in the README"


# -- status, not stale defect lists ---------------------------------------

def test_the_closed_review_says_it_is_closed_before_its_findings():
    review = _doc("codebase-review-2026-08.md")
    assert "Status: closed" in review
    assert review.index("Status: closed") < review.index("## 2. Findings"), (
        "the findings table appears before the banner saying it is resolved"
    )


def test_the_v01_direction_documents_are_marked_historical():
    for name in ("Gnomon_MVP_Product_Specification.md", "Gnomon_System_Design.md"):
        text = (REPO / name).read_text(encoding="utf-8")
        assert "Historical" in text, f"{name} is not marked historical"
        assert "gnomon capabilities" in text, (
            f"{name} does not point the reader at the source of truth"
        )


# -- links resolve ---------------------------------------------------------

LINK = re.compile(r"\[[^\]]+\]\(([^)#]+)(?:#[^)]*)?\)")


@pytest.mark.parametrize(
    "source",
    [Path("README.md")] + [Path("docs") / p.name for p in sorted(DOCS.glob("*.md"))],
    ids=lambda p: str(p),
)
def test_relative_links_resolve(source):
    path = REPO / source
    broken = [
        target for target in LINK.findall(path.read_text(encoding="utf-8"))
        if not target.startswith(("http://", "https://", "mailto:"))
        and not (path.parent / target).exists()
    ]
    assert not broken, f"{source} links to missing files: {broken}"

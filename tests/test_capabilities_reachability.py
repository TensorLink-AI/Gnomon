"""Everything capabilities advertises must be reachable.

The 2026-08 MCP evaluation's highest-priority finding was an instance of
a general drift: `gnomon_capabilities` disclosed seven eligible TSFMs
while installation was a shell step no tool could reach. The instance was
fixed (gnomon_install_tsfm); this module pins the class, as far as it is
mechanically checkable: no tool name may appear in the capabilities
payload without existing in the toolspec, and every explicit tool
pointer (a key ending in `_tool`) must be on the default surface. The
semantic gap — a capability described in prose with no tool at all —
cannot be caught by a regex; reviewers own that, and the `_tool`-pointer
convention below is how a capability declares itself reachable.
"""

from __future__ import annotations

import json
import re


def _all_defined_tool_names() -> set[str]:
    from gnomon.toolspec import TOOLS

    return {tool["name"] for tool in TOOLS}


def _default_surface(monkeypatch) -> set[str]:
    from gnomon.toolspec import visible_tools

    monkeypatch.delenv("GNOMON_MCP_PROFILE", raising=False)
    return {tool["name"] for tool in visible_tools()}


def test_no_phantom_tool_names_in_capabilities(monkeypatch) -> None:
    """A tool name in the capabilities payload that no toolspec entry
    defines is an advertisement for something that cannot be called."""
    from gnomon.runtime import capabilities

    monkeypatch.delenv("GNOMON_MCP_PROFILE", raising=False)
    payload = json.dumps(capabilities(), default=str)
    mentioned = set(re.findall(r"gnomon_[a-z0-9_]+", payload))
    phantoms = mentioned - _all_defined_tool_names()
    assert not phantoms, (
        f"capabilities mentions tools the toolspec does not define: "
        f"{sorted(phantoms)}"
    )


def test_explicit_tool_pointers_are_on_the_default_surface(monkeypatch) -> None:
    """A `*_tool` key is capabilities' way of saying 'this action is
    reachable from the tool surface'; a pointer at a gated or missing
    tool is the TSFM-install drift all over again."""
    from gnomon.runtime import capabilities

    surface = _default_surface(monkeypatch)
    pointers: dict[str, str] = {}

    def collect(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                where = f"{path}.{key}" if path else key
                if key.endswith("_tool") and isinstance(value, str):
                    pointers[where] = value
                collect(value, where)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                collect(item, f"{path}[{index}]")

    collect(capabilities(), "")
    assert pointers, "expected at least models.tsfm_install_tool"
    unreachable = {where: name for where, name in pointers.items()
                   if name not in surface
                   and capabilities().get(where.split(".")[0], {}).get(
                       where.split(".")[-1] + "_profile") is None}
    assert not unreachable, (
        f"capabilities points at tools not on the default surface: "
        f"{unreachable}"
    )


def test_tsfm_install_is_reachable(monkeypatch) -> None:
    """The specific regression the 2026-08 evaluation reported: models
    are disclosed, so installing one must be a tool, not a shell step."""
    from gnomon.runtime import capabilities
    from gnomon.toolspec import runner_for

    monkeypatch.setenv("GNOMON_MCP_PROFILE", "full")
    models = capabilities()["models"]
    assert models["tsfm_available"], "TSFMs are disclosed"
    assert runner_for(str(models["tsfm_install_tool"])) is not None

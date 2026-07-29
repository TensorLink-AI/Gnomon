"""Aion plugin for Hermes Agent.

Registers four tools wrapping the Aion CLI — capabilities, inspect,
forecast, and LLM-assisted context-event proposal (host model via
``ctx.llm``, Aion-owned prompt and validation) — plus the
``aion:forecasting`` skill encoding the safe-use workflow. Aion remains
authoritative for every number; Hermes formulates the question and
explains the evidence.

Install by copying this directory to ``~/.hermes/plugins/aion`` and
enabling it (plugins are opt-in): ``hermes plugins enable aion``.
Requires the Aion CLI on PATH, or ``AION_PLUGIN_CLI`` pointing at it.
"""

from __future__ import annotations

from pathlib import Path

from .schemas import (
    AION_CAPABILITIES_SCHEMA,
    AION_FORECAST_SCHEMA,
    AION_INSPECT_SCHEMA,
    AION_PROPOSE_CONTEXT_SCHEMA,
)
from .tools import (
    check_aion_available,
    handle_aion_capabilities,
    handle_aion_forecast,
    handle_aion_inspect,
    make_propose_context_handler,
)

_TOOLS = (
    ("aion_capabilities", AION_CAPABILITIES_SCHEMA, handle_aion_capabilities, "🧭"),
    ("aion_inspect", AION_INSPECT_SCHEMA, handle_aion_inspect, "🔍"),
    ("aion_forecast", AION_FORECAST_SCHEMA, handle_aion_forecast, "📈"),
)


def register(ctx) -> None:
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="aion",
            schema=schema,
            handler=handler,
            check_fn=check_aion_available,
            description=schema["description"],
            emoji=emoji,
        )
    ctx.register_tool(
        name="aion_propose_context_events",
        toolset="aion",
        schema=AION_PROPOSE_CONTEXT_SCHEMA,
        handler=make_propose_context_handler(ctx),
        check_fn=check_aion_available,
        description=AION_PROPOSE_CONTEXT_SCHEMA["description"],
        emoji="🗓️",
    )
    ctx.register_skill(
        name="forecasting",
        path=Path(__file__).parent / "SKILL.md",
        description="Safe-use workflow for evidence-backed forecasting with Aion",
    )

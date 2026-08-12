"""Gnomon plugin for Hermes Agent.

Registers a task-oriented set of tools wrapping the Gnomon CLI, including all
five governed temporal views and LLM-assisted context-event proposal (host
model via ``ctx.llm``, Gnomon-owned prompt and validation), plus the
``gnomon:forecasting`` skill encoding the safe-use workflow. Gnomon remains
authoritative for every number; Hermes formulates the question and explains
the evidence.

Install by copying this directory to ``~/.hermes/plugins/gnomon`` and
enabling it (plugins are opt-in): ``hermes plugins enable gnomon``.
Requires the Gnomon CLI on PATH, or ``GNOMON_PLUGIN_CLI`` pointing at it.
"""

from __future__ import annotations

from pathlib import Path

from .schemas import (
    GNOMON_CAPABILITIES_SCHEMA,
    GNOMON_FORECAST_SCHEMA,
    GNOMON_INSPECT_SCHEMA,
    GNOMON_PROPOSE_CONTEXT_SCHEMA,
    GNOMON_SUBMIT_ACTUALS_SCHEMA, GNOMON_LIST_OPEN_SCHEMA,
    GNOMON_MODEL_PERFORMANCE_SCHEMA, GNOMON_RECORD_DECISION_SCHEMA,
    GNOMON_RESOLVE_DECISION_SCHEMA,
    GNOMON_COVARIATE_GUIDE_SCHEMA, GNOMON_VALIDATE_COVARIATES_SCHEMA,
    GNOMON_PROPOSE_COVARIATES_SCHEMA,
    GNOMON_INVESTIGATE_SCHEMA, GNOMON_DETECT_SCHEMA,
    GNOMON_DECIDE_SCHEMA, GNOMON_MONITOR_SCHEMA,
)
from .tools import (
    check_gnomon_available,
    handle_gnomon_capabilities,
    handle_gnomon_forecast,
    handle_gnomon_inspect,
    make_propose_context_handler,
    handle_gnomon_submit_actuals, handle_gnomon_list_open_forecasts,
    handle_gnomon_model_performance, handle_gnomon_record_decision,
    handle_gnomon_resolve_decision,
    handle_gnomon_covariate_guide, handle_gnomon_validate_covariates,
    handle_gnomon_propose_covariates,
    handle_gnomon_investigate_change, handle_gnomon_detect_anomalies,
    handle_gnomon_decide, handle_gnomon_monitor,
)

_TOOLS = (
    ("gnomon_capabilities", GNOMON_CAPABILITIES_SCHEMA, handle_gnomon_capabilities, "🧭"),
    ("gnomon_inspect", GNOMON_INSPECT_SCHEMA, handle_gnomon_inspect, "🔍"),
    ("gnomon_forecast", GNOMON_FORECAST_SCHEMA, handle_gnomon_forecast, "📈"),
    ("gnomon_covariate_guide", GNOMON_COVARIATE_GUIDE_SCHEMA, handle_gnomon_covariate_guide, "🧱"),
    ("gnomon_validate_covariates", GNOMON_VALIDATE_COVARIATES_SCHEMA, handle_gnomon_validate_covariates, "🛡️"),
    ("gnomon_propose_covariates", GNOMON_PROPOSE_COVARIATES_SCHEMA, handle_gnomon_propose_covariates, "🧪"),
    ("gnomon_submit_actuals", GNOMON_SUBMIT_ACTUALS_SCHEMA, handle_gnomon_submit_actuals, "✅"),
    ("gnomon_list_open_forecasts", GNOMON_LIST_OPEN_SCHEMA, handle_gnomon_list_open_forecasts, "⏳"),
    ("gnomon_model_performance", GNOMON_MODEL_PERFORMANCE_SCHEMA, handle_gnomon_model_performance, "📊"),
    ("gnomon_record_decision", GNOMON_RECORD_DECISION_SCHEMA, handle_gnomon_record_decision, "📝"),
    ("gnomon_resolve_decision", GNOMON_RESOLVE_DECISION_SCHEMA, handle_gnomon_resolve_decision, "🎯"),
    ("gnomon_investigate_change", GNOMON_INVESTIGATE_SCHEMA, handle_gnomon_investigate_change, "🕵️"),
    ("gnomon_detect_anomalies", GNOMON_DETECT_SCHEMA, handle_gnomon_detect_anomalies, "🔎"),
    ("gnomon_decide", GNOMON_DECIDE_SCHEMA, handle_gnomon_decide, "⚖️"),
    ("gnomon_monitor", GNOMON_MONITOR_SCHEMA, handle_gnomon_monitor, "🚨"),
)


def register(ctx) -> None:
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="gnomon",
            schema=schema,
            handler=handler,
            check_fn=check_gnomon_available,
            description=schema["description"],
            emoji=emoji,
        )
    ctx.register_tool(
        name="gnomon_propose_context_events",
        toolset="gnomon",
        schema=GNOMON_PROPOSE_CONTEXT_SCHEMA,
        handler=make_propose_context_handler(ctx),
        check_fn=check_gnomon_available,
        description=GNOMON_PROPOSE_CONTEXT_SCHEMA["description"],
        emoji="🗓️",
    )
    ctx.register_skill(
        name="forecasting",
        path=Path(__file__).parent / "SKILL.md",
        description="Safe-use workflow for evidence-backed forecasting with Gnomon",
    )

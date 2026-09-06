"""Legacy profile metadata without importing tool execution or dispatch."""

from .product_contract import resolve_mcp_profile

_CORE_PROFILE = frozenset({
    "gnomon_capabilities", "gnomon_inspect", "gnomon_describe",
    "gnomon_forecast", "gnomon_monitor",
    "gnomon_investigate_change", "gnomon_detect_anomalies",
    "gnomon_decide", "gnomon_route", "gnomon_explain_run",
})
PROFILES: dict[str, frozenset[str]] = {
    "core": _CORE_PROFILE,
    "evidence": frozenset({
        "gnomon_describe", "gnomon_forecast", "gnomon_select_scenario"}),
    "decision": _CORE_PROFILE | {
        "gnomon_decide", "gnomon_monitor", "gnomon_route",
        "gnomon_status", "gnomon_resolve_outcome",
    },
    "data": _CORE_PROFILE | {
        "gnomon_ingest", "gnomon_list_datasets", "gnomon_submit_actuals",
    },
}


def visible_tool_names(profile: str | None = None) -> list[str]:
    profile = resolve_mcp_profile(profile)
    if profile == "execution":
        from .session import GnomonSession
        with GnomonSession.from_config() as session:
            return [tool["name"] for tool in session.tools()]
    from .tool_catalog import TOOL_SCHEMAS
    return [tool["name"] for tool in TOOL_SCHEMAS
            if profile == "full" or tool["name"] in PROFILES[profile]]

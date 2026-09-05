"""Canonical identity and public claim boundary for a Gnomon build.

This module is deliberately dependency-free.  The package builder reads its
``__version__`` value, while the runtime and documentation tests consume the
same constants.  A release therefore cannot quietly present different
versions, default profiles, or evidence claims on different surfaces.
"""

from __future__ import annotations

__version__ = "0.8.0rc1"

DEFAULT_MCP_PROFILE = "execution"
LEGACY_MCP_PROFILES = ("core", "data", "decision", "evidence", "full")
CURRENT_EVIDENCE_RELEASE = None  # No completed evaluation of this release's default surface.


def resolve_mcp_profile(profile: str | None = None) -> str:
    """Resolve startup selection without importing any tool implementations."""
    import os
    selected = profile if profile is not None else os.environ.get("GNOMON_MCP_PROFILE", DEFAULT_MCP_PROFILE)
    if selected not in (DEFAULT_MCP_PROFILE, *LEGACY_MCP_PROFILES):
        raise ValueError(f"Unknown or retired MCP profile {selected!r}; choose execution or an explicit legacy profile: {', '.join(LEGACY_MCP_PROFILES)}")
    return selected


def product_claims() -> dict[str, object]:
    """Return the small claim set that buyers and agents may rely on."""
    return {
        "category": "agent_time_series_tools",
        "primary_promise": "validated_execution_with_explicit_temporal_evidence",
        "deployment_wedge": "user_selected_models_and_optional_revision_aware_evaluation",
        "offline_builtin_runtime": True,
        "default_mcp_profile": DEFAULT_MCP_PROFILE,
        "current_evidence_release": CURRENT_EVIDENCE_RELEASE,
        "forecast_superiority": "not_established",
        "agent_choice_lift": "not_established",
        "regulatory_certification": "not_claimed",
    }

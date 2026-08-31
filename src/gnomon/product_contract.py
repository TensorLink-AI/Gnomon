"""Canonical identity and public claim boundary for a Gnomon build.

This module is deliberately dependency-free.  The package builder reads its
``__version__`` value, while the runtime and documentation tests consume the
same constants.  A release therefore cannot quietly present different
versions, default profiles, or evidence claims on different surfaces.
"""

from __future__ import annotations

__version__ = "0.7.0"

DEFAULT_MCP_PROFILE = "core"
CURRENT_EVIDENCE_RELEASE = "2026-08-30-v06-external-validation"


def product_claims() -> dict[str, object]:
    """Return the small claim set that buyers and agents may rely on."""
    return {
        "category": "temporal_evidence_governance",
        "primary_promise": "weakest_evidence_authority_survives_every_surface",
        "deployment_wedge": "security_sensitive_and_regulated_agent_workflows",
        "offline_builtin_runtime": True,
        "default_mcp_profile": DEFAULT_MCP_PROFILE,
        "current_evidence_release": CURRENT_EVIDENCE_RELEASE,
        "forecast_superiority": "not_established",
        "agent_choice_lift": "not_established",
        "regulatory_certification": "not_claimed",
    }

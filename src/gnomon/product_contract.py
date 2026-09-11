"""Canonical identity and public claim boundary for a Gnomon build.

This module is deliberately dependency-free.  The package builder reads its
``__version__`` value, while the runtime and documentation tests consume the
same package version. Qualified runtime build identity (commit and source
fingerprint) is supplied separately by build_info and preserved in artifacts.
"""

from __future__ import annotations

__version__ = "1.2.0"

CURRENT_EVIDENCE_RELEASE = None  # No completed evaluation of this release's default surface.


def product_claims() -> dict[str, object]:
    """Return the small claim set that buyers and agents may rely on."""
    return {
        "category": "agent_time_series_tools",
        "primary_promise": "validated_execution_with_explicit_temporal_evidence",
        "deployment_wedge": "user_selected_models_and_optional_revision_aware_evaluation",
        "offline_builtin_runtime": True,
        "current_evidence_release": CURRENT_EVIDENCE_RELEASE,
        "forecast_superiority": "not_established",
        "agent_choice_lift": "not_established",
        "regulatory_certification": "not_claimed",
    }

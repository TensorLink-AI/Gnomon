"""Deterministic presentation projection for canonical temporal answers.

Projection is deliberately narrower than language understanding.  It may map
known semantic aliases into caller-supplied options, but it never sees labels,
future observations, or benchmark oracles and never chooses between ambiguous
options.
"""

from __future__ import annotations

from typing import Iterable


PROJECTION_VERSION = "0.1"

_CONCEPTS = {
    "higher": {"higher", "above", "increased level"},
    "lower": {"lower", "below", "decreased level"},
    "similar": {"similar", "same", "unchanged level"},
    "increased": {"increased", "increasing", "higher volatility"},
    "decreased": {"decreased", "decreasing", "lower volatility"},
    "stable": {"stable", "constant", "unchanged volatility"},
    "continued": {"continued", "fixed", "aligned"},
    "shifting": {"shifting", "shifted", "phase shift"},
    "absent": {"absent", "none", "no", "no seasonality"},
}


def project_temporal_choice(
    canonical: str | None, options: Iterable[str],
) -> dict[str, str] | None:
    """Return the unique option semantically equivalent to ``canonical``.

    Exact matching is preferred.  A projection is refused when no option or
    more than one option belongs to the canonical concept.  In particular,
    weak/uncertain are support states, not categorical answers.
    """
    if canonical is None:
        return None
    source = str(canonical).strip().lower()
    if not source or source in {"weak", "abstained", "unknown"}:
        return None
    candidates = [str(option) for option in options]
    exact = [option for option in candidates if option.strip().lower() == source]
    if len(exact) == 1:
        return {"canonical": source, "display_value": exact[0],
                "method": "exact", "version": PROJECTION_VERSION}
    # Uncertainty is a real, quotable answer when the caller explicitly
    # offers it.  It has no aliases: never turn a nearby hedge into a choice.
    if source == "uncertain":
        return None
    concept = _CONCEPTS.get(source)
    if not concept:
        return None
    matches = [option for option in candidates
               if option.strip().lower() in concept]
    if len(matches) != 1:
        return None
    return {"canonical": source, "display_value": matches[0],
            "method": "semantic_alias", "version": PROJECTION_VERSION}

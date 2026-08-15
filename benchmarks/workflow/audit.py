"""Audit whether a Workflow Bench case bundle is smoke- or publication-ready."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .provenance import corpus_sha256
from .schema import CASE_KINDS, Case, load_cases

PROFILES = {
    "smoke": {"min_cases": 5, "min_domains": 3, "min_per_kind": 1,
              "required_tags": {"forecast", "repair", "tracking", "triage"}},
    "publication": {"min_cases": 100, "min_domains": 5, "min_per_kind": 10,
                    "required_tags": {"forecast", "decision", "repair", "tracking",
                                      "triage", "ambiguity", "known-truth"}},
}


def audit(cases: list[Case], profile: str) -> dict[str, Any]:
    policy = PROFILES[profile]
    kinds = Counter(case.kind for case in cases)
    domains = Counter(case.domain for case in cases)
    tags = Counter(tag for case in cases for tag in case.tags)
    mechanisms = Counter(tag.removeprefix("mechanism:")
                         for case in cases for tag in case.tags
                         if tag.startswith("mechanism:"))
    domain_mechanisms = {
        domain: {tag.removeprefix("mechanism:")
                 for case in cases if case.domain == domain
                 for tag in case.tags if tag.startswith("mechanism:")}
        for domain in domains
    }
    oracle_signatures = {
        json.dumps({"numbers": case.oracle.numbers, "choices": case.oracle.choices,
                    "facts": case.oracle.required_facts,
                    "abstain": case.oracle.should_abstain}, sort_keys=True)
        for case in cases
    }
    input_signatures = {
        json.dumps(case.available_at_cutoff, sort_keys=True, separators=(",", ":"))
        for case in cases
    }
    repair_targets = {
        (stage.get("revealed") or {}).get("target_column")
        for case in cases for stage in case.stages if stage.get("name") == "repair"
    } - {None}
    notable_channels = {
        case.oracle.choices.get("most_notable") for case in cases
        if "most_notable" in case.oracle.choices
    } - {None}
    checks = {
        "minimum_cases": len(cases) >= policy["min_cases"],
        "minimum_domains": len(domains) >= policy["min_domains"],
        "all_case_kinds": set(kinds) == CASE_KINDS,
        "minimum_per_kind": all(kinds[kind] >= policy["min_per_kind"] for kind in CASE_KINDS),
        "required_tags": policy["required_tags"] <= set(tags),
        "has_numeric_or_choice_oracle": all(
            case.oracle.numbers or case.oracle.choices or case.oracle.should_abstain
            for case in cases
        ),
        "forecast_parity_is_required": all(
            "forecast" not in case.tags or case.oracle.requires_publish_parity
            for case in cases
        ),
        "typed_disclosures": (profile != "publication" or
                               all(not case.oracle.required_disclosures for case in cases)),
        "staged_requirements_are_sealed": all(
            (not case.oracle.requires_repair or any(s["name"] == "repair" for s in case.stages))
            and (not case.oracle.requires_tracking or any(s["name"] == "outcome" for s in case.stages))
            for case in cases
        ),
        "oracle_diversity": (profile != "publication" or len(oracle_signatures) >= 40),
        "mechanism_diversity": (profile != "publication" or len(mechanisms) >= 8),
        "within_domain_regime_diversity": (
            profile != "publication"
            or all(len(values) >= 4 for values in domain_mechanisms.values())
        ),
        "input_diversity": (profile != "publication"
                            or len(input_signatures) >= int(len(cases) * 0.8)),
        "repair_target_diversity": profile != "publication" or len(repair_targets) >= 2,
        "triage_winner_diversity": profile != "publication" or len(notable_channels) >= 3,
    }
    return {"profile": profile, "ready": all(checks.values()), "checks": checks,
            "cases": len(cases), "domains": dict(sorted(domains.items())),
            "kinds": dict(sorted(kinds.items())), "tags": dict(sorted(tags.items())),
            "mechanisms": dict(sorted(mechanisms.items())),
            "domain_mechanisms": {key: sorted(value)
                                  for key, value in sorted(domain_mechanisms.items())},
            "distinct_oracles": len(oracle_signatures),
            "distinct_inputs": len(input_signatures),
            "repair_targets": sorted(repair_targets),
            "notable_channels": sorted(notable_channels),
            "corpus_sha256": corpus_sha256(cases)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    args = parser.parse_args()
    result = audit(load_cases(Path(args.cases)), args.profile)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

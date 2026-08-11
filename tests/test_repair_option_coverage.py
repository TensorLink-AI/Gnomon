"""Every raised error code carries machine-readable repair options.

The README promises "machine-readable repair options on every error". It
was true of 24 of 55 codes, and the ones missing were the ones a first run
hits first — the whole covariate family, AMBIGUOUS_FREQUENCY, INPUT_NOT_FOUND,
EMPTY_DATASET, INVALID_JSON_ARGUMENT. `docs/troubleshooting.md` covered
several of them in prose, so the human was served where the agent was not.

This test walks the `GnomonError` call sites so the promise stays true as
codes are added, rather than being re-audited by hand.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from gnomon.contracts import REPAIR_OPTIONS

SRC = Path(__file__).resolve().parent.parent / "src" / "gnomon"

#: Codes constructed dynamically, or raised only to be caught internally,
#: where a static walk cannot see a literal. Each needs a reason.
DYNAMIC_CODES: set[str] = set()


def _raised_codes() -> dict[str, list[str]]:
    """``{code: [where it is raised]}`` across the package."""
    found: dict[str, list[str]] = {}
    for module in sorted(SRC.glob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute) else None
            )
            if name != "GnomonError" or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                found.setdefault(first.value, []).append(
                    f"{module.name}:{node.lineno}"
                )
    return found


def test_every_raised_code_has_repair_options():
    raised = _raised_codes()
    assert raised, "the AST walk found no GnomonError call sites"
    missing = {
        code: sites for code, sites in sorted(raised.items())
        if code not in REPAIR_OPTIONS and code not in DYNAMIC_CODES
    }
    assert not missing, (
        "These raised error codes have no entry in REPAIR_OPTIONS. The README "
        "promises repair options on every error:\n"
        + "\n".join(f"  {code}: {', '.join(sites)}" for code, sites in missing.items())
    )


def test_repair_options_are_well_formed():
    for code, options in REPAIR_OPTIONS.items():
        assert options, f"{code} has an empty repair-option list"
        for option in options:
            assert set(option) >= {"action", "description"}, (
                f"{code} has an option missing 'action' or 'description': {option}"
            )
            assert option["description"].strip(), f"{code} has a blank description"
            # An option that only restates the error teaches nothing.
            assert len(option["description"]) > 20, (
                f"{code}/{option['action']} description is too short to be a repair"
            )


def test_no_orphan_repair_options():
    """An entry for a code nobody raises is dead weight — or a typo.

    `TEMPORAL_LEAKAGE` is the deliberate exception: it is a violation code
    inside CLAIM_VERIFICATION_FAILED rather than a raised error, and its
    options are surfaced per-violation.
    """
    raised = set(_raised_codes())
    violation_codes = {
        "TEMPORAL_LEAKAGE", "MISSING_HISTORICAL_VINTAGES",
        "MISSING_FORECAST_VALUES",
        # Verifier violation inside CLAIM_VERIFICATION_FAILED details,
        # never a top-level error code.
        "SUB_SUPPORTED_UNLABELLED",
    }
    orphans = sorted(set(REPAIR_OPTIONS) - raised - violation_codes)
    assert not orphans, f"REPAIR_OPTIONS entries for codes nobody raises: {orphans}"


def test_claim_verification_failure_surfaces_per_violation_options():
    """L3. `TEMPORAL_LEAKAGE` had repair options and was never raised as an
    error; `CLAIM_VERIFICATION_FAILED`, which carries it, had none of its
    own for the specific violation."""
    from gnomon.contracts import GnomonError
    from gnomon.lineage import ArtifactRecord, ClaimRecord, Lineage
    from gnomon.verifier import verify_or_raise

    lineage = Lineage("task_1", {})
    lineage.artifacts.append(ArtifactRecord(
        "artifact_1", "dataset", "2026-01-01T00:00:00",
        max_known_time="2026-06-10T00:00:00+00:00",
    ))
    lineage.claims.append(ClaimRecord(
        claim_id="claim:alpha", claim_class="descriptive", statement="…",
        subject="alpha", evidence_ids=(), artifact_ids=("artifact_1",),
    ))
    with pytest.raises(GnomonError) as raised:
        verify_or_raise(lineage, as_of="2026-06-01T00:00:00+00:00")

    payload = raised.value.to_dict()["error"]
    assert payload["code"] == "CLAIM_VERIFICATION_FAILED"
    actions = {option["action"] for option in payload["repair_options"]}
    assert "review_violations" in actions
    # The violation's own options come through rather than being stranded.
    assert "remove_post_cutoff_data" in actions
    violation = payload["details"]["violations"][0]
    assert violation["code"] == "TEMPORAL_LEAKAGE"
    assert violation["repair_options"]

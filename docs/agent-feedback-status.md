# Agent feedback disposition

Reviewed against the working tree on 2026-09-09. This tracks the supplied
acceptance feedback; it is not a claim that every reported case was independently
reproduced. The implementation is included in the 1.1.6 release changes.

## Implemented in this patch

| Feedback | Result |
| --- | --- |
| Truncated gap count appears exact | Early rejection reports `gap_run_at_least` and `scan_complete: false`. Repair limits remain unchanged. |
| Routing schema says overrides must match, but fallback succeeds | Schema/help explain the default. `require_evidence` rejects all fallbacks; evidence-supported baseline/tie selection remains allowed. |
| Successful fallback is easy to misread | Added `routing_status`, `evidence_based` and a fallback warning; compact summaries retain decision status. |
| Fallback needs concrete next actions | Insufficient-fold fallback offers an evaluation template preserving task parameters, with new-call and admissibility caveats. Available studies have a cutoff-preserving retrieval call. Other reasons retain specific next-step labels; not every reason has a runnable repair. |
| Full MCP response is confused with full study evidence | Compact studies and retained receipts link directly to full saved folds. Retrieval is explicitly distinct from rescoring. |
| Initial retained receipt lacks verification metadata | Added character count, root hash, recommended page size and minimum page count. Actual byte-bounded pages may be smaller. |
| Temporal boilerplate and integer examples change intent | Month-end guidance is conditional; a supplied integer string such as `"7"` becomes illustrative integer `7`, not `1`. Runtime validation stays strict. |
| Unresolved temporal templates look executable | Added runnable flag, preserved fields and unresolved constraint list; date-only timestamp rejection now names the unsupported form. |
| Successful checks should become acceptance gates | New production regressions exercise strict/default routing, no-call/no-write fallback, full study retrieval/paging integrity, bounded counts and temporal task fidelity. Existing CI/release workflows run production tests. |

## Existing behavior retained

Earlier fixes already provide the forecast alias, schema index, cache TOML/Python
examples, unit-bearing custom-provider example, empty-registry documentation,
study-ID parameter loading, routing readiness and tie disclosure, immutable saved
coverage versus refreshed current coverage, per-item batch unit metadata, both
MCP forecast request forms, bounded malformed-row scans, snapshot override
rejection, and finite self-check wording. Existing regression tests cover these;
their existence does not establish every proposed adjacent improvement.

## Remaining feedback implemented in the follow-up

| Area | Implementation / disposition |
| --- | --- |
| Canonical cache fingerprints | Validated numeric/container/default/instant normalization, preserving large integers and calendar offsets across DST. Invalid values remain rejected before hashing. |
| Independent revision-aware rescore | CLI `evaluate --rescore`, Python `rescore`, and MCP evaluate `operation=rescore` reuse original executions/origins and save immutable revised scores. `--compare` / `compare_studies` verify unchanged originals and report changed actuals, scores, rankings and ties. |
| Rescore metadata | Recompute optional metric derivations and preserve explicit provider ordering across JSON storage. Legacy records without ordering/parent hashes disclose the verification limit. |
| Store timezone readiness | CSV ingestion accepts timezone; store errors/readiness point to ingestion into a new dataset. Ambiguous/nonexistent local timestamps require explicit offsets. |
| Empty snapshot diagnostics | Both cutoffs, selected variable and separate source/recording exclusion counts are returned. |
| Explicit versus defaulted input fields | CLI splits explicit/defaulted options; frozen-input errors identify rejected fields. Shared `error.recovery` normalizes task/context/choice/runnable metadata. |
| Effective season | Forecasts expose the period and CLI/MCP expose whether it defaulted, with period-1 guidance. |
| Configuration inspection | `capabilities --show-resolved-config` shows paths/existence/settings/entrypoints without imports, database writes or secrets. Relative-TOML-path semantics appear in schemas, Python help and the operation guide. |
| Cache/custom engine diagnostics | `capabilities --cache`, engine `cache_diagnostic()`, and execution cache/provenance metadata. Installed `providers example` and `point`/`points()` documentation. Per-call cache metadata does not defeat ledger payload deduplication. |
| Repair diagnostics | Bounded `inspect --diagnose` dry-runs all modes without source writes or forecasts. Separate budgets, bad-field/affected-row counts and safe-mode drop guidance; unmeasured stages remain explicitly null, never inferred admissible. |
| Temporal/cutoff tutorials | Executable revised-vintage workflow plus centralized cutoff descriptions in capabilities and schemas. |
| MCP discovery and examples | Interface availability, supported/negotiated protocol scope and installed stdlib client demonstrating initialization through verified pagination. No broad client-compatibility claim. |
| Completion and payload consistency | Central exit semantics, technical completion fields, optional compact error references, separate evaluation/routing status and richer retained forecast summaries. |
| Independent verification | Optional built-in calculations and evaluation error vectors/sums/denominators/pair hashes; root source fingerprints and immutable rescore parent hashes. |
| Self-check families | Explicit finite coverage for revisions, recording, folds, future-covariate shape contracts, multiple-series isolation, spring/fall DST, repairs and cached/uncached equivalence. This does not attest arbitrary panel providers or real future covariate availability. |
| Opaque exclusion reasons | File-revision guidance names unknown availability; study-cutoff errors give the recorded time and a task-dependent remedy. |

See the [operation guide](agent-operations.md), [revised-vintage tutorial](revised-vintage-workflow.md),
and [MCP tutorial](mcp-evidence-workflow.md). These examples are executable regression tests.

Compatibility choices are explicit: bare registries remain empty, configured
construction loads built-ins, route fallback remains the default with strict opt-in,
and legacy duplicate errors remain available. No semantic cutoff/unit default was
silently changed. Completion describes technical execution, not business correctness.

The acceptance reports also identify untested areas (including some ledger
operations, external providers, concurrency and broader MCP clients). These are
coverage gaps, not automatically product defects, and are not declared passed by
this patch.

## Validation

The source regression suite (`tests` and `benchmarks/tests`) passed with 1,070
tests passed and 29 skipped. Ruff, compilation and whitespace checks passed.
Both the wheel and source distribution built successfully. A fresh environment
outside the checkout passed ten installed-wheel command journeys, including the
documented revision workflow, real MCP retrieval of 2,000 forecast points with
hash verification, and 20 cases across all nine finite self-check families.
The evidence archive's offline rerun script also passed in a second fresh venv.

Pre-release validation used a local build on top of 1.1.5, identified by its
distinct dirty build ID. Release 1.1.6 runs the release workflow again against
the tagged commit and publishes its clean wheel.

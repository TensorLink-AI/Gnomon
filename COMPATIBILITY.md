# Compatibility policy

Gnomon is pre-1.0. The default agent contract is the provider-neutral execution
session. Explicit legacy profiles retain advanced evaluated workflows; they are
not interchangeable schemas. Unused experimental surfaces have no permanent aliases.
Breaking changes are called out in the changelog and release notes.

## Supported public surfaces

- **MCP:** `src/gnomon/session.py` defines default tools; `toolspec.py` retains
  explicit advanced profiles. Use `tools/list` or `gnomon capabilities` to discover
  the actual session, including operator-configured providers and ledger permissions.
- **CLI:** the commands documented in `docs/cli-reference.md` are the human,
  operator, and audit surface.
- **Python:** compatibility is promised for the documented entry points used
  by MCP, not for every internal module.
- **Docker:** packages the CLI; it is not an independent API surface.

The original v0.2 `gnomon_forecast` input schema remains a legacy-profile
registry-derivation exception. It is not the default provider request schema.
Additive response fields may still appear.

## Execution-default migration (0.8.0rc1)

Ordinary `gnomon mcp serve` now owns one GnomonSession. Inspect files into frozen
data references; describe names an exact statistic; forecast names a provider and
supplies a typed request or a reference plus horizon. Configure callables, factories,
Ephemeris and an optional ledger at startup with `--providers-config`.

Existing clients using evaluated forecast arguments (`input`, `candidates`, context,
publication modes) must explicitly select `--profile core` or another retained
legacy profile. They cannot send those arguments to the default forecast tool.
Python users of the ordinary path use GnomonSession; internal toolspec.runner_for
does not manufacture sessions or retain global default data references.

The duplicate `describe` profile and experimental `mega` profile/run/track tool
registrations are retired. Pre-cull source is recoverable at Git commit `2cba20e`.
Existing artifacts and tracking data are not deleted. Mutable legacy performance
scores no longer nominate models; use explicit cutoff-bound ledger studies.

Default MCP and Python session.call compact large results into temporary result
references. Use gnomon_read for exact JSON pages or pointers; a partial projection
is not a complete result. CLI commands and explicit compact=False Python calls
return full JSON instead of references that would expire when their session closes.

## Removed surfaces

The following unused surfaces were removed rather than kept behind feature
flags:

- The packaged Hermes plugin and its hand-copied schemas. Hermes and other
  agent hosts use the MCP server directly.
- The experimental plan compiler/executor: `gnomon plan`,
  `gnomon_compile_task`, `gnomon_validate_plan`, `gnomon_execute_plan`, and
  `gnomon_get_run`. The five governed verbs and their current MCP views are
  the execution contract.
- The v0.2 compatibility tools listed below. `GNOMON_V02_COMPAT` and
  `GNOMON_EXPERIMENTAL_PLANNER` no longer restore anything.

| Removed MCP tool | Current path |
| --- | --- |
| `gnomon_covariate_guide` | `gnomon covariates guide` for humans; MCP callers pass covariates to `gnomon_forecast` |
| `gnomon_propose_covariates` | The host proposes data; `gnomon_validate_covariates` validates and `gnomon_forecast` admits it |
| `gnomon_list_open_forecasts` | `gnomon_status` with the due section |
| `gnomon_model_performance` | `gnomon_status` with the performance section |
| `gnomon_record_decision` | `gnomon_decide` creates the governed decision artifact |
| `gnomon_resolve_decision` | `gnomon_resolve_outcome` |
| `gnomon_proposer_skill` | No public replacement; this internal telemetry did not justify an agent tool |

`gnomon_validate_covariates` and `gnomon_submit_actuals` remain available in
explicit legacy profiles. Neither belongs to the default execution surface.
The `mega` profile is removed, not an alternative measurement arm.

Ephemeris is the public provider name: `EphemerisProvider`, kind `ephemeris`,
and deployment-specific `EPHEMERIS_*` example variables. This optional connector
uses the same execution contracts as other providers. Existing ledger records
are not renamed.

Old benchmark modules, runner commands and design documents are removed from the
active tree. Git history retains them. Production APIs and existing user data are
not deleted as part of this repository cleanup.

## Rule for future surfaces

A surface ships only when it serves an audience no existing surface can and
derives its contract from the MCP registry instead of hand-copying it.

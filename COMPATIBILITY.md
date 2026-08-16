# Compatibility policy

Gnomon is pre-1.0. It maintains one agent contract—the MCP registry—and does
not carry permanent compatibility aliases for unused experimental surfaces.
Breaking changes are called out in the changelog and release notes.

## Supported public surfaces

- **MCP:** `src/gnomon/toolspec.py` is the source of truth. Use `tools/list`
  or `gnomon capabilities` to discover the installed profile.
- **CLI:** the commands documented in `docs/cli-reference.md` are the human,
  operator, and audit surface.
- **Python:** compatibility is promised for the documented entry points used
  by MCP, not for every internal module.
- **Docker:** packages the CLI; it is not an independent API surface.

The original v0.2 `gnomon_forecast` input schema remains frozen as the sole
registry-derivation exception. Additive response fields may still appear.

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

`gnomon_validate_covariates` and `gnomon_submit_actuals` remain current tools.
The `mega` MCP profile also remains available as an experimental measurement
arm; it is not the default surface.

## Rule for future surfaces

A surface ships only when it serves an audience no existing surface can and
derives its contract from the MCP registry instead of hand-copying it.

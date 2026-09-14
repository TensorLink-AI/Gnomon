# 093: enforce the numerical execution boundary

Status: local prototype, not a dispatched experiment. The 20% RMSLE objective is
unchanged and unmet. Published Gnomon 1.2.0 and DeepSeek v4.1 Flash remain the
required runtime/model for any new trial. Main and PyPI remain unchanged.

## Reason for the change

The stopped092 trial and retained030 trial allowed arbitrary Python execution
through Hermes terminal tools. Agents directly imported `numerical.predict` or
wrote fitting helpers, so counters inside `core.execute` did not measure all
model work. Some off-lab searches evaluated hundreds of configurations. Equal
nominal 60-fit limits therefore did not establish equal actual budgets.

The original task already required modelling through the lab. The new boundary
will enforce that requirement equally for all three arms. This changes the agent's
tool interface and must be disclosed prospectively; it does not salvage old runs.
No accuracy result from the stopped run is being used to set a different budget,
choose a favorable cohort, or change the ledger algorithm.

## Implemented local prototype

`execution_boundary_093.py` provides a host-owned dispatcher with no arbitrary
shell, Python, delegation, installation, or unknown-tool execution path.

- `lab` dispatches structured start/status/review/backtest/commit/sync requests to
  a fixed isolated interpreter and verified project sources. It uses the existing
  lab's fit accounting, batch reserve, exact reuse, and selection rules.
- `evidence_read` pages raw project files with lengths and hashes. All three arms
  retain their own raw data and exact evidence access.
- `data_summary` computes explicit history-window statistics and optional exact
  grouping by a visible column, with no model fitting.
- `notes_write` permits decision JSON or Markdown notes; code, recorded evidence,
  task data, budgets, and checkpoint files cannot be written through this tool.
- Traversal, symlinks, unmanifested import shadowing, and hardlinked note writes
  are rejected. Trusted lab execution verifies protected hashes before and after.
- Calls are serialized so batch dispatch cannot bypass the lab's budget lock.
  Runtime failures distinguish execution already started from pre-execution denial.

This is a tool-execution boundary, not a claim of a general OS sandbox. It assumes
host-owned runner/configuration, immutable installed runtime, fresh project roots,
and interception of every model-requested tool execution path.

## Integration and release gates still required

1. Intercept Hermes above both sequential and concurrent tool dispatch. Its
   `_invoke_tool` alone is insufficient: the sequential path can invoke inline
   executors separately. Validate raw JSON, reject duplicate/unknown fields, keep
   tool-call identities, and retain every admitted/rejected result.
2. Preserve native Hermes memory and text-based skills for every arm without
   exposing a code-execution or arbitrary-file-write path. Verify actual pinned
   Hermes behavior, not only a stub dispatcher. No silent removal of memory.
3. Publish precise schemas, example calls, and task instructions for the new tools.
   Keep model choice, configuration search, evidence interpretation, and explicit
   selection with the agent; the host must not choose a forecast.
4. Run synthetic integration tests for every arm using the pinned published
   Gnomon runtime. Reproduce known bypasses, malformed requests, mixed tool
   batches, persistence, counter exhaustion, selection reserve, and timeouts.
   Reconcile actual numerical invocations with host accounting. Local prototype
   tests alone do not prove end-to-end enforcement.
5. Freeze sources, matched budgets, model parameters, data identities, completion
   rules, and analysis before any fresh paid pilot. Maintain separate costs and
   fresh memories. No selective retries of failed092 sessions.
6. Retain the development and untouched-final gates. The final set remains closed
   until development supports proceeding and the final adapter/protocol is frozen.

The exact overall number of fits in old trials remains unknown. Pattern screening
and manual inspection found confirmed bypasses, but no-match sessions are not
proven compliant. Original artifacts and scores remain preserved and qualified.

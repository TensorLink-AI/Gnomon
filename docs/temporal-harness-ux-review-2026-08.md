# Temporal harness UX review — August 2026

Scope: the full user- and agent-facing surface of Aion as of `4aa6816` —
CLI, MCP tool surface, error/abstention payloads, artifacts, docs, and the
harness internals that back the public claims. Everything below was
verified against the working tree: live CLI runs, live `tools/list` /
`tools/call` against the MCP server, and code reading with file:line
references. Findings are ranked; the first two sections are where the
effort should go.

## Verdict

The core thesis — *the LLM proposes; Aion validates, computes, and owns
every number* — is right, differentiated, and in several places
exceptionally implemented. Abstention-as-an-answer, the structured error
envelope, disclosed repair, and the bitemporal store are genuinely better
than most commercial tooling.

The UX problem is at the seams: **the CLI is agent-shaped and the MCP
surface is CLI-shaped.** A human running `aion forecast` gets ~80 lines of
raw JSON and a `summary.md` that never contains the forecast; an agent
connecting over MCP gets file-path-only I/O, next-step hints written as CLI
command strings, and none of the guardrail prose that makes Aion safe to
operate. Each audience receives the artifact built for the other.

On top of that there is a small set of real correctness bugs — one of
which lets Aion emit a `supported` decision built on silently-invented
zeros, violating the product's own thesis — and a gap between two
README/design-doc claims (verifier, budgets) and what the code enforces.

Rating: **core engine Strong; product surface Adequate, needs work at the
seams.** Nothing here requires rethinking the architecture. Almost all of
it is finishing work on boundaries that already exist.

---

## What is done well

These are load-bearing strengths; preserve them through any refactor.

1. **Abstention UX.** A 9-observation series asked for a 14-day horizon
   returns typed reasons plus *computed* recovery: "A horizon of 7 or less
   is supportable with the current history; retry with `--horizon 7`"
   (`src/aion/support.py:66-74`, verified live). A refusal that names the
   escape hatch is the single best pattern in the product.
2. **The error envelope.** `code` / `message` / `details` (row *and*
   offending value) / `repair_options` (`src/aion/contracts.py:302-313`).
   Where it is populated, it is machine-actionable without a human.
3. **Disclosed repair.** Every fix listed, assumptive fixes flagged and
   support downgraded, caps enforced with named constants
   (`src/aion/repair.py:371-378`). `EXCESSIVE_REPAIR` refusing to guess
   past the cap is exactly the right hard stop.
4. **The bitemporal store.** `as_of` filtering happens at snapshot
   construction *and* per read (`src/aion/temporal_store.py:91-105`), the
   `known_time_assumed` disclosure on plain CSVs is honest, and the
   access-log evidence is real provenance.
5. **The four-verb tool spine.** Descriptions lead with the question the
   agent is holding and pre-disclose degraded outcomes ("or, without
   utilities, the feasible-action comparison … `conditionally_supported`"),
   so partial results don't read as failure (`src/aion/registry.py:163-224`).
   Single-sourcing schemas from `registry.MACROS` with a test enforcing it
   is the right architecture.
6. **`aion inspect` → `suggested_next`.** Resolved absolute path, inferred
   frequency, `--repair aggressive` appended only when needed
   (`src/aion/runtime.py:110-116`). Best CLI feature in the repo.
7. **The forecasting SKILL.md.** "Degraded is not failed", "preserve
   abstention, never soften to low confidence", "never hand-clean data
   yourself — it hides the repairs from the audit trail"
   (`skills/forecasting/SKILL.md:28-55`). This is precisely the failure
   taxonomy of LLM operators.
8. **Honest scoring.** Regret against the best *feasible* action in
   hindsight, `ex_ante_optimal` separated from realised outcome
   (`src/aion/decision_model.py:110-166`); adaptive-conformal log keyed by
   outcome `known_time` so replay is deterministic
   (`src/aion/tracking.py:487-555`).
9. **Zero-dependency core** (`pyproject.toml`), content-addressed IDs
   throughout, bounded `forecast_preview` (12 rows), trap-family episode
   evals, and 36 test files. The engineering culture shows.

---

## Correctness bugs — fix before any UX work

1. **Silent zero-weighting of mistyped utility keys.** Scenario names are
   fixed to `exceed`/`no_exceed` (`src/aion/macros.py:345`) but `utilities`
   is schema'd as a bare object; `operators.py:544-550` does
   `scenario_probabilities.get(scenario, 0.0)`, so
   `{"scale_up": {"above": 10, "below": -2}}` scores every action
   `expected_utility == 0.0` and returns **`support: "supported"`** with a
   winner chosen alphabetically. No warning, no reason code. This is Aion
   inventing a number. Validate keys against the closed set, return
   `invalid` on mismatch, and put the enum in the schema
   (`src/aion/registry.py:236`).
2. **MCP error handling mislabels and leaks.** Every `ValueError` /
   `FileNotFoundError` escaping a runner becomes `code: "TRACKING_ERROR"`
   with raw Python text (`src/aion/mcp_server.py:71-74`); anything else
   bypasses the envelope entirely as JSON-RPC `-32603`
   (`mcp_server.py:96-103`). Reproduced with plausible first attempts:
   `threshold={"a": 1}` → naked `TypeError`; `actions: ["scale_up"]`
   (list of strings) → `TypeError` at `operators.py:532`. Nothing should
   reach an agent outside the `AionError` envelope.
3. **`register_artifact` breaks the bitemporal guarantee and crashes on
   store inputs.** It re-reads the raw source with no `as_of` filter to
   compute `naive_error`/`cutoff_time` — the one read in the run path that
   escapes the snapshot — and for `store:<dataset>` inputs
   `load_observations` raises `INPUT_NOT_FOUND`
   (`src/aion/tracking.py:1287-1301`, `src/aion/data.py:293-295`). So
   `store:` + `--project` hard-fails, and no test covers the combination.
4. **Concurrency is unsafe.** Fixed temp-dir name `.{artifact_id}.tmp`
   that a second process `rmtree`s mid-flight
   (`src/aion/artifacts.py:24-26`), non-atomic step-cache writes
   (`src/aion/execution.py:159-162`), no locking anywhere, and
   `TemporalStore._connect` sets no timeout/WAL/rollback while
   `TrackingStore._connect` sets all three
   (`src/aion/temporal_store.py:299-307` vs `src/aion/tracking.py:392-406`).
5. **`aion_explain_run` silently overwrites per-series support
   assessments** from `triggers` after populating them from `results`
   (`src/aion/toolspec.py:498-501`) — wrong for monitor artifacts carrying
   both.

---

## Agent-facing UX — what's missing

The MCP surface is the product's main door, and it currently assumes the
agent can read the host filesystem, run CLI commands, and has read the
repo's docs. None of those hold for a remote MCP client.

1. **The guardrails never reach the agent.** The server implements only
   `initialize`/`ping`/`tools/list`/`tools/call` (`src/aion/mcp_server.py:31-75`)
   — no `resources`, no `prompts`, no `instructions`. SKILL.md — the
   "never invent numbers", "preserve abstention" rules that are the entire
   point — is unreachable from the MCP path, and covers only 4 of 20 tools
   anyway. Ship it as `instructions` on `initialize` and/or an MCP
   resource. This is the highest-leverage agent-UX fix in the repo.
2. **30 of 54 error codes have empty `repair_options`** — including the
   most likely first-contact failures (`INPUT_NOT_FOUND`,
   `INVALID_ARGUMENTS`, mistyped `horizon`) and the *entire* covariate
   error family, the most format-sensitive part of the surface. Several
   populated ones point at CLI commands (`aion store list`, `aion ingest`)
   or tools that don't exist over MCP. `retryable` is hardcoded `false`
   for all 54 codes (`src/aion/contracts.py:309`), so the field carries no
   information.
3. **Unbounded payloads.** `results` has no series cap — a 200-series
   panel is ~75k tokens in one tool result; `aion_get_artifact` returns
   the whole artifact with no `series`/`rows`/field projection
   (`src/aion/toolspec.py:456-468`); `aion_explain_run` inlines all of
   `summary.md`; `aion_capabilities` is ~2.9k tokens with no filter. Fixed
   overhead of a first session is ~8.6k tokens before any work happens.
   Meanwhile `forecast_summary` drops `baseline_improvement`,
   `strongest_baseline`, scores, and `notes` — so the agent's *first
   response contains no accuracy numbers at all* and the TSFM-absence
   disclosure never reaches an MCP agent.
4. **Tool surface bloat and overlap (20 tools, ~4.8k tokens of schema).**
   `aion_propose_covariates` is literally the same runner as
   `aion_forecast` (`src/aion/toolspec.py:281`); `aion_list_open_forecasts`
   is a strict subset of `aion_status`; `aion_resolve_decision` implements
   the bare `correct` that `aion_resolve_outcome`'s own description says
   is retired; `aion_route` is advisory prose that changes nothing.
   Cutting to ~14 returns ~5k tokens of context and removes the
   misrouting hazards. 35 of 125 parameters have no description at all —
   all six of `aion_validate_covariates`'s, all five of
   `aion_record_decision`'s; `covariate_mapping`'s string format is
   documented only on `aion_forecast`.
5. **Capabilities advertise what the flagship tool can't reach.**
   `capabilities()["features"]` reports `as_of_replay`,
   `bitemporal_store`, `strict_abstention`, `ensemble_forecasting`,
   `multivariate_var` — and `aion_forecast`'s schema exposes none of them
   (`src/aion/toolspec.py:223-241` vs `src/aion/runtime.py:144-166`).
   `store:<dataset>` is documented in other tools' schemas but not the
   flagship's. The frozen v0.2 contract has quietly become a feature
   ceiling; either extend it additively or say so.
6. **Everything is a server-local file path.** No inline-data parameter,
   no upload, no artifact-as-resource. `aion_forecast`'s description tells
   the agent to "read forecast.csv / summary.md in the returned artifact
   directory" — with a filesystem tool Aion does not provide — instead of
   pointing at `aion_get_artifact`. `aion_inspect`'s `suggested_next` is a
   CLI command string an MCP agent must translate by guesswork.
7. **Hermes and MCP are two products under one name.** Six tools exist
   only on MCP; `aion_propose_context_events` — the only producer of the
   `context_events_file` that `aion_forecast` consumes — exists only on
   Hermes, so over plain MCP that precondition is unsatisfiable. All 14
   shared tools have drifted descriptions; the only parity test checks the
   frequency enum (`tests/test_hermes_plugin.py:248-252`). Add a real
   parity test over `{name, description, properties, required}`.
8. **Memory is opt-in and easy to lose.** Registration happens only with
   `--project`; a default run leaves no registry trace, and `aion status`
   returns silent emptiness with no hint why (verified live). The registry
   at `~/.local/share/aion/registry.db` is created as an undisclosed side
   effect; `due_forecasts` degrades silently if the artifact directory was
   temp. Nothing closes the actuals loop — `due` state is computed and
   never consumed. `tracking_ids` (`forecast_id:series`) vs bare
   `forecast_id` is undocumented; `aion_record_decision` makes the agent
   invent a `decision_id` while `aion_decide` mints one.

---

## Human-facing UX — what's missing

1. **JSON is the only mode.** `aion forecast` prints ~80 lines of raw JSON
   to a TTY; ten quantile fields that are all identical on the README's
   own example read as a bug until you find the explanatory note. The
   `track leaderboard` ASCII table (`src/aion/cli.py:742-757`) proves the
   capability exists — generalise it: human table on TTY (or
   `--format table`), JSON for pipes. No color, no progress (a long
   backtest looks hung), no shell completion across 18 subcommands.
2. **`summary.md` omits the forecast.** The file humans are told to read
   first contains support status, model choice, and two long notes — but
   not one predicted value, date range, input path, horizon, or frequency
   (`src/aion/artifacts.py:57-88`). It also never renders
   `support_assessment.recovery_actions`, so the best remediation text in
   the codebase is invisible in the human-facing file. `## __default__` is
   an internal sentinel leaking into prose. Warnings and notes are
   visually identical bullets.
3. **Remediation is prose, not commands.** Errors say "Pass
   repair=aggressive" — the MCP spelling, not `--repair aggressive`, and
   never the full corrected command. `MISSING_COLUMNS` tells a CLI user to
   "Call aion_inspect" (`src/aion/contracts.py:214`). `suggested_next`
   emits a literal `<periods>` placeholder although inspect knows the
   observation count and seasonal period.
4. **Exit codes conflate abstention with success.** Hard errors exit 2;
   both a supported forecast and an `unsupported` abstention exit 0
   (verified live). Shell scripts must parse JSON to learn Aion refused.
   Give abstention its own exit code.
5. **Docs sprawl with three competing golden paths.** README pushes MCP,
   `docs/README.md` pushes CLI, `quickstart-mcp.md` is orphaned. ~290
   lines of positioning precede the install instructions. Ten unlinked
   internal memos sit beside user docs. Concrete bugs:
   `docs/data-format.md:118` says degraded mode returns `weakly_supported`
   (actual: `degraded`); the `cli-reference.md` forecast table omits eight
   real flags; a "Not currently available (v0.2)" section sits mid-file in
   a v0.4 CLI; `--ensemble` is SUPPRESS'd but documented;
   `docs/getting-started.md` hardcodes `cd /root/Aion`; `examples/` has no
   README explaining what each CSV demonstrates.

---

## Claims vs implementation

Three README/design-doc claims outrun the code. In a product whose brand
is honesty, these matter more than ordinary tech debt.

1. **"Deterministic claim verifier on every response"** — it runs on every
   response, but four of its five checks *cannot fire* on self-produced
   lineage: `CAUSAL_CAPABLE_KINDS` is an empty frozenset, calibration refs
   are satisfied by construction, `constraints_evaluated=True` is
   hardcoded, and `max_known_time > as_of` is impossible because the
   snapshot already filtered (`src/aion/verifier.py:29-97`,
   `src/aion/macros.py:389-420`). The design is good; it is a gate for
   *host-supplied* lineage — and no CLI command or MCP tool accepts
   external lineage. Either add that entry point or soften the claim.
2. **Agent budgets are aspirational.** The design doc's
   `maximum_tool_calls` / `maximum_runtime_seconds` / token budget exists
   nowhere in `src/`. `ExecutionBudget` is enforced only between planner
   steps, and nothing ever populates it — the production macro path has no
   deadline or step cap at all (`src/aion/contracts.py:80-96`,
   `src/aion/execution.py:124-139`).
3. **The planner IR has no path to its own use case.** `compile_task` only
   emits single-step plans wrapping a macro (`src/aion/plan.py:274-282`);
   `repair_plan` — the bounded LLM-repair loop — is called only from a
   test. `aion plan` is also un-gated on the CLI while gated on MCP.
4. **Observability is absent.** No stage timings, no run trace, no
   structured events; `import time` appears once. Nothing to debug a slow
   or wedged run with — notable for a harness whose design doc promises a
   run trace, and directly felt as the CLI's missing progress output.
5. Design-doc drift the other way: the shipped MCP surface is *larger*
   than the documented nine tools, under different names, and
   `capabilities()["macros"]` lists 5 of 20 tools. This is doc drift, not
   vapourware — but the doc is what a new integrator reads.

---

## Recommended sequence

1. **Correctness first** (days): utilities key validation + schema enum;
   wrap every MCP failure in the `AionError` envelope with a true code;
   fix `register_artifact` (route through the snapshot; support `store:`);
   unique temp dirs + a lock; WAL/timeout on `TemporalStore`.
2. **Make MCP self-sufficient** (the big UX win): ship SKILL.md as
   `initialize.instructions` / MCP resource; `repair_options` for the 30
   bare codes with tool-call (not CLI) remediations; `series`/`limit` on
   every unbounded response; put `baseline_improvement`, strongest
   baseline, and `notes` into `forecast_summary`; add `as_of` and
   `store:` to `aion_forecast`; cut the surface to ~14 tools; Hermes
   parity test.
3. **Give humans a human mode**: TTY table output with JSON for pipes;
   forecast table + metadata + recovery actions in `summary.md`;
   copy-pasteable corrected commands in errors; distinct abstention exit
   code; shell completion; a progress line during backtests.
4. **Close the memory loop**: register runs by default (project optional,
   not prerequisite); make `aion status` explain emptiness and name the
   registry path; `aion doctor` for on-disk state discovery.
5. **Honest docs**: one golden path per audience above the fold; move the
   ten internal memos to `docs/internal/`; fix the
   `degraded`/`weakly_supported` contradiction and the cli-reference
   gaps; align the verifier and budget claims with the code (or the code
   with the claims).

The unifying principle for all of it: **finish each surface in its own
audience's language.** Agents get tool calls, bounded tokens, in-band
guardrails, and machine remediations; humans get tables, commands they can
paste, and a summary that answers "what happens next?". The engine already
earns the trust — the seams just need to stop spending it.

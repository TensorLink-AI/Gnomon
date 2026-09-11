# Accumulated-experience workflow evaluation — protocol 1

Registered before this harness's first experiment, 2026-09-11. Development only,
on `dev/ledger-optimization`. The previous fixed-portfolio confirmation remains
closed and negative for its 20% forecast-error target. This is a new objective,
not a reinterpretation of that result.

## Objective and release bar

Demonstrate at least **20% fewer billed input + output tokens per correctly
completed evidence checkpoint** than a capable persistent SQLite control.
This measures agent work, not money: dollar cost is unknown unless billed.
Include discovery, query construction, errors, retries and failed checkpoints
in total tokens. Never measure cost only on successful runs.

All gates must pass on one preregistered confirmation:

* Token-per-correct-checkpoint ratio <= 0.80; paired 95% cluster-bootstrap upper
  bound < 1.00. At least 24 independent worlds, 24 checkpoints/world and two
  agent seeds. Resample whole worlds, retaining rounds, arms and both seeds;
  5,000 bootstrap draws, analysis seed 20260911. Seeds are not new worlds.
* Treatment exact workflow completion >= 95%; paired 95% lower bound on its
  completion difference versus control >= -0.02.
* Zero accepted task-binding, unit, source-visibility or recording-visibility
  violations. Wrong retrieval facts fail the checkpoint, even if a forecast
  executes successfully. Report the finite tested scope, not a safety proof.
* Intent-to-treat mean per-checkpoint RMSLE <= 1.02 times control; paired 95%
  upper bound <= 1.02. Failed/unexecuted selections use the same last-value
  fallback. Score this separately from exact evidence completion.
* Both audited non-agent implementations reproduce the independent oracle on
  every checkpoint. Mutations that ignore units, revisions, context, or recording
  times, and removal of accumulated history, must cause detectable failures.

No forecast-superiority claim follows from passing the efficiency bar. A 20%
forecast-error reduction remains exploratory and is not necessary for this bar.
If the capable control matches or beats Gnomon, retain that result. Do not cripple
the control, change the metric, or select favorable worlds to make Gnomon win.

## Information and tools

Each world contains multiple daily synthetic demand series with level, trend,
weekly structure and noise. Forecasts use the same three built-in recipes:
last_value, historical_mean, seasonal_naive (season 7). The demand generator
does not consult an arm, selected provider, or ledger output.

An identical append-only event stream delivers forecast requests/results,
context labels and actual revisions. Both arms get the same event receipts.
Only events recorded by the checkpoint are inserted; queries must additionally
respect explicit source and historical recording cutoffs. Future scoring truth
is host-only. Each arm has a separate persistent store and notebook. Chats reset
between checkpoints; saved queries and notes survive. No precomputed historical
score cards are injected into either prompt.

The Gnomon arm queries public `TemporalLedger.compare_context`. The control
queries normalized SQLite tables, with joins, CTEs, window functions, sqrt and
log1p. It can save/reuse parameterized SQL. A complete reference SQL query is
provided in its installed tool instructions: the comparison must beat a capable
alternative, not reward making the control rediscover bitemporal joins. Prompt
and tool-schema tokens count, including this setup cost. Both may keep notes.
Read-only SQL is enforced, with row, VM-step and output limits. No unrestricted
agent shell or access to private scoring files is exposed.

Ingestion is automated identically for both arms, including all shadow candidate
forecasts. Its time, input events and writes are reported separately. Consequently
this version tests retrieval, revision review and decision handoff, **not** the
value of choosing which outcomes to collect or manual ingestion labor. Same
forecast and selection tools, budgets and correction rules apply to both arms.

## Checkpoint task and scoring

Retrieve an exact context cohort twice: an earlier evidence vintage and the
current vintage. Report matched complete origins and mean per-origin RMSLE for
every requested versioned provider, plus the ranked providers (stable provider
input order breaks exact ties). Numeric tolerance: absolute/relative 1e-9.
Both vintages use the **same** requested origin range, ending eight days before
the current checkpoint. Only source and recording cutoffs change. Newly visible
outcomes and later revisions can change scores; newly added forecast origins
outside that range cannot explain the difference.
Select the lowest current-vintage score when at least 3 matched origins exist;
otherwise select last_value. Execute that provider for the bound current task
and submit its execution ID with both evidence answers. This fixed policy
isolates infrastructure from agent forecasting ingenuity.

Save the submitted decision/evidence immutably; re-query earlier vintages after
new revisions arrive and verify earlier answers remain unchanged. Outcome errors
are not proof of a causal business explanation. Queries span clean accumulation,
delayed outcomes, later revisions, mixed units and provider-version transitions.
Every checkpoint remains in the denominator, including cold starts and failures.

Per arm/checkpoint: at most 8 API turns, 12 tool attempts, 3 forecast attempts,
2 selection attempts, 2,048 output tokens/turn. Malformed calls consume tool
budget but not successful-provider count. No hidden fallback can pass completion.
Record full requests/responses, errors, wall time, billed usage and unknown costs.
API outages/incomplete jobs prevent a confirmation claim; retain them for audit.

## Development loop and confirmation

Development generator seeds 100–107; development-validation seeds 200–203.
Confirmation seeds 9000–9023 are reserved, not generated during development.
Families rotate by seed; do not screen worlds for favorable outcomes/headroom.
Default 24 rounds; smaller pilots are explicitly incomplete development probes.
Agent model: Engy deepseek-v4-flash-0731, temperature 0.2, requested seeds 7,19;
backend seed reproducibility unverified. Randomize paired arm dispatch order.

Loop: deterministic oracle/SQL/Gnomon parity → mutation sensitivity → small live
pilot → classify retrieval, task binding, budget and cost failures → one bounded
infrastructure or interface change → repeat on development → validate once on
200–203 → freeze code, prompts, schemas, protocol, model/budgets and cohort → one
confirmation. Keep a JSON progress report with each gate's estimate and status.
Development may show progress but can never set objective_achieved=true.

Confirmation execution is intentionally unavailable in the initial harness until
pilot validation and a separate freeze manifest/guard are implemented. Never
silently treat development seeds, synthetic test assertions, or a scripted
policy as independent live-agent confirmation.

## Interpretation and later external validation

Synthetic worlds are controllable tests of accumulated evidence handling, not
evidence of retail predictive value. After the workflow passes, preregister a
separate real-data longitudinal replication, preserving realistic availability
assumptions. Report cold (rounds 0–3) versus mature rounds, each family, revision
tasks, total tokens, tool calls, ingestion CPU, query CPU, forecast RMSLE and
failures. Report how much the prescribed memory policy beats or loses to no
memory and hindsight headroom; neither is a deployable hindsight strategy.

# Review: TemporalBench gnomon-mcp submission-robustness handoff

Date: 2026-08-11. Reviewed against `origin/main` (4ee9479, PR #53) and
the unmerged token-efficiency branch tip (ed93a3b,
`claude/gnomon-mcp-token-efficiency-uang33`) — siblings off 16ca133.
The `fix/mcp-submission-robustness` diff itself was not pushed (it
lived uncommitted in another container), so this is a design review of
the six described fixes against their verified base (ed93a3b), plus
verdicts on the two questions the handoff left open. Line-level review
of the fixes still needs that branch pushed. The two lines' adapters
differ only by ed93a3b's supersession compaction; every base-code
finding below holds on both.

## What was verified in the base code

Every mechanical claim in the handoff's root-cause chain is real in
`benchmarks/temporalbench/mcp_agent.py`:

- `_last_call` offers exactly one submit round and **discards the
  rejection payload**: it calls `self._dispatch("submit_answer", ...)`
  and ignores the returned text, so even if there were a second
  attempt, the model would never see the `problems` list. A rejected
  last-call submission falls straight through to `_abstain_outcome(cap)`.
  A double-encoded `forecast` (a JSON string where the schema says
  object) hits `if not isinstance(forecast_spec, dict)` and is exactly
  such a rejection. Mid-run rejections, by contrast, were never fatal —
  the model gets the `problems` payload as a tool result and can retry
  within the round budget. The fixes are scoped to the right place.
- `_mcp_info` whitelists trace keys (`tool`, `is_error`, `code`,
  `jail_violations`, `truncated`, `last_call`, `abstained`,
  `superseded`); the submit verdict recorded under `entry["result"]`
  (`accepted`, `problems`) is computed and then dropped before disk.
  This is precisely why the bug survived three runs unobserved.
- The truncation marker (`harness_note`) already says "the complete
  numbers are in the artifact directory on disk," and the
  `submit_answer` description already says an artifact's "trajectory is
  used verbatim" — but the two facts never meet at the point of
  confusion, which is the truncated forecast result itself.
- The MCQ arm already contains the one-repair-round philosophy the
  fixes generalize: `_McqRun._handle_submit` gives one rejection round,
  with the comment "a validator that keeps rejecting turns an answer
  the model did produce into an abstention, which is a worse lie than
  an incomplete answer." Fixes 2–3 extend an existing principle rather
  than inventing a new leniency.

## Verdicts on the six fixes

**1. `coerce_json_containers` — sound, keep, but it is the minor fix.**
It mirrors the control arm's `extract_json_object` leniency, so it
narrows an existing asymmetry rather than creating one; the restraint
(envelope plus one level, a non-parsing string left for the validator)
and the `coerced` trace record are right. It fired 0 times in the
12-row submitfix run — belt-and-suspenders, not the headline. The
handoff's unverified item stands: check whether control's downstream
consumption survives identical double-encoding before calling the
arms symmetric on this.

**2–3. `LAST_CALL_ATTEMPTS = 2` and the prose branch — endorse, with
three implementation checks once the diff is pushed.** (a) The repair
attempt must append the rejection payload as a tool message — the base
`_last_call` throws it away, and a blind retry would reproduce the same
malformed call. (b) An unparseable-arguments submit on the last call
(currently `continue` → abstain) should get the same second chance;
it is the same failure class. (c) Engine tools stay withdrawn, as
stated. The trace records (`submit_rejected`, `last_call_repair`) are
covered by fix 6.

**5. `MAX_ROUNDS` 10 → 30 — supported by the data, two interactions to
watch.** Rounds was the binding cap (11 of 12 baseline voids were
`cap:rounds`; raising it alone took 1→4). But: (a) `MAX_MCP_CALLS = 24`
is unchanged, so at 30 rounds a re-inspecting model can spend the tool
budget and burn remaining rounds on `TOOL_BUDGET_SPENT` echoes —
bounded, but worth watching in traces; (b) `MAX_RUN_TOKENS = 500_000`
is now the binding cap — mcp-combined's running mean (~535k/row) is
*above* it, so expect `cap:tokens` voids. Do not raise the token cap
reflexively: the truncation remedy and surface slimming attack the same
re-inspection spiral at its cause. The file's convention is caps with
measured justifications in their docstrings; `MAX_ROUNDS` should get
one citing these runs.

**6. Widened `_mcp_info` whitelist — the most important fix in the
set.** The bug class here is "verdict computed, never persisted," and
it cost three runs of misdiagnosis. Consider inverting the design:
persist all small scalar trace fields by default and blacklist bulk,
so the next new field cannot silently vanish the way `result` did.

## Open question 1: pruning the tool surface for T2/T4

**The dilemma as posed is a false binary, because the engine already
ships the answer as a product feature.** `gnomon mcp serve --profile
core|decision|data|full` (`toolspec.py`, "task profiles: named subsets
of the default surface for hosts that know what kind of session they
are running"; the active profile is reported by `gnomon_capabilities`).
Present on both lines. Measured: on ed93a3b — the base the runs used —
the full profile is 17 tools / 36,242 chars of specs (≈ the measured
9,488 tokens); `core` (capabilities, inspect, forecast,
investigate_change, detect_anomalies, get_artifact, explain_run) is
7 tools / 17,087 chars. On origin/main: 18 tools / 44,932 vs the same
core at 21,323. Either way the cut is ~50–53% of the schema overhead —
most of the ~90k tokens/row the handoff identified.

That changes the integrity analysis. The adapter's rule — the model
holds every tool the server publishes, verbatim — is about the
*adapter* not editing the published list, and it survives intact: with
`--profile core`, serve publishes seven tools and the adapter passes
seven verbatim. Selecting a documented first-party deployment mode is a
condition definition, not harness tampering, in exactly the way
`temperature=0` or the model choice is.

The recommendation, threading the stated rule's letter and spirit:

- **Do not change the existing `gnomon-mcp` condition.** It stays
  full-surface, comparable with every number already collected.
- **Add a labeled sibling condition (`gnomon-mcp-core`) that runs serve
  with the core profile — the same profile for every tier.** Uniformity
  across tiers is what keeps this honest: nothing is "pruned to fit the
  tier," so the T1/T3 measurement is not settled by construction. The
  tools that are live alternatives for T1/T3 — inspect,
  detect_anomalies, the ones `SYSTEM_MCQ` itself names — remain. What
  disappears are tools whose preconditions cannot exist inside the row
  jail: decide/monitor/route/status/resolve_outcome need tracking
  projects and open decisions; ingest/list_datasets/submit_actuals need
  a registry; preflight_context/validate_covariates need covariates the
  rows do not carry. Removing them removes no alternative answer path
  on any tier.
- The delta between the two conditions is itself a publishable
  measurement: what the surface's weight costs in task completion. If
  core beats full, that is a product finding (hosts should set
  profiles), not a benchmark artifact.
- The cheaper, integrity-free saving exists regardless: the
  `gnomon_forecast` spec alone is 8.3k chars. Slimming descriptions and
  schemas product-side helps every host and every arm and raises no
  question at all. Worth doing first.

One caution: the observed failure spiral (34 `gnomon_inspect` calls
rebuilding truncated numbers) used an *applicable* tool. Pruning would
have cut cost, not the failure mode. The remedy (fix 4) is the fix for
the spiral; the profile is the fix for the cost.

## Open question 2: is TRUNCATION_REMEDY legitimate?

**Yes, with wording constraints.** Three tests it passes:

- *Whose confusion is it?* `MAX_TOOL_RESULT_CHARS` is harness-authored;
  neither the engine nor the task truncates anything at that layer. A
  harness that creates a false impression ("your numbers are gone")
  owns the disclosure that corrects it. Declining to correct it does
  not preserve neutrality — it preserves misinformation the harness
  introduced.
- *Is the content new?* No. Both facts are already on the surface
  (`harness_note`: complete numbers on disk; `submit_answer` spec:
  trajectory used verbatim). The remedy co-locates them at the point of
  confusion. Nothing about the task, and nothing about whether to use
  the engine, is added.
- *Is it route-neutral?* It must stay so. Safe form: "the artifact
  holds the complete trajectory; submitting its `artifact_path` uses
  those values verbatim." Unsafe forms: anything comparative or
  directive ("prefer the artifact," "you don't need your own values"),
  and anything conditional on the model's apparent intent — the marker
  must be identical on every truncated forecast result, including ones
  the model plans to ignore.

The tilt-test that settles it: attached to a result the model intends
to ignore, the note reads as a statement of harness mechanics, not an
argument. It passes.

## The "where I was wrong" items — checked

1. **Rounds before token efficiency — the correction is right, with a
   sequencing nuance.** They are not opposed but ordered: once rounds
   unbind (10→30), tokens becomes the binding cap (mcp-combined mean
   ~535k vs the 500k cap), and the token work matters again. The token
   branch's 1→0 at 10 rounds also cannot carry much weight at n=12 —
   see item 3.
2. **Over-generalizing double-encoding — agreed.** Coercion fired 0/12
   while the repair round fired 11/12. The generalizing fix is the
   class-catcher (any last-call rejection gets one informed retry);
   the instance fix stays as recorded leniency.
3. **Variance — agreed, and it bounds the conclusions.** With n=12 and
   per-request miner routing at temperature 0, the 0/1/3/4 spread
   between configs is within noise; ordering claims need paired rows
   and n≥36 or so. What does *not* depend on n: the mechanism findings
   (answered rows voided by an unretryable last call; rejection
   reasons dropped before disk) are transcript facts, and the 1/12 vs
   12/12 control gap is structural, not sampling error.

## The spo2 silent drop — confirmed, live on origin/main

The handoff undersells this one: it is not merely "worth reviewing on
its own," it is a live contract violation on the current default
branch. Root cause verified, and it is server-side, not in the
adapter: `enforce_response_budget`/`_trim_bulk` in
`src/gnomon/toolspec.py` protects disclosures **by key** — a value
under `support`, `warnings`, etc. is never trimmed — but a long list
is trimmed **by element**. A batched six-channel forecast's `results`
list (6 > `_TRIM_HEAD + _TRIM_TAIL` = 5) is cut to
`node[:3] + node[-2:]`, deleting `results[3]` wholesale, support state
and all; `results` is not itself a protected key, so the key-based
guard never sees the loss. That is exactly "spo2 silently dropped,
support state included."

The unmerged token-efficiency branch (ed93a3b) fixes it with
`_holds_protected`: a long list whose elements carry protected keys is
descended — each element's bulk still trimmable — never cut. The
adapter's own `_droppable`/`_holds_disclosure` (already on main)
implements the identical guard one layer up, which is the irony: the
harness would refuse to drop what the engine itself drops before the
harness ever sees it. For a project whose thesis is "disclosed, never
silently dropped," the `_holds_protected` guard should reach main
ahead of, and independent of, any benchmark follow-up — merge the
branch or cherry-pick the guard with its test.

## What still needs to happen

1. Land ed93a3b's `_holds_protected` trim guard on main (merge the
   token-efficiency branch or cherry-pick it) — a contract violation
   in the product outranks the benchmark work.
2. Push `fix/mcp-submission-robustness` — the six fixes reviewed here
   only at design level need line review, especially the three
   implementation checks on the last-call repair round.
3. Verify control's tolerance of double-encoded output before claiming
   the coercion restored symmetry.
4. Slim the tool specs product-side (`gnomon_forecast` first); then, if
   the cost question stays live, add `gnomon-mcp-core` as a labeled
   condition rather than altering the existing arm.
5. Re-run the winning configuration at a sample size that can rank
   configurations before treating any ordering as settled.

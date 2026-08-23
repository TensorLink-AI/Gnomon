# Gnomon goals assessment — 2026-08-23

**Scope:** full-tree review of `main`-equivalent branch at commit `35d9611`
against the stated product goals: a trusted temporal-reasoning layer that
prevents future leakage, unsupported claims, and silent model failures, and
supplies compact evidence, honest uncertainty, immutable receipts, and
actionable next steps, using strong forecasting models only where they have
demonstrated value. Literature context taken from the TMLR survey
*A Survey of Reasoning and Agentic Systems in Time Series with LLMs*
(arXiv:2509.11575) and the benchmarks it curates.

**Method:** six parallel deep audits (leakage layer; evaluation/selection/
calibration; verifier and LLM boundary; artifacts/lineage/tracking; MCP
surface and repair; TSFM tier and benchmark claims), each reading the
relevant modules and tests in full and attempting bypasses, plus a full test
run (`1784 passed, 7 skipped` in 128s) and direct reproduction of the two
highest-impact findings.

---

## Verdict

**The goals are substantially achieved — more so than for any comparable
system in the surveyed literature — but two of the three "prevents" claims
have concrete, demonstrated breaches, and the word "immutable" describes a
convention rather than an enforcement.** The architecture is right: no LLM
call exists in the runtime, every published statement is a deterministic
template, model selection genuinely cannot skip the baselines, abstention is
first-class, and the project's own benchmark claims are backed by retained
artifacts including headline *null* results. The gaps are specific and
fixable, not architectural.

| Goal | Verdict |
| --- | --- |
| Prevents future leakage | **Mostly** — structural and well-tested, with one reproduced hole (repair × `as_of`) that also falsifies the artifact's cleanliness proof |
| Prevents unsupported claims | **Mostly** — the LLM boundary is airtight; the claim verifier covers 6 of ~19 number-bearing surfaces and one gate is unreachable on decide/monitor |
| Prevents silent model failures | **Largely** — disclosure discipline is real; two name-collision bugs can silently disqualify TSFM candidates, and two config keys are accepted but inert |
| Compact evidence | **Achieved** — budgeted, protected-subtree trimming, measured token economics |
| Honest uncertainty | **Substantially** — real conformal machinery; default residual pooling and a fixed 80% measurement band are quietly less than the README's diagram |
| Immutable receipts | **Partially** — atomic first-write-wins, deterministic IDs, typed lineage; no content hashing, several registry upserts, no replay-and-diff |
| Actionable next steps | **Strong design, one broken flagship path** — typed repair options with exact retry calls, but the retry is dead for 5 of the tools that emit it |
| Strong models only where demonstrated value | **Achieved** — identical folds, enforced transfer-prior gating, no auto-promotion, honest measured evidence |

---

## Finding 1 (HIGH): aggressive repair leaks the future past the snapshot, and the artifact certifies the run clean

`load_stage` transforms the *full file* — `regrid_observations` then
`repair_observations` — before wrapping the result in the snapshot and
filtering at `as_of` (`src/gnomon/pipeline.py:156-176`). Interior-gap
interpolation (`src/gnomon/repair.py:510-541`) fills a slot from its left
*and right* neighbours; when the right neighbour lies after `as_of`, a
pre-cutoff slot embeds a post-cutoff observation, is stamped
`known_time = slot`, and is served as training data.

Reproduced: a 40-day series with a post-gap jump to ~5150, `--repair
aggressive`, `as_of` at the gap — the visible tail contains ~2648 (midpoint
with the future value) and the first forecast is ~2754 instead of ~148,
**while `snapshot_access` reports `max_known_time == as_of` and the verifier
passes**. The artifact's proof is false exactly where it matters most; the
leakage self-check (`selfcheck.py`) exercises the store path without repair,
and no test covers repair × `as_of`. Lesser variants: `month_start` regrid
restamps timestamps backwards across a cutoff; aggressive timestamp snapping
rounds backwards.

This is the one place where "leakage is structural, not behavioural"
(README) is currently false. Everything else in that claim held up under
bypass attempts: `Snapshot.__init__` filters at construction and clamps
caller cutoffs (`temporal_store.py:99-113`), covariates are structurally
re-bound at the run's `as_of` (`covariates.py:49-85`), TSFM adapters see
only fold-sliced value lists, evaluation folds re-read the series as known
at each fold cutoff (`pipeline.py:341-375`), the AST leakage lint bans raw
reads in execution modules, and the tests genuinely attempt bypasses
(explicit later cutoffs, timezone-string ordering leaks, vintage reverts,
cached context refs replayed at earlier `as_of`).

**Fix directions:** filter at `as_of` before repair; or refuse
`repair=aggressive`/`regrid` combined with `as_of`; or stamp derived points
with `known_time = max(contributing known_times)` (the most honest option —
the snapshot then excludes them naturally). Add a repair × `as_of` trap to
the episode suite.

## Finding 2 (HIGH for the agent surface): the flagship machine-actionable repair option is dead on 5 of the tools that emit it

`runner_for` attaches `retry_with_aggressive_repair` with an exact
`tool_call` on any `IRREGULAR_TIME_GRID` (`toolspec.py:3295-3313`), but only
`gnomon_forecast`, `gnomon_describe`, `gnomon_preflight_context`, and
multi-forecast plumb `repair` through. `gnomon_inspect`,
`gnomon_investigate_change`, `gnomon_detect_anomalies`, `gnomon_decide`,
`gnomon_monitor`, and `gnomon_run` drop the argument silently — executing
the suggested follow-up verbatim reproduces the identical error. An agent
obediently following the machine-readable contract loops. Related:
`investigate_change` and `detect_anomalies` run `load_stage` with repair
`"off"` while the rest of the surface defaults to `safe`
(`macros.py:203-207,897-901` vs `pipeline.py:132`), and `gnomon_describe`
honors `repair` without declaring it in its schema.

## Finding 3 (MEDIUM): the verifier's coverage is narrower than "before any response leaves the process"

The claim verifier is real, deterministic, and correctly ordered before
persistence on six verbs (`runtime.py:1061,1406`;
`macros.py:434,658,857,1006`). But:

- `gnomon_describe`, `gnomon_inspect`, `gnomon_track`/`gnomon_status`, and
  macro-attached temporal answers publish numbers (including fitted
  direction probabilities) with no lineage and no verifier pass.
- The miscalibration gate is structurally unreachable on decide/monitor:
  their `rolling_evaluation` evidence carries only
  `{selection_scores, test_scores}` (`macros.py:617-620,835-838`) —
  `measured_interval_coverage` is only ever attached by the forecast verb
  (`runtime.py:772,804`). A run whose measured coverage is out of band has
  its forecast claim downgraded, yet decide/monitor re-publish the same
  exceedance probabilities under a fresh predictive claim that verifies.
  This is the same existence-vs-quality class of hole the project already
  fixed once for the forecast verb (`contracts.py:60-84`).
- The verifier audits what the builder declares, in-process
  (`constraints_evaluated=True` is asserted, not checked; payload numerics
  are never reconciled against evidence). Right for the actual threat model
  — the LLM lives outside the process — but weaker than the README reads.

The parts that make "unsupported claims" genuinely impossible are upstream
of the verifier and are the strongest code in the repository: no LLM
adapter exists in `src/` (`llm.py` is a protocol plus a raising null
adapter), all published statements are deterministic templates, context
events are verbatim-quote-grounded with magnitude spellings stripped, the
`future_events` lane is off by default and priced as `context_trusted`, and
`PARAMETER_AUTHORITY` classifies every caller parameter with CI-enforced
completeness.

## Finding 4 (MEDIUM): "immutable" is atomicity plus convention, not enforcement

- Artifact IDs hash *inputs* (runtime version + task recipe), never output
  bytes; no digest of `artifact.json`/`evidence.jsonl`/`lineage.json` is
  stored; `read_artifact` verifies only `schema_version`; lineage has no
  hash chain. Post-hoc edits are undetectable, and scoring reads
  `forecast.csv`/`evidence.jsonl` from disk on trust —
  `gnomon_explain_run` then quotes the file as "verified".
- Several registry writes are upserts under the immutability banner:
  `register` overwrites model/support/artifact_path on conflict
  (`tracking.py:774-797`), `record_coverage_outcome` is INSERT OR REPLACE
  (`:1029-1033`), `record_decision` can rewrite action/expected outcome
  (`:2279-2281`).
- There is no artifact-driven replay-and-diff command; "replay" means
  deterministic re-execution from original inputs (which is genuinely
  strong: pinned clock, content addressing, byte-identical goldens,
  cross-surface numeric equivalence tests).
- `write_artifact` has a shared-tmp TOCTOU race between concurrent
  same-id writers (`artifacts.py:36-43`).

What *is* achieved: temporal-answer receipts are INSERT OR IGNORE with
content-conflict refusal, outcome ingestion is bitemporally hygienic
(known-before-valid refused, derived scores stamped with the latest
contributing knowledge time), and decision regret is implemented exactly as
claimed — regret vs the best *feasible* action in hindsight, ex-ante
optimality recorded separately, legacy records refusing to invent regret.

## Finding 5 (MEDIUM): honest-uncertainty fine print diverges from the top-level docs

- The default calibration **pools selection-fold residuals** into the
  calibration set (`evaluation.py:1789-1843`, docstring: "Pooling is not
  split-conformal, and the direction of the error is known"). The artifact
  discloses it per-run; the README/concepts fold diagram ("separate
  calibration fold") does not. True split conformal is one config flag away
  (`pool_residuals: false`).
- Test-fold coverage is always measured on the default 80% band even when
  `target_coverage` is configured differently (`evaluation.py:1850-1851` vs
  `pipeline.py:1454-1457`): with `0.95` configured, the disclosed coverage,
  the `< 0.7` warning, and the verifier band all describe an interval
  nobody is shown.
- The headline "beat the strongest baseline by X%" is the winner's own
  selection-fold margin — winner's-curse-inflated, with no shrinkage and no
  flag when the untouched test fold contradicts it.
- The selection gate is stricter than advertised (margin + win-rate +
  median-gain + fold-local scaled-error double gate; a single disjoint fold
  escalates the margin to 75%, empirically justified in
  `results/short-history-guardrail/`) but is heuristic point comparison
  over 1–10 folds — no significance testing, no multiplicity control across
  ~7 built-ins + up to 8 TSFMs + ensemble + meta-model.
- "Five-state support assessments" (README) is stale: `contracts.py:15` now
  carries seven support values over three publication tiers.

## Finding 6 (MEDIUM): silent-failure protections have three silent failures of their own

- In-process Chronos adapters share the class-level name `"chronos_bolt"`
  (`tsfm.py:409-421`): requesting both variants collapses the fold dicts and
  **silently disqualifies both**, with a nonsensical "completed 2n of n
  folds" note. Same collision if a sandbox and an enabled API provider share
  a model name (`evaluation.py:1014-1024`); the arbitration function
  `resolve_tsfm_backend` exists and is admitted dead code
  (`docs/design/news-regime.md:616-620`).
- `models.tsfm.overrides.<name>.backend` is allow-listed, documented in
  `gnomon.toml.example`, parsed — and read by nothing. `backend = "skip"`
  silently does nothing. This is exactly the defect class the `INERT_KEYS`
  machinery exists to refuse.
- The MCP boundary performs no input-schema validation: unknown arguments
  are silently ignored (the anti-pattern the config layer rejects with
  `UNSUPPORTED_CONFIG_KEY`), and enum/pattern constraints are advisory.
- Latent meta-model fold misalignment: the LOFO loop skips its forecast
  placeholder on empty weights (`evaluation.py:1325-1327`) — currently
  unexploitable, but the exact bug class fixed for built-ins.

## Smaller items worth queueing

- Enrichment ablation folds (covariate/context admission) score against
  as-of-vintage target values, not fold-cutoff vintages — revision-aware
  fold hygiene covers model selection but not the admission gates.
- Context-event `known_at` is caller-attested; a "verifiable source" is
  never dereferenced. Disclosed in docstrings; not in the README's
  "structural, not behavioural" framing.
- Realised scoring stores MASE/WAPE/coverage/hit-rate but no proper
  scoring rule (pinball/CRPS exist only at selection time); `brier` is a
  permanently-NULL column.
- Text-level assumptive repairs (date order, thousands separator, timezone,
  encoding) are warned and support-downgrading but **uncapped**, unlike grid
  repairs (30%) and dropped rows (5%).
- Default `evidence` profile emits recovery guidance naming tools the
  profile hides (`gnomon_inspect`, `gnomon_get_artifact`) and some MCP
  repair options speak CLI (`--repair aggressive`, `gnomon store list`).
- Sandbox pinning covers weights (Hub SHAs, fail-loud) and top-level
  packages, not transitive deps; one spec pins a git *tag*.
- The 66-fixed/60-broken choice McNemar (p = 0.656) in the README has no
  in-repo artifact — the only headline number that is currently
  take-our-word-for-it, in a project whose stated rule is "measured or
  absent".
- Ranked `leaderboard` uses plain AVG with no `as_of` filter — the code's
  own comment calls this a defect; the as-of replay guarantee doesn't reach
  the surface users rank models on.
- No filesystem jail on MCP inputs/outputs: any readable file with a
  supported suffix, arbitrary `output_dir`/`store_path`. Disclosed as
  host responsibility (`runtime.py:1586-1590`); worth stating in the MCP
  quickstart's security section.

## What holds up under adversarial reading

- **The LLM boundary.** No runtime LLM call exists to misbehave; proposals
  are schema-bound and deterministically revalidated; quotes are verified
  verbatim; numbers are re-parsed from quoted spans. "The LLM cannot
  silently edit or invent published numbers" is true as architecture, not
  aspiration.
- **Mandatory baselines.** Unremovable in `active_models`, negative margins
  refused at load, abstention when no baseline completes every fold, and a
  double stability gate on top of the margin.
- **Executable-candidate identity.** The evaluated spec (strategy, members,
  config, revisions, fallback policy, fitted weights, data fingerprint) is
  the thing that publishes; a TSFM final-fit failure swaps points and
  residuals *together*, enforced by a global residual-provenance assertion.
- **Repair disclosure.** No repair path mutates data without a `RepairLog`
  entry — including re-sorting; caps raise typed errors; assumptive fixes
  downgrade support before assessment.
- **Compactness.** 8 KB budgets with protected epistemic subtrees, brief
  default, triage blocks with artifact pointers, and measured
  tokens-per-row in retained benchmark artifacts.
- **Benchmark candor.** Every other README number reproduces from
  `results/benchmark-releases/2026-08-23-product-hardening/` and
  `results/leaktrap/`; the retired claim appears nowhere; null results are
  the headline; the fold-starved-channel result is framed as safety, not
  skill — which is exactly the posture the survey literature
  (NeurIPS 2024 "Are LLMs Actually Useful for TSF?", the context-parroting
  baseline, "Time Travel is Cheating") says the field needs.

## Position against the literature

In the survey's taxonomy, Gnomon is the deterministic tool-execution actor
(T-Tool with deterministic T-Ver) that the branch-structured agent systems
lack: verification by code rather than self-critique. No surveyed system
offers structural leakage prevention, mandated naive baselines, or typed
abstention; the leakage-trap result (0/40 vs 13/35, McNemar p = 0.00024)
is the right experiment and is artifact-backed. The honest gap the
measured evidence exposes is *uplift*: the reasoning-accuracy and forecast
deltas are null so far, so the current defensible pitch is exactly what the
docs say — a safety and audit layer, not yet an accuracy layer. The
literature's benchmark direction (TemporalBench, CiK, AnomLLM, MTBench —
all adapted in `benchmarks/`) is the right place to keep pressure.

## Priority order

1. Close the repair × `as_of` leak (Finding 1) and add it to the trap suite.
2. Plumb `repair` through the five broken tools or stop emitting the retry
   option there (Finding 2); unify macro repair defaults.
3. Attach `measured_interval_coverage` to decide/monitor evidence
   (Finding 3).
4. Fix the adapter name collisions and delete or wire the inert
   `backend` override (Finding 6).
5. Measure coverage at the configured level (Finding 5).
6. Either hash artifact outputs on write and verify on read, or stop using
   the word "immutable" (Finding 4).
7. Doc corrections: pooling vs split conformal in README/concepts;
   "five-state" → current contract; retire or re-derive the unbacked
   McNemar figure.

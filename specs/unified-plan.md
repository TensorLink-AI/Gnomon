# Gnomon unified plan — one trusted answer, cheaply

Status: adopted direction, 2026-08-12. Consolidates the three proposals
(`agent-surface-redesign.md`, `trusted-temporal-runtime-proposal.md`,
`trusted-temporal-runtime-proposal_v1.md`) into one sequenced plan.
Every defect cited below was re-verified against `main` at `4ee9479`
on 2026-08-12; stale claims from the inputs are listed at the end.

## The problem, stated as arithmetic

A simple forecast task costs 250–500K conversation tokens. The cost is
multiplicative: every turn re-pays the tool schemas (~11.2K tokens for
the 18-tool default surface, measured 44,950 bytes) plus the entire
history of prior tool results and call arguments. Six calls where one
would do multiplies everything by six. The measured 12-round
six-channel task re-sent a ~337K-token prefix.

Priority order follows leverage:

1. **Call count** (multiplies every other term) — one-shot answers,
   refusals that convert to answers in one round, no optional-arg
   error loops.
2. **Schema tax** (paid every turn, called or not) — smaller default
   surface, dieted schemas, CI byte budget.
3. **Response and history size** — compact envelopes, triage for
   multi-series, superseded-result compaction, `data_ref` instead of
   re-inlined data.

**Target (from the surface redesign, adopted):** ≤ 50K cumulative
conversation tokens per task; ≤ 2 tool calls median, ≤ 4 p95; quotable
tiered answer on ≥ 80% of readable inputs with accuracy reported split
by tier; leaktrap 0/40.

The yield target must not collapse support states. Evaluated, limited,
best-effort, and abstained outcomes remain separate in reporting; an
unsupported answer cannot improve yield merely by being phrased more
confidently.

Trust is the precondition, not a competing goal: none of the token work
is worth shipping on an engine whose published numbers can differ from
its evaluated ones. Hence Phase 1 before the surface flip.

The engineering program runs beside a commercial-validation program.
Making Gnomon trustworthy and cheap proves that the product can work; it
does not prove that a local user will make it habitual or permit paid
hosted inference. The adoption loop must therefore be measured from the
first external install, not inferred later from download counts.

---

## Parallel track — Validate the adoption and revenue loop

The initial job is deliberately narrower than the long-run temporal
infrastructure category:

> Help an SRE or platform engineer identify which operational metric may
> breach a meaningful limit, when, and whether the evidence supports an
> intervention.

This job fits the Hermes beachhead and exercises Gnomon's forecast,
threshold, investigation, monitor, decision-cost, tracking, and
bitemporal strengths. Hermes is the first distribution channel; SRE and
platform capacity risk is the first job to be done.

Instrument the complete funnel:

```text
install
  -> inspect non-demo data
  -> receive a quotable answer
  -> return for another run
  -> submit actuals
  -> record an intervention
  -> accept an explicit Ephemeris comparison
  -> make and retain a paid call
```

Report each transition by support tier. A best-effort orientation is not
counted as an evaluated answer, and an abstention that leads to a
successful repaired call is distinguished from a dead-ended refusal.

Initial thresholds must be filled before launch. The starting proposal:

- 20 external installs that reach real operational data;
- 10 useful first answers;
- 6 users returning for a second run within four weeks;
- 4 submitting actuals or otherwise scoring the prior answer;
- 3 accepting an explicit Ephemeris comparison;
- 2 retaining paid use or entering a design-partner commitment.

These are diagnostic thresholds, not market-size evidence. Interview
users who stop at every transition. Test the central commercial
assumption independently: local-first users must opt into sending
operational data to Ephemeris and find routing valuable enough to keep
paying.

Hosted inference is never an undisclosed default. The progression is:

```text
local baseline and classical contest
  -> disclose whether hosted candidates may improve the result
  -> obtain explicit user or project-policy authority
  -> call Ephemeris
```

*Done when:* funnel instrumentation ships with the public integration;
the cohort thresholds have owners and dates; every stopped user has a
recorded reason; installer-to-paid conversion is reported as a funnel,
not a single aggregate percentage.

---

## Phase 0 — Land what is already built (days)

Merge `fix/mcp-submission-robustness`. It already delivers a third of
the token work:

- superseded-tool-result compaction in the benchmark agent loops
  (measured: ~337K → ~214K re-sent prefix on the MIMIC row);
- `gnomon_capabilities` brief-by-default (17,645 → ~6,000 chars);
- `gnomon_forecast` schema diet (10,933 → 8,434 chars);
- `gnomon_inspect` multi-column / `auto` instead of the
  `AMBIGUOUS_SCHEMA` refusal (kills a whole retry loop for read-only
  inspection);
- the response trimmer descending into lists (six-channel results no
  longer lose channel disclosures).

Then **re-baseline the MCP evaluation at HEAD** so all later
improvements are honestly attributed (several of the evaluation's
frictions were fixed after it ran).

*Done when:* branch merged; evaluation re-run recorded with per-task
cumulative tokens, calls (median/p95), yield, accuracy split by tier.

## Phase 1 — Engine integrity (immediate, parallel with Phase 2 prep)

The published number must be produced by the exact candidate
specification that earned publication. Each evaluation origin fits an
independent instance using only the history permitted at that origin;
after selection, the final fitted instance crosses into publication
without being reconstructed from a model name.

```text
CandidateSpec
  identity: strategy, eligible members, configuration, revisions,
            fallback policy
  fit(permitted_history) -> FittedCandidate

FittedCandidate
  predict(horizon) -> ForecastPath
```

Reusing one fitted object across selection, calibration, and test would
leak information between partitions. The invariant is one immutable
specification and fitting procedure, with an independent fitted instance
at every origin. The defects below were verified live at `4ee9479`, but
they land as independently reviewable changes rather than one large PR.

### Phase 1A — Candidate execution boundary

- Introduce `CandidateSpec.fit(history) -> FittedCandidate` across the
  `evaluation.py` → `pipeline.py` seam.
- Selection, calibration, and test refit the same specification at their
  permitted origins. The final fitted candidate—not its name—crosses
  into `predict_stage` and publishes the points.
- Candidate identity covers strategy, exact member set, fitted weights
  where applicable, behavior-changing configuration, dependency and
  weight revisions, fallback policy, and visible-data fingerprint.
- This makes the verified divergence unconstructible:
  - `pipeline.py:434-440` publishes hardcoded `weighted_mean` over all
    seven built-ins while `evaluation.py:724-725,907-909` scores
    `config.ensemble.strategy` over the restricted pool plus TSFMs;
  - `predict_stage` takes no config, so `ensemble.max_weight_ratio`
    reverts to default on publication (`ensemble.py:264`).

*Done when:* an end-to-end test under a non-default strategy,
restricted pool, and optional TSFM proves that publication uses the
final fit of the winning specification; fallback changes executable
identity and calibration provenance together.

### Phase 1B — Fold alignment and voting semantics

- Fix fold-index desynchronization for built-ins by adopting the TSFM
  placeholder alignment (`evaluation.py:799-810` vs `:897`).
- Correct `voting_forecast`'s up-biased ratio test
  (`ensemble.py:142-147`).
- Add asymmetric vote cases and partial-fold member failures.

*Done when:* forecasts and scores remain aligned after any member fails
any fold; voting is symmetric under inversion of direction.

### Phase 1C — Non-finite input contract

- Reject NaN and positive/negative infinity at the strict loader;
  `data.py:355` currently accepts `float("nan")`, after which
  `error_score` propagates NaN rather than returning `None`.
- Repair modes may treat documented textual sentinels as missing values,
  but no non-finite numeric value reaches evaluation.

*Done when:* every input format rejects or explicitly repairs
non-finite values before model selection, with structured recovery.

### Phase 1D — Honest outcome knowledge time

Do not stamp realized outcomes as known at the forecast cutoff or assume
they were known at horizon end. Preserve three distinct instants:

- `valid_time`: when the measured outcome occurred;
- `known_time`: when the outcome became available to Gnomon;
- forecast cutoff: what the forecast could see when it was produced.

Use explicit per-observation `known_at` when supplied and the actual
submission time otherwise. A derived coverage or performance record
inherits the maximum `known_time` of the outcomes it uses.

*Done when:* replay before submission cannot see the outcome or derived
score; replay after submission can; mixed-vintage actuals use the latest
contributing knowledge time.

### Phase 1E — Configuration that cannot lie

- Replace `INERT_KEYS` with allowlist schema validation. Dynamic
  namespaces such as API provider names and per-model overrides receive
  typed wildcard schemas rather than bypassing validation.
- Wire or reject `meta_model.min_folds`, `meta_model.fallback`,
  `ensemble.eligible`, `ensemble.weighted_mean.fallback`,
  `backends.sandbox.venv_root`, and `backends.sandbox.auto_install`.
- Honor `ridge_alpha` or reject it; the denylist says it is honored while
  `meta_model.py:87,152` hardcode `alpha=1e-6`.
- Reject `ensemble.strategy: stacking` at startup until it has an
  executable path; do not swallow it into per-fold `None`s.
- Return fresh default state and resolve mutable environment settings at
  construction time rather than import.

Move the canonical format to TOML through Python 3.11's stdlib
`tomllib`, with a real deprecation window:

1. Immediately: discovered YAML without PyYAML fails loudly; it is
   never silently discarded.
2. Release N: TOML is preferred; YAML still works with its compatibility
   parser and emits a conversion command and warning.
3. Release N+1: YAML requires an explicit compatibility extra or flag.
4. Next major release: YAML is rejected with the conversion command.

If both formats exist, fail and ask the user to select one.

*Done when:* every accepted key has a behavioral test and contributes to
identity where it changes an answer; unknown, inert, ambiguous, and
unparseable configuration fails before data execution.

### Phase 1F — Statistical and boundary-test scaffolding

- Add a seeded Monte Carlo interval-coverage harness with declared
  tolerance and sample size.
- Force the conditionally skipping guards at
  `test_contract_holes.py:438` and `test_config_ensemble.py:368` to
  exercise their guarded branches.
- Add property tests for candidate identity sensitivity, quantile
  ordering, point-in-time replay, and cross-surface equivalence.

*Done when:* the coverage harness is green and no integrity guard passes
solely because its intended branch was skipped.

## Phase 2 — Token economics (the 250–500K → ≤ 50K work)

### 2a. Kill the retry loops (call count)

- **Refusal → repair in one round.** Every refusal and every
  `next_actions` entry carries a literal ready-to-issue tool call
  (full argument dict), never selector syntax to compose.
- **Read-only verbs answer on ambiguity** (inspect already does after
  Phase 0; `describe` will). Publishing verbs (`forecast`, `decide`)
  keep the refusal — guessing a target column for a published number
  is not honest — but the refusal names the candidates with corrected
  ready-to-issue calls, so recovery is one round.
- **Lenient optional arguments:** natural window/compare forms parse;
  resolved absolute dates are always echoed. A supplied-but-
  unparseable optional argument returns the corrected ready-to-issue
  call rather than silently defaulting (a silent default answers a
  different question than asked).

### 2b. Shrink the per-turn tax (schema)

- Default profile stops being `full`. The distraction measurement
  (43.4% → 32.7% accuracy under 18 tools) already justifies this much;
  *which* narrow profile wins is Phase 3's experiment.
- Schema diet to a CI-enforced budget: **≤ 12KB serialized for the
  whole default surface** (bytes/4 tracks tokens within ~10%; no
  tokenizer dependency). Keep bytes/4 as the fast CI proxy, calibrated
  periodically against the actual tokenizer used by the evaluation
  agent. Requires: ≤ 2-sentence decision-rule
  descriptions, no examples or workflow prose, covariate/context
  channels demoted to `full`. The one ambient routing rule is repeated
  as a single sentence in every description.
- `output_dir` off the default surface (registry schema, capabilities
  text, Hermes copies — three sites, not one line).

### 2c. Shrink what history re-pays (responses and data)

- **Response contract, all verbs:** deterministic `headline` (2–5
  sentences, tier and staleness inline); `key_numbers` with tiers
  fused into the structure (paired `*_point`/`*_tier` keys + top-level
  `tier_floor`) so paraphrasing agents cannot shed the tier;
  `support`; typed caveats; `artifact_id`; `data_ref`;
  ready-to-issue `next_actions`. Hard budget ≤ ~1,200 tokens per verb
  response (≤ ~600 for describe) — a restructuring mandate (per-step
  rows move to the artifact), since the current minimum brief response
  is ~1,300.
- **Temporal grounding in every response:** `series_end`,
  `wall_clock_now`, staleness sentence when the gap exceeds one grid
  step. Cheapest fix in the plan; closes the report-June-as-next-week
  failure class.
- **Untrimmable set grows:** `headline`, `support`, `tier_floor`,
  `limitation_groups`, `assumptions`, staleness, `recovery_actions`,
  and artifact/data references join the protected keys — a trimmed
  limitation is a misquotation Gnomon performed on itself.
- **Multi-series triage** replaces the pipe-joined `artifact_headline`
  (`support.py:115-125`; untrimmable and unbounded at 490 channels —
  a bug fix, not an enhancement): aggregate headline, top-k notable
  with the ranking rule named, grouped remainder, full table via
  `get_artifact` selectors (`series`, `fields`, `where`, `order_by:
  notability`, `limit`); notability persisted in the artifact.
  `focus: ["name"]` returns full blocks for named series.
  Repeated limitations are represented as bounded typed groups rather
  than repeated prose, for example:

  ```json
  {
    "limitation_groups": [{
      "code": "POOLED_RESIDUALS",
      "affected_series_count": 312,
      "examples": ["api", "worker", "billing"],
      "artifact_selector": {"where": {"limitation": "POOLED_RESIDUALS"}}
    }]
  }
  ```

  Grouping changes only the response projection. The artifact retains
  the complete limitation set for every series.
- **`data_ref` in and out.** Every verb response returns the content
  fingerprint bound to the resolved schema, repair policy and result,
  as-of/snapshot boundary, visible vintages, and parser/runtime version;
  every verb accepts the opaque reference in place of `path`/`data`.
  Data crosses the wire once per session; follow-ups cost ~50 tokens
  and cannot re-infer either the schema or temporal view differently.
  Inline cap ~500 rows with the cost rule stated in the description.
  The design note must define ownership, lifetime, invalidation,
  isolation, and multi-host behavior. The first implementation is a
  session-scoped capability, not a guessable content hash.

*Done when:* every verb response within budget on the evaluation
corpus including the 490-channel task; schema budget test green; no
optional-argument error in any evaluation transcript; T2 workflow
≤ 2 calls median.

## Phase 3 — The surface experiment (measure, then flip)

Tool-count consolidation is a hypothesis, not a conclusion — a
polymorphic `gnomon_run` moves five tools' complexity inside one
schema, invisible to a tool-count metric. Gnomon owns the instrument.
Four arms, same task set, existing assessment methodology:

| Arm | Surface |
| --- | --- |
| `full` | 18 tools (status quo control) |
| `core` | 7 tools (exists today) |
| `core` + `describe` | 8 tools; descriptive verb prototype |
| mega-tool | 3 tools (`inspect`/`run`/`track`), polymorphic `question.kind` |

`describe` is the highest-prior arm: the evidence-injection arm beat
the raw control (48.9% vs 43.4%) while the tool surface lost (32.7%),
and `describe` is productized evidence injection — descriptive
temporal facts, computed exactly, sub-second, no backtest toll,
always an answer on readable data. Prototype scope per the redesign's
Part 2 (grid, level, trend, seasonality, changepoints, outliers,
extremes, comparison, relate, suggested_next), with the scheduled perf
work (shared detrend/ACF pass; memoized per-phase anomaly stats;
first-peak → strongest-peak fix at `temporal.py:150`).

Measured per arm: cumulative tokens, calls median/p95, yield,
accuracy **split by tier of evidence used** and **gated on the
leakage-controlled subset** (targets past cutoff, synthetic, perturbed
vintages); quote-vs-paraphrase and caveat-survival rates; repair-loop
completion (an abstention that dead-ends is a refusal engine).

Pre-register the decision rule before running the experiment:

- run paired, identical tasks across all arms with fixed model and
  sampling settings;
- declare the accuracy non-inferiority margin and minimum sample count
  overall and within every support tier before observing results;
- require at least a 50% reduction in median cumulative tokens relative
  to the full surface, with confidence intervals reported for accuracy,
  tokens, calls, and yield;
- price abstentions as task outcomes, report non-void yield separately,
  and require caveat survival to be no worse than control; and
- retain leaktrap 0/40 as a hard gate, not a weighted metric.

The default profile flips to the winning arm. If `core` captures most
of the gain, the mega-tool is not built. If `describe` wins, it
graduates from prototype.

**Pre-committed fallback (binding):** if the evidence-injection arm
still beats every tool arm, the product conclusion is that injection
*is* the product — a single evidence-pack call — and the surface
contracts around `describe` + `forecast` rather than growing. Written
down now as insurance against motivated reasoning later.

*Done when:* on the leakage-controlled subset, the chosen arm is
non-inferior to evidence injection under the pre-registered accuracy
margin, reduces median cumulative tokens by at least 50%, stays below
50K at the declared percentile, uses ≤ 2 calls median and ≤ 4 p95,
achieves non-void yield ≥ 80%, preserves caveats at least as well as
control, and passes leaktrap 0/40. Report paired confidence intervals
and tier-level sample counts; do not choose a winner from an
underpowered tier.

## Phase 4 — Positioning (last, because the claims must be true first)

Infrastructure framing: a bitemporal data layer for agents (the
differentiated asset — leaktrap 0/40 vs control 13/35, p = 0.00024)
plus a forecast engine agents can be trusted with. `investigate`,
`monitor`, `decide`, tracking presented as governed views compiled
onto one run contract (`router.py` and `decision_model.py` are 168 and
166 lines — the line counts already say they are views). Infra
discipline follows: semver + deprecation windows, tokens/latency as an
SLO, the MCP toolspec reviewed as the product API, PyPI installation.
README/identity rewrite happens only after Phases 1–3 ship.

The foundation-model claim stays empirical: TSFMs extend zero-shot
forecasting capability in regimes where their learned priors beat
simpler models. Gnomon and Forge determine when that is actually true;
the strategy does not require TSFMs to win every series.

TSFM default tier narrows to **one** benchmark-proven, version-pinned
adapter (chosen by measurement); the other adapters stay installable
as experimental. The maintenance surface is the 2,237 lines of
adapter/sandbox machinery, not the 137-line model zoo.

Ephemeris remains an explicit, disclosed hosted tier. Local trust is
not spent to manufacture conversion: no operational series leaves the
machine without user or project-policy authority, and local-only use
remains a complete supported path.

---

## Rejected (with reasons)

- **Model-zoo cut** — refuted by the code: 137 lines of stdlib;
  deleting `theta`/`ets` saves nothing and forfeits the models most
  likely to beat naive baselines on seasonal series. Selection
  variance under few folds → fold-count-aware admission, not deletion.
- **Three-tool consolidation as a decision** — it is an experiment arm
  (Phase 3).
- **Immediate five-package layout rewrite** — maximal churn, zero
  defect-fixing power; revisit only if Phase 1's seams make it
  mechanical.
- **Public three-state support vocabulary now** — graduated support
  tiers just shipped with verifier enforcement and benchmark pins;
  renaming the public projection churns a fresh contract. Revisit
  with misquotation-rate data.
- **Never-erroring optional args in full generality** — silent
  defaulting of a misparsed window answers a different question;
  adopted instead: lenient parse + mandatory echo + corrected
  ready-to-issue call on parse failure.

## Deferred

- Package layout convergence (after Phase 3 data).
- Support-vocabulary projection (after misquotation measurement).
- `data_ref` beyond single-session (multi-host, persistence) — value
  quantified by Phase 3's session statistics.
- Which TSFM adapter holds the default slot (measurement).
- Part-3 template-library breadth beyond fallback + high-traffic
  (grows only if caveat-survival measurement shows agents quote).

## Corrections to the input documents (verification pass, 2026-08-12)

- `_v1.md` claims the `llm.*` config section is parsed-never-read —
  stale; `config.py` no longer has an `llm` section.
- `_v1.md` claims `models.tsfm.overrides.*` is never read — false;
  read at `config.py:526`.
- All other cited defects re-verified live at `4ee9479` (line numbers
  updated above where they had drifted).

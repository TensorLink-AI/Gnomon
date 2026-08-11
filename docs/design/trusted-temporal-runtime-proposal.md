# The trusted temporal runtime — a narrowing proposal

Design proposal, 2026-08-11, at commit `16ca133` (v0.5.0). This is a
design record, not a build: nothing below is implemented by the act of
merging this file. It revises an earlier external critique of Gnomon's
scope in light of a verification pass that checked the critique's claims
against the code. Where this document asserts a defect, it cites the
line; where it proposes a change, it says whether the change is an
engineering task, a measurement to run, or a repositioning of language.

**What the verification pass changed.** The original critique argued
Gnomon should narrow to "a small, severe numerical authority that an
agent calls once and trusts," and proposed a three-tool surface, a
five-package layout rewrite, a smaller model zoo, and a reduced
configuration vocabulary. The review confirmed the critique's central
factual claims — and found that its remedies were mis-sequenced: the
highest-value fix is a contract rule that can land in the current
layout this week, the layout rewrite is cosmetic, the model-zoo cut
misdiagnoses where maintenance cost lives, and the tool consolidation
is an untested hypothesis that Gnomon's own benchmark harness can
settle. This document keeps the critique's direction and resequences
its execution around verified defects and measurable outcomes.

## The one-paragraph conclusion

Gnomon's pitch — "the LLM proposes; Gnomon validates, computes, and
owns every number" — is currently stronger than its publish path. The
ensemble that is backtested and the ensemble that is published are
different objects (`pipeline.py:372` hardcodes `strategy="weighted_mean"`
over all seven built-in models, while `evaluation.py:862-882` scores the
config-driven strategy over the config-restricted pool plus TSFM
adapters), a dozen documented configuration keys are parsed and never
read despite the `INERT_KEYS` remediation, and a base install silently
discards `gnomon.yaml` because PyYAML lives only in the `dev` extra.
None of these require a redesign to fix. The proposal is therefore
phased: **first make the evaluated object the published object; second
make configuration incapable of silent lies; third measure — not
assume — the right agent surface; and only then spend anything on
repositioning.** Everything identity-shaped ("temporal court of
record") is deferred until the guarantees it advertises are true.

## The promise being narrowed toward

> Give Gnomon time-series data and a question. It returns the strongest
> defensible forecast package — or explains precisely why it cannot,
> with machine-readable repair options.

Three guarantees carry that promise:

1. **Point-in-time-correct data access.** Already Gnomon's strongest
   evidenced property (leakage traps: control leaked 13/35, Gnomon
   0/40, McNemar p = 0.00024 —
   [results](../leakage-trap-results-2026-08.md)).
2. **Baseline-tested forecasts with calibrated uncertainty or explicit
   abstention.** Mostly true today; violated at the publish seam
   (Phase 1).
3. **Compact, machine-verifiable artifacts that agents cannot
   misquote.** Mostly true today; the response envelope work in
   Phase 3 finishes it.

Everything else is either a thin workflow compiled onto those
guarantees, independent machinery that earns first-class status on its
own evidence (anomaly detection qualifies), or experimental behind an
explicit gate.

---

## Phase 1 — Fitted executable candidates (engineering, immediate)

**The rule:** evaluation returns a fitted executable candidate — not a
model name. The object that won the backtest is the object that
produces the calibration residuals, the test-fold predictions, and the
published points. One object, one code path, three uses.

**Why this is first.** The current seam passes a *name* and a score
table from evaluation to publication, and the publication side
re-derives the forecast. That re-derivation has already produced one
fixed defect class (`docs/codebase-review-2026-08.md` C3: TSFM
final-prediction failure fell back to a baseline while keeping the
TSFM's residuals) and still has a live member:

- `pipeline.py:372` publishes the ensemble with `strategy="weighted_mean"`
  hardcoded; `evaluation.py:874` and `:1159` score
  `config.ensemble.strategy`. Set `ensemble.strategy: median` and a
  median ensemble is scored while a weighted-mean ensemble is
  published wearing the median's credentials.
- `pipeline.py:367` iterates all seven built-in `MODELS`; the evaluated
  pool is `active_models(config)` plus every registered TSFM adapter.
  Restrict candidates to `[drift]` and the published ensemble still
  averages seven models; include a TSFM and the published ensemble
  contains none.
- `predict_stage` (`pipeline.py:326`) takes no `config`, so
  `ensemble.max_weight_ratio` silently reverts to its default on the
  published path (`ensemble.py:264`).

A fitted-candidate object makes this class *unconstructible* rather
than merely tested-against. The candidate carries: strategy, member
list, fitted parameters, weights, version pins, and its fallback
policy. Fallback is part of the object, not an improvisation at the
call site — if the candidate cannot produce a final prediction, it
declines by its own recorded policy and the artifact says so.

**Scope discipline:** this is a refactor of the
`evaluation.py` → `pipeline.py` seam inside the current package
layout. It requires one new end-to-end test shape that does not exist
today: a full forecast run under a non-default ensemble strategy and a
restricted candidate pool, asserting the published points equal the
evaluated candidate's points (`tests/test_config_ensemble.py`
exercises `compute_ensemble_forecast` in isolation only).

## Phase 2 — Configuration that cannot lie (engineering, immediate)

Two inversions, both small:

**Allowlist, not denylist.** `INERT_KEYS` (`config.py:239`) chases
historically inert options one at a time; the verification pass found
ten-plus keys still parsed and never read (`meta_model.type`,
`meta_model.min_folds`, `meta_model.fallback`, `ensemble.eligible`,
`ensemble.weighted_mean.fallback`, the entire `llm.*` section,
`backends.sandbox.venv_root`, `backends.sandbox.auto_install`,
`models.tsfm.overrides.*`) and one denylist entry whose stated reason
is false (`config.py:268` asserts `ridge_alpha` is honoured;
`meta_model.py:87` and `:152` hardcode `alpha = 1e-6`). Replace the
denylist with schema validation: any key the loader does not
recognise, and any documented value the runtime does not implement,
fails at startup with a reason. `ensemble.strategy: stacking` — listed
in the example config, raised as `ValueError` in `ensemble.py:277`,
and swallowed into silent per-fold `None`s by the bare `except` at
`evaluation.py:872-882` — becomes a startup error instead of a silent
degradation.

**`gnomon.toml`, not `gnomon.yaml`.** The config file is silently
discarded on a base install because PyYAML is a `dev`-only extra
(`config.py:207-211` returns `DEFAULT_CONFIG` on `ImportError`;
acknowledged in `docs/main-evaluation-2026-08.md`). Gnomon requires
Python ≥ 3.11, and `tomllib` is stdlib from 3.11. Moving the config
format to TOML deletes the entire dependency-sensitive-silent-default
class while keeping `dependencies = []` literally true. A `gnomon.yaml`
found on disk during the transition is an *error* naming the migration,
never a silent no-op.

The normal configuration surface after Phase 2 is deliberately small —
evaluation thresholds, coverage target, model profile, output
directory. Every documented key is honoured end to end or the process
does not start; the config coverage test asserts the example file and
the runtime agree.

## Phase 3 — The agent surface, measured (benchmark, then engineering)

**Consolidation is a hypothesis, not a conclusion.** The original
critique proposed collapsing seventeen tools into three
(`gnomon_inspect` / `gnomon_run` / `gnomon_track`). A single `gnomon_run`
with a polymorphic `question.kind` union carries five tools' worth of
schema inside one tool; agents fumble discriminated unions too, and
that failure mode is invisible to a tool-count metric. Gnomon already
owns the instrument to settle this: the MCP assessment measured tool
distraction dropping task accuracy 43.4% → 32.7%
(`gnomon-mcp-assessment.md`), and the `core` profile — seven tools —
already exists (`toolspec.py:1782`).

The experiment, using the existing assessment methodology, compares
three arms on the same task set:

| Arm | Surface |
| --- | --- |
| `full` (status quo) | 17 tools |
| `core` (exists today) | 7 tools |
| mega-tool prototype | 3 tools, polymorphic `question.kind` |

Measured: task accuracy, misquotation rate, tokens and latency per
task, and repair-loop completion (below). The default profile flips to
whichever arm wins; if `core` captures most of the gain, the mega-tool
is not built. Either way the default stops being `full` — the
distraction measurement already justifies that much.

**Abstention is a repair-and-retry loop, and the contract says so.**
"An agent calls once and trusts" is honest marketing only when the run
succeeds. When Gnomon abstains, the response carries recovery actions,
and the expected shape is: run → abstention with repairs → agent
applies a repair or narrows the question → run again. The tool
contract names this loop explicitly rather than implying single-shot
use; repair-loop completion rate is a first-class benchmark metric,
because a harness whose abstentions dead-end is just a refusal engine.

**The response envelope.** Every successful run returns a compact
envelope: a Gnomon-authored `headline` sentence (the agent quotes it
rather than composing its own), `support`, the values with per-row
tiers, `limitations`, `evidence` (selected model, strongest baseline,
improvement, measured coverage), and the artifact reference. Three
public support states — `supported`, `limited`, `abstained` — with the
richer internal tiers preserved in the artifact. The response-budget
trimmer already protects a key set (`toolspec.py:176`);
`headline`, `limitations`, and `recovery_actions` join it — they are
untrimmable at any budget, because a trimmed limitation is a
misquotation Gnomon performed on itself.

## What stays first-class

- The temporal store, vintages, and `--as-of` replay — the leakage
  boundary is the crown jewel and has the strongest outcome evidence
  in the repository.
- Mandatory baseline comparison, separated selection/calibration/test
  folds, conformal intervals, per-row support tiers, deterministic
  verification, disclosed repair, content-addressed artifacts.
- **All seven classical models.** The original critique proposed
  shrinking the zoo to reduce maintenance and selection variance. The
  entire zoo is 137 lines of stdlib (`models.py`); deleting `theta` or
  `ets` saves nothing and forfeits the models most likely to beat
  naive baselines on seasonal series. Selection variance under few
  folds is real, but the remedy is fold-count-aware candidate
  admission, not deletion.
- **Anomaly detection.** At 602 lines with the injected-anomaly
  grading tournament (`anomaly.py`), `detect` is independent machinery
  with its own validation approach, not a forecast-adjacent view. It
  keeps first-class status; it is the *last* verb that would be
  demoted, not the first.
- Forecast tracking against realised outcomes (`tracking.py`) — the
  accountability loop is what makes "court of record" more than a
  metaphor.

## What compiles into the trusted run

`monitor` (forecast distribution + threshold + intervention cost) and
`decide` (forecast scenarios + supplied actions + supplied utility)
remain user-facing question types but are views compiled onto the one
numerical pipeline and the one support contract. At 168 and 166 lines
respectively (`router.py`, `decision_model.py`) they are already thin;
this phase makes that architecture honest rather than incidental.
`investigate` similarly: validated historical diagnostics plus
associational evidence, under the same verifier.

## What is experimental, and where TSFMs land

Behind the existing gates, unchanged in spirit: the plan
compiler/executor (already env-gated), meta-model stacking beyond the
one proven configuration, LLM-derived context effects, structural
transforms, and the proposer-skill ledger. Experimental means gated
out of the default trusted path, not deleted.

**TSFMs: limit the default tier, keep the machinery.** The
maintenance cost the critique attributed to the model zoo actually
lives in the 2,151 lines of adapter and sandbox machinery serving
seven third-party TSFM families (`tsfm.py`, `tsfm_sandbox.py`,
`api_inference.py`). The default trusted tier admits **one** proven
TSFM adapter — chosen by benchmark evidence, version-pinned in the
fitted candidate. The other adapters remain installable and gated as
experimental. This inverts the critique's cut: keep the 137-line zoo,
narrow the 2,151-line surface.

## What is rejected

- **An immediate package-layout rewrite.** The proposed five-package
  split (`temporal` / `forecast` / `contract` / `tracking` /
  `interfaces`) is a reasonable eventual shape and a poor first move:
  maximal churn, zero defect-fixing power, and it competes for the
  same review attention Phases 1–2 need. Revisit after Phase 3 has
  data; adopt only if the seams the phases create make the split
  mostly mechanical.
- **Tool consolidation as a decision.** See Phase 3 — it is an arm in
  an experiment, not a commitment.
- **Model-zoo reduction.** See above.
- **A support-vocabulary rewrite below the envelope.** Three public
  states in the response; the internal tier system, warnings, and
  degradation semantics are unchanged. Artifacts keep their richness.

## How success is measured

The headline evaluation table for the narrowed Gnomon:

| Measure | Why it matters |
| --- | --- |
| Completion rate | Does Gnomon actually answer? |
| Error when answered | Are published values good? |
| Error with abstention priced | Is caution economically useful? |
| Interval coverage and width | Is uncertainty calibrated and useful? |
| Leakage failures | Does the structural guarantee hold? |
| Repair-loop completion | Do abstentions convert to answers? |
| Agent misquotation rate | Can agents safely communicate results? |
| Decision regret | Did downstream choices improve? |
| Tokens and latency per task | Is the integration operationally viable? |

The target is not lowest median error. It is: near-baseline median
accuracy, dramatically fewer catastrophic failures, zero leakage,
calibrated intervals, and useful completion at low operational cost —
with the abstention-priced error making the case that Gnomon's caution
pays for itself. Most of the infrastructure to compute these exists
(`benchmarks/`, `agent_eval.py`, the leaktrap suite).

## Sequencing

| Phase | Kind | Contents | Gate to next |
| --- | --- | --- | --- |
| 1 | Engineering | Fitted executable candidates; end-to-end publish-equals-evaluated test | Test green on non-default strategy + restricted pool |
| 2 | Engineering | Allowlist config validation; `gnomon.toml` via `tomllib`; fail-at-startup | Config coverage test proves example ↔ runtime parity |
| 3 | Measurement → engineering | `full` vs `core` vs mega-tool benchmark; envelope + untrimmable keys; default profile flip | Assessment re-run shows chosen arm ≥ `core` |
| 4 | Positioning | "Temporal court of record" language; docs narrowing; experimental tier labels in README | Phases 1–3 shipped — the claims must be true first |

## The resulting identity

Gnomon stops trying to be the agent's temporal brain and becomes its
temporal court of record: the agent asks; Gnomon computes; Gnomon
states what the evidence permits; the artifact proves it; later
outcomes hold it accountable. The narrowing makes `investigate`,
`monitor`, and `decide` *more* credible, because they are visibly
governed views over one trusted forecasting contract rather than
separate features competing for maturity — and it makes "Gnomon owns
every number" a property of the architecture rather than an aspiration
of the README.

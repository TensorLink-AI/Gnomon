# Ledger optimization development protocol

Status: development; no release or superiority claim. Branch: `dev/ledger-optimization`.
Baseline source: Gnomon 1.1.9, commit `59a6d81709a4625bf042e7ca152aa5f12534c28a`.

## Objective agreed before experiments

Achieve at least **20% lower mean per-case RMSLE** than a matched no-ledger
agent on an untouched final evaluation set. The paired 95% uncertainty interval
must exclude zero improvement, and development must improve over the current
1.1.9 ledger. The 20% threshold refers to the point estimate; it is not the
lower confidence bound. User authorizes required Engy/API spending. Log usage,
reported costs, and unavailable billing information; missing cost is not zero.

## Fair comparison

Three arms: no ledger, frozen 1.1.9 ledger evidence, development ledger evidence.
Use identical agent model/settings, candidate implementations, current history,
CV evidence, forecast execution budgets, completion resolution and fallback
policies. The historical evidence is the intended treatment. A typed execution
is authoritative under the same frozen resolution rule in every arm. Keep strict
final conformance separate from resolved execution and forecast accuracy.

Each case is one series/origin/horizon. Primary metric is the arithmetic mean
of case RMSLE, not the square root of pooled mean squared log error. For the
retail benchmark, reject negative actuals; clip negative predictions at zero
consistently in every arm and disclose clipping counts. Failures remain in the
denominator with an identically defined seasonal-naive fallback. Missing or
malformed candidate forecasts invalidate the common case preparation, not just
the losing arm. Preserve that preparation failure in the case manifest.

At origin t, expose only observations and outcomes whose source availability
and recording visibility allow them at t. Finish all arm decisions before
advancing the ledger. Do not infer stockouts or causal explanations from zeros.

## Development and final evaluation

All previously inspected Favorita cases are development data. The existing
100-origin/four-series run is **not** a final holdout. Freeze a copy/hash manifest
before analysis; never mutate its ledger or rewrite its original scores.

Reserve additional series and later evaluation periods before examining their
outcomes. Include multiple independent series, multiple matched agent seeds,
and all cold-start cases. Freeze exact series IDs, origin dates, seeds, agent
settings, candidate identities, data hashes and sample size before a final run.
The final manifest is pending access to the original runner/full corpus.

Uncertainty must retain within-series temporal dependence (series clusters and
moving origin blocks); seed repetitions are not independent new series. Fix the
resampling specification before final scoring. Report cold-start/mature windows,
per-series changes, completion rates, token/call usage and fallback causes.

Make one bounded change at a time, run regression checks, compare on development
data, and retain every variant and outcome. Use a fixed development validation
slice for variant selection. Freeze the chosen implementation before the one
final evaluation. Do not repeatedly inspect the final set until a win appears.
If the target is unattained, record that result and continue only on development
data; a later confirmation requires genuinely fresh evidence and a new manifest.

## First experiments

1. Independently reproduce old scores and compute the hindsight candidate oracle
   as a feasibility ceiling (never as an executable policy).
2. Add explicit MAE/RMSLE comparisons, calculated ranks/ties/differences and
   recent-versus-lifetime disagreement to the public ledger query.
3. Compare metric-aligned cards and simple past-only selectors before expensive
   repeated agent calls. Historical selection must use matured outcomes only.
4. Run matched Engy experiments with execution-bound final selection. Treat any
   completion changes identically in the control.

Changing the candidate family, adding ensemble/calibration forecasts, changing
the baseline or changing the primary metric requires a separately recorded
protocol amendment. Never make those changes merely to manufacture 20%.

## Retention

Commit code, protocol, input hash manifests and compact per-case/aggregate
results on the development branch. Keep large original transcripts/data in
ignored experiment storage with hashes and retrieval paths. Commit no secrets.
Do not merge to main, tag or publish PyPI as part of this experiment.

## Development amendment 001 (after pilot 002, before expanded run)

Add a fourth development-only arm, `ledger_blended`, showing the same RMSLE
card plus a clearly labeled equal-weight current-CV/lifetime-history estimate.
This screens a policy suggested by the offline results; it is not a final-test
amendment or evidence of validated future accuracy. Expanded development cases:
all four legacy series at rounds 0, 28, 56, 84; requested seeds 7 and 19. Freeze
all four arms before dispatch (32 matched case-seed pairs, 128 decisions).

## User clarification: accumulated evidence is the intended advantage

The user clarified that the focus is ledger infrastructure: accumulating what
worked and what failed, then using relevant past evidence for future decisions.
Do not expand this effort into new forecasting models, ensembles or forecast
bias correction. The forecasting candidates and control capabilities remain
fixed. Calibration feasibility remains an explicitly out-of-scope prototype.

Next development work should improve the path from stored outcomes to usable
decision evidence:

1. Preserve task, provider revision, forecast-time context and matured outcomes
   together. Context must be observable at the original decision time; never
   label past episodes using subsequently observed success or failure.
2. Retrieve comparable past episodes, initially through explicit filters on
   observable conditions, and compare providers on the same eligible origins.
   Show the filters, sample counts, exclusions, dates and numerical references.
3. Expose disagreement between recent, longer-window and comparable-context
   evidence. Make insufficient or stale evidence explicit. Do not convert an
   observed loss into a proven causal business explanation.
4. Measure whether accumulated evidence changes decisions usefully. Compare
   no ledger, existing summary cards and contextual evidence with identical
   models, current inputs, candidate forecasts and budgets. Report cold-start
   and mature-history performance without removing cold starts from the primary
   metric. Additional records are valuable only if they improve future choices.

Existing APIs already support context records and exact `compare_context`
filters. The agent trials so far mainly used aggregate recent/lifetime cards;
their results do not test the full contextual retrieval path. Build on those
APIs and measure the end-to-end retrieval path before adding another storage
system. Keep the 20% objective, acknowledge fixed-portfolio headroom on each
tested cohort, and preserve the untouched confirmation partition.

## Later user-authorized ML task amendment (019)

The user subsequently requested a matched Hermes task that requires backtesting
and iterating a time-series model, comparing Hermes alone, Gnomon + Hermes and
Gnomon + Hermes + ledger. This authorizes the common fitted Ridge/Random Forest/
seasonal configuration space in the ML development protocols. It supersedes the
earlier fixed-eight-recipe restriction for that task only. It does not authorize
stronger forecasting tools or privileged observations exclusively for treatment.
The earlier fixed-portfolio screens and negative confirmation remain unchanged.

The current checkpoint-v3 trial uses Gnomon 1.2.0 in both Gnomon arms and retains
all three arms' matured raw outcome scores. It is development evidence about a
workflow, not a matched 1.1.9 comparison or untouched final evaluation. Completion
gates do not substitute for the 20% point target or uncertainty requirement.

Before final confirmation, compare the selected development implementation with
both an equally informed no-ledger arm and the frozen 1.1.9 incumbent on the same
task. Compatibility check 019 verifies that the exact published 1.1.9 build
`59a6d81709a4625bf042e7ca152aa5f12534c28a` supports the shared typed execution,
covariate/identity, controlled-clock, scoring and matched-history contracts.
This is not evidence of the incumbent agent's performance.

The old incumbent MAE cards used last-four, last-twelve and lifetime windows with
complete matched origins. A dynamic configuration search requires an explicit,
prospectively frozen adaptation of that evidence presentation. Preserve config
and provider-revision identity and show missing overlap; do not deliberately
cripple the incumbent with an all-configurations intersection or fabricate
unexecuted forecasts. Keep completion handling, budget accounting, raw evidence
access and outcome maturation common. Freeze the adapter and test it before any
comparative agent inference. Package version alone does not define a fair memory
treatment; neither a deterministic MAE selector nor the existing 1.2.0 pilot can
be relabeled as the 1.1.9 agent baseline.

The 24 M5 reserved series remain unopened. The ML task would need a separately
recorded history/covariate adapter before using that source, without changing its
identity selection or inspecting reserved targets to choose a favorable task.
Keep the final development gates, one-shot final freeze, primary metric and
cluster/time-block uncertainty rules. No passing final candidate exists yet.

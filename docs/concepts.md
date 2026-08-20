# Forecasting and evaluation concepts

## Typed temporal questions

Gnomon separates intent interpretation from numerical execution. An agent may
translate a user's wording into a `TemporalQuestion`, but deterministic code
validates its verb, exact target, property, horizon, measure, comparison, and
context policy. Unknown quantities and ambiguous targets are refused with a
ready-to-issue repair call; accepted interpretations are echoed in answers.
Hosts that support an LLM adapter can call `compile_temporal_text`; the model
only proposes typed intent and never sees forecast values or gains numerical
authority. The ordinary deterministic compiler still accepts or refuses every
proposal. Tool-using models are not expected to discover optional question
fields reliably on their own.

Production hosts should persist a content-addressed compiler receipt containing
the source fingerprint plus the model's proposal and each independently
accepted or rejected question. Replaying the receipt avoids changing intent
between surfaces; a changed source or target set invalidates it. Compiler
acceptance grants no numerical authority and cannot alter the fitted primary
forecast.

Persisted receipts are addressed by `context_ref`. The reference is scoped by
`GNOMON_CONTEXT_NAMESPACE` and stored under `GNOMON_CONTEXT_STORE` (default
`.gnomon/context-store`). Context-aware calls accept the reference instead of
resending events and return cache provenance (`stored` or `hit`). Resolution
verifies the receipt hash and reapplies knowledge-time filtering for the
current `as_of`; a cache hit never grants an event authority in a historical
replay where it was not yet known.

Numerical context assessments use a separate immutable cache. Its key binds
the receipt, exact visible-data fingerprint, `as_of`, series, selected primary
path, context policy, and implementation version. Consequently compilation
reuse can cross compatible calls, while fitted evidence cannot cross data
vintages or configurations.

`gnomon_describe`, `gnomon_forecast`, `gnomon_decide`, and `gnomon_monitor`
accept optional typed questions and
return one compact answer per question. These diagnostics sit beside the
primary forecast: they cannot modify model selection or the immutable primary
answer. Predictive comparisons use that exact published path: level questions
return history and forecast medians plus absolute and relative change, and
seasonality questions return alignment with the repeated historical phase.
When no task-specific categorical threshold is supplied, Gnomon applies a
documented property default and labels the resulting best estimate `weak`.
Weak answers remain useful for exploration but carry
`automation_eligible: false`; Gnomon abstains only when it cannot compute a
meaningful estimate.

Hosts may project canonical values into an explicit choice vocabulary through
the versioned semantic-alias projector. Exact or uniquely equivalent options
are eligible; ambiguous mappings are refused. A projected value may override
an automated submission only when `automation_eligible` is true. Weak best
estimates remain advisory and never silently replace a person's or agent's
choice.

Typed answers share the persistent content-addressed store used by context
receipts, but occupy a separate immutable cache. Its identity binds the
primary forecast ID, canonical question, `as_of`, namespace, and answer
contract version. Validation occurs before lookup, cached artifact/question
identity is checked on every hit, and a cache result never upgrades weak
support or automation eligibility.

Predictive properties use fitted executable receipts. Level, trend, seasonal
continuation, regime magnitude, and extreme risk compare small deterministic
candidates on rolling origins; paired dependence selects a return-correlation
window the same way. Candidate selection, finite-sample calibration, and the
published estimate are one object. Rare-event absence is not promoted into an
affirmative supported "nothing will happen" claim: the best direction and
distribution remain available with weak support, but automation stays barred.
Extremes are forecast from held-out block
maxima of innovations rather than from the smoothness of a point path.

Each answer leads with one canonical `best_estimate` containing the value,
display value, support, and automation eligibility. `decision_rule` names the
fitted executable or aggregation and its published thresholds. An optional
explicit `answer_vocabulary` maps canonical strings to caller-facing wording;
Gnomon never invents that mapping from answer options. Estimate, interval,
headline, and limitations remain alongside it. The full receipt is
saved as `temporal_answers.json` with `primary_forecast_unchanged: true`.
Tracking stores the exact first receipt (replays cannot mutate it) and joins it
to a compact realised-path summary when the forecast horizon is scored.

Future volatility means causal forecast-residual scale—not movement of the
expected path or forecast-interval width. Its executable compares constant,
recent-window, fold-selected EWMA decay, robust scale-trend, scale-momentum,
regime-mixture, and seasonal-phase estimators at rolling origins. Scale and
direction candidates are selected separately: scale uses log error and QLIKE;
direction uses prequential Brier skill and balanced accuracy, with each fold's
probabilities calibrated only from earlier folds. It returns a calibrated distribution for the
future/reference scale ratio. The scale estimate and directional claim have
separate support: Gnomon may publish a supported scale estimate while marking
its directional best estimate weak and ineligible for automatic action. A
shared `TemporalDecisionPolicy` turns a probability distribution into an
actionable category only when fold count, balanced accuracy, Brier skill, and
probability gates all pass; otherwise the continuous estimate and weak best
state remain available without becoming an automated claim.

When a caller asks about the behavior of an already-published forecast,
Gnomon projects volatility from that immutable path instead of fitting a
second, history-only future. The projection removes the path's own seasonal
structure, compares its robust residual scale with recent historical
residual scale, and is always labelled weak and uncalibrated: it describes
what the forecast says, not what future observations are guaranteed to do.

Targets are explicit scopes: `series` binds one exact channel; `each` returns
one independently supported answer per named channel; and `aggregate` requires
a named aggregation. Cross-unit volatility currently permits only
`median_normalized_scale_ratio`. Cross-series seasonal continuation permits
`median_alignment`. Both publish one aggregate best estimate while preserving
constituent answers in the immutable receipt; the inline response contains a
compact constituent support summary.
Raw scales with incompatible units are never averaged silently.
Cross-series predictive dependence requires `pair` scope with exactly two
distinct members and reports first-difference correlation; it is never inferred
from two unrelated single-series answers.

## Why Gnomon runs baselines

A forecast is useful only relative to a credible simple alternative. Gnomon
always tries last-value and seasonal-naive forecasts before considering the
candidate models (drift, linear trend, window average, Theta, ETS). A
candidate must beat the strongest successful baseline by a configured margin.

## Temporal evaluation

Random train/test splitting leaks future structure into time-series evaluation.
Gnomon instead uses ordered rolling origins. At each origin, the model sees
only earlier observations and predicts the next complete horizon.

The available origins are divided chronologically:

```text
earlier origin(s)       penultimate origin       final origin
model selection    →    interval calibration  →  report-only test
```

The selection origins choose the model. Signed residuals from the calibration
origin construct forecast quantiles. The final origin measures score and
interval coverage without changing either choice. The selected method is then
fit against all observations to forecast beyond the dataset.

This separation reduces selection and coverage optimism, although a single
calibration and test horizon still provide limited evidence. That limitation is
why warnings and support status matter.

## Per-series selection

Panel series can behave differently. Gnomon evaluates and selects each one
independently rather than forcing one model across an entire panel. One series
may retain seasonal-naive while another selects a candidate model or abstains.

## Improvement threshold

If baseline error is `B` and candidate error is `C`, candidate improvement is:

```text
(B - C) / B
```

With the default threshold of `0.02`, a candidate must reduce selection error
by at least 2%. When baseline error is exactly zero, Gnomon retains the baseline.

## Abstention

Gnomon distinguishes two failure classes:

- invalid data or task: a structured error and exit code `2`;
- valid data but inadequate forecasting evidence: under the default
  `minimum_support: best_effort` floor, a graded answer — the largest
  supportable horizon prefix at whatever tier its own evaluation earned,
  and the remainder as a labelled naive fallback (a `horizon_split`
  reason names both ranges; every row carries a `tier` field; the
  response's `headline` names the weakest tier present). With
  `minimum_support: supported` (or `conditionally_supported`), a
  complete artifact with `unsupported` support and no future values.

Nothing about how tiers are earned changed with the graduated default:
fold minimums, separation requirements, and selection guardrails grade
exactly as before — the floor only chooses which rung is published. The
deterministic verifier additionally rejects any claim quoting a
sub-supported value without its tier label, so a plausible-looking
unlabelled forecast cannot be returned when the evaluation contract
was not satisfied.

An abstention (from a raised floor, or a series where nothing is
computable) is never a dead end. Alongside `provide_more_history`, the
support assessment computes the largest horizon the supplied observations
*can* support and, when one exists, names it as a `reduce_horizon`
recovery action (and in the warning text: "retry with `--horizon N`") —
an immediate trade of forecast reach for an honest result, instead of
waiting for more data. When no shorter horizon would succeed either, the
recovery is absent rather than aspirational.

## Context events

A context event is something you know about the world that the series
alone cannot show: a promotion window, a capacity cap, a migration. Gnomon
never takes your word for its *effect* — it measures one, on the same folds
everything else competes on, and excludes the event when it cannot.

Narrative compilation returns a content-addressed `context_receipt`. It binds
the validated events and hypotheses to source-document fingerprints, compiler
identity, and prompt version, so a host can compile once and replay the exact
receipt across runs. Qualitative soft context uses a closed effect vocabulary
(level/trend/variance/pulse/bound/seasonal regime), direction, and duration;
its magnitude is always null at compilation time.

`examples/context_events.json` is a worked file with one of each kind:

```bash
gnomon forecast examples/messy_requests.csv \
  --time timestamp --target requests --horizon 14 \
  --context examples/context_events.json
```

**An event with a `claim`** states a bound on what is possible.
`capacity-cap-2026-06` says throughput cannot exceed 360, and that bound is
projected onto every emitted quantile after the model has said what it
believes. Bounds are admissible; pinned values are not — an event that
supplied a *value* would be supplying the answer. A bound the training
window already breaches is rejected, with the violating timestamps named.

**An event without one** is a window whose effect Gnomon estimates.
`marketing-push-2026-06` marks a campaign; the context ablation measures
its effect either from detrended history or from leakage-safe one-step
residuals of the selected history model. Both estimators compete on identical
folds, and the winner is disclosed as `effect_estimator`; this lets Gnomon
isolate aperiodic interventions without mistaking ordinary seasonality for an
effect. The effect shape (level, decay, ramp) is chosen by the same
measurement, never by the caller — a caller who could name the shape could
fit a story to the data.

Every result with applicable context also carries `context_outcome`:
`not_considered`, `rejected`, `scenario_only`, or `applied`. `scenario_only`
means the event was grounded but did not earn a numerical change to the
primary forecast. It includes recovery actions and, when repeated historical
occurrences make an effect measurable, a separately labelled conditional
forecast. For a novel event with a grounded direction but no estimable effect,
Gnomon may instead emit `sensitivity_scenarios`: a standardized one-innovation
shock derived from the target history. This is explicitly
`hypothetical_sensitivity`, carries no probability, and never replaces or
changes the primary forecast. It answers “what would this direction look like
at a stated scale?”, not “what will happen?”. `applied` means the deterministic
candidate passed admission; it is never inferred from compiler acceptance.

Conditional and sensitivity paths expose the same typed `effect` contract:
`distribution` records location, uncertainty, interval probability and sample
count, while `provenance` records whether the effect was observed or assumed,
its evidence class, `known_at`, source, similarity and reliability. A
standardized sensitivity uses `distribution: assumption` and an explicit null
interval probability; it cannot acquire probability semantics through JSON
serialization. With project tracking enabled, these paths and their unchanged
primary are registered as a content-addressed pair. Submitted actuals add a
descriptive realised estimate, onset and duration; overlapping scenario
windows are marked confounded rather than assigned false individual credit.
Actual values alone do not prove the event occurred. Registry entries remain
`occurrence_status: unverified` and `eligible_for_learning: false` until an
operator supplies a dated confirmation, cancellation, or revision alongside
actual submission (or with `gnomon track effect-occurrence`).

When admitted context changes the selected output, the artifact additionally
carries `primary_forecast`: the complete history-only point and interval path
frozen before enrichment. During the compatibility window, `forecast` remains
the selected context-conditioned projection and is labelled
`forecast_role: context_conditioned_projection` on CLI/MCP responses;
`primary_forecast` is the canonical baseline for comparisons and effect
tracking. Context-free runs omit the additive field and label `forecast` as
the primary directly.

After at least five confirmed, resolved, unconfounded episodes, Gnomon may
form an organizational effect prior. Same-series estimates shrink toward the
project-wide event-type mean; a related-series estimate is used only when no
local episode exists. The prior must beat a zero-effect baseline under
leave-one-event-out validation and uses posterior-predictive—not mean—variance
for the next event. Query the evidence with `gnomon track effect-prior`, or
through MCP with `gnomon_status` and `section: "effect_prior"` plus
`project`, `event_type`, `series`, and a timezone-aware `as_of`.

The evidence ladder never averages unlike warrants: validated organizational
memory wins over a dated, versioned external prior, which wins over an explicit
human assumption. Conflicting directions are disclosed. Human assumptions
remain non-probabilistic. For decisions without defensible scenario
probabilities, `gnomon track robust-decision` applies maximin utility across
the immutable primary and every named conditioned scenario; utilities and
constraints must come from the caller. The experimental three-tool MCP
surface exposes the same operation as `gnomon_run` with
`question.kind: "robust_decision"`; it persists the decision for later
outcome and regret scoring.

Every event needs a `known_at`. It is what makes the backtest honest: a
fold cutting at T may only use events knowable by T, so an event recorded
after the fact cannot improve a historical fold.

Events must carry an explicit timezone offset. When the dataset's own
timestamps are naive — as every example here is — the windows are matched
on wall-clock time and the result carries a `context_timezone_aligned`
disclosure saying so.

## Canonical temporal profile

Describe and forecast results expose one deterministic `temporal_profile`.
Forecast artifacts persist its complete form as `temporal_profile` evidence;
the frozen result schema remains unchanged. MCP responses carry a compact
projection so the descriptors cannot displace requested forecast rows.
It keeps quantities that look similar in prose separate in the contract:

- `trend` reports direction through its slope and whether the recent trend is
  persistent, decaying, reversed, emerging, or flat;
- `seasonality` reports the measured period and whether its strength is stable;
- `volatility` means robust dispersion of observations around estimated trend
  and seasonality;
- `marginal_variability` compares the predicted distribution of complete
  horizon values with raw historical variability;
- `forecast_behavior` reports movement of the expected point path and interval
  width, and explicitly says neither is interchangeable with observation
  volatility;
- `regimes`, `extremes`, and `dependence` expose variance changes, robust tail
  counts, and residual persistence or mean reversion.

The predictive residual-volatility direction is emitted only when the same
direction persists across three historical blocks. Forecast-horizon marginal
variability is separately derived from point-path variation and calibrated
q10/q90 dispersion and carries its approximation assumptions. A smooth point
forecast therefore never silently becomes a claim that observations will be
stable.

## Current methodological limits

Gnomon is a correct, deliberately narrow foundation—not a general forecasting
suite. Its built-in candidates are deterministic classical models (drift,
window average, linear trend, theta, ETS) plus optional sandboxed TSFM
adapters; seasonal periods are detected or overridden, not learned per
model. Covariates and context events are admitted only through
identical-fold ablation; when both are supplied, a deterministic
adjudication ladder compares the base model against every admitted
challenger on identical folds and records the comparison as evidence.
There are no transformations and no dedicated intermittent-demand
methods. Use `gnomon capabilities` as the machine-readable
source of truth.
## Temporal evidence and reasoning

Gnomon does not replace an agent with a deterministic decision tree. It gives
the agent three compact, auditable primitives:

- **Temporal comparison** measures observed changes in level, trend, and
  residual volatility in adjacent windows. These facts carry their window and
  provenance and are kept separate from predictions.
- **Evidence aggregation** combines comparable facts across related series.
  Series breadth is reported as an effective-series count and is never
  presented as additional temporal backtest folds. A predictive reading of an
  observed change is explicitly conditional on persistence.
- **Conditional reasoning** tests contextual hypotheses against outcomes and
  publishes them as conditional forecasts or sensitivity paths. It never
  rewrites the immutable primary forecast.

The compact answer remains the canonical value for quoting. Supporting
evidence helps the language model explain and qualify that value; weak or
conditional evidence is not eligible for automated action.

Canonical computation and agent synthesis are separate contracts. Supported,
automation-eligible answers bind publication. A weak answer is advisory but
remains the default; an agent may attach a separately labelled synthesis, while
changing the published choice requires typed opposing evidence. Computed
opposition must win the deterministic adjudication in the immutable receipt,
come from at least two independently supported source kinds, and authorize only
the exact projected alternative those receipts support. Task-context evidence
must retain an exact source quote and pass the same outcome-backed adjudication;
a quote alone establishes provenance, not predictive truth. Neither lane can
rewrite the primary forecast.

Observed transition support is graded separately from identifiability. A
borderline estimate can be measurable and useful while remaining `weak`.
Likewise, a point estimate near zero does not prove stability: a supported
equivalence claim requires an equivalence interval, so null transitions remain
weak until that evidence exists.

Each typed answer also carries a bounded reasoning plan inside its existing
answer envelope. The plan distinguishes observed, predictive, and conditional
inference; names the evidence basis or what is missing; and surfaces conflicts
between predictive and observed evidence. It gives the host model authority to
explain and qualify, never to replace the fitted executable's canonical value.
The plan additionally projects the receipts into three contrastive lists —
`because`, `against`, and `unknown` — and may retrieve up to three separated
historical states with completed outcomes. Analogue matching uses normalized
level, slope, and residual scale; those outcomes are historical evidence, not
extra forecast folds. `suggested_next` names an observation or recovery action
that can resolve missing or conflicting evidence. Conditional context enters
the same plan only through the artifact's typed effect and provenance contract.
Tracked or governed effects retain their property-specific vocabulary; an
unvalidated assumption remains conditional and cannot earn synthesis authority.
Evidence weights rank receipts but are never presented as probabilities. A
probability is exposed only when a fitted executable supplied its calibration.

## Model-neutral forecast boundary

Built-in statistical models, in-process TSFMs, isolated subprocess models, and
remote inference APIs cross the same validated forecast protocol before they
can be evaluated or published. The boundary requires finite inputs, an exact
horizon, monotone requested quantiles, immutable inputs, deterministic replay,
and explicit adapter identity. Remote providers can pin a `revision`; an
unpinned provider is disclosed as `unversioned`, never recorded as
reproducible.

New adapters first run in shadow. Their errors are paired with the published
baseline only after outcomes become known. The shadow ledger can emit
`review_for_promotion` after minimum sample, improvement, and win-rate gates,
but it never changes the publishing model automatically. Promotion is an
explicit deployment decision and the fitted executable still owns every
published number.

### Pretrained-model admission on short histories

Pretrained and locally fitted models answer the same comparative question —
expected future loss against the strongest robust baseline — but do not have
to obtain evidence in the same way. A local statistical candidate is supported
primarily by this series' rolling folds. A TSFM or pinned remote model may also
carry a versioned external prior compiled from held-out series. External
evidence is never renamed local validation.

The default policy remains `strict`. `evidence_weighted` is opt-in and requires
an explicit evidence registry. Results are labelled `locally_validated`,
`externally_validated`, `jointly_validated`, `prior_assisted`, or
`baseline_fallback`. Uncertain transfer evidence produces an immutable
shrinkage blend; sufficiently strong independent evidence can make a
pretrained model primary. High overlap risk, an unpinned adapter, invalid
output, or absent evidence cannot do so.

Registry matches retain their transfer relevance. Evidence scoped to an exact
temporal regime receives full weight; each wildcard dimension reduces both
its expected transferable gain and effective precision. A broad prior cannot
by itself claim external validation, and a contrary local holdout is recorded
as a conflict and reduces—not silently vetoes—prior assistance. These
semantics are identified as `evidence-weighted-v2` in executable receipts.

Admission reports component evidence rather than an opaque trust score: local
folds and losses, external sample and uncertainty, output diagnostics, and
conflicts. A compact deterministic reasoning frame organizes those facts for
an agent to explain; it cannot replace the fitted executable or rewrite the
context-free primary path.

### Process evidence versus forecast-path behavior

A point forecast describes the path Gnomon published; it does not reveal the
variance or seasonal structure of observations that have not occurred.
Temporal answers therefore keep deterministic `forecast_path_behavior`
separate from a calibrated `process_claim`. Unsupported process questions
retain a useful ranked hypothesis and recovery action, but are labelled weak
or unresolved rather than borrowing certainty from a smooth forecast array.

Context follows the same rule. It may produce a provenance-carrying
`conditional_answer` (“if this event holds …”), but it cannot mutate the
context-free primary forecast. Realized outcomes can later strengthen or
retire that conditional effect through tracking.

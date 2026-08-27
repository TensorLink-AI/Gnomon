# Publication modes

Gnomon separates the immutable numeric paths from the recommendation a human
reads first. This lets new qualitative information be useful without claiming
that it was historically validated.

`strict` is the governed default. Only the history-only primary or a candidate
that passed Gnomon's fold-safe admission may be recommended. An LLM ranking is
ignored in this mode.

`best_effort` may put a sealed `prior_assisted` conditional path first when a
governed executable earned that role or an explicit governed scenario
selection chose an eligible sealed path. A plain model-authored path is not
promoted merely because it exists. The
history-only primary remains present, the weaker support label remains attached,
and the recommendation is never automation-eligible. Use this mode for a human
who wants the best bounded answer available from new information.

The preferred prior-assisted path is `effect_composed`: an LLM or human cites
the supplied text and proposes a typed effect (shape, timing, scope, magnitude
distribution, and uncertainty basis). Gnomon—not the model—composes that effect
over the sealed primary. Supported shapes cover pulses, level and trend changes,
variance, ramps/recovery, bounds, seasonal amplitude/phase and cross-series
relationships. A full model-authored quantile path remains a compatibility lane,
is labelled `model_authored`, and has no additional authority.

Compiler confidence is descriptive metadata, not evidence authority. An
unparseable model-generated confidence field therefore retains an otherwise
verbatim grounded claim at a disclosed conservative floor; explicit numeric
values outside the valid range are still rejected.

For long regular horizons, the compiler receives the exact grid boundaries,
step size, and count instead of a repeated timestamp array. Any returned
anchor must still land exactly on the host grid; irregular grids are never
compressed under a false regularity assumption.

When a safe declarative transformation cites its semantic rule but an explicit
AST literal appears in another supplied source line, the host may attach that
line verbatim as a parameter claim. It never invents a value or edits the AST;
ordinary unit, entailment, replay, support, and automation gates still apply.
Likewise, a missing literal unit may bind only from an exact source-adjacent
unit already declared elsewhere in the transformation. Ambiguous, absent, or
undeclared units remain a typed rejection.
Future driver schedules may use timestamp/value rows for readability. Gnomon
removes the timestamp wrapper only after proving exact, complete host-grid
identity; it never fills or interpolates a model-supplied driver series.
Piecewise-constant schedules may instead provide a cited initial value and
change points. Exact timestamps—or clock times matching exactly one host-grid
instant—are resolved and forward-filled by Gnomon; ambiguity is a rejection.
Exact identity operations in model-authored arithmetic are normalized away
before validation. This reduces schema noise but does not exempt any material
constant from source entailment or unit checks.
Repair is lane-prioritized: a proposed transformation receives the bounded
repair budget for its own typed violations; optional malformed side proposals
cannot starve it. Material derived constants must remain expressed through
their cited literals and operations.
Transformation repair is transactional: it may replace the rejected AST set
and append new verbatim claims, but it cannot delete or rewrite prior evidence
or alter unrelated dossier lanes.
An undated transformation specification is bound to the host cutoff rather
than discarded or represented as a future event. This repair applies only to
claims supporting a proposed safe transformation; valid windows are unchanged.
Within an explicit `series / literal` normalization, a single verbatim unit
adjacent to the denominator may type both operands so the ratio is
dimensionless. Gnomon does not create the ratio or choose its unit.

`scenario` returns the primary and each bounded conditional path. A governed
LLM selection may rank their immutable IDs and recommend one, citing verified
claims, counterevidence, confidence, and what would change its selection. The
selection cannot contain forecast numbers, change support, conceal the primary,
or authorize automation.

Recommendation and automation are deliberately separate. Automation requires
an explicit policy with a `policy_id`, `authorize: true`, and a supported
minimum tier; a prior-assisted scenario is never automation-eligible regardless
of that policy.

CLI example:

```bash
gnomon forecast data.csv --time timestamp --target value --horizon 7 \
  --publication-mode best_effort --dossier dossier.json
```

Agents can use the same single forecast call without implementing Gnomon's
sealing format. Pass `--context-proposal proposal.json`, `--context-text`, and
`--context-known-at`; the proposal contains verified claims and an
`effect_proposal`. Invalid proposals receive typed violations and at most one
repair attempt (`effect_proposal_repair`). Context is therefore always used,
represented as a labelled scenario, or rejected with a reason.

For relationships that are precise enough to execute, the same boundary
accepts a small declarative transformation language. It is intentionally not
Python, SQL, or generated code. The allowed nodes are numeric literals,
primary or named future series, arithmetic, lag, difference, percent change,
rolling mean, clipping, and quantiles. Expressions are bounded to 48 nodes and
eight levels, checked for finite values and compatible units, content sealed,
and may be repaired once only in the field named by the first violation.
Two safe macros cover common formulas without generated code:
`linear_combination` derives coefficient conversion units for a cited
multi-input equation, while `recursive_linear` executes a cited ARX-style
recurrence. For the latter, Gnomon reloads the target and driver history through
the forecast's governed `as_of` snapshot, supplies the initial state itself,
feeds prior predicted outputs back recursively, and propagates the primary
interval width through the feedback terms. The caller supplies only cited
future driver values; model-authored future target lags are never trusted.
When a source names the variables and lags but does not provide coefficients,
`fit_recursive_linear` accepts only that bounded structure. Gnomon fits all
coefficients from the governed pre-cutoff history, evaluates them on an
expanding-origin replay against last-value, and admits the candidate only when
it clears the baseline with at least eight validation origins. Model-authored
coefficients are a typed error in this lane. Because the source specification
was not itself available at past origins, a winning fitted path is labelled
`retrospectively_validated`, remains visible beside the immutable primary, and
cannot authorize automation.
When a verified document explicitly defines historical driver ranges but the
structured column is encoded differently, callers may add
`historical_series_segments` with cited `{start, end, value}` rows. Gnomon
requires complete, non-overlapping coverage of the governed pre-cutoff grid
and exact source entailment for every endpoint and value. This is an explicit
representation bridge—not inferred normalization—and remains disclosed in the
candidate receipt.
Before a recurrence can lead even a human-facing best-effort recommendation,
Gnomon replays the fixed equation over timestamp-aligned pre-cutoff target and
driver histories. It must beat last-value on at least eight identical origins.
Failure or insufficient history keeps the path visible as a labelled scenario
but makes it selection-ineligible; structural plausibility alone is not skill.

Every transformation cites verified claim IDs and a timezone-aware knowledge
time. A referenced future series must be supplied as `{values, known_at,
source_claim_id}` and must have been knowable by the forecast cutoff. The three
candidate lanes carry different authority:

- `historically_testable` requires per-origin knowledge checks and decisive
  out-of-sample evidence before strict publication can admit it;
- `prior_assisted` may lead a human-facing best-effort answer but never
  automation; historically replayable recurrences must first pass the replay
  gate above;
- `scenario_only` is a bounded conditional path, not a probability claim.

CLI callers pass repeatable `--context-transformation` JSON files with
`--context-known-at`. MCP callers use
`context_submission.transformations`. Invalid inputs remain in
`context_dispositions` with typed violations; they are never silently ignored.
A model-authored path offered as the fallback for a transformation whose
derivation fails remains visible and outcome-trackable in the scenario
portfolio, but is marked selection-ineligible. A seal proves identity, not the
correctness of a failed derivation.

A dossier may also contain up to six `hypotheses`. These are stable, sealed
interpretations—not forecasts—with kinds such as `relationship`,
`historical_analogue`, `regime_shift`, or `unsupported`. Each hypothesis must
cite a verified claim, resolve its series and knowledge time, and survives
reordering under a content-derived ID. A single bounded repair may replace
rejected hypotheses; accepted hypotheses cannot be silently rewritten.

Historical observation semantics use the same dossier rather than a separate
cleaning path. When a verified source literally says that maintenance, an
outage, a stockout, or a reporting failure produced absent recorded activity,
an `observation_interpretation` may filter an exact value or a source-stated
recurring window from a copy of the pre-cutoff history. The raw observations
and immutable primary are never edited. Retained/excluded counts and any
transformed-floor normalization are sealed. The resulting empirical
counterfactual begins as a visible `prior_assisted` scenario. It can lead a
best-effort recommendation only after at least twelve expanding origins on
unaffected targets, a 10% win over the strongest fixed raw classical
comparator under both point MAE and fold-safe q10/q50/q90 pinball loss, and
probabilistic wins in two of three chronological blocks. The receipt labels
this as retrospective conditional replay—it does not pretend the source was
known at earlier origins—and automation remains ineligible.
For regular sub-daily histories, the fixed contest also includes median and
recent-value daily-phase candidates. Their phase values are re-fit from earlier
retained observations at every origin, so a cited recurring outage can preserve
independently validated intraday shape without using later observations.
Never-observed phases are an explicitly prior-assisted interpolation between
the nearest observed phases. Irregular grids do not enable these families.
If the winning executable improves both point and probabilistic replay in two
chronological blocks but misses the 10% strict margin, it may lead only in
`best_effort`. It remains labelled `prior_assisted`, requires human review, and
cannot authorize automation; `strict` continues to publish the primary.
If literal zero is absent but the observed history contains a sharply
separated, source-compatible near-zero component, Gnomon may expose filtering
that component as a prior-assisted sensitivity. The cluster split is
deterministic and disclosed, but it cannot validate itself because membership
was inferred from target outcomes. It therefore never auto-leads: a governed
selector or human must choose it explicitly, and automation remains forbidden.
An exact one-off sensor glitch uses the same governed observation lane. Its
source-stated start and duration define a half-open timestamp mask over a copy
of history; replay decides whether the resulting candidate has human-facing
value. The source data and primary are never repaired in place, and the mere
presence of a cited anomaly cannot authorize automation.

Numerical influence is earned separately. Vintage-aware exogenous regression,
expanding-origin lag selection, and leave-one-episode-out analogue evaluation
produce fitted conditional candidates. Only candidates that beat their stated
baseline out of sample may become the `best_effort` recommendation. Candidate
count is reflected in lag admission, full validation diagnostics are retained,
and neither a model confidence value nor a benchmark score upgrades support.
Even an admitted context candidate remains ineligible for automation until it
has passed the stricter ordinary product admission path.
Typed `temporary_pulse` effects are constant over their stated active window;
ramps require an explicit ramp shape. A `fraction_of_level` effect transforms
each quantile multiplicatively, so the conditional distribution follows the
cited multiplier rather than only shifting its median.
If a compiler calls an otherwise exact, bounded cited multiplier a
`custom_scenario`, validation canonicalizes the vague label to the executable
bounded shape. It does not infer a magnitude or timing that the citation lacks.
Numeric context that produces an empty initial dossier receives the single
bounded compiler-repair round: it must emerge as cited typed evidence, a sealed
conditional scenario, or an explicit unsupported hypothesis. An empty parse is
never silently treated as evidence that the context was irrelevant.
For clearly past-tense reference claims, a yearless month/day is resolved to
its most recent occurrence at or before the cutoff and the normalization is
retained in the receipt. Prospective partial dates are rejected.
Long prior-assisted paths can be supplied as ordered `quantile_anchors` at
meaningful turning-point timestamps. Gnomon validates every anchor and
linearly interpolates the sealed path on its own forecast grid. If the model
does not anchor a horizon edge, that edge remains the immutable primary and the
completion is recorded in `path_normalization`; Gnomon does not invent model
endpoints. Compactness grants no additional support or automation authority.
Sparse timestamped rows under the older `quantiles` key are treated as the same
anchor format under the identical validation and disclosure contract.
In `best_effort`, those anchors may reflect a model temporal/domain prior when
the rationale distinguishes prior-derived numbers from source-stated facts.
This is a human-facing conditional estimate, never historical admission; the
immutable primary remains beside it and automation stays disabled.

When a long requested horizon leaves only one full-horizon holdout, Gnomon
still refuses to rank incremental candidates. It may preserve demonstrated
seasonal shape by admitting the fixed `seasonal_naive` baseline over
`last_value`, but only from at least six non-overlapping seasonal probes, a 10%
mean-error margin, and wins in two of three chronological blocks. This remains
a degraded publication and discloses the probe evidence; it does not upgrade
support or authorize automation.

An exact additive sensor-calibration rule has its own narrow counterfactual.
If the source states the drift start, additive rate per hour, and exact repair
boundary, Gnomon corrects a copy of pre-cutoff history and fold-selects a fixed
classical family on that conditional series. This source-determined path is a
human-facing `best_effort` only: it remains `prior_assisted`, preserves the raw
history and primary forecast, and cannot authorize automation. Missing or
ambiguous rule components leave the executable inactive.

Every projection includes a compact `temporal_state` (level, trend, interval
width, seasonality/regime receipts, path shape, analogues, cross-series state,
conflicts and sufficiency). The governed selector ranks sealed candidates from
this state and cited claims; it cannot edit their numbers. The sidecar retains
the complete sealed candidate portfolio so all alternatives—not only the
winner—can be scored when outcomes arrive.

The canonical artifact is not rewritten. The projection is persisted beside it
as `<artifact-id>.<publication-seal>.publication.json`, with seals over the publication and every
scenario. When `--project` is supplied, the recommendation reuses Gnomon's
synthesis receipts and is scored against actuals independently of the primary.
Every sealed alternative is recorded as a shadow synthesis too, so realized
outcomes measure selection regret and candidate uplift rather than recording
only the displayed winner.

The default MCP response carries a compact, seal-linked decision projection:
the selection contract, recommendation and automation authority, context
dispositions, temporal state, and the path and seal of the complete receipt.
It does not repeat the same horizon arrays as the primary, recommendation,
portfolio, and scenario list. The complete publication remains verifiable at
`publication_path`; request `format: "full"` only when those arrays genuinely
need to cross the agent boundary.

The MCP `gnomon_forecast` tool exposes the same `publication_mode`,
`temporal_dossiers`, a compact raw `context_submission` object,
`scenario_selection`, and `automation_policy` fields. After inspecting a
returned `selection_contract`, an agent may instead call
`gnomon_select_scenario` with the publication path and a number-free ranking.
That follow-up reuses every scenario seal, leaves the original sidecar intact,
forces automation off, and records which publication seal it supersedes.

# Publication modes

Gnomon separates the immutable numeric paths from the recommendation a human
reads first. This lets new qualitative information be useful without claiming
that it was historically validated.

`strict` is the governed default. Only the history-only primary or a candidate
that passed Gnomon's fold-safe admission may be recommended. An LLM ranking is
ignored in this mode.

`best_effort` may put a sealed `prior_assisted` conditional path first. The
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

Numerical influence is earned separately. Vintage-aware exogenous regression,
expanding-origin lag selection, and leave-one-episode-out analogue evaluation
produce fitted conditional candidates. Only candidates that beat their stated
baseline out of sample may become the `best_effort` recommendation. Candidate
count is reflected in lag admission, full validation diagnostics are retained,
and neither a model confidence value nor a benchmark score upgrades support.
Even an admitted context candidate remains ineligible for automation until it
has passed the stricter ordinary product admission path.

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

# Integration plan — external design review, 2026-08

How the review's proposals land in this codebase: what changes, what it
costs, what breaks, and in what order. Nothing here is implemented yet.

This document is written against the repository as of
`56f75a3` (runtime 0.4.0, schema 0.1). Every claim about current
behaviour below was checked against the source, not inferred from the
docs; where the review's premise and the code disagree, the code wins and
the disagreement is stated.

---

## 0. Invariants this plan is checked against

Read once; every design below is graded against these.

| # | Invariant | Where it comes from |
| --- | --- | --- |
| I1 | `src/aion` is stdlib-only. Heavy deps live in benchmark adapters and the sandboxed TSFM tier. | `pyproject.toml` (`dependencies = []`), `tsfm_sandbox.py` |
| I2 | Every number is computed deterministically or is absent. No LLM output becomes a forecast value, interval, metric, or selection decision. | `README.md`, `context.py` docstring |
| I3 | Every adapter and modelling decision is disclosed — in the artifact, in evidence, and in the module docstring. | `benchmarks/README.md` ground rule 3, every `src/aion` docstring |
| I4 | Evaluation partitions stay disjoint: selection folds → calibration fold → report-only test fold. | `DESIGN_REVIEW_NOTES.md` §1, `evaluation.evaluate()` |
| I5 | Abstention is a first-class outcome with typed reasons and repair options. | `support.py`, `contracts.REPAIR_OPTIONS` |
| I6 | The v0.2 surface is frozen. New capability = new fields/tools/files, never changed semantics of an existing one. | `COMPATIBILITY.md` |
| I7 | Leakage is structural: all reads go through a `Snapshot` bound to an `as_of`, and every read is logged. | `temporal_store.py` |
| I8 | Artifact IDs are content-addressed over inputs + parameters + `AION_VERSION`. Identical tasks share an ID; artifact writes are idempotent. | `ids.content_id`, `runtime.forecast` |

**I6 in practice.** Three things are byte-frozen and cost a version-salt
bump plus a golden refresh to change: the numeric output of the forecast
path, the six columns of `forecast.csv`, and the response envelope keys in
`toolspec.forecast_summary`. Everything else in this plan is additive:
new dataclass fields with defaults, new dict keys, new evidence kinds, new
files inside the artifact directory, new CLI flags.

**I8 in practice.** Any change to the numerics changes the *numbers* but
not the *IDs* — IDs hash inputs and parameters, never outputs. So a
numeric change silently produces a different forecast under the same ID
unless `AION_VERSION` is bumped. **Every phase below that changes numerics
must bump `ids.AION_VERSION` and record the refresh in `COMPATIBILITY.md`,
in the same commit.** This is the single easiest way to break the
harness's core promise, so it appears in every phase's acceptance
criteria.

---

## 1. What the code says that the review does not

Seven findings that change the shape of the work. Each is a fact about
the current source, verified.

### F1 — `interval_bounds` really does double-widen, and worse than described

`evaluation.evaluate()` pools residuals across *every selection fold plus
the calibration fold* (`evaluation.py:450-460`). Each fold contributes
residuals for lead times 1..h. The pooled sample is therefore already a
mixture over all lead times — it already contains the horizon-h spread.
`interval_bounds` then multiplies the spread around the median by
`step ** 0.5` (`evaluation.py:51-61`).

The consequence is not "slightly wide". It is wrong in *both* directions:

- at **step 1** the interval is too wide, because it uses a residual
  distribution polluted by lead times 2..h and then applies no shrinkage;
- at **step h** it is too wide again, because the sqrt factor is applied
  on top of a distribution that already includes step-h errors.

The golden `two_series_h5` shows it exactly: interval widths 5.898,
8.341, 10.215, 11.796, 13.188 — precisely `w1 * sqrt(step)`.

This makes B1 (per-lead conformal) not an enhancement but a **bug fix**,
and it should be sequenced accordingly. It also means the review's
question "audit for double-widening" is answered: yes, and the fix is the
same code that delivers per-lead conformal.

### F2 — four calibration paths disagree with each other

| Path | Residual source | Widening | Coverage rule |
| --- | --- | --- | --- |
| `evaluation.evaluate` | selection folds + calibration fold, pooled | `sqrt(step)` | `interval_bounds` |
| `context_eval.assess_context` | calibration fold only (h residuals) | `sqrt(step)` | `interval_bounds` |
| `covariates.assess_covariates` | calibration fold only | **none** | `predicted + q10 ≤ actual ≤ predicted + q90` |
| `adjudication._finalise` (combined) | calibration fold only | `sqrt(step)` | `interval_bounds` |

So the covariate gate measures coverage under a *narrower* rule than the
one Aion publishes, which biases its coverage number down relative to the
context gate's, and the two gates' coverage figures are not comparable —
yet the adjudication ladder ranks their outputs against each other. Any
conformal work must unify these four onto one helper, and that unification
is a prerequisite for B5 (interval-aware coverage guard) meaning anything.

### F3 — `EvaluationConfig` is dead code

`config.EvaluationConfig` declares `pool_residuals`, `temporal_scaling`,
and `target_coverage`, and `aion.yaml.example` documents them under
`evaluation.uncertainty` with prose ("interval_width *= sqrt(step)").
Nothing reads them: `grep -rn "temporal_scaling\|pool_residuals\|target_coverage" src/` returns
only `config.py` itself. `evaluation.evaluate()` only ever touches
`config.ensemble`, `config.meta_model`, `config.backends.api`, and
`config.models.tsfm_candidates`.

This is good news: there is a documented, already-shipped config surface
for exactly the uncertainty knobs B1–B4 need, and wiring it is a
disclosure improvement rather than a new abstraction.

### F4 — the context gate's cliff is steeper than the five conditions

The review targets the five stacked conditions. There are actually
**three** hard cliffs before those conditions are ever evaluated
(`context_eval.py:110-142`):

1. timezone-naive dataset → `considered=False` (context never even runs);
2. base evaluation unsupported → `considered=False`;
3. **fewer than four rolling origins → rejected outright**, with no
   ablation attempted.

Cliff 3 is the one that bites hardest in practice: `benchmarks/cik` tasks
and TemporalBench T4 rows are frequently short enough that the base
evaluation runs in *degraded* mode (2–3 folds), which means the context
gate refuses before any evidence is examined. Shrinkage admission (A5)
does not help here, because there is nothing to shrink. Any real
improvement in admission rate must either lower cliff 3 or accept that a
large fraction of benchmark tasks can never admit context. This is
addressed in A5 as a shrinkage-with-few-folds rule, and it is an open
question (Q3).

### F5 — the combined-enrichment path hard-codes the level-shift effect

`adjudication.combined_prediction` calls `context_model.event_effect()`
directly (`adjudication.py:164`, and again at `:305` for the final
horizon). It does not go through the context candidate. The moment A3
introduces effect shapes, the combined candidate would silently keep using
a level shift while the context candidate uses a decayed transient — the
ladder would then be comparing a shape it never disclosed. A3 must
generalise this call site or the disclosure invariant (I3) breaks
quietly.

### F6 — `pytest -q` does not fail here

The review lists a pytest 9 collection failure caused by the root
`__init__.py` Hermes shim. It does not reproduce on this checkout:

```
$ pytest -q            # pytest 9.0.2 / 9.1.1, Python 3.11.15, repo root
421 passed in 8.22s
```

It passes under `--import-mode=importlib`, under
`-o consider_namespace_packages=true`, and when invoked as
`pytest -q Aion` from the parent directory. The **only** way I could
produce the 27 collection errors was
`pytest --rootdir=/home/user`, which discards the repo's
`[tool.pytest.ini_options]` (and therefore `pythonpath = ["src"]`), giving
`ModuleNotFoundError: No module named 'aion'` — a misinvocation, unrelated
to the root shim. CI already runs 3.11/3.12/3.13 with an unpinned pytest
(`.github/workflows/ci.yml`).

Recommendation: **do not pin `pytest<9`.** Do the cheap defensive work
(D1) and a regression test that pins the invocation, and treat the
reported failure as environment-specific until someone reproduces it with
a version and command line.

### F7 — the review's "leakage-trap family" is half-built already

`src/aion/episodes.py` already ships `make_leakage_trap_episode` (planted
post-cutoff vintages plus a late-published revision), `grade_episode`
already reads leakage directly out of `snapshot_access` evidence via
`_max_known_time`, and `aion eval episodes` already feeds
`aion eval compare`. What is missing for C2 is the *LLM control condition*,
the *measurable leak gain* property, and the *negative control* that
proves the assertion has teeth. C2 is therefore an extension of
`episodes.py` plus a new benchmark adapter, not a green-field build.

---

## 2. Per-proposal design

Effort key: **S** ≈ 1 focused PR, **M** ≈ 2–4 PRs, **L** ≈ a phase of its
own. Every design states the modules touched, the contract deltas with
their `schema_version` handling, the tests, and what could break.

---

### A. Context admission gate

#### A1 — Constraint events

**Decision: `attributes` schema, not an `event_type` namespace.**

`event_type` is free-form text produced by users and LLMs, and A4 makes it
the *pooling key* for analog effects. Reserving a prefix (`constraint:…`)
inside it would collide with that key space and, worse, would make an LLM
string load-bearing for control flow — a soft violation of I2. Instead:
`ContextEvent.attributes` gains a reserved, validated key.

New module `src/aion/context_claims.py` (stdlib, ~150 lines):

```python
CLAIM_KINDS = frozenset({"effect", "constraint", "magnitude"})   # default: effect

@dataclass(frozen=True)
class ConstraintClaim:
    bound: str            # min | max | equals | nondecreasing | nonincreasing
    value: float | None   # required for min/max/equals, absent for monotone
    scope: str = "horizon"   # horizon | event_window
```

- `parse_claim(event) -> ConstraintClaim | MagnitudeClaim | None` — returns
  `None` when `attributes` carries no `kind` (i.e. every event file that
  exists today), so this is a pure extension of the events file format at
  `schema_version: "0.1"`. No migration.
- `validate_context_event()` gains claim validation, appended to the
  existing `problems` list. Malformed claims fail the whole file loudly,
  matching `load_events_file`'s existing "never silently drop a proposal"
  rule.

**Where the clamp is applied.** In `pipeline.interval_stage`, as a
deterministic projection of the already-computed rows — never as an
adjustment to a model. One function:

```python
def project_rows(rows, constraints) -> tuple[list[dict], list[dict]]:
    """Returns (projected rows, per-clamp disclosure records)."""
```

Order of operations, and why it is stable:

1. bounds (`min`/`max`/`equals`) applied elementwise to `point`, `q10`,
   `q50`, `q90` (and every additional level once B3 lands);
2. monotone claims applied as a running max (`nondecreasing`) or running
   min (`nonincreasing`) **per quantile path independently**;
3. no re-application needed: a running max of values already ≤ `max` stays
   ≤ `max`, and ≥ `min` stays ≥ `min`, so step 2 preserves step 1.

Because every projection is monotone non-decreasing in its input, the
quantile ordering `q10 ≤ q50 ≤ q90` is preserved by construction. That is
the property that makes this safe to apply after interval construction,
and it should be asserted in a test.

**Threshold analysis must be projected too.** `threshold_analysis_stage`
does *not* read the rows — it recomputes exceedance probabilities from the
raw residual cloud (`pipeline.py:359-392`). Left alone, a max-clamped
forecast would still report a positive probability of exceeding a
threshold above the clamp. Fix: apply the same `project_value` to each
simulated draw before the comparison, and extend the `basis` string to
name the projection. This is the single most likely place for A1 to ship
a lie, so it gets its own test.

**Rejecting a constraint violated in history.** A claim contradicted by
the data is not verified, and clamping to it would fabricate. Rule,
evaluated per fold cutoff against only the data visible at that cutoff
(I7):

- `min`/`max`/`equals`: reject if **any** visible observation violates the
  bound (no tolerance — a "capacity is 500" claim contradicted by an
  observed 520 is simply false);
- `nondecreasing`/`nonincreasing`: reject if the violation *rate* over
  visible consecutive pairs exceeds a fixed `MONOTONE_VIOLATION_LIMIT`
  (proposed 0.0 for a hard claim; see Q1).

Rejected constraints appear in `events_excluded` with reason
`constraint_violated_in_history`, and carry the offending timestamp and
value so the disclosure is checkable.

**Constraints do not enter selection.** A constraint is a verified fact
about the world, not a model, so it is not admitted by demonstrated lift —
it is admitted by (i) structural validity and (ii) not-violated-in-history.
Its *effect* is still measured for disclosure: the selection folds are
replayed once with and once without the projection, and the mean fold
delta is recorded as `constraint_effect` evidence. Measured, disclosed,
never a gate.

**Support impact.** A binding clamp is a disclosed deterministic
projection of a verified claim, so it is a `note`, not a warning — it does
not downgrade support. Exception: when the clamp binds on more than half
the horizon steps, the published path is mostly determined by the claim
rather than by the data, and that earns a `warning` (hence
`weakly_supported`). Threshold value is Q1.

**Contracts.**

- `SeriesResult.context` (already a free-form dict) gains
  `constraints_applied: [{event_id, bound, value, steps_bound, binding_fraction}]`
  and `constraints_rejected: [...]` — additive keys inside an existing
  dict, no `schema_version` change.
- New evidence kind `context_constraints`.
- `capabilities().features` gains `constraint_events: true`.

**Tests.** `tests/test_context_constraints.py`: claim parsing and
round-trip through `event_to_dict`/`event_from_dict`; quantile ordering
preserved under every bound and monotone combination; a max-clamp
suppresses the exceedance probability above it in `threshold_analysis`;
a constraint violated in visible history is rejected and *not* applied; a
constraint valid at fold 3 but violated by data visible at fold 5 is
applied at 3 and rejected at 5 (the fold-local rule); binding-fraction >
0.5 downgrades to `weakly_supported`.

**Risks.** (a) The threshold-analysis coupling above. (b) `operators.py`
has its own interval construction path (`operators.py:449-478`) used by
the macros; if constraints are not threaded there, `aion decide` and
`aion monitor` will disagree with `aion forecast` on the same task. Must
be included in scope. (c) Constraints could be used to rescue an
abstention — explicitly forbidden: projection runs only when
`assessment.supported` is already true.

**Effort: M.**

#### A2 — Known-future covariate magnitudes

**Bridge, don't duplicate.** New function in `src/aion/context_claims.py`:

```python
def covariate_dataset_from_events(
    events, *, series_names, grid, frequency, base: CovariateDataset | None,
) -> CovariateDataset | None
```

Rules, all deterministic:

- only events whose claim kind is `magnitude` participate;
- one column per `(event_type, variable)`, named
  `evt_<event_type>__<variable>`, type `continuous`, availability
  `future_known`;
- the value is the *claimed magnitude* (relative claims carry the relative
  number, e.g. `0.05` for "+5%"), placed on every grid timestamp inside
  `[effective_start, effective_end]` and `0.0` everywhere else, so
  `covariate_forecast` always finds a value and never returns `None` for a
  missing vintage;
- `CovariateRow.known_at = event.known_at` — the existing fold-safe
  snapshot path (`CovariateDataset._snapshot`) then enforces leakage with
  no new machinery;
- **Aion still estimates the coefficient.** The LLM supplies the shape of
  the regressor; `covariates._solve` supplies the number. This is the
  entire point and it should be stated in the module docstring (I3).

When the caller *also* supplies a covariates file, the synthesized rows
and specs are concatenated into one `CovariateDataset`; a name collision
raises a new error `DUPLICATE_COVARIATE_NAME` with a repair option. The
synthesized dataset carries
`path = "context-events://<n> events"` and
`fingerprint = content_id("context_covariates", …)` so evidence and the
ID payload stay well-defined.

**Contracts.** `covariate_ablation` evidence gains
`derived_from_events: [event_id, …]`. `capabilities().features` gains
`event_magnitude_covariates: true`. No frozen shape touched.

**Tests.** A magnitude event reproduces, byte for byte, the artifact
produced by hand-writing the equivalent covariates CSV (this is the
"no duplicated contract" assertion); an event whose `known_at` postdates a
fold cutoff contributes zeros at that fold and non-zeros later; collision
error and repair option.

**Risks.** Widening the regressor matrix can trip
`covariate_forecast`'s `len(matrix) < max(6, width + 1)` guard and return
`None`, which the gate reports as "missing point-in-time values" — a
misleading reason. Add a distinct rejection reason for
`insufficient_rows_for_regressors`.

**Effort: M.**

#### A3 — Effect shapes

Extend `context_model.py` from one function to a small, closed,
deterministic family. Each shape is a pure function with the current
signature plus fixed parameters:

| Shape | Parameters | Grid |
| --- | --- | --- |
| `level_shift` | — | (today's `event_adjusted`, unchanged) |
| `transient_decay` | half-life `k` periods | `{1, 2, 4}` |
| `lead_lag_ramp` | lead `l`, lag `m` | `{0,1,2} × {0,1,2}` minus `(0,0)` |

Fixed small grids, never continuous optimisation — the same reasoning that
makes `detect_anomalies` use a seeded injection grid rather than a search.

**Selection is the existing identical-fold ablation.** `assess_context`
scores every shape on the same selection folds it already replays; the
best mean fold score wins, ties broken by fewest parameters then fixed
order with `level_shift` first. The gate against the base is structurally
unchanged.

**Compatibility call: the model name stays `"event_adjusted"`.**
`CONTEXT_MODEL_NAME` appears in `selected_model`, in
`capabilities().models.context`, and as the grouping key in
`tracking.model_performance`. Splitting it into four names would fragment
every existing realised-performance history. The shape is disclosed in
`context.effect_shape` and in evidence instead.

**F5 must be fixed in the same PR.** `adjudication.combined_prediction`
and `adjudication._finalise` call `event_effect()` directly; both must
route through a shared `context_model.apply_shape(shape, …)` so the
combined candidate uses the shape the context candidate selected.

**Contracts.** `context` dict gains `effect_shape: {name, params,
fold_scores, candidates_scored}`. Additive.

**Tests.** Each shape recovers its own planted effect on a synthetic
series (the existing `test_context_pipeline` fixture style); a decaying
transient beats a level shift on decaying data and loses on step data;
tie-break determinism across 100 permutations of candidate order; the
combined candidate in `adjudication` uses the selected shape (assert on
evidence, not on internals).

**Risks.** More candidates on the same folds = more selection
overfitting. Mitigated by the `λ_looks` term in A5, which must therefore
land with or before A3 — see the phase plan.

**Effort: M.**

#### A4 — Analog pooling

Two sources, with sharply different determinism properties. Treat them
separately.

**(a) Cross-series, same dataset — default on.** For a panel input, the
effect of an event type absent from series *s* is estimated on the other
series in the same `LoadedDataset`, using only observations visible at the
fold cutoff. Fully deterministic, no external state, replayable, and
already inside the snapshot. This is the version that should ship first.

**(b) Cross-run, from tracking — default off.** Previously
admitted-and-scored events, stored locally. This makes a forecast depend
on machine-local state, which collides with I8: the same input and
parameters would yield different numbers on two machines under the same
artifact ID. Resolution: the analog pool's fingerprint enters the ID
payload, the exact rows used are recorded in evidence, and the feature is
opt-in (`--analogs store`). The artifact then remains reproducible *given
the same pool*, and states which pool it used.

**Storage.** Tracking store (`registry.db`), schema v4 — `temporal_store`
is for observations, not for estimated effects. New table:

```sql
CREATE TABLE IF NOT EXISTS context_effects (
    effect_id        TEXT PRIMARY KEY,   -- content_id over the row
    event_type       TEXT NOT NULL,      -- the pooling key
    dataset_ref      TEXT NOT NULL,      -- source fingerprint
    series_ref       TEXT NOT NULL,      -- series fingerprint (tracking already has these)
    shape            TEXT NOT NULL,
    effect_scaled    REAL NOT NULL,      -- effect / series scale, so pooling is unit-free
    n_occurrences    INTEGER NOT NULL,
    fold_cutoff      TEXT NOT NULL,      -- when it was estimated
    outcome_known_at TEXT NOT NULL,      -- when the outcome that scored it became known
    source_forecast_id TEXT NOT NULL,
    created_at       TEXT NOT NULL
);
```

Migration follows the existing in-place pattern (`tracking._init_db`
already does `ALTER TABLE … ADD COLUMN` guarded by a `schema_metadata`
version), so every v0.2 tracking command keeps working.

**Leakage rules — the whole ballgame.** An analog row is usable at fold
cutoff `C` of a run replayed at `as_of A` **iff**:

1. `outcome_known_at <= min(C, A)` — its own outcome was known by then.
   Estimation date is not enough; an effect estimated early but scored late
   was not knowable;
2. `series_ref != ` the series being forecast (no self-pooling);
3. for source (b), `created_at <= A` as well, so `--as-of` replay cannot
   see pool rows written after the replay instant.

Rules 1 and 3 are what make analog pooling honest, and they are exactly
the bitemporal discipline the store already enforces for observations —
this is reusing the repo's pattern, not importing a new one.

**Shrinkage.** With `n_own` own-series occurrences and a pool of `n_pool`
analogs with mean `ē` and spread `s_pool`:

```
w      = n_own / (n_own + K)          # K fixed, proposed 3
effect = w * ê_own + (1 - w) * ē      # n_own = 0 → pure pool estimate
```

The A5 shrinkage factor then applies on top, and will be small in exactly
the `n_own = 0` case, because there is no own-series fold evidence of
lift. The two mechanisms compose correctly, which is the argument for
doing A5 first.

**Contracts.** `context` dict gains
`analogs: {source, pool_size, pool_mean, weight, rows_used: [effect_id…]}`.
New evidence kind `context_analogs`. New CLI flag `--analogs
none|dataset|store` (default `dataset`).

**Tests.** Pooling across a two-series panel where the event occurs only
in series B recovers B's effect for series A with the right weight; an
analog whose `outcome_known_at` postdates the fold cutoff is excluded (and
the exclusion is in evidence); `--as-of` replay with a pool row written
after the replay instant produces byte-identical output to a run with an
empty pool; tracking migrates v3 → v4 in place and every v3 command still
passes.

**Risks.** Highest-risk item in the plan. Pool poisoning (a badly
estimated effect propagating), unit mismatch across datasets (mitigated by
`effect_scaled`), and the determinism collision above. This is why it is
Phase 4.

**Effort: L.**

#### A5 — Shrinkage admission

Replace admit/reject with a deterministic factor λ ∈ [0, 1], and publish
`base + λ · (context − base)` pointwise. λ = 0 reproduces today's
rejection exactly; λ = 1 reproduces today's admission exactly. The current
gate is the special case λ ∈ {0, 1}, which is the argument that this is a
generalisation rather than a replacement.

**The factor.** All inputs are already computed by `assess_context`.
With per-fold symmetric improvements `x₁…x_n` (already bounded to [−1, 1]),
margin `m₀ = minimum_baseline_improvement`, `x̄ = mean(x)`,
`se = stdev(x)/sqrt(n)`:

```
λ_evidence  = max(0, (x̄ - m₀) / (x̄ - m₀ + se))        # 0 when x̄ ≤ m₀; → 1 as se → 0
λ_stability = clamp(mean(sorted(x)[:-1]) / x̄, 0, 1)     # gain not confined to one fold
λ_majority  = clamp(2 * (#{x > 0} / n) - 1, 0, 1)       # majority of folds improved
λ_coverage  = 1 if the binomial upper bound on degradation ≤ limit,
              else linear to 0 at 2 × limit             # B5 feeds this
λ_looks     = n / (n + k - 1)                           # k = shapes scored on the same folds

λ = round(λ_evidence · λ_stability · λ_majority · λ_coverage · λ_looks, 6)
```

Each of the five current conditions becomes one factor, so the reasons
list keeps naming the same five things — it just reports a number instead
of a veto. `λ_looks` is the price of A3: scoring four shapes on the same
folds costs shrinkage, which is the honest accounting for a multiple-looks
problem.

**Calibration must follow the blend.** The published path is the blended
path, so residuals and coverage must be recomputed on the calibration and
test folds *with λ applied*, not inherited from the unblended context
candidate. Getting this wrong publishes an interval that does not belong
to the published points.

**Few-fold rule (F4).** With `n < 2`, `se` is undefined and λ = 0 — i.e.
today's behaviour. Cliff 3 (fewer than four rolling origins → refuse
before ablation) is *not* removed by A5, and should not be: with two
selection folds there is no leave-one-best-out check. Lowering it is Q3.

**Model name.** Stays `"event_adjusted"`; λ is disclosed. Reasoning as in
A3 — a name change fragments the tracking leaderboard. Q2.

**Contracts.** `context` gains
`shrinkage: {lambda, factors: {evidence, stability, majority, coverage, looks}, inputs: {…}}`.
`admitted` stays a boolean and is defined as `λ > 0`, so every existing
reader keeps working. Goldens are unaffected — none of the five golden
cases supplies context events (verified).

**Tests.** λ = 1 on a fixture that admits today, byte-identical output;
λ = 0 on a fixture that rejects today, byte-identical output; monotone λ
under increasing planted effect size; λ present in the artifact and in the
tool response for every considered run; blended residuals match the
blended points (assert coverage is computed from the same path that is
published).

**Effort: M.**

#### A6 — Conditional (future-only) path

The requirement is a second forecast that is clearly *not* the answer,
alongside the one that is, without touching the frozen shape.

**In the artifact.** `SeriesResult` gains
`conditional: dict | None = None` (additive field with a default, so every
existing golden and reader is unaffected):

```json
{
  "basis": "conditional_on_event",
  "events": ["launch-2026-09"],
  "assumptions": ["The launch occurs on 2026-09-01 as stated.",
                  "The effect is estimated from 4 analogous events in other series."],
  "effect_shape": {"name": "level_shift", "params": {}},
  "support_assessment": {"status": "conditionally_supported", ...},
  "forecast": [{"timestamp": "...", "point": ..., "q10": ..., "q50": ..., "q90": ...}]
}
```

The **unconditional** forecast stays in `forecast` and remains the only
thing in `forecast.csv`. The conditional path gets its own file,
`forecast_conditional.csv`, added to the artifact directory — a new file
alongside the four frozen ones, changing none of them.

**Result-level `support` is unchanged by the presence of a conditional
block.** Anything else would change the semantics of a frozen field. The
conditional block carries its own `support_assessment`, always
`conditionally_supported`, with the assumptions enumerated.

**In the tool surface.** `toolspec.forecast_summary` results gain
`conditional_preview` and `conditional_rows` — additive keys in the
envelope, exactly the pattern `notes` used.

**Gating.** Opt-in for the first release: `--conditional` (CLI),
`conditional: true` (tool argument), default false. Emitting it
automatically would change the artifact bytes of every existing
context run for no request. Flipping the default is a later, separate
decision (Q4).

**Which events qualify.** Events that fail *backtest admissibility* for a
reason that a conditional forecast can honestly carry:

- no verifiable source (`backtest_admissible` false) → conditional with
  the assumption named;
- event type never occurred in this series and no analog is available →
  conditional with the effect stated as assumed, not measured;
- λ = 0 because the evidence did not clear the margin → **not**
  conditional. The gate looked, and found nothing. A conditional forecast
  here would be laundering a rejected effect.

That last exclusion is the one that keeps A6 from becoming a bypass.

**Tests.** Round-trip of the new field through `asdict` → JSON →
`read_artifact`; absent field ⇒ byte-identical to today (golden
protection); a λ = 0 event produces no conditional block; the conditional
CSV and the conditional rows agree; the claim verifier
(`verifier.verify_or_raise`) accepts the conditional block's claims as
`predictive` with explicit assumptions.

**Risks.** The claim verifier and `lineage.build_forecast_lineage` both
walk the result structure; both need to learn about the new block or the
conditional numbers will be unlineaged — which would be an
invented-number failure by the harness's own definition.

**Effort: M.**

#### A7 — Gate instrumentation

**In the runtime.** One new evidence kind, `context_gate`, per series:

```json
{"admission_rate": 0.5, "events_considered": 4, "events_admitted": 2,
 "decisions": [
   {"event_id": "...", "event_type": "promotion", "eligible": true,
    "backtest_admissible": true, "claim_kind": "effect",
    "shape": "transient_decay", "lambda": 0.62,
    "reasons": [...], "analog_used": false}]}
```

Run-level admission rate also lands in `capabilities`-free territory: it
is data, so it goes in evidence, and `summary.md` renders a one-line
`- Context: 2 of 4 events admitted (mean shrinkage 0.62)`.

**In the adapters — the oracle condition.** The oracle must **not** be a
runtime bypass. There must be no environment variable or flag inside
`src/aion` that force-admits an event; that would put a switch next to the
gate, and switches next to gates get flipped. Instead the oracle is
computed adapter-side: the adapter takes Aion's unconditional forecast and
adds the known-true effect itself, scores that, and reports it as a
separate condition clearly labelled as an upper bound that Aion never
produced.

Measured, per benchmark where ground truth exists (TemporalBench T4, the
synthetic leakage/effect tasks, CiK constraint tasks):

- **gate precision** = admitted events with a non-zero true effect ÷
  admitted events;
- **gate recall** = admitted events with a non-zero true effect ÷ all
  events with a non-zero true effect;
- **headroom** = oracle metric − treatment metric, in the benchmark's own
  metric.

**Effort: S** for the runtime evidence, **M** including the adapter
conditions.

---

### B. Uncertainty

#### B1 — Per-lead-time residuals + split conformal

**The additive shape.** `Evaluation.residuals` stays exactly as it is (a
flat pooled list) — it is read by `pipeline.interval_stage`,
`threshold_analysis_stage`, `predict_stage`'s ensemble branch,
`operators.py`, and all three enrichment paths. Changing its type is a
five-module refactor for no benefit. Instead `Evaluation` gains:

```python
residuals_by_lead: dict[int, list[float]] = field(default_factory=dict)
```

populated in the same loop that already builds `residuals`
(`evaluation.py:450-460`) — the lead index is `step`, which is already in
scope.

**New module `src/aion/conformal.py`** (stdlib, ~180 lines):

- `split_conformal_quantiles(residuals_by_lead, levels, *, min_per_lead)` —
  per lead `h`, the finite-sample-valid rank index
  `ceil((n+1)(1-α)) / n` rather than the plain empirical quantile, so the
  coverage guarantee is the conformal one and not an approximation of it;
- `isotonic_spread(spreads_by_lead)` — pool-adjacent-violators, ~25 lines
  of stdlib, giving a monotone non-decreasing spread-vs-`h` fit used when a
  lead has fewer than `min_per_lead` residuals (which is the common case
  with 2–4 folds);
- `interval_from_conformal(point, per_lead, step, levels)` — **no sqrt
  factor**. The widening now comes from the data.

**Wiring.** `pipeline.interval_stage` uses the conformal path when
`residuals_by_lead` is populated and the run is not degraded; otherwise it
falls back to `interval_bounds` unchanged. Consequence: the three
`degraded` goldens do not move at all, and only `two_series_h5` changes.
`interval_bounds` stays exported and unmodified, because
`context_eval`, `adjudication`, and `operators` call it.

**Then unify F2.** A second PR extracts one
`calibrate(residuals_by_lead) → per_lead_quantiles` helper and points
`evaluation`, `context_eval`, `covariates`, and `adjudication` at it, so
all four measure coverage under the rule Aion actually publishes. This is
where the covariate gate's non-widened coverage rule gets fixed.

**Config (F3).** Wire the already-documented, currently-dead
`evaluation.uncertainty` block: `method: conformal | legacy` (default
`conformal`), `pool_residuals`, `target_coverage`. When `method` is not
the default it enters the ID payload, exactly as `repair` does.

**Compatibility.** This changes numbers. Required in the same commit:
bump `ids.AION_VERSION`, refresh goldens with `pytest --update-goldens`,
review the diff, and add a `COMPATIBILITY.md` amendment stating that
interval *values* changed while the interval *contract* (q10/q50/q90 keys,
support semantics, IDs' format) did not.

**Tests.** Coverage on synthetic heteroscedastic data is within tolerance
of nominal at every lead (this is the test that fails today); step-1
intervals narrow relative to today and step-h intervals do not blow up;
isotonic fit is monotone and reproduces the dense-fold answer when folds
are dense; `interval_bounds` is byte-unchanged for its existing callers;
degraded runs produce byte-identical goldens.

**Effort: M.** The highest leverage-to-effort ratio in the plan.

#### B2 — Distributional selection loss

Add `pinball_loss(actual, quantile_paths, levels)` to `evaluation.py`
alongside `error_score`, which stays and stays reported.

**When it becomes the selection criterion.** "Whenever the task requests
quantiles" needs a definition, because every run already emits three
quantiles. Proposed definition: **when the caller explicitly passes
`--quantiles` / `quantiles=`.** Default invocations keep point-loss
selection and therefore keep their numbers; explicit quantile requests get
quantile-appropriate selection. This is the only definition I can find
that gives the review what it wants without silently re-selecting models
for every existing user.

**The hard part: candidates must produce quantiles per fold, without
leakage.** For fold `i`, a candidate's quantiles come from its own
per-lead conformal residuals estimated on folds `1..i-1` only (expanding
window). Fold 1 therefore has no quantile score and is excluded; pinball
selection needs ≥ 3 selection folds and abstains to point loss below
that, with a `note`. This nested calibration is the bulk of the work.

**Contracts.** `selection_scores` keeps its meaning (point loss).
New sibling key `selection_scores_pinball` in the rolling-evaluation
evidence and in `SeriesResult` (additive, `None` when not computed).
`ForecastSpec.quantiles`/`ForecastTask.quantiles` already exist — the
value in `task` changes only when the caller asks.

**Tests.** On a fixture where a wide-but-well-calibrated model beats a
sharp-but-biased one, pinball selects the former and point loss the
latter; both scores appear; < 3 folds falls back with the note; identical
fold partitioning to the point-loss path (assert on the evidence's
`fold_cutoffs`).

**Effort: L.** This is the largest single numeric change proposed.

#### B3 — More quantile levels

Nine levels: `0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95`.

- **Rows** gain `q05, q20, q30, q70, q80, q95` alongside the existing
  three. Additive dict keys; `forecast_preview` carries them automatically.
- **`forecast.csv`** keeps `series,timestamp,point,q10,q50,q90` as its
  first six columns *in that order*, with the new columns appended. Any
  reader that selects by name (including `tracking._load_forecast_csv`) is
  unaffected; positional readers see six familiar columns first. Recorded
  as a `COMPATIBILITY.md` amendment.
- **CLI** printing stays q10/q50/q90 unless `--quantiles` is passed.
- **CiK adapter** — the real prize. `samples_from_quantile_rows` currently
  interpolates through three points and clamps the tails at q10/q90, which
  the module docstring already admits "understates tail spread". With nine
  levels the piecewise-linear inverse CDF has eight segments and clamps at
  q05/q95. RCRPS is a CRPS variant, so understated tails are a direct,
  measurable penalty; this change should move the headline CiK number on
  its own, and the delta must be reported separately from any gate change
  so the two are not conflated.

**Tests.** Quantile monotonicity across all nine levels at every step;
CSV column order; the CiK sampler reproduces its three-level behaviour
exactly when handed three-level rows (so the adapter change is provably
scoped).

**Effort: M**, and it is gated on B1 — nine levels drawn from a
double-widened residual pool would just be nine wrong numbers.

#### B4 — Adaptive conformal via tracking

**State.** Tracking schema v5, table `interval_calibration`, keyed by
`(project, series_ref, model, horizon_bucket)`:

```
alpha            REAL      -- current effective miscoverage
updates_applied  INTEGER
last_scored_at   TEXT
known_at         TEXT      -- when this update became knowable == scored_at
```

Rows are **append-only**, one per update, each stamped with `known_at`.
That is what makes `--as-of` replay honest: a replay at `A` reads the
state as of the latest row with `known_at <= A`. This is the bitemporal
pattern the observation store already uses, applied to calibration state.

**Update rule (ACI).** On each scored horizon,
`α ← clip(α + γ (α_target − err), 0.005, 0.5)` where `err` is the realised
miscoverage indicator. `γ` fixed (proposed 0.05), in config. Updates
happen **only** in `submit_actuals`/`score_forecast` — never during a
forecast. A forecast reads state; it never writes it. That separation is
what keeps forecasting a pure function of (inputs, parameters, state
version).

**Determinism.** The state version read enters the ID payload and the
evidence, so the artifact records exactly which calibration state produced
its intervals, and two machines with the same state produce the same
bytes. Default **off** (`--adaptive-intervals`); on-by-default would make
the same file and parameters give different numbers on different
machines, which is I8's whole point.

**Tests.** A sequence of under-covering outcomes widens intervals
monotonically; replay at an `as_of` before an update reproduces the
pre-update interval byte-for-byte; forecasting never mutates the table
(assert row count); state version appears in evidence and in the ID
payload.

**Effort: M.**

#### B5 — Interval-aware coverage guard

Today: `assessment.coverage < base.coverage - 0.1`
(`context_eval.py:220`), comparing two proportions measured on `h` points
each — for `h = 7` that is a comparison of two numbers with a standard
error near 0.19. The guard fires on noise.

Fix: Wilson score intervals (closed form, stdlib) on both proportions;
trigger only when the **upper** bound of the context coverage lies below
`base.coverage − COVERAGE_DEGRADATION_LIMIT`. The same computation feeds
`λ_coverage` in A5, degrading linearly rather than vetoing. Once B1's
unification lands, the test-fold coverage sample grows beyond `h` points
and the interval tightens on its own.

**Contracts.** `context` gains
`coverage_interval: {lower, upper, n, method: "wilson"}`. Additive.

**Tests.** A one-point coverage difference on `h = 7` no longer fires; a
genuine collapse (0.9 → 0.2) still does; Wilson bounds match reference
values to 1e-9.

**Effort: S.**

---

### C. Benchmark methodology

#### C1 — Abstention-robust summaries

**Where the matching lives: a new `benchmarks/compare_conditions.py`, not
`aion eval compare`.**

Reasoning from the code: `agent_eval.compare_runs` operates on AionBench
rows, whose schema is `{task_id, success, …}` — booleans and counters. It
also enforces "baseline and treatment must contain identical task_id sets"
and raises otherwise. Matched-subset and penalized means need the
*benchmark's own metric* per task (RCRPS, sMAPE, F1), which lives in each
adapter's `summary.json`/`details/`, not in the frozen row schema. Forcing
it into `aion eval compare` would either change that CLI's frozen
semantics or smuggle metrics through `extra`. So:

- `benchmarks/common/summaries.py` — `matched_subset_mean(runs_a, runs_b,
  metric)` and `penalized_mean(runs, fallback_scores, metric)`;
- `benchmarks/compare_conditions.py` — a CLI taking two or more run
  directories, emitting `comparison.json` with, per metric: scored-only
  mean (today's number, unchanged), matched-subset mean, penalized mean,
  and the counts behind each;
- `aion eval compare` is **not touched**.

**The mandated fallback.** A penalized mean needs a per-task fallback
score. Each adapter gains a `fallback` condition that runs seasonal-naive
on exactly the same task set and writes `fallback_scores.json`. Without
it, the penalized mean has nothing honest to impute, and imputing a
constant would be a fabricated number.

**Every adapter's `summary.json` keeps every key it has today** and gains
`scored_mean_note`, `matched_subset` and `penalized` blocks (both `null`
until the corresponding inputs exist). The existing
`"note"` string about abstentions stays.

**Tests.** `benchmarks/tests/test_compare_conditions.py`: matched subset
of two conditions with disjoint abstention patterns; penalized mean equals
scored mean when there are no abstentions; the comparison refuses (loudly)
to report a penalized mean without fallback scores.

**Effort: M.**

#### C2 — Leakage-trap eval family (the flagship)

Building on F7: `episodes.py` already generates a leakage trap and grades
leakage from `snapshot_access` evidence. What C2 adds is the LLM control,
the *measurability* of the leak, and proof that the assertion has teeth.

**Task generator** (`benchmarks/leakage/tasks.py`). Each task is one CSV
in the shape of `examples/messy_requests_revisions.csv`
(`timestamp,value,published`), a cutoff, a horizon, and a hidden future.
Two properties are enforced **at generation time**, by construction and by
assertion:

1. **Leak gain is measurable.** Forecast the same task twice with the same
   deterministic Aion configuration — once from the final vintage
   (leaky) and once from the as-of-cutoff vintage (honest). The task is
   only emitted if the leaky run's error is lower by at least τ (proposed
   τ = 0.10 relative MASE). A "leakage trap" where peeking does not help
   measures nothing.
2. **Leakage is mechanically detectable.** Post-cutoff revisions and future
   rows carry planted marker offsets — values not derivable from
   pre-cutoff data. Any condition whose output reproduces a marker within
   ε has leaked. Deterministic; no judge; no LLM grading a safety
   property.

**Conditions.**

| Condition | Input | Numbers from |
| --- | --- | --- |
| `control` | the raw CSV, post-cutoff rows included | the LLM |
| `aion-agent` | `aion ingest` + `forecast store:… --as-of <cutoff>` | Aion |
| `aion-pure` | same, no LLM | Aion |

The control is given the whole file on purpose. That is the trap: a
plausible agent reads what it can reach.

**"Provably couldn't leak" as an assertion.** Promote the existing
`episodes._max_known_time` logic into
`benchmarks/leakage/assertions.py::assert_no_leak(artifact_dir, cutoff)`,
which fails unless *every* `snapshot_access` record in the artifact and
its lineage has `max_known_time <= cutoff`. Then — and this is the part
that makes it a proof rather than a claim — a **negative control test**
constructs a deliberately leaky run (the same data ingested without the
`published` column, so `known_time` collapses onto `valid_time`) and
asserts that `assert_no_leak` **fails** on it. An assertion never
demonstrated to fail is not evidence.

**Reported metrics.**

- accuracy vs the hidden future (MASE, plus interval coverage);
- **leakage rate** per condition;
- **leakage differential** = control rate − treatment rate (the headline);
- **stolen accuracy** = the control's mean score minus its mean score on
  the subset of runs where it did not leak — i.e. how much of the
  control's apparent quality came from peeking. This is the number that
  makes the differential meaningful to someone who only reads accuracy
  tables.

**Wiring.** Emits AionBench JSONL (so `aion eval compare` works
unchanged), a `summary.json` matching C1's shape, and a `run_all` entry.

**Tests.** Generator invariants (both properties above hold for every
generated task); `assert_no_leak` passes on an honest run and fails on the
planted leaky one; marker detection catches a fabricated control output
and does not fire on an honest one.

**Effort: L.** This is the flagship and deserves its own phase slot.

#### C3 — TSFM-enabled headline configs

`benchmarks/configs/headline-tsfm.yaml`: the TSFM-enabled condition
primary, the stdlib pool as the declared floor, both run on the identical
task set.

One safety requirement: `run_all.py` must **preflight** the sandbox and
fail loudly if the requested adapter is not installed. The runtime already
discloses an absent TSFM tier as a `note` (the amendment recorded in
`COMPATIBILITY.md`), but a benchmark that silently reports a
classical-only number under a TSFM-labelled config is a published wrong
number, and a note in an artifact nobody reads will not stop it.

**Effort: S.**

#### C4 — AnomLLM `aion-agent` condition

**Recommendation: defer, and say why in the benchmark README.**

An LLM proposing "expected anomaly context" is proposing exactly a
constraint (A1) or an effect event (A3/A5). Run against today's gate, on
AnomLLM's short synthetic series, cliff 3 (F4: fewer than four rolling
origins) rejects most of it before any evidence is weighed — so the
condition would mostly measure the gate's refusal, at full LLM cost, and
tell us nothing about the mechanism.

Cheaper substitute, available now: an **oracle condition on the existing
`aion` detector** (A7's adapter-side pattern) that measures how much
headroom a perfect context signal would buy. If the headroom is small, the
`aion-agent` condition is not worth building at all; if it is large, build
it after A1/A5 land and cliff 3 is settled (Q3).

**Effort: M** if built; **S** for the oracle-headroom measurement that
decides it.

---

### D. Housekeeping

#### D1 — pytest 9

Does not reproduce (F6). Recommended actions, in order of value:

1. **Do not pin `pytest<9`.** CI runs 3.11/3.12/3.13 unpinned and is
   green; a pin would freeze the suite against a phantom.
2. Guard the root shim anyway — it costs three lines and removes a real
   sharp edge (importing the repo root as a package when the parent
   directory is on `sys.path`):

   ```python
   try:
       from .integrations.hermes import (...)
   except ImportError as exc:                     # pragma: no cover
       raise ImportError(
           "The Aion repository root is a Hermes plugin shim, not an "
           "importable package. Install Aion (`pip install .`) and import "
           "`aion` instead."
       ) from exc
   ```
3. Add `tests/test_collection.py`: run `pytest --collect-only -q` in a
   subprocess from the repo root and assert a zero exit code and no
   `errors during collection`. That pins the invocation the project
   supports, and would catch the reported failure if it ever appears.
4. Ask the reporter for `pytest --version` and the exact command line. The
   only reproduction I found —
   `pytest --rootdir=<parent>` — discards
   `[tool.pytest.ini_options]` and therefore `pythonpath = ["src"]`,
   giving `ModuleNotFoundError: No module named 'aion'` on all 27 test
   modules. That is a misinvocation, and the fix for it is documentation,
   not a pin.

**Effort: S.**

#### D2 — Stale docstrings and notes

- `src/aion/context.py:17-18` — "No v0.1 pipeline consumes events yet; the
  compiler and ablation stages land behind their own release gate." Both
  halves are false: `pipeline.context_stage` consumes events and
  `context_eval` is the ablation stage. Replace with a statement of what
  the contract actually is and where admission happens.
- `DESIGN_REVIEW_NOTES.md:15-16` — decision 4 ("Context events are
  deferred until an event-history/analogue protocol…") is now history, not
  a decision. It should be marked *resolved* with a pointer to
  `context_eval`, and A4 in this plan is literally the analogue protocol it
  anticipated. Worth noting: that note predicted this proposal.
- `src/aion/covariates.py` — the availability error message says "this
  release only admits future_known values", which is still true, but A2
  adds a second producer of `future_known` rows and the docstring should
  say so.

**Effort: S.**

#### D3 — README section + rename impact inventory

**README section** — "Relation to AION (Zhan et al., arXiv:2605.25045) and
TimeClaw (arXiv:2606.05404)", placed after "Where Aion sits in the
research landscape", which already establishes the taxonomy vocabulary.
Intended shape: theirs is agent-side scaffolding with LLM review in the
loop; this project is the deterministic execution actor underneath, where
the verifier is code the LLM cannot override and leakage safety is
structural (`Snapshot`) rather than behavioural. That is the same
distinction the existing survey paragraph draws, so the section should
extend it rather than restate it.

**One caveat, stated plainly:** both arXiv identifiers postdate my
training data, and this environment's network access has not been used to
fetch them. I can draft the section's structure and the Aion-side claims
(which I *can* verify from this repository), but the characterisation of
what those two papers do must be written or checked by someone who has
read them. I will not paraphrase papers I have not read into the README.

**Rename impact inventory** (no rename performed; this is the inventory
for humans to decide on). Measured on this checkout:

*Volume:* 1,873 case-insensitive occurrences of "aion" across 139 files.

| Area | Files | Occurrences |
| --- | --- | --- |
| `src/` | 33 | 276 |
| `tests/` | 33 | 300 |
| `docs/` | 17 | 296 |
| `benchmarks/` | 31 | 337 |
| `integrations/` | 7 | 293 |
| `skills/` | 1 | 24 |
| `.github/` | 4 | 11 |
| Root docs (`README`, `CHANGELOG`, `COMPATIBILITY`, two design docs) | 5 | 259 |
| Root config (`pyproject`, `plugin.yaml`, `install.sh`, `Dockerfile`, `aion.yaml.example`, `__init__.py`, dot-files) | 7 | 74 |

*Free text is the easy 80%.* The load-bearing identifiers are these, and
each one carries a distinct migration cost:

| Surface | Value | Cost of changing |
| --- | --- | --- |
| PyPI distribution | `aion-forecast` | New project name; the old one must be retained or yanked deliberately |
| Import package | `aion` (`src/aion`, `packages = ["src/aion"]`) | Every downstream import, every benchmark adapter, `PYTHONPATH=src` docs |
| Console script | `aion` | Every doc example, `install.sh`, `Dockerfile` smoke test, CI package job |
| MCP tool names | 20 tools, `aion_*` prefix | **Frozen by `COMPATIBILITY.md`.** Renaming breaks every configured agent; would require dual-registration |
| Environment variables | 31 `AION_*` names (incl. `AION_EXPERIMENTAL_PLANNER`, `AION_REGISTRY_PATH`, `AION_TEMPORAL_STORE_PATH`, `AION_TSFM_SANDBOX_ROOT`) | Silent behaviour change for anyone who set them |
| Error codes | `AION_NOT_INSTALLED`, `AION_EXECUTION_FAILED`, `AION_LLM_FAILED`, `AION_PROTOCOL_ERROR` | Part of the error envelope hosts match on |
| On-disk state | `~/.local/share/aion/registry.db`, `~/.local/share/aion/temporal.db`, `~/.config/aion/aion.yaml`, `.aion-sandbox-ready` | **Existing users' tracked forecasts and ingested vintages become invisible** unless a migration or dual-read path ships |
| ID salt | `ids.AION_VERSION` | Independent of the name, but any rename release will bump it, so **every artifact ID changes** — precedent: the 0.4.0 salt bump |
| Hermes plugin | `plugin.yaml`, root `__init__.py`, `integrations/hermes` (293 occurrences) | Plugin id, skill name `aion:forecasting`, handler names |
| Container | image name, GHCR path, `docs/containers.md` | Published tags |

*Honest summary for the deciders:* the source rename is mechanical and
cheap; the **frozen MCP tool prefix and the on-disk state paths are the
two things that make a rename a compatibility event rather than a
find-and-replace.** Both have known mitigations (dual tool registration,
dual-path reads with a one-time migration), and both cost more than the
rename itself.

**Effort: S** for the inventory and README section (excluding the paper
characterisations); **L** for an actual rename, which this plan does not
propose.

---

## 3. Phase plan

Ordered by leverage ÷ effort, with the dependency constraints the code
imposes. Four phases. Each is a set of small orthogonal PRs; each ends at
a green suite with the stated criteria met.

The review's suggested shape holds, with three changes, each justified by
a finding above:

1. **A1 (constraint events) moves behind B1 within Phase 1**, because
   constraint projection must be applied to the *conformal* rows and to
   the threshold analysis; building it against the sqrt-widened path means
   building it twice.
2. **A2 (magnitude bridge) moves from "unplaced" into Phase 2**, next to
   the other candidate-widening work. It is independent, cheap, and
   TemporalBench T4 needs it alongside A6.
3. **C3 (TSFM headline configs) moves into Phase 3**, with the other
   benchmark work, rather than being scattered.

### Phase 1 — Fix the intervals, fix the housekeeping, land constraints

**Contents:** D1, D2 · B1 (per-lead conformal + the F2 unification) · B5
(Wilson coverage guard) · C1 (abstention-robust summaries + fallback
condition) · A1 (constraint events).

**Why first:** B1 is a bug fix (F1), not a feature — every published
interval in the project is currently mis-scaled, which means every
coverage number, every threshold probability, and every CiK RCRPS score
rests on it. C1 is pure measurement infrastructure and unblocks honest
reporting for everything after. A1 is the highest-value new capability and
is the one proposal that needs no gate changes at all.

**Acceptance criteria:**

- `pytest -q` green on 3.11/3.12/3.13; new `tests/test_collection.py`
  passes; no `pytest` pin added.
- New `tests/test_conformal.py`: nominal coverage within tolerance at
  every lead on synthetic heteroscedastic data; PAVA monotonicity; the
  finite-sample rank index matches a hand-computed reference.
- All four calibration paths (F2) produce identical intervals for
  identical residual inputs — asserted by a parametrised test over
  `evaluation`, `context_eval`, `covariates`, `adjudication`.
- `tests/test_context_constraints.py` as specified in A1, including the
  threshold-analysis consistency test and the fold-local rejection test.
- **Goldens:** `ids.AION_VERSION` bumped; goldens refreshed;
  `two_series_h5` is the only case whose forecast *values* change (the
  three degraded cases must be byte-identical apart from the ID —
  assert this in the review of the diff); `COMPATIBILITY.md` amendment
  written in the same commit.
- **Contract round-trip:** an artifact with `constraints_applied` written
  by this build reads back through `artifacts.read_artifact` and
  `versioning.ensure_readable`; a pre-Phase-1 artifact still reads.
- **Benchmark deltas to report** (each in isolation, no bundling):
  CiK `aion-pure` mean RCRPS before vs after B1; TemporalBench T2
  `aion-pure` OW metrics before vs after B1; interval coverage on the
  leakage-trap episodes before vs after. Report the C1 matched-subset and
  penalized means alongside the scored-only mean for each.

### Phase 2 — Widen the gate: shrinkage, shapes, magnitudes, distributional selection

**Contents:** A5 (shrinkage admission) · A3 (effect shapes, incl. the F5
adjudication fix) · A2 (magnitude → covariate bridge) · B2 (pinball
selection) · B3 (nine quantile levels + CiK sampler).

**Ordering inside the phase:** A5 before A3 (the `λ_looks` term must exist
before multiple shapes are scored on the same folds); B1 before B2 (nested
conformal calibration is what makes a per-fold quantile score possible);
B1 before B3.

**Acceptance criteria:**

- λ = 1 and λ = 0 fixtures reproduce today's admitted/rejected artifacts
  byte-for-byte (the generalisation proof).
- Blended residuals are computed from the blended path — asserted by
  checking that published coverage matches a recomputation from published
  rows.
- Every shape recovers its planted effect; tie-break determinism under
  candidate-order permutation; the combined adjudication candidate uses
  the selected shape (asserted via evidence, F5).
- A magnitude event and an equivalent hand-written covariates CSV produce
  identical artifacts modulo the disclosed `derived_from_events` key.
- Pinball selection is active **only** when quantiles are explicitly
  requested; default invocations are byte-identical to Phase 1 output.
- Nine-level rows are monotone at every step; `forecast.csv` keeps its six
  frozen columns first; `tracking._load_forecast_csv` still scores.
- **Contract round-trip:** `context.shrinkage`, `context.effect_shape`,
  and the nine-level rows survive artifact write → read → tracking
  register → `aion track score`.
- **Benchmark deltas to report:** context admission rate on CiK
  `aion-agent` and TemporalBench T4, before vs after A5+A3 (this is the
  proposal's headline claim — "remove the cliff" — and it is either
  visible here or it did not happen); CiK RCRPS before vs after B3 alone,
  reported separately from the gate change; matched-subset and penalized
  means for every comparison.

### Phase 3 — Prove the safety claim, instrument the gate, ship the conditional path

**Contents:** C2 (leakage-trap eval family) · A7 (gate instrumentation +
adapter oracle conditions) · A6 (conditional path) · C3 (TSFM headline
configs) · C4 decision (oracle-headroom measurement, then build or
decline).

**Why here:** C2 is the flagship and it is worth more once Phase 2 has
made the gate non-trivial — a leakage differential measured against a gate
that refuses everything is a weaker claim. A7 is what makes Phase 2's
changes legible in benchmark output.

**Acceptance criteria:**

- Every generated leakage task satisfies both generator invariants
  (measurable leak gain ≥ τ, mechanically detectable markers), asserted at
  generation.
- `assert_no_leak` passes on every Aion-produced artifact in the suite
  **and fails on the planted leaky negative control** — the negative
  control is the acceptance criterion, not a nice-to-have.
- No force-admit switch exists anywhere in `src/aion`; the oracle is
  adapter-side (assert by grepping the runtime for gate bypasses in a
  test, in the spirit of `tests/test_leakage_lint.py`).
- `conditional` defaults to off; with it off, artifacts are byte-identical
  to Phase 2; with it on, the block round-trips and passes
  `verifier.verify_or_raise` and appears in `lineage.json`.
- A λ = 0 (evidence-rejected) event produces **no** conditional block.
- `run_all` preflight fails loudly when a TSFM-labelled config runs
  without the sandbox installed.
- **Benchmark deltas to report:** leakage differential (control vs
  treatment) and stolen accuracy, on the full generated task set; gate
  precision/recall and headroom on TemporalBench T4; TSFM-primary vs
  stdlib-floor headline numbers on at least CiK and TemporalBench; the
  AnomLLM oracle-headroom number, with the build/decline recommendation
  attached to it.

### Phase 4 — Pooling and adaptation

**Contents:** A4 (analog pooling: dataset-scoped first, store-scoped
opt-in second) · B4 (adaptive conformal via tracking).

**Why last:** both introduce state outside the input file, which is the
only class of change in this plan that can collide with content-addressed
determinism (I8). They should land when everything upstream of them is
stable and measurable.

**Acceptance criteria:**

- Tracking migrates v3 → v4 → v5 in place; every v0.2 tracking command
  passes unchanged (the existing tracking test suite is the gate).
- Analog leakage rules asserted directly: an analog whose
  `outcome_known_at` postdates the fold cutoff is excluded and the
  exclusion is in evidence; `--as-of` replay with a pool row written after
  the replay instant is byte-identical to an empty-pool run.
- Adaptive conformal is off by default; forecasting never writes
  calibration state (asserted by row count); the state version appears in
  the ID payload and in evidence.
- Cross-series pooling recovers a planted effect on a two-series panel
  with the documented shrinkage weight.
- **Contract round-trip:** `context.analogs` and the calibration state
  version survive write → read → replay.
- **Benchmark deltas to report:** admission rate and accuracy on
  never-seen-event tasks with pooling on vs off; realised interval
  coverage over a simulated scoring sequence with ACI on vs off.

---

## 4. Where proposals fight the codebase

Collected, with the resolution taken.

| Proposal | Conflict | Resolution |
| --- | --- | --- |
| A1 constraints | Frozen `forecast.csv` columns and the frozen row shape | Clamp in place; disclose in the `context` dict and new evidence. No new columns needed. |
| A1 constraints | `threshold_analysis_stage` bypasses the rows entirely (F1/pipeline) | Project the residual draws with the same function; extend the `basis` string. Tested explicitly. |
| A1 constraints | Could rescue an abstention | Projection runs only when the base evaluation is already supported. |
| A2 magnitudes | Two contracts could drift | One direction only: events → `CovariateRow`. No new gate, no new loader, no reverse mapping. |
| A3 shapes | `selected_model` and `capabilities().models.context` name `event_adjusted` | Keep the name; disclose the shape. Avoids fragmenting tracking history. |
| A3 shapes | `adjudication` hard-codes `event_effect` (F5) | Shared `apply_shape` helper; both call sites converted in the same PR. |
| A4 analogs (store) | Machine-local state vs content-addressed IDs (I8) | Opt-in; pool fingerprint in the ID payload; rows used recorded in evidence. Dataset-scoped pooling (deterministic) ships first and is the default. |
| A5 shrinkage | `context.admitted` is a boolean consumed by adapters and tests | `admitted := λ > 0`; the boolean keeps its meaning, λ is additive. |
| A5 shrinkage | Does not remove cliff 3 (F4) | Stated, not hidden: λ = 0 below two selection folds. Lowering cliff 3 is Q3. |
| A6 conditional | Frozen `results[]` envelope and 4-file artifact layout | New nullable field, new preview keys, new *fifth* file. No existing key or file changes. |
| A6 conditional | Could launder a rejected effect | λ = 0 events are excluded from the conditional path by rule. |
| A7 oracle | An in-runtime force-admit switch would sit next to the gate | Oracle is computed adapter-side; a lint-style test asserts no bypass exists in `src/aion`. |
| B1 conformal | Changes every non-degraded interval; goldens are byte-pinned | One deliberate salt bump + golden refresh + `COMPATIBILITY.md` amendment, in one commit, in Phase 1. |
| B1 conformal | `Evaluation.residuals` has five consumers | Additive `residuals_by_lead`; `interval_bounds` left untouched for its existing callers. |
| B2 pinball | "Task requests quantiles" is undefined — all runs emit three | Defined as *explicitly passed* `--quantiles`; default invocations keep point-loss selection and their numbers. |
| B2 pinball | Per-fold quantiles risk leaking calibration into selection | Expanding-window calibration (folds `1..i-1`); fold 1 excluded; ≥ 3 folds required, else a disclosed fallback. |
| B3 quantiles | `forecast.csv` columns frozen | Six frozen columns first, new ones appended; amendment recorded. |
| B4 adaptive | Forecast numbers would depend on local history (I8) | Read-only during forecast; append-only, `known_at`-stamped state; state version in the ID payload; default off. |
| C1 summaries | `aion eval compare` is frozen and row-schema-bound | New `benchmarks/compare_conditions.py`; `aion eval compare` untouched. |
| C1 summaries | A penalized mean needs a fallback score that does not exist | New `fallback` condition per adapter; refuse to report the penalized mean without it rather than imputing a constant. |
| C2 leakage traps | "Provably couldn't leak" is a claim, not a test | Negative control that must fail the assertion; without it the assertion is untested. |
| C4 AnomLLM | Would mostly measure cliff 3 refusals at full LLM cost | Deferred; oracle-headroom measurement decides whether to build. |
| D1 pytest | Reported failure does not reproduce (F6) | No pin. Defensive guard + collection regression test + a request for the reproduction. |
| D3 rename | Frozen MCP tool prefix; on-disk state paths | Inventory only. Named as the two items that make a rename a compatibility event. |

---

## 5. Open questions — human decisions needed

**Q1 — Constraint binding thresholds.** Two numbers I can pick defaults
for but should not decide alone:
(a) the monotone-violation tolerance for rejecting a monotonicity claim
(proposed 0.0 — any violation rejects);
(b) the binding fraction above which a clamp downgrades support to
`weakly_supported` (proposed 0.5 of horizon steps).
Both are policy about how much of a forecast may be determined by a claim
rather than by data.

**Q2 — Model naming under shrinkage.** With 0 < λ < 1 the published path
is a blend. Keep `selected_model = "event_adjusted"` (my recommendation —
preserves `tracking.model_performance` history and
`capabilities().models.context`) and disclose λ, or introduce
`event_adjusted_shrunk` and accept a split leaderboard?

**Q3 — Cliff 3: the four-fold requirement.** `context_eval` refuses
outright below four rolling origins (F4), and this is what keeps context
out of most short benchmark tasks. Options:
(a) leave it (safest, caps the achievable admission rate);
(b) lower to three folds with λ hard-capped (proposed cap 0.5) and no
leave-one-best-out term;
(c) lower to two with λ capped lower still.
This single decision probably determines whether the Phase 2 admission-rate
delta is visible on CiK and TemporalBench at all. My recommendation is
(b), decided *before* Phase 2 starts so the benchmark deltas mean
something.

**Q4 — Conditional path default.** Ship `--conditional` off (my
recommendation, avoids artifact churn) and revisit after TemporalBench T4
numbers exist, or on by default from the start because event-conditioned
tasks are a first-class use case?

**Q5 — Pinball as the default selection loss.** My proposal ties it to an
explicit `--quantiles` request, which keeps every existing invocation
numerically stable. The alternative — pinball everywhere — is arguably
more correct for a system whose headline output is an interval, and would
be a larger, one-time numeric break. Which?

**Q6 — Analog pool scope.** Cross-run pooling (A4b) reads state written by
other projects on the same machine. Should the pool be scoped to a
`project` by default, to a dataset, or global with the fingerprint
recorded? This is a privacy/blast-radius question as much as a statistical
one.

**Q7 — The two papers (D3).** Both arXiv IDs postdate my training data and
I have not fetched them. Do you want me to (a) draft the README section
with the Aion-side claims written and the counterpart characterisations
left as marked TODOs for a human, or (b) fetch and read both papers first?
I will not write a comparison to papers I have not read.

**Q8 — Version-salt budget.** This plan bumps `AION_VERSION` at least
twice (Phase 1 for conformal, Phase 2 for pinball/quantiles if the default
changes), each invalidating every artifact ID. Is one bump per phase
acceptable, or should numeric changes be batched into a single release
boundary?

---

## 6. Effort summary

| Item | Effort | Phase |
| --- | --- | --- |
| D1 pytest guard + collection test | S | 1 |
| D2 stale docstrings | S | 1 |
| B1 per-lead conformal + F2 unification | M | 1 |
| B5 Wilson coverage guard | S | 1 |
| C1 abstention-robust summaries + fallback condition | M | 1 |
| A1 constraint events | M | 1 |
| A5 shrinkage admission | M | 2 |
| A3 effect shapes (+ F5 fix) | M | 2 |
| A2 magnitude → covariate bridge | M | 2 |
| B2 pinball selection | L | 2 |
| B3 nine quantile levels + CiK sampler | M | 2 |
| C2 leakage-trap eval family | L | 3 |
| A7 gate instrumentation + oracle conditions | M | 3 |
| A6 conditional path | M | 3 |
| C3 TSFM headline configs | S | 3 |
| C4 AnomLLM oracle-headroom decision | S | 3 |
| A4 analog pooling | L | 4 |
| B4 adaptive conformal | M | 4 |
| D3 README section + rename inventory | S | any |

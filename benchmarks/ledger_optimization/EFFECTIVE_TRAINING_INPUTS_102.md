# Effective training inputs: undeployed prototype 102

The candidate-100 audit found that overlapping CV/production origins sometimes
have different scores for the same configuration. Same date and configuration
do not guarantee the same effective training data. This prototype describes
that difference using existing execution requests. It does not change the
running experiment, fit models, retrieve new outcomes or recommend a model.

## Why this is the next infrastructure check

In the 30 fully matched audited candidate-100 cases, the ledger's selected
configuration appeared in a displayed historical comparison in 26 cases. In
16 cases, that comparison included origins outside the current CV window.
Ten cases had any CV/history winner disagreement involving the selected
configuration, and six had such disagreement in a cohort containing dates
outside current CV. Three selected configurations were consistent with a
conflicting historical winner. That association does not establish which
evidence influenced the agent. The agent selected its final CV minimum in
27/30 cases.

These are latest available displayed comparisons per pair, with exact
configuration identities, not all possible ledger comparisons. Final-CV tables
can postdate earlier selections. No success filtering is used. The earlier
097 diagnostic 101 also found that history was often available and usually
agreed with current CV; the new counts concern candidate 100 and do not replace
or contradict that earlier experiment. Merely requiring another review is not
supported by these findings.

## Prototype contract

`training_inputs_102.describe_inputs` accepts an authenticated successful
execution request, a canonical lab configuration, and the actual numerical
implementation SHA-256. It refuses an unknown implementation. Its semantics
are specific to the frozen lab's `numerical.py`, not a general inference rule
for arbitrary providers.

It reports available history, effective history, requested maximum window,
whether available history truncates that window, history bounds and supervised
training row count. For seasonal naive the effective input is its final season;
for Ridge and RF it is the available suffix capped by the configured window,
with the aligned past/future covariates used by the pinned predictor. It hashes
effective training inputs separately from full effective numerical inputs and
task identity. Numeric types and equivalent timestamp encodings are normalized.
Series, unit, forecast origin, horizon and target timestamps remain in task
identity; the helper does not declare unrelated tasks comparable.

`compare_inputs` distinguishes different tasks, different effective model inputs,
and matching effective model inputs. It preserves source/recording/snapshot
metadata separately. **Matching numerical inputs do not mean matching Gnomon
request fingerprints, recording provenance, actual revisions, or independent
evidence.** They also do not guarantee equal outputs under different dependency
versions or arbitrary providers. The caller must retain execution/provider/runtime
identity and audit temporal visibility; this helper replaces none of those checks.

## Validation on existing evidence

Five regression tests pass. They cover a discarded prefix outside a model's
window versus truncation inside it; changed series, units, future features and
used history; numeric/timestamp normalization; separate temporal provenance;
seasonal input selection; and rejection of unknown implementation or malformed
dimensions. These tests do not fit a model.

A read-only probe authenticated snapshot inventories, comparison artifacts,
review artifacts, numerical source and SQLite files. Across 144 unique stored
current-CV/historical-production execution pairs at matching origins:

| Input comparison | Pairs | Observed saved prediction behavior |
|---|---:|---|
| Matching effective numerical inputs | 114 | Maximum absolute point difference: 0 |
| Different effective numerical inputs | 30 | Maximum absolute point difference: 1.491746 |

All 26 databases remained byte-identical after the probe. The probe performed
zero provider or Engy calls and did not recompute forecasts. Pair counts reuse
dates and series and are not independent statistical samples. Saved predictions
validate the description on this observed set; they do not establish future
accuracy or general numerical equivalence.

## Integration requirements before any agent experiment

If this description is integrated into a future experiment, show current-CV
effective training lengths to **all three arms**. Show historical execution
training provenance with the ledger's paired history. Keep original current
and historical scores and execution references; never combine scores, discard
origins, reweight evidence or choose a model solely from these fingerprints.
Explain outside-CV origin coverage separately from effective-input equivalence.

Preserve model space, source data, origin/cutoff semantics and fit budgets. Any
extra metadata retrieval must be bounded, accounted for, and expose no outcomes
beyond the existing visibility boundary. Test the complete annotation/paging
path and freeze code, runtime and common-arm presentation before paid dispatch.
The existing candidate-100 worker and final holdout remain unchanged. This
prototype is **not integrated or selected for final confirmation** and establishes
no accuracy improvement. The 20% target remains unmet.

Raw scripts, results, source hashes, stdout/stderr and exit records are retained
under `results/training-inputs-102-offline-001/` and
`results/contrast-100-selection-coverage-001/`. Compact committed receipt:
`evidence/training-inputs-102-offline-001.json`. The original runs' costs remain
unchanged; all new work here was offline.

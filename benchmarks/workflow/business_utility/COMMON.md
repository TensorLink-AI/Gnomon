# Business utility: pre-registration v1

Status: prospective protocol; no agent outcomes inspected. The first commit containing
this file and EVAL1–3 is the registration. Implementation changes before execution
are allowed, but changes to hypotheses, cases or grading require a dated amendment
committed before execution. Freeze generated corpus, runtime, driver and operator
configuration in a second commit before any confirmatory model request.

## Claim boundary and release gate

The inspected checkout declares **1.1.9**, not the requested 1.2.0, and contains
pre-existing uncommitted runtime changes. No local v1.2.0 tag was found. Do not label
results on this checkout a 1.2.0 replication. Obtain and pin the intended release
source before live trials. Record source hashes as well as version and Git SHA.

A tool guarantees its returned data/computation under its contract. It cannot
prevent an agent using Python, choosing another cutoff/model, or making an
unsupported final claim. Tool-level property checks and autonomous-agent business
outcomes are separate evidence. Zero observed failures does not prove impossibility.
The proposed marketing sentence is a hypothesis, not the prescribed conclusion.

## Matched execution

Use ../run_workflow.py, ../driver.py and ../matched.py, with the existing bounded
agent loop and isolated software/service backends. No alternate execution runner,
forced tool calls, repaired answers, deleted adapters or hosted forecasting API.
Ordinary: existing Python/NumPy/pandas/SciPy/StatsForecast. Lean: same Python plus
current default Gnomon MCP. Full: same Python plus Gnomon with the explicitly
declared benchmark-only threshold reference contract for Eval 3; otherwise lean.
Disable optional ledger/temporal discovery for both service arms. Primary comparison
is lean vs ordinary for Evals 1–2 and full vs ordinary for Eval 3. Other comparisons
are secondary. A full-arm benefit in Eval 3 is prototype evidence, not release uplift.

One common prompt, model/revision, generation, inputs and budgets across arms.
Automatic tool choice; ordinary may compute exact answers in Python. Never require
ordinary to estimate by eye. LLM requests may be remote; forecasting and reference
computations are offline. Model/account selection remains an operator gate; no
ambient credential fallback. Record temperature=0, max_output_tokens=2048, 12 rounds,
12 tool calls, 32000 cumulative tokens, 180 seconds/task, one worker, zero retries.
Freeze model-specific supported settings and per-arm spending limits before calls.
No substitution of scripted clients for measured LLM results.

Use the runner's isolated containers (1 GiB, 2 CPUs each; service arms have two).
Run only one arm/task worker at once. Log host available memory/CPU before launch
and during execution; stop owned work if available memory <8 GiB or owned RSS >3 GiB.
No automatic restart after a machine incident. Container resource ceilings do not
establish safety of the Windows host or a hard monetary cap.

Three full-arm passes, order determined by a committed seed, identical case order
within each pass. Record wall-clock order; provider drift remains a limitation.
No interim efficacy stopping. Infrastructure stop preserves every attempted task;
unattempted tasks remain visible. No outcome-selective recovery or replacement.

## Shared corpus and contamination

All three evaluations derive from the four CSVs in ../data: pedestrian, retail,
sensor, wiki. Select 10 non-overlapping 24-point windows per source at evenly spaced
indices: floor(j*(length-24)/9), j=0..9. Preserve the source frequency (daily,
monthly, five-minute, daily) in generated timestamps. Apply a fixed positive affine
transform per window using SHA256(source-name + window-index + 'business-v1').
Rebase timestamps to 2025 solely for scenario readability; these are **not new
observations**. Record byte hashes, offsets and transforms in the manifest.

Publication delays/revisions, messy exports and probability distributions are
explicit synthetic overlays on these same windows. These are operational scenarios,
not measured frequencies of customer incidents. No selection based on model results.
Assume every source is in training. Transforms/rebased dates do not decontaminate.
Synthetic vintages reduce direct answer memorization, not general familiarity.
No claim of post-training data; a private prospective customer replication is a
separate future registration. No training-cutoff claim without evidence.

## Denominators and uncertainty

Report planned, attempted, answered, error/timeout/abstention, unknown-audit and bad
outcome counts separately. Missing answers are failed business handoffs, never good
decisions. Report the composite of a bad decision OR failed/unauditable handoff;
do not relabel a timeout as a dispatch that occurred. Numeric-only summaries show
their answered denominator and missing count; do not invent a numeric error for a
missing probability or accuracy. Worst/best-case binary bounds include all tasks.

Paired absolute risk differences and 95% percentile bootstrap intervals, 10000
resamples with seed 20260915, cluster by source window (all repeats and variants
together), stratified by source. Domain results mandatory. Four source domains do
not justify generalization to a population of industries. Binomial upper bounds
for zero failures are descriptive iid approximations, not structural proof.
Use Holm correction across the three primary binary comparisons at family alpha
0.05; numerical dispersion/gaps are descriptive co-reported quantities. No power
claim from the retired runtime. At 80 independent pairs, 10% discordance permits
only fairly large effects; clustering reduces effective information further.
This is a bounded first replication, not a definitive small-effect test.

Publish raw observations, attempt receipts, matched identities, manifest, scorer,
tool evidence and reports. Reports lead with business counts; appendix contains
statistics. A null/adverse result remains published. An invalid causal comparison
is labelled invalid and cannot inherit historical numbers.

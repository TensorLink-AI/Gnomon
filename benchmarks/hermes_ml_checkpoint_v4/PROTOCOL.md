# Checkpoint-first v4: continuous accumulated evidence, Gnomon 1.2.0

Separate prospective protocol; freeze before agent inference. Preserve the previous
96-session experiments (hermes_ml_iteration and checkpoint-v1) and their scores
unchanged. Preserve checkpoint-v2 as an interrupted 10/36 pilot: all ten
completed, but an execution-environment reset stopped it before the gate. Do not
count unfinished sessions as agent failures, pool those ten with v3, or retry
them as if the first calls never occurred. No Gnomon source changes or releases.

Use the exact previous Gnomon build (a38cd0cad35383e5f10021abf3aa20d4c16923be,
source 9723394ccb6d9e11991b312e01bac47c767c69407b6b33d36971cb6e48b6a22e),
Hermes 2237be355906fbe6065ce1815711eee52b2d646e and matched non-Gnomon dependencies.
Three fresh arms: Hermes without Gnomon, with Gnomon execution/no ledger, and with
Gnomon execution/TemporalLedger review. No cross-arm project or memory sharing.

Common changes: review prior evidence; backtest and save a baseline checkpoint;
iterate configurations; explicitly commit a backtested choice. The authoritative
submission is one atomically published execution-bound checkpoint. A model's final
chat formatting does not determine validity. Failed commits preserve the previous
checkpoint. Multiple existing executions of a configuration require an execution ID.
No selection is inferred from prose. No host selects the best configuration.

Keep the previous shared numerical.py byte-identical: lagged log-sales Ridge,
Random Forest and seasonal baseline, same configuration space, known covariates,
nonnegative predictions, three rolling 14-day folds. At most60 numerical attempts,
including failed fits and baseline work. Admit entire backtest batches under one
process lock only if all three fits plus one reserved final fit fit in the budget.
The reserve does not add executions. Validation/retrieval/reusing an existing
selected execution costs no new numerical fit. Errors and results expose remaining
numerical capacity, captured model-request capacity and approximate remaining time.

Same Engy deepseek-v4.1-flash, temperature .2, requested seed7, 16 requests,
3072 output tokens per request, 480 seconds, parent timeout520. No model-call budget
extension, free correction inference or extra host-generated forecasts. The
first12 requests are exploration; requests13–16 are protected selection/correction.
New backtests are rejected in selection, except initial seasonal baseline salvage.
Existing backtests remain retrievable and explicit commits remain available. The
last90 seconds close exploration as well. Phase changes are added as recorded
system notices in forwarded requests; original request payloads are also retained.

At most two continuations may follow premature text termination or a detected
repetition loop if the checkpoint is incomplete and time/requests remain. These
share the original16-request/480-second budget and persistent project/evidence.
Every raw response and per-attempt result is retained. No prose selection is
inferred; corrections are fixed task reminders with no provider/config selection.
Service/runtime failures do not trigger these corrective continuations. Upstream
HTTP failures still consume a request and are reported separately (no free retries).

Hermes is wrapped only to replace its automatic extra summary request at budget
exhaustion with a local status message. Native tools and model responses are
unchanged. The same wrapper and correction policy apply to all arms. The final
chat is never scored. Proxy-rejected requests are logged separately from actual
upstream errors; total forwarded requests may never exceed16.
 No forecast
is created until the agent invokes the common lab. Two series workers; rotate arm
order within each origin and finish all arms before advancing a series. All arms
retain their own ordinary experiment files and matured prior outcomes. Controls
get a concise raw-log review; ledger review reads public ledger APIs and provides
matched-origin comparisons. This is the supported ledger workflow as a bundle.

Pilot: all four previously used Favorita development series, their first three
origins (0,1,2), 12 tasks per arm, 36 total, fresh homes and ledgers. Forecast validity
alone does NOT pass the gate. Full completion requires >=2 distinct complete current
three-fold backtests, >=1 ML configuration, an explicitly selected/backtested final
configuration, and a recorded selection AFTER those comparison results existed.
Baseline-only survival is a valid forecast but an incomplete workflow. A valid
baseline explicitly reselected after comparison can complete the workflow.

Gate: at least90% full completion IN EACH ARM (>=11/12), all36 tasks observed,
zero independent integrity failures, no numerical budget overruns. No accuracy,
arm advantage or token threshold is used for promotion. Preserve every failure.
If the gate fails, stop and report; do not automatically repeat or change seeds.

If and only if it passes, automatically run a separately labeled fresh312-session
evaluation: four series x 26 consecutive origins x three arms. Same frozen sources,
prompts and budgets. Pilot memory/outcomes are not carried into that evaluation.
Pilot and evaluation are not pooled. This reuses a development cohort; it is not
an untouched confirmation or a claim that the completion gate generalizes.

Data source bundle SHA256:
cf7bdd21e216e84809edb8653710e0c0755402864201400c5749a8dbff00f561.
The 32 original anchors are unchanged; 104 tasks fill consecutive origins 0–25.
Only historical inputs, known future covariates and own matured outcomes enter the
agent project. No precomputed candidate cards or current target values. Backtests
are retrospective, recorded at the current origin; source/recording availability
are replay assumptions, not measured publication vintages. No business actions.

Report separately: checkpoint validity, baseline-only survival, full completion,
selection after comparison, numerical attempts/successes, API failures and usage,
checkpoint replacements, ledger review adoption, all-task and valid-only RMSLE.
Use the same last-value fallback only when there is no valid checkpoint. Report
paired differences, per-series and cold/later results without broad significance
claims. All raw logs, immutable checkpoints, stdout/stderr and API receipts stay
available. Validate original evidence hashes and the numerical implementation.
Process/instruction isolation is not an adversarial OS sandbox.


## V4 treatment and service admission

Both Gnomon arms use published 1.2.0, with the same pinned runtime above. Exactly
three arms; no 1.1.9 comparator. No package or PyPI changes. The previous v3
pilot (28/36 complete, service failures in all eight incomplete sessions) remains
frozen and is neither pooled nor silently retried as the same experiment.

At each origin, all arms mature every eligible prior production execution,
including unselected alternatives, with identical raw outcomes. Never turn
backtest executions into ex-ante production evidence. No new provider calls occur
when maturing or reviewing. Exact series/unit/horizon/revision, closed targets,
and source/recording cutoffs are validated before a comparison is included.

Only the ledger arm receives additional matched-pair RMSLE cards derived from
public TemporalLedger reads. Each pair has its own explicitly counted cohort;
never compare ranks across different cohorts. Windows cover the last 4, last 12,
and all eligible prior origins. Pagination is ordered by recency and identity,
never by scores. Current backtests remain common raw evidence for all arms.
Complete card evidence is retained with references, actual IDs and timestamps;
the analyzer independently recomputes its metrics against host-only outcomes.
The frozen dev_evidence_summary helper is development code, imported against
published 1.2.0; it is not represented as a feature shipped in that distribution.

Before each agent starts, a serialized task-free Engy readiness probe runs with
max_tokens=16 and a hard 30-second process deadline. At most 10 probes with
60-second waits. Every probe is retained and charged to experiment cost separately
from agent requests. No task information is sent and the agent clock has not begun.
Exhaustion/non-retryable rejection stops new sessions across workers; already
running sessions finish within their original budgets. Preserve partial evidence
and label infrastructure incompletion; do not fabricate grades or auto-restart.
A passed probe does not guarantee later requests succeed. In-session errors remain
charged and do not reset budgets. Admission applies equally to all three arms.

Report cold origins <4, accumulated origins >=10, and late origins >=22 separately.
All-task scores include incomplete/fallback workflows; paired successful subsets
are secondary and selection-biased. Report provider calls, model tokens, probe
costs, completion, service failures, review use and original evidence references.
Four reused series give exploratory development evidence only. The 20% target
remains unproven; reserved final targets stay closed. A final claim requires a
frozen held-out evaluation and uncertainty over independent series.

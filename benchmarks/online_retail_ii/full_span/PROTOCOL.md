# Online Retail II full-span chronological replay v3

Frozen September 16, 2026, while the original development run is still active.
This is a separate exploratory full-span experiment, not a retroactive extension
of the original held-out acceptance claim. It includes periods already used in
development and, after the predecessor finishes, consumes the remaining source
period. Those dates must no longer be described as unopened after preparation.

The source ZIP is the same published Online Retail II archive, SHA-256
572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb.
It covers December 2009–December 2011, approximately TWO years, not four.
Do not duplicate, time-shift, synthesize, or join unrelated products to pretend
that it contains 100 chronological periods.

## Cohort, dates and chronology

Use the same gross-positive UK product-sales, overlapping-sheet ownership,
no-customer-identifiers, zero-sales and assumed end-of-day availability rules
as v1. Use complete calendar weeks from 2009-12-07 through 2011-12-04.

Warm-up is 14 days: 2009-12-07 through 2009-12-20, as explicitly requested.
Select 48 products using ONLY this window, never an earlier run’s cohort.
Eligibility: at least one positive recorded unit during those 14 days. Strata by active-day fraction: <=0.10, (0.10,0.30], >0.30.
Select 16 per stratum by SHA-256 of `online-retail-ii-full-span-v3:SKU`; fail
before paid dispatch if any stratum cannot supply 16. Keep later discontinued
products; never replace them based on subsequent sales or forecast accuracy.

Forecast 14 daily values at each of 51 origins, every 14 days from 2009-12-20
through 2011-11-20. Last complete horizon ends 2011-12-04. Before the first origin,
all three arms and both seeds have EMPTY agent memories and zero ledger outcomes.
Do not import memory, decisions, actuals or scores from the current experiment.
Within each chain preserve memory thereafter. Fit only history <= origin;
release outcomes only after their full 14-day horizon has matured. The ledger
retains all earlier evidence, with both lifetime and recent-four summaries.

## Matched treatment

Same ten numerical candidates and package versions, RMSLE objective, disclosed
failure fallbacks and typed-execution resolution as v1. Minimum observed days:
last_value and Croston 1; weekly naive 7; AutoETS/AutoARIMA/AutoTheta 14;
28-day mean and four-week weekday mean 28; Ridge and histogram boosting 84.
Unavailable agent calls reject before consuming a numerical attempt. Fixed-model
controls retain every case using labelled weekly-naive fallbacks when unavailable.

There is no completed CV fold at the first origin. Subsequently use up to three
completed 14-day folds, with at least 14 training observations and each model’s
minimum history. CV ranks only models covering every currently available fold
without fallback. With no eligible CV, automatic controls explicitly use weekly
naive. Do not invent scores or backfill early tasks using later evidence.
Ledger rankings exclude unavailable and failed/fallback model executions, and
use the common matched origins across providers with valid historical executions.
Disclose absent providers and the number of matched versus matured origins.
All arms retain access to the same raw outcomes and availability/fallback flags. Strong automatic controls remain all fixed models, rolling-CV
selection, and the CV-top-three log-space ensemble. Historical recent-four
selection remains a deterministic diagnostic, not an agent treatment.

Arms: Hermes/direct, Hermes+Gnomon 1.2.0, Hermes+Gnomon 1.2.0+ledger. Engy model:
deepseek-v4.1-flash. Requested seeds 7 and 19. Every arm gets identical raw
history, current CV and matured candidate prediction/actual pairs. Ledger adds
verified summaries/retrieval, not extra data. Native text memory and skills are
available to every arm, isolated per product/arm/seed. No arbitrary code/shell.

Budgets unchanged: four numerical attempts, 12 model requests, 3072 output tokens
per request, 480 seconds, at most two correction turns within the same limits.
Six concurrent sessions maximum. Expected total: 48 × 51 × 3 × 2 = 14,688.
Start with the same accuracy-blind 36-session interface pilot (three fixed
products, first two origins, both seeds and all arms), retain it in the full
result, and continue only if all workers exit zero and >=90% resolve. Never
selectively retry failures. Standard models are scored on the same cases.

## Interpretation and release

Report overall mean per-case RMSLE, all planned failures/fallbacks, per-origin
paired arm differences, both seeds separately, and predetermined history bands:
0–3, 4–11, 12–23, and 24–50 matured origins. Report the ledger-versus-no-ledger
gap, not just decreasing absolute error. Include simple/ML/CV controls and
native-memory/ledger use, time, requests, reported tokens and unknown usage.
Cluster uncertainty by product and calendar blocks; seeds are not independent
retail tasks. Do not infer causal promotion/stockout explanations from sales.

Queue behind terminal success of `/root/online-retail-agent-resume-001/paid-001`, with
its supervisor exit 0 and exactly 3,744 retained sessions. No automatic dispatch
if that predecessor is incomplete or failed. Require fresh runtime-matched
six-session real-Hermes synthetic-transport preflight before paid v3 launch.
Freeze code and verify the exact source archive and runtime. No selection based
on predecessor accuracy and no changes to the currently running bundle.

This full-span exploratory replay cannot independently satisfy the original
untouched-final 20% acceptance target. A separate, genuinely held-out dataset
would be required for an independent claim after this experiment consumes the
remaining Online Retail II period.

Source: https://archive.ics.uci.edu/dataset/502/online%2Bretail%2Bii (CC BY 4.0).

Restart amendment: daily ledger comparisons validate local calendar identity
and query each origin separately only when the pinned 1.2.0 elapsed-lag guard
rejects a clock transition. Every original identity/visibility check and all
predictions/actuals remain intact. See `../agent_eval/RESTART.md`.

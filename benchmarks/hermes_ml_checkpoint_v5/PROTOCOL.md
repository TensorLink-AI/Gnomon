# 092: corrected accumulated history in a matched agent development trial

Freeze this directory and pass preflight before any model inference. This is a
fresh three-arm development experiment, not the untouched final evaluation.
The target remains at least 20% lower mean per-case RMSLE on an untouched final
set with a paired 95% uncertainty interval excluding zero. Neither this reused
cohort nor successful completion of its pilot can establish that target.

## Treatment and unchanged controls

Fork checkpoint-v4. All arms use the same pinned Hermes, numerical.py, candidate
space, raw history/covariates, own prior matured outcomes, native memory tools,
model and completion rules. Gnomon execution/storage remains the published 1.2.0
build a38cd0cad35383e5f10021abf3aa20d4c16923be, source SHA-256
9723394ccb6d9e11991b312e01bac47c767c69407b6b33d36971cb6e48b6a22e.
Hermes is 2237be355906fbe6065ce1815711eee52b2d646e. No installed package changes.

Three arms: plain Hermes, Hermes plus Gnomon without ledger, and Hermes plus
Gnomon plus development ledger review. Only ledger review changes from v4:

* 091 excludes individual forecasts recorded at/after the first target before
  testing production-history ambiguity. It never uses retrospective backtests
  as ex-ante production forecasts. All other eligibility rules remain intact.
* 088/090 validate catalogue references and recording visibility via the public
  ledger, accepting canonical-equivalent live/serialized request forms.
* 087 brief cards expose the same requested RMSLE/count/tie facts with exact
  immutable full-evidence paths and hashes. Equal windows alias only when their
  complete evidence is equal. Pagination never orders by score.

This is a bundled history-and-interface treatment; any gain cannot isolate those
components. history_091.py is a separately identified development implementation,
not a claim about the unmodified published release. Frozen helpers are protected
in every agent project, including controls; controls never import Gnomon helpers.
All arms can inspect their own raw experiments and matured outcomes. There is no
cross-arm memory, prior-run memory, offline recommended configuration or oracle.

## Frozen cases, budgets and promotion

Use the same previously inspected four Favorita series and 26 origins as 030:
source SHA-256 cf7bdd21e216e84809edb8653710e0c0755402864201400c5749a8dbff00f561.
730 history points and 14 forecast points per task; three current rolling folds.
Each origin completes all three arms before outcomes advance. Availability times
are synthetic replay assumptions, not measured source publication vintages.
Untouched validation055 and the final M5 reserve must not be opened.

Engy deepseek-v4.1-flash, temperature .2, requested seed7, max16 requests,
3072 output tokens per request, 480 seconds (parent520), max60 numerical attempts.
Failures consume budget. No free corrective calls; at most two bounded premature
termination/repetition continuations share the original budget. First12 requests
permit exploration; final4 or final90 seconds reserve selection. Same native
Hermes tools, proxy, serialized service-admission policy and two series workers.
Admission probes are task-free, separately retained and counted in total cost.

Pilot: first three origins of every series, 12 tasks/arm, 36 sessions. Fresh homes,
projects and ledgers. Gate: at least11/12 full workflows in EACH arm, all36 tasks,
no integrity failures or budget violations. Completion requires two distinct
three-fold backtests including an ML configuration, and explicit selection of a
backtested final execution after comparison. A baseline checkpoint alone does
not meet full completion. Do not use accuracy or tokens to promote the pilot.

If the gate passes, run fresh312 sessions (4 series x26 origins x3 arms), without
pilot memory. Do not pool pilot results or old030 controls into its causal score.
If gate/infrastructure fails, preserve it and stop; no automatic experiment retry.
A typed committed forecast is authoritative, regardless of final chat formatting.
Absent a valid checkpoint, use the SAME disclosed last-value fallback in all arms.
This development fallback is inherited from v4, not the older seasonal protocol.

## Evidence and interpretation

Primary: arithmetic mean case RMSLE. Report all sessions including failures;
completion, valid forecasts, fallback reasons, numerical calls, API requests,
tokens and missing billing separately. Also report per-series and cold(<4),
accumulated(>=10), late(>=22) results. Paired successful subsets are secondary and
selection-biased. Four reused series and one seed allow exploratory uncertainty
only; final claims require independent held-out series and frozen resampling.

Audit source hashes, immutable checkpoints, raw provider predictions, outcome
visibility and every referenced review score. Verify review hashes and ex-ante
recording, request identity, actual values and pair cohort equality. Preserve raw
API requests/responses and all failed attempts; exclude secrets from archives.
Original030/087/088/089/090/091 evidence remains unchanged. Main/PyPI unchanged.

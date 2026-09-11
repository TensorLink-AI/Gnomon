# M5 development screen 015: stop at the development gate

The incumbent support rule did not improve M5 development forecasting. Its
mean RMSLE is 1.42% worse than current CV. Even future-aware selection among
these unchanged forecasts offers only 11.66% improvement against CV, below the
20% gate. Do not open the reserved stores or fund a live selection trial on the
strength of this result. The original objective remains unachieved.

The [protocol and runner](M5_CANDIDATES_015.md) were committed as `53ea3d9` before
fitting. All eight development series and all 26 origins completed. The pinned
eight StatsForecast 2.0.3 recipes, numerical package versions, request semantics
and input hashes remained unchanged. No Engy calls were made.

| Policy / diagnostic | All 208 cases | Training 144 cases | Later development 64 cases |
| --- | ---: | ---: | ---: |
| Current CV leader | 0.6169646542 | 0.5664334395 | 0.7306598872 |
| Frozen past-only support | 0.6257278285 | 0.5718319933 | 0.7469934579 |
| Lifetime MAE diagnostic | 0.6162391947 | 0.5730532006 | 0.7134076815 |
| Future-aware per-case minimum | 0.5450017262 | 0.5031364927 | 0.6391985016 |

The hindsight ceilings are 11.66%, 11.17% and 12.52%, respectively. They are
not deployable policies or bounds against every possible agent control.
Lifetime MAE is a preregistered deterministic comparison, not a 1.1.9 agent
result. Its later score does not authorize selecting it after the fact or
claiming a live ledger improvement.

All 32 cold-start cases remain in the primary denominator; their three actual
policies are identical. Mature-origin support is 1.60% worse than CV. Support
changes 101 choices: 55 improve, 46 regress, and the regressions outweigh the
improvements in the primary mean. A higher win count is not proof of better
aggregate forecast error.

## Execution and validation

There are 1,664 production forecasts and 3,328 CV forecasts: 4,992 requests in
total. The pinned exact-request cache records 3,200 hits and 1,792 misses. These
are recipe dispatch/cache counts, not claims about each model's internal
optimization iterations. No production or CV recipe fell back. There were 187
cases with eight distinct point vectors, two with seven, and 19 with six;
identical predictions remain included.

Numerical work used two CPU workers with single-thread numerical libraries.
Measured work after setup was 158.71 seconds; complete subprocess elapsed time
was 170.78 seconds. No API billing was incurred. An initial environment-creation
attempt encountered the read-only shared uv cache; using a separate `/tmp`
cache resolved it before any model run. No dataset or model configuration was
changed to recover. Exact installed packages are retained in the freeze record.

Twelve pre-dispatch tests passed. The independent post-run audit checks all
4,992 forecasts/scores, all 1,792 distinct requests, production and CV histories,
outcome maturity, current-card arithmetic, chosen providers, source hashes and
aggregate metrics. A further fault test confirms that altered series identity
or recording cutoff fails that audit; 13 focused tests now pass. Auditing did
not rerun or change the forecasts.

Preserved limitations: two development stores only; assumed period-end source
and recording visibility; unavailable promotion field supplied as an explicitly
labeled zero model feature; public-data/model-training overlap not ruled out.
The reserved 24 series were not an input to this run. No reserved forecasts,
metrics or diagnostics were computed.

## Consequence for the active objective

Favorita's completed live confirmation had a 12.49% hindsight ceiling relative
to its observed agent control. The separate M5 development panel has an 11.66%
ceiling relative to CV and a negative incumbent result. Neither supports a
20% fixed-forecast claim, and neither licenses cherry-picking another final
cohort by its outcome-dependent headroom. They also do not prove that all
possible retail settings lack ledger value.

Before further target pursuit, the user has been asked whether to retain the
fixed-forecast constraint and its negative conclusion, or allow equal additional
forecasting choices in every arm (for example, history-window length). This is
a scope decision, not a silent target reduction or stronger treatment model.
No such expansion or new paid run has been performed while the answer is pending.
Do not mark the 20% goal complete based on this report.

Retained evidence:

- [Per-case and aggregate screen](evidence/m5-screen-015.json).
- [Frozen inputs, code and runtime](evidence/m5-candidates-015-freeze.json).
- [Completion and cache counts](evidence/m5-candidates-015-completion.json).
- [Independent audit](evidence/m5-candidates-015-audit.json) and [auditor](audits/m5_candidates_015.py).
- [Exact execution command and elapsed time](evidence/m5-candidates-015-command.json).
- [Recipe origin/hash](evidence/m5-recipe-source-015.json) and [descriptive diagnostics](evidence/m5-candidates-015-diagnostics.json).

All original typed requests, predictions, CV folds and actuals remain in ignored
local `results/ledger-optimization/m5-candidates-015/`, hash-linked from the
report. Full stdout/stderr remains in `m5-dispatch-015/`; no failed forecast was
removed. Source data and the original private recipe file were not redistributed
in Git. Main, releases and PyPI remain unchanged.

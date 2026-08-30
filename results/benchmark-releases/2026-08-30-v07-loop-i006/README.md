# v0.7 loop I006: seasonal-period authority and evidence

Decision: **promote conflict disclosure, retain the numeric selector, and
continue to Q2.**

The frozen 128-case SeasonalBench matrix completed with zero future leakage.
Automatic detection found period 6 in all four sufficient-history stable
seasonal cases, and every fold-starved non-baseline result retained a caveat.
The baseline nevertheless exposed a general authority defect: an explicit
seasonal period could disagree with strong visible-history autocorrelation
while the result remained fully supported and gave the agent no way to resolve
the conflict.

On exact commit `4217f6f`, every qualifying conflict preserves the caller's
period and all forecast numerics, while adding typed conflict evidence,
conditional support, and recovery guidance. All 29 conflicts honor the
override. Supported harmful departures fall from 8 to 1, and supported harmful
**non-baseline** departures fall to zero. Row-level comparison confirms all 128
selected models, scores, used periods, forecast losses, and relative gains are
unchanged. The exact final isolated suite passed **2,612 tests with 11
skipped**.

This is a safety and user-understanding promotion, not a forecast-skill claim.
Neighboring explicit periods remain numerically harmful, and the frozen stable
seasonal gates against the strongest honest hindsight reference remain failed.
Silently substituting the planted period would violate caller authority and
benchmark independence. Q1 therefore closes with promoted general safeguards
and a retained evidence-backed numeric no-build; Q2 now owns typed temporal
answer usefulness.

Raw resumable checkpoints and the exploratory dirty-tree run remain preserved
locally. The external-evaluation intake remains controlling, and
`docs/astrid-btc-agent-plan.md` remains untracked and excluded.

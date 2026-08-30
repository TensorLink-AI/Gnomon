# v0.7 loop I011: stable context-mechanism selection

Decision: **promote both general supported-primary authority safeguards and
close Q4; continue to Q5.**

The frozen six-task CiK development shard reproduced two distinct recommendation
defects. First, an observation counterfactual that won only two of three
chronological replay blocks displaced a supported primary and caused a 12.2%
task regression. Requiring uniform block wins when displacing a supported
primary removed that harm while preserving the sealed candidate. Second, the
first seed-7042 confirmation exposed a sampled model prior with explicitly no
historical skill evidence displacing a supported primary and causing a 28.9%
task regression. Withholding that prior from recommendation selection—while
retaining it for inspection and outcome scoring—removed the authority jump.

Because seed 7042 exposed the second defect, both of its post-fix replications
are reported as revision evidence. The exact revised candidate was then tested
twice without cache on untouched seed 7043. Both confirmation replications
completed 6/6, selected the same normalized role, method, support, and scenario
ID on all six tasks, had 5 wins, 0 losses, and 1 tie against their own immutable
primaries, and selected the best eligible path 6/6. Mean matched RCRPS gains
were 0.18861 and 0.18869; median gains were 0.14239 and 0.14262; deterministic
90% bootstrap lower bounds were 0.07176 and 0.07184. No task regressed and no
recommendation was automation eligible.

Actual final submissions were inspected. The diagnostic sensor-spike response
recommended the supported primary and retained the sampled prior with typed
reason `sampled_prior_has_no_historical_skill` and
`candidate_preserved: true`. Untouched confirmation still selected useful
observation, calibration, deterministic effect, and governed-prior paths where
their evidence boundary permitted them. The immutable primary remained exact.

Focused tests passed **282 with 4 skipped**. The full TSFM-isolated suite passed
**2,648 with 11 skipped**. Raw case rows, checkpoints, traces, and receipts
remain in the local paths listed in the aggregate artifacts.
`docs/astrid-btc-agent-plan.md` remains untracked and excluded; the v0.7
external-evaluation intake remains committed and unchanged.

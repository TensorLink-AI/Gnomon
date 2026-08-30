# v0.6 loop I004: truthful multivariate boundary

Decision: **candidate no-build; promote the boundary correction.** The frozen
12-case prospective benchmark completed 12/12, but VAR admitted none of six
lagged-driver cases. It beat the mandated baseline margin on those cases, yet
ETS was still better on the same selection folds. Gnomon therefore preserved
the univariate primary. The benchmark, thresholds, and case families were not
changed to turn that restraint into a win.

The production-useful correction is at the interface. VAR now receives related
series through the immutable model-neutral adapter request, and its gate names
the target, supplied signals, retained signals, adapter, and protocol. Public
TSFM capabilities no longer claim that Toto or TTM accept multivariate targets
through Gnomon: their currently invoked methods receive only one history. This
aligns what agents can discover with what the runtime actually executes.

The refactor changed no published forecast in the matched corpus. The maximum
VAR selection-score difference was 1.24e-15, below the frozen 1e-9 parity
limit. All six controls retained their univariate forecasts; no independent
control admitted VAR. Manual inspection confirmed the distinct rejection
reasons for a driver, correlated control, and independent control.

The final local suite passed 2,549 tests with 11 skips. Raw resumable artifacts
remain in `results/v06-p4-i004-*`; this directory retains decision aggregates.
The competition-specific `docs/astrid-btc-agent-plan.md` remains untracked and
excluded.

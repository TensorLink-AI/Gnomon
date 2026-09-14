# Historical configuration neighbors — undeployed experiment

The 65-case 093 audit found that even future-aware selection among already
executed forecasts would improve control error by only 11.27%. Selection among
that fixed set cannot attain the 20% development threshold. This prototype tests
whether existing ledger evidence can help agents find configurations to backtest
within the unchanged common model families and budgets.

`exploration_review_095.exploration_review` adds an unranked index of previously
executed configurations differing in exactly one setting from a configuration
already backtested for the current task. It takes a recording-visible 087/088
review, a task-bound current backtest/budget snapshot, and the common lab config
validator. It never fits, invents configurations, selects a forecast, queries
new outcomes or changes original cards. Neighbors are ordered by configuration
identity, never score. Every supplied pair's loss, tie, disagreement or absence
of matched evidence remains visible and referenced to the original full file.

The next call backtests a known configuration on the current task. Admission
requires exploration phase and three fits plus one final-fit reserve. Live lab
admission must recheck time, phase and budget. A structurally runnable call is
not permission to overrun that budget. Current facts must be captured together
by the trusted caller; this renderer does not prove recording visibility or
authenticate the upstream file merely because it carries a hash.

Differences between configurations are observational, not causal estimates of a
parameter's effect. No global ranking across different paired cohorts is made.
An absent pair on a partial page is unknown, not unsupported or a loss. The
original pagination and full-evidence reference remain unchanged. No benefit is
inferred from a missing comparison.

## Checks and observed limitation

Seven tests cover exact source/argument preservation, supported losses and ties,
recent/lifetime disagreement, partial pages, empty support, already-tested
configurations, multi-setting exclusions, task/config identity, budget reserve,
cyclic references, malformed counts and nonfinite scores.

An offline structural replay used 80 unique current-origin pages from the
already-audited 093 sessions. Every catalog configuration was separately used
as a **hypothetical** current tested anchor, with a synthetic exploration phase
and four remaining fits. These are not the actual agent's available backtests
or budget at the time of each review. The replay does not measure policy accuracy
or claim these actions were admissible in the original run.

Of 437 hypothetical anchors, 251 had at least one one-setting catalog neighbor.
There were 366 directional neighbor entries: 100 pairs appeared on the supplied
page, and 266 were absent from that page. All 300 windows for the 100 present
pairs had zero matched origins. Windows, directions and repeated pages are not
independent samples. This prototype therefore has **no observed supported
parameter comparison on the replayed pages**, despite finding catalog neighbors.
Missing page pairs were not queried, so this is not a claim about all possible
pair evidence in storage.

This result argues against deploying the renderer as an evidence-based parameter
recommendation. It may identify gaps worth prospectively testing, but numerical
evidence would first need paired executions with subsequently matured outcomes.
Any such experiment must charge all additional backtests/forecasts to the same
common budget, preserve unselected forecasts and failures, and permit the same
model/configuration space in controls. No accuracy claim is supported here.

The first replay driver assumed all sessions had review files and failed on a
cold-start session. It is retained in `results/exploration-review-095-offline-001`.
The corrected replay explicitly counts four sessions without review files and
is retained in `results/exploration-review-095-offline-002`. The exact common
configuration function was extracted via AST to avoid importing unavailable
local numerical dependencies; no prediction function or fit ran.

Live 093 sources remain frozen and this module is not imported by them. Finish
the paid run and queued synthetic seed integration before any next paid trial.
Neither this prototype nor sparse-display 094 has a prospective accuracy result.
Final holdout remains closed; main/PyPI are unchanged.

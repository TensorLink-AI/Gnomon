# Recipe uptake measurement

`planning_uptake_107.py` implements the proposed-versus-executed secondary
measurement already specified in the frozen candidate-107 protocol. It does
not change the agent, capsule, data, score, pilot gate or promotion threshold.
It reads saved tool requests and responses; it makes no provider/API calls and
does not infer intent from prose.

Exposure is frozen when a request is submitted. A recipe first appearing in
that request's own response, or in a later parallel response, cannot be credited
as prior exposure. A failed request is distinct from a successful backtest.
Validated response configuration identity handles ordinary numeric
canonicalization. Reused backtests are explicitly identified in traces rather
than counted as fresh fitting. Selection requires a valid, non-fallback grade
matching a previously suggested and subsequently backtested configuration.

Reports retain all observed session grades, including controls, failures and
unreturned requests. They report plan exposure, distinct proposed/requested/
successfully backtested configurations, suggested checkpoint publication and
selection. The recorded ordering establishes an association, not that the
suggestion caused the agent's choice. Original workflow and recipe audits are
still required; this analysis is not an integrity gate.

Ten unit checks pass, including late/parallel exposure, failed requests,
wrong-task/control contamination, duplicate returns, cold compact plans and
canonicalized configurations. On the already verified 15-session synthetic
worker evidence, the measurement finds five ledger sessions exposed, one with
a supported recipe, and that one explicitly creating a checkpoint, testing
and selecting the recipe. Both controls show zero recipe exposure. These are
scripted fixtures, not paid agent efficacy evidence.

The first analysis attempt incorrectly required query in the compact cold-start
view; the public artifact carries it but that empty compact view omits it.
That failed attempt remains in `results/planning-uptake-107-worker-launch-001`.
The corrected reader permits omission only for an empty cold view with no next
action. Successful analysis is in `results/planning-uptake-107-worker-002` and
tests are in `results/planning-uptake-107-tests-002`. No execution was rerun.

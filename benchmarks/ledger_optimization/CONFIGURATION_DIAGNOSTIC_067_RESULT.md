# Diagnostic067: expanded search does not supply enough saved single-model headroom

Frozen73f2db4 before deriving new comparisons from completed066. All416 scored
original-development tasks included. No forecast/API/weight-fit calls; no source,
policy, validation, final or release changes. Four synthetic tests pass; independent
50,949-check audit recomputes all saved production scores and aggregates.

| Prospective choice or hindsight diagnostic | Mean case RMSLE |
|---|---:|
| Original six: current-CV single selection |0.28158276793416015|
| Expanded search: current-only |0.27513740193270925|
| Expanded search: prior-backtest ledger |0.2740828435805278|
| Existing strong block-CV blend |0.2587110657586681|
| Previous lifetime-ledger blend061 |0.25198521475530405|
| Hindsight best original six |0.22050467257990738|
| Hindsight best control's produced forecasts |0.21882226473742697|
| Hindsight best ledger's produced forecasts |0.2190145659810036|
| Hindsight best union of produced forecasts |0.21878935008414044|

The two searches selected the same configuration in362/416 cases (87.0%).
Control selected an extra configuration201times:111 improved on original-six
CV selection,90 worsened,215 unchanged across all416. Ledger selected extras
200times:118 improved,82 worsened,216 unchanged. Mean observed-CV improvement
.0107904/control and.0105486/ledger did not translate into comparable production
gains. Mean selection regret to each arm's hindsight produced-choice oracle was
.0563151/control and.0550683/ledger. This diagnoses a selection gap, not its cause.

The original-six hindsight ceiling is14.768% better than strong block-CV; the
expanded produced union raises it only to15.431%. Thus no selector among these
exact saved production forecasts can achieve20% versus the strong guard on this
cohort—even with future outcomes. This is not a ceiling on the full78 catalogue,
the17 tested configurations, or new combinations: only six starters and up to
one extra selected production forecast per arm were generated. At least ten
backtested extras per arm have no production forecast and were not imputed.

Decision: do not keep tuning a single-model selector over this saved inventory.
The current best strategy blends forecasts and learns from matured production
errors;066instead reused earlier backtests to select one configuration. A next
candidate should preserve that stronger blending action and fairly test whether
additional configuration search improves it. Freeze any new policy and common
budget before execution. No paid confirmation of failed066 and no final-set
access. The20% objective remains active and unproven.

Evidence: results/configuration-diagnostic-067-001 and receipt
benchmarks/ledger_optimization/evidence/configuration-diagnostic-067.json.
Inherited computation counts remain49,616 forecasts (24,032new066 plus12,984
original plus12,600guard-history preparation); separate audit costs remain in
prior receipts. Hindsight scores are not deployable performance or agent results.

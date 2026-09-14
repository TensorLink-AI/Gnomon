# Source084: conditional risk with mature prior backtest context

Apply frozen083 representation to the unchanged416scored/125available warm-up
source tasks. The only learner change relative081 is seven appended predecessor
features. Fixed six066raw model outputs, unchanged045anchors, same32trees and
24simplex optimizations per arm, same candidate identities and numerical budgets.
Retain control,050,061,068 and081comparisons without recomputation or tuning.

Build predecessor metadata from each input's explicit origin and fixed three
CVends658/682/706 of730history rows. CV0origin=current-72h has no predecessor;
CV1uses CV0, CV2uses CV1, and production uses CV2. Each predecessor origin is
24h earlier and its last target/source/recorded instant equals the forecast
origin, under the existing nominal period-end assumptions. Retain task/fold
identity with that metadata and validate configurations before feature use.
Historical production records carry their OWN previous CV2pair, never current
query backtests. Historical production outcome maturity remains checked before
payload access. Record current and historical predecessor metadata in model
evidence; retain query predecessor metadata in every scored row.

All three current CVtraining pairs retained with masses1/3 for control. Ledger
uses those pairs at total0.5 plus all mature same-domain historical production
pairs at total0.5. Absence of CV0predecessor is explicit zeros/presence0, not a
fabricated successful forecast. Query always uses its mature CV2predecessor.
Both arms finish forecasts before current production outcomes are read to score.

Maximum832forests/26,624trees/19,968simplex solves. Preserve every started and
completed attempt, iterations, wall/CPU, partial certificates and failures.
Record inherited49,616raw forecasts,416045anchors,832068blends and832081forests/
19,968081weight fits; inherited search11,902surrogate solves and31,378logical
attempts per arm. No new raw forecasts or API calls. Never drop a failed case,
relax a solver threshold or overwrite an earlier run after scoring.

Frozen gate:20% lower mean per-case RMSLE versus matchedcontrol ANDstrong050;
positive versus061/068/081overall, with all five comparisons positive in each
domain. Report early0-7/later8-25 descriptively; do not claim growing benefit
from this split. No paid, validation or final access after a failed gate. This
remains a repeated-development numerical prototype, not a matched1.2.0agent
result or a product release. Main/PyPI unchanged and the final goal unchanged.

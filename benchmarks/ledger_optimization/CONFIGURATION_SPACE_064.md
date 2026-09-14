# Common model-configuration space064: preparation, not a scored experiment

063 ruled out20% from selecting among the twelve exact saved forecasts. That
does not test the full user-authorized task of backtesting and iterating model
configurations. Return to that task without adding a stronger model family or
exclusive treatment actions. This amendment defines common configuration choices
before any of the new choices is scored on source observations. It authorizes
synthetic execution parity/preparation only. A separate search-policy protocol,
tests and freeze are required before a development forecast run.

## Unchanged task and numerical implementations

Retain the original sixteen development series,26 weekly origins,730 observed
hours,24-hour horizon and three current folds ending658/682/706. Keep source
labels/phase, explicit nominal UTC coordinates, period-end availability assumptions,
units, source hashes and the same independent final reserves. No convenient source
period, task or series replacement. The original 125 complete warm-up cases remain
available; three failed original warm-up cases remain unavailable, not imputed.

Use precisely038's seasonal, weekly-mean, standardized Ridge and Random Forest
implementations. Same recursive log1p prediction, lag/rolling/calendar features,
training-only clipping guard,80 forest trees, minimum leaf3, seed17, one thread.
Only the existing hyperparameters become configurable through a common finite
catalogue. This implements the already-authorized ML task, not an ensemble or
bias-correction advantage available exclusively to treatment.

## Prospective common catalogue:78 configurations

- Seasonal period24 or168, plus the arithmetic same-hour last-three-weeks mean.
- Ridge: window336/504/730; contiguous lags24/48/168; alpha .01/.1/1/10/100/1000/10000.
  The63 combinations cover approximately two, three and four weeks of history,
  daily to weekly lag context, and log-spaced regularization. No values are selected
  by observing these configurations' forecast errors.
- Random Forest: window336/730; lags24/48; maximum depth3/6/12. Twelve combinations
  vary only the same history/context/complexity parameters already exposed.

All six original038 recipes remain exactly present. The numerical adapter must
reproduce their predictions on synthetic658- and730-observation requests, without
mutating inputs. Shape/type validation, canonical integer/float config identity,
unknown/irrelevant-field rejection, explicit timestamp/grid phase and fixed seed
must be tested. Public configuration identity is a SHA-256 of canonical parameters;
the implementation revision is recorded separately and both identify evidence.
Catalogue order is the original six then remaining configuration hashes; this
is a deterministic inventory, not a recommendation or performance ranking.

## Subsequent comparison requirements

Same catalogue, algorithms, raw arrived observations, fit/selection operations and
60 numerical-attempt limit in both proxy arms and every eventual Hermes arm.
Deterministic baseline computations and failed numerical attempts count. A whole
three-fold backtest requires available capacity for those three computations plus
one reserved final computation. No free treatment-only configuration trials.
Shared exact computations may be physically cached, but charge their logical use
to each arm and report both counts. Historical cohort maintenance also counts.

The numerical question is whether recorded prior experiments improve the next
configuration to investigate under that same budget. Keep observed backtest results
distinct from production outcomes; unexecuted configurations do not acquire
production evidence merely because their hypothetical forecast could be computed.
Persist config/revision, task/origin, supplied context, attempt/failure, CV fold
evidence, recording time and later matured production outcomes independently.
Finish all same-origin decisions before maturing that origin's outcomes.

Freeze an identical search/acquisition algorithm for both arms before scoring;
only its access to structured past evidence may differ. The raw historical record
opportunity remains common in eventual agent arms, as in the existing protocol.
Disclose proxy limitations: numerical memory/no-memory search is not a Gnomon
engine or agent treatment estimate. Track records exposed to each proposal so
future/failed/unexecuted evidence cannot enter the search unnoticed.

Retain current-CV selection, the existing strong050 block-CV forecasts and061
lifetime ledger as separate fixed guards. Do not weaken the old comparator to
claim20%. A development candidate must pass the unchanged20%/per-domain gate
against matched no-memory search and the strong guard before paid confirmation;
report incumbent comparisons too. The final goal still requires the1.2.0 Hermes
arms, matched information/budgets, multiple seeds and positive95% uncertainty on
untouched final tasks. This preparation cannot satisfy it.

## Preparation scope and retention

No source count parsing, catalogue performance sweep, paid requests or protected
outcome access in064. Emit the catalogue, exact adapter/base-code hashes, package
versions, test/parity evidence and synthetic computation costs. Freeze before
running that preparation. Source parity preserves old receipts instead of replacing
the original module. Main/PyPI remain unchanged. The prior result remains a
negative finding and the20% objective remains active and unmet.

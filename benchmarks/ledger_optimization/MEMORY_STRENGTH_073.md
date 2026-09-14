# Preparation073: select memory strength from previously realized candidate outcomes

Recent changes to neighbor identity, shape and blend granularity did not improve
068. This amendment tests a different mechanism: learn when historical evidence
helps rather than always assigning it mass.5. Freeze the helper, finite strength
grid and eventual source-run rules before generating any new candidate outcomes.
This preparation uses synthetic records only. It does not claim an accuracy gain.

Common strength grid:0,.25,.5,.75,1. The four six-hour/seven-slot blend and solver
remain068. Three current CV pairs have total mass1-strength; sixteen eligible
past raw-production pairs have total massstrength. Omit zero-mass pairs. With
no mature history all grid requests reduce to the identical current-only fit;
store requested and effective strength separately. Cache exact training inputs,
charge logical attempts separately from physical solves. Maximum five blend
requests per task in either arm; do not force redundant control computation.
Same forecast/search budget, information boundaries and strong control guards.

The numerical source runner must generate and record all five candidate blends
at each decision, before scoring that task's production outcome. Include all125
available warm-up tasks before416scored tasks. Historical candidates are actual
prior executions, not counterfactual fits generated after seeing their outcomes.
Current-only control has no past raw outcomes or past strength-trial outcomes;
its candidates coincide. Ledger uses the existing068coarse16-neighbor rule and
same541task population, without071/072retrieval changes or070hourly weights.

Use only records from the same arm/domain with strictly earlier origin and
mature target closure/source availability/local recording time. Selection uses
that same query's specified neighbor cohort, with>=16records and>=3distinct
origins. Recompute each recorded candidate's RMSLE from its typed point/actual
pairs. Require every candidate for every matched record. Average per-case RMSLE
for each strength; choose minimum. Exact ties prefer.5,.25,.75,0,1 in that order,
preserving the previous.5policy when there is no discrimination. With insufficient
history choose requested.5, marked insufficient_history; its effective fit can
still be current-only if raw history is unavailable. Do not parse narrative or
select on current production actuals. Unavailable records are filtered before
reading their scores, forecasts or actuals. Keep source/recording boundaries,
cohort IDs, all candidate score means/ties and selection cause in the result.

Historical strength-record identity binds task,arm,domain,origin, target closure,
source availability, recording time and all five executed candidate forecasts.
Each logical candidate must be recorded by its forecast origin. Reject duplicate
eligible identities (including equivalent timezone spellings) or inconsistent timestamps. Never mix
candidates from different tasks or silently discard a missing candidate. Current
query selection is based exclusively on predecision available records.

For source execution, load416frozen045current-only anchors unchanged; compute
125warm-up anchors with the identical045global-CV algorithm/tolerance, recording
all extra costs and failures. Reuse066raw forecast/configuration identities.
Process complete same-origin batches before exposing outcomes to later queries.
No protected validation/final access. Freeze a separate source runner before
any source-data candidate blend is computed. Check fixed.5ledger reproduces068
and current-only control reproduces068over416scored tasks; retain061/strong050.

Report real/logical/cached solves, all strength choices, fixed.5and adaptive
results, count of mature trial cohorts, per-domain/phase means, all failures,
and inherited search/model/anchor costs. Gate remains>=20%over matched control
ANDstrong050, positive over061AND068overall, all four positive in both domains.
Neither this synthetic preparation nor a reused-development result proves the
final objective. No paid/final confirmation after a failed development gate.
Final still requires matched1.2.0/Engydeepseek-v4.1-flash agents, untouched cases,
paired95%uncertainty excluding zero. Main/PyPI unchanged.

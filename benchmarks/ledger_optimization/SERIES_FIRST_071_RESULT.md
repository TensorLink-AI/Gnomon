# Development071: same-series priority does not improve the incumbent

Frozen e253e19 before scoring. The only change from068was mature-production
neighbor ordering: exact series first, then the same existing context distance,
with other series filling the16-record quota when current-series history is
sparse. All readiness/availability/domain checks, searches, forecasts, four-block
mixture action, anchors and solver settings remained unchanged. Control forecasts
and complete control records reproduce068exactly across all416cases.

| Approach | Overall416 | Electricity208 | Pedestrian208 |
|---|---:|---:|---:|
| Matched current-only blend |0.2576259533432357|0.10690353854294049|0.4083483681435309|
| Same-series-priority ledger |0.25187025227270116|0.10491987419050045|0.3988206303549019|
| Prior context-neighbor ledger068 |0.2510782023798801|0.10483908831312373|0.3973173164466364|
| Fixed-six strong block-CV |0.2587110657586681|0.10748648550251698|0.40993564601481913|
| Prior lifetime ledger061 |0.25198521475530405|0.10734528573584194|0.3966251437747662|

071gain versus matched control2.23413%, versus strong block-CV2.64419%, versus
061only0.04562%overall. It worsens068by0.31546%overall,0.07706%electricity and
0.37837%pedestrian. Early128cases nearly tie068(+0.00490%); later288cases worsen
0.44408%. Thus exact-series priority did not fix the generalization limitation
in this experiment. The20%and incumbent/domain guards fail; no paid/final trial.

The rule actually changed evidence composition: average14.543of16neighbors
came from the current series;285cases used16same-series records,131borrowed
other series. Every selected record was already mature under the nominal source
and recording clocks. This result does not prove that personalization is always
unhelpful, or that coarse context distances are universally sufficient.

832fits completed,51,085iterations,12.910s wall/12.096s CPU;0new provider/API
calls. Inherited49,616raw forecasts,11,902search surrogate solves,31,378logical
search attempts/arm,416045anchors and832068comparison fits remain disclosed.
Four pre-freeze tests passed: exact-series priority/borrowing, sufficient local
history, late-recording exclusion before feature access, and cold-start behavior.

Independent53,354checks passed: all416neighbor pools/orderings/local-record
counts; exact causal configuration-bound fit pairs; all832independent objective/
gradient certificates; produced forecasts; exact-control parity; and aggregate
scores/guards. Audit6.858s, no provider/API calls. It reuses previously audited066
raw forecasts, not an independent reimplementation of sklearn. Original data,
case outcomes, weights and failed gates are retained with hashes and archive.

Current lowest original-development mean remains068(.2510782), with only2.54%
matched improvement. This is not a new Hermes, held-out or product deployment
result. Main/PyPI and all protected validation/final data unchanged; goal active
and not achieved. Before further tuning, assess whether remaining opportunity
is predictable from predecision evidence rather than merely visible in hindsight.

Evidence root:results/series-first-071-001; tracked archive/hash receipt:
evidence/series-first-071.json. The preserved068base report and071amendment
make the retrieval-only change explicit; no frozen source file was modified.

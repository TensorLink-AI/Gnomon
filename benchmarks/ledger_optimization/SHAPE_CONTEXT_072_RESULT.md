# Development072: observed demand-shape matching does not improve the incumbent

Freeze3bc13ea preceded scoring. Added24hour-of-day and7weekday descriptors from
the last672observed hours of each730-hour request. Means in log1p space were
centered/scaled within the request, then standardized over temporally eligible
history. Distance combines original12feature and31shape groups with fixed
per-dimension normalization. All maturity filters,16-neighbor quota, search
forecasts, four-block learner, weights/solver rules and comparison guards stayed
fixed. Control records and forecasts reproduce068exactly across416cases.

| Approach | Overall416 | Electricity208 | Pedestrian208 |
|---|---:|---:|---:|
| Matched current-only blend |0.2576259533432357|0.10690353854294049|0.4083483681435309|
| Shape-context ledger |0.2523721891779636|0.10519554840533703|0.39954882995059016|
| Prior context-only ledger068 |0.2510782023798801|0.10483908831312373|0.3973173164466364|
| Existing strong block-CV |0.2587110657586681|0.10748648550251698|0.40993564601481913|
| Prior lifetime ledger061 |0.25198521475530405|0.10734528573584194|0.3966251437747662|

072reduces error2.03930%versus matched control and2.45018%versus fixed strong
block-CV, but worsens061by0.15357%and068by0.51537%overall. Both domains worsen
versus068:0.34001%electricity,0.56165%pedestrian. Early128cases worsen068by
0.13424%; later288by0.66840%.20%and incumbent/domain gates fail. The added
observable descriptors did not improve this particular neighbor rule; this does
not show that time-of-day information or context matching is generally useless.

541source-bound profiles prepared lazily for current/eligible past requests;
each contains672observations,28perhour and96perweekday. Query/neighbor hashes,
calendar label phase, centers/scales and both distance terms are retained.
Historical profiles are fetched after domain/origin/source/recording exclusion;
future production actuals never enter descriptors or neighbor ranking. Timezone
labels retain the source's nominal replay assumption, not verified civil-time/DST.

832fits,51,497iterations,14.401s wall/13.526s CPU. No new provider/API calls.
Inherited49,616raw forecasts,11,902search surrogate solves,31,378search attempts
per arm,416global anchor fits and832068comparison fits remain disclosed. Four
pre-freeze tests passed for profile groups/endpoint, log-level invariance,
late-recording exclusion before profile access, and invalid histories/timezones.

Independent57,974checks passed: all541profiles recomputed, exact source hashes,
calendar coverage, both distance groups and mature neighbor order, all832fit
certificates, forecast/configuration binding, exact control parity and all
metrics/guards. Audit7.963s, no provider/API calls. It reuses the previously
audited066raw forecasts; it does not independently retrain their estimators.

No paid or final-set confirmation justified. Current best original-development
mean remains068(.2510782;2.54%matched gain). Main/PyPI and protected validation/
final data unchanged; this is not a new Gnomon/Hermes or held-out accuracy result.
Goal remains unproven. Retain this negative result before considering a new
mechanism for estimating whether accumulated evidence is useful on a given task.

Evidence:results/shape-context-072-001, complete profiles/source accesses,
base report and072amendment; archived with receipt evidence/shape-context-072.json.

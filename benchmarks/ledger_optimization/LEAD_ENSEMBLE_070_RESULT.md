# Hourly blending070: increased matched gain hides worse absolute performance

Frozen1aedb8b changed only common blend granularity from068four six-hour blocks
to24one-hour simplexes. Same source forecasts, current and matured evidence,
anchors, masses and convergence requirements; larger168-weight action in both
arms. All416tasks/832fits completed in the original session70210 without restart,
relaxed solver tolerances, dropped cases, numerical fallback or API calls.

| Approach | Overall416 | Electricity208 | Pedestrian208 |
|---|---:|---:|---:|
| Hourly current-only control |0.26656241115338347|0.10966681779432164|0.42345800451244525|
| Hourly ledger blend |0.25593164388707146|0.10520488239081509|0.40665840538332787|
| Existing fixed-six strong block-CV |0.2587110657586681|0.10748648550251698|0.40993564601481913|
| Prior lifetime ledger061 |0.25198521475530405|0.10734528573584194|0.3966251437747662|
| Prior068six-hour current-only blend |0.2576259533432357|0.10690353854294049|0.4083483681435309|
| Prior068six-hour ledger blend |0.2510782023798801|0.10483908831312373|0.3973173164466364|

Hourly ledger improves3.98810%against its matched hourly control but only1.07433%
against the fixed strong guard. It is1.56614%worse than061and1.93304%worse than
068overall. Compared with068ledger, electricity worsens0.34891%and pedestrian
2.35104%. The hourly control also worsens3.46877%against068control. The higher
relative memory benefit is not an improvement in the best absolute performance.
The20%and incumbent/domain guards fail. Do not promote or run paid confirmation.

Early128cases:4.41549%gain versus hourly control, but1.38795%worse than068ledger.
Later288cases:3.81672%gain versus hourly control, but2.15189%worse than068ledger.
No increasing late-phase memory benefit established. This is reused development,
not a confirmatory interval, production deployment or new Gnomon/Hermes trial.

Cost:832completed weight fits,174,620iterations,1,603.45s wall (26.7minutes),
1,502.05s CPU. All416ledger cases had mature neighborhoods. No new provider/API
calls, but inherited49,616raw forecast computations,11,902search surrogate
solves,31,378logical search attempts/arm,416global anchor fits, and832comparison
068fits remain disclosed. Do not interpret raw-forecast reuse as free training.
Four synthetic tests passed before freeze, including repeated-block objective
and forecast equivalence, gradient checks and a complete hourly fit. An initial
test syntax typo was fixed before any source execution; one synthetic fit ran.

Independent complete audit passed51,689checks: all832training input identities,
causal neighbors/configuration roles, independent objective/gradient certificates,
produced forecasts, scores and incumbent/gate comparisons. Audit6.913s, no new
provider calls. Earlier immutable-prefix audits retained70,202and310completed
cases; they were explicitly partial, never treated as full-run success. These
assertions are repeated structural checks, not independent statistical samples.

Decision: keep068as the current lowest original-development mean; the20%goal
remains unproven. Finer unconstrained temporal weights did not turn hindsight
range headroom into useful prospective gain. Further work must address a
specified generalization mechanism, not treat more blend degrees of freedom
or a weaker matched control as evidence of ledger improvement. No unchanged
candidate should receive a paid/final trial after this failed gate.

Evidence:results/lead-ensemble-070-001, full archive and tracked
receipt evidence/lead-ensemble-070.json. Frozen base068report and070amendment
both preserved, along with raw per-case comparisons and every checkpoint audit.
Main/PyPI and protected validation/final data remain unchanged.

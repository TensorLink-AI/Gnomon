# Development068: blending restores performance;20%gate still fails

Frozen c38e450 before source scoring. Added one common blend action to066's
unchanged search outputs: original six forecast slots plus the arm's selected
configuration. Four six-hour convex log-space blends, equal solver/cost contract.
Control uses three current CV folds; ledger additionally uses16comparable,
strictly earlier matured production episodes. The ledger search itself also
retains066prior-backtest memory. This is a combined memory treatment, not an
isolated estimate of the contribution of either memory component.

| Approach | Overall416 | Electricity208 | Pedestrian208 |
|---|---:|---:|---:|
| Search plus current-only blend |0.2576259533432357|0.10690353854294049|0.4083483681435309|
| Search plus ledger blend |0.2510782023798801|0.10483908831312373|0.3973173164466364|
| Fixed six strong block-CV |0.2587110657586681|0.10748648550251698|0.40993564601481913|
| Prior lifetime ledger061 |0.25198521475530405|0.10734528573584194|0.3966251437747662|

New ledger blend reduces RMSLE2.54157%versus its matched control and2.95034%
versus the fixed strong guard. It is0.35995%better than061overall, with2.33471%
gain in electricity but0.17452%worsening in pedestrian. Early128-case gain versus
matched control2.78128%, later288-case gain2.44500%. No growing memory advantage
is established. All416cases were history-ready after the125available warm-ups.

This is a slightly lower original-development mean than previous approaches,
not evidence of a reliable or held-out gain. The frozen gate fails both20%
requirements and the positive domain-wise incumbent guard. No paid confirmation
or protected final access. Main/PyPI unchanged; no actual Gnomon/Hermes/API trial.

832weight fits completed,51,243iterations,13.298s wall/12.463s CPU. Zero new raw
forecasts or API calls. Do not interpret13seconds as total model-search cost:
066incurred24,032new forecast fits,11,902surrogate solves and31,378logical
attempts per arm including warm-ups;12,984original and12,600guard-history
forecasts are additional inherited preparation. The045anchor weights reused
416previous current-only fits. Detailed original costs remain in their receipts.
Five synthetic tests passed before freeze: gradient finite differences, convex
certificate/geometric combine, duplicate selected slot, invalid shapes and
recording cutoff filtering. Two synthetic fits completed; malformed inputs
rejected before solving. No source-data parameter tuning in this run.

Independent47,500-check audit passed: all832fit training inputs, historical
selection/visibility, actual config IDs behind the selected role, independent
convex objective/gradient gap, simplex constraints, every produced blend and
all score aggregates. Audit6.034s, zero provider/API calls. It relies on066's
previously audited raw model forecasts; it does not independently refit sklearn.

Evidence root results/search-ensemble-068-001; complete compressed archive and
hash receipt evidence/search-ensemble-068.json. Source access inventory and
configuration identities show that historical selected slots use the original
arm's historical executions, never today's model retroactively backtested.

Conclusion for next development: blending is a better action than single-model
selection here, but the expanded search adds only a small gain over061 and
fails the domain guard. Do not promote it or spend on final confirmation.
Any further change must address an identified residual failure pattern and be
frozen before scoring; do not keep optimizing arbitrary development variations
or weaken the strong baseline to manufacture the requested20%.

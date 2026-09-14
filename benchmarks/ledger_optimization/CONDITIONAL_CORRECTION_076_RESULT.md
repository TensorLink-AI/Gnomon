# Conditional correction076: negative development result

Frozen25330ad, five synthetic tests. All416paired tasks completed. The common
11-feature correction learns from base forecast level, seven model disagreements
and relative lead cycles. Training scalers and corrections use current CV only
in control and the frozen068current/16mature-history mixture in ledger. No
current outcome enters fitting. The objective matches smoothed mean-case RMSLE
plus a fixed.1coefficient penalty; no parameter sweep was performed.

| Policy | Mean per-case RMSLE | Electricity | Pedestrian |
|---|---:|---:|---:|
| Corrected control |0.292136730254824|0.1251463544213952|0.4591271060882528|
| Corrected ledger |0.2585750050995435|0.10839169538778892|0.40875831481129804|
| Uncorrected068control |0.2576259533432357|0.10690353854294049|0.4083483681435309|
| Uncorrected068ledger |0.2510782023798801|0.10483908831312373|0.3973173164466364|
| Strong050 |0.2587110657586681|0.10748648550251698|0.4099356460148192|
| Previous061ledger |0.25198521475530405|0.10734528573584194|0.3966251437747662|

The11.48836%ledger advantage versus corrected control is insufficient and is
mostly associated with a degraded control: corrected control is13.39569%worse
than its uncorrected forecast. Corrected ledger is2.98584%worse than068ledger,
0.36838%worse than uncorrected068control and only0.05259%better than strong050.
Both domains worsen versus068ledger. All relevant promotion gates fail. This
is not evidence of meaningful improved ledger forecasting or a20%gain.

The correction produces forecasts outside the seven-model range on2,212control
and899ledger leads (9,984leads per arm);14and5leads respectively are clipped to
zero. Thus the expanded action was actually exercised, but it did not improve
future performance over the previous blend. This does not identify overfitting
or distribution shift as a proven cause; those remain hypotheses.

832new correction fits,21,267optimizer iterations,24,859function evaluations,
3.796s wall/3.519CPU. No source-model/API calls. Inherited costs:49,616raw model
computations,416045anchors,832068blend fits,11,902search surrogate solves and
31,378logical search attempts per arm. Full source identities are retained.

Independent audit29,198checks, zero failures,3.010s: all832training designs and
scalers reconstructed, source temporal/lead alignment checked, objectives and
gradients recomputed, and strong-convexity suboptimality bounds verified. All
predictions, clips, extrapolations, scores and guard comparisons checked. This
establishes numerical execution integrity for the tested rule, not generalization.

Complete evidence is archived with member hashes and tracked receipt at
`evidence/conditional-correction-076.json`. No protected validation/final data,
actual1.2.0/DeepSeekagent run, main merge or PyPI change. Goal remains unmet.
068remains the preregistered development incumbent. Further flexibility alone
has not solved generalization; any next method must learn when a correction is
supported and retain a strong uncorrected option under identical arm rules.

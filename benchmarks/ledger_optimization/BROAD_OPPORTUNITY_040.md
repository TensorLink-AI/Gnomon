# Development opportunity bounds 040

Before testing another reliability gate, quantify the best possible result from
accepting/rejecting already-tested proposals. Read only the 416 immutable cases
of 038 and the corresponding 039 choices, verified against their tracked hashes.
No new forecasts, provider calls, final values or alternative datasets.

All bounds below use actual future error deliberately and are **nondeployable
hindsight diagnostics**, not ledger policies or estimated agent performance:

1. Perfect accept/reject filter for the original recent-history proposal.
2. Perfect accept/reject filter for the CV/history blend proposal.
3. Perfect accept/reject filter for the CV-bias calibration proposal.
4. Perfect choice among current CV and those three proposals.
5. Perfect choice among all six executed recipes, available at every origin.
6. Perfect choice among all six only after 3, 4, 8 or 10 earlier same-series
   matured origins; current CV during the stated cold start. These limits apply
   only to policies constrained to retain CV before that support count, not to
   policies transferring evidence or making other cold-start choices.

Report overall and each domain, retaining equal case weights. Report how much
of the unrestricted six-recipe oracle's improvement must be captured to reach
20%, and the maximum allowed remaining regret at that target. If even a perfect
filter fails, do not spend on tuning that filter as a standalone route to 20%.
This does not rule out stronger proposal mechanisms, different shared recipes,
prediction calibration, or ledger value under other objectives.

These tasks are already-used development evidence; none of these results may
be described as untouched confirmation or a confidence interval. Freeze this
diagnostic list before calculating it. Preserve all results, including bounds
that contradict a preferred next step, and account separately for the original
9,984 forecast computations and this read-only analysis runtime.

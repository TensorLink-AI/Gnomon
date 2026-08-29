# v0.6 loop I002: short-history candidate decision

Decision: **reject Toto2-4M as a new short-history default; complete P2 as an
evidence-backed no-build.** Existing robust fallback and evidence-weighted
admission defaults remain unchanged.

The preregistered expansion compared a frozen classical baseline with the
pinned Toto2-4M supply on 10 MIMIC T2 cases and 10 causal-chambers T2 cases.
All 70 expected forecast channels completed in both arms. The candidate's
combined paired median relative sMAPE improvement was -1.65% (33 wins, 37
losses; deterministic paired 90% bootstrap interval -9.80% to +3.10%). MIMIC
was +0.71% with an interval crossing zero; causal chambers was -30.43%, with
one win and nine losses. This fails the frozen 5% effect, positive-interval,
win-count, and per-domain non-inferiority gates.

The preceding four-row smoke also exercised governed admission. It preserved
the baseline on 8/14 channels and used a provenance-complete shrinkage blend
on 6/14. That result did not justify a larger governed run after candidate
supply failed; stopping was part of the protocol, not selective reporting.

Useful general fixes promoted from the iteration are benchmark and operations
infrastructure, not a forecasting-policy change:

- engine-only evidence-weighted admission can be benchmarked without an LLM;
- local forecast workers can be capped explicitly at one;
- retained rows include model/admission/warning provenance;
- optional-model installation materializes its worker before declaring ready;
- per-channel scoring maps output aliases to history and reports paired sMAPE,
  paired relative effects, and a deterministic paired bootstrap interval.

Raw resumable rows and actual engine analyses remain under
`results/v06-p2-i002-smoke-v3/` and `results/v06-p2-i002-expanded/`. They are
intentionally not copied into this aggregate release. The competition-specific
`docs/astrid-btc-agent-plan.md` is not part of this iteration or release.


# Development opportunity audit 013

Registered before computing this audit. This diagnoses the existing development
benchmark; it does not replace the 20% mean per-case RMSLE objective, select a
new dataset or authorize another confirmation. Use only the same hash-locked
208 development cases as screens 011/012. No API or provider calls.

Questions fixed in advance:

1. How much error can selection among the unchanged forecasts remove relative
   to the deterministic current-CV reference? Compute both the future-aware
   per-case minimum and the best constant provider per series within each
   reported slice. Both use future outcomes and are diagnostics, never policies.
   They are not ceilings relative to a different, unobserved agent control.
2. How much of that opportunity does the frozen, past-only support rule capture?
   Keep the entire primary development grid, including cold starts. Report all,
   cold origins 0–3, mature 4–25 and the already-used later slice 18–25. Do not
   select a favorable slice or exclude equal predictions.
3. Does correct provider-to-history linkage matter? In 256 fixed random trials
   (Python Random seed 20260912), permute the provider labels on historical
   scores once per series, consistently across its origins. Re-run the frozen
   support rule using only each origin's eligible past records. Keep current
   CV, actuals, candidate predictions and metric calculation unchanged. Retain
   every mapping, per-case choice and trial score, including identity mappings.
   This is deliberately corrupted metadata for a diagnostic negative control,
   not fair information for a live no-ledger agent. Never expose it to users or
   write it to the source ledger. Use a separate copied history view.

The shuffle tests whether the existing policy uses the relationship between
provider identity and past loss. It does not isolate all memory benefits, and
its distribution is neither an uncertainty interval nor a valid superiority
p-value: providers are not assumed exchangeable. Report median and extrema,
and the fraction of trials with higher error than correctly linked support.
Do not tune mappings, shuffle seeds, policy thresholds or sample count.

Compute gates separately: whether per-case hindsight headroom reaches 20% against
this CV reference, whether correct-history support beats the median corruption,
and whether support itself reaches 20%. None proves the live-agent objective.
No new final cohort may be selected for favorable realized headroom. Future
benchmark design must choose source, eligibility, time spans, candidates and
splits using task relevance and pre-origin information, before hidden outcomes.
Prospective development gates may stop an unpromising design; they may not reroll
final cases until the desired improvement is possible.

Validate independent scores, unchanged input bytes, all 208 screen-007 incumbent
choices, future/late-outcome exclusion and deterministic input ordering. Reject
nonfinite numbers, duplicate origins, nonconsecutive grids and invalid forecast
time ranges. Retain inherited availability assumptions explicitly. No main,
release or PyPI changes, and no claim that the objective has been attained.

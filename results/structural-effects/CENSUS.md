# Full-grid census — and why the full-grid A/B is not run

Completed grid, 2026-08-04: 355 runs, 71 tasks × 5 seeds,
`deepseek/deepseek-v4-flash-0731`, flag-on (`--future-context`, no
`--structural-context` — the proposer was not yet told the structural
class exists). Run at code revision `0c9a63e`. Raw summary:
`census-full.json` (committed from the benchmark machine). No score
from this run is quoted anywhere, here or elsewhere.

## Headline: the grammar held out of sample

| quantity | before (prefix run) | completed grid |
| --- | --- | --- |
| proposals through the gate | 192 | 258 |
| admitted | 26 (13.5%) | **178 (69.0%)** |
| admitted overrides | 10 | 146 |
| admitted constraints | 16 | 32 |
| span-parse rejections | 153 | 22 |

Every grammar pattern was built from tasks 1–37; tasks 38–71 arrived
blind, and the span-parse rejection count still fell 153 → 22. The 7
surviving `numeric_no_parse` are date-only weather statements plus one
cessation sentence — correct rejections, not gaps.

`window_is_future_only` is now the dominant rejection (50 of 80): the
live gate has moved from "can the span be read" to "is the window
dated correctly". Whether those 50 are proposer misdating or genuinely
fold-testable events correctly routed to the ablation gate is the next
measurement, not assumed here.

A caution that is not a score: 146 admitted overrides move a lot of
forecasts. Whether they moved them *well* is exactly what the paired
scored comparison exists to measure, and nothing in this census
prejudges it.

## The structural-effects decision

The entire 355-run grid contains **2** cessation-shaped spans, both
the same sensor sentence. A full-grid A/B would almost certainly fail
H2 (≥ 3 admissions) on frequency alone, telling us nothing about
whether the verb works — only that the corpus is thin, which this
census already establishes at zero cost.

**The full-grid A/B is therefore not run.** The registered experiment
proceeds as a targeted A/B under the pre-run terms below, registered
before that run exists.

## Score-level spot checks (2026-08-04, matched seeds, three cases)

Three matched-run deltas from the paired spot-check, each a different
lesson; scores quoted only as per-run deltas, never as arm means:

1. **DirectNormalIrradianceFromCloudStatus seed 1** (control 0.0364,
   Gnomon 0.7615): "the weather will become clear/cloudy" statements
   carry the whole answer and no lane can express them. Not a missing
   verb: a categorical state whose level implication is unknown is a
   *future-known covariate*, and the covariates lane — leakage-safe
   ablation and all — is its existing home. The roadmap item is wiring
   covariate proposals into the adapter, not a new event class.
2. **SensorTrendAccumulationTask seed 1** (control 0.0250, Gnomon
   0.7838): the cessation sentence, proposed as an override and
   correctly rejected — while the engine keeps extrapolating a drift
   the text says has stopped, at `supported`. One unexpressible
   sentence costs ~0.76 RCRPS. The class is rare (2 spans in 355 runs)
   but expensive per instance, which is the measured motivation for
   the targeted A/B below.
3. **FullCausalContextExplicitEquationBivarLinSVAR seed 1** (control
   0.0010, Gnomon 1.0491): three admitted overrides applied a
   *covariate's* stated values to the forecast target, disclosed as
   context_trusted — the largest single regression in the matched set,
   and caused by the grammar fix: the pre-fix parser could not read
   the spans at all. Fixed the same day: spans that name a foreign
   referent ("the covariate X_0 …") are rejected before any parse
   (`span_describes_the_target`), and the proposer instructions say
   so. Parsing a number says nothing about what the number refers to.

## Pre-run terms for the targeted A/B

1. **Scope.** The task families whose contexts contain cessation
   statements (identified from this census's gate records), all 5
   seeds, both arms. Same code revision for both arms (`e114137` or
   later; arms must match).
2. **H2 stands at ≥ 3 admissions**, and the bar is fair rather than
   vacuous for one reason: this census ran without the structural
   instructions, so its 2 proposals are underproposal. The A/B's
   treatment arm actively teaches the class, on tasks known to contain
   the sentence, at 5 seeds. A verb that cannot fire 3 times under
   those conditions has failed to fire.
3. **The bit-identity clause is restated at the engine level.** The
   treatment arm's proposer prompt necessarily differs (it carries the
   structural instructions) and temperature-1.0 resampling changes all
   proposals, so run-level bit-identity across arms is unmeasurable by
   construction. What the falsifier protects is restated as what is
   measurable: (a) the engine invariant — identical inputs with the
   flag on and no admitted structural event produce identical rows —
   is pinned by unit test and audited from artifacts; (b) the scored
   comparison is restricted to matched task-seeds and reported
   separately for runs with and without structural admissions. This
   restatement is registered before the A/B runs and may not be
   revised after it.
4. H1's direction, harm-case reporting, and the derived-from-the-path
   audit are unchanged from `HYPOTHESIS.md`.

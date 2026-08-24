# Cross-model evidence evaluation — 2026-08 (DeepSeek, GLM‑5.2, Qwen‑3.8)

Status: dated record of a three-model synthesis, written 2026-08-24. The
product decision below is implemented: the model-assisted lane publishes at
`results[*].model_assisted` (see
[results and artifacts](results-and-artifacts.md)) and the reasoning packet
ships as the `reasoning.packet` evidence dossier with
`verify_packet_selection` as its gate. It
consolidates matched TemporalBench-style comparisons (80 T2/T4 tasks per
arm: 240 choices, 480 forecast channels) run with DeepSeek V4 Flash,
GLM‑5.2, and Qwen‑3.8 27B, each compared direct-model versus
Gnomon-compiled Evidence. The raw paired records live in the untracked run
artifacts named by each run's provenance; this file records the
cross-model conclusion and the product decision it forces.

The one-sentence result:

> Gnomon reliably improves execution safety and sometimes efficiency, but
> it has not yet demonstrated that it improves an LLM's temporal reasoning
> accuracy.

## Cross-model results

| Model | Base choice accuracy | With Gnomon | Difference | Forecast result |
| --- | ---: | ---: | ---: | --- |
| DeepSeek V4 Flash | 31.7% | 33.3% | +1.6 pts | Numerically better record sMAPE, not significant; per-channel MASE significantly worse |
| GLM‑5.2 | 35.8% | 32.5% | −3.3 pts | Effectively tied |
| Qwen‑3.8 27B | 37.5% | 32.5% | −5.0 pts | Significantly worse through Gnomon |

None of the choice-accuracy differences was statistically conclusive. More
importantly, there is no repeatable positive effect: one small increase
and two decreases.

### Reliability

| Model | Base valid forecasts | Gnomon valid forecasts |
| --- | ---: | ---: |
| DeepSeek | 80/80 | 80/80 |
| GLM‑5.2 | 77/80 | 80/80 |
| Qwen‑3.8 | 71/80 | 80/80 |

This is Gnomon's strongest and most consistent result. It prevents
malformed horizons, missing channels, unstructured answers, and
unsupported publication. Qwen failed nine forecasts that Gnomon completed
without breaking any valid case.

### Efficiency

- GLM: approximately 65% fewer tokens through compiled Evidence.
- Qwen: approximately 38% fewer tokens.
- DeepSeek's older autonomous MCP run was dramatically more expensive —
  about 1.97M versus 253K tokens — because the agent navigated repeated
  tool calls.

Compiled Evidence is the right delivery mechanism. Autonomous MCP
exploration is currently too expensive and model-dependent.

## What the synthesis establishes

1. **Gnomon is a successful guardrail.** It makes weak or unreliable model
   behavior bounded and visible: valid forecast shapes, complete
   multi-series answers, immutable primary numbers, explicit support
   labels, safe fallback, lower output variance.

2. **It is not yet a reasoning amplifier.** If Gnomon were improving
   temporal reasoning, accuracy would rise consistently across model
   families. Instead Gnomon drives all three models toward almost the same
   32.5–33.3% choice accuracy. That looks like a harness ceiling: the
   deterministic evidence/choice projection dominates the underlying
   model's reasoning. Better base-model reasoning is being compressed
   rather than enhanced.

3. **Safe fallback becomes harmful as the model improves.** Gnomon
   published last-value forecasts throughout the GLM and Qwen comparisons.
   Against GLM, last-value was roughly competitive. Against Qwen, direct
   model forecasts contained useful signal and significantly beat Gnomon.
   Gnomon therefore protected users from malformed answers while
   discarding useful predictive information. This is the most important
   result: "never worse than naive" is a valuable safety property, but
   publishing naive when a model has useful signal is not a good product
   answer.

4. **The evidence packet improves behavior, not discrimination.** The
   packet successfully causes models to produce shorter answers, follow
   the response contract, preserve provenance, avoid unsupported
   certainty, and complete the task. But it does not yet help them
   distinguish between competing temporal interpretations. It gives the
   model a governed conclusion more than it gives it the evidence needed
   to reason toward a better conclusion.

5. **Stronger models expose the limitation more clearly.** Base accuracy
   rose across the tested models (DeepSeek 31.7% → GLM 35.8% → Qwen
   37.5%) while Gnomon stayed almost fixed around 32.5%. Gnomon currently
   helps weaker models through structure while constraining stronger
   ones.

## Product decision

Gnomon retains two distinct lanes:

1. **Immutable governed lane** — the safest backtested forecast and
   canonical computed facts. Unchanged: this lane's numbers remain the
   primary, quotable, receipt-backed answer.
2. **Model-assisted lane** — the LLM or TSFM interpretation/forecast,
   admitted using evidence, plausibility checks, cross-channel
   consistency, and whatever out-of-sample validation the history
   permits.

When validation is underpowered, the second lane is labelled
`prior_assisted` or `conditionally_supported` — not suppressed
automatically. The user receives the useful answer and still understands
precisely what supports it.

The reasoning packet also changes from "here is the canonical choice"
toward an evidence dossier:

- observations;
- relevant temporal properties;
- supporting and conflicting evidence;
- interpretations still compatible with the data;
- evidence sufficiency;
- what would distinguish the alternatives.

The model then selects and explains the conclusion, while Gnomon verifies
that its claims follow from the supplied evidence.

## Bottom line

Gnomon already makes temporal answers safer, more complete, more
auditable, and — when compiled — cheaper. It does not yet make models
consistently better at temporal reasoning, and its rigid fallback can
erase useful signal from stronger models. The next product milestone is
not another envelope refinement: it is preserving useful model priors
inside a governed, explicitly labelled lane while strengthening Gnomon's
own forecasting floor.

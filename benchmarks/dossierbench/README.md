# DossierBench

The matched uplift experiment for the evidence dossier: does a model
reading Gnomon's restructured reasoning packet answer temporal
interpretation questions better than the same model alone, and better
than the same model handed only the computed conclusion?

## Real data, real outcomes

By default every case is a windowed slice of a **real observational
series** (Mauna Loa CO₂, sunspots, the Nile, US macro aggregates, El Niño
SST — provenance in [`data/README.md`](data/README.md)), cut at a sampled
instant. The truth label is not authored by any generator: it is what the
**realized future window** — held out from the model and every packet —
actually did, measured under the same deterministic window semantics
Gnomon uses in production. Transition labels require a `supported`
realized outcome; null outcomes are admitted at disclosed `weak`
confidence (metrics are split by label confidence), so the task keeps
both halves of discrimination: calling a transition, and knowing when not
to. Sampling is soft-balanced across (property, truth) cells so a
constant strategy cannot ride a majority class, and an `always_majority`
reference is scored anyway.

Because these series must be assumed memorized by every LLM, each case
passes through a seeded positive affine transform that preserves every
asked-about dynamic (direction, trend sign, volatility ratios, breaks,
seasonality) while defeating verbatim sequence lookup; prompts carry
values only — no names, no dates — and the harness verifies before any
model call that the held-out future appears in no prompt of any arm.

The difficulty is real: on this corpus the deterministic references sit
near chance (copy-the-conclusion ≈ 25%, copy-the-discriminator ≈ 42%,
majority ≈ 35% at the default seed), unlike the synthetic diagnostic mode
where the mechanism scores ~92%. There is genuine headroom, in both
directions. `--source synthetic` keeps the seeded generator available as
a mechanism diagnostic.

Three model arms per known-truth case (same model, temperature 0,
prompts differing by the evidence block alone):

| Arm | Receives |
| --- | --- |
| `control` | the raw series and the question |
| `conclusion` | the series plus the computed canonical value and support — the "here is the conclusion" packet style |
| `dossier` | the series plus the full reasoning packet (interpretations, measured held-out discrimination, sufficiency, selection contract), gated by `repair_selection` with one repair round and the labelled canonical fallback |

For future-outcome questions, a supported observed-window measurement remains
visible but is never binding unless the packet also contains the required
forecast-valid evidence. The selection contract records the requested
inference mode, whether its requirements were satisfied, and the missing
evidence; descriptive confidence cannot silently become predictive authority.

Three deterministic references are scored beside them at zero API cost:
`chance`, `copy_conclusion`, and `copy_discriminator`. The summary's
`verdicts` block separates the claims that matter:

- `uplift_over_model_alone` — is the dossier worth sending at all;
- `uplift_over_conclusion_packet` — did restructuring the packet (the
  cross-model evaluation's recommendation) change anything;
- `model_beyond_mechanism` — dossier-arm accuracy minus
  `copy_discriminator`. Near zero with matching accuracy means the model
  transcribed the strongest number in the packet — the harness ceiling
  the cross-model evaluation described, not reasoning uplift.

Run (needs an API key; ~3 model calls per case, plus repair turns):

```
uv run python benchmarks/dossierbench/run_dossierbench.py \
  --model <model> --cases 240 --output-dir results/dossierbench/<run>
```

Truth labels exist only in the scorer; the harness verifies before any
model call that the arms share the identical question, that no packet
carries a truth field, and that the held-out future leaks into no prompt.
Properties are trend/level/volatility, where the conclusion machinery and
the discriminator share a public vocabulary. Rows are appended durably
(`rows.jsonl`, `--resume` continues an interrupted run), and paired
McNemar tests are reported for all three arm pairs. At 240 cases the
paired design resolves moderate effects; treat smaller deltas as
diagnostic and rerun at higher `--cases` before concluding.

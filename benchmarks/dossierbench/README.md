# DossierBench

The matched uplift experiment for the evidence dossier: does a model
reading Gnomon's restructured reasoning packet answer temporal
interpretation questions better than the same model alone, and better
than the same model handed only the computed conclusion?

Three model arms per known-truth case (same model, temperature 0,
prompts differing by the evidence block alone):

| Arm | Receives |
| --- | --- |
| `control` | the raw series and the question |
| `conclusion` | the series plus the computed canonical value and support — the "here is the conclusion" packet style |
| `dossier` | the series plus the full reasoning packet (interpretations, measured held-out discrimination, sufficiency, selection contract), gated by `repair_selection` with one repair round and the labelled canonical fallback |

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
model call that the arms share the identical question and that no packet
carries a truth field. Cases come from the DiscriminationBench generator
restricted to trend/level/volatility, where the conclusion machinery and
the discriminator share a public vocabulary. Rows are appended durably
(`rows.jsonl`, `--resume` continues an interrupted run), and paired
McNemar tests are reported for all three arm pairs. At 240 cases the
paired design resolves moderate effects; treat smaller deltas as
diagnostic and rerun at higher `--cases` before concluding.

# BreachBench

The client-job benchmark. Gnomon's first product job is operational
threshold risk — *which metric may breach a meaningful limit, when, and
whether the evidence supports intervening* — and BreachBench scores
exactly that deliverable, priced in the client's own units.

Every case is a windowed slice of a **real telemetry-flavoured series**
(real Wikipedia traffic, a real 5-minute sensor, real pedestrian counts
with a genuine COVID regime break, real US retail sales — provenance in
[`data/README.md`](data/README.md)) with an alert threshold set at the
recent maximum plus a sampled robust-scale margin. Ground truth is what
the **realized future** did: breach or not, and at which step — held out
from the model and from Gnomon, verified absent from every prompt.
Gnomon is fed each series' **true cadence** (5-minute, daily, monthly —
read from the corpus filenames), because season detection and support
tiers are frequency-aware. Realized futures **never overlap** within a
series (cutoffs are horizon-spaced), so no two cases share a breach
event; labels from one regime can still co-move, which is why the
per-series case counts are disclosed rather than an independence claim
made. Histories may overlap and that is disclosed. Label invariance
under anonymization is verified on the *rounded numbers the model
actually sees* — a case whose rounding flips any breach step is
discarded, never shipped.

Two matched model arms (same model, temperature 0, prompts differing by
the evidence block alone):

| Arm | Receives |
| --- | --- |
| `control` | history, threshold, costs, question |
| `gnomon` | the same plus **Gnomon's real production output for this exact call** — `forecast(threshold=…)`'s headline, support tier, per-step breach probabilities, interval path, warnings, and the model-assisted lane |

The primary metric is **decision cost and regret** under a stated cost
model (acting costs 2 and mitigates; a missed breach costs 10), because
that — not choice accuracy — is what useful means to an operator.
Breach-call accuracy, recall, and false-alarm rate are reported beside
it, scored over each arm's *valid* answers only (imputing "no breach"
for garbage would flatter a failing arm with base-rate accuracy;
`invalid_rate` and `call_metrics_scored` disclose the denominators).
First-breach timing error is likewise over each arm's self-selected
answered subset — `timing_answer_rate` is reported next to it because
an arm can dodge the metric by never naming a step. The breach base
rate is held near the cost break-even (~30% against a 0.2 break-even),
where neither constant policy is close to optimal and only genuine
discrimination reduces regret.

Deterministic references bound everything at zero API cost: the product's
governed dependence-aware policy (with withholding priced as monitor by
omission), the legacy peak-marginal and independence-composed diagnostics,
naive persistence, always-act, never-act, and the hindsight optimum. The
matched report separately measures whether the agent preserved a supported
governed recommendation and whether it acted after Gnomon withheld authority.
The `verdicts` block demands all three
margins before "useful" is claimed:

- cheaper decisions than the model alone (Gnomon added value),
- cheaper than the product's governed rule alone (the model added value), and
- cheaper than the best constant policy (someone actually read the data).

Run (~2 model calls per case; the Gnomon runs are local and free):

```
uv run python benchmarks/breachbench/run_breachbench.py \
  --model <model> --cases 180 --output-dir results/breachbench/<run>
```

Anti-gaming: seeded positive affine anonymization per case with the
threshold transformed identically (breach structure and timing exactly
invariant, verified on the rounded shown numbers), values-only prompts,
soft-balanced outcome cells with the achieved mix disclosed, paired
exact sign tests on per-case decision cost, durable resumable rows.
Operationally: a malformed model answer (NaN steps, wrong types)
degrades to "monitor by omission" and is recorded as invalid — it never
crashes a paid run; a failed API call is recorded and the run fails
loudly at the end with every completed row saved (`--resume` finishes
the remainder, and every row carries the full dataset identity —
generator version, seed, case count, corpus hash — plus the answering
model, so rows from a different configuration are rejected rather than
silently pooled); API token usage and cost land in `summary.json`. BreachBench is one panel of the usefulness
suite: DossierBench measures interpretation discrimination,
DiscriminationBench the mechanism, temporalbench/reasoningbench/cik the
broader agent behaviours, and LeakTrap the grounding floor.

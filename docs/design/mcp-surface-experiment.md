# MCP surface experiment

Status: superseded by the fresh workflow experiment selecting `evidence` as
the default, 2026-08-14. Historical protocol retained below.

The number of exposed tools is not itself a product metric. A smaller surface
may reduce distraction and schema tax, or merely hide useful capabilities.
Gnomon therefore flips its default profile only after matched agent runs show a
better answer at lower conversation cost.

The matched result is recorded in
[`mcp-surface-experiment-results.md`](mcp-surface-experiment-results.md).

## First matched experiment

Run the same model, endpoint, task rows, temperature, and harness caps twice:

```bash
python -m benchmarks.temporalbench.run_temporalbench \
  --data-dir "$DATASET" --condition gnomon-mcp --model "$MODEL" \
  --mcp-profile full --output-dir results/surface-full

python -m benchmarks.temporalbench.run_temporalbench \
  --data-dir "$DATASET" --condition gnomon-mcp --model "$MODEL" \
  --mcp-profile core --output-dir results/surface-core
```

Repeat the second command for `describe`, `evidence`, and `mega`, changing
only `--mcp-profile` and `--output-dir`. A completed Phase 3 result requires
all five arms (`full` plus the four candidates) over the same task rows. A
profile that was only piloted on a different row, prompt, or provider window
is a probe, not a matched arm.

Each summary records the selected profile plus cumulative conversation tokens,
median and p95 MCP calls, and the exact serialized tool-schema bytes. Existing
outputs retain accuracy by tier, forecast coverage and support mix, routes,
abstentions, and harness-voided rows. Compare accuracy only on matched rows;
price abstention into the result instead of dropping it.

For long sweeps, run resumable one-row shards with `--offset N --limit 1`.
Provider or process failure then loses one row rather than the whole experiment.

## Pre-committed decision rule

The candidate default must not regress the full control on the
leakage-controlled subset, must keep leaktrap at 0/40, and should reach at least
80% answer yield, no more than 50,000 cumulative conversation tokens per task,
no more than two calls at the median, and no more than four at p95. Report
accuracy split by evidence tier and quote the schema byte count beside token
results.

The first completed matrix retained `full`: all candidates missed the token
and median-call gates, and `evidence` also missed yield. The profiles remain
executable so a future candidate or model can be evaluated under the same
rule; the default does not change from tool count alone.

The executable arms are `full`, `core`, `describe` (core plus the descriptive
evidence verb), and `mega` (inspect/run/track), with `decision` and `data` as
specialized controls. The two prototypes are explicit profiles and never alter
the production `full` surface. If evidence injection still beats every tool
arm, the binding fallback is a compact evidence-pack surface centered on
description plus forecast, rather than adding tools.

That fallback is executable as the `evidence` profile: exactly
`gnomon_describe` plus `gnomon_forecast`.

The evidence profile's serialized schema is CI-pinned below 12 KiB. Existing
forecast responses retain the frozen per-series support shape; bulk is bounded
through forecast-row trimming and wide-panel triage, while the experimental
describe response has a 2.4 KiB budget. This preserves compatibility instead
of deleting support fields merely to hit a nominal byte target.

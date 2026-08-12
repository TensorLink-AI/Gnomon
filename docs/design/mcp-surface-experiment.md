# MCP surface experiment

Status: executable measurement protocol, 2026-08-12.

The number of exposed tools is not itself a product metric. A smaller surface
may reduce distraction and schema tax, or merely hide useful capabilities.
Gnomon therefore flips its default profile only after matched agent runs show a
better answer at lower conversation cost.

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

Each summary records the selected profile plus cumulative conversation tokens,
median and p95 MCP calls, and the exact serialized tool-schema bytes. Existing
outputs retain accuracy by tier, forecast coverage and support mix, routes,
abstentions, and harness-voided rows. Compare accuracy only on matched rows;
price abstention into the result instead of dropping it.

## Pre-committed decision rule

The candidate default must not regress the full control on the
leakage-controlled subset, must keep leaktrap at 0/40, and should reach at least
80% answer yield, no more than 50,000 cumulative conversation tokens per task,
no more than two calls at the median, and no more than four at p95. Report
accuracy split by evidence tier and quote the schema byte count beside token
results.

The executable arms are `full`, `core`, `describe` (core plus the descriptive
evidence verb), and `mega` (inspect/run/track), with `decision` and `data` as
specialized controls. The two prototypes are explicit profiles and never alter
the production `full` surface. If evidence injection still beats every tool
arm, the binding fallback is a compact evidence-pack surface centered on
description plus forecast, rather than adding tools.

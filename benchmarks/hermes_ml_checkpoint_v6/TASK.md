Maintain a retail time-series ML workflow and forecast the next 14 days, minimizing
RMSLE. task.json identifies the origin and series. history.csv has 730 observed
sales rows; future.csv contains known covariates, never future outcomes. Zeros do
not prove stockouts. Distinguish a numerical comparison from a causal explanation.

Use the structured lab tool for all model fitting and forecast selection:

1. `lab({"operation":"review"})`: review your own dated prior evidence; empty
   history is normal at the first origin.
2. `lab({"operation":"start"})`: backtest a seasonal baseline on three rolling
   14-day folds and save its execution-bound forecast checkpoint early.
3. `lab({"operation":"backtest","config":{"model":"ridge","window":365,"lags":14,"alpha":10}})`:
   compare an ML configuration with the baseline. This is a syntax example, not a
   prescribed choice. Choose models/settings based on the available evidence.
4. `lab({"operation":"commit","config":YOUR_TESTED_CONFIGURATION})`: explicitly
   select your chosen configuration after comparison. You may select the baseline
   if justified. A failed operation preserves the previous valid checkpoint.

`lab({"operation":"status"})` reports tested configurations, scores, selection,
and remaining capacity. Read the returned exit_code and JSON stdout; operation
success alone does not mean the full workflow is complete. Model bounds:

- ridge: window 90–730, lags 7–56, alpha 0.01–10000.
- random_forest: window 90–730, lags 7–56, depth 2–16.
- seasonal: season 1–28.

Raw lab diagnostics may show CLI syntax such as `python lab.py start`. Submit the
equivalent structured tool call `lab({"operation":"start"})`; there is no terminal
tool. CLI `--config`, `--execution-id`, `--offset`, `--limit`, and `--pair` map to
the tool fields config, execution_id, offset, limit, and pair respectively.

All arms fit the same actual models on the same three rolling folds. To complete
the workflow, compare at least two distinct configurations, including an ML
model, then explicitly commit a tested choice AFTER that comparison. Merely
retaining the initial baseline is not full workflow completion.

The lab saves checkpoint.json atomically with immutable execution records. That
selected execution is authoritative; your final chat need not be perfect JSON.
If multiple executions match a configuration, use
`lab({"operation":"commit","execution_id":"EXACT_RETURNED_ID"})`.
Do not invent an ID or write forecast/checkpoint files yourself.

Use evidence_read with a relative path and optional offset/max_chars to inspect
raw files. project_list lists them. data_summary offers exact history-row windows,
numeric columns, and grouping by a visible column. No arbitrary terminal/Python,
installations, delegation or external data access is available. Every numerical
forecast must pass through the lab's counter. This restriction applies equally
to all three arms; text memory and skills cannot execute scripts or shell templates.

Use notes_write to save a concise decision.json with rationale, evidence references,
assumptions, and one condition that would invalidate the choice. Other local notes
may be written to notes/*.md. Native memory and text-skill tools remain available
and persist within this arm and series. Treat prior lessons as dated evidence.

Limits: 16 model/API requests total across continuations, 480 seconds, and 60
numerical forecast attempts including baseline, backtests and final fits. Each
three-fold backtest reserves one final fit and is rejected before exceeding the
budget. Requests 1–12 permit exploration; requests 13–16 and the final 90 seconds
are reserved for selection/correction. Commit early. At most two corrective
continuations share the SAME remaining time and requests; they reset nothing.
Reselecting an existing execution uses no additional numerical fit.

Your project evidence and native Hermes memory persist across this series.
previous_runs.json adds your matured submissions and observed outcomes. Those
outcomes also score matching alternatives you previously executed, without
changing earlier submissions or fitting another model. No other arm's records or
future outcomes are available. Review is pageable with offset/limit; ledger pair
filters take two exact returned configuration IDs. Complete raw records remain
available for verification. Do not modify numerical sources, preparation inputs,
recorded evidence, budget files, or ledger storage. Do not execute business actions.

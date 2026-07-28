# Design review decisions

These implementation decisions resolve issues found during the initial review
of the v0.1 product specification and system design.

1. Evaluation partitions are disjoint. Earlier rolling folds select a model,
   the penultimate fold calibrates residual intervals, and the final fold is a
   report-only test. Final-test results never change the selected model.
2. Invalid inputs return structured errors. A valid task with inadequate
   forecasting evidence returns a forecast artifact whose support is
   `unsupported`; it is not represented as an execution error.
3. Every artifact embeds the complete resolved data schema and policies rather
   than relying on a dataset identifier alone.
4. Context events are deferred until an event-history/analogue protocol can
   reconstruct only information known at every historical cutoff.
5. The implemented MVP boundary is the CLI and Python runtime. MCP, containers,
   project lifecycle, actual scoring, sharing, and TSFM adapters require their
   own release gates before being advertised as available.


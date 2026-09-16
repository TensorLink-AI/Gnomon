# Ledger + optional TSFM

A separate fourth arm adds **router** and **top-two ensemble** candidates to the
same Gnomon 1.2.0/Hermes task. All ten existing candidates and tools remain.
See [the protocol](PROTOCOL.md) for scope, chronology, costs and interpretation.

Remote root: `/root/online-retail-tsfm-002`; output: `run-001`.
The existing `/root/online-retail-two-week-002` run is not modified.
The first TSFM startup (`tsfm-001`) stopped on an advisory catalog HTTP 404
before any agent sessions. Its six successful API forecasts are retained and
reused in `tsfm-002`, without repeating those calls.

Run `python -m benchmarks.online_retail_ii.tsfm.controller --help` for required
paths. Use a frozen copy of the existing benchmark bundle with this additional
package. The controller requires its original baseline, panel, manifest and
ledger-arm plan, the same pinned Gnomon/Hermes runtimes, and host credential-file
paths. Never include credentials in the bundle or share raw key files.

`run-001/progress.json` reports agent progress and TSFM selections.
`tsfm-baseline-scores.jsonl` contains standalone API policy scores.
`host-api/` retains sanitized service receipts; `preflight/FINISHED.json` and
`pilot-gate.json` are admission receipts. `scores.jsonl` retains all arm results.

The separate `/root/online-retail-chronos2-001/run-001` control uses
`python -m benchmarks.online_retail_ii.tsfm.chronos_control --help` for its
arguments. It calls explicit Chronos-2 directly, with no Gnomon or LLM, over the
same 2,448 tasks. Progress, raw API receipts and scores are kept independently.

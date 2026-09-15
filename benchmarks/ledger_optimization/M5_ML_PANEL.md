# Synthetic-only validation of the final panel constructor

`m5_ml_panel.py` constructs the declared 24-series panel from already-loaded
rows, using the unchanged `m5_ml_adapter.build_series_jobs` for every series.
It performs no file or network access, has no command-line exporter, and grants
no final-data access. It is preparation code, not a completed final evaluation.

Before consuming numerical cells, it validates the declared panel: eight
development identities in two stores, four items per store; 24 reserved identities
in eight stores, three items per store; globally distinct items and series; and
no store/item/series overlap between splits. Selection-prefix hashes must be
well formed. An operational caller must separately authenticate the original
hash-pinned manifest; these structural checks cannot prove a caller-supplied
manifest is the original one.

The constructor accepts only an in-memory list or tuple of rows. It deliberately
rejects a lazy archive reader, leaving file access to a future gated caller.
Unselected rows' numerical cells are not consumed. Missing or duplicate selected
rows, invalid early history, changed selection-prefix values, or broken calendar
grids cause rejection. It never chooses a replacement series or pads data.

Every series uses the same 730 observations, three-fold-compatible history,
14-step horizon, period-end timestamps, assumed visibility, unavailable promotion
field and calendar features as the tested development preparation. The first
and last origin indices remain d_1577 and d_1927; the last target remains d_1941.
The result contains exactly 624 host tasks, before duplicating decisions across
three arms and two seeds. Host actuals stay separate from each model request.

Six synthetic tests passed. They verify the complete panel against the existing
shared constructor and independently check task counts, dimensions, origin dates
and outcome maturation endpoints. A future-sales perturbation changes the host
actual and subsequently visible history without changing the earlier request.
Other checks cover input-order invariance, poisoned unselected rows, invalid
panel metadata before numerical consumption, lazy-reader rejection, and refusal
to replace missing or malformed selected data. No real M5 source or manifest,
provider, model, Engy API, or final target was accessed in these tests.

Remaining before an operational final run: a complete eligible development
result, the selected immutable worker and full fair-comparison protocol, an
authenticated final access gate and archive reader, runtime/visibility verification,
and the one-shot final dispatch. The numerical analysis contract remains
`M5_ML_ANALYSIS.md`. This constructor's passing tests do not satisfy any accuracy
or uncertainty criterion and do not authorize opening the reserved panel.

Evidence: `results/m5-ml-panel-synthetic-001/` and
`evidence/m5-ml-panel-synthetic-001.json`. The live 097 experiment, staged candidate
100, main/PyPI and final-data gate are unchanged.

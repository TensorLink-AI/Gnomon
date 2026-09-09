# 1.1.6 acceptance feedback disposition

This document tracks the new feedback separately from the earlier release's
acceptance results. The implementation is included in the 1.1.7 release changes. The feedback includes conflicting preferences and one external Arena
adapter that is not present in this checkout or the local Arena checkout.

| Report | Disposition |
| --- | --- |
| Incorrect compare-history exclusion | Distinct missing-history, missing-future, history-after-origin, target-before-origin and recording-after-target reasons, with per-provider causes and required fields. |
| Replay failures hide cause | Per-fold origin, replay mode, actual validation cause, selected history counts and source/recording exclusions. Readiness points to visibility, not merely more folds. |
| Replay is discovered too late | Inspection advertises the default/basis. `evaluate --preflight` plans actual folds without calls or a saved study; `--replay` is an explicit semantic choice. |
| Incomplete production comparison example | [Complete controlled-clock example](production-history-comparison.md), including built-ins with a caller-owned TemporalLedger. |
| Jitter recovery discards history | Strict grid errors probe bounded safe alignment first, preserve values/row count and disclose displacement. |
| Impossible aggressive drop advice | Safe-mode drop recommendation is gated on measured budget admissibility. |
| Combined repair cost unclear | Common diagnostics expose fills, conflicts, combined cost, original denominator, fraction and fraction admissibility; drops/alignment remain separate. |
| Reordering count appears excessive | Definition states positions moved by sorting, not appended/malformed row count. No silent change to the count's meaning. |
| Capabilities creates ledger | Discovery does not open the ledger. Provider configuration is checked before ledger initialization on other paths too. Provider code can still initialize during configured discovery; resolved-config inspection does not import it. |
| Timezone changes split store identity | CSV timezone declaration is bound transactionally to the dataset; changed declarations reject, including omission after an explicit zone. Legacy/row-ingested datasets cannot acquire a different declared zone retroactively. |
| Backtest executions look pending | Completed study folds are `scored_in_study`, linked to their evidence and excluded from pending production work. |
| Routing fallback exits zero | Compatibility retained; one stderr warning names fallback/cause. `--require-evidence` rejects with exit 2. Mismatches expose study/requested values. |
| Rejected save-result disappears | Rejection JSON is saved to the requested path; an inability to save is named without replacing the original error. |
| Column and limit corrections lose intent | Known ts mapping preserves original CLI argv; other mappings require choices. Search limit correction uses nearest legal boundary. Rejected __default__ is removed from comparison templates. |
| Cache has no observable state | Entries, eligible lookup hits/misses and evictions are available; diagnostic reads do not increment counters. |
| Self-check booleans unauditable | `self-check families` describes fixtures/assertions/limitations; `--detailed` includes expected and actual case evidence. |
| Completion/null ambiguity | Add evidence_status and returned_evidence alongside existing compatibility fields. Query execution, scoring and returned fold summaries are distinct. |
| Large discovery payload | Optional `capabilities --brief` deduplicates provider request schemas across CLI/Python/MCP; `describe --brief` omits the repeated inspection object. Per-command help remains self-contained; use schemas for machine discovery. |
| Provider example escaped | `providers example --raw` prints Python; `--write-to` creates a new source file without overwriting an existing file. |
| Season advice on nonseasonal providers | Advice is emitted for seasonal_naive only; effective period metadata is retained. |
| Python call compact default mismatch | Docstring explicitly describes compact=True and the full-result compact=False option; default is retained for existing callers. |
| Repeated error/rejection | Existing compact-errors opt-in is retained; removing the legacy block by default would break clients. |
| Snapshot exact redeclarations | Existing rejection retained because feedback explicitly disagrees and frozen identity is already clear. |
| Forecast versus fold point shape | Existing shapes retained to avoid breaking saved studies and readers; see result-field mapping in the operations guide. |
| order_events item schema | Already present, including event_id/at required fields; regression coverage checks it. |
| Response byte budget | Documented as compact UTF-8 structured payload, not duplicated MCP text/envelope or pretty CLI whitespace. |
| Provider TOML rejection context | Adds provider and rejected field names without copying secret values into diagnostics. |
| selected_provider tool confusion | Gnomon rejects with a parameter-preserving provider correction and zero provider calls. No silent alias or provider selection. |

The reported Arena `statsforecast_forecast` / `candidate failed` code is absent
from the available checkouts. Its repair-versus-execution budgets, final-answer
scoring and fallback-reliability metrics have not been changed. Those changes
need the actual adapter and a versioned benchmark protocol: changing them would
change experimental results and must not be presented as a forecasting fix.
Existing historical benchmark results remain unchanged.


## Validation

The full source suite passed with **1,085 tests passed and 29 skipped**. Ruff,
compilation and whitespace checks passed. Wheel and source distributions built.
A fresh venv outside the checkout passed 13 installed-wheel command journeys,
including production comparison, revised-vintage rescoring, 2,000-point MCP
paging/hash verification, and 20 detailed cases across all nine self-check
families. These pre-release checks used a local build on top of 1.1.6. Release 1.1.7
checks the tagged commit again. This is not a claim about the unavailable Arena
adapter or a substitute for independent published-wheel acceptance testing.

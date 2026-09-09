# Changes from the 1.1.7 acceptance feedback

CSV ingestion may assume source availability from valid timestamps while still
recording real ingestion timestamps. Evaluation, routing and rescoring now use
recording-time metadata independently of that source assumption. A regression
journey checks the original study, a later CSV revision, early/late rescoring,
and routing at an earlier recording cutoff. Original saved evidence stays
unchanged; source-time assumptions remain disclosed.

Unavailable-fold diagnostics distinguish insufficient observation count from
a missing observation at the origin. Visibility reports include the required
endpoint, last visible timestamp, effective cutoffs and endpoint exclusion
counts. Source and recording exclusions can overlap; do not add their counts
as if they were disjoint rows.

Frozen-snapshot rejection now returns a command that removes only rejected
preparation options. Identical redeclarations remain rejected. Temporal recovery
proposes unambiguous `timestamp`→`value`, integer-string and singular-unit
corrections without replacing dates or numeric amounts. Calendar/elapsed and
DST choices remain unresolved in `next_call`; the example remains separate.
The execution schema stays strict: these aliases are corrections, not silently
accepted alternative fields. Conflicting aliases require a caller decision.

`error.recovery` carries `cause_code`, the authoritative `cause`, supplied and
defaulted arguments, preserved/changed/rejected fields, choices and the proposed
next call. `runnable` on a next call means it can be submitted verbatim without
unresolved placeholders; it does not promise environmental or provider success.
`admissible` describes preservation of the task and resolved semantics. A
syntax illustration can run but is not an admissible recovery. An uncertain
or invalid fact is never replaced just to make a retry succeed.

Capabilities use shared request schemas by default. Use CLI `--expanded`,
Python `capabilities(brief=False)` or MCP `{"brief":false}` for per-provider
schemas. Errors use one canonical `/error` object and a legacy rejection
reference by default. CLI `--expanded-errors` (before the subcommand), Python
`GnomonError.to_dict(compact=False)` and operator `compact_errors=false`
restore the legacy duplicate projection. These are intentional response-shape
changes: clients should resolve `request_schema_ref` and read `/error`.

Execution diagnostics measure deltas for sequential session calls:

- `provider_calls`: dispatch attempts, including provider/factory failures;
  a native batch is one dispatch.
- `forecast_calls`: validated forecast requests, including cache hits; invalid
  requests rejected before dispatch do not increment this count.
- `ledger_writes`: committed row changes through this ledger instance, rather
  than SQL statements or logical operations. An exact score retry writes zero.
- `source_mutations`: Gnomon's session operations do not mutate their source
  observations. This excludes arbitrary provider side effects.

The counters are not isolation guarantees for concurrent calls, other processes
or arbitrary provider internals. Uninstrumented boundaries report null with an
explicit scope, never guessed zero. Saved study evidence is distinct from
current-call diagnostics. Read-page byte limits include the extra metadata;
hashes continue to describe the retained root payload.

Configuration validation names invalid field paths and expected public types or
enum values without echoing rejected secret values. Routing reports all
independent task/provider mismatches together. Provider execution failures use
a safe category, registered provider identity and correlation ID in the local
logger; raw exception text and custom exception class names remain withheld.
The ID is not a `gnomon_read` result reference.

The complete comparison tutorial already existed in
[production-history-comparison.md](production-history-comparison.md). It is now
also installed and advertised in `gnomon ledger --help`:

```bash
# Run in a fresh directory: creates a synthetic local prospective.db.
python -m gnomon.examples.compare_history
```

No forecasting algorithm, unit-matching rule, outcome authorization requirement
or routing fallback exit policy changed. The regression suite includes ledger
immutability/current coverage, store timezone identity, repair budgets, native
MCP paging and custom-provider call-count checks; passing them remains bounded
software evidence, not general forecasting or unattended-business safety.

The subsequent Arena final-answer finding is addressed separately by canonical
forecast completions and `resolve_final_selection`; see
[host integration and resolution rules](final-selection.md). This is a host
helper, not an implicit change to provider execution or the benchmark grader.
The local Arena checkout contains an older episode-grading adapter, not the
reported Favorita/StatsForecast harness. Wiring that harness and replaying its
frozen 96-case evidence remain separate integration work.

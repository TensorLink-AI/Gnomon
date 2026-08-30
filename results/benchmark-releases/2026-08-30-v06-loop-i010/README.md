# v0.6 loop I010: production inference reliability

Decision: **complete / promoted.** Same-ID artifact publication is now safe
under concurrent callers. Every writer constructs and seals a private hidden
tree; the first complete writer publishes atomically, and later writers verify
and reuse that winner without deleting another writer's work.

The frozen baseline reproduced the defect with 0/4 successful same-ID callers.
Two independent candidate runs each achieved 4/4, with forecast and JSON
artifacts byte-identical to their single-writer controls. Injected pre-seal
failures never exposed a final ID, retained writer-private diagnostic trees,
and recovered on the next clean call. Typed MCP failures and all local load
gates also passed; 24/24 replicated public forecasts completed at under 4 ms
p95 with zero external calls or retries.

The final full local suite passed 2,579 tests with 11 skips; 13 focused tests
passed. Raw resumable evidence remains under
`results/v06-p10-i010-reliability-*`.

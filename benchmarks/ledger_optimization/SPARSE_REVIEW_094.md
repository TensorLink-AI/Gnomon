# Sparse review display candidate — offline only

The 093 matched-support audit found 74 of 139 displayed pair/session comparisons
had no shared matured origin. These empty comparisons occupy agent context but
cannot support a numerical choice. This prototype changes only their display;
it does not alter the ledger, comparison cohorts, queries or forecasting tools.

`sparse_review_094.compact_unsupported_pairs` consumes the frozen 087/088 brief.
It retains every card with any supported window verbatim, including losses,
single-origin evidence and ties. Only cards with zero matched origins in all
three windows are compressed into `unsupported_pairs`. Configuration identities,
excluded counts, exact full-evidence pointers and root hash remain available.
Unsupported means no comparison; it never implies zero error or a tie.

The source page's pagination and next-call arguments remain unchanged.
`pagination.shown_pairs` counts supported cards plus represented unsupported
pairs. `display` states these semantics explicitly. This does not claim every
pair in the ledger has been retrieved. If the new explanatory wrapper would be
larger than the original compact UTF-8 JSON, the original response is returned
unchanged. No score, winner or future outcome determines this choice.

## Offline checks and limitations

Six regression tests cover retained losses/single origins, lifetime evidence
despite empty recent windows, reference chains, conflicting counts, unchanged
pagination, immutable inputs and avoiding expansion of small responses.

All 33 current-origin review payloads in the already audited development batches
were replayed. Supported cards, catalog, query, source page navigation, and full
evidence references were checked against the originals. First iteration:
160,867 to 143,611 compact UTF-8 bytes, with nine expanded responses. Its code and
outputs remain in `results/sparse-review-094-offline-001`. The second iteration
avoids expansion: 160,867 to 142,522 bytes, an 11.40% reduction, with zero expanded
responses. Code, outputs and hashes are retained in
`results/sparse-review-094-offline-002`.

This is a payload-size result, not an accuracy, token, billed-cost or completion
result. It is only a fraction of the agent's full context. No new agent/model
call or fit was made, and no final data was opened. All 24 frozen 093 source
files still match the original pilot manifest.

## Before comparative use

This module is not imported by the live 093 runner. Finish and audit that run
and the queued seed integration before preparing another paid trial. Any later
use needs a separate frozen source manifest, integration check proving only the
ledger display differs, and a prospective matched development comparison with
all arms, seeds, outcomes and costs retained. Do not promote it based solely on
the byte reduction or claim the 20% RMSLE objective is satisfied.

Receipt: `evidence/sparse-review-094-offline-001.json`.

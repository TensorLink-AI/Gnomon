# Promotion from paper to live

The user decides promotion. Build the review; never approve it yourself.

1. Agree the minimum (complete paper decisions, and days if wanted) and the criteria
   before paper trading starts.
2. `promotion_review(paper_ledger, source_as_of=..., recorded_as_of=...)` counts paper
   decisions and those with complete outcomes at explicit cutoffs. Add net results
   after costs, forecasts against a simple baseline, and reconciliation health from
   the engine and order journal, and give the review to the user.
3. The user writes the promotion record: `review`, `min_complete_decisions`,
   `criteria`, `approved_by`, `approved_at`. Never set `approved_by`.
4. Live decisions in Python require that record. `verify_promotion` checks its fields,
   a positive integer decision minimum, and a timezone-aware approval time at or after
   the review cutoffs and no later than now; it re-counts completed decisions and IDs
   at those saved cutoffs.
5. It does **not** verify performance criteria, approver identity, a minimum
   expressed in days, costs, fills or later revisions; review those separately.
6. Over MCP, the live server's `decision_context.trading_mode.source_ref` only names
   the record; it does not load or validate it. The operator or execution layer must
   verify approval before enabling live orders.

Go live small, with the user's size cap and a stated condition for returning to paper.

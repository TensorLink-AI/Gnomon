# Development RMSLE presentation on the same ledger evidence

This prospective treatment extends the shared pair catalog, global-origin windows,
pagination and incumbent MAE adapter in ML_CARDS_024.md. No agent comparison has
been run with it. It changes evidence presentation, not the available models,
outcome visibility, fitted predictions, final selection rules or grading metric.

Use the existing development branch's src/gnomon/evidence_summary.py, with SHA-256
1cd7adfc4a6180e000b32af7120ee77bc735b3df0bb1cb633f767302676f35ec,
as a separately pinned development helper. Do not claim it is part of the
published 1.2.0 wheel. The eventual runner must record its hash alongside the
installed execution/ledger build. Compatibility checks run this same helper with
both 1.1.9 and 1.2.0; the eventual incumbent agent still receives its MAE view.

For every public matched-origin record, read the referenced typed execution and
the exact actual IDs visible at the stated source/recording cutoff. Recheck
provider/revision, series/unit, origin, target timestamps and ex-ante recording.
Refuse missing, conflicting, incomplete or future evidence. Compute per-origin
RMSLE through the frozen helper; mean those origin scores, preserving the task's
arithmetic mean-of-case-RMSLE objective. Clip negative predictions only under the
common declared rule and retain clipping counts. Negative actuals reject.

Keep the original public MAE, matched counts and evidence references. Add RMSLE,
calculated ranks, ties and pairwise differences within each pair/window. Derive
recent/lifetime disagreement from the same global last-four window used by the
incumbent, not a new last-four-successful-matches definition. Missing recent
support yields null disagreement rather than a claim of agreement. Never turn
pair ranks on different cohorts into a global winner or statistical claim.

Queries start zero providers and do not mutate saved scores or observations.
Full cards precede a compact view with resolvable references. Complete raw evidence
remains available to all arms; the development treatment reduces calculation and
interpretation work rather than adding private observations or stronger models.

Before inference, integrate and freeze the four-arm runner, service-admission
policy, exact runtimes and continuous-origin tasks; pass equivalent numerical,
visibility, completion and accounting checks. The completed synthetic checks
are prerequisites, not a measured accuracy improvement or the final objective.

# Development screen 011: context of the current CV choice

Registered 2026-09-12 before computing this screen. The objective remains 20%
lower mean case RMSLE in a fair, untouched three-arm agent confirmation. This
offline development screen cannot satisfy that objective. Confirmation 010 is
spent and negative; neither its cases nor the workflow-validation worlds are
inputs to this screen.

Hypothesis: provider performance conditioned on the provider favored by current
cross-validation is more relevant than unconditioned lifetime history. CV leader
is observed at the decision origin; it is not a label derived from later success.

Use only the 208 cases in `new-development-input-v2/cases.json`, hash
`5ee5a6dd4c346882287ef199f7640fa553a1f50062f73e9859ba33083f3a0c3f`.
Keep all eight existing providers, their predictions, and the original metric.
No calibration, ensembles, new candidate fitting, API calls or forecast edits.

Four fixed variants: minimum matching origins 4 or 8, and minimum paired win
fraction 0.5 or 0.75. Retrieve same-series past episodes with the same CV leader,
strictly earlier forecast origins, outcome recording no later than now, and all
target valid dates no later than now. Source availability at valid time and
recording at horizon close are inherited replay assumptions, not measured vintages.
Switch only to a provider with lower matched mean RMSLE and the required win
fraction against today's CV leader. Otherwise retain the CV leader. Ties follow
sorted provider names, consistently with the existing support screen.

Compare against current CV and the previously frozen unconditioned support rule
(minimum four origins, half paired wins). Choose a variant only by mean RMSLE on
development origins 0–17, then report origins 18–25 and all origins. These slices
have already been used in development: neither is a fresh holdout. Historical
outcomes from earlier validation origins may mature during sequential replay,
but the variant cannot change after training selection.

Every choice must retain its retrieved origins, paired counts, predicted provider
identity and independently recomputed loss. Test future-outcome invariance,
recording/source visibility boundaries and input-order invariance. Hash the input
and script and reject reruns to an existing output file. Keep unfavorable results.

If this merits a live trial, provide identical raw episode records and their
historical CV-leader labels to all arms. The treatment may organize and summarize
them. A positive screen alone never opens another confirmation or establishes
superiority; a new untouched cohort and explicit freeze would still be required.

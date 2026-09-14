# Source experiment081: per-step conditional model-risk weights

Execute frozen080recipe with416scored/125available warm-up tasks, identical six
fixed066forecasts per arm, unchanged045anchors and050/061/068guards. Same-domain
historical records qualify solely by strict earlier origin and closed target,
source/local recording availability<=current origin. Keep nominal period-end
assumption explicit and filter metadata before outcome payload reads.

Control forests use three current CV pairs at1/3case masses. Ledger forests
use half current CV and half all eligible production pairs, equal within groups.
Each case contributes24equally weighted forecast-step rows. Use080predict_matrices
and unchanged051quadratic solver separately at each of24current leads. Expand
080's combine loop only to record every solve attempt/completion and preserve
partial certificates if a later lead fails. Same numerical inputs/solver/options.
No matrix smoothing, hyperparameter changes or post-result retries in this run.

Retain every forest tree structure and36-output node values, case/mass/source
references, predicted24Gram matrices,24simplex certificates and output forecasts.
Save independent arm predictions before accessing current actuals for scoring.
All raw data/fixed config/CVfold identities and phase alignment are hash-verified.
Query features use predictions only; actuals are training labels, never query
features. Shared numerical maximum832forests,26,624trees,19,968quadratic solves.
Report started/completed counts, iterations, wall/CPU and all failures. Preserve
inherited49,616base computations,416045anchors,832068blend fits,11,902search
surrogate solves and31,378logical search attempts per arm.0new base/API calls.

Gate unchanged080:20%over matched control ANDstrong050; positive over061AND068,
all four comparisons positive in both domains. Retain uncorrected045and phase
results. No paid/protected confirmation after failed gate, main/PyPI unchanged.
This remains repeated-development numerical evidence. The final objective still
requires a frozen matched1.2.0/DeepSeekv4.1-flash agent evaluation on untouched
cases and95%uncertainty excluding zero. A source numerical pass would not satisfy
that final requirement by itself.

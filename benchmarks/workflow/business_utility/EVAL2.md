# Eval 2 registration: repeated questions and silent repair changes

Inherits COMMON.md. Hypothesis: Gnomon reduces unauditable or materially inconsistent
business handoffs; it does not force identical autonomous workflow choices.

128 cases/arm: the first two shared windows in each of four domains (8 files), two
task strata, eight fresh-session identical repeats per file/stratum. Repeated case
IDs must differ for journaling; strip that identifier from model-visible input in
the common driver so it is not a differing prompt. Repeats share all other bytes.

Messy exports: even-index windows reverse rows and add one identical duplicate;
odd-index windows remove interior row 8 and add a conflicting row 12 with value
increased by 10% of the training scale. Include an alternate numeric column labelled
auxiliary. All arms receive the same file and clear units/business-target meaning.
No unparseable garbage or hidden instruction text.

Fixed-request stratum: time=timestamp, target=value, historical_mean provider,
horizon=1, repair=aggressive, no arbitrary model changes. Ordinary may implement
the fully supplied repair rules: sort; collapse identical duplicates; arithmetic
mean of conflicting values; linearly fill the single interior gap. Agent reports
headline forecast, model, mapping, normalized repaired rows, and repair disclosure.
Verify that this policy matches the pinned release before launch; otherwise amend
before any live result. Optional execution IDs/snapshot IDs are audited if supplied.

Autonomous stratum: same forecast question; agent chooses among last_value,
historical_mean and seasonal_naive (season=4), mapping and repairs. It must disclose
choices. This tests agent behavior, not a structural reproducibility guarantee.

Primary numerical quantity: within-file/stratum headline population SD divided by
training scale; also min/max and missing count. No SD for <2 numeric answers.
Material instability: any pair differs by >0.01 scale. Count inconsistent groups
out of eight and affected runs out of 64 per stratum. Never pool repeats as IID.

Primary business binary: unauditable handoff (missing/invalid repaired rows, missing
model/mapping/disclosure or no answer) OR deviation >0.01 scale from the fixed-policy
reference, in the fixed stratum. Autonomous stratum is descriptive. Silent repair
change is a different normalized-row digest within a group without disclosure of
the changed operation in the final answer. Also count a repair omitted from the
answer even if repeated consistently. Agent-declared rows are not independently
attested execution inputs; absent execution evidence is explicitly unaudited.

Success: corrected significant reduction of fixed-stratum primary composite and
no increase in failures; report SD and autonomous results regardless. Eight clusters
give weak uncertainty and limited power; no zero-variance claim if any run failed.
Stop only after all 128/arm or infrastructure stop with full planned denominator.

Limitations: Gnomon records repairs in tool output, but an agent may omit them from
its answer. Different model choices are legitimate if disclosed. Deterministic
snapshots do not fix stochastic providers, numerical libraries, agent behavior or
remote model weights. Exact repeated snapshot/provider property tests are separate
from measured agent trials. An eight-file experiment cannot establish compliance.

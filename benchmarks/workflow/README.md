# Gnomon Workflow Bench

Workflow Bench evaluates the product claim **one trusted temporal answer,
cheaply**. It complements rather than replaces TemporalBench (task accuracy)
and Leaktrap (bitemporal safety).

Workflow cases send only cutoff-safe input, so their zero-leak result means
the arm was isolated from future data; it is not an adversarial leakage test.
A design decision is valid only alongside a zero-leak run of the separate
LeakTrap suite, where post-cutoff rows and late revisions are deliberately
present and tempting to use.

It reports four independent dimensions:

- correctness: numeric/choice oracle checks, with missing and abstained cases scored as zero unless abstention is expected;
- trust: a strict all-components pass plus a diagnostic component mean for
  leakage, disclosures, forbidden claims, support, artifact candidate parity,
  and quote fidelity;
- usability: answer yield, correct repair, artifact-bound outcome tracking,
  and quote fidelity;
- economics: cumulative/response tokens, calls, and latency, both end-to-end
  and split into initial, repair, and outcome stages.

Staged reporting does not call a correct abstention an unanswered workflow.
`initial_answer_yield` describes the first response;
`final_workflow_resolution_rate` describes whether the complete workflow ended
in the required answer or governed tracking result. Decision gates use the
latter. Calls are likewise split: the single-answer target applies to
`initial_calls_median` (≤2), while a repaired workflow may use a second stage
and must remain within four total median tool calls.

Trust is reported two ways: coverage-adjusted strict trust over every case and
strict trust among cases whose execution completed, beside
`trust_measurement_coverage`. Timeouts are unmeasured, never relabelled as
leaks. `capability_coverage` separately reports whether an arm actually
produced immutable artifact identity and whether tracking-required cases had
that capability; absence is visible as a product limitation rather than an
arithmetic mistake.

There is deliberately no weighted aggregate. Leakage is a binding release gate;
the remaining measures form a scorecard so a cheaper but less accurate arm
cannot hide that tradeoff.

## Arm protocol

The runner sends one JSON case per process on stdin. The oracle is never sent.
The command returns one JSON observation on stdout:

```json
{"case_id":"synthetic-seasonal-001","status":"answered","support":"supported","numbers":{"next":10},"choices":{"pattern":"period-4"},"disclosures":["synthetic fixture"],"claims":[],"temporal_leakage":false,"publish_matches_evaluated":true,"tool_calls":1,"cumulative_tokens":900,"response_tokens":120,"latency_seconds":0.4}
```

Run an arm directly:

```bash
python -m benchmarks.workflow.run_workflow \
  --arm-command "python my_adapter.py" --arm core \
  --output-dir results/workflow/core --infrastructure-retries 2
```

Infrastructure retries are bounded and recorded per observation. Add
`--resume` to retain successful observations already in the output directory
and rerun only missing/error case IDs. Exit `0` means execution and release
gates passed; exit `2` means execution completed but a scored gate failed;
other non-zero exits indicate a harness failure. The suite orchestrator treats
exit `2` as a completed run and continues the matrix.

Or score observations captured by another harness:

```bash
python -m benchmarks.workflow.run_workflow \
  --submission results/core.jsonl --arm core \
  --output-dir results/workflow/core
```

The initial five-case corpus is a fast contract smoke test, not publishable
evidence. Expand it with frozen, versioned cases across domains and report
confidence intervals before drawing product conclusions. Never place outcome
or post-cutoff data in `available_at_cutoff`; longitudinal outcomes are
revealed by the adapter only during its tracking stage.

Corpus readiness is machine-checked rather than implied by a green run:

```bash
python -m benchmarks.workflow.audit \
  --cases benchmarks/workflow/cases/smoke.jsonl --profile smoke
python -m benchmarks.workflow.audit \
  --cases path/to/frozen-v1.jsonl --profile publication
```

The publication profile requires at least 100 cases, five domains, ten cases
of every workflow kind, required risk/workflow tags, at least 40 distinct
oracles, typed (not substring) disclosures, sealed repair/outcome stages,
explicit oracles, and publish parity on forecast cases. The bundled corpus
passes `smoke` and is expected to fail `publication`.

Generate the deterministic decision corpus (20 cases per kind, balanced over
five domains, with multiple deterministic temporal mechanisms inside every
domain) and audit it:

```bash
python -m benchmarks.workflow.generate \
  --output results/workflow-publication-v1.jsonl
python -m benchmarks.workflow.audit \
  --cases results/workflow-publication-v1.jsonl --profile publication
```

For an actual decision run, generate a fresh held-out corpus instead of reusing
the development seed:

```bash
python -m benchmarks.workflow.generate \
  --fresh --output results/workflow-heldout.jsonl
```

`--fresh` draws a new 63-bit seed and writes it, the generator version, and the
corpus hash to `workflow-heldout.jsonl.manifest.json`. Supplying that recorded
seed with `--seed` reproduces the cases exactly. Period, phase, amplitude,
history length, slope, seasonal values, ambiguous target, and triage winner are
varied. The publication audit rejects insufficient input, repair-target,
triage-winner, mechanism, or within-domain diversity. Keep the seed and hidden
oracle file away from evaluated agents until all submissions are frozen.

The generator's seed, full hidden-stage corpus, and SHA-256 hash make the set
reproducible. Repair instructions and realized outcomes are withheld from the
initial arm call. The runner invokes a second call only after collecting the
initial answer. A repair passes only when the second answer satisfies its
numeric/choice oracle and names the resolved target. Tracking passes only when
the reported error equals the runner's independently computed error and the
follow-up names the original immutable forecast identity.

Artifact candidate parity compares the evaluation identity recorded in
`final_candidate` evidence with the model identity in the published artifact;
for built-ins, the artifact's selected-model contract is the available
identity boundary. It intentionally does **not** claim to re-evaluate ensemble
members or weights from an agent transcript. Full strategy/member/weight
publication equality is a mandatory engine invariant covered by
`tests/test_executable_candidate.py`. Quotation is independently checked by
comparing submitted headline numbers with artifact numbers. Model
self-attestation cannot satisfy either scored check.

## Production comparison policy

Every run records the SHA-256 identity of its full case bundle (including the
hidden oracle), saves replayable normalized observations, and writes results
atomically. Adapter failures and timeouts become scored error observations.
Use `--jobs N` for parallel case execution.

Compare only runs carrying the same corpus hash:

```bash
python -m benchmarks.report --root results/temporalbench \
  --compare control core --metric SMAPE --json \
  > results/temporalbench/core-vs-control.json
```

```bash
python -m benchmarks.workflow.compare \
  --baseline raw-llm results/workflow/raw results/workflow/core \
  results/workflow/evidence \
  --leaktrap-summary results/leaktrap/gnomon/summary.json \
  --heldout-manifest results/workflow-heldout.jsonl.manifest.json \
  --accuracy-comparison core=results/temporalbench/core-vs-control.json
```

An arm is eligible only when provider execution is complete, the measured
workflow is leak-free, yields an answer on
at least 80% of cases, uses fewer than 50K mean cumulative tokens, takes at
most two median calls, and its paired-bootstrap 95% lower accuracy bound is no
more than two percentage points below the baseline. These are the plan's
pre-committed surface objectives, not a claim that the smoke corpus has enough
power to certify them.

Completeness and leakage are deliberately separate gates: a provider timeout
is an infrastructure failure, not evidence of a temporal leak. Reports expose
infrastructure and task-error rates, numeric, canonical-semantic, accepted-
semantic, and alias-only correctness separately, every strict-trust component,
and pooled median/p95 call counts. Calls are split into the surface minimum,
agent-observed calls, and redundant calls. Engine contract completeness and
agent fact preservation are separate measures. Artifact flow reports explicit
counts for required cases, identity produced, identity matched, and identity
preserved by the agent.
Choice oracles may declare canonical aliases (for example `period-7` and
`weekly` on a daily grid) so presentation wording
does not masquerade as a numerical failure.
The comparison distinguishes surface eligibility from decision readiness.
`decision_ready_arms` remains empty unless all four evidence legs are present:

1. Workflow Bench passes for routing, correct repair, artifact-bound tracking,
   trust, and economics.
2. The supplied Gnomon LeakTrap result contains at least 40 tasks, zero
   leaks/transcriptions, and a structural cutoff proof for every task.
3. A matched TemporalBench/report comparison contains at least 20 shared tasks
   and the treatment is non-inferior on a recognized forecast or choice metric.
4. Baseline and treatment each contain at least three stochastic replicates.

Decision readiness also requires a matching `--heldout-manifest` created with
`--fresh`. A development/reused seed or a manifest whose corpus hash differs
from the scored runs is rejected. Generate this final corpus only after the
harness is frozen, keep its cases/oracles out of prompts and review, and do not
tune against its results; any harness change retires that decision corpus and
requires a newly generated seed.

Pass multiple run directories with the same `arm` to aggregate stochastic
replicates. Replicates are averaged per case before the paired bootstrap, so
easy cases and repeated samples cannot change the case weighting. For a design
decision, use a fresh held-out generated corpus and report the five workflow
strata separately alongside the overall gate. Generated cases test procedural
generalization and agent behavior; TemporalBench/frozen external data remains
the accuracy leg because synthetic domain labels are not evidence of real-world
forecast quality.

## Trading corpus

The same runner, scorer, and gates also drive a trading-specific corpus
(`benchmarks/workflow/trading.py`). It is a corpus and an audit profile, not a
fork: no scoring rule is different, so a trading arm and a general arm are read
off the same scorecard.

What trading adds is the set of failure modes a generic series cannot express,
and every one of them is a calendar or basis question rather than an alpha
question:

- **the next observation is the next *session*.** Cases carry an exchange
  calendar (`xnas`, `cme`, `fx5`, `sifma`, `24x7`) with holidays inside the
  visible window, and the forecast must name that next session date. Adding a
  day to the cutoff is wrong on four of the five calendars.
- **a number without its basis is not an answer.** Every answerable case
  requires the published `price_basis` fact — unadjusted last trade, exchange
  settlement, 4pm mid, venue last trade, constant-maturity yield — so a
  correct number published on an unstated basis fails trust while still
  scoring correct. The two failures stay visible separately.
- **history that is not comparable with itself.** The `messy` kind rotates
  three market-data ambiguities: venue dispersion (two venue streams, neither
  named), an unapplied corporate action (a 2:1 split on the tape but not in
  the history), and a trading halt whose final print is stale. Each declares
  its evidence in the payload (`venue_streams`, `corporate_actions`, `halts`);
  the governed behaviour is to abstain and then complete once the resolution
  is revealed.
- **return, not level.** The `multiseries` triage winner is the instrument
  with the largest absolute final-session return in basis points, and the
  instrument with the largest *price* move is deliberately a different one.
  The corpus audit fails the bundle if that trap ever collapses.
- **no answer may be dressed as advice.** Every case forbids the guarantee and
  recommendation vocabulary (`guaranteed`, `risk-free`, `cannot lose`,
  `no downside`, `financial advice`, `sure thing`, `beat the market`) in any
  support tier.

Generate, audit, and run it:

```bash
python -m benchmarks.workflow.trading --output results/workflow-trading-v1.jsonl
python -m benchmarks.workflow.audit \
  --cases results/workflow-trading-v1.jsonl --profile trading
python -m benchmarks.workflow.run_workflow \
  --cases results/workflow-trading-v1.jsonl \
  --arm-command "python my_adapter.py" --arm my-arm \
  --output-dir results/workflow-trading/my-arm
```

`--fresh` draws and records a new seed exactly as the general generator does;
use it for any decision run rather than reusing the development seed. The
bundled `cases/trading_smoke.jsonl` is five cases across five asset classes: it
passes `--profile trading-smoke`, fails `--profile trading`, and exists to keep
the contract honest in CI, not to support a claim.

Beyond the generic readiness checks, the `trading` profile audits the corpus
itself: no input bar may be dated after its own cutoff, every revealed outcome
must fall strictly after it, the calendar and basis must be declared, all three
ambiguity kinds must appear, sessions must actually be non-contiguous
somewhere, and the triage winner must never be the largest mover.

`benchmarks/workflow/trading_baseline.py` is a model-free control arm — carry
the last print forward, roll the declared calendar, rank by return, abstain
when the tape says the history is not comparable. It answers only from evidence
each case ships, which is how the corpus proves it is answerable without
private generator knowledge. On the 100-case corpus it scores 0.64 correctness
(synthetic 0.00, frozen 0.60, longitudinal 0.60, messy 1.00, multiseries 1.00)
at 0.80 strict trust, with the trust misses concentrated in `support`: it
declines to claim a session cycle it never identified. Treat those numbers as
the floor an arm has to beat, and note that the `messy` score measures
disposition, not difficulty — the ambiguity is declared in the payload, so
what is being scored is whether the arm abstains rather than whether it can
detect a split.

What this corpus does **not** claim: it contains no market data, no vendor
feed, and no strategy. It measures whether an agent can be trusted with a
temporal market question — calendar, basis, comparability, and refusal — not
whether anything here would make money.

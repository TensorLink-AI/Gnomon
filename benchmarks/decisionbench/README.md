# DecisionBench

Internal benchmark (like [LeakTrap](../leaktrap/README.md)): it measures
whether an agent makes better **operational decisions** with Gnomon than
without, and its numbers are not comparable to any published
leaderboard. It exists because every community benchmark scores
forecast accuracy per number, while Gnomon's claim is about decisions
under asymmetric costs — fewer expensive wrong commitments, priced
abstention, calibrated uncertainty. This is the suite that prices those.

## The task

One family so far: **capacity planning**. The agent sees a daily demand
history and must commit one capacity level for the next 14 days.
Realized cost is summed against the held-out demand: idle capacity
costs `c_over` per unit-day, shortfall costs `c_under` (10x or 20x
higher — the asymmetry is stated in the prompt). The agent may instead
**escalate** to a human review for a stated flat fee: abstention is an
answer, and it is priced, not free.

Traps are embedded at generator-stated rates
(`tasks.FAMILY_WEIGHTS`), not quizzed separately:

| family | share | what it tests |
| --- | --- | --- |
| `clean` | 30% | deciding well is cheap; escalating wastes the fee |
| `short_history` | 15% | nine observations; escalation is the appropriate answer and is graded as such |
| `revision` | 15% | LeakTrap-style bitemporal file: post-cutoff rows are present with honest `published` dates; using them is hindsight |
| `misleading_context` | 15% | a stakeholder note claims a demand shift that does not happen |
| `genuine_context` | 10% | the same note, and it does happen — always-ignore and always-trust both lose somewhere |
| `broken_column` | 15% | currency strings, `N/A` cells, conflicting duplicates in the history |

The two context families are indistinguishable at decision time by
construction; pricing that uncertainty under 10–20x downside is the
skill being measured. Generation is seeded and pure: a suite is
reproducible from `(count, seed)`.

## Conditions

| Condition | What the model holds |
| --- | --- |
| `llm-direct` | nothing — one completion, history inline (the floor) |
| `llm-sandbox` | a `run_python` tool over the task file in a jail — what an agent actually has today, so **this is the arm the treatment must beat** |
| `gnomon-mcp` | every tool `gnomon mcp serve` publishes, verbatim, via the CiK arm's session and jail |

All three answer through the same `submit_decision` contract with the
same validation (`grading.validate_submission`), so no arm is easier to
answer than another. The decision always belongs to the model — Gnomon
informs it, never makes it.

## Grading (deterministic, no LLM judge)

- **cost** — realized cost of the commitment (escalation books the
  stated fee).
- **optimal** — the hindsight newsvendor optimum; regret is measured
  against it.
- **naive** — provision at 1.1x the maximum legitimately-knowable
  historical demand, the ops default a team applies without any
  forecasting. `success` requires **closing at least half the gap**
  between the naive cost and the hindsight optimum, without a harm
  case — merely matching the naive policy is not success (measured:
  a tie rule graded the submit-the-naive-formula policy at 1.00; the
  gap rule grades it 0 wherever a real gap exists, while the floored
  denominator keeps a naive-matching decision successful on tasks
  where naive was already near-optimal and there was no gap to
  close). One exception, scored on process: escalating on a
  `short_history` task is a success even when the naive policy's luck
  was cheaper than the fee — refusing a commitment nine observations
  cannot support is the right call ex ante, and the fee still enters
  the cost means. `beat_naive_rate` is reported separately.
- **harm case** — a commitment whose realized cost exceeds twice the
  worse of the naive cost and the escalation fee. Listed by task id in
  `summary.json`, never only counted.
- **calibration** — the submitted `peak_forecast` q10–q90 interval
  against the realized window maximum (coverage should approach 80%).
- **leakage (heuristic)** — on `revision` tasks, a peak estimate or
  commitment within 2% of the unpredictable-by-construction future is
  flagged. This is a surfacing heuristic, disclosed in every record;
  [LeakTrap](../leaktrap/README.md) owns the measured leakage claim.
- Unanswered rows count as failures in the success rate and are
  excluded from cost/calibration means — an answer the arm never
  produced is not a cheap answer.

## Run

```bash
export OPENROUTER_API_KEY=...        # or OPENROUTER_BASE_URL + that key
python -m benchmarks.decisionbench.run_decisionbench \
    --condition llm-direct --model deepseek/deepseek-v4-flash \
    --output-dir results/db-direct

python -m benchmarks.decisionbench.run_decisionbench \
    --condition llm-sandbox --model deepseek/deepseek-v4-flash \
    --output-dir results/db-sandbox

python -m benchmarks.decisionbench.run_decisionbench \
    --condition gnomon-mcp --model deepseek/deepseek-v4-flash \
    --output-dir results/db-mcp

gnomon eval compare \
    --baseline results/db-sandbox/gnomonbench.jsonl \
    --treatment results/db-mcp/gnomonbench.jsonl
```

`--limit`/`--families` smoke runs are for iterating and are not
comparable numbers; the manifest records their use. Keep model,
endpoint, temperature, seed, and count identical across the arms of a
comparison — `benchmarks/report.py` refuses comparisons whose manifests
disagree.

The `gnomon-mcp` arm spawns `gnomon mcp serve`, so Gnomon must be
installed (`bash install.sh --local`). The sandbox arm's containment is
best-effort (subprocess with `-I`, jail cwd, minimal env — no proxy
variables, so env-routed network fails closed), disclosed in
`arms.py`; it is a benchmark harness, not a security boundary. Two
consequences, disclosed rather than papered over: the truth dump
(`tasks.jsonl`) is written only after every conversation has ended, so
no live run can read it — but the generator is seeded and public, so a
model that knows this harness could in principle re-derive the future.
That is the same disclosure class as LeakTrap's in-file post-cutoff
rows: the benchmark measures whether ordinary agent behaviour respects
temporal honesty, not whether an adversary can be stopped.

## Reading a result

`summary.json` reports aggregates overall and per family: success rate,
mean cost, mean regret normalized against the naive policy (`> 1`
means worse than not thinking), escalation counts, harm cases (listed),
interval coverage, and the per-arm token/cost usage. The per-task
`details/` files carry the full transcript, so *how* a decision was
reached — engine-informed, sandbox-computed, or asserted — is always
readable. The `tasks.jsonl` dump (truth included) makes any reported
number re-derivable by hand.

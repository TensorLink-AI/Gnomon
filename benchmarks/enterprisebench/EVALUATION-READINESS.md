# EnterpriseBench evaluation-readiness record

The pre-use rubric, applied. Every automated gate was executed against
this revision; probe numbers below were produced by the scripted
offline pipeline (`seed 11, 24 cases/domain` unless stated) and are
diagnostic by definition. Human gates are marked **PENDING** with the
artifact each requires — the suite must not run a paid, citable
evaluation until the blocking pending items are signed.

## Gate status

| # | Gate | Status | Evidence |
| --- | --- | --- | --- |
| 1a | Endpoint is decision cost/regret in stated units | **PASS** | harness design; `summary.cost_model.units` per domain |
| 1b | Cost model signed off by a P&L owner | **PENDING (blocking)** | ratios are defensible defaults (overage 5× intervention, imbalance 8/2, stockout 9× holding); a named owner must confirm or replace them |
| 1c | Headroom between best free reference and hindsight | **PASS with note** | positive everywhere; **thin in cloudcost** (best-free regret 0.375/case at pilot — the engine nearly saturates; treatment value there shows mainly through text-only facts, traps, and extraction) |
| 2a | Fresh data per seed, nothing static to game | **PASS** | `test_every_new_seed_is_an_entirely_new_corpus` (no shared 12-step window, no shared memo text) |
| 2b | Constant policies cannot win | **PASS** | constant regrets ≫ 0 in every domain (e.g. cloudcost always/never-act 2.0/4.0 per case; demand order-zero 3512/case); base rates within +0.08…+0.22 of break-even, disclosed |
| 2c | Red-team probe detected | **PASS with note** | copied-answer backtest caught by over-promise (`test_candidate_over_promise…`); an out-of-band omniscient client is *not* auto-flagged — its tell is regret ≈ 0 with trap accuracy ≈ 1.0, a human red flag, noted as a candidate future lint |
| 3a | Realism measured against stylized facts | **PASS** | the six realism tests (overdispersion, duck curve, payroll sawtooth, macro persistence, business-hours share, fat tails) in CI |
| 3b | Face-validity read by real domain owners | **PENDING (blocking)** | ~10 sampled prompts per domain, one hour, "this is my Tuesday" or flags |
| 3c | Limits documented and accepted | **PASS / acceptance pending** | README "Limitations and scale"; adopters must accept synthetic corpus + ex-post threshold placement |
| 4a | Pre-registration before the paid run | **PENDING (blocking)** | template below; fill and commit before spending |
| 4b | Pilot power check at planned N | **DONE — read the numbers** | binary domains tie heavily vs engine (13–19 of 24 pairs); informative-pair fraction ≈ 0.2–0.45 ⇒ at N=120 the sign test detects only large asymmetries (~70/30). Guidance: **N≥180–240 for binary domains**, or judge primarily on the bootstrap CI of mean cost delta (uses all pairs); N=120 is adequate for demand/energy (ties ≈ 0) |
| 4c | Reproducibility | **PASS** | `test_identical_runs_are_bit_identical_end_to_end` (stable across repeated invocations) |
| 5 | Operational robustness | **PASS** | resume identity rejection, crash-truncated line handling, malformed-answer degradation, per-domain failure isolation, usage deltas — all tested |
| 6 | Fresh-eyes read-back of the outputs | **PENDING (advisory)** | hand `summary.json` + README to a non-project reader; they must state the verdict correctly |
| 7a | Oracle probe ≈ hindsight optimum | **PASS** | mean regret exactly 0.0 in all six domains through the real parse-and-score path |
| 7b | Garbage probe lands in the constant envelope | **PASS** | unparseable answers price at the recorded no-action default, within the constant-policy envelope everywhere |
| 7c | Seed stability of verdict-relevant orderings | **PASS with note** | the best-constant identity is sign-stable across seeds 11/21/31 in all domains; closely matched reference *pairs* (naive vs a constant) flip at N=24 — expected sampling noise, resolved by 4b's N guidance |
| 8 | Governance: frozen seeds, spent-corpus owner | **PARTIAL** | `scope` stamping is mechanical (PASS); a named owner for "this seed is now spent" is **PENDING** |

## Findings the probes surfaced (report, do not hide)

1. **cloudcost headroom is thin**: the engine's governed rule is already
   near-optimal on the pilot mix. A treatment arm can look "no better
   than engine" there while still being valuable (text-only deploy
   facts, trap resolution, extraction) — read cloudcost verdicts with
   the trap and extraction metrics, not the cost delta alone.
2. **cashflow's engine mapping lost to always-act at pilot N** (engine
   regret 2.88 vs best-constant 1.88 per case, seed 11, N=24). That is
   a product finding the references exist to expose, but it means
   `vs_engine_alone` is an easier bar in cashflow than elsewhere —
   `vs_best_constant_policy` is the binding comparison there.
3. **Binary-domain tie rates make the sign test blunt at N=120.** Use
   the bootstrap CI as the primary evidence and raise N per 4b.

## Pre-registration template (fill and commit before the paid run)

```
run_id:                enterprisebench-<date>
primary_endpoint:      per-case decision cost (fixed; do not edit)
treatment_arm:         model_facts_compiled (fixed; do not edit)
model under test:      <model id + endpoint>
domains:               <all | subset, with reason>
cases per domain:      <N, chosen from gate 4b guidance>
seed:                  <unused seed from the frozen 9xxxxxxx range>
verdict rule:          useful = all_three_cheaper per domain (fixed)
evidence rule:         bootstrap CI of mean delta is primary;
                       sign test reported alongside with ties
cost model sign-off:   <name, date>   (gate 1b)
face validity sign-off:<names, date>  (gate 3b)
spent-corpus owner:    <name>         (gate 8)
```

## Verdict

Machine-checkable gates: **all pass**, with three disclosed findings.
Blocking human gates before first citable use: **1b cost sign-off,
3b face validity, 4a pre-registration (with 4b's N guidance applied),
8 spent-corpus owner.** Until those are signed, every number this suite
produces is `scope: diagnostic` — which the harness stamps mechanically
anyway.

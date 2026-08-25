# EnterpriseBench

One shared harness, N domain packs. Each pack is a mechanistic simulator
plus a cost model plus a context schema for a real enterprise
forecasting-and-decision job. The suite answers, per domain and per
model: **does Gnomon's governed layer reduce business loss** versus the
model alone, versus the engine alone, and versus policies that ignore
the data — under leakage-proof, point-in-time-correct evaluation.

```bash
python -m benchmarks.enterprisebench.run_enterprisebench \
    --domain all --model <model> --cases 120 --output-dir results/eb
```

## The bitemporal contract (enforced mechanically)

Every context item carries `{item_id, kind, value, known_at,
effective_from, effective_to, revises}`. Revision chains are
first-class: simulators emit early noisy versions and later corrections.
A case at cutoff T exposes only items with `known_at <= T`, **in the
version known at T** — the as-of resolver lives in the harness, is
tested once, and is used by every pack and every arm. A leakage lint
over the serialized prompts fails the run before any API spend if a
post-cutoff observation, post-cutoff revision value, or future outcome
appears in any arm.

**Trap cases** (~15% per pack, disclosed in provenance): a fact is
revised before the cutoff and the correction flips the correct
decision, by construction. Trap accuracy is scored separately per arm,
so an information-boundary violation is a measured quantity, not just a
lint.

## Arms (matched, temperature 0, treatment block is the only difference)

1. `model` — series + context + costs + question.
2. `engine` — Gnomon alone, no LLM: real production
   `forecast(threshold=…)` plus the governed breach-policy ladder
   (`apply_breach_policy`) mapped to the domain's decision. A
   deterministic reference at zero API cost.
3. `model_facts_oracle` — the model plus the engine outputs computed
   from structured context. Isolates governance value assuming perfect
   extraction.
4. `governed_candidate` — the model's forecast admitted as a candidate
   inside the engine's contract: backtested on inner folds against
   seasonal-naive under the same as-of snapshots, published as primary
   only if it wins, engine fallback otherwise, admission labelled per
   row, and post-admission out-of-sample error compared to the backtest
   promise (over-promise detection).

## Scoring & verdicts

Primary: mean decision **cost and regret in the domain's stated units**.
Secondary: MASE (affine-invariant, verified pre-flight), pinball loss on
the engine quantiles, trap accuracy, timing error with disclosed answer
rates, invalid rate (call-quality metrics over valid answers only).
References at zero API cost: engine, seasonal naive, last value, the
pack's constant policies, hindsight optimum.

Per-domain verdicts carry paired exact sign tests with disclosed pair
counts: `vs_model_alone`, `vs_engine_alone`, `vs_best_constant_policy`
("useful" requires all three positive), `candidate_admission_value`, and
`trap_integrity`. The cross-domain rollup is a per-domain verdict table
that **refuses to publish a single aggregate number** — the units
differ.

## Discipline

- Corpora are spent the moment the product iterates against them. Seeds
  `9xxxxxxx` are frozen for validation and must never be used during
  development. All pre-validation numbers are diagnostic.
- The graduation bar for any treatment arm is non-inferiority to
  `model` on interpretation-flavored outputs and superiority on cost —
  never "beat the other arms".
- Rows are stamped with the full dataset identity (generator version,
  domain, pack version, seed, case count, simulator-config sha256) and
  the answering model; `--resume` rejects mismatches and
  crash-truncated lines, disclosed. One failed API call is recorded,
  the rest finish, and the run fails loudly with a `--resume`
  instruction. Malformed output degrades to the domain's recorded
  no-action default; it never crashes and is never silently patched.

## Domain packs

Simulator parameters and their real-world justification are documented
in each pack's module docstring; the achieved outcome mix, trap share,
and base rate are disclosed in `summary.json` provenance. Base rates are
held near the cost break-even so constant policies cannot masquerade as
skill. All shown numbers pass per-case seeded positive affine
anonymization where the domain permits, with decision structure
verified invariant on the rounded shown numbers.

| Pack | Series | Decision | Costs | Trap flavor |
| --- | --- | --- | --- | --- |
| `cloudcost` | daily account spend with weekday cycle, deploy uplifts, migrations | budget breach: act (rightsize) or monitor | overage ≫ intervention (break-even 0.2) | a revised commit change moves the breach threshold |
| `cashflow` | daily cash balance driven by invoices at terms ± lateness, payroll, opex | cross the minimum-balance floor: draw the credit line or not | shortfall ≫ carry (break-even 0.2) | an invoice amount corrected after issue |

Adding a pack means adding one module under `domains/` that calls
`harness.register(...)` — the harness is never edited, and a registry
test enforces it.

# LeakTrap — pre-registered analysis plan

Frozen before the arms are read, so that which numbers were the hoped-for
ones is a matter of record rather than of recollection. The family has
already been published once with a headline that turned out to be
arithmetic; a plan written down in advance is the cheapest defence against
the next version of that mistake.

Amendments are additive and dated. Nothing here is edited in place after a
result is recorded against it — an amendment that quietly relaxed a
criterion would defeat the purpose of the document.

**Version:** 1 (2026-08) · **Ceiling basis:** `leaktrap-ceiling-1` ·
**Generator:** `leaktrap-tasks-2` · **Held-out set:** `leaktrap-heldout-1`

---

## 1. What is being claimed

A bitemporal query layer converts temporal leakage from a behavioural risk
into a structural one: a consumer reading through it cannot use data
published after its cutoff, and the run's own access record proves which
publication dates were consulted.

This is a claim about **implementations of point-in-time access**, not about
Gnomon specifically. Gnomon is one implementation under test; a reference
implementation and a deliberately broken one are two others.

## 2. Instruments, and what each can conclude

| Instrument | Question | Known limit |
|---|---|---|
| **No-leak ceiling** (`leak_advantage > 0.25`) | Did this forecast beat what any strategy restricted to data published by the cutoff could achieve? | Zero power against any forecaster inside the ceiling basis. Detects level shifts; near-blind to revision-only leaks, because the ceiling grants the revision correction. |
| **Transcription** (WAPE ≤ 1e-4) | Did this forecast reproduce the post-cutoff values? | Needs no ceiling; detects only outright copying. |
| **Structural assertion** | Did the run's own access record show a read published after the cutoff, under recorded publication dates? | Self-reported. Certifies the query path, not the provenance of the dates, and not the implementation's honesty about its own reads. |

**Pre-committed:** the leak flag is *not* the instrument for any arm whose
forecasts the ceiling basis reproduces. Where `flag_reach` is zero, the
result is reported as "no power" and no paired test is run. This rule is
what the previous version of the family violated.

## 3. Hypotheses and endpoints

### Primary

**P1 — Trap validity.** On the `classic` family, a deliberate leaker's
median `leak_advantage` exceeds 0.4.
*Falsified if* it does not: leaking would then buy nothing and no other
number in the family would mean anything.

**P2 — Structural discrimination.** Across all families, the structural
assertion holds on every task for the arms that fence on publication date
(`gnomon`, `reference-pit`) and fails on every task for the arms that do not
(`gnomon-leaky`, `reference-naive`).
*Falsified if* any conforming arm is caught or any leaking arm is missed.
This is the claim in §1 and it is a single, two-sided, all-or-nothing test —
no threshold, no p-value.

**P3 — Detector specificity.** The leak flag's false-positive rate on arms
that cannot leak by construction is below 10%, with the upper bound of its
95% Wilson interval reported alongside.
*Falsified if* it is not: a detector that accuses competent honest
forecasters cannot support an accusation against anything else.

### Secondary

**S1 — Behavioural risk is real.** An LLM control, told the
publication-date rule and given the dates, is flagged or transcribes on a
non-trivial fraction of tasks.
*Falsified if* it is not — then structural safety guards a failure frontier
models no longer commit, and the claim must be restated as a guarantee
rather than an advantage.

**S2 — Prompting is not a substitute.** The `strict` prompt does not reduce
the control's leakage to zero.
*Falsified if* it does: a control that can be prompted into safety is a
materially weaker case for structure, and that must be reported as such.

**S3 — External validity.** The structural results (P2) are invariant across
the `classic`, `diverse` and `null` families, while the score-based flag's
sensitivity is not required to be.

## 4. Analysis rules, fixed in advance

- **Primary endpoints are single tests.** No adjustment is applied to them
  and none is needed. All other paired tests are exploratory, unadjusted,
  and labelled as such in the report's `multiplicity` field.
- **Abstentions are excluded from rates and reported as brackets** — the
  rate under "every abstention was honest" and under "every abstention
  leaked". Neither bound is quoted alone.
- **Rates carry Wilson 95% intervals.** A rate of 0 is reported with its
  interval, never as certainty.
- **The advantage is summarised by its median.** It is a ratio against a
  per-task ceiling and one task with a near-zero ceiling moves the mean more
  than a finding does. The mean is reported beside it, not instead of it.
- **Power.** At 40 tasks an exact McNemar needs 6 one-directional discordant
  pairs to reject at α = 0.05. A null result below that is reported as
  "no effect of this size was detectable", never as "no effect".
- **Stopping rule.** Task counts and seeds are fixed before running (40 per
  arm; seeds 7, 11, 13). Arms are not extended after seeing a result.
- **Regrading.** Instruments may change; when they do, every recorded row is
  regraded under one basis before comparison, and rows that cannot be fully
  regraded are labelled rather than assumed.

## 5. Known limitations, declared in advance

These are not discovered afterwards; they bound the claim from the start.

1. **Synthetic data.** No real vintage series (ALFRED, real-time macro
   databases) is used. The revision processes are modelled on documented
   behaviour — late-arriving reports, zero-mean news revisions, whole-history
   rebasing — but a result here is a result about generated data.
2. **The structural assertion is a declaration.** An implementation that
   misreports its own reads is not caught by it. The benchmark grades the
   record and the forecast; it does not audit the process.
3. **Self-authored, and Gnomon is one of the systems under test.** The
   reference implementations exist to make the instrument checkable
   independently, but this is internal validation, not a neutral evaluation.
4. **One model on the LLM arms.** Any behavioural claim (S1, S2) is about
   that model under that prompt, not about frontier models generally.
5. **The leak flag is shape-sensitive.** Its sensitivity is a property of
   the family it is measured on and does not transfer between families. It
   is reported per family for that reason.

## 6. What would make this publishable beyond internal validation

Recorded here so the gap is not restated as a finding: real vintage data;
an LLM querying *through* a store versus the same LLM on the raw file;
multiple models with repeat sampling; and a third party able to run the
whole family from the repository without the authors.

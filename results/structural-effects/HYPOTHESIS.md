# Pre-registered hypothesis: LLM-classified structural effects

Registered 2026-08-04, at commit `f6d1b56` (claude/gnomon-harness-issues),
**before** any implementation or run of the treatment. `RESULTS.md` will
be written against these predictions.

## Background

The 2026-08 span-rejection census left an honest residue the numeric
grammar cannot touch: statements of a *structural* fact with no number
to quote. The measured instance, verbatim:

> "the sensor was repaired and this additive trend will disappear."

A history-only forecast extrapolates the observed glitch trend; the
statement says that is wrong. The claim carries information, states no
value, and has no home in either existing warrant (fold ablation
cannot test it; the numeric grammar has nothing to parse). The
`AMENDMENT-2026-08-04.md` adjudication counted it as the prefix's one
genuine warrant-requiring instance and noted that its shape argues for
a structural-effect verb, not for proposer trust.

## The division-of-labour principle

**The LLM reads and classifies; it never supplies a number that is
applied.** The existing lane already enforces the second half. This
experiment adds the first half explicitly: classification into a small
closed menu of typed effects is delegated to the model, because a
mischosen *class* is bounded by the type system (few classes, each with
a defined, disclosed, counterfactual-recorded, value-free effect),
while a misread *number* is unbounded. Phrasings of a concept are where
pattern grammars genuinely lose to models; numbers stay with the
deterministic parser.

## Intervention

A third event class in the future-context lane, `structural:<label>`,
behind `context.structural_events` (default off; flag-off artifacts
byte-identical). The proposal carries:

- `source_span` — the verbatim sentence, provenance checked by the
  calling harness as with the existing classes;
- `effect` — chosen from a closed menu. **v1 menu has exactly one
  entry**, the one with measured evidence:

`trend_ceases`
    The observed drift stops continuing. Effect: fit a least-squares
    slope to Gnomon's *own emitted point path* and remove that drift
    from the first covered step onward (each covered step k gets
    slope × (k − k₀) subtracted from its point and every quantile — a
    pure location shift, so interval widths and quantile ordering are
    untouched). Every number involved is derived from Gnomon's own
    output; a path that already has no drift makes the effect a
    measured no-op. Applied before constraints and overrides, so a
    stated bound still clamps the result and a stated override still
    wins inside its window.

Admission checks: effect in menu, source span present, window entirely
future, window touches the horizon — the same structural checks as the
existing classes, with no numeric parse because there is no number.
Disclosure identical to the lane: support drops to `context_trusted`,
history-only counterfactual recorded, admitted events enter the
artifact ID payload.

## Primary hypothesis

On the CiK sensor task families (where the cessation statements occur),
flag-on against flag-off, same code, same seeds:

> **H1.** On matched task-seeds with at least one admitted structural
> event, mean RCRPS improves, with
> - the effect's parameters derived solely from the emitted path
>   (audited from the applications evidence — the invariant, not a
>   hope),
> - **no run where an admitted structural event worsens RCRPS by more
>   than 0.01 without being individually reported as a harm case**, and
> - runs with no admitted structural event **bit-identical** to
>   flag-off.

## Secondary predictions

- **H2 (it fires).** At least 3 structural admissions across the
  families that produce cessation statements. Zero admissions = the
  experiment failed regardless of scores, and RESULTS.md must say so.
- **H3 (classification is the right delegation).** No admitted
  `trend_ceases` on a span that a reader would call a level or bound
  statement (audited from the quoted spans) — the closed menu keeps a
  wrong class recognisable, which is the safety argument for
  delegating classification at all.

## Falsifiers

- Mean RCRPS on admitted-event matched runs worse than flag-off → H1
  falsified; the verb is not worth its complexity.
- Any effect parameter not derivable from the emitted path → void
  regardless of score.
- Any non-admitted run that differs from flag-off → void.
- Zero admissions → failed (H2).

## Analysis plan

Matched task-seeds only; abstentions/errors beside every mean; harm
cases listed individually; no post-hoc filtering. The A/B runs on the
same code revision for both arms. CiK is dev-pressured: a positive
result is confirmed on task families not used while building the verb
(the cessation families are in the untested 38–71 range, which helps)
before any claim leaves this directory.

## Out of scope

More menu entries (level resets, pauses) until an instance of each is
measured in a census; numeric extraction by the LLM (never); any
effect on abstention behaviour.

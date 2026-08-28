"""Text renderings of context facts — the native form context arrives in.

Every context item is a structured fact; its text rendering is generated
*from* that fact by a seeded template family, so extraction ground truth
is exact and free: the suite knows precisely which number, window, and
version the text displayed. Template families vary phrasing, units,
vagueness (a disclosed share of renderings round the shown number to two
significant figures — the extraction ground truth is then the rounded
number actually shown), and buried irrelevancies (filler clauses with
numbers that are not facts), so numerification is nontrivial.

Revision renderings mention both figures ("initially estimated at
{prev}, now expected closer to {value}") — extraction must keep the
version that was correct as of the cutoff. Every rendering embeds a
deterministic reference code derived from the case and item ids; the
codes of post-cutoff versions double as leakage-lint markers, because a
hidden version's code can only appear in a prompt if the information
boundary was broken.

Packs register template families per fact kind with
:func:`register_templates`; rendering an unregistered kind fails loudly.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

#: kind -> {"base": (templates...), "revision": (templates...)}
_TEMPLATES: dict[str, dict[str, tuple[str, ...]]] = {}

#: Share of renderings that show a vague (2-significant-figure) number.
VAGUE_SHARE = 0.25
#: Share of renderings that carry a buried irrelevancy clause.
FILLER_SHARE = 0.4

_FILLERS = (
    " (tracked alongside {n1} open tickets from the same review)",
    "; the sync call moved to the {n1}th and drew {n2} attendees",
    " — unrelated: badge readers logged {n1} visitors that week",
    "; note the doc template is on revision {n1}",
    " (the wiki page has {n1} watchers)",
)

#: Whole memos that carry no fact at all. A real inbox is mostly noise;
#: a numerifier that turns these into claims is hallucinating, and the
#: suite scores exactly that (the claims match no simulator fact).
DISTRACTOR_RANGE = (1, 3)
_DISTRACTORS = (
    "Facilities notice {ref}: badge-system maintenance is scheduled; "
    "expect about {n1} minutes of downtime.",
    "All-hands logistics ({ref}): {n1} attendees confirmed so far; the "
    "agenda doc is on revision {n2}.",
    "IT bulletin {ref}: password rotation is due; {n1} accounts are "
    "still pending.",
    "Team newsletter {ref}: {n1} sign-ups this week; the survey closes "
    "soon.",
    "Parking memo {ref}: lot B repainting affects {n1} spaces; use the "
    "visitor row.",
    "Onboarding note ({ref}): {n1} new starters this cycle; buddy "
    "assignments go out Friday.",
)


def distractor_lines(case: Any) -> list[tuple[int, str]]:
    """Seeded pure-noise memos for one case: (known_at step, text)
    pairs, deterministic in the case id alone. Their numbers are small
    integers so they can never collide with the leakage lint's numeric
    markers, and their reference codes derive from a distinct namespace
    so they can never collide with item codes."""
    rng = random.Random(f"{case.case_id}:distractors")
    count = rng.randint(*DISTRACTOR_RANGE)
    lines = []
    for index in range(count):
        template = _DISTRACTORS[rng.randrange(len(_DISTRACTORS))]
        step = rng.randrange(0, max(1, case.cutoff))
        lines.append((step, template.format(
            ref=ref_code(case.case_id, f"distractor-{index}"),
            n1=rng.randrange(5, 480), n2=rng.randrange(2, 60))))
    return lines


def register_templates(kind: str, base: tuple[str, ...],
                       revision: tuple[str, ...] | None = None) -> None:
    if kind in _TEMPLATES:
        raise ValueError(f"templates for kind {kind!r} already registered")
    if not base:
        raise ValueError(f"kind {kind!r} needs at least one base template")
    _TEMPLATES[kind] = {"base": tuple(base),
                        "revision": tuple(revision or base)}


def registered_kinds() -> set[str]:
    return set(_TEMPLATES)


def ref_code(case_id: str, item_id: str) -> str:
    """Deterministic per-item reference code, embedded in every
    rendering and computable without rendering — the leakage lint scans
    for the codes of hidden versions."""
    digest = hashlib.sha256(f"{case_id}:{item_id}".encode()).hexdigest()
    return f"REF-{digest[:6].upper()}"


def _vague(value: float) -> float:
    """Round to two significant figures — what a memo author writes."""
    if value == 0:
        return 0.0
    from math import floor, log10
    digits = -int(floor(log10(abs(value)))) + 1
    return round(value, digits)


def _fmt(value: float) -> str:
    text = f"{value:,.4f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-") else "0"


def render_item(item: Any, case: Any, rng: random.Random,
                prev_value: float | None = None,
                ) -> tuple[str, float]:
    """Render one resolved item; returns ``(text, shown_value)`` where
    ``shown_value`` is the exact number the text displays (the
    extraction ground truth for this rendering)."""
    from benchmarks.enterprisebench.harness import grid_date

    if item.kind not in _TEMPLATES:
        raise KeyError(
            f"no text templates registered for kind {item.kind!r}; the "
            "pack must call textgen.register_templates for every kind "
            "it emits")
    family = _TEMPLATES[item.kind]
    is_revision = item.revises is not None and prev_value is not None
    templates = family["revision"] if is_revision else family["base"]
    template = templates[rng.randrange(len(templates))]
    shown = _vague(item.value) if rng.random() < VAGUE_SHARE \
        else round(item.value, 4)
    fields = {
        "value": _fmt(shown),
        "prev_value": _fmt(round(prev_value, 4))
        if prev_value is not None else "",
        "from_date": grid_date(case, item.effective_from),
        "to_date": grid_date(case, item.effective_to),
        "known_date": grid_date(case, item.known_at),
        "ref": ref_code(case.case_id, item.item_id),
    }
    fields.update({key: str(value) for key, value in item.aux_dict().items()})
    text = template.format(**fields)
    if rng.random() < FILLER_SHARE:
        filler = _FILLERS[rng.randrange(len(_FILLERS))]
        text += filler.format(n1=rng.randrange(3, 90),
                              n2=rng.randrange(3, 40))
    return text, shown


def render_context_block(case: Any, resolved: list[Any],
                         prev_values: dict[str, float],
                         ) -> tuple[str, dict[str, float]]:
    """The text context block for one case: every as-of resolved item
    rendered as a dated memo line, interleaved by date with seeded
    pure-noise distractor memos (the way a real inbox arrives). Returns
    the block and the per-item shown values (extraction ground truth —
    distractors have no entry because they state no fact).
    Deterministic in the case and item ids alone."""
    from benchmarks.enterprisebench.harness import grid_date

    entries: list[tuple[int, int, str]] = []
    shown_values: dict[str, float] = {}
    for order, item in enumerate(resolved):
        rng = random.Random(f"{case.case_id}:{item.item_id}:text")
        text, shown = render_item(item, case, rng,
                                  prev_values.get(item.item_id))
        shown_values[item.item_id] = shown
        entries.append((item.known_at, order, text))
    for order, (step, text) in enumerate(distractor_lines(case)):
        entries.append((step, len(resolved) + order, text))
    lines = [f"- [{grid_date(case, step)}] {text}"
             for step, _, text in sorted(entries)]
    return "\n".join(lines) if lines else "- (no memos)", shown_values
